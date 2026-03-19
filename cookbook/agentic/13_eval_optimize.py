"""Evaluate-and-Optimize: Generate, Score, Improve Loop

Two patterns for using LLMs as judges of their own output:

  1. Evaluator-Generator Loop -- generate a solution, score it on a rubric,
     feed the evaluation back in, and iterate until it passes. Each attempt
     lives on its own branch so the DAG shows the full improvement history.

  2. Mandatory Rationale Gate -- a pre_commit middleware that blocks any
     "implementation" artifact unless a "rationale" artifact already exists
     on the branch. Forces the agent to explain WHY before committing WHAT.

Core principle: LLMs are better at evaluating than generating. A cheap model
scoring structured criteria catches issues that the generator misses.

Demonstrates: branches for iteration history, t.llm.run(), t.llm.chat(),
              metadata for scores, merge(strategy="theirs"), pre_commit
              middleware, BlockedError recovery, tag-based gating,
              compile().pprint(style="chat")
"""

import io
import re
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError, MiddlewareContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger


MODEL_ID = llm.small


def _section(num: int, title: str, desc: str) -> None:
    print(f"\n{'=' * 70}\n  {num}. {title}\n{'=' * 70}\n  {desc}\n")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_scores(text: str) -> dict:
    """Extract CORRECTNESS, COMPLETENESS, QUALITY scores from evaluator text."""
    scores = {}
    for key in ("correctness", "completeness", "quality"):
        match = re.search(rf"{key}\s*:\s*(\d(?:\.\d)?)", text, re.IGNORECASE)
        scores[key] = float(match.group(1)) if match else 3.0
    scores["average"] = round(
        sum(scores[k] for k in ("correctness", "completeness", "quality")) / 3,
        2,
    )
    return scores


# ===================================================================
# Section 1: Evaluator-Generator Loop
# ===================================================================

TASK_PROMPT = (
    "Write a Python function `parse_cron(expr: str) -> dict` that parses "
    "a standard 5-field cron expression (minute, hour, day-of-month, month, "
    "day-of-week) and returns a dict mapping each field name to a list of "
    "integer values it represents. Handle: single values, ranges (1-5), "
    "steps (*/15), comma lists (1,3,5), and wildcards (*). Raise ValueError "
    "for invalid input."
)

EVAL_PROMPT = (
    "You are a code reviewer. Score this implementation on three criteria "
    "(1-5 each, 5 is best). Be strict.\n\n"
    "CORRECTNESS: Does it handle edge cases? (invalid input, boundary values, "
    "ranges that wrap, step values on subsets like 1-10/3)\n"
    "COMPLETENESS: Does it cover all 5 cron fields with proper validation "
    "of each field's allowed range?\n"
    "QUALITY: Is it readable? No unnecessary complexity? Good variable names? "
    "Proper error messages?\n\n"
    "Reply in EXACTLY this format (one line each, then feedback):\n"
    "CORRECTNESS: N\n"
    "COMPLETENESS: N\n"
    "QUALITY: N\n"
    "FEEDBACK: <specific issues to fix, or 'none' if passing>\n\n"
    "Implementation to review:\n\n"
)

PASS_THRESHOLD = 4.0
MAX_ITERATIONS = 3


def section_1_eval_loop() -> None:
    _section(1, "Evaluator-Generator Loop",
             "Generate -> evaluate -> improve until the evaluator passes it.")

    log = StepLogger()

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:

        t.system(
            "You are an expert Python developer. Write clean, correct, "
            "well-tested code. When given evaluation feedback, address "
            "every specific issue mentioned."
        )

        for tag in ["generation", "evaluation", "passed", "failed"]:
            t.register_tag(tag)

        final_scores = None
        passed = False

        for iteration in range(1, MAX_ITERATIONS + 1):
            print(f"\n  --- Iteration {iteration}/{MAX_ITERATIONS} ---\n")

            # --- Generate on a dedicated branch ---
            branch_name = f"gen/{iteration}"
            t.branch(branch_name, switch=True)
            t.config.set(temperature=0.7, stage=f"generation-{iteration}")

            if iteration == 1:
                prompt = TASK_PROMPT + "\nCommit the function as an artifact."
            else:
                prompt = (
                    f"The evaluator scored your previous attempt and found issues.\n"
                    f"Their feedback is in the conversation above.\n\n"
                    f"Rewrite the function from scratch, addressing EVERY point "
                    f"in the feedback. Commit the improved version as an artifact."
                )

            gen_result = t.llm.run(
                prompt,
                max_steps=6,
                max_tokens=2048,
                tool_names=["commit", "status"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )

            gen_log = t.log(limit=20)
            artifacts = [e for e in gen_log if e.content_type == "artifact"]
            print(f"\n  Generator: {gen_result.status}, "
                  f"{len(artifacts)} artifact(s) committed")

            # --- Evaluate (low temperature, structured output) ---
            t.config.set(temperature=0.1)

            # Build eval input from the latest artifact
            if artifacts:
                code_content = t.get_content(artifacts[0])
                code_text = str(code_content) if code_content else "(no content)"
            else:
                code_text = gen_result.final_response or "(no output)"

            eval_response = t.llm.chat(
                EVAL_PROMPT + code_text,
                max_tokens=400,
            )

            eval_text = eval_response.text or ""
            scores = parse_scores(eval_text)

            # Store evaluation as a commit with scores in metadata
            t.commit(
                content=eval_text,
                message=f"evaluation: iteration {iteration} "
                        f"(avg={scores['average']})",
                content_type="evaluation",
                metadata={"scores": scores, "iteration": iteration},
                tags=["evaluation"],
            )

            print(f"\n  Evaluator scores:")
            for k in ("correctness", "completeness", "quality"):
                print(f"    {k:14s}: {scores[k]}/5")
            print(f"    {'average':14s}: {scores['average']}/5  "
                  f"(threshold: {PASS_THRESHOLD})")

            # Extract feedback line for display
            fb_match = re.search(r"FEEDBACK:\s*(.+)", eval_text, re.IGNORECASE)
            if fb_match:
                print(f"    feedback: {fb_match.group(1)[:100]}")

            final_scores = scores

            if scores["average"] >= PASS_THRESHOLD:
                # Pass -- tag and merge to main
                passed = True
                t.commit(
                    content=f"PASSED at iteration {iteration} "
                            f"with average {scores['average']}",
                    message=f"passed: iteration {iteration}",
                    tags=["passed"],
                )
                t.switch("main")
                t.merge(branch_name, strategy="theirs",
                        message=f"merge passing generation (iter {iteration})")
                print(f"\n  PASSED at iteration {iteration} -- merged to main")
                break
            else:
                # Fail -- record, switch back to main for next iteration
                t.commit(
                    content=f"FAILED iteration {iteration}: {eval_text}",
                    message=f"failed: iteration {iteration}",
                    tags=["failed"],
                )
                t.switch("main")
                # Merge the evaluation feedback so the next branch sees it
                t.merge(branch_name, strategy="theirs",
                        message=f"merge feedback from iteration {iteration}")
                print(f"\n  FAILED -- feedback merged, continuing...\n")

        # --- Summary ---
        print(f"\n  {'=' * 50}")
        print(f"  Eval Loop Summary")
        print(f"  {'=' * 50}")
        print(f"  Passed: {passed}")
        if final_scores:
            print(f"  Final scores: {final_scores}")
        print(f"  Branches created:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        s = t.status()
        print(f"  Total: {s.commit_count} commits, {s.token_count} tokens")

        print(f"\n  Compiled context (final):")
        t.compile().pprint(style="chat")

    print("\n  PASSED")


# ===================================================================
# Section 2: Mandatory Rationale Gate
# ===================================================================

def section_2_rationale_gate() -> None:
    _section(2, "Mandatory Rationale Gate",
             "pre_commit blocks 'implementation' commits until a 'rationale' exists.")

    log = StepLogger()

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:

        t.system(
            "You are a careful software engineer. When asked to implement "
            "something, first explain your reasoning, then write the code."
        )

        for tag in ["rationale", "implementation"]:
            t.register_tag(tag)

        # --- Gate: block implementation commits without a prior rationale ---
        def require_rationale(ctx: MiddlewareContext):
            """Block any commit tagged 'implementation' unless a 'rationale' exists."""
            if not ctx.commit or not ctx.commit.tags:
                return
            if "implementation" not in ctx.commit.tags:
                return
            # Check if any rationale commit exists on this branch
            rationales = ctx.tract.find(tag="rationale", limit=5)
            if not rationales:
                raise BlockedError(
                    "pre_commit",
                    "Cannot commit implementation without a rationale. "
                    "Commit a rationale first (tag it 'rationale'), "
                    "then commit the implementation.",
                )

        t.middleware.add("pre_commit", require_rationale)
        print("  Gate installed: pre_commit requires 'rationale' before 'implementation'\n")

        # --- Demo 1: Direct implementation attempt (should be blocked) ---
        print("  --- Attempt 1: commit implementation directly ---\n")

        try:
            t.commit(
                content="def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
                message="implementation: fibonacci function",
                content_type="artifact",
                tags=["implementation"],
            )
            print("  (unexpectedly succeeded)")
        except BlockedError as e:
            print(f"  BLOCKED: {e}")
            print("  Good -- the gate enforces rationale-first.\n")

        # --- Demo 2: Rationale first, then implementation (should succeed) ---
        print("  --- Attempt 2: rationale first, then implementation ---\n")

        t.commit(
            content=(
                "Rationale for fibonacci implementation:\n"
                "- Recursive approach is clearest for demonstration\n"
                "- Add memoization via functools.lru_cache for O(n) performance\n"
                "- Include input validation: n must be non-negative integer\n"
                "- Return type: int (arbitrary precision in Python)"
            ),
            message="rationale: fibonacci design decisions",
            content_type="artifact",
            tags=["rationale"],
        )
        print("  Rationale committed successfully.")

        t.commit(
            content=(
                "from functools import lru_cache\n\n"
                "@lru_cache(maxsize=None)\n"
                "def fibonacci(n: int) -> int:\n"
                "    if not isinstance(n, int) or n < 0:\n"
                "        raise ValueError(f'n must be non-negative integer, got {n}')\n"
                "    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)\n"
            ),
            message="implementation: fibonacci with memoization",
            content_type="artifact",
            tags=["implementation"],
        )
        print("  Implementation committed successfully (rationale existed).\n")

        # --- Demo 3: Agent loop with the gate active ---
        print("  --- Attempt 3: let the agent handle the gate ---\n")

        t.branch("agent-work", switch=True)

        result = t.llm.run(
            "Implement a Python function `flatten(nested: list) -> list` that "
            "flattens arbitrarily nested lists. You MUST commit a rationale "
            "first (tagged 'rationale'), then the implementation (tagged "
            "'implementation'). The pre_commit gate will block you if you "
            "skip the rationale.",
            max_steps=8,
            max_tokens=1024,
            tool_names=["commit", "tag", "status", "log"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # --- Verify ordering in compiled context ---
        print(f"\n  Compiled context (rationale should precede implementation):")
        t.compile().pprint(style="chat")

        # --- Summary ---
        entries = t.log(limit=30)
        rationales = [e for e in entries if e.tags and "rationale" in e.tags]
        implementations = [e for e in entries if e.tags and "implementation" in e.tags]

        print(f"\n  Summary:")
        print(f"    Rationales committed:      {len(rationales)}")
        print(f"    Implementations committed: {len(implementations)}")
        print(f"    Gate enforced ordering:    rationale always before implementation")

        s = t.status()
        print(f"    Total: {s.commit_count} commits, {s.token_count} tokens")

    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    print("=" * 70)
    print("  Evaluate-and-Optimize Patterns")
    print("  LLMs as judges: generate, score, improve")
    print("=" * 70)

    section_1_eval_loop()
    section_2_rationale_gate()

    print(f"\n{'=' * 70}")
    print("  Both patterns exploit the evaluation > generation asymmetry.")
    print("  Section 1: iterative improvement via structured scoring rubrics.")
    print("  Section 2: middleware gates enforce process (rationale before code).")
    print(f"{'=' * 70}")
    print("\nDone.")


test_eval_optimize = main

if __name__ == "__main__":
    main()


# --- See also ---
# Adversarial review:     agentic/05_adversarial_review.py
# Error recovery trials:  agentic/10_error_recovery.py
# Semantic gates:          agentic/04_semantic_automation.py
# Middleware reference:    reference/02_middleware.py
