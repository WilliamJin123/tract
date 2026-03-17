"""Semantic Quality Gates: LLM-Powered Transition Enforcement

A semantic gate blocks stage transitions until the LLM judges that
the context meets a quality criterion. Unlike deterministic middleware
(which counts commits or checks tags), semantic gates evaluate the
*meaning* of the context.

This example runs an adversarial research workflow:
  1. An agent researches microservices vs monolith trade-offs
  2. A semantic gate on pre_transition blocks moving to "synthesis"
     unless the research has sufficient depth and diversity
  3. The agent hits the gate, gets blocked, does more research, retries

The gate is NOT scripted -- it makes a real LLM call to evaluate
whether the commit log shows substantive analysis from multiple angles.
If the initial research happens to be deep enough, the gate passes on
the first attempt (and that's fine -- honest behavior).

Demonstrates: t.middleware.gate() for registering semantic gates, condition
callbacks for efficient pre-checks, BlockedError recovery, genuine
agent interaction with quality enforcement.

Requires: LLM API key (uses Cerebras provider)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    print("=" * 70)
    print("Semantic Quality Gates: LLM-Powered Transition Enforcement")
    print("=" * 70)
    print()
    print("  A semantic gate evaluates research QUALITY via LLM judgment,")
    print("  not just commit count. The agent must produce genuinely diverse")
    print("  analysis before the gate allows transition to synthesis.")
    print()

    log = StepLogger()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
    ) as t:
        # ─── Setup: stages and system prompt ──────────────────────
        t.system(
            "You are a technology research analyst. Your job is to produce "
            "thorough, multi-perspective analysis. When researching a topic, "
            "commit each distinct finding or perspective separately so your "
            "research log shows breadth and depth."
        )
        t.config.set(stage="research")

        # Create the synthesis branch (but stay on main for research)
        t.branches.create("synthesis", switch=False)

        # ─── Register semantic gate ───────────────────────────────
        # The gate fires on pre_transition, but ONLY when transitioning
        # to "synthesis". It uses a cheap model to judge whether the
        # research commits show substantive analysis from genuinely
        # distinct angles.
        t.middleware.gate(
            "research-depth",
            event="pre_transition",
            check=(
                "Does the research contain substantive analysis from at least "
                "2 genuinely different angles or perspectives? Look at the "
                "commit messages and content types. The commits should cover "
                "distinct viewpoints (e.g., technical trade-offs AND organizational "
                "impact, or performance AND developer experience). Repetition of "
                "the same angle with different wording does NOT count. "
                "Be strict: superficial one-liners do not count as substantive."
            ),
            model=llm.small,  # cheap model for the gate evaluation
            condition=lambda ctx: ctx.target == "synthesis",
        )

        print(f"  Branch: {t.current_branch}")
        print(f"  Gates registered: {t.middleware.list_gates()}")
        print(f"  Gate fires on: pre_transition (to synthesis only)")

        # ─── Phase 1: Initial research ────────────────────────────
        # Give the agent a research problem, NOT a procedure.
        # The agent decides how to structure its research.
        print("\n" + "=" * 70)
        print("Phase 1: Initial Research")
        print("=" * 70 + "\n")

        result = t.llm.run(
            "Research the trade-offs of microservices vs monolith architecture. "
            "Focus on ONE specific angle (e.g., deployment complexity OR data "
            "consistency). Commit your findings as you go. Keep it focused.",
            max_steps=6,
            max_tokens=1024,
            profile="full",
            tool_names=["commit", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Show what was committed
        entries = t.search.log(limit=20)
        print(f"\n  Commits after Phase 1: {len(entries)}")
        for entry in entries[:5]:
            msg = (entry.message or "(no message)")[:60]
            print(f"    [{entry.commit_hash[:8]}] {entry.content_type:12s} \"{msg}\"")

        # ─── Attempt transition -- gate evaluates quality ─────────
        print("\n" + "=" * 70)
        print("Transition Attempt 1: Does the gate pass?")
        print("=" * 70 + "\n")

        gate_blocked = False
        try:
            t.transition("synthesis")
            print("  Gate PASSED on first attempt -- research deemed sufficient.")
            print("  (The LLM judge found enough depth/diversity in Phase 1.)")
        except BlockedError as e:
            gate_blocked = True
            print(f"  Gate BLOCKED: {e.reasons[0]}")
            print()
            print("  The semantic gate judged that the research lacks diversity.")
            print("  The agent must research from a DIFFERENT angle before retrying.")

        # ─── Phase 2: Deeper research (if blocked) ────────────────
        if gate_blocked:
            print("\n" + "=" * 70)
            print("Phase 2: Expanding Research (Different Angle)")
            print("=" * 70 + "\n")

            result = t.llm.run(
                "The quality gate blocked our transition to synthesis because "
                "the research lacks diverse perspectives. Research from a "
                "COMPLETELY DIFFERENT angle than before. If you covered "
                "technical trade-offs, now cover organizational/team impact. "
                "If you covered deployment, now cover data management. "
                "Commit your findings.",
                max_steps=6,
                max_tokens=1024,
                profile="full",
                tool_names=["commit", "status"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )
            result.pprint()

            entries = t.search.log(limit=20)
            print(f"\n  Total commits after Phase 2: {len(entries)}")
            for entry in entries[:8]:
                msg = (entry.message or "(no message)")[:60]
                print(f"    [{entry.commit_hash[:8]}] {entry.content_type:12s} \"{msg}\"")

            # ─── Retry transition ─────────────────────────────────
            print("\n" + "=" * 70)
            print("Transition Attempt 2: Retry after deeper research")
            print("=" * 70 + "\n")

            try:
                t.transition("synthesis")
                print("  Gate PASSED on retry -- research now has sufficient diversity.")
            except BlockedError as e:
                print(f"  Gate BLOCKED again: {e.reasons[0]}")
                print("  (In production, you would loop: research more, retry.)")

        # ─── Final report ─────────────────────────────────────────
        print("\n" + "=" * 70)
        print("Final State")
        print("=" * 70 + "\n")

        print(f"  Current branch: {t.current_branch}")

        entries = t.search.log(limit=50)
        print(f"  Total commits: {len(entries)}")

        print(f"  Gates: {t.middleware.list_gates()}")

        status = t.search.status()
        print(f"  Tokens: {status.token_count}")

        # Show the compiled context summary
        print(f"\n  Compiled context:")
        t.compile().pprint(style="compact")

        reached_synthesis = t.current_branch == "synthesis"
        print(f"\n  Reached synthesis: {reached_synthesis}")
        if gate_blocked:
            print("  Gate blocked initial transition: YES")
            print("  Agent adapted with deeper research: YES")
        else:
            print("  Gate blocked initial transition: NO (research was sufficient)")

    # ─── Why semantic gates matter ──────────────────────────────
    print("\n" + "=" * 70)
    print("WHY SEMANTIC GATES")
    print("=" * 70)
    print("""
  Deterministic gates (count commits, check tags):
    - Easy to game: agent commits 3 empty artifacts, gate passes
    - No quality judgment: quantity != quality
    - Brittle: hardcoded thresholds don't adapt to context

  Semantic gates (LLM evaluates meaning):
    - Evaluates the CONTENT of the research, not just metadata
    - Catches thin/repetitive analysis even with many commits
    - Natural language criteria adapt to any domain
    - Condition callbacks skip the LLM call when irrelevant
      (e.g., transitioning to a different branch)

  Cost: one cheap LLM call per gate evaluation (~100 tokens).
  The gate model can be smaller/cheaper than the main agent model.
""")


if __name__ == "__main__":
    main()


# --- See also ---
# Semantic maintenance:         agent/09_semantic_maintenance.py
# Implicit discovery:           agent/01_implicit_discovery.py
# Profile-based stages:         agent/04_profile_stages.py
# Middleware basics:             config_and_middleware/02_event_automation.py
