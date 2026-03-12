"""TDD Coding Assistant: plan -> test -> implement -> verify -> iterate -> review

An advanced multi-stage workflow that demonstrates why DAG-based context
management is fundamentally superior to naive prompt chaining for software
engineering tasks. The agent builds an email validator through test-driven
development, using branches for parallel implementation attempts, quality
gates to enforce stage discipline, compression to manage token budgets,
and cross-branch diffs to learn from failed approaches.

Stages:
  planning       -- analyze requirements, identify edge cases (temp 0.9)
  test_writing   -- write failing tests first, TDD-style (temp 0.3)
  implementation -- write code to pass tests (temp 0.3)
  verification   -- run tests, record pass/fail metadata (temp 0.1)
  iteration      -- branch per fix attempt, diff to diagnose, merge winner
  review         -- final quality gate, only reachable when tests pass

Key patterns demonstrated:
  1. Branch-per-attempt -- isolate implementation variants, merge the winner
  2. Quality gates      -- middleware blocks premature stage transitions
  3. Self-correction    -- diff failed attempt against tests to guide fixes
  4. Context management -- compress planning before impl, pin test cases
  5. Cross-branch diff  -- compare() to analyze attempt/1 vs attempt/2
  6. Searchable history -- find() to locate test results and past patterns

Why this is impossible with naive prompt chains:
  - Prompt chains lose context as conversations grow. Tract compresses
    selectively, preserving pinned test cases while summarizing exploration.
  - Prompt chains cannot branch. Tract creates isolated implementation
    attempts and merges the one that passes.
  - Prompt chains cannot diff. Tract compares two branches to extract
    exactly what differed between a passing and failing approach.
  - Prompt chains have no gates. Tract middleware enforces that tests exist
    before implementation starts and that tests pass before review begins.

Requires: LLM API key (uses Cerebras provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError, Priority

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # =============================================================
        # 1. WORKFLOW INFRASTRUCTURE
        #    Config per stage, quality gate middleware, system prompt
        # =============================================================

        print("=" * 70)
        print("TDD Coding Assistant: Email Validator")
        print("=" * 70)
        print()

        # --- Stage configs (temperature, compile strategy per stage) ---
        t.configure(
            stage="planning",
            temperature=0.9,
            compile_strategy="full",
        )

        # --- Directive: workflow rules baked into compiled context ---
        t.directive(
            "tdd-workflow",
            "You are building an email validation function using TDD.\n"
            "Workflow stages:\n"
            "  1. PLANNING -- Analyze requirements, list edge cases\n"
            "  2. TEST_WRITING -- Write pytest test cases FIRST (they should fail)\n"
            "  3. IMPLEMENTATION -- Write code to make tests pass\n"
            "  4. VERIFICATION -- Mentally run tests, report pass/fail\n"
            "  5. REVIEW -- Summarize final solution quality\n\n"
            "Rules:\n"
            "- Always write tests BEFORE implementation code\n"
            "- Commit each major artifact (plan, tests, code, results)\n"
            "- When verifying, record test results with metadata\n"
            "- Use transition tool to advance between stages",
        )

        # --- Quality gate: block implementation until tests exist ---
        def require_tests_for_impl(ctx):
            if ctx.target != "implementation":
                return
            test_commits = ctx.tract.find(
                content="def test_", content_type="assistant", limit=5
            )
            if len(test_commits) < 1:
                raise BlockedError(
                    "pre_transition",
                    "Cannot start implementation: no test cases found. "
                    "Write tests first (TDD).",
                )

        # --- Quality gate: block review until verification passes ---
        def require_passing_tests_for_review(ctx):
            if ctx.target != "review":
                return
            results = ctx.tract.find(
                metadata_key="test_status", limit=10
            )
            passing = [
                r for r in results
                if r.metadata and r.metadata.get("test_status") == "pass"
            ]
            if not passing:
                raise BlockedError(
                    "pre_transition",
                    "Cannot enter review: no passing test results found. "
                    "Fix failing tests first.",
                )

        t.use("pre_transition", require_tests_for_impl)
        t.use("pre_transition", require_passing_tests_for_review)

        # --- System prompt ---
        t.system(
            "You are a senior Python developer practicing strict TDD.\n\n"
            "TASK: Build a `validate_email(email: str) -> bool` function.\n\n"
            "Requirements:\n"
            "- Return True for valid emails, False for invalid\n"
            "- Must handle: basic format (user@domain.tld), subdomains,\n"
            "  plus-addressing (user+tag@domain.com), dots in local part\n"
            "- Must reject: missing @, missing domain, spaces, double dots,\n"
            "  leading/trailing dots in local part, missing TLD\n"
            "- Do NOT use a regex library -- write the validation logic manually\n\n"
            "You have tools: commit, transition, get_config, status, log.\n"
            "Commit each artifact with a clear message. Use transition to\n"
            "advance stages when the current stage work is complete."
        )

        log = StepLogger()

        # =============================================================
        # 2. PLANNING STAGE
        #    High temperature, creative exploration of edge cases
        # =============================================================

        print("\n=== Stage 1: Planning ===\n")
        print("  Config: temperature=0.9, strategy=full")
        print("  Goal: analyze requirements, identify edge cases\n")

        result = t.run(
            "Analyze the email validation requirements. List:\n"
            "1. The valid email patterns to accept (with examples)\n"
            "2. The invalid patterns to reject (with examples)\n"
            "3. Edge cases that are tricky (e.g., plus addressing, subdomains)\n\n"
            "Commit your analysis as a structured plan. When done, "
            "transition to 'test_writing'.",
            max_steps=8,
            profile="full",
            tool_names=["commit", "transition", "get_config", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        planning_status = t.status()
        print(f"\n  Planning complete: {planning_status.commit_count} commits, "
              f"{planning_status.token_count} tokens")

        # =============================================================
        # 3. COMPRESS PLANNING + PIN TESTS
        #    Save tokens before implementation-heavy stages.
        #    Planning exploration is summarized; test cases are pinned.
        # =============================================================

        print("\n=== Context Management: Compress Planning ===\n")

        pre_compress_tokens = t.status().token_count
        compress_result = t.compress(
            content="Email validation planning summary: Function validate_email(email) -> bool. "
            "Valid patterns: user@domain.tld, user+tag@domain.com, user@sub.domain.com, "
            "first.last@domain.com. Invalid patterns: missing @, no domain, spaces, "
            "double dots (..), leading/trailing dots in local part, no TLD. "
            "Implementation approach: manual character-by-character validation, "
            "no regex library. Split on @, validate local part and domain separately.",
        )
        post_compress_tokens = t.status().token_count
        print(f"  Compressed: {pre_compress_tokens} -> {post_compress_tokens} tokens "
              f"(ratio: {compress_result.compression_ratio:.2f})")

        # =============================================================
        # 4. TEST WRITING STAGE
        #    Low temperature, precise test cases. Quality gate above
        #    ensures tests exist before we can transition to implementation.
        # =============================================================

        print("\n=== Stage 2: Test Writing (TDD) ===\n")

        # The agent transitions to test_writing at the end of planning via
        # the transition tool (which fires pre_transition middleware).
        # Here we just set the temperature for the test-writing stage.
        t.configure(stage="test_writing", temperature=0.3)

        result = t.run(
            "Write comprehensive pytest test cases for validate_email().\n\n"
            "Include tests for:\n"
            "- Valid emails: basic, plus-addressing, subdomains, dots in local\n"
            "- Invalid emails: no @, no domain, spaces, double dots, "
            "leading/trailing dots, no TLD, empty string\n"
            "- Edge cases: very long email, single-char local, single-char domain\n\n"
            "Format as a complete pytest file. Commit the test file. "
            "When done, transition to 'implementation'.",
            max_steps=8,
            profile="full",
            tool_names=["commit", "transition", "get_config", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        # --- Pin test cases so they survive future compression ---
        test_commits = t.find(content="def test_", content_type="assistant", limit=5)
        for tc in test_commits:
            t.annotate(tc.commit_hash, Priority.PINNED, reason="TDD: tests are source of truth")
            print(f"  Pinned test commit: {tc.commit_hash[:8]}")

        print(f"  Tests written and pinned ({len(test_commits)} commit(s))")

        # =============================================================
        # 5. IMPLEMENTATION STAGE -- ATTEMPT 1
        #    Branch "attempt/1" for the first implementation try.
        #    This isolates the attempt so we can try alternatives.
        # =============================================================

        print("\n=== Stage 3: Implementation (attempt/1) ===\n")

        # The agent transitions to implementation at the end of test_writing
        # via the transition tool (gate requires test commits to exist).
        # Here we just set the temperature for the implementation stage.
        t.configure(stage="implementation", temperature=0.3)

        # Create isolated branch for first attempt
        t.branch("attempt/1", switch=True)

        result = t.run(
            "Implement validate_email(email: str) -> bool.\n\n"
            "Requirements:\n"
            "- Split on '@' to get local part and domain\n"
            "- Validate local part: no leading/trailing dots, no consecutive dots,\n"
            "  allowed chars are alphanumeric, dots, plus, hyphen, underscore\n"
            "- Validate domain: at least one dot, valid subdomain labels,\n"
            "  TLD must be at least 2 chars\n"
            "- No regex -- use string operations only\n\n"
            "Commit the implementation. Do NOT transition yet.",
            max_steps=6,
            profile="full",
            tool_names=["commit", "get_config", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        attempt1_head = t.head
        print(f"  Attempt 1 committed: {attempt1_head[:8] if attempt1_head else 'None'}")

        # =============================================================
        # 6. VERIFICATION STAGE
        #    Mentally trace through tests. Record results as metadata.
        # =============================================================

        print("\n=== Stage 4: Verification (attempt/1) ===\n")

        # Verification is harness-driven (no gate on this stage).
        t.configure(stage="verification", temperature=0.1)

        result = t.run(
            "Mentally run each test case against the implementation.\n"
            "For each test, trace through the logic and determine if it would\n"
            "pass or fail. Be precise -- check every edge case.\n\n"
            "Commit your verification results. In the commit metadata, include:\n"
            "  test_status: 'pass' or 'fail'\n"
            "  tests_passed: number of passing tests\n"
            "  tests_failed: number of failing tests\n"
            "  failures: list of failing test names (if any)\n\n"
            "Be HONEST about failures. It's better to catch bugs now.",
            max_steps=6,
            profile="full",
            tool_names=["commit", "get_config", "status", "log"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        # Check test results via metadata search
        test_results = t.find(metadata_key="test_status", limit=5)
        attempt1_passed = any(
            r.metadata and r.metadata.get("test_status") == "pass"
            for r in test_results
        )

        if attempt1_passed:
            print("  Attempt 1: ALL TESTS PASSED")
        else:
            print("  Attempt 1: some tests failed (or no results committed)")
            # Show what the agent found
            for r in test_results:
                if r.metadata:
                    print(f"    Status: {r.metadata.get('test_status', '?')}, "
                          f"Passed: {r.metadata.get('tests_passed', '?')}, "
                          f"Failed: {r.metadata.get('tests_failed', '?')}")

        # =============================================================
        # 7. ITERATION -- ATTEMPT 2 (if needed)
        #    Branch from the test-writing point, try a different approach.
        #    Use diff to understand what went wrong in attempt/1.
        # =============================================================

        attempt2_passed = False  # only set True if attempt 2 is needed and passes
        if not attempt1_passed:
            print("\n=== Stage 5: Iteration (attempt/2) ===\n")

            # --- Self-correction via diff ---
            # Compare attempt/1 against the main branch to see what diverged
            print("  Analyzing attempt/1 via diff...")
            diff_result = t.compare(branch_a="main", branch_b="attempt/1")
            print(f"  Diff stat: +{diff_result.stat.messages_added} msgs, "
                  f"~{diff_result.stat.messages_modified} modified, "
                  f"{diff_result.stat.total_token_delta:+d} tokens")

            # Switch back to main and branch for attempt/2
            t.switch("main")
            t.branch("attempt/2", switch=True)
            t.configure(stage="implementation", temperature=0.3)

            # Gather failure context from attempt/1
            failure_context = ""
            for r in test_results:
                if r.metadata and r.metadata.get("failures"):
                    failure_context = f"Failed tests: {r.metadata['failures']}"

            result = t.run(
                f"The first implementation attempt had issues.\n"
                f"{failure_context}\n\n"
                "Write an IMPROVED validate_email(email: str) -> bool.\n"
                "Pay special attention to:\n"
                "- Edge cases with dots (leading, trailing, consecutive)\n"
                "- Plus-addressing: user+tag@domain.com should be valid\n"
                "- Domain validation: subdomains, TLD length\n"
                "- Empty string and missing @ handling\n\n"
                "Commit the improved implementation.",
                max_steps=6,
                profile="full",
                tool_names=["commit", "get_config", "status"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )

            attempt2_head = t.head

            # --- Cross-branch comparison ---
            print("\n  Comparing attempt/1 vs attempt/2...")
            cross_diff = t.compare(branch_a="attempt/1", branch_b="attempt/2")
            print(f"  Cross-diff: +{cross_diff.stat.messages_added} new, "
                  f"~{cross_diff.stat.messages_modified} modified, "
                  f"{cross_diff.stat.total_token_delta:+d} tokens")

            # Verify attempt/2
            t.configure(stage="verification", temperature=0.1)

            result = t.run(
                "Mentally run each test case against this IMPROVED implementation.\n"
                "Trace through carefully. Commit results with metadata:\n"
                "  test_status: 'pass' or 'fail'\n"
                "  tests_passed: number\n"
                "  tests_failed: number\n"
                "  failures: list (if any)",
                max_steps=6,
                profile="full",
                tool_names=["commit", "get_config", "status", "log"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )

            # Check attempt/2 results
            test_results_2 = t.find(metadata_key="test_status", limit=5)
            attempt2_passed = any(
                r.metadata and r.metadata.get("test_status") == "pass"
                for r in test_results_2
            )

            if attempt2_passed:
                print("  Attempt 2: ALL TESTS PASSED")

            # --- Merge winning attempt back to main ---
            print("\n  Merging successful attempt into main...")
            winning_branch = "attempt/2" if attempt2_passed else "attempt/1"
            t.switch("main")
            merge_result = t.merge(winning_branch, strategy="theirs")
            print(f"  Merged '{winning_branch}' -> main "
                  f"(type: {merge_result.merge_type})")

        else:
            # Attempt 1 passed -- merge it back
            print("\n  Merging attempt/1 into main...")
            t.switch("main")
            merge_result = t.merge("attempt/1", strategy="theirs")
            print(f"  Merged 'attempt/1' -> main "
                  f"(type: {merge_result.merge_type})")

        # =============================================================
        # 8. REVIEW STAGE
        #    Quality gate: only reachable if tests passed.
        #    Compress iteration context first to save tokens.
        # =============================================================

        print("\n=== Stage 6: Review ===\n")

        # Compress iteration history before review (save tokens)
        pre_review_tokens = t.status().token_count
        if pre_review_tokens > 1500:
            compress_result = t.compress(
                content="Implementation complete. Email validator written and tested. "
                "Tests cover: valid basic emails, plus-addressing, subdomains, "
                "dot handling, rejection of invalid formats (missing @, spaces, "
                "double dots, missing TLD). All tests passing.",
            )
            post_review_tokens = t.status().token_count
            print(f"  Compressed iteration context: {pre_review_tokens} -> "
                  f"{post_review_tokens} tokens")

        # Record test verification on main so the review gate can find it.
        # Why: get_ancestors() walks only the primary parent chain, so metadata
        # committed on attempt branches is invisible from main after merge.
        # This commit lands after compression, so it won't be wiped.
        any_passed = attempt1_passed or attempt2_passed
        if any_passed:
            t.commit(
                content={
                    "content_type": "note",
                    "text": "All tests passing after merge.",
                },
                message="Post-merge test verification",
                metadata={"test_status": "pass"},
            )
            print("  Recorded test_status=pass on main (post-merge)")

        # Transition to review -- gate checks for passing tests.
        # This is the one harness-driven transition: the agent does NOT have
        # the transition tool during review, so we call t.transition() here
        # to fire the pre_transition gate.
        review_allowed = True
        try:
            t.transition("review")
            print("  Review gate passed -- transition to review allowed")
        except BlockedError as e:
            review_allowed = False
            print(f"  GATE BLOCKED: {e}")
            print("  (Review stage skipped because quality gate rejected transition)")

        if review_allowed:
            t.configure(stage="review", temperature=0.1)

            result = t.run(
                "Perform a final code review of the email validator.\n\n"
                "Evaluate:\n"
                "1. Correctness: Does it handle all specified cases?\n"
                "2. Edge cases: Any patterns we missed?\n"
                "3. Code quality: Is it readable and maintainable?\n"
                "4. Performance: Any unnecessary complexity?\n\n"
                "Commit your review summary with a final assessment.",
                max_steps=6,
                profile="full",
                tool_names=["commit", "get_config", "status", "log"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )

        # =============================================================
        # 9. FINAL STATE -- Show why tract is superior
        # =============================================================

        print(f"\n{'=' * 70}")
        print("FINAL STATE")
        print(f"{'=' * 70}\n")

        # Branches created during workflow
        print("  Branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        # Pinned commits (survived all compressions)
        pinned = t.pinned()
        print(f"\n  Pinned commits (survived compression): {len(pinned)}")
        for p in pinned[:5]:
            print(f"    {p.commit_hash[:8]}  {p.content_type:10s}  {(p.message or '')[:50]}")

        # Searchable history: find all test-related commits
        test_artifacts = t.find(content="validate_email", limit=10)
        print(f"\n  Commits mentioning 'validate_email': {len(test_artifacts)}")

        # Final compiled context stats
        final_status = t.status()
        print(f"\n  Final: {final_status.commit_count} commits, "
              f"{final_status.token_count} tokens")

        # Commit history
        print(f"\n  Log (last 10 commits):")
        for ci in t.log(limit=10):
            meta = ""
            if ci.metadata and "test_status" in ci.metadata:
                meta = f"  [{ci.metadata['test_status']}]"
            print(f"    {ci.commit_hash[:8]}  {ci.content_type:12s}  "
                  f"{(ci.message or '')[:45]}{meta}")

        print(f"\n{'=' * 70}")
        print("WHY TRACT > NAIVE PROMPT CHAINS:")
        print(f"{'=' * 70}")
        print("  1. BRANCHING: Isolated attempt/1 and attempt/2 without losing context")
        print("  2. COMPRESSION: Planning shrank from exploration to summary, tests stayed pinned")
        print("  3. QUALITY GATES: Middleware enforced TDD order (tests before code, pass before review)")
        print("  4. CROSS-DIFF: compare() revealed exactly what changed between attempts")
        print("  5. SEARCH: find() located test results by metadata across all branches")
        print("  6. MERGE: Winning implementation merged cleanly back to main")
        print()


if __name__ == "__main__":
    main()


# --- See also ---
# Basic coding workflow:  workflows/01_coding_assistant.py
# Quality gates:          agent/07_quality_gates.py
# Self-correction:        agent/03_self_correction.py
# Staged workflows:       agent/05_staged_workflow.py
