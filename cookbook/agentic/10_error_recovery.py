"""Error Recovery via DAG History

When an agent makes a bad commit or goes down a wrong path, tract's DAG
operations let it revert and retry from a known-good state. Two patterns:

  1. Rollback After Bad Commit -- post_commit middleware detects a problematic
     commit and resets HEAD to the parent, isolating the bad node in the DAG
  2. Branch-Isolate-Retry      -- agent works on trial branches, abandons
     failures, and merges the first successful attempt back to main

Both patterns exploit the fact that tract's DAG is append-only: bad commits
are never deleted, just made unreachable from HEAD. This preserves full
audit history while keeping the compiled context clean.

Requires: LLM API key
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, MiddlewareContext
from tract.formatting import pprint_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# ---------------------------------------------------------------------------
# Section 1 -- Rollback After Bad Commit
# ---------------------------------------------------------------------------
# An agent builds an API design step by step. A post_commit middleware
# watches for a forbidden pattern ("DEPRECATED_PATTERN") and automatically
# resets HEAD to the parent commit when one is detected. The bad commit
# stays in the DAG (audit trail) but is unreachable from HEAD, so the
# compiled context never includes it.

def section_rollback() -> None:
    print("=" * 70)
    print("  Section 1: Rollback After Bad Commit")
    print("=" * 70)
    print("  Middleware auto-reverts commits containing a forbidden pattern.")
    print()

    rollback_log: list[str] = []

    def rollback_bad_commits(ctx: MiddlewareContext):
        """Post-commit guard: revert if commit contains a forbidden pattern."""
        if not ctx.commit:
            return
        content = ctx.tract.get_content(ctx.commit)
        if content and "DEPRECATED_PATTERN" in str(content):
            # Walk log to find parent -- log returns newest-first
            history = ctx.tract.log(limit=2)
            if len(history) >= 2:
                parent_hash = history[1].commit_hash
                ctx.tract.reset(target=parent_hash)
                msg = f"Rolled back {ctx.commit.commit_hash[:8]} -> {parent_hash[:8]}"
                rollback_log.append(msg)
                print(f"  >> {msg}")

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
        auto_message=llm.small,
    ) as t:
        t.middleware.add("post_commit", rollback_bad_commits)

        t.system(
            "You are designing a REST API for a task management system. "
            "Commit each design component separately."
        )

        # --- Step 1: Good commit (endpoint design) ---
        t.commit(
            content="API endpoints: POST /tasks, GET /tasks/{id}, "
            "PUT /tasks/{id}, DELETE /tasks/{id}. "
            "All endpoints return JSON. Auth via Bearer token.",
            message="Endpoint design",
            content_type="artifact",
        )
        print("  Committed: Endpoint design")

        # --- Step 2: Good commit (data model) ---
        t.commit(
            content="Task model: id (uuid), title (str), status (enum: "
            "todo/in_progress/done), assignee (str), created_at (datetime). "
            "Stored in PostgreSQL with UUID primary key.",
            message="Data model",
            content_type="artifact",
        )
        print("  Committed: Data model")

        # --- Step 3: Bad commit (contains forbidden pattern) ---
        print("\n  Committing a bad decision (uses DEPRECATED_PATTERN)...")
        t.commit(
            content="Error handling: use DEPRECATED_PATTERN for all error "
            "responses. Return XML error bodies with nested status codes. "
            "Clients must parse XML even though the API is JSON.",
            message="Error handling (bad approach)",
            content_type="artifact",
        )
        # Middleware fires post_commit and rolls back automatically

        # --- Step 4: Good commit (correct error handling) ---
        print("\n  Committing the correct error handling approach...")
        t.commit(
            content="Error handling: use RFC 7807 Problem Details JSON. "
            "All errors return {type, title, status, detail, instance}. "
            "Standard HTTP status codes. Content-Type: application/problem+json.",
            message="Error handling (RFC 7807)",
            content_type="artifact",
        )
        print("  Committed: Error handling (RFC 7807)")

        # --- Show results ---
        print(f"\n  Rollbacks triggered: {len(rollback_log)}")
        for entry in rollback_log:
            print(f"    - {entry}")

        # The compiled context should NOT contain the bad commit
        compiled = t.compile()
        compiled_text = str(compiled.to_dicts())
        has_deprecated = "DEPRECATED_PATTERN" in compiled_text
        print(f"\n  Compiled context contains forbidden pattern: {has_deprecated}")
        print(f"  Compiled context: {compiled.commit_count} commits, "
              f"{compiled.token_count} tokens")

        # The log from HEAD shows only good commits
        print("\n  Commit log (reachable from HEAD):")
        pprint_log(t.log(limit=10))

        print()


# ---------------------------------------------------------------------------
# Section 2 -- Branch-Isolate-Retry Pattern
# ---------------------------------------------------------------------------
# The agent attempts a task on a trial branch. If the result fails a
# verification check, the trial is abandoned and a new branch is created
# with adjusted directives. On success, the winning branch is merged back
# to main. The DAG retains the full topology: main, trial/1 (abandoned),
# trial/2 (merged).

def section_branch_retry() -> None:
    print("=" * 70)
    print("  Section 2: Branch-Isolate-Retry Pattern")
    print("=" * 70)
    print("  Agent retries on fresh branches until verification passes.")
    print()

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
        auto_message=llm.small,
    ) as t:
        t.system(
            "You are a software architect designing a caching strategy "
            "for a high-traffic web application."
        )

        # Seed main with the problem statement
        t.commit(
            content="Requirements: design a caching layer for an e-commerce "
            "product catalog. 50k products, 10k req/sec reads, 100 writes/sec. "
            "Must handle cache invalidation correctly. Budget: single Redis node.",
            message="Caching requirements",
            content_type="artifact",
        )
        print("  Seeded main with requirements.\n")

        log = StepLogger()
        winning_branch = None

        for attempt in range(1, 4):
            branch_name = f"trial/{attempt}"
            t.branch(branch_name, switch=True)
            print(f"  --- Trial {attempt} (branch: {branch_name}) ---")

            # Add progressively better guidance on each retry
            if attempt == 1:
                hint = "Design a caching strategy. Focus on simplicity."
            elif attempt == 2:
                hint = (
                    "Design a caching strategy. The previous attempt was too "
                    "simplistic -- address cache invalidation on writes and "
                    "race conditions during concurrent updates."
                )
            else:
                hint = (
                    "Design a caching strategy with: (1) write-through for "
                    "consistency, (2) TTL-based expiry as safety net, "
                    "(3) pub/sub invalidation for multi-instance deployments. "
                    "Include specific Redis commands."
                )

            result = t.llm.run(
                hint + "\n\nCommit your design as a single artifact.",
                max_steps=6,
                max_tokens=1024,
                tool_names=["commit", "status"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )

            # --- Verification: ask the LLM to judge the design ---
            verification = t.llm.chat(
                "Review the caching design above. Does it address ALL of: "
                "(1) cache invalidation on writes, "
                "(2) race conditions / thundering herd, "
                "(3) consistency guarantees? "
                "Reply YES if all three are covered, NO otherwise. "
                "One word only.",
                max_tokens=10,
            )
            verdict = (verification.text or "").strip().upper()
            passed = "YES" in verdict

            if passed:
                winning_branch = branch_name
                t.switch("main")
                t.merge(branch_name, strategy="theirs")
                print(f"\n  Trial {attempt} PASSED -- merged to main.\n")
                break
            else:
                t.switch("main")
                print(f"  Trial {attempt} FAILED verification "
                      f"(verdict: {verdict}). Abandoned.\n")

        if not winning_branch:
            print("  All trials failed. Using last attempt as best effort.")
            t.merge("trial/3", strategy="theirs")
            winning_branch = "trial/3"

        # --- Show final branch topology ---
        print("\n  Branch topology:")
        for b in t.list_branches():
            marker = " (merged)" if b.name == winning_branch else ""
            marker = marker if b.name != "main" else " (current)"
            # Check if branch was abandoned (not merged)
            if b.name.startswith("trial/") and b.name != winning_branch:
                marker = " (abandoned)"
            print(f"    - {b.name}{marker}")

        # --- Final compiled context (main only) ---
        status = t.status()
        print(f"\n  Main branch: {status.commit_count} commits, "
              f"{status.token_count} tokens")
        print("\n  Final log (main):")
        pprint_log(t.log(limit=10))

        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    print()
    print("=" * 70)
    print("  Error Recovery via DAG History")
    print("  Two patterns for reverting bad work and retrying cleanly.")
    print("=" * 70)
    print()

    section_rollback()
    section_branch_retry()

    print("\nDone. Both error recovery patterns demonstrated.")


if __name__ == "__main__":
    main()


# --- See also ---
# DAG operations (no LLM):      reference/03_dag_operations.py
# Middleware reference:          reference/02_middleware.py
# Checkpoint/resume:             agentic/08_checkpoint_resume.py
