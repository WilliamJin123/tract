"""GC, Rebase, and Merge Hooks

Part 1 — PendingGC: intercept garbage collection to inspect what will
be removed, exclude specific commits from the removal plan, then approve.

Part 2 — PendingRebase: intercept rebase to review the replay plan,
exclude commits you want to drop, and check warnings before approving.

Part 3 — PendingMerge: intercept merge conflicts, inspect resolutions,
edit individual conflict resolutions with edit_resolution(), and steer
the resolver with edit_guidance().

Part 4 — Merge Retry and Validate: use validate() / retry() / auto_retry()
on PendingMerge to check resolutions, re-resolve via LLM when validation
fails, and run the automated retry loop. Shows HookRejection on exhaustion.

Demonstrates: gc(review=True), PendingGC.exclude(), rebase(review=True),
              PendingRebase.exclude(), merge(review=True), PendingMerge,
              edit_resolution(), set_resolution(), edit_guidance(),
              validate(), retry(), auto_retry(), HookRejection,
              t.on("gc"|"rebase"|"merge", handler), pprint()
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.gc import PendingGC
from tract.hooks.merge import PendingMerge
from tract.hooks.rebase import PendingRebase
from tract.hooks.retry import auto_retry
from tract.hooks.validation import HookRejection, ValidationResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# ---------------------------------------------------------------------------
# Part 1: PendingGC — Selective Garbage Collection
# ---------------------------------------------------------------------------

def part1_gc_hooks():
    print("=" * 60)
    print("PART 1 — PendingGC: Selective Garbage Collection")
    print("=" * 60)
    print()
    print("  GC removes orphaned commits (not reachable from any branch).")
    print("  Create orphans by branching, adding commits, then deleting the branch.")

    with Tract.open() as t:
        # Build a main conversation
        t.system("You are a helpful assistant.")
        t.user("Main question.")
        t.assistant("Main answer.")

        # Create a throwaway branch with several commits
        t.branch("throwaway")
        orphan_count = 0
        for i in range(4):
            t.user(f"Throwaway question {i + 1}")
            t.assistant(f"Throwaway answer {i + 1}")
            orphan_count += 2

        # Delete the branch — its commits become orphaned
        t.switch("main")
        t.delete_branch("throwaway", force=True)
        print(f"\n  Deleted 'throwaway' branch — {orphan_count} commits now orphaned")

        # --- review=True: get PendingGC without executing ---
        # orphan_retention_days=0 makes them immediately eligible
        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)

        print(f"\n  PendingGC returned:")
        print(f"    status:            {pending.status}")
        print(f"    commits_to_remove: {len(pending.commits_to_remove)}")
        print(f"    tokens_to_free:    {pending.tokens_to_free}")

        for h in pending.commits_to_remove:
            print(f"      {h[:12]}")

        # --- Exclude one commit: keep it despite being orphaned ---
        if len(pending.commits_to_remove) > 1:
            keep_hash = pending.commits_to_remove[0]
            original_count = len(pending.commits_to_remove)
            pending.exclude(keep_hash)
            print(f"\n  Excluded {keep_hash[:12]} from removal")
            print(f"    commits_to_remove: {len(pending.commits_to_remove)} (was {original_count})")

        # --- Approve the reduced plan ---
        result = pending.approve()
        print(f"\n  Approved! GC complete")
        print(f"    status: {pending.status}")

    # --- Hook handler pattern: auto-exclude by token count ---
    print(f"\n  Hook pattern: protect high-value orphans")

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Main context.")
        t.assistant("Main reply.")

        # Create and delete two branches to make orphans
        for branch_name in ["experiment-a", "experiment-b"]:
            t.branch(branch_name)
            for i in range(3):
                t.user(f"{branch_name} Q{i}")
                t.assistant(f"{branch_name} A{i} — " + "x" * (50 * (i + 1)))
            t.switch("main")
            t.delete_branch(branch_name, force=True)

        def protect_large_orphans(pending: PendingGC):
            """Keep orphans that might have substantial content."""
            # Note: PendingGC doesn't expose per-commit token counts publicly.
            # In practice, you'd use your own tracking or inspect via t.log().
            # Here we demonstrate exclude() by keeping every other commit.
            for i, h in enumerate(list(pending.commits_to_remove)):
                if i % 2 == 0:
                    pending.exclude(h)
                    print(f"    Protected {h[:12]} (kept every other orphan)")
            pending.approve()

        t.on("gc", protect_large_orphans)
        t.gc(orphan_retention_days=0)
        print(f"    GC complete (large orphans preserved)")


# ---------------------------------------------------------------------------
# Part 2: PendingRebase — Review Before Replay
# ---------------------------------------------------------------------------

def part2_rebase_hooks():
    print("\n" + "=" * 60)
    print("PART 2 — PendingRebase: Review Before Replay")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Base question about algorithms.")
        t.assistant("Base answer about algorithms.")

        # Create a feature branch with several commits
        t.branch("feature")
        feature_commits = []
        for i in range(4):
            ci = t.user(f"Feature question {i + 1}")
            feature_commits.append(ci)
            ci2 = t.assistant(f"Feature answer {i + 1}")
            feature_commits.append(ci2)

        # Add a commit on main so rebase has something to do
        t.switch("main")
        t.user("New main question.")
        t.assistant("New main answer.")
        t.switch("feature")

        # --- review=True: get PendingRebase ---
        pending: PendingRebase = t.rebase("main", review=True)

        print(f"\n  PendingRebase returned:")
        print(f"    status:      {pending.status}")
        print(f"    target_base: {pending.target_base[:12]}")
        print(f"    replay_plan: {len(pending.replay_plan)} commits")
        print(f"    warnings:    {len(pending.warnings)}")

        for h in pending.replay_plan:
            print(f"      {h[:12]}")

        # --- Exclude a commit: skip it during replay ---
        drop_hash = pending.replay_plan[0]
        pending.exclude(drop_hash)
        print(f"\n  Excluded {drop_hash[:12]} from replay")
        print(f"    replay_plan: {len(pending.replay_plan)} commits (was {len(pending.replay_plan) + 1})")

        # --- Approve ---
        result = pending.approve()
        print(f"\n  Approved! Rebase complete")
        print(f"    status: {pending.status}")

    # --- Hook handler pattern ---
    print(f"\n  Hook pattern: warn-and-approve")

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Main setup.")
        t.assistant("Main reply.")

        t.branch("experiment")
        for i in range(3):
            t.user(f"Experiment Q{i}")
            t.assistant(f"Experiment A{i}")

        t.switch("main")
        t.user("Main diverged.")
        t.assistant("Main diverged reply.")
        t.switch("experiment")

        def warn_and_approve(pending: PendingRebase):
            """Log the replay plan, then approve."""
            print(f"    [hook] Rebasing {len(pending.replay_plan)} commits onto {pending.target_base[:8]}")
            if pending.warnings:
                for w in pending.warnings:
                    print(f"    [hook] WARNING: {w}")
            pending.approve()

        t.on("rebase", warn_and_approve)
        t.rebase("main")  # Handler fires
        print(f"    Rebase via hook complete")


# ---------------------------------------------------------------------------
# Part 3: PendingMerge — Conflict Resolution Hooks
# ---------------------------------------------------------------------------

def part3_merge_hooks():
    print("\n" + "=" * 60)
    print("PART 3 — PendingMerge: Conflict Resolution Hooks")
    print("=" * 60)
    print()
    print("  Only merges WITH conflicts fire hooks.")
    print("  Fast-forward and clean merges proceed without interception.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Build a conflict: both branches EDIT the same message
        sys_ci = t.system("You are a helpful assistant.")
        user_ci = t.user("What is Python?")
        t.assistant("Python is a programming language.")

        # Feature branch edits the assistant message
        t.branch("feature")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Python is a high-level, interpreted language created by Guido van Rossum.",
        )

        # Main also edits the same message → conflict
        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Python is a versatile language popular in data science and web development.",
        )

        # --- review=True: get PendingMerge ---
        pending: PendingMerge = t.merge("feature", review=True)

        print(f"\n  PendingMerge returned:")
        print(f"    status:        {pending.status}")
        print(f"    source_branch: {pending.source_branch}")
        print(f"    target_branch: {pending.target_branch}")
        print(f"    conflicts:     {len(pending.conflicts)}")
        print(f"    resolutions:   {len(pending.resolutions)} keys")

        # Show each conflict and its current resolution
        for key, resolution in pending.resolutions.items():
            print(f"\n    Conflict key: {key[:12]}")
            print(f"    Resolution:   {resolution[:80]}...")

        # --- edit_resolution: replace one ---
        first_key = list(pending.resolutions.keys())[0]
        pending.edit_resolution(
            first_key,
            "Python is a high-level language popular in data science, web dev, and automation.",
        )
        print(f"\n  After edit_resolution({first_key[:8]}, ...):")
        print(f"    Resolution: {pending.resolutions[first_key][:80]}")

        # --- Guidance ---
        pending.edit_guidance("Prefer the version that mentions more use cases.")
        print(f"\n  guidance: {pending.guidance}")
        print(f"  guidance_source: {pending.guidance_source}")

        # --- Approve ---
        result = pending.approve()
        print(f"\n  Approved! Merge complete")
        print(f"    status: {pending.status}")
        result.pprint()

    # --- Hook handler pattern: auto-resolve ---
    print(f"\n  Hook pattern: auto-pick incoming version")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("What is Rust?")
        t.assistant("Rust is a language.")

        t.branch("feature2")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Rust is a systems programming language focused on safety.",
        )

        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Rust is a modern compiled language.",
        )

        def prefer_incoming(pending: PendingMerge):
            """Always pick the incoming (source branch) version."""
            for conflict in pending.conflicts:
                key = getattr(conflict, "target_hash", None)
                if key and hasattr(conflict, "content_b_text"):
                    pending.set_resolution(key, conflict.content_b_text)
                    print(f"    [hook] Auto-picked incoming for {key[:8]}")
            pending.approve()

        t.on("merge", prefer_incoming)
        result = t.merge("feature2")
        print(f"    Merge via hook complete")
        print(f"    Result type: {type(result).__name__}")


# ---------------------------------------------------------------------------
# Part 4: PendingMerge — Retry and Validate
# ---------------------------------------------------------------------------

def part4_merge_retry_and_validate():
    print("\n" + "=" * 60)
    print("PART 4 — PendingMerge: Retry and Validate")
    print("=" * 60)
    print()
    print("  validate() checks all resolutions are present and non-empty.")
    print("  retry() re-resolves all conflicts via LLM with optional guidance.")
    print("  auto_retry() is the automated validate->retry loop.")

    # -- Manual validate / retry cycle --
    print(f"\n  --- Manual validate/retry cycle ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Build a conflict: both branches EDIT the same message
        sys_ci = t.system("You are a helpful assistant.")
        user_ci = t.user("Explain machine learning.")
        t.assistant("Machine learning is a subset of AI.")

        # Feature branch edits the assistant message
        t.branch("feature-ml")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Machine learning uses statistical models to learn patterns from data.",
        )

        # Main also edits the same message -> conflict
        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Machine learning enables computers to improve through experience.",
        )

        # --- review=True: get PendingMerge ---
        pending: PendingMerge = t.merge("feature-ml", review=True)

        print(f"\n  PendingMerge returned:")
        print(f"    status:      {pending.status}")
        print(f"    conflicts:   {len(pending.conflicts)}")
        print(f"    resolutions: {len(pending.resolutions)} keys")

        # --- Clear resolutions to demonstrate validate() failing ---
        # (In real use, resolutions might be missing if review=True
        #  was called without a resolver, or if the resolver failed.)
        saved_resolutions = dict(pending.resolutions)
        pending.resolutions.clear()

        print(f"\n  Cleared resolutions to simulate missing data:")
        print(f"    resolutions: {len(pending.resolutions)} keys")

        # --- validate() should fail: no resolutions ---
        result: ValidationResult = pending.validate()
        print(f"\n  validate() #1:")
        print(f"    passed:    {result.passed}")
        print(f"    diagnosis: {result.diagnosis}")
        print(f"    index:     {result.index}")

        # --- retry() re-resolves all conflicts via LLM ---
        print(f"\n  Calling retry(guidance='Be concise')...")
        pending.retry(guidance="Be concise")
        print(f"    resolutions after retry: {len(pending.resolutions)} keys")
        for key, resolution in pending.resolutions.items():
            print(f"    {key[:12]}: {resolution[:80]}")

        # --- validate() again: should pass now ---
        result2: ValidationResult = pending.validate()
        print(f"\n  validate() #2:")
        print(f"    passed:    {result2.passed}")
        print(f"    diagnosis: {result2.diagnosis}")

        # --- Approve ---
        merge_result = pending.approve()
        print(f"\n  Approved! Merge complete")
        print(f"    status: {pending.status}")

    # -- auto_retry(): automated validate->retry loop --
    print(f"\n  --- auto_retry(): automated loop ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("What is deep learning?")
        t.assistant("Deep learning uses neural networks.")

        t.branch("feature-dl")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Deep learning uses multi-layered neural networks for representation learning.",
        )

        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Deep learning is a branch of ML with layered architectures.",
        )

        pending2: PendingMerge = t.merge("feature-dl", review=True)

        print(f"\n  PendingMerge with {len(pending2.conflicts)} conflicts")
        print(f"  Calling auto_retry(pending, max_retries=3)...")

        result = auto_retry(pending2, max_retries=3)
        print(f"\n  auto_retry returned: {type(result).__name__}")
        print(f"    pending status: {pending2.status}")

        if isinstance(result, HookRejection):
            print(f"    REJECTED: {result.reason}")
        else:
            print(f"    Merge approved and complete")

    # -- HookRejection on exhausted retries --
    print(f"\n  --- HookRejection when retries exhaust ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("Explain transformers.")
        t.assistant("Transformers are a neural network architecture.")

        t.branch("feature-tx")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Transformers use self-attention to process sequences in parallel.",
        )

        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Transformers replaced RNNs for most NLP tasks.",
        )

        pending3: PendingMerge = t.merge("feature-tx", review=True)

        # Force resolutions to be empty so every retry's validate() fails
        # (simulates a resolver that keeps producing bad output)
        original_retry = pending3.retry

        def broken_retry(**kwargs):
            """Simulate a resolver that always produces empty resolutions."""
            original_retry(**kwargs)
            # Wipe resolutions after each retry to force failure
            pending3.resolutions.clear()

        pending3.retry = broken_retry

        print(f"\n  Simulating a resolver that always fails...")
        print(f"  Calling auto_retry(pending, max_retries=2)...")

        result = auto_retry(pending3, max_retries=2)
        print(f"\n  auto_retry returned: {type(result).__name__}")

        if isinstance(result, HookRejection):
            print(f"    reason:           {result.reason}")
            print(f"    rejection_source: {result.rejection_source}")
            print(f"    pending status:   {pending3.status}")
            if result.metadata:
                print(f"    metadata:         {result.metadata}")


# ---------------------------------------------------------------------------

def main():
    part1_gc_hooks()
    part2_rebase_hooks()
    part3_merge_hooks()
    part4_merge_retry_and_validate()


if __name__ == "__main__":
    main()
