"""Branching: Create, Switch, Merge, Import, Rebase

Quick reference for branch operations:
- Branch lifecycle: branch(), switch(), list_branches(), delete_branch()
- Merge strategies: fast-forward, clean, conflict
- Conflict resolution: resolutions dict + commit_merge()
- import_commit() — cherry-pick across branches
- rebase() — replay commits onto a new base
"""

from tract import Tract, InstructionContent, CommitOperation
from tract.formatting import pprint_log


def main() -> None:
    # =================================================================
    # 1. BRANCH LIFECYCLE
    # =================================================================
    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Initial message on main.")

    # Create + switch to new branch (default: switch=True)
    t.branches.create("feature")
    print(f"Current: {t.current_branch}")  # "feature"

    t.user("Work on feature branch.")

    ctx = t.compile()
    ctx.pprint(style="compact")

    # Switch back
    t.branches.switch("main")
    print(f"Current: {t.current_branch}")  # "main"

    ctx = t.compile()
    ctx.pprint(style="compact")

    # Create without switching
    t.branches.create("draft", switch=False)
    print(f"Still on: {t.current_branch}")  # "main"

    # List all branches
    for b in t.branches.list():
        marker = "*" if b.is_current else " "
        print(f"  {marker} {b.name:12s} @ {b.commit_hash[:8]}")

    # Delete a branch
    t.branches.delete("draft", force=True)
    t.close()

    # =================================================================
    # 2. MERGE — FAST-FORWARD
    # =================================================================
    # Main hasn't moved since branching -> pointer slides forward.
    t = Tract.open()
    t.system("Assistant.")
    t.user("Base message.")

    t.branches.create("feature")
    t.user("Feature work.")

    t.branches.switch("main")
    result = t.merge("feature")
    print(f"Merge type: {result.merge_type}")  # "fast_forward"

    ctx = t.compile()
    ctx.pprint(style="compact")

    result.pprint()
    t.close()

    # =================================================================
    # 3. MERGE — CLEAN (diverged, APPEND-only)
    # =================================================================
    # Both branches added content, no edits to same commits.
    t = Tract.open()
    t.system("Assistant.")
    t.user("Shared base.")

    t.branches.create("feature")
    t.user("Feature content.")

    t.branches.switch("main")
    t.user("Main content.")

    result = t.merge("feature")
    print(f"Merge type: {result.merge_type}")  # "clean"

    ctx = t.compile()
    ctx.pprint(style="compact")

    t.close()

    # =================================================================
    # 4. MERGE — CONFLICT (both edit same commit)
    # =================================================================
    t = Tract.open()
    sys_ci = t.system("You are helpful.")
    t.user("Hello.")

    # Feature edits the system prompt
    t.branches.create("formal")
    t.commit(
        InstructionContent(text="You are a formal academic assistant."),
        operation=CommitOperation.EDIT,
        edit_target=sys_ci.commit_hash,
    )

    # Main also edits the system prompt -> conflict
    t.branches.switch("main")
    t.commit(
        InstructionContent(text="You are a casual friendly assistant."),
        operation=CommitOperation.EDIT,
        edit_target=sys_ci.commit_hash,
    )

    # Merge returns MergeResult; conflict merges have committed=False
    result = t.merge("formal")
    if result.conflicts and not result.committed:
        # Inspect conflicts and set resolutions
        for conflict in result.conflicts:
            if conflict.target_hash:
                result.resolutions[conflict.target_hash] = (
                    "You are a knowledgeable yet approachable assistant."
                )
        # Finalize the merge
        result = t.commit_merge(result)
        print(f"Conflict resolved, committed: {result.committed}")

        ctx = t.compile()
        ctx.pprint(style="compact")

    # With LLM resolver (auto-resolves conflicts, requires LLM config):
    # result = t.merge("formal", resolver="llm")

    t.close()

    # =================================================================
    # 5. MERGE OPTIONS — no_ff, delete_branch
    # =================================================================
    t = Tract.open()
    t.system("Assistant.")
    t.branches.create("quick-fix")
    t.user("Fix content.")

    t.branches.switch("main")
    result = t.merge("quick-fix", no_ff=True, delete_branch=True)
    # no_ff=True -> forces merge commit even when FF is possible
    # delete_branch=True -> auto-deletes source branch after merge
    branches = [b.name for b in t.branches.list()]
    print(f"Branches after: {branches}")  # ["main"] — quick-fix deleted
    t.close()

    # =================================================================
    # 6. IMPORT COMMIT (cherry-pick)
    # =================================================================
    # Copy specific commits from one branch to another.
    t = Tract.open()
    t.system("Assistant.")
    t.user("Main base.")

    t.branches.create("experiment")
    good_ci = t.user("This insight is worth keeping.")

    t.branches.switch("main")
    ir = t.import_commit(good_ci.commit_hash)
    print(f"Original: {ir.original_commit.commit_hash[:8]}")
    print(f"New copy: {ir.new_commit.commit_hash[:8]}")
    # Same content, new hash (different lineage). Source branch untouched.
    t.close()

    # =================================================================
    # 7. REBASE — replay commits onto new base
    # =================================================================
    t = Tract.open()
    t.system("Assistant.")
    t.user("Shared base.")

    t.branches.create("examples")
    t.user("Example 1.")
    t.user("Example 2.")

    t.branches.switch("main")
    t.user("New main content.")  # main advances

    t.branches.switch("examples")
    result = t.rebase("main")

    print(f"Replayed: {len(result.replayed_commits)} commits")
    print(f"New HEAD: {result.new_head[:8]}")
    # examples branch now sits on top of main's latest commit.
    # Hash changes (new lineage) but content is preserved.
    for orig, new in zip(result.original_commits, result.replayed_commits):
        print(f"  {orig.commit_hash[:8]} -> {new.commit_hash[:8]}")

    t.close()
    print("Branching reference complete.")


if __name__ == "__main__":
    main()
