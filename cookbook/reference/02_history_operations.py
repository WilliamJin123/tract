"""History Operations: Log, Diff, Reset, Edit

Quick reference for inspecting and modifying commit history:
- t.search.log() — walk history, CommitInfo fields, pinned/skipped filters
- t.search.show() — inspect a single commit
- t.search.diff() — compare compiled contexts at two points
- t.branches.reset() — roll back HEAD, undo via ORIG_HEAD
- Edit — t.assistant(edit=hash), t.llm.revise(), edit_history(), restore()
"""

from tract import Tract, InstructionContent, CommitOperation
from tract.formatting import pprint_log


def main() -> None:
    t = Tract.open()

    # Build a small conversation
    sys_ci = t.system("You are a geography tutor.")
    u1 = t.user("What is the capital of France?")
    a1 = t.assistant("The capital of France is Paris.")
    u2 = t.user("And Germany?")
    a2 = t.assistant("The capital of Germany is Berlin.")

    print("=== Conversation ===\n")
    pprint_log([sys_ci, u1, a1, u2, a2])

    # =================================================================
    # 1. LOG — walk commit history from HEAD backward
    # =================================================================
    history = t.search.log()                    # list of CommitInfo, newest first
    print(f"\n=== 1. Log ({len(history)} commits) ===\n")
    history_limited = t.search.log(limit=3)     # last 3 commits only

    pprint_log(history)

    # Chronological order: reversed()
    pprint_log(list(reversed(history)))

    # Quick filters
    pinned = t.search.pinned()    # commits that survive compression (instructions, etc.)
    skipped = t.search.skipped()  # commits hidden from compile (reasoning, etc.)
    print(f"\n  Pinned: {len(pinned)}, Skipped: {len(skipped)}")

    # =================================================================
    # 2. SHOW — inspect a single commit with full content
    # =================================================================
    print(f"\n=== 2. Show ===\n")
    t.search.show(a1)  # prints rich detail for that commit

    # Get raw content
    content = t.search.get_content(a1)
    print(f"Content: {content}")

    # =================================================================
    # 3. DIFF — compare compiled context at two points
    # =================================================================
    print(f"\n=== 3. Diff ===\n")
    # diff(A, B) compares FULL compiled context at commit A vs commit B
    result = t.search.diff(u1.commit_hash, a2.commit_hash)
    result.pprint()                  # full diff view
    result.pprint(stat_only=True)    # summary like "git diff --stat"

    # diff(A) compares A vs HEAD
    result2 = t.search.diff(u1.commit_hash)
    # result.open()                  # open in VS Code / $EDITOR

    # DiffResult fields:
    #   result.message_diffs  — list of MessageDiff entries (per-message changes)
    #   result.stat           — DiffStat with aggregate counts:
    #     .messages_added     — messages only in B
    #     .messages_removed   — messages only in A
    #     .messages_modified  — messages changed between A and B
    #     .messages_unchanged — same in both
    #     .total_token_delta  — net token change (B - A)

    # =================================================================
    # 4. RESET — roll HEAD back to an earlier commit
    # =================================================================
    print(f"\n=== 4. Reset ===\n")
    head_before = t.search.log()[0].commit_hash
    t.branches.reset(u1.commit_hash)  # HEAD now points to u1

    ctx = t.compile()
    print(f"After reset: {len(ctx.messages)} messages")  # system + user1 only

    # Undo via ORIG_HEAD (saved automatically on reset)
    t.branches.reset("ORIG_HEAD")
    ctx = t.compile()
    print(f"After undo: {len(ctx.messages)} messages")   # all restored

    # =================================================================
    # 5. EDIT — modify previous commits without losing history
    # =================================================================
    print(f"\n=== 5. Edit ===\n")

    # Style 1: t.assistant(edit=hash) — replace content of a prior commit
    fix = t.assistant(
        "Paris is the capital of France, also known as the City of Light.",
        edit=a1.commit_hash,
        message="Add City of Light detail",
    )
    # Creates a new EDIT commit; original stays in history. Compile uses latest edit.

    # Style 2: t.llm.revise(hash, prompt) — LLM-driven rewrite (requires LLM config)
    # e = t.llm.revise(a1.commit_hash, "Add info about the Eiffel Tower")

    # View edit chain
    versions = t.search.edit_history(a1.commit_hash)  # [original, edit1, edit2, ...]
    pprint_log(versions)

    # Restore an earlier version (creates a new edit, preserves full history)
    restored = t.search.restore(a1.commit_hash, version=0)  # back to original text
    print(f"Restored to v0: {restored.commit_hash[:8]}")

    # The compiled context always uses the latest edit for each target
    ctx = t.compile()
    ctx.pprint(style="chat")

    t.close()
    print("History operations reference complete.")


if __name__ == "__main__":
    main()
