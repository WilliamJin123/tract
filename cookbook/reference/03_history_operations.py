"""History Operations: Log, Diff, Reset, Edit

Quick reference for inspecting and modifying commit history:
- t.log() — walk history, CommitInfo fields, pinned/skipped filters
- t.show() — inspect a single commit
- t.diff() — compare compiled contexts at two points
- t.reset() — roll back HEAD, undo via ORIG_HEAD
- Edit — t.assistant(edit=hash), t.revise(), edit_history(), restore()
"""

from tract import Tract, InstructionContent, CommitOperation


def main():
    t = Tract.open()

    # Build a small conversation
    sys_ci = t.system("You are a geography tutor.")
    u1 = t.user("What is the capital of France?")
    a1 = t.assistant("The capital of France is Paris.")
    u2 = t.user("And Germany?")
    a2 = t.assistant("The capital of Germany is Berlin.")

    # =================================================================
    # 1. LOG — walk commit history from HEAD backward
    # =================================================================
    history = t.log()                    # list of CommitInfo, newest first
    history_limited = t.log(limit=3)     # last 3 commits only

    for entry in history:
        print(entry)                     # "hash[:8] message"

    # Chronological order: reversed()
    for entry in reversed(history):
        # Fields: commit_hash, content_type, message, effective_priority,
        #         generation_config, created_at, parent_hash, ...
        pri = entry.effective_priority or "normal"
        print(f"  {entry.commit_hash[:8]}  [{pri:<7}]  {entry.message or ''}")

    # Quick filters
    pinned = t.pinned()    # commits that survive compression (instructions, etc.)
    skipped = t.skipped()  # commits hidden from compile (reasoning, etc.)

    # =================================================================
    # 2. SHOW — inspect a single commit with full content
    # =================================================================
    t.show(a1)  # prints rich detail for that commit

    # Get raw content
    content = t.get_content(a1)
    print(f"Content: {content}")

    # =================================================================
    # 3. DIFF — compare compiled context at two points
    # =================================================================
    # diff(A, B) compares FULL compiled context at commit A vs commit B
    result = t.diff(u1.commit_hash, a2.commit_hash)
    result.pprint()                  # full diff view
    result.pprint(stat_only=True)    # summary like "git diff --stat"

    # diff(A) compares A vs HEAD
    result2 = t.diff(u1.commit_hash)
    # result.open()                  # open in VS Code / $EDITOR

    # DiffResult fields:
    #   result.added      — messages only in B
    #   result.removed    — messages only in A
    #   result.modified   — messages changed between A and B
    #   result.unchanged  — same in both

    # =================================================================
    # 4. RESET — roll HEAD back to an earlier commit
    # =================================================================
    head_before = t.log()[0].commit_hash
    t.reset(u1.commit_hash)  # HEAD now points to u1

    ctx = t.compile()
    print(f"After reset: {len(ctx.messages)} messages")  # system + user1 only

    # Undo via ORIG_HEAD (saved automatically on reset)
    t.reset("ORIG_HEAD")
    ctx = t.compile()
    print(f"After undo: {len(ctx.messages)} messages")   # all restored

    # =================================================================
    # 5. EDIT — modify previous commits without losing history
    # =================================================================

    # Style 1: t.assistant(edit=hash) — replace content of a prior commit
    fix = t.assistant(
        "Paris is the capital of France, also known as the City of Light.",
        edit=a1.commit_hash,
        message="Add City of Light detail",
    )
    # Creates a new EDIT commit; original stays in history. Compile uses latest edit.

    # Style 2: t.revise(hash, prompt) — LLM-driven rewrite (requires LLM config)
    # e = t.revise(a1.commit_hash, "Add info about the Eiffel Tower")

    # View edit chain
    versions = t.edit_history(a1.commit_hash)  # [original, edit1, edit2, ...]
    for i, v in enumerate(versions):
        label = "ORIGINAL" if i == 0 else f"EDIT {i}"
        print(f"  v{i} ({label}) [{v.commit_hash[:8]}] {v.message or ''}")

    # Restore an earlier version (creates a new edit, preserves full history)
    restored = t.restore(a1.commit_hash, version=0)  # back to original text
    print(f"Restored to v0: {restored.commit_hash[:8]}")

    # The compiled context always uses the latest edit for each target
    ctx = t.compile()
    ctx.pprint(style="chat")

    t.close()
    print("History operations reference complete.")


if __name__ == "__main__":
    main()
