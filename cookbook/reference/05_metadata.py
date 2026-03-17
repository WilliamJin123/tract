"""Metadata reference: tags, priority annotations, edit-in-place, surgical edits.

Covers: register_tag/tag/untag/get_tags/query_by_tags, annotate() with
PINNED/SKIP/NORMAL/IMPORTANT, system(edit=), tool_result(edit=).
"""

from tract import Priority, Tract
from tract.formatting import pprint_log


def main() -> None:
    # =================================================================
    # 1. Tags: auto-classification + explicit + mutable
    # =================================================================

    t = Tract.open()

    # Auto-classification: every commit gets tags based on content type/role
    sys_ci = t.system("You are a research assistant.")
    print(f"system()    auto-tags: {t.tags.get(sys_ci.commit_hash)}")

    usr_ci = t.user("Summarize the attention paper.")
    print(f"user()      auto-tags: {t.tags.get(usr_ci.commit_hash)}")

    ast_ci = t.assistant("The Transformer replaces recurrence with self-attention.")
    print(f"assistant() auto-tags: {t.tags.get(ast_ci.commit_hash)}")

    # Explicit tags at commit time (merge with auto-tags, immutable)
    obs_ci = t.user("Sparse attention scales linearly.", tags=["observation"])
    print(f"user(tags=) all tags: {t.tags.get(obs_ci.commit_hash)}")

    # Mutable annotation tags (add/remove after the fact)
    t.tags.add(ast_ci.commit_hash, "decision")
    print(f"after tag():   {t.tags.get(ast_ci.commit_hash)}")

    removed = t.tags.remove(ast_ci.commit_hash, "decision")  # returns True if existed
    print(f"after untag(): {t.tags.get(ast_ci.commit_hash)} (removed={removed})")

    # Tag registry: register custom tags with descriptions
    t.tags.register("dead_end", "Agent determined this path was unproductive")
    ci = t.assistant("Approach A failed.")
    t.tags.add(ci.commit_hash, "dead_end")

    t.close()

    # =================================================================
    # 2. Tag queries: query_by_tags + log(tags=)
    # =================================================================

    t = Tract.open()
    t.system("You are a data analyst.")
    t.user("Load the CSV.", tags=["observation"])
    t.assistant("Loaded 10k rows.")
    t.assistant("Q4 revenue is strongest.", tags=["decision"])

    # match="any" (OR): commits with reasoning OR observation
    any_hits = t.tags.query(["reasoning", "observation"], match="any")
    print(f"\nany-match: {len(any_hits)} commits")

    # match="all" (AND): commits with BOTH reasoning AND decision
    all_hits = t.tags.query(["reasoning", "decision"], match="all")
    print(f"all-match: {len(all_hits)} commits")

    # log(tags=) filters the commit log
    reasoning_log = t.search.log(tags=["reasoning"])
    print(f"log(tags=['reasoning']): {len(reasoning_log)} entries")
    t.close()

    # =================================================================
    # 3. Priority annotations: PINNED, SKIP, NORMAL, IMPORTANT
    # =================================================================

    t = Tract.open()

    sys_ci = t.system("You are a research assistant.")
    # system() is auto-PINNED (survives compression)
    print(f"\nsystem default priority: PINNED")

    t.user("What reduces inference cost?")
    verbose = t.assistant("1. Quantization 2. Pruning 3. Distillation ...")
    t.user("Focus on quantization.")
    t.assistant("INT4 cuts memory 4x with <1% accuracy loss.")

    # Unpin the system prompt (allow compression to summarize it)
    t.annotations.set(sys_ci.commit_hash, Priority.NORMAL, reason="temporary persona")

    # Skip verbose content (hidden from compile, still in history)
    t.annotations.set(verbose.commit_hash, Priority.SKIP, reason="user narrowed focus")

    ctx = t.compile()
    print(f"after SKIP: {len(ctx.messages)} messages (verbose hidden)")

    # Reset back to normal
    t.annotations.set(verbose.commit_hash, Priority.NORMAL)
    ctx = t.compile()
    print(f"after reset: {len(ctx.messages)} messages (restored)")

    # Convenience filters
    print(f"skipped: {len(t.search.skipped())} commits")
    print(f"pinned:  {len(t.search.pinned())} commits")

    # Effective priority visible in log()
    pprint_log(list(reversed(t.search.log())))

    t.close()

    # =================================================================
    # 4. Edit-in-place: system(edit=hash), user(edit=hash)
    # =================================================================

    t = Tract.open()

    bad = t.system("Return policy: 60 days.")  # mistake
    t.user("What's the return policy?")
    t.assistant("You can return within 60 days.")

    # Fix the system prompt in-place (original preserved in history)
    fix = t.system("Return policy: 30 days.", edit=bad.commit_hash)
    print(f"\nedit: {bad.commit_hash[:8]} -> {fix.commit_hash[:8]}")

    # compile() now serves the corrected version
    ctx = t.compile()
    sys_msg = ctx.messages[0].content
    assert "30 days" in sys_msg
    print(f"compiled system: {sys_msg}")

    # Both versions preserved in log for audit
    log = t.search.log()
    edit_count = sum(1 for e in log if e.operation.value == "edit")
    print(f"log: {len(log)} entries, {edit_count} edits")
    t.close()

    # =================================================================
    # 5. Surgical tool edits: tool_result(edit=hash)
    # =================================================================

    t = Tract.open()
    t.system("You are a debugging agent.")
    t.user("Find the auth bug.")

    t.assistant("Searching...", metadata={"tool_calls": [
        {"id": "c1", "name": "grep", "arguments": {"pattern": "auth"}},
    ]})
    original = t.tool_result("c1", "grep",
        "login.py:15: def authenticate()\n"
        "login.py:22: if not auth_ldap()\n"
        "session.py:8: from login import authenticate\n"
        "middleware.py:12: from login import authenticate\n"
        "docs/api.md:88: The authenticate() function..."
    )

    before_tokens = t.compile().token_count

    # Trim to only relevant lines (original preserved for audit)
    edited = t.tool_result("c1", "grep",
        "login.py:15: def authenticate()\n"
        "login.py:22: if not auth_ldap()",
        edit=original.commit_hash,
    )

    after_tokens = t.compile().token_count
    print(f"\nsurgical edit: {before_tokens} -> {after_tokens} tokens")
    print(f"original at {original.commit_hash[:8]} preserved in history")
    t.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
