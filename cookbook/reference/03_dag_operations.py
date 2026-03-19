"""DAG Operations: History, Branching, Compression, and GC

Comprehensive reference for inspecting, modifying, and maintaining the
commit DAG. All scenarios run without an LLM — no API keys needed.

Covers:
  History    — log, show, diff, reset, edit (replace/revise/edit_history/restore)
  Branching  — create, switch, list, delete, merge (FF/clean/conflict),
               merge options (no_ff, delete_branch), import_commit, rebase
  Compression — manual compress, range compress with preserve, guided
                priorities (IMPORTANT + retain), GC with retention policies
  Compile     — reorder via compile(order=), selective compress_tool_calls
"""

from tract import Tract, InstructionContent, CommitOperation, Priority
from tract.formatting import pprint_log


# =====================================================================
# HISTORY
# =====================================================================


def scenario_log() -> None:
    """Log — walk commit history from HEAD backward."""
    print("\n=== Log ===\n")

    t = Tract.open()
    sys_ci = t.system("You are a geography tutor.")
    u1 = t.user("What is the capital of France?")
    a1 = t.assistant("The capital of France is Paris.")
    u2 = t.user("And Germany?")
    a2 = t.assistant("The capital of Germany is Berlin.")

    # Full history, newest first
    history = t.log()
    print(f"Total commits: {len(history)}")
    pprint_log(history)

    # Limited to last 3
    history_limited = t.log(limit=3)
    print(f"Limited: {len(history_limited)} commits")

    # Chronological (oldest first)
    pprint_log(list(reversed(history)))

    # Quick filters
    pinned = t.search.pinned()
    skipped = t.search.skipped()
    print(f"Pinned: {len(pinned)}, Skipped: {len(skipped)}")

    t.close()


def scenario_show() -> None:
    """Show — inspect a single commit with full content."""
    print("\n=== Show ===\n")

    t = Tract.open()
    t.system("You are a geography tutor.")
    t.user("What is the capital of France?")
    a1 = t.assistant("The capital of France is Paris.")

    # Rich detail view
    t.show(a1)

    # Raw content
    content = t.get_content(a1)
    print(f"Content: {content}")

    t.close()


def scenario_diff() -> None:
    """Diff — compare compiled context at two points."""
    print("\n=== Diff ===\n")

    t = Tract.open()
    t.system("You are a geography tutor.")
    u1 = t.user("What is the capital of France?")
    t.assistant("The capital of France is Paris.")
    t.user("And Germany?")
    a2 = t.assistant("The capital of Germany is Berlin.")

    # diff(A, B) compares FULL compiled context at commit A vs commit B
    result = t.diff(u1.commit_hash, a2.commit_hash)
    result.pprint()
    result.pprint(stat_only=True)

    # diff(A) compares A vs HEAD
    result2 = t.diff(u1.commit_hash)

    # DiffResult fields:
    #   result.message_diffs  — per-message changes
    #   result.stat           — DiffStat with:
    #     .messages_added / .messages_removed / .messages_modified
    #     .messages_unchanged / .total_token_delta
    print(f"Messages added between u1 and HEAD: {result2.stat.messages_added}")

    t.close()


def scenario_reset() -> None:
    """Reset — roll HEAD back to an earlier commit, undo via ORIG_HEAD."""
    print("\n=== Reset ===\n")

    t = Tract.open()
    t.system("You are helpful.")
    u1 = t.user("First question.")
    t.assistant("First answer.")
    t.user("Second question.")
    t.assistant("Second answer.")

    t.reset(u1.commit_hash)
    ctx = t.compile()
    print(f"After reset: {len(ctx.messages)} messages")  # system + user1 only

    # Undo via ORIG_HEAD (saved automatically on reset)
    t.reset("ORIG_HEAD")
    ctx = t.compile()
    print(f"After undo:  {len(ctx.messages)} messages")  # all restored

    t.close()


def scenario_edit() -> None:
    """Edit — modify previous commits without losing history."""
    print("\n=== Edit ===\n")

    t = Tract.open()
    t.system("You are a geography tutor.")
    t.user("What is the capital of France?")
    a1 = t.assistant("The capital of France is Paris.")

    # Style 1: t.assistant(edit=hash) — replace content of a prior commit
    fix = t.assistant(
        "Paris is the capital of France, also known as the City of Light.",
        edit=a1.commit_hash,
        message="Add City of Light detail",
    )
    # Creates a new EDIT commit; original stays in history. Compile uses latest.

    # Style 2: t.llm.revise(hash, prompt) — LLM-driven rewrite (requires LLM)
    # e = t.llm.revise(a1.commit_hash, "Add info about the Eiffel Tower")

    # View edit chain: [original, edit1, edit2, ...]
    versions = t.edit_history(a1.commit_hash)
    pprint_log(versions)

    # Restore an earlier version (creates a new edit, preserves full history)
    restored = t.restore(a1.commit_hash, version=0)
    print(f"Restored to v0: {restored.commit_hash[:8]}")

    # Compiled context always uses the latest edit for each target
    ctx = t.compile()
    ctx.pprint(style="chat")

    t.close()


# =====================================================================
# BRANCHING
# =====================================================================


def scenario_branch_lifecycle() -> None:
    """Branch lifecycle — create, switch, list, delete."""
    print("\n=== Branch Lifecycle ===\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Initial message on main.")

    # Create + switch (default: switch=True)
    t.branch("feature")
    print(f"Current: {t.current_branch}")  # "feature"

    t.user("Work on feature branch.")
    ctx = t.compile()
    ctx.pprint(style="chat")

    # Switch back
    t.switch("main")
    print(f"Current: {t.current_branch}")  # "main"

    # Create without switching
    t.branch("draft", switch=False)
    print(f"Still on: {t.current_branch}")  # "main"

    # List all branches
    for b in t.list_branches():
        marker = "*" if b.is_current else " "
        print(f"  {marker} {b.name:12s} @ {b.commit_hash[:8]}")

    # Delete a branch
    t.delete_branch("draft", force=True)

    t.close()


def scenario_merge_fast_forward() -> None:
    """Merge — fast-forward (main hasn't moved since branching)."""
    print("\n=== Merge: Fast-Forward ===\n")

    t = Tract.open()
    t.system("Assistant.")
    t.user("Base message.")

    t.branch("feature")
    t.user("Feature work.")

    t.switch("main")
    result = t.merge("feature")
    print(f"Merge type: {result.merge_type}")  # "fast_forward"

    result.pprint()
    t.close()


def scenario_merge_clean() -> None:
    """Merge — clean (diverged, APPEND-only, no overlapping edits)."""
    print("\n=== Merge: Clean ===\n")

    t = Tract.open()
    t.system("Assistant.")
    t.user("Shared base.")

    t.branch("feature")
    t.user("Feature content.")

    t.switch("main")
    t.user("Main content.")

    result = t.merge("feature")
    print(f"Merge type: {result.merge_type}")  # "clean"

    ctx = t.compile()
    ctx.pprint(style="chat")

    t.close()


def scenario_merge_conflict() -> None:
    """Merge — conflict (both branches edit the same commit)."""
    print("\n=== Merge: Conflict ===\n")

    t = Tract.open()
    sys_ci = t.system("You are helpful.")
    t.user("Hello.")

    # Feature edits the system prompt
    t.branch("formal")
    t.commit(
        InstructionContent(text="You are a formal academic assistant."),
        operation=CommitOperation.EDIT,
        edit_target=sys_ci.commit_hash,
    )

    # Main also edits the system prompt -> conflict
    t.switch("main")
    t.commit(
        InstructionContent(text="You are a casual friendly assistant."),
        operation=CommitOperation.EDIT,
        edit_target=sys_ci.commit_hash,
    )

    result = t.merge("formal")
    if result.conflicts and not result.committed:
        # Inspect conflicts and provide resolutions
        for conflict in result.conflicts:
            if conflict.target_hash:
                result.resolutions[conflict.target_hash] = (
                    "You are a knowledgeable yet approachable assistant."
                )
        # Finalize the merge
        result = t.commit_merge(result)
        print(f"Conflict resolved, committed: {result.committed}")

        ctx = t.compile()
        ctx.pprint(style="chat")

    # With LLM resolver (auto-resolves conflicts, requires LLM):
    # result = t.merge("formal", resolver="llm")

    t.close()


def scenario_merge_options() -> None:
    """Merge options — no_ff, delete_branch."""
    print("\n=== Merge Options ===\n")

    t = Tract.open()
    t.system("Assistant.")
    t.branch("quick-fix")
    t.user("Fix content.")

    t.switch("main")
    result = t.merge("quick-fix", no_ff=True, delete_branch=True)
    # no_ff=True   -> forces merge commit even when FF is possible
    # delete_branch -> auto-deletes source branch after merge
    branches = [b.name for b in t.list_branches()]
    print(f"Branches after: {branches}")  # ["main"] — quick-fix deleted

    t.close()


def scenario_import_commit() -> None:
    """Import commit (cherry-pick) — copy specific commits across branches."""
    print("\n=== Import Commit ===\n")

    t = Tract.open()
    t.system("Assistant.")
    t.user("Main base.")

    t.branch("experiment")
    good_ci = t.user("This insight is worth keeping.")

    t.switch("main")
    ir = t.import_commit(good_ci.commit_hash)
    print(f"Original: {ir.original_commit.commit_hash[:8]}")
    print(f"New copy: {ir.new_commit.commit_hash[:8]}")
    # Same content, new hash (different lineage). Source branch untouched.

    t.close()


def scenario_rebase() -> None:
    """Rebase — replay commits onto a new base."""
    print("\n=== Rebase ===\n")

    t = Tract.open()
    t.system("Assistant.")
    t.user("Shared base.")

    t.branch("examples")
    t.user("Example 1.")
    t.user("Example 2.")

    t.switch("main")
    t.user("New main content.")  # main advances

    t.switch("examples")
    result = t.rebase("main")

    print(f"Replayed: {len(result.replayed_commits)} commits")
    print(f"New HEAD: {result.new_head[:8]}")
    # examples branch now sits on top of main's latest commit.
    # Hashes change (new lineage) but content is preserved.
    for orig, new in zip(result.original_commits, result.replayed_commits):
        print(f"  {orig.commit_hash[:8]} -> {new.commit_hash[:8]}")

    t.close()


# =====================================================================
# COMPRESSION & GC
# =====================================================================


def scenario_manual_compress() -> None:
    """Manual compress — summarize history into a single commit (no LLM)."""
    print("\n=== Manual Compress ===\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What are black holes?")
    t.assistant("Black holes are regions of spacetime with extreme gravity.")
    t.user("What about neutron stars?")
    t.assistant("Neutron stars are ultra-dense remnants of supernovae.")

    result = t.compress(
        content="User learned about black holes (extreme gravity) and neutron stars (dense remnants).",
    )
    result.pprint()

    t.close()


def scenario_range_compress() -> None:
    """Range compress — compress a range while preserving specific commits."""
    print("\n=== Range Compress with Preserve ===\n")

    t = Tract.open()
    sys_ci = t.system("You are a contract reviewer.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    q1 = t.user("What are the payment terms?")
    a1 = t.assistant("Net 45, 1.5% late penalty.")
    q2 = t.user("What are the risks?")
    a2 = t.assistant("Uptime SLA is aggressive at 99.95%.")

    # Compress but preserve specific commits verbatim
    result = t.compress(
        content="Contract: Net 45 payment, 1.5% penalty, 99.95% SLA.",
        preserve=[q1.commit_hash, a1.commit_hash],
    )
    print(f"Preserved: {len(result.preserved_commits)} commits kept verbatim")

    t.close()


def scenario_guided_compression() -> None:
    """Guided compression — IMPORTANT priority with retain criteria."""
    print("\n=== Guided Compression (Priority + Retain) ===\n")

    t = Tract.open()
    t.system("You are a finance assistant.")
    report = t.user("Q3 revenue: $4.2M, margin: 23.5%, churn: 2.1%")
    t.assistant("Revenue is strong. Churn is below industry average.")

    # Mark report as IMPORTANT with retention criteria
    t.annotate(
        report.commit_hash,
        Priority.IMPORTANT,
        retain="Preserve all dollar amounts and percentages",
    )

    # LLM-based compress with target token count (requires LLM config):
    # result = t.compress(target_tokens=100)

    # Verify the annotation stuck
    history = t.log()
    for ci in history:
        if ci.commit_hash == report.commit_hash:
            print(f"Report priority: {ci.effective_priority}")
            break

    t.close()


def scenario_gc() -> None:
    """GC — reclaim storage after compression."""
    print("\n=== GC with Retention Policies ===\n")

    t = Tract.open()
    sys_ci = t.system("You are helpful.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)
    t.user("Hello")
    t.assistant("Hi there!")

    t.compress(content="User greeted the assistant.")

    # Conservative: keep archives forever (default)
    gc1 = t.gc(archive_retention_days=None)
    print(f"gc(None): removed {gc1.commits_removed} commits")

    # Aggressive: remove archives immediately
    gc2 = t.gc(archive_retention_days=0)
    print(f"gc(0): removed {gc2.commits_removed}, freed {gc2.tokens_freed} tokens")

    # Production recommendation: archive_retention_days=30
    t.close()


# =====================================================================
# COMPILE REORDERING & TOOL COMPRESSION
# =====================================================================


def scenario_compile_reorder() -> None:
    """Compile reorder — rearrange message order at compile time."""
    print("\n=== Compile Reorder ===\n")

    t = Tract.open()
    t.system("You are a nutrition expert.")
    t.user("What are macros?")
    t.assistant("Proteins, carbs, and fats.")
    t.user("Explain fasting.")
    t.assistant("Time-restricted eating pattern.")

    ctx = t.compile()
    hashes = ctx.commit_hashes
    # [0]=system, [1]=q1_user, [2]=q1_asst, [3]=q2_user, [4]=q2_asst

    # Swap the two Q&A pairs
    new_order = [hashes[0], hashes[3], hashes[4], hashes[1], hashes[2]]
    reordered, warnings = t.compile(order=new_order)

    print(f"Reorder warnings: {len(warnings)}")
    for w in warnings:
        print(f"  [{w.severity}] {w.warning_type}: {w.description}")
    if not warnings:
        print("  (none -- safe to reorder APPEND-only commits)")

    t.close()


def scenario_tool_tracking() -> None:
    """Tool result tracking and selective compress_tool_calls (LLM path shown)."""
    print("\n=== Tool Tracking + Selective Compression ===\n")

    t = Tract.open()
    t.system("You are a debugging agent.")
    t.user("Find the auth bug.")

    # Simulate tool calls and results
    t.assistant("Searching...", metadata={"tool_calls": [
        {"id": "c1", "name": "grep", "arguments": {"pattern": "auth"}},
    ]})
    t.tool_result("c1", "grep", "login.py:15: def authenticate()\n" * 10)

    t.assistant("Reading file...", metadata={"tool_calls": [
        {"id": "c2", "name": "bash", "arguments": {"cmd": "cat login.py"}},
    ]})
    t.tool_result("c2", "bash", "import hashlib\ndef authenticate(): pass")

    # Compress only grep results, leave bash untouched (requires LLM):
    # grep_result = t.compression.compress_tool_calls(
    #     name="grep",
    #     instructions="One line per file: 'filename: finding'",
    # )
    # print(f"compressed {grep_result.turn_count} grep turns")

    # Query tool results for token analysis
    grep_results = t.tools.find_results(name="grep")
    bash_results = t.tools.find_results(name="bash")
    print(f"grep results: {len(grep_results)}, bash results: {len(bash_results)}")

    t.close()


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:
    # --- History ---
    scenario_log()
    scenario_show()
    scenario_diff()
    scenario_reset()
    scenario_edit()

    # --- Branching ---
    scenario_branch_lifecycle()
    scenario_merge_fast_forward()
    scenario_merge_clean()
    scenario_merge_conflict()
    scenario_merge_options()
    scenario_import_commit()
    scenario_rebase()

    # --- Compression & GC ---
    scenario_manual_compress()
    scenario_range_compress()
    scenario_guided_compression()
    scenario_gc()

    # --- Compile reordering & tool tracking ---
    scenario_compile_reorder()
    scenario_tool_tracking()

    print("\nDAG operations reference complete.")


if __name__ == "__main__":
    main()
