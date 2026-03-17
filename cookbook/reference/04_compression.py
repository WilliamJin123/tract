"""Compression, GC, and reordering reference.

Covers: manual/range compress, guided compression with priorities,
gc() with retention policies, compile(order=), compress_tool_calls().
"""

from tract import Priority, Tract


def main() -> None:
    # =================================================================
    # 1. Manual compress (no LLM needed)
    # =================================================================

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What are black holes?")
    t.assistant("Black holes are regions of spacetime with extreme gravity.")
    t.user("What about neutron stars?")
    t.assistant("Neutron stars are ultra-dense remnants of supernovae.")

    # Compress all non-pinned history into a single summary commit
    result = t.compression.compress(
        content="User learned about black holes (extreme gravity) and neutron stars (dense remnants).",
    )
    result.pprint()
    t.close()

    # =================================================================
    # 2. Range compress + preserve
    # =================================================================

    t = Tract.open()
    sys_ci = t.system("You are a contract reviewer.")
    t.annotations.set(sys_ci.commit_hash, Priority.PINNED)

    q1 = t.user("What are the payment terms?")
    a1 = t.assistant("Net 45, 1.5% late penalty.")
    q2 = t.user("What are the risks?")
    a2 = t.assistant("Uptime SLA is aggressive at 99.95%.")

    # Compress but preserve specific commits verbatim
    result = t.compression.compress(
        content="Contract: Net 45 payment, 1.5% penalty, 99.95% SLA.",
        preserve=[q1.commit_hash, a1.commit_hash],  # keep these intact
    )
    print(f"\npreserved: {len(result.preserved_commits)} commits kept verbatim")
    t.close()

    # =================================================================
    # 3. Guided compression with IMPORTANT + retain
    # =================================================================

    t = Tract.open()
    t.system("You are a finance assistant.")
    report = t.user("Q3 revenue: $4.2M, margin: 23.5%, churn: 2.1%")
    t.assistant("Revenue is strong. Churn is below industry average.")

    # Mark the report as IMPORTANT with retention criteria
    t.annotations.set(
        report.commit_hash,
        Priority.IMPORTANT,
        retain="Preserve all dollar amounts and percentages",
    )

    # LLM-based compress with target token count (requires LLM)
    # result = t.compression.compress(target_tokens=100)  # requires LLM
    t.close()

    # =================================================================
    # 4. GC: reclaim storage after compression
    # =================================================================

    t = Tract.open()
    sys_ci = t.system("You are helpful.")
    t.annotations.set(sys_ci.commit_hash, Priority.PINNED)
    t.user("Hello")
    t.assistant("Hi there!")

    t.compression.compress(content="User greeted the assistant.")

    # Conservative: keep archives forever (default)
    gc1 = t.compression.gc(archive_retention_days=None)
    print(f"\ngc(None): removed {gc1.commits_removed} commits")

    # Aggressive: remove archives immediately
    gc2 = t.compression.gc(archive_retention_days=0)
    print(f"gc(0): removed {gc2.commits_removed}, freed {gc2.tokens_freed} tokens")

    # Production recommendation: archive_retention_days=30
    t.close()

    # =================================================================
    # 5. Reorder: compile(order=)
    # =================================================================

    t = Tract.open()
    t.system("You are a nutrition expert.")
    t.user("What are macros?")
    t.assistant("Proteins, carbs, and fats.")
    t.user("Explain fasting.")
    t.assistant("Time-restricted eating pattern.")

    ctx = t.compile()
    hashes = ctx.commit_hashes
    # [0]=system, [1]=q1_user, [2]=q1_asst, [3]=q2_user, [4]=q2_asst

    # Custom order: swap the two Q&A pairs
    new_order = [hashes[0], hashes[3], hashes[4], hashes[1], hashes[2]]
    reordered, warnings = t.compile(order=new_order)

    print(f"\nreorder: {len(warnings)} warnings")
    for w in warnings:
        print(f"  [{w.severity}] {w.warning_type}: {w.description}")
    if not warnings:
        print("  (none -- safe to reorder APPEND-only commits)")
    t.close()

    # =================================================================
    # 6. Selective compression: compress_tool_calls(name=)
    # =================================================================
    # Compresses only specific tool types, leaving others untouched.
    # Requires LLM for the summarization step.

    t = Tract.open()
    t.system("You are a debugging agent.")
    t.user("Find the auth bug.")

    # Simulate tool results
    t.assistant("Searching...", metadata={"tool_calls": [
        {"id": "c1", "name": "grep", "arguments": {"pattern": "auth"}},
    ]})
    t.tool_result("c1", "grep", "login.py:15: def authenticate()\n" * 10)

    t.assistant("Reading file...", metadata={"tool_calls": [
        {"id": "c2", "name": "bash", "arguments": {"cmd": "cat login.py"}},
    ]})
    t.tool_result("c2", "bash", "import hashlib\ndef authenticate(): pass")

    # Compress only grep results, leave bash untouched (requires LLM)
    # grep_result = t.compression.compress_tool_calls(
    #     name="grep",
    #     instructions="One line per file: 'filename: finding'",
    # )
    # print(f"compressed {grep_result.turn_count} grep turns")

    # Query tool results for token analysis
    grep_results = t.tools.find_results(name="grep")
    bash_results = t.tools.find_results(name="bash")
    print(f"\ngrep results: {len(grep_results)}, bash results: {len(bash_results)}")
    t.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
