"""Sub-agent delegation: branch, delegate, compress, merge.

Branch, child works, compress(content=...), parent.merge()

Session management (deploy, collapse, merge) is a developer-side concern --
the Orchestrator does not handle it.
"""

from tract import Session


# =====================================================================
# Branch-delegate-compress-merge
# =====================================================================

def manual():
    print("=" * 60)
    print("Branch-Delegate-Compress-Merge")
    print("=" * 60)

    session = Session.open()
    parent = session.create_tract(display_name="architect")

    parent.system("You are a distributed systems architect.")
    parent.user("We need a caching strategy for our microservices.")
    parent.assistant("I will research caching patterns and report back.")

    # Deploy child on a branch for focused research
    child = session.deploy(
        parent,
        purpose="research caching patterns",
        branch_name="research-caching",
    )

    # Child does extensive work (simulating many turns)
    topics = [
        ("What is write-through caching?", "Write-through writes to cache and DB simultaneously, ensuring consistency at the cost of latency."),
        ("What is write-back caching?", "Write-back defers DB writes until eviction, improving write performance but risking data loss."),
        ("What is cache-aside?", "Cache-aside loads on miss and updates on write. The application manages the cache explicitly."),
        ("Compare TTL vs LRU eviction.", "TTL expires entries after a fixed time. LRU evicts the least recently used entry when full."),
        ("What about distributed cache invalidation?", "Use pub/sub or versioned keys. Eventual consistency is acceptable for most read-heavy workloads."),
    ]
    for question, answer in topics:
        child.user(question)
        child.assistant(answer)

    print(f"\n  Child commits: {len(child.log())}")
    print(f"  Parent commits: {len(parent.log())} (unchanged)")

    # Compress child's work into a summary
    child.compress(
        content="Caching patterns: write-through (consistent, slower), "
                "write-back (faster, risk), cache-aside (app-managed). "
                "Eviction: TTL for time-based, LRU for space-based. "
                "Distributed invalidation via pub/sub or versioned keys."
    )

    # Merge child branch back to parent
    result = parent.merge("research-caching")
    print(f"\n  Merge type: {result.merge_type}")
    print(f"  Parent commits after merge: {len(parent.log())}")
    print(f"  10 child turns -> compressed into parent's history")

    session.close()


if __name__ == "__main__":
    manual()
