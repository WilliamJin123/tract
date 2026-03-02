"""Sub-agent delegation: branch, delegate, compress, merge.

  PART 1 -- Manual:      Branch, child works, compress(content=...), parent.merge()
  PART 2 -- Interactive:  Review child work, click.confirm("Accept findings?"), merge
  PART 3 -- LLM / Agent:  Full session.deploy + compress + session.collapse
"""

import click

from tract import Session


# =====================================================================
# PART 1 -- Manual: branch-delegate-compress-merge
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Branch-Delegate-Compress-Merge")
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
    child._seed_base_tags()

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


# =====================================================================
# PART 2 -- Interactive: review before merge
# =====================================================================

def part2_interactive():
    print("\n" + "=" * 60)
    print("PART 2 -- Interactive: Review Before Merge")
    print("=" * 60)

    session = Session.open()
    parent = session.create_tract(display_name="lead-engineer")

    parent.system("You are a database architect.")
    parent.user("Evaluate indexing strategies for our query patterns.")

    child = session.deploy(
        parent,
        purpose="evaluate indexing strategies",
        branch_name="indexing-research",
    )
    child._seed_base_tags()

    child.user("Compare B-tree and hash indexes.")
    child.assistant("B-tree supports range queries and ordering. Hash indexes "
                    "are O(1) for equality lookups but cannot do range scans.")
    child.user("When should we use composite indexes?")
    child.assistant("Composite indexes help when queries filter on multiple "
                    "columns. Column order matters: most selective first.")

    # Review child's work
    ctx = child.compile()
    print(f"\n  Child context: {len(ctx.messages)} messages, {ctx.token_count} tokens")
    for m in ctx.messages:
        snippet = m.content[:65] + ("..." if len(m.content) > 65 else "")
        print(f"    {m.role:>10}: {snippet}")

    if click.confirm("\n  Accept child's indexing research?", default=True):
        child.compress(
            content="Indexing: B-tree for range/ordering, hash for equality. "
                    "Composite indexes: most selective column first."
        )
        parent.merge("indexing-research")
        print(f"  Merged. Parent commits: {len(parent.log())}")
    else:
        print("  Research rejected, branch not merged.")

    session.close()


# =====================================================================
# PART 3 -- LLM / Agent: deploy + compress + collapse
# =====================================================================

def part3_agent():
    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Deploy-Compress-Collapse")
    print("=" * 60)

    session = Session.open()
    parent = session.create_tract(display_name="orchestrator")

    parent.system("You are a systems design coordinator.")
    parent.user("Design a message queue architecture.")

    # Deploy child for focused research
    child = session.deploy(
        parent,
        purpose="evaluate message queue options",
        branch_name="mq-research",
    )
    child._seed_base_tags()

    # Child does thorough research
    child.user("Compare RabbitMQ, Kafka, and Redis Streams.")
    child.assistant("RabbitMQ: AMQP-based, great for task queues. "
                    "Kafka: distributed log, high throughput, exactly-once. "
                    "Redis Streams: lightweight, good for simple pub/sub.")
    child.user("What about ordering guarantees?")
    child.assistant("Kafka guarantees per-partition ordering. RabbitMQ "
                    "guarantees per-queue FIFO. Redis Streams preserve "
                    "insertion order within a stream.")
    child.user("Recommendation for our event-driven architecture?")
    child.assistant("Kafka for event sourcing (ordering + replay). "
                    "RabbitMQ for task distribution. Redis Streams for "
                    "lightweight internal pub/sub between services.")

    # Collapse entire child history into one parent commit
    collapse_result = session.collapse(
        child, into=parent,
        content="Message queue evaluation: Kafka for event sourcing (ordered, "
                "replayable), RabbitMQ for task distribution (FIFO), Redis "
                "Streams for lightweight internal pub/sub. Recommendation: "
                "Kafka as primary event bus, RabbitMQ for worker queues.",
        auto_commit=True,
    )

    print(f"\n  Collapse: {collapse_result.summary_tokens} summary tokens")
    print(f"  Source: {collapse_result.source_tokens} tokens compressed")
    print(f"  Parent commits: {len(parent.log())}")

    ctx = parent.compile()
    print(f"  Parent context: {len(ctx.messages)} messages, {ctx.token_count} tokens")

    session.close()


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
