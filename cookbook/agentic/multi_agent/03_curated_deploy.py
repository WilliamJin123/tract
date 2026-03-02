"""Curated Deploy -- Branch-based sub-agent deployment with context curation.

Deploy a child agent on a new branch, curate which commits it sees using
tags, drop lists, and compact_before. Then merge results back or collapse.
Runs fully in-memory, no API key needed.
"""

from tract import Session


# =====================================================================
# Helpers
# =====================================================================

def _enable_tags(tract):
    """Enable mutable tag support and base-tag registry on a Session-created tract."""
    from tract.storage.sqlite import (
        SqliteTagAnnotationRepository,
        SqliteTagRegistryRepository,
    )
    tract._tag_annotation_repo = SqliteTagAnnotationRepository(tract._session)
    tract._tag_registry_repo = SqliteTagRegistryRepository(tract._session)
    tract._seed_base_tags()


def _seed_child_tags(child):
    """Seed base tags on a deployed child so auto-classified tags pass validation."""
    child._seed_base_tags()


def _show_compiled(label, ctx):
    """Print compiled messages with a header."""
    print(f"\n  [{label}] {len(ctx.messages)} messages, {ctx.token_count} tokens:")
    for m in ctx.messages:
        snippet = m.content[:70] + ("..." if len(m.content) > 70 else "")
        print(f"    {m.role:>10}: {snippet}")


# =====================================================================
# Part 1: Basic Deploy
# =====================================================================

def part1_basic_deploy():
    print("=" * 60)
    print("Part 1: BASIC DEPLOY")
    print("=" * 60)
    print()
    print("  deploy() creates a branch from the parent and returns a")
    print("  child Tract on that branch.  Parent stays on its original")
    print("  branch.  Both share the same commit DAG.")
    print()

    session = Session.open()
    parent = session.create_tract(display_name="orchestrator")

    # Build a conversation on parent
    parent.system("You are a planning assistant.")
    parent.user("What topics should we research?")
    parent.assistant("I suggest three topics: caching, indexing, and replication.")

    parent_branch = parent.current_branch

    # Deploy a child for research
    child = session.deploy(
        parent,
        purpose="research caching strategies",
        branch_name="research-caching",
    )
    _seed_child_tags(child)

    print(f"  Parent branch:  {parent_branch}")
    print(f"  Child branch:   {child.current_branch}")
    print(f"  Same tract_id:  {parent.tract_id == child.tract_id}")

    # Child sees the same history
    child_log = child.log()
    print(f"  Child log:      {len(child_log)} commits (inherited from parent)")

    # Child adds its own work -- parent is unaffected
    child.user("Explain write-through vs write-back caching.")
    child.assistant(
        "Write-through writes to cache and backing store simultaneously. "
        "Write-back defers the backing store write until eviction."
    )

    print(f"  Child commits:  {len(child.log())} (added 2)")
    print(f"  Parent commits: {len(parent.log())} (unchanged)")

    session.close()


# =====================================================================
# Part 2: Curated Deploy with Tags
# =====================================================================

def part2_curated_deploy_with_tags():
    print()
    print("=" * 60)
    print("Part 2: CURATED DEPLOY WITH TAGS")
    print("=" * 60)
    print()
    print("  Tag commits at creation time (immutable) or afterward (mutable).")
    print("  Deploy with keep_tags to SKIP everything that does not match.")
    print()

    session = Session.open()
    parent = session.create_tract(display_name="orchestrator")
    _enable_tags(parent)

    # Register a custom tag (base tags like "instruction" are pre-registered)
    parent.register_tag("hypothesis", "Commits containing hypotheses to test")

    # Immutable tags: set at commit time via tags= parameter
    c1 = parent.system("You are a research assistant.", tags=["instruction"])
    c2 = parent.user("Gather data on distributed caching.", tags=["instruction"])
    c3 = parent.assistant("Sure, I will research Redis, Memcached, and Hazelcast.")
    c4 = parent.user("Here is a paper on cache coherence.", tags=["hypothesis"])
    c5 = parent.assistant("The paper suggests MESI protocol for coherence.")

    # Mutable tag: added after the fact with t.tag()
    parent.tag(c3.commit_hash, "reasoning")

    print("  Tags on each commit:")
    for ci in [c1, c2, c3, c4, c5]:
        tags = parent.get_tags(ci.commit_hash)
        label = ci.commit_hash[:8]
        print(f"    {label}: {tags or '(none)'}")

    # Deploy with keep_tags -- only instruction + hypothesis survive
    child = session.deploy(
        parent,
        purpose="focused cache coherence research",
        branch_name="coherence-only",
        curate={"keep_tags": ["instruction", "hypothesis"]},
    )
    _seed_child_tags(child)

    ctx = child.compile()
    _show_compiled("child (curated)", ctx)
    print()
    print("  Commits without 'instruction' or 'hypothesis' tags are SKIPPED.")
    print(f"  Child sees {len(ctx.messages)} messages instead of 5.")

    session.close()


# =====================================================================
# Part 3: Drop and Compact
# =====================================================================

def part3_drop_and_compact():
    print()
    print("=" * 60)
    print("Part 3: DROP AND COMPACT")
    print("=" * 60)
    print()
    print("  Drop specific commits by hash, or compact older commits")
    print("  into a summary (compact_before requires LLM for smart")
    print("  summaries; without LLM it concatenates content).")
    print()

    session = Session.open()
    parent = session.create_tract(display_name="planner")

    parent.system("You are a planning assistant.")
    parent.user("What is our roadmap?")
    irrelevant = parent.assistant("Let me check the weather first...")  # noise
    parent.user("Focus on the roadmap please.")
    recent = parent.assistant("The roadmap has three milestones: alpha, beta, GA.")

    print(f"  Parent has {len(parent.log())} commits")
    print(f"  Irrelevant commit: {irrelevant.commit_hash[:12]}")

    # --- Drop a specific commit ---
    child_drop = session.deploy(
        parent,
        purpose="roadmap planning (clean context)",
        branch_name="roadmap-clean",
        curate={"drop": [irrelevant.commit_hash]},
    )
    _seed_child_tags(child_drop)

    ctx = child_drop.compile()
    _show_compiled("after drop", ctx)
    print(f"\n  Dropped the irrelevant weather tangent.")

    # --- Compact older commits ---
    # compact_before compresses everything before the marker hash.
    # Without an LLM, it concatenates content as a simple summary.
    child_compact = session.deploy(
        parent,
        purpose="roadmap planning (compacted history)",
        branch_name="roadmap-compact",
        curate={"compact_before": recent.commit_hash},
    )
    _seed_child_tags(child_compact)

    ctx_compact = child_compact.compile()
    _show_compiled("after compact", ctx_compact)
    print()
    print("  Everything before the recent milestone commit was compacted.")
    print("  (With an LLM configured, compact_before produces a smart summary.)")

    session.close()


# =====================================================================
# Part 4: Merge-Back Workflow
# =====================================================================

def part4_merge_back():
    print()
    print("=" * 60)
    print("Part 4: MERGE-BACK AND COLLAPSE")
    print("=" * 60)
    print()
    print("  After a child finishes work on its branch, bring results")
    print("  back to the parent via merge or collapse.")
    print()

    session = Session.open()
    parent = session.create_tract(display_name="orchestrator")

    parent.system("You are a project manager.")
    parent.user("We need to evaluate two database options.")

    # Deploy a child for research
    child = session.deploy(
        parent,
        purpose="evaluate PostgreSQL vs MySQL",
        branch_name="db-evaluation",
    )
    _seed_child_tags(child)

    # Child does research work
    child.user("Compare PostgreSQL and MySQL for our use case.")
    child.assistant(
        "PostgreSQL offers better JSON support and extensibility. "
        "MySQL has simpler replication and broader hosting support."
    )
    child.user("Which do you recommend?")
    child.assistant("PostgreSQL -- the JSON column support aligns with our schema.")

    print(f"  Child added {len(child.log()) - len(parent.log())} new commits")

    # --- Option A: Merge the child branch back ---
    # Parent merges the child's branch.  No conflicts => fast-forward.
    result = parent.merge("db-evaluation")

    print(f"\n  Merge type: {result.merge_type}")
    print(f"  Parent now has {len(parent.log())} commits (includes child work)")

    # --- Option B: Collapse (summarize child into one commit) ---
    # For a collapse demo, create a fresh parent+child pair.
    parent2 = session.create_tract(display_name="orchestrator-2")
    parent2.system("You are a project manager.")
    parent2.user("Research monitoring solutions.")

    child2 = session.spawn(parent2, purpose="evaluate monitoring tools")
    child2.user("Compare Prometheus, Grafana, and Datadog.")
    child2.assistant("Prometheus is open-source, Datadog is SaaS, Grafana visualizes both.")
    child2.user("Pricing considerations?")
    child2.assistant("Prometheus is free; Datadog starts at $15/host/month.")

    # Collapse: compress the child's entire history into one parent commit
    collapse = session.collapse(
        child2,
        into=parent2,
        content=(
            "Monitoring evaluation: Prometheus (free, self-hosted), "
            "Datadog ($15/host/mo, SaaS), Grafana (visualization layer). "
            "Recommendation: Prometheus + Grafana for cost efficiency."
        ),
        auto_commit=True,
    )

    print(f"\n  Collapse summary tokens: {collapse.summary_tokens}")
    print(f"  Source tokens:           {collapse.source_tokens}")
    print(f"  Parent2 commits:         {len(parent2.log())}")

    ctx = parent2.compile()
    _show_compiled("parent after collapse", ctx)

    session.close()


# =====================================================================
# Main
# =====================================================================

def main():
    part1_basic_deploy()
    part2_curated_deploy_with_tags()
    part3_drop_and_compact()
    part4_merge_back()
    print()
    print("=" * 60)
    print("Done. All four patterns demonstrated in-memory.")
    print("=" * 60)


if __name__ == "__main__":
    main()
