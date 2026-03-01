"""Semantic Tags -- Classify, Annotate, and Query

Every commit is auto-classified with semantic tags (instruction, reasoning,
tool_call, etc.).  You can also attach explicit tags at commit time, add
mutable annotation tags after the fact, register custom tags with
descriptions, and query history by tag.  This cookbook covers all five facets.
"""

from tract import Priority, Tract, TagNotRegisteredError
from tract.formatting import pprint_log, pprint_tag_registry


# =============================================================================
# Part 1: Auto-Classification
# =============================================================================

def part1_auto_classification():
    print("=" * 60)
    print("Part 1: AUTO-CLASSIFICATION")
    print("=" * 60)
    print()
    print("  Every commit is auto-classified based on content type and role.")
    print("  system() -> 'instruction', assistant() -> 'reasoning',")
    print("  assistant(metadata={'tool_calls': ...}) -> 'tool_call'.")
    print()

    t = Tract.open()

    sys_ci = t.system("You are a research assistant specializing in ML papers.")
    print(f"system()    -> tags: {t.get_tags(sys_ci.commit_hash)}")

    usr_ci = t.user("Summarize the key ideas in the attention paper.")
    print(f"user()      -> tags: {t.get_tags(usr_ci.commit_hash)}")

    ast_ci = t.assistant(
        "The paper introduces the Transformer architecture, replacing "
        "recurrence with multi-head self-attention."
    )
    print(f"assistant() -> tags: {t.get_tags(ast_ci.commit_hash)}")

    tool_ci = t.assistant(
        "Let me search for that paper.",
        metadata={"tool_calls": [{"name": "search", "args": {"q": "attention"}}]},
    )
    print(f"assistant(tool_calls) -> tags: {t.get_tags(tool_ci.commit_hash)}")

    print()
    print("  Auto-tags are immutable -- they are baked into the commit at creation")
    print("  time and cannot be removed.  They cost nothing to add and make every")
    print("  commit searchable from the start.")
    print()

    t.close()


# =============================================================================
# Part 2: Explicit Tags at Commit Time
# =============================================================================

def part2_explicit_tags():
    print("=" * 60)
    print("Part 2: EXPLICIT TAGS AT COMMIT TIME")
    print("=" * 60)
    print()
    print("  Pass tags=[...] to system(), user(), or assistant() to attach")
    print("  your own semantic labels.  They merge with auto-classified tags.")
    print()

    t = Tract.open()

    # Explicit tag on a user message
    hyp_ci = t.user(
        "I hypothesize that sparse attention scales linearly.",
        tags=["observation"],
    )
    print(f"user(tags=['observation']):")
    print(f"  all tags -> {t.get_tags(hyp_ci.commit_hash)}")

    # Explicit + auto: assistant gets 'reasoning' auto-tag + our 'decision'
    finding_ci = t.assistant(
        "After reviewing the evidence, linear scaling holds for local patterns "
        "but breaks down for global dependencies.",
        tags=["decision"],
    )
    tags = t.get_tags(finding_ci.commit_hash)
    print(f"assistant(tags=['decision']):")
    print(f"  all tags -> {tags}")
    print(f"  'reasoning' (auto) present: {'reasoning' in tags}")
    print(f"  'decision' (explicit) present: {'decision' in tags}")

    print()
    print("  Explicit tags are also immutable -- set once at commit time.")
    print("  Use them for domain-specific labels the auto-classifier cannot infer.")
    print()

    t.close()


# =============================================================================
# Part 3: Mutable Annotation Tags
# =============================================================================

def part3_mutable_tags():
    print("=" * 60)
    print("Part 3: MUTABLE ANNOTATION TAGS")
    print("=" * 60)
    print()
    print("  t.tag(hash, name) adds a tag after the fact.  t.untag() removes it.")
    print("  Annotation tags are stored separately from immutable commit tags")
    print("  but get_tags() merges both sources seamlessly.")
    print()

    t = Tract.open()

    t.system("You are a debugging assistant.")
    ast_ci = t.assistant("Let me try approach A: rewrite the parser from scratch.")

    print(f"Before tagging: {t.get_tags(ast_ci.commit_hash)}")

    # Retrospectively mark this as a dead end
    t.tag(ast_ci.commit_hash, "decision")
    print(f"After t.tag('decision'): {t.get_tags(ast_ci.commit_hash)}")

    # Changed our mind -- remove the annotation
    removed = t.untag(ast_ci.commit_hash, "decision")
    print(f"After t.untag('decision'): {t.get_tags(ast_ci.commit_hash)}")
    print(f"  untag returned: {removed}  (True = tag existed and was removed)")

    # untag on a nonexistent tag returns False
    removed_again = t.untag(ast_ci.commit_hash, "decision")
    print(f"  untag again:    {removed_again}  (False = tag was already gone)")

    print()
    print("  Mutable tags are ideal for retrospective labels: dead_end,")
    print("  approved, needs_review -- things you learn after the commit.")
    print()

    t.close()


# =============================================================================
# Part 4: Tag Registry and Strict Mode
# =============================================================================

def part4_tag_registry():
    print("=" * 60)
    print("Part 4: TAG REGISTRY AND STRICT MODE")
    print("=" * 60)
    print()
    print("  Base tags (instruction, reasoning, tool_call, etc.) are pre-seeded.")
    print("  register_tag() adds custom tags.  list_tags() shows all tags with")
    print("  descriptions and usage counts.")
    print()

    t = Tract.open()

    # Register a custom tag
    t.register_tag("dead_end", "Agent determined this path was unproductive")

    # Use it
    ci = t.assistant("Approach A failed due to memory limits.")
    t.tag(ci.commit_hash, "dead_end")

    # List all tags -- shows base + custom
    print("  Registered tags (showing first 5):")
    pprint_tag_registry(t.list_tags()[:5])

    t.close()

    # --- Strict mode demonstration ---
    print()
    print("  Strict mode is ON by default.  Unregistered tags raise errors:")
    print()

    t2 = Tract.open()
    assert t2._strict_tags is True

    try:
        t2.user("test", tags=["totally_made_up_tag"])
        print("    (no error -- unexpected)")
    except TagNotRegisteredError as e:
        print(f"    TagNotRegisteredError: {e}")

    # Register it, then it works
    t2.register_tag("totally_made_up_tag")
    ci = t2.user("test", tags=["totally_made_up_tag"])
    print(f"    After register_tag(): tags = {t2.get_tags(ci.commit_hash)}")

    # Turn strict mode off -- any tag is accepted
    t2._strict_tags = False
    ci2 = t2.user("anything goes", tags=["wild_west"])
    print(f"    Strict=False: tags = {t2.get_tags(ci2.commit_hash)}")

    print()

    t2.close()


# =============================================================================
# Part 5: Tag Queries
# =============================================================================

def part5_tag_queries():
    print("=" * 60)
    print("Part 5: TAG QUERIES")
    print("=" * 60)
    print()
    print("  query_by_tags() and log(tags=...) let you slice history by tag.")
    print("  match='any' (OR) and match='all' (AND) control filtering.")
    print()

    t = Tract.open()

    # Build a mixed conversation
    sys_ci = t.system("You are a data analyst.")
    t.user("Load the sales CSV.", tags=["observation"])
    t.assistant("Loaded 10,000 rows. Revenue column has 2% nulls.")
    t.user("What is the average revenue per quarter?")
    t.assistant(
        "Q1: $42k, Q2: $55k, Q3: $48k, Q4: $61k. Q4 is strongest.",
        tags=["decision"],
    )
    t.user("Pin the Q4 finding for the report.", tags=["observation"])

    # --- match="any" (OR): commits with reasoning OR observation ---
    any_results = t.query_by_tags(["reasoning", "observation"], match="any")
    print(f"  query_by_tags(['reasoning', 'observation'], match='any'):")
    print(f"    {len(any_results)} commits matched (at least one tag present)")

    # --- match="all" (AND): commits with BOTH reasoning AND decision ---
    all_results = t.query_by_tags(["reasoning", "decision"], match="all")
    print(f"  query_by_tags(['reasoning', 'decision'], match='all'):")
    print(f"    {len(all_results)} commits matched (both tags required)")
    for r in all_results:
        print(f"      {r.commit_hash[:8]} tags={t.get_tags(r.commit_hash)}")

    # --- log(tags=...) filters the commit log ---
    reasoning_log = t.log(tags=["reasoning"])
    print(f"\n  t.log(tags=['reasoning']):")
    pprint_log(reasoning_log)

    # --- Tags + annotations work together ---
    print(f"\n  Combining tags with priority annotations:")
    # The system prompt is already PINNED by default
    sys_tags = t.get_tags(sys_ci.commit_hash)
    print(f"    System commit {sys_ci.commit_hash[:8]}: tags={sys_tags}, PINNED by default")
    print(f"    query_by_tags(['instruction']) finds it alongside any other instructions")

    instruction_results = t.query_by_tags(["instruction"])
    print(f"    -> {len(instruction_results)} instruction-tagged commit(s) found")

    print()

    t.close()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_auto_classification()
    part2_explicit_tags()
    part3_mutable_tags()
    part4_tag_registry()
    part5_tag_queries()
    print("=" * 60)
    print("Done -- all 5 parts demonstrated the tag system end-to-end.")
    print("=" * 60)


if __name__ == "__main__":
    main()
