"""Research Pipeline: ingest -> organize -> synthesize

An agent-driven research workflow. The agent ingests information, organizes
it with tags and metadata, and synthesizes findings -- all governed by rules
that gate transitions and configure compile strategies per stage.

Stages:
  ingest    -- full compile strategy, gather raw information
  organize  -- tag taxonomy, metadata classification
  synthesize -- adaptive compile, produce final synthesis

Demonstrates: tagging tools, metadata tools, transition gates with commit
              thresholds, agent-driven stage navigation, compile strategies

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract, resolve_all_configs

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.small


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # =============================================================
        # Stage rules and transition gates
        # =============================================================

        print("=== Setting Up Research Pipeline ===\n")

        # Initial stage config
        t.rule("stage", trigger="active",
               action={"type": "set_config", "key": "stage", "value": "ingest"})
        t.rule("ingest-strategy", trigger="active",
               action={"type": "set_config", "key": "compile_strategy", "value": "full"})
        t.rule("ingest-temp", trigger="active",
               action={"type": "set_config", "key": "temperature", "value": 0.7})

        # Gates: require enough content before advancing
        gate = lambda v: {"type": "require", "condition": {
            "type": "threshold", "metric": "commit_count", "op": ">=", "value": v}}
        t.rule("organize-gate", trigger="transition:organize", action=gate(6))
        t.rule("synthesize-gate", trigger="transition:synthesize", action=gate(3))

        print(f"  Initial configs: {resolve_all_configs(t.rule_index)}")

        # =============================================================
        # Register tags the agent can use
        # =============================================================

        for tag_name in ["source", "primary", "secondary", "comparison", "synthesis"]:
            t.register_tag(tag_name)

        print(f"  Registered 5 research tags")

        # =============================================================
        # System prompt: describe the research workflow
        # =============================================================

        t.system(
            "You are a research assistant working through a structured pipeline.\n\n"
            "PIPELINE STAGES:\n"
            "1. INGEST -- Gather and record information about the topic.\n"
            "   Commit facts, summaries, and key points as user messages.\n"
            "   Tag commits with 'source', 'primary', or 'secondary'.\n"
            "2. ORGANIZE -- Classify and structure the information.\n"
            "   Use create_metadata to store structured taxonomies.\n"
            "   Use tag/register_tag to create a tagging system.\n"
            "3. SYNTHESIZE -- Produce a final synthesis of findings.\n"
            "   Use compile to review all context, then write a synthesis.\n\n"
            "Tools available: commit, compile, status, log, tag, register_tag,\n"
            "query_by_tags, create_metadata, get_config, transition.\n\n"
            "Use get_config to check current stage. Use transition to advance.\n"
            "Complete all three stages."
        )

        # =============================================================
        # Seed some initial research content
        # =============================================================

        t.user("Topic: Compare database indexing strategies -- B-trees, "
               "hash indexes, and LSM trees.")

        # =============================================================
        # Run: agent drives through ingest -> organize -> synthesize
        # =============================================================

        print("\n=== Running Agent (ingest -> organize -> synthesize) ===\n")

        result = t.run(
            "Research database indexing strategies. Start by ingesting key facts:\n"
            "- B-trees: balanced, O(log n) lookups, good for range queries\n"
            "- Hash indexes: O(1) point lookups, no range support\n"
            "- LSM trees: write-optimized, compaction-based, used in RocksDB\n\n"
            "Commit each as a separate fact. Tag each with 'source' and the "
            "relevant strategy name (register new tags as needed).\n\n"
            "When you have enough content, transition to 'organize'. Create "
            "metadata entries to classify the strategies. Then transition to "
            "'synthesize' and produce a comparative summary.",
            max_steps=20,
            on_step=lambda step, _resp: print(f"  step {step}..."),
        )

        result.pprint()

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final State ===\n")

        print(f"  Stage: {t.get_config('stage')}")
        print(f"  Branch: {t.current_branch}")

        print(f"\n  Branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Registered tags:")
        for entry in t.list_tags():
            print(f"    {entry['name']:20s} count={entry['count']}")

        print(f"\n  Log (last 10 commits):")
        for ci in t.log()[-10:]:
            tags_str = f" [{', '.join(ci.tags)}]" if ci.tags else ""
            print(f"    {ci.commit_hash[:8]}  {ci.content_type:10s}{tags_str}  "
                  f"{ci.message[:40]}")


if __name__ == "__main__":
    main()


# --- See also ---
# Coding workflow:       workflows/01_coding_assistant.py
# Customer support:      workflows/03_customer_support.py
# Tagging patterns:      agent/04_knowledge_organization.py
