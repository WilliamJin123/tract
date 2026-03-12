"""Agent-Driven Knowledge Organization (Implicit)

After a multi-topic conversation (12 exchanges across 4 disciplines),
the agent is asked to organize content by discipline and then perform
cross-cutting retrieval. Tagging provides structured access.

Tools available: register_tag, tag, untag, get_tags, list_tags,
                 query_by_tags, log, get_commit

Demonstrates: Can the model build a taxonomy (tags) to organize large-scale
              content and use structured retrieval for cross-cutting queries?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.xlarge


PROFILE = ToolProfile(
    name="librarian",
    tool_configs={
        "register_tag": ToolConfig(enabled=True),
        "tag": ToolConfig(enabled=True),
        "untag": ToolConfig(enabled=True),
        "get_tags": ToolConfig(enabled=True),
        "list_tags": ToolConfig(enabled=True),
        "query_by_tags": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
        "get_commit": ToolConfig(enabled=True),
    },
)


# 12 exchanges across 4 disciplines
CONVERSATION = [
    # Biology (3)
    ("What is photosynthesis?",
     "Converts sunlight, water, CO2 into glucose and oxygen via chlorophyll."),
    ("How does DNA replication work?",
     "Helicase unwinds the helix; polymerase synthesizes complementary strands."),
    ("What is CRISPR?",
     "Guide RNA targets DNA sequences; Cas9 cuts for editing."),

    # Physics (3)
    ("How does gravity work?",
     "General relativity: gravity is spacetime curvature from mass-energy."),
    ("What is quantum entanglement?",
     "Correlated quantum states — measuring one determines the other instantly."),
    ("Explain the photoelectric effect.",
     "Light above threshold frequency ejects electrons. Proves photon nature."),

    # Chemistry (3)
    ("What are covalent bonds?",
     "Shared electron pairs between atoms. Single/double/triple bonds."),
    ("How does catalysis work?",
     "Catalysts lower activation energy without being consumed."),
    ("What is electrochemistry?",
     "Electron transfer in redox reactions. Galvanic vs electrolytic cells."),

    # Computer Science (3)
    ("How does public key cryptography work?",
     "Paired keys: public encrypts, private decrypts. Based on prime factoring."),
    ("What is a hash table?",
     "Maps keys to indices via hash functions. O(1) average lookup."),
    ("Explain the CAP theorem.",
     "Distributed systems: at most 2 of Consistency, Availability, Partition tolerance."),
]


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent-Driven Knowledge Organization (Implicit)")
    print("=" * 60)
    print()
    print("  12 exchanges across 4 disciplines. Agent is asked to")
    print("  organize by discipline and do cross-cutting retrieval.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        t.system(
            "You are a knowledgeable research assistant with tagging tools.\n"
            "When asked to organize content, ACT IMMEDIATELY using your tools: "
            "register_tag to create categories, then tag to apply them to commits. "
            "Use log to find commits by hash, then tag each one. Do not ask for "
            "confirmation — just do it."
        )

        # Build the large multi-topic conversation
        for q, a in CONVERSATION:
            t.user(q)
            t.assistant(a)

        print(f"  Conversation seeded with {len(CONVERSATION)} exchanges.")
        print(f"  Topics: biology, physics, chemistry, CS")

        log = StepLogger()

        # Task 1: organize by discipline (high volume makes manual scanning tedious)
        print("\n  --- Task 1: Organize by discipline ---")
        result = t.run(
            "Organize everything by discipline — biology, physics, "
            "chemistry, and CS. Register a tag for each discipline, use log "
            "to find each commit's hash, then tag each commit with its discipline. "
            "I'll be querying this repeatedly.",
            max_steps=15, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Task 2: cross-cutting retrieval (tests whether the taxonomy is queryable)
        print("\n\n  --- Task 2: Cross-cutting query ---")
        result = t.run(
            "Find everything related to energy transfer — which topics "
            "involve energy across disciplines?",
            max_steps=10, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Report
        all_tags = t.list_tags()
        if all_tags:
            print(f"\n  Tags created: {[tag['name'] for tag in all_tags]}")
        else:
            print("\n  Agent did not create any tags.")


if __name__ == "__main__":
    main()


# --- See also ---
# Tag basics (no LLM):  tags/01_tag_basics.py
# Tag queries (no LLM): tags/02_tag_queries.py
