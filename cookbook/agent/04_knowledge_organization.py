"""Agent-Driven Knowledge Organization

An LLM agent organizes conversation history using tags -- registering custom
tag vocabularies, tagging commits by topic, querying by tags to find related
content, and removing stale tags. All through genuine tool calls.

The agent builds its own taxonomy, tags commits retrospectively, and uses
tag queries to find all commits related to a specific topic.

Tools exercised: register_tag, tag, untag, get_tags, list_tags,
                 query_by_tags, log, get_commit

Demonstrates: LLM-driven taxonomy creation, retrospective tagging,
              tag-based retrieval, agent building its own search index
"""

import io
import json
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile: tagging and search tools
TAG_PROFILE = ToolProfile(
    name="tagger",
    tool_configs={
        "register_tag": ToolConfig(
            enabled=True,
            description=(
                "Register a new tag name before using it. Provide a "
                "description of what the tag means. Required before tagging "
                "commits with this name."
            ),
        ),
        "tag": ToolConfig(
            enabled=True,
            description=(
                "Add a tag to a commit. The tag must be registered first. "
                "Use log to find commit hashes, then tag them by topic, "
                "importance, or any category."
            ),
        ),
        "untag": ToolConfig(
            enabled=True,
            description="Remove a tag from a commit.",
        ),
        "get_tags": ToolConfig(
            enabled=True,
            description=(
                "Get all tags on a specific commit. Returns both auto-classified "
                "and manually added tags."
            ),
        ),
        "list_tags": ToolConfig(
            enabled=True,
            description=(
                "List all registered tags with descriptions and usage counts. "
                "Use this to see the current taxonomy."
            ),
        ),
        "query_by_tags": ToolConfig(
            enabled=True,
            description=(
                "Find commits by tag. Use match='any' (OR) to find commits "
                "with at least one matching tag, or match='all' (AND) to "
                "require all tags. Returns matching commit hashes."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description=(
                "View recent commits to find hashes for tagging. Returns "
                "commit hashes, content types, and messages."
            ),
        ),
        "get_commit": ToolConfig(
            enabled=True,
            description="Get full details about a commit including its content.",
        ),
    },
)


def _log_step(step_num, response):
    """on_step callback -- print step number."""
    print(f"    [step {step_num}]")


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent-Driven Knowledge Organization")
    print("=" * 60)
    print()
    print("  The agent will: create a tag taxonomy, tag commits by topic,")
    print("  then query to find all commits about a specific subject.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Register tools from the profile
        tools = t.as_tools(profile=TAG_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are an information organizer. You have tagging tools to "
            "categorize and search conversation history. Register tags "
            "before using them."
        )

        # Build a multi-topic conversation
        topics = [
            ("What is photosynthesis?",
             "Photosynthesis is the process by which plants convert sunlight into glucose."),
            ("How does gravity work?",
             "Gravity is a fundamental force that attracts objects with mass toward each other."),
            ("What is DNA replication?",
             "DNA replication is the process of copying a cell's DNA before division."),
            ("What is orbital mechanics?",
             "Orbital mechanics describes the motion of objects in gravitational fields."),
        ]
        for q, a in topics:
            t.user(q)
            t.assistant(a)

        print("  Conversation to organize:")
        t.compile().pprint(style="compact")

        # Ask the agent to organize
        print("\n  --- Task: Create taxonomy and tag everything ---")
        result = t.run(
            "Look at the conversation history with log. Create appropriate "
            "topic tags (register them first with descriptions), then tag "
            "each commit by its subject. Finally, use query_by_tags to find "
            "all commits related to biology, and list_tags to show the "
            "final taxonomy.",
            max_steps=15, on_step=_log_step,
        )
        result.pprint()


if __name__ == "__main__":
    main()
