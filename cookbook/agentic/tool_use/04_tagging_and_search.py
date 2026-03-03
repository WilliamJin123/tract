"""Tagging and Search via Tools

An LLM agent organizes conversation history using tags — registering custom
tag vocabularies, tagging commits by topic, querying by tags to find related
content, and removing stale tags. All through genuine tool calls.

Scenario: A research conversation covers multiple topics. The agent is asked
to organize it by tagging commits by subject, then use tag queries to find
all commits related to a specific topic.

Tools exercised: register_tag, tag, untag, get_tags, list_tags,
                 query_by_tags, log, get_commit

Demonstrates: LLM-driven taxonomy creation, retrospective tagging,
              tag-based retrieval, agent building its own search index
"""

import json
import sys
from pathlib import Path

import httpx

from tract import Tract
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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


def run_agent_loop(t, executor, tools, task, max_turns=15):
    """Generic agentic loop: user task -> tool calls -> final response."""
    t.user(task)

    for turn in range(max_turns):
        response = t.generate()

        if not response.tool_calls:
            print(f"\n  Agent: {response.text[:200]}")
            if len(response.text) > 200:
                print(f"         ...({len(response.text)} chars total)")
            return response

        for tc in response.tool_calls:
            result = executor.execute(tc.name, tc.arguments)
            t.tool_result(tc.id, tc.name, str(result))
            args_short = json.dumps(tc.arguments)[:60]
            print(f"    -> {tc.name}({args_short})")
            output_short = str(result.output)[:80]
            print(f"       {output_short}")

    print("  (max turns reached)")
    return None


# =====================================================================
# PART 1 -- Manual: Show tagging tools, execute directly
# =====================================================================

def part1_manual():
    """Show tag tools and execute them manually."""
    print("=" * 60)
    print("PART 1 -- Manual: Tagging and Search Tools")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.user("Tell me about photosynthesis.")
        t.assistant("Photosynthesis converts light energy into chemical energy.")
        t.user("What about cellular respiration?")
        t.assistant("Cellular respiration breaks down glucose to produce ATP.")

        executor = ToolExecutor(t)
        executor.set_profile(TAG_PROFILE)

        # Show available tools
        tools = t.as_tools(profile=TAG_PROFILE, format="openai")
        print(f"\n  Tag profile: {len(tools)} tools")
        for tool in tools:
            print(f"    - {tool['function']['name']}")

        print("\n  Conversation:")
        t.compile().pprint(style="chat")

        # Register and apply a tag
        executor.execute("register_tag", {"name": "biology", "description": "Biology topics"})
        commits = t.log(limit=4)
        if commits:
            result = executor.execute("tag", {
                "commit_hash": commits[0].commit_hash,
                "tag": "biology",
            })
            print(f"\n  tag(): {result.output}")

        # List tags
        result = executor.execute("list_tags", {})
        print(f"\n  list_tags():\n    {result.output[:200]}")

        print()


# =====================================================================
# PART 2 -- Agent: LLM organizes history with tags
# =====================================================================

def part2_agent():
    """LLM agent creates a taxonomy and tags conversation history."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no API key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: LLM Organizes with Tags")
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
        executor = ToolExecutor(t)
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
        print("  --- Task: Create taxonomy and tag everything ---")
        run_agent_loop(
            t, executor, tools,
            "Look at the conversation history with log. Create appropriate "
            "topic tags (register them first with descriptions), then tag "
            "each commit by its subject. Finally, use query_by_tags to find "
            "all commits related to biology, and list_tags to show the "
            "final taxonomy."
        )


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/metadata/01_tags.py         -- Manual tag operations
# cookbook/agentic/sidecar/04_auto_tagger.py     -- Orchestrator-driven tagging
