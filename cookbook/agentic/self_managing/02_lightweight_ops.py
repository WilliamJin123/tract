"""Lightweight Inline Operations

An agent that tags, pins, and checks status as part of its normal workflow.
These are simple enough meta-decisions that the agent handles inline --
no sidecar needed.

Demonstrates: ToolExecutor for tag/annotate/status, inline agent decisions
"""

import json
import os

import click
import httpx
from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =====================================================================
# PART 1 -- Manual: Direct ToolExecutor calls for tag, annotate, status
# =====================================================================

def part1_manual():
    """Execute tag, annotate, and status operations via ToolExecutor."""
    print("=" * 60)
    print("PART 1 -- Manual: Tag, Annotate, Status via ToolExecutor")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a research assistant tracking experiment results.")

        ci_hyp = t.user("Hypothesis: the new optimizer converges 2x faster.")
        ci_res = t.assistant(
            "Experiment results: the optimizer converged in 45 epochs "
            "vs 90 for the baseline. Hypothesis confirmed."
        )
        ci_note = t.user("This is a key finding -- save it for the paper.")

        executor = ToolExecutor(t)

        # --- Tag: mark commits with semantic labels ---
        print(f"\n  Tagging commits:\n")

        result = executor.execute("register_tag", {
            "name": "hypothesis",
            "description": "A stated hypothesis to test",
        })
        print(f"    register_tag: {result.output}")

        result = executor.execute("register_tag", {
            "name": "key-finding",
            "description": "A significant result worth preserving",
        })
        print(f"    register_tag: {result.output}")

        result = executor.execute("tag", {
            "commit_hash": ci_hyp.commit_hash,
            "tag": "hypothesis",
        })
        print(f"    tag hypothesis: {result.output}")

        result = executor.execute("tag", {
            "commit_hash": ci_res.commit_hash,
            "tag": "key-finding",
        })
        print(f"    tag key-finding: {result.output}")

        # --- Annotate: pin critical context ---
        print(f"\n  Annotating commits:\n")

        result = executor.execute("annotate", {
            "target_hash": ci_res.commit_hash,
            "priority": "pinned",
            "reason": "Key experiment result -- protect from compression",
        })
        print(f"    annotate pin: {result.output}")

        # --- Status: check current state ---
        print(f"\n  Checking status:\n")

        result = executor.execute("status", {})
        print(f"    status: {result.output}")

        # --- Query: find tagged commits ---
        print(f"\n  Querying by tags:\n")

        result = executor.execute("query_by_tags", {"tags": ["key-finding"]})
        print(f"    key-finding commits: {result.output}")

        result = executor.execute("get_tags", {
            "commit_hash": ci_res.commit_hash,
        })
        print(f"    tags on result commit: {result.output}")


# =====================================================================
# PART 2 -- Interactive: Agent suggests ops, human confirms
# =====================================================================

def part2_interactive():
    """Agent analyzes conversation and suggests tag/pin operations."""
    print(f"\n{'=' * 60}")
    print("PART 2 -- Interactive: Agent Suggests, Human Confirms")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are an AI debugging assistant.")

        ci1 = t.user("The API returns 500 errors on POST /users.")
        ci2 = t.assistant(
            "Root cause: the users table has a NOT NULL constraint on "
            "email, but the request body schema marks email as optional."
        )
        ci3 = t.user("Good find. Fix the schema to require email.")
        ci4 = t.assistant(
            "Fixed: updated the OpenAPI spec and Pydantic model to "
            "require email. POST /users now returns 201."
        )

        executor = ToolExecutor(t)

        # Simulate agent analysis: decide what to tag/pin
        suggestions = [
            ("tag", {
                "commit_hash": ci1.commit_hash,
                "tag": "observation",
            }, "Tag bug report as 'observation'"),
            ("tag", {
                "commit_hash": ci2.commit_hash,
                "tag": "reasoning",
            }, "Tag root cause as 'reasoning'"),
            ("annotate", {
                "target_hash": ci4.commit_hash,
                "priority": "pinned",
                "reason": "Fix confirmation -- keep for regression tracking",
            }, "Pin the fix confirmation"),
        ]

        print(f"\n  Agent suggests {len(suggestions)} inline operations:\n")
        for tool_name, args, reason in suggestions:
            print(f"  {reason}")
            if click.confirm(f"    Execute {tool_name}?", default=True):
                result = executor.execute(tool_name, args)
                status = "OK" if result.success else "FAIL"
                print(f"    [{status}] {result.output}\n")
            else:
                print(f"    [SKIPPED]\n")

        # Status check
        result = executor.execute("status", {})
        print(f"  Final status: {result.output}")


# =====================================================================
# PART 3 -- Agent: LLM autonomously tags and pins during conversation
# =====================================================================

def part3_agent():
    """Full agentic loop where the LLM tags and pins inline."""
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("PART 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: LLM Tags and Pins Inline")
    print("=" * 60)
    print()
    print("  The agent has tag, annotate, and status tools alongside")
    print("  its normal conversation. It makes inline meta-decisions:")
    print("  tagging important findings, pinning critical context,")
    print("  and checking status mid-conversation.")
    print()

    # Profile with hint-driven descriptions for inline ops
    inline_ops_profile = ToolProfile(
        name="inline-ops",
        tool_configs={
            "commit": ToolConfig(enabled=True),
            "status": ToolConfig(
                enabled=True,
                description=(
                    "Check your context status. Call after 3+ exchanges "
                    "to monitor token usage."
                ),
            ),
            "tag": ToolConfig(
                enabled=True,
                description=(
                    "Tag a commit for retrieval. Call when a message contains "
                    "a key finding, decision, or action item. Use semantic "
                    "labels like 'decision', 'action_item', 'key_finding'."
                ),
            ),
            "annotate": ToolConfig(
                enabled=True,
                description=(
                    "Pin or skip a commit. Pin (priority='pinned') when "
                    "content is critical and must survive compression. "
                    "Skip (priority='skip') when content is superseded."
                ),
            ),
            "register_tag": ToolConfig(
                enabled=True,
                description=(
                    "Register a custom tag before first use. Call this once "
                    "per tag name before using it with the tag tool."
                ),
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commit history to find hashes for tagging.",
            ),
        },
    )

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        executor = ToolExecutor(t)
        tools = t.as_tools(profile=inline_ops_profile)
        t.set_tools(tools)

        t.system(
            "You are a research assistant. Answer questions and use your "
            "tools to organize the conversation: tag important findings, "
            "pin critical context, and check status periodically."
        )

        # Seed a multi-turn conversation
        exchanges = [
            "What are the main approaches to neural architecture search?",
            "Which approach is most compute-efficient for our 8-GPU budget?",
            "Good analysis. Pin that recommendation -- we'll use it for planning.",
        ]

        for user_msg in exchanges:
            print(f"  User: {user_msg[:60]}...")
            t.user(user_msg)

            # Agentic loop: let the LLM respond and/or call tools
            for turn in range(8):
                response = t.generate()

                if not response.tool_calls:
                    print(f"  Assistant: {response.text[:100]}...")
                    print()
                    break

                for tc in response.tool_calls:
                    result = executor.execute(tc.name, tc.arguments)
                    t.tool_result(tc.id, tc.name, str(result))
                    args_short = str(tc.arguments)[:60]
                    print(f"    [tool] {tc.name}({args_short})")
                    print(f"           -> {result.output[:80]}")

        # Show what the agent organized
        print(f"  {'=' * 50}")
        print("  Tags applied by the agent:")
        for entry in t.log():
            tags = t.get_tags(entry.commit_hash)
            if len(tags) > 1:  # More than just auto-classified
                msg = (entry.message or entry.content_text or "")[:50]
                print(f"    {entry.commit_hash[:8]}  tags={tags}  {msg}")


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/metadata/01_tags.py          -- Full tag system (register, query, strict mode)
# cookbook/developer/metadata/02_priority.py      -- Priority annotations (pin, skip, normal)
# cookbook/agentic/sidecar/04_auto_tagger.py      -- LLM-driven auto-tagging via orchestrator
# cookbook/agentic/self_managing/01_tool_hints.py  -- Description-driven tool selection
