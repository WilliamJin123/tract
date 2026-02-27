"""Agent Serialization Interface

Part 1 — to_dict(): serialize any Pending to a structured dict for
LLM consumption. Shows the JSON output with operation, status, fields,
and available_actions. Works identically on every Pending subclass.

Part 2 — to_tools(): generate JSON Schema tool definitions compatible
with OpenAI/Anthropic function calling. One tool per public action.
Show how an LLM can see available actions as callable functions.

Part 3 — describe_api(): human/LLM-readable markdown documentation
auto-generated from type hints and docstrings.

Part 4 — apply_decision() and execute_tool(): dispatch an LLM's
structured decision to the correct method. Whitelist guards prevent
calling private methods. Full end-to-end: Pending → to_tools() →
LLM picks action → apply_decision() executes it.

Demonstrates: to_dict(), to_tools(), describe_api(), apply_decision(),
              execute_tool(), _public_actions whitelist, pprint()
"""

import json
import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.gc import PendingGC
from tract.hooks.tool_result import PendingToolResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# ---------------------------------------------------------------------------
# Part 1: to_dict() — Structured Serialization
# ---------------------------------------------------------------------------

def part1_to_dict():
    print("=" * 60)
    print("PART 1 — to_dict(): Structured Serialization")
    print("=" * 60)
    print()
    print("  Every Pending subclass serializes to a consistent dict format.")
    print("  Send this to an LLM to let it inspect state and pick an action.")

    with Tract.open() as t:
        # --- PendingGC ---
        # GC removes orphaned commits. Create orphans by branching then deleting.
        t.system("You are a helpful assistant.")
        t.user("Main context.")
        t.assistant("Main reply.")

        t.branch("temp")
        for i in range(3):
            t.user(f"Temp Q{i}")
            t.assistant(f"Temp A{i}")
        t.switch("main")
        t.delete_branch("temp", force=True)

        pending_gc: PendingGC = t.gc(orphan_retention_days=0, review=True)
        gc_dict = pending_gc.to_dict()

        print(f"\n  PendingGC.to_dict():")
        print(json.dumps(gc_dict, indent=4))

        print(f"\n  Key structure:")
        print(f"    operation:         {gc_dict['operation']}")
        print(f"    pending_id:        {gc_dict['pending_id'][:12]}...")
        print(f"    status:            {gc_dict['status']}")
        print(f"    fields:            {list(gc_dict['fields'].keys())}")
        print(f"    available_actions: {gc_dict['available_actions']}")

        pending_gc.approve()  # Clean up

    # --- PendingToolResult ---
    print(f"\n  PendingToolResult.to_dict():")

    with Tract.open() as t:
        t.system("Assistant.")
        t.assistant("Calling tool.", metadata={
            "tool_calls": [{"id": "x1", "name": "grep",
                            "arguments": {"pattern": "TODO"}}],
        })

        pending_tr: PendingToolResult = t.tool_result(
            "x1", "grep", "main.py:10: # TODO fix this", review=True,
        )
        tr_dict = pending_tr.to_dict()
        print(json.dumps(tr_dict, indent=4))

        print(f"\n  Same structure, different fields:")
        print(f"    operation:         {tr_dict['operation']}")
        print(f"    fields:            {list(tr_dict['fields'].keys())}")
        print(f"    available_actions: {tr_dict['available_actions']}")

        pending_tr.approve()


# ---------------------------------------------------------------------------
# Part 2: to_tools() — JSON Schema Tool Definitions
# ---------------------------------------------------------------------------

def part2_to_tools():
    print("\n" + "=" * 60)
    print("PART 2 — to_tools(): JSON Schema Tool Definitions")
    print("=" * 60)
    print()
    print("  Auto-generates OpenAI/Anthropic function-calling schemas.")
    print("  One tool definition per public action on the Pending.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Get a PendingCompress to show rich tool schemas
        sys_ci = t.system("You are a science tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)
        t.chat("What is DNA?")
        t.chat("How does DNA replication work?")
        t.chat("What are the main enzymes involved?")

        pending: PendingCompress = t.compress(target_tokens=300, review=True)
        tools = pending.to_tools()

        print(f"\n  PendingCompress.to_tools() — {len(tools)} tools:")
        for tool in tools:
            fn = tool["function"]
            params = fn["parameters"].get("properties", {})
            required = fn["parameters"].get("required", [])
            print(f"\n    {fn['name']}()")
            print(f"      description: {fn['description'][:70]}")
            if params:
                for pname, pschema in params.items():
                    req = " (required)" if pname in required else ""
                    print(f"      {pname}: {pschema['type']}{req}")

        # Show the raw JSON for one tool
        print(f"\n  Raw JSON for 'edit_summary':")
        edit_tool = next(t for t in tools if t["function"]["name"] == "edit_summary")
        print(json.dumps(edit_tool, indent=4))

        # These can be passed directly to an LLM API
        print(f"\n  Pass to OpenAI: client.chat.completions.create(tools=pending.to_tools())")

        pending.approve()


# ---------------------------------------------------------------------------
# Part 3: describe_api() — Markdown Documentation
# ---------------------------------------------------------------------------

def part3_describe_api():
    print("\n" + "=" * 60)
    print("PART 3 — describe_api(): Markdown Documentation")
    print("=" * 60)
    print()
    print("  Auto-generated markdown from type hints and docstrings.")
    print("  Useful for system prompts or documentation.")

    with Tract.open() as t:
        t.system("Assistant.")
        t.user("Main.")
        t.assistant("Reply.")
        t.branch("temp")
        t.user("Temp Q")
        t.assistant("Temp A")
        t.switch("main")
        t.delete_branch("temp", force=True)

        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)
        api_doc = pending.describe_api()

        print(f"\n  PendingGC.describe_api():\n")
        print(api_doc)

        pending.approve()

    # --- PendingToolResult API ---
    with Tract.open() as t:
        t.system("Assistant.")
        t.assistant("Calling.", metadata={
            "tool_calls": [{"id": "y1", "name": "test",
                            "arguments": {}}],
        })

        pending_tr: PendingToolResult = t.tool_result(
            "y1", "test", "test output", review=True,
        )
        api_doc_tr = pending_tr.describe_api()
        print(f"\n  PendingToolResult.describe_api():\n")
        print(api_doc_tr)

        pending_tr.approve()


# ---------------------------------------------------------------------------
# Part 4: apply_decision() and execute_tool()
# ---------------------------------------------------------------------------

def part4_dispatch():
    print("\n" + "=" * 60)
    print("PART 4 — apply_decision() / execute_tool()")
    print("=" * 60)
    print()
    print("  Let an LLM pick an action from to_tools(), then dispatch")
    print("  the structured response through apply_decision().")

    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Main context.")
        t.assistant("Main reply.")

        t.branch("work")
        for i in range(3):
            t.user(f"Work Q{i}")
            t.assistant(f"Work A{i}")
        t.switch("main")
        t.delete_branch("work", force=True)

        pending: PendingGC = t.gc(orphan_retention_days=0, review=True)

        # --- execute_tool(): call by name ---
        print(f"\n  execute_tool('exclude', {{'commit_hash': ...}}):")
        first_hash = pending.commits_to_remove[0]
        pending.execute_tool("exclude", {"commit_hash": first_hash})
        print(f"    Excluded {first_hash[:12]}")
        print(f"    Remaining: {len(pending.commits_to_remove)} commits")

        # --- apply_decision(): structured LLM output ---
        print(f"\n  apply_decision() — simulating LLM response:")

        # Simulate what an LLM would return after seeing to_tools()
        llm_decision = {
            "action": "approve",
            "args": {},
        }
        print(f"    LLM decision: {json.dumps(llm_decision)}")

        result = pending.apply_decision(llm_decision)
        print(f"    Dispatched! status={pending.status}")

    # --- Whitelist security ---
    print(f"\n  Whitelist security:")

    with Tract.open() as t:
        t.system("Assistant.")
        t.assistant("Tool.", metadata={
            "tool_calls": [{"id": "z1", "name": "test",
                            "arguments": {}}],
        })
        pending2: PendingToolResult = t.tool_result(
            "z1", "test", "output", review=True,
        )

        # Try to call a private method — blocked
        try:
            pending2.execute_tool("_execute_fn")
        except ValueError as e:
            print(f"    execute_tool('_execute_fn') -> ValueError: {e}")

        # Try to call a non-whitelisted method — blocked
        try:
            pending2.execute_tool("pprint")
        except ValueError as e:
            print(f"    execute_tool('pprint') -> ValueError: {e}")

        # Only whitelisted actions work
        print(f"\n    Allowed actions: {sorted(pending2._public_actions)}")
        pending2.approve()

    # --- Full pipeline: Pending → to_dict + to_tools → LLM → apply_decision ---
    print(f"\n  Full agent pipeline:")
    print(f"    1. Operation returns Pending (review=True or via hook)")
    print(f"    2. Send to_dict() as context + to_tools() as tools to LLM")
    print(f"    3. LLM returns {{action, args}} in tool_call response")
    print(f"    4. pending.apply_decision(llm_response) dispatches safely")
    print(f"    5. Whitelist blocks private/unauthorized methods")


# ---------------------------------------------------------------------------

def main():
    part1_to_dict()
    part2_to_tools()
    part3_describe_api()
    part4_dispatch()


if __name__ == "__main__":
    main()
