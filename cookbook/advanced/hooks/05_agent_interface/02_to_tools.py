"""to_tools(): generate JSON Schema tool definitions compatible with
OpenAI/Anthropic function calling. One tool per public action. Shows
how an LLM can see available actions as callable functions.
"""

import json
import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def to_tools_demo():
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

        # Show pending state, then the tool schemas (the lesson)
        pending.pprint()

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

        # Raw JSON for one tool — this is what you pass to an LLM API
        print(f"\n  Raw JSON for 'edit_summary':")
        edit_tool = next(t for t in tools if t["function"]["name"] == "edit_summary")
        print(json.dumps(edit_tool, indent=4))

        print(f"\n  Pass to OpenAI: client.chat.completions.create(tools=pending.to_tools())")

        pending.approve()


if __name__ == "__main__":
    to_tools_demo()
