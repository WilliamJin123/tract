"""to_tools(): generate JSON Schema tool definitions compatible with
OpenAI/Anthropic function calling. One tool per public action. Shows
how an LLM can see available actions as callable functions.
"""

import json
import sys
from pathlib import Path

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.commit import CommitInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm  

MODEL_ID = llm.large


def to_tools_demo() -> None:
    print("\n" + "=" * 60)
    print("PART 2 — to_tools(): JSON Schema Tool Definitions")
    print("=" * 60)
    print()
    print("  Auto-generates OpenAI/Anthropic function-calling schemas.")
    print("  One tool definition per public action on the Pending.")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Get a PendingCompress to show rich tool schemas
        sys_ci: CommitInfo = t.system("You are a science tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)
        t.chat("What is DNA?", max_tokens=500)
        t.chat("How does DNA replication work?", max_tokens=500)
        t.chat("What was the first user question verbatim?", max_tokens=500)

        pending: PendingCompress = t.compress(target_tokens=300, review=True)

        # Show pending state, then the tool schemas (the lesson)
        t.compile().pprint()
        pending.pprint()

        tools: list[dict] = pending.to_tools()
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

        # Separate read vs write tools for clarity
        read_tools = [t for t in tools if t["function"]["name"].startswith(("get_", "list_"))]
        write_tools = [t for t in tools if not t["function"]["name"].startswith(("get_", "list_"))]
        print(f"\n  Read tools ({len(read_tools)}): {[t['function']['name'] for t in read_tools]}")
        print(f"  Write/control tools ({len(write_tools)}): {[t['function']['name'] for t in write_tools]}")

        # Raw JSON for one read tool — get_summary takes an index param
        print(f"\n  Raw JSON for 'get_summary' (read tool):")
        get_tool: dict = next(t for t in tools if t["function"]["name"] == "get_summary")
        print(json.dumps(get_tool, indent=4))

        # Raw JSON for one write tool
        print(f"\n  Raw JSON for 'edit_summary' (write tool):")
        edit_tool: dict = next(t for t in tools if t["function"]["name"] == "edit_summary")
        print(json.dumps(edit_tool, indent=4))

        print(f"\n  Pass to OpenAI: client.chat.completions.create(tools=pending.to_tools())")

        pending.approve()


if __name__ == "__main__":
    to_tools_demo()
