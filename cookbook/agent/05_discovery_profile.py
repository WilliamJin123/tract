"""Discovery Profile: 3 meta-tools for progressive capability discovery

Instead of exposing 29 individual tools (which burns context tokens on schemas
the agent may never use), the discovery profile collapses everything into 3
meta-tools that let the agent discover and use capabilities on-demand:

- tract_help(topic?) -- progressive 3-level help drill-down
- tract_do(action, params?) -- single execution surface for all operations
- tract_inspect(what?) -- unified state inspection dashboard

This is the --help pattern: start broad, drill down only when needed.

Demonstrates: get_discovery_tools(), ToolExecutor.set_profile("discovery"),
              3-level help, tract_do execution, tract_inspect views,
              error-as-navigation pattern

No LLM required -- this example calls the tools directly.
"""

import sys
from pathlib import Path

from tract import Tract
from tract.toolkit.executor import ToolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    with Tract.open() as t:

        # =============================================================
        # Setup: create a ToolExecutor with the discovery profile
        # =============================================================

        print("=" * 60)
        print("Discovery Profile: 3 Meta-Tools")
        print("=" * 60)

        executor = ToolExecutor(t)
        executor.set_profile("discovery")

        print(f"\n  Available tools: {executor.available_tools()}")
        print("  (Only 3 tools instead of 29!)\n")

        # =============================================================
        # tract_help: 3-level progressive drill-down
        # =============================================================

        print("-" * 60)
        print("LEVEL 1: tract_help() -- high-level overview")
        print("-" * 60)

        result = executor.execute("tract_help", {})
        print(result.output)

        print()
        print("-" * 60)
        print("LEVEL 2: tract_help(topic='context') -- actions in a domain")
        print("-" * 60)

        result = executor.execute("tract_help", {"topic": "context"})
        print(result.output)

        print()
        print("-" * 60)
        print("LEVEL 3: tract_help(topic='commit') -- full parameter schema")
        print("-" * 60)

        result = executor.execute("tract_help", {"topic": "commit"})
        print(result.output)

        # =============================================================
        # Error-as-navigation: bad topic gives helpful guidance
        # =============================================================

        print()
        print("-" * 60)
        print("ERROR AS NAVIGATION: tract_help(topic='bogus')")
        print("-" * 60)

        result = executor.execute("tract_help", {"topic": "bogus"})
        print(result.output)
        print("\n  ^ The error itself lists all valid domains and actions.")

        # =============================================================
        # tract_do: execute operations through the single surface
        # =============================================================

        print()
        print("-" * 60)
        print("EXECUTION: tract_do(action='commit', params={...})")
        print("-" * 60)

        # Add some content to work with
        t.system("You are a helpful research assistant.")

        result = executor.execute("tract_do", {
            "action": "commit",
            "params": {
                "content": {
                    "content_type": "freeform",
                    "payload": {
                        "text": "The discovery profile reduces tool count from 29 to 3.",
                    },
                },
                "message": "research note",
            },
        })
        print(f"  Result: {result.output[:120]}...")

        # Check status via tract_do
        result = executor.execute("tract_do", {"action": "status"})
        print(f"  Status: {result.output}")

        # Create a branch via tract_do
        result = executor.execute("tract_do", {
            "action": "branch",
            "params": {"name": "experiment"},
        })
        print(f"  Branch result: {result.output}")

        # Error-as-navigation with tract_do: bad action
        print()
        print("-" * 60)
        print("ERROR AS NAVIGATION: tract_do(action='nonexistent')")
        print("-" * 60)

        result = executor.execute("tract_do", {"action": "nonexistent"})
        print(f"  {result.output[:120]}...")
        print("\n  ^ Lists all valid actions, plus a hint to use tract_help.")

        # =============================================================
        # tract_inspect: unified state views
        # =============================================================

        print()
        print("-" * 60)
        print("INSPECT: tract_inspect() -- dashboard overview")
        print("-" * 60)

        result = executor.execute("tract_inspect", {})
        print(result.output)

        print()
        print("-" * 60)
        print("INSPECT: tract_inspect(what='branches')")
        print("-" * 60)

        result = executor.execute("tract_inspect", {"what": "branches"})
        print(result.output)

        print()
        print("-" * 60)
        print("INSPECT: tract_inspect(what='history')")
        print("-" * 60)

        result = executor.execute("tract_inspect", {"what": "history"})
        print(result.output)

        print()
        print("-" * 60)
        print("INSPECT: tract_inspect(what='tags')")
        print("-" * 60)

        result = executor.execute("tract_inspect", {"what": "tags"})
        print(result.output)

        # Error-as-navigation for inspect too
        print()
        print("-" * 60)
        print("ERROR AS NAVIGATION: tract_inspect(what='bogus')")
        print("-" * 60)

        result = executor.execute("tract_inspect", {"what": "bogus"})
        print(result.output)
        print("\n  ^ Lists valid inspection targets.")

        # =============================================================
        # Alternative: get_discovery_tools() for direct access
        # =============================================================

        print()
        print("-" * 60)
        print("ALTERNATIVE: get_discovery_tools() returns ToolDefinitions")
        print("-" * 60)

        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(t)
        for tool in tools:
            print(f"  {tool.name:16s}  {tool.description[:60]}...")

        # Call a tool handler directly (no executor needed)
        help_tool = tools[0]  # tract_help
        overview = help_tool.handler()
        print(f"\n  Direct handler call returned {len(overview)} chars")

        # =============================================================
        # Summary
        # =============================================================

        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print("  The discovery profile provides 3 meta-tools that cover")
        print("  all 29 tract operations. An LLM agent starts with")
        print("  tract_help() for a high-level overview, drills into")
        print("  domains and actions as needed, then uses tract_do()")
        print("  to execute. tract_inspect() gives state at a glance.")
        print()
        print("  Key pattern: errors are navigation aids. Bad inputs")
        print("  return lists of valid options, not dead ends.")


if __name__ == "__main__":
    main()


# --- See also ---
# Presentation layer:     agent/06_presentation_layer.py
# Compact profile:        (uses 7 domain-grouped tools instead of 3 meta-tools)
# Tool profiles overview: getting_started/03_custom_tools.py
