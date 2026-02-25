"""Tool Result Hooks

Intercept tool results before they enter the commit chain.  The
PendingToolResult hook lets you inspect, edit, summarize, or reject
any tool result — useful for filtering sensitive output, enforcing
token budgets on verbose tools, or routing different tools through
different summarization strategies.

Part 1: Hook basics — t.on("tool_result", handler), PendingToolResult
         fields (tool_name, content, token_count), approve/reject
Part 2: Edit and summarize — pending.edit_result() for manual replacement,
         pending.summarize() for LLM-driven summarization
Part 3: Declarative config — configure_tool_summarization() with per-tool
         instructions, auto_threshold, and default_instructions
Part 4: Custom routing — write a handler that routes different tools
         through different strategies (pass-through, edit, summarize, reject)

Parts 1-2 demonstrate the hook primitives.  Part 3 shows the sugar layer.
Part 4 combines both for a production-ready pattern.

Demonstrates: t.on("tool_result", handler), PendingToolResult,
              edit_result(), summarize(instructions=, target_tokens=),
              approve(), reject(), original_content preservation,
              configure_tool_summarization(instructions=, auto_threshold=,
              default_instructions=), review=True for manual inspection,
              ToolSummarizationConfig
"""


def main(): pass


if __name__ == "__main__":
    main()
