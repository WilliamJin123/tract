"""Tool Query and Audit

Find, filter, and inspect tool-related commits across a conversation.
Use the query API to answer questions like "which tools were called?",
"how many tokens did grep results consume?", and "which tool turns
are worth compressing?"  Then act on the answers — bulk-edit verbose
results, selectively compress by tool name, or export an audit log.

No LLM required for the query and edit operations (Parts 1-2).
Part 3 uses an LLM for selective compression by tool name.

Part 1: Query API — find_tool_results(), find_tool_calls(), find_tool_turns()
Part 2: Surgical edits — tool_result(edit=) to trim verbose results
Part 3: Selective compression — compress_tool_calls(name=) to compress
        only specific tool types while leaving others untouched

Each part uses its own Tract instance for clarity.

Demonstrates: find_tool_results(name=, after=), find_tool_calls(name=),
              find_tool_turns(name=), ToolTurn (all_hashes, result_hashes,
              total_tokens, tool_names), tool_result(edit=) for surgical
              replacement, compress_tool_calls(name=) for selective
              compression, token accounting before/after edits
"""


def main(): pass


if __name__ == "__main__":
    main()
