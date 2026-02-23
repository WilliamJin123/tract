"""Programmatic Tool Calling

Setup: An agent loop with Tract operations exposed as OpenAI-format tool
       definitions via as_tools(). The LLM receives tool schemas and decides
       which operations to call. Conversation is at 85% token budget.
Decision: The LLM must identify budget pressure and choose the right Tract
          tool to invoke (compress, skip, branch, etc.).
Evaluates: Tool selection appropriateness, argument validity, budget outcome.

Demonstrates: as_tools(format="openai"), ToolExecutor, chat() with tools
Compares: Baseline for 02 (MCP) and 03 (CLI) — same scenario, different interface
"""


def main():
    # --- Setup: load fixture conversation, configure budget ---
    # --- Expose tools: as_tools(format="openai", profile="self") ---
    # --- Agent loop: LLM sees status + tools, decides action ---
    # --- Execute: ToolExecutor runs chosen operation ---
    # --- Verify: check budget, log decision trace ---
    # --- Report: print metrics ---
    pass


if __name__ == "__main__":
    main()
