"""CLI + Skill Interface

Setup: An agent interacts with Tract through CLI commands (shell execution)
       or natural language skill invocations that map to CLI operations.
Decision: Same budget pressure scenario. The agent must formulate correct CLI
          commands to manage context.
Evaluates: Command generation accuracy, output parsing, end-state equivalence.

Demonstrates: CLI commands (tract status, tract compress, etc.), output parsing
Compares: Natural language command generation vs structured tool calling
"""


def main():
    # --- Setup: same fixture conversation, CLI available ---
    # --- Agent loop: LLM generates CLI commands as text ---
    # --- Execute: shell out to tract CLI ---
    # --- Parse: agent interprets CLI output ---
    # --- Verify: end state matches programmatic/MCP ---
    # --- Report: print metrics ---
    pass


if __name__ == "__main__":
    main()
