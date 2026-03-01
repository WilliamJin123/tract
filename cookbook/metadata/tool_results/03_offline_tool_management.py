"""Offline Tool Management

Part 2 of Tool Calling: error handling with is_error=True and
drop_failed_tool_turns(), the tool query API (find_tool_turns/results/calls),
and tool_result(edit=) for surgical edits. No LLM needed.

Demonstrates: tool_result(is_error=True), drop_failed_tool_turns(),
              find_tool_turns(), find_tool_results(), tool_result(edit=),
              pprint(), log()
"""

from tract import Tract


def main():
    # =================================================================
    # PART 2: Offline Tool Management (no LLM needed)
    # =================================================================
    # Error handling, query API, surgical edits — all with pprint()
    # so you can see exactly what the context window looks like.

    print(f"\n\n{'=' * 60}")
    print("PART 2: Offline Tool Management")
    print("=" * 60)
    print()
    print("  No LLM needed. We manually commit tool interactions, then")
    print("  show error handling, the query API, and surgical edits.")
    print()

    with Tract.open() as t3:

        # --- Build a realistic multi-tool session ---

        t3.system("You are a deployment agent.")
        t3.user("Deploy the application to staging.")

        # Turn 1: Health check (success)
        t3.assistant(
            "Checking server health...",
            metadata={"tool_calls": [
                {"id": "call_1", "name": "health_check", "arguments": {}},
            ]},
        )
        t3.tool_result("call_1", "health_check",
                        "Server is healthy. CPU: 23%, Memory: 45%")

        # Turn 2: Deploy (fails — marked as error)
        t3.assistant(
            "Deploying to staging...",
            metadata={"tool_calls": [
                {"id": "call_2", "name": "deploy", "arguments": {"env": "staging"}},
            ]},
        )
        t3.tool_result(
            "call_2", "deploy",
            "Error: Connection refused. Could not reach staging server "
            "at 10.0.1.5:8080.\n"
            "Traceback (most recent call last):\n"
            "  File '/deploy/runner.py', line 42, in deploy_to\n"
            "    conn = ssh.connect(host, port)\n"
            "ConnectionRefusedError: [Errno 111] Connection refused",
            is_error=True,
        )

        # Turn 3: Deploy retry (success)
        t3.assistant(
            "Retrying with backup server...",
            metadata={"tool_calls": [
                {"id": "call_3", "name": "deploy", "arguments": {"env": "staging-backup"}},
            ]},
        )
        deploy_ci = t3.tool_result("call_3", "deploy",
                                    "Deployed successfully to staging-backup. Build #1847.")

        # --- Full context before dropping errors ---

        print("--- Before drop_failed_tool_turns() ---\n")
        ctx_before = t3.compile()
        print(f"  {len(ctx_before.messages)} messages  |  {ctx_before.token_count} tokens\n")
        ctx_before.pprint(style="compact")

        # --- Drop error turns ---

        drop_result = t3.drop_failed_tool_turns()

        print(f"\n  drop_failed_tool_turns() -> ToolDropResult:")
        print(f"    turns_dropped:   {drop_result.turns_dropped}")
        print(f"    commits_skipped: {drop_result.commits_skipped}")
        print(f"    tokens_freed:    {drop_result.tokens_freed}")
        print(f"    tool_names:      {drop_result.tool_names}")

        # --- Context after dropping: error turn gone ---

        print(f"\n--- After drop_failed_tool_turns() ---\n")
        ctx_after = t3.compile()
        saved = ctx_before.token_count - ctx_after.token_count
        print(f"  {len(ctx_after.messages)} messages  |  {ctx_after.token_count} tokens")
        print(f"  (freed {saved} tokens)\n")
        ctx_after.pprint(style="chat")

        # --- Query API ---

        print(f"\n--- find_tool_turns() ---")
        turns = t3.find_tool_turns()
        print(f"  {len(turns)} tool turn(s) in history:")
        for turn in turns:
            names = ", ".join(turn.tool_names)
            print(f"    {names}: {turn.total_tokens} tokens, "
                  f"{len(turn.results)} result(s)")

        print(f"\n--- find_tool_results() ---")
        for r in t3.find_tool_results():
            print(f"    {r.metadata['name']}: {r.token_count} tokens")

        # --- Surgical edit: shorten the deploy result in-place ---

        print(f"\n--- tool_result(edit=) ---")
        print(f"  Before: deploy result is {deploy_ci.token_count} tokens")

        edited_ci = t3.tool_result(
            "call_3", "deploy",
            "Deployed to staging-backup. Build #1847.",
            edit=deploy_ci.commit_hash,
        )
        print(f"  After:  deploy result is {edited_ci.token_count} tokens")
        print(f"  Original preserved at {deploy_ci.commit_hash[:8]}...\n")

        print("\n--- Full history (originals preserved for audit) ---\n")
        for entry in reversed(t3.log()):
            print(f"  {entry}")


if __name__ == "__main__":
    main()
