"""Offline Tool Management

Three tiers of offline tool management: manual error handling and queries,
interactive editing of tool results, and agent-driven compression.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: tool_result(is_error=True), drop_failed_tool_turns(),
              find_tool_turns(), find_tool_results(), tool_result(edit=),
              pprint(), log(), click.confirm(), click.edit(), ToolExecutor
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract
from tract.toolkit import ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


def part1_offline_management():
    # =================================================================
    # PART 1: Offline Tool Management (no LLM needed)
    # =================================================================
    # Error handling, query API, surgical edits — all with pprint()
    # so you can see exactly what the context window looks like.

    print(f"\n\n{'=' * 60}")
    print("Part 1: OFFLINE TOOL MANAGEMENT  [Manual Tier]")
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


# =============================================================================
# Part 2: Interactive Tool Result Editing  (PART 2 — Interactive)
# =============================================================================

def part2_interactive_edit():
    print("=" * 60)
    print("Part 2: INTERACTIVE TOOL RESULT EDITING  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  Walk tool results with find_tool_results(). For each,")
    print("  display content and offer to edit in $EDITOR.")
    print()

    with Tract.open() as t:
        t.system("You are a deployment agent.")
        t.user("Check all services.")

        # Build some tool results
        t.assistant(
            "Checking services...",
            metadata={"tool_calls": [
                {"id": "call_a", "name": "health_check",
                 "arguments": {"service": "api"}},
                {"id": "call_b", "name": "health_check",
                 "arguments": {"service": "worker"}},
            ]},
        )
        t.tool_result("call_a", "health_check",
                       "api: healthy, CPU 12%, Mem 34%, 200 req/s, "
                       "uptime 30d, last deploy 2026-02-28")
        t.tool_result("call_b", "health_check",
                       "worker: healthy, CPU 45%, Mem 67%, 50 jobs/min, "
                       "queue depth 12, uptime 15d, last restart 2026-02-25")

        # Walk results interactively
        for r in t.find_tool_results():
            name = r.metadata.get("name", "unknown")
            content = r.content_text or ""
            print(f"  [{name}] {r.commit_hash[:8]}: {content[:60]}...")

            if click.confirm("    Edit this result?", default=False):
                edited = click.edit(content)
                if edited and edited.strip() != content.strip():
                    call_id = r.metadata.get("tool_call_id", "")
                    t.tool_result(call_id, name, edited.strip(),
                                  edit=r.commit_hash)
                    print(f"    -> edited")
                else:
                    print(f"    -> no changes")
            else:
                print(f"    -> skipped")

        print()
        t.compile().pprint(style="compact")

    print()


# =============================================================================
# Part 3: Agent-Driven Tool Compression  (PART 3 — LLM / Agent)
# =============================================================================

def part3_agent_compression():
    print("=" * 60)
    print("Part 3: AGENT-DRIVEN TOOL COMPRESSION  [Agent Tier]")
    print("=" * 60)
    print()
    print("  ToolExecutor lets an agent compress tool results or manage")
    print("  tool interactions programmatically.")
    print()

    t = Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    )

    t.system("You are a deployment agent.")
    t.user("Deploy to staging.")

    t.assistant(
        "Running health check...",
        metadata={"tool_calls": [
            {"id": "call_1", "name": "health_check", "arguments": {}},
        ]},
    )
    t.tool_result("call_1", "health_check",
                   "Server healthy. CPU: 23%, Memory: 45%, Disk: 67%.")

    # Agent uses ToolExecutor to drop failed turns or manage tools
    executor = ToolExecutor(t)
    result = executor.execute("drop_failed_tool_turns", {})
    print(f"  executor.execute('drop_failed_tool_turns') -> success={result.success}")
    print(f"  result: {result.result}")
    print()
    print("  For LLM-driven compression of verbose tool results, use:")
    print("    t.compress_tool_calls(hashes, target_tokens=100)")
    print("  See 01_agentic_loop.py Part 3 for a full example.")
    print()

    t.close()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_offline_management()
    part2_interactive_edit()
    part3_agent_compression()
    print("=" * 60)
    print("Done -- all 3 tiers of offline tool management demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
