"""Rollback

Three tiers of rollback -- manual reset, interactive numbered log
with confirmation, and agent-driven toolkit execution.

PART 1 -- Manual           reset() permanently rolls back, no interaction
PART 2 -- Interactive       Numbered log, click.prompt, click.confirm, reset
PART 3 -- LLM / Agent      ToolExecutor reset/checkout for agent rollback

Demonstrates: reset(), compile(), compile(at_commit=), compile(at_time=),
              checkout(), status(), log(), show(),
              click.prompt, click.confirm, ToolExecutor
"""

import os
from datetime import datetime, timezone

import click
from dotenv import load_dotenv

from tract import Tract, ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


# =============================================================================
# PART 1 -- Manual: reset() permanently rolls back
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Permanent Rollback with reset()")
    print("=" * 60)
    print()
    print("  Build a conversation, then reset() to an earlier commit.")
    print("  Later turns become orphaned -- invisible to compile().")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise geography tutor. One sentence answers.")

        r1 = t.chat("What are the 3 largest countries by area?")
        early_hash = r1.commit_info.commit_hash

        t.chat("Which of those has the highest population density?")
        t.chat("What's the capital of that country?")

        print("\n  Full conversation (7 messages):")
        t.compile().pprint(style="chat")

        # Permanently roll back to turn 1
        print(f"\n  Resetting to turn 1 ({early_hash[:8]})...\n")
        t.reset(early_hash)

        print("  After reset:")
        ctx = t.compile()
        ctx.pprint(style="chat")
        print(f"\n  {len(ctx.messages)} messages -- turns 2-3 are orphaned.")


# =============================================================================
# PART 2 -- Interactive: Numbered log, click.prompt, confirm, reset
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Rollback with Log and Confirmation")
    print("=" * 60)
    print()
    print("  Chat over several turns, then use a numbered log to pick")
    print("  a rollback target. Confirm before resetting.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise geography tutor. One sentence answers.")

        r1 = t.chat("What are the 3 largest countries by area?")
        turn1_hash = r1.commit_info.commit_hash

        midpoint = datetime.now(timezone.utc)

        r2 = t.chat("Which of those has the highest population density?")
        r3 = t.chat("What's the capital of that country?")

        # Show the full conversation
        print("\n  Full session:")
        t.compile().pprint(style="chat")

        # Time-travel: what did the LLM see after turn 1?
        print(f"\n  Time-travel: context at turn 1 ({turn1_hash[:8]}):")
        past_ctx = t.compile(at_commit=turn1_hash)
        past_ctx.pprint(style="chat")

        # Time-travel by timestamp
        print(f"\n  Time-travel: context at midpoint timestamp:")
        mid_ctx = t.compile(at_time=midpoint)
        print(f"  {len(mid_ctx.messages)} messages (same as turn 1)")

        # Show numbered log for rollback selection
        entries = list(t.log(limit=20))
        print(f"\n  Commit log:")
        for i, entry in enumerate(entries):
            print(f"    [{i}] {entry.commit_hash[:8]}  {entry.role:10s}  {str(entry.message or '')[:40]}")

        choice = click.prompt(
            "\n  Roll back to which commit? (number)",
            type=int,
            default=0,
        )

        if 0 <= choice < len(entries):
            target = entries[choice]
            # Preview the commit
            preview = t.show(target.commit_hash)
            print(f"\n  Target: {target.commit_hash[:8]}")
            print(f"    role:    {preview.role}")
            print(f"    content: {str(preview.content)[:80]}")

            orphan_count = choice  # entries above the target get orphaned
            if click.confirm(f"  Reset to {target.commit_hash[:8]}? This orphans {orphan_count} commits.", default=False):
                t.reset(target.commit_hash)
                print(f"\n  After reset:")
                t.compile().pprint(style="chat")
            else:
                print("  Reset cancelled.")

        # Checkout for non-destructive inspection
        print(f"\n  Checkout (non-destructive) to turn 1:")
        t.checkout(turn1_hash)
        status = t.status()
        print(f"  HEAD: {status.head_hash[:8]}, detached: {status.is_detached}")

        # Return to main
        t.checkout("main")
        print(f"  Back on main: {t.current_branch}")


# =============================================================================
# PART 3 -- LLM / Agent: ToolExecutor reset/checkout
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: ToolExecutor Rollback")
    print("=" * 60)
    print()
    print("  An LLM agent uses ToolExecutor for rollback operations.")
    print("  reset() is destructive; checkout() is non-destructive.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise geography tutor.")

        r1 = t.chat("What are the 3 largest countries by area?")
        early_hash = r1.commit_info.commit_hash

        t.chat("Which has the highest population density?")
        t.chat("What's the capital?")

        executor = ToolExecutor(t)

        # Non-destructive: checkout to inspect past state
        print(f"\n  checkout({early_hash[:8]}) -- non-destructive inspection:")
        result = executor.execute("checkout", {"target": early_hash})
        print(f"  Result: {result}")

        status = t.status()
        print(f"  Detached: {status.is_detached}")

        # Return to main
        result = executor.execute("checkout", {"target": "main"})
        print(f"\n  checkout('main'): {result}")

        # Destructive: reset to roll back permanently
        print(f"\n  reset({early_hash[:8]}) -- permanent rollback:")
        result = executor.execute("reset", {"commit_hash": early_hash})
        print(f"  Result: {result}")

        print(f"\n  After reset:")
        ctx = t.compile()
        ctx.pprint(style="chat")
        print(f"  {len(ctx.messages)} messages remaining.")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
