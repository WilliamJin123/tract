"""GC After Compression

Three tiers of garbage collection -- manual compress+gc, interactive
review with editor, and agent-driven triggers for auto-maintenance.

PART 1 -- Manual           compress(content=) + gc(), inspect GCResult
PART 2 -- Interactive       compress(review=True), edit summaries, gc()
PART 3 -- LLM / Agent      CompressTrigger + GCTrigger compound auto-maintenance

Demonstrates: gc(), GCResult, compress(content=), compress(review=True),
              PendingCompress, edit_summary(), approve(),
              CompressTrigger, GCTrigger, configure_triggers()
"""

import os

import click
from dotenv import load_dotenv

from tract import CompressTrigger, GCTrigger, Priority, TokenBudgetConfig, Tract, TractConfig
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


# =============================================================================
# PART 1 -- Manual: compress(content=) + gc(), no LLM, deterministic
# =============================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Compress + GC")
    print("=" * 60)
    print()
    print("  Simple compress with your own summary text, then gc() to")
    print("  reclaim storage. No LLM, no interaction.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        sys_ci = t.system("You are a concise travel advisor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("Best time to visit Japan?")
        t.chat("What about Iceland?")
        t.chat("Tips for visiting Morocco.")
        t.chat("Best way to travel through Southeast Asia?")

        ctx_before = t.compile()
        print(f"\n  Built conversation: {len(ctx_before.messages)} messages, "
              f"{ctx_before.token_count} tokens")

        # Manual compression with explicit summary
        result = t.compress(
            content=(
                "User asked about travel: Japan (spring/autumn), Iceland "
                "(summer), Morocco (spring/fall), Southeast Asia (Nov-Feb)."
            ),
        )

        print(f"\n  CompressResult:")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    source_commits:    {len(result.source_commits)} archived")

        ctx_after = t.compile()
        print(f"\n  Post-compression context:")
        ctx_after.pprint(style="chat")

        # GC with 0-day archive retention (immediate cleanup for demo)
        print(f"\n  Running gc(archive_retention_days=0)...\n")
        gc_result = t.gc(archive_retention_days=0)

        print(f"  GCResult:")
        print(f"    commits_removed:        {gc_result.commits_removed}")
        print(f"    blobs_removed:          {gc_result.blobs_removed}")
        print(f"    tokens_freed:           {gc_result.tokens_freed}")
        print(f"    source_commits_removed: {gc_result.source_commits_removed}")
        print(f"    duration_seconds:       {gc_result.duration_seconds:.3f}s")

        print(f"\n  Context unchanged after GC:")
        t.compile().pprint(style="chat")
        print(f"\n  Archived source commits are gone. Reachable data is safe.")


# =============================================================================
# PART 2 -- Interactive: compress(review=True), edit summaries, gc()
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: Collaborative Compression + GC")
    print("=" * 60)
    print()
    print("  LLM drafts summaries, you review and edit them in $EDITOR,")
    print("  then approve and run gc() to reclaim storage.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        sys_ci = t.system("You are a concise travel advisor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("Best time to visit Japan?")
        t.chat("What about Iceland?")
        t.chat("Tips for visiting Morocco.")
        t.chat("Best way to travel through Southeast Asia?")

        ctx_before = t.compile()
        print(f"\n  Built conversation: {len(ctx_before.messages)} messages, "
              f"{ctx_before.token_count} tokens")

        # Collaborative compression: LLM drafts, user reviews & edits
        pending = t.compress(target_tokens=100, review=True)

        print(f"\n  PendingCompress (NOT yet committed):")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    original_tokens:  {pending.original_tokens}")
        print(f"    estimated_tokens: {pending.estimated_tokens}")

        # Interactive review: open each draft in $EDITOR for real editing
        for i, summary in enumerate(pending.summaries):
            print(f"\n  Opening summary [{i}] in your editor...")
            edited = click.edit(summary)
            if edited is not None and edited.strip() != summary.strip():
                pending.edit_summary(i, edited.strip())
                print(f"  Summary [{i}] updated with your edits.")
            else:
                print(f"  Summary [{i}] kept as-is.")

        # Approve -- NOW it commits
        compress_result = pending.approve()

        print(f"\n  Approved! CompressResult:")
        print(f"    {len(compress_result.source_commits)} commits archived")
        print(f"    {compress_result.original_tokens} -> {compress_result.compressed_tokens} tokens")
        print(f"    compression_ratio: {compress_result.compression_ratio:.1%}")

        ctx_after = t.compile()
        print(f"\n  Post-compression context:")
        ctx_after.pprint(style="chat")

        # GC with 0-day archive retention (immediate cleanup for demo)
        print(f"\n  Running gc(archive_retention_days=0)...\n")
        gc_result = t.gc(archive_retention_days=0)

        print(f"  GCResult:")
        print(f"    commits_removed:        {gc_result.commits_removed}")
        print(f"    tokens_freed:           {gc_result.tokens_freed}")

        print(f"\n  Context unchanged after GC:")
        t.compile().pprint(style="chat")


# =============================================================================
# PART 3 -- LLM / Agent: CompressTrigger + GCTrigger compound auto-maintenance
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("PART 3 -- Agent: Auto-Maintenance via Triggers")
    print("=" * 60)
    print()
    print("  CompressTrigger fires when budget fills up. GCTrigger fires")
    print("  when dead commits accumulate. Together they auto-maintain")
    print("  the context window without human intervention.")

    compress_trigger = CompressTrigger(threshold=0.7, summary_content="Auto-condensed.")
    gc_trigger = GCTrigger(max_dead_commits=5)

    print(f"\n  CompressTrigger: threshold=0.7, fires_on={compress_trigger.fires_on}")
    print(f"  GCTrigger:       max_dead_commits=5, fires_on={gc_trigger.fires_on}")

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=300))

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
        config=config,
    ) as t:
        t.configure_triggers([compress_trigger, gc_trigger])

        # Pin the system prompt so it survives compression
        sys_ci = t.system("You are a concise travel advisor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        # Add messages to fill up the budget
        for i in range(8):
            t.user(f"Travel question {i}: recommend a destination with padding text here.")

        print(f"\n  Added 8 messages. Compiling (triggers may fire)...")

        # compile() fires CompressTrigger if budget exceeded
        ctx = t.compile()
        print(f"  After compile: {ctx.token_count} tokens, {len(ctx.messages)} messages")

        # Check if GC trigger would fire
        gc_action = gc_trigger.evaluate(t)
        if gc_action:
            print(f"\n  GCTrigger fired: {gc_action.reason}")
        else:
            print(f"\n  GCTrigger: not yet triggered (dead commits below threshold)")

        print(f"\n  Triggers auto-manage compression and garbage collection")
        print(f"  as the conversation grows -- no human intervention needed.")


# =============================================================================
# main
# =============================================================================

def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
