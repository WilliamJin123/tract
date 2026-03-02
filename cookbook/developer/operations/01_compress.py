"""Core compression: manual, interactive, and agent-driven.

Three ways to compress conversation history:

  PART 1 -- Manual:      compress(content="your summary"), no LLM needed
  PART 2 -- Interactive:  compress(target_tokens=150, review=True), human edits + approves
  PART 3 -- LLM / Agent:  compress(target_tokens=200) with instructions=, automated
"""

import os

import click
from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: compress(content=), no LLM needed")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # Pin the system prompt so it survives compression
        sys_ci = t.system("You are a concise astronomy guide.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("How do stars form?")
        t.chat("What are black holes?")
        t.chat("Explain neutron stars.")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # Manual summary -- no LLM needed
        result = t.compress(
            content=(
                "User learned about three stellar phenomena: "
                "star formation (nebulae collapsing under gravity), "
                "black holes (collapsed massive stars with event horizons), "
                "and neutron stars (ultra-dense remnants of supernovae)."
            ),
        )

        # Inspect the CompressResult
        print(f"\n  CompressResult:")
        print(f"    compression_id:    {result.compression_id[:8]}")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    source_commits:    {len(result.source_commits)} archived")
        print(f"    preserved_commits: {len(result.preserved_commits)} kept (PINNED)")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")
        print(f"\n  3 Q&A pairs -> 1 summary. PINNED system prompt survived.")


def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: review=True, human edits + approves")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise biology explainer.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is CRISPR and how does it work?")
        t.chat("How does mRNA deliver instructions to cells?")
        t.chat("Explain epigenetics and why it matters.")

        print("\n  BEFORE compression:\n")
        t.compile().pprint(style="chat")

        # review=True returns a PendingCompress -- nothing committed yet
        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        pending.pprint(verbose=True)

        # Interactive: open each draft in $EDITOR for human editing
        for i, summary in enumerate(pending.summaries):
            print(f"\n  Opening summary [{i}] in your editor...")
            edited = click.edit(summary)
            if edited is not None and edited.strip() != summary.strip():
                pending.edit_summary(i, edited.strip())
                print(f"  Summary [{i}] updated with your edits.")
            else:
                print(f"  Summary [{i}] kept as-is.")

        # Approve -- NOW it commits
        if click.confirm("\n  Approve and commit?", default=True):
            result = pending.approve()
            print(f"\n  Approved! {result.compression_ratio:.1%} compression ratio")
            t.compile().pprint(style="compact")
        else:
            pending.reject("User declined")
            print("  Cancelled. Nothing was committed.")


def part3_agent():
    print("=" * 60)
    print("PART 3 -- LLM / Agent: instructions= steers the summary")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        sys_ci = t.system("You are a concise economics tutor.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("What is inflation and what causes it?")
        t.chat("Explain supply and demand.")

        # Mark debug noise as SKIP -- excluded from compression entirely
        noise = t.user("[debug] latency=342ms | cache=miss")
        t.annotate(noise.commit_hash, Priority.SKIP)

        t.chat("What causes a recession?")

        print("\n  BEFORE compression:\n")
        ctx_before = t.compile()
        ctx_before.pprint(style="chat")
        print(f"\n  {ctx_before.token_count} tokens, {len(ctx_before.messages)} messages")

        # LLM generates the summary, steered by instructions=
        result = t.compress(
            target_tokens=200,
            instructions="Focus on key decisions and action items. Omit historical background.",
        )

        print(f"\n  CompressResult:")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="table")
        print(f"\n  PINNED system survived. SKIP noise excluded. Rest summarized by LLM.")


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
