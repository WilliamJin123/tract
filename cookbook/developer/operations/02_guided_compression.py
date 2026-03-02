"""Guided compression with priorities and retention guarantees.

Three ways to ensure critical facts survive compression:

  PART 1 -- Manual:      annotate(IMPORTANT), compress(content=..., preserve=[h1,h2])
  PART 2 -- Interactive:  compress(review=True), inspect retain_match, click.confirm
  PART 3 -- LLM / Agent:  compress(target_tokens=300, instructions=..., retain_match=[...], max_retries=5)
"""

import os
from pathlib import Path

import click
from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"

CONTRACT_PATH = Path(__file__).parent / "sample_contract.md"


def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: IMPORTANT + preserve= keeps critical facts")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a contract review assistant. Be concise and precise.")

        # Seed a multi-turn contract discussion
        contract_msg = t.user(
            f"Please review this contract:\n\n{CONTRACT_PATH.read_text(encoding='utf-8')}",
            message="Full contract loaded for review",
        )
        t.chat("What are the most important financial terms?")
        t.chat("What are the biggest risks for the client?")

        # Annotate the raw contract as IMPORTANT
        t.annotate(
            contract_msg.commit_hash,
            Priority.IMPORTANT,
            retain="Preserve all dollar amounts, payment terms, and penalty clauses",
        )
        print(f"  {contract_msg.commit_hash[:8]}: annotated IMPORTANT")

        # Get hashes for the financial-terms Q&A to preserve verbatim
        entries = list(t.log(limit=20))
        entries.reverse()
        # [0]=system, [1]=contract_user, [2]=q1_user, [3]=q1_asst, [4]=q2_user, [5]=q2_asst
        financial_hashes = [entries[2].commit_hash, entries[3].commit_hash]

        print(f"  Preserving financial Q&A: [{financial_hashes[0][:8]}, {financial_hashes[1][:8]}]")

        # Manual summary, preserving the financial Q&A pair
        result = t.compress(
            content=(
                "Contract: $2,847,000 over 36 months. Net 45 payment terms. "
                "1.5% monthly late penalty. 3.5% annual escalation. "
                "99.95% uptime SLA. Go-live June 15, 2026."
            ),
            preserve=financial_hashes,
        )

        print(f"\n  Compressed: {result.original_tokens} -> {result.compressed_tokens} tokens")
        print(f"  Preserved:  {len(result.preserved_commits)} commits kept verbatim")
        print(f"  Archived:   {len(result.source_commits)} commits summarized")

        print("\n  AFTER compression:\n")
        t.compile().pprint(style="compact")


def part2_interactive():
    print("=" * 60)
    print("PART 2 -- Interactive: review=True with retention inspection")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a contract review assistant. Be concise and precise.")

        contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
        contract_msg = t.user(
            f"Please review this contract:\n\n{contract_text}",
            message="Full contract loaded for review",
        )

        t.chat("Summarize the key financial terms and deadlines.")
        t.chat("What are the penalty clauses?")

        # Annotate with IMPORTANT and retention patterns
        t.annotate(
            contract_msg.commit_hash,
            Priority.IMPORTANT,
            retain="Preserve all financial terms and dates",
            retain_match=[r"\$2,847,000", r"Net[\s-]45", r"99\.95%"],
            retain_match_mode="regex",
        )

        # review=True for human inspection before committing
        pending: PendingCompress = t.compress(target_tokens=300, review=True)

        pending.pprint(verbose=True)

        # Human decides: are the retention patterns satisfied in the draft?
        if click.confirm("\n  Retention patterns satisfied? Approve compression?", default=True):
            result = pending.approve()
            print(f"\n  Approved! {result.compression_ratio:.1%} compression ratio")
            t.compile().pprint(style="compact")
        else:
            pending.reject("Retention criteria not met in draft")
            print("  Rejected. Original context preserved.")


def part3_agent():
    print("=" * 60)
    print("PART 3 -- LLM / Agent: retain_match= with auto-retry")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a contract review assistant. Be concise and precise.")

        contract_text = CONTRACT_PATH.read_text(encoding="utf-8")
        contract_msg = t.user(
            f"Please review this contract:\n\n{contract_text}",
            message="Full contract loaded for review",
        )

        t.chat("What are the most important financial terms?")
        t.chat(
            "What are the biggest risks for the client? "
            "Focus on penalties, termination costs, and liability caps."
        )

        # Mark debug noise as SKIP
        noise = t.user("[debug] trace_id=x9f2-k3m1 | latency=342ms | cache=miss")
        t.annotate(noise.commit_hash, Priority.SKIP)

        # Annotate with regex retention patterns and retry enforcement
        t.annotate(
            contract_msg.commit_hash,
            Priority.IMPORTANT,
            retain="Preserve all financial terms and dates",
            retain_match=[
                r"\$2[,.]?847[,.]?000",
                r"[Nn]et[\s\-]*45",
                r"1\.5\s*%",
                r"3\.5\s*%",
                r"[Jj]une\s*15.*2026",
                r"99\.95\s*%",
            ],
            retain_match_mode="regex",
        )

        print(f"\n  Annotated {contract_msg.commit_hash[:8]} with 6 regex retention patterns")

        ctx_before = t.compile()
        print(f"  Before: {ctx_before.token_count} tokens, {len(ctx_before.messages)} messages")

        # LLM compresses with instructions + max_retries for retention enforcement
        result = t.compress(
            target_tokens=300,
            instructions="Preserve all financial terms and dates",
            max_retries=5,
        )

        print(f"\n  CompressResult:")
        print(f"    original_tokens:   {result.original_tokens}")
        print(f"    compressed_tokens: {result.compressed_tokens}")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")

        print("\n  AFTER compression:\n")
        ctx_after = t.compile()
        ctx_after.pprint(style="table")

        # Verify key terms survived
        print("\n  Verification: asking LLM about key facts post-compression...\n")
        verify = t.chat(
            "Quick: what's the total contract value, payment terms, "
            "late penalty rate, and go-live date?"
        )
        verify.pprint()


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
