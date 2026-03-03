"""Guided compression with priorities and retention guarantees.

  PART 1 -- Manual:      annotate(IMPORTANT), compress(content=..., preserve=[h1,h2])
"""

import sys
from pathlib import Path

from tract import Priority, Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

MODEL_ID = llm.large

CONTRACT_PATH = Path(__file__).parent / "sample_contract.md"


def main():
    print("=" * 60)
    print("PART 1 -- Manual: IMPORTANT + preserve= keeps critical facts")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
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


if __name__ == "__main__":
    main()
