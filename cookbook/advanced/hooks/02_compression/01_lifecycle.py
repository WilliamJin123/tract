"""PendingCompress Lifecycle

compress(review=True) returns a PendingCompress with draft summaries.
Inspect fields, edit_summary(), then approve() to commit -- nothing
lands until you approve.
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult
from tract.protocols import CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _seed_conversation(t: Tract) -> None:
    """Build a multi-turn support conversation to give compress something to work with."""
    sys_ci = t.system("You are a customer support agent for TechFlow, a project management SaaS platform.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("Hi, I can't export my project reports to PDF. The button just spins and nothing happens.")
    t.chat("I'm on Chrome, macOS. The project has about 200 tasks with file attachments.")
    t.chat("Tried Firefox too — same issue. Is there a file size limit for exports?")
    t.chat("Can you just email me the report directly? My deadline is tomorrow.")


def pending_lifecycle() -> None:
    print("=" * 60)
    print("PART 1 -- PendingCompress Lifecycle")
    print("=" * 60)
    print()
    print("  compress(review=True) returns a PendingCompress.")
    print("  Nothing is committed until you call approve().")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        print("\n  BEFORE compression:")
        ctx_before: CompiledContext = t.compile()
        print(f"    {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")
        ctx_before.pprint(style="compact")

        # --- Get the pending (not committed yet) ---
        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # pprint(verbose=True) shows status, token ratio, summary previews,
        # and guidance -- all the fields we used to print manually.
        print("\n  PendingCompress returned:")
        pending.pprint(verbose=True)

        # --- Edit a summary before committing ---
        original: str = pending.summaries[0]
        pending.edit_summary(0, original.rstrip(".") + " -- reviewed and approved.")
        print(f"\n  After edit_summary(0, ...):")
        print(f"    Draft [0]: {pending.summaries[0][:100]}...")

        # Context hasn't changed yet (nothing committed)
        ctx_still: CompiledContext = t.compile()
        print(f"\n  Context unchanged: {ctx_still.token_count} tokens (nothing committed yet)")

        # --- Approve: NOW it commits ---
        result: CompressResult = pending.approve()

        # pprint shows the updated status (approved) and final token ratio
        print("\n  Approved!")
        pending.pprint()

        # CompressResult details
        print(f"\n  CompressResult:")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    new_head:          {result.new_head[:12]}")

        print("\n  AFTER compression:")
        ctx_after: CompiledContext = t.compile()
        print(f"    {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
        ctx_after.pprint(style="compact")


if __name__ == "__main__":
    pending_lifecycle()
