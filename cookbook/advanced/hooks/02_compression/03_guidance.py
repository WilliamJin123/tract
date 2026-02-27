"""Guidance (Steering the Summarizer)

edit_guidance() tells the LLM what to focus on during compression.
The guidance_source field tracks where guidance came from (user, llm,
or user+llm).
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.models.compression import CompressResult

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


def guidance() -> None:
    print("\n" + "=" * 60)
    print("PART 3 -- Guidance (Steering the Summarizer)")
    print("=" * 60)
    print()
    print("  edit_guidance() tells the LLM what to focus on.")
    print("  guidance_source tracks where guidance came from.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        # Get pending to inspect guidance
        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # guidance_source is important context that pprint may not highlight,
        # so we print it explicitly alongside pprint's output.
        print(f"\n  Initial state:")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # User provides guidance
        pending.edit_guidance("Focus on the technical troubleshooting steps. Skip pleasantries.")
        print(f"\n  After edit_guidance('Focus on the technical troubleshooting...'):")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # Edit again -- source stays "user"
        pending.edit_guidance("Emphasize the PDF export issue and browser details.")
        print(f"\n  After second edit_guidance(...):")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # Approve with the user's guidance in place
        result: CompressResult = pending.approve()
        print(f"\n  Approved with guidance. Compression ratio: {result.compression_ratio:.1%}")

        # --- Hook handler that uses guidance ---
        print("\n  Hook handler that auto-sets guidance:")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        def guided_compressor(pending: PendingCompress) -> None:
            """Set guidance before approving to steer the output."""
            pending.edit_guidance("Keep only actionable details. Remove conversational filler.")
            pending.approve()

        t.on("compress", guided_compressor, name="guided-compressor")
        _seed_conversation(t)

        result: CompressResult | PendingCompress = t.compress(target_tokens=150)

        if isinstance(result, CompressResult):
            print(f"    Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        t.print_hooks()


if __name__ == "__main__":
    guidance()
