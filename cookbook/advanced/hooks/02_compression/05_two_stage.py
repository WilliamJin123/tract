"""Two-Stage Compression (Guidance Generation)

compress(two_stage=True) generates guidance *before* summaries, letting
the LLM decide what matters. The guidance is editable via edit_guidance()
and refreshable via regenerate_guidance(). Combines well with review=True
for full human-in-the-loop control over both the plan and the output.
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


def two_stage() -> None:
    print("\n" + "=" * 60)
    print("PART 5 -- Two-Stage Compression (Guidance Generation)")
    print("=" * 60)
    print()
    print("  two_stage=True generates guidance before summaries.")
    print("  The LLM first decides *what matters*, then summarizes.")

    # --- 5a: two_stage=True generates LLM guidance ---
    print("\n  --- 5a: two_stage=True with review=True ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        print(f"\n  Conversation BEFORE compression:")
        ctx_before = t.compile()
        print(f"    {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")
        ctx_before.pprint(style="compact")

        pending: PendingCompress = t.compress(
            target_tokens=150, two_stage=True, review=True,
        )

        # guidance_source is the key distinguisher for two_stage --
        # show it explicitly alongside pprint's output.
        print(f"\n  After compress(two_stage=True, review=True):")
        print(f"    guidance_source: {pending.guidance_source}")
        pending.pprint(verbose=True)

    # --- 5b: regenerate_guidance() for fresh LLM guidance ---
    print("\n  --- 5b: regenerate_guidance() ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(
            target_tokens=150, two_stage=True, review=True,
        )

        print(f"\n  Original guidance_source: {pending.guidance_source}")
        pending.pprint(verbose=True)

        # Regenerate guidance from LLM (gets a fresh take)
        new_guidance: str = pending.regenerate_guidance()
        print(f"\n  After regenerate_guidance():")
        print(f"    guidance_source: {pending.guidance_source}")
        pending.pprint(verbose=True)

    # --- 5c: edit_guidance() after two_stage (user + LLM) ---
    print("\n  --- 5c: edit_guidance() after two_stage ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(
            target_tokens=150, two_stage=True, review=True,
        )

        print(f"\n  Initial guidance_source: {pending.guidance_source}")

        # User edits the LLM-generated guidance
        pending.edit_guidance("Focus only on the technical issue details. Skip pleasantries.")
        print(f"\n  After edit_guidance(...):")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # Approve with user-edited guidance
        result: CompressResult = pending.approve()
        print(f"\n  Approved with user guidance: ratio={result.compression_ratio:.1%}")

        print(f"\n  Conversation AFTER compression:")
        ctx_after = t.compile()
        print(f"    {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
        ctx_after.pprint(style="compact")

    # --- 5d: two_stage=False (default) skips guidance ---
    print("\n  --- 5d: two_stage=False (default) skips guidance ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # No guidance when two_stage is False (the default)
        print(f"\n  After compress(review=True) -- no two_stage:")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")
        print(f"    (None because two_stage defaults to False)")

        result: CompressResult = pending.approve()
        print(f"\n  Approved without guidance: ratio={result.compression_ratio:.1%}")

        print(f"\n  Conversation AFTER compression:")
        ctx_after = t.compile()
        print(f"    {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
        ctx_after.pprint(style="compact")


if __name__ == "__main__":
    two_stage()
