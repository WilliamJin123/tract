"""Retry and Validate

validate() checks summaries against quality criteria, returning a
ValidationResult. edit_summary() can fix problems, then re-validate.
auto_retry() automates the loop (validate -> retry -> validate).
Shows what HookRejection looks like when retries are exhausted.
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.retry import auto_retry
from tract.hooks.validation import HookRejection, ValidationResult
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


def retry_and_validate() -> None:
    print("\n" + "=" * 60)
    print("PART 4 -- Retry and Validate")
    print("=" * 60)
    print()
    print("  validate() checks summaries against quality criteria.")
    print("  edit_summary() fixes problems, then re-validate.")
    print("  auto_retry() automates the validate->retry loop.")

    # --- 4a: Manual validate / edit_summary / re-validate ---
    print("\n  --- 4a: Manual validate -> edit_summary -> re-validate ---")

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

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # Validate the initial summaries.
        # ValidationResult fields (passed, diagnosis, index) are important
        # to show explicitly -- they drive the retry decision.
        result: ValidationResult = pending.validate()
        print(f"\n  Initial validate():")
        print(f"    passed:    {result.passed}")
        print(f"    diagnosis: {result.diagnosis}")
        print(f"    index:     {result.index}")

        # Force a bad summary to demonstrate failure + fix
        original: str = pending.summaries[0]
        pending.edit_summary(0, "Bad.")  # Suspiciously short (< 10 chars)

        result = pending.validate()
        print(f"\n  After edit_summary(0, 'Bad.'):")
        print(f"    passed:    {result.passed}")
        print(f"    diagnosis: {result.diagnosis}")
        print(f"    index:     {result.index}")

        # Fix it by restoring the original
        pending.edit_summary(0, original)
        result = pending.validate()
        print(f"\n  After restoring original summary:")
        print(f"    passed:    {result.passed}")
        print(f"    diagnosis: {result.diagnosis}")

        # Approve after passing validation -- pprint shows final state
        compress_result: CompressResult = pending.approve()
        print(f"\n  Approved after validation:")
        pending.pprint()
        print(f"    ratio={compress_result.compression_ratio:.1%}")

        print(f"\n  Conversation AFTER compression:")
        ctx_after = t.compile()
        print(f"    {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
        ctx_after.pprint(style="compact")

    # --- 4b: auto_retry() -- automated validate->retry loop ---
    print("\n  --- 4b: auto_retry() -- automated loop ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # auto_retry validates, retries failing summaries with diagnosis
        # as guidance, then approves if all pass.
        result: CompressResult | HookRejection = auto_retry(pending, max_retries=3)

        print(f"\n  auto_retry() result: {type(result).__name__}")
        if isinstance(result, HookRejection):
            print(f"    Rejected: {result.reason}")
            print(f"    Source:   {result.rejection_source}")
        else:
            print(f"    Compression ratio: {result.compression_ratio:.1%}")
            pending.pprint()

            print(f"\n  Conversation AFTER auto_retry compression:")
            ctx_after = t.compile()
            print(f"    {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
            ctx_after.pprint(style="compact")

    # --- 4c: HookRejection when retries are exhausted ---
    print("\n  --- 4c: HookRejection when retries are exhausted ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # Sabotage the summary so validation always fails
        pending.edit_summary(0, "X")  # Too short, will fail validate()

        # auto_retry will retry (which regenerates via LLM), but if the
        # regenerated summary also fails, it retries again up to max_retries.
        # With max_retries=1, it gets one shot.
        result: CompressResult | HookRejection = auto_retry(pending, max_retries=1)

        print(f"\n  auto_retry(max_retries=1) result: {type(result).__name__}")
        if isinstance(result, HookRejection):
            print(f"    reason:           {result.reason}")
            print(f"    rejection_source: {result.rejection_source}")
            print(f"    metadata:         {result.metadata}")
            pending.pprint()
        else:
            # LLM may have produced a valid summary on retry
            print(f"    Succeeded anyway: ratio={result.compression_ratio:.1%}")


if __name__ == "__main__":
    retry_and_validate()
