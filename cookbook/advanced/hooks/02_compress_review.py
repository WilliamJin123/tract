"""Compression Review via Hooks

Part 1 — PendingCompress lifecycle: compress(review=True) returns a
PendingCompress with draft summaries. Inspect fields, edit_summary(),
then approve() to commit. Nothing lands until you approve.

Part 2 — Hook handler patterns: register a handler that auto-edits
summaries (e.g. enforcing a word limit) and approves, versus one that
rejects poor quality. Shows the handler contract: call approve() or
reject() before returning.

Part 3 — Guidance: edit_guidance() steers what the LLM focuses on
during compression. The guidance field tracks its source (user, llm,
or user+llm). Useful for telling the summarizer "focus on the code
examples, skip the small talk."

Part 4 — Retry and validate: validate() checks summaries against
quality criteria, returning a ValidationResult. edit_summary() can
fix problems, then re-validate. auto_retry() from tract.hooks.retry
automates the loop (validate -> retry -> validate). Shows what
HookRejection looks like when retries are exhausted.

Part 5 — Two-stage compression: compress(two_stage=True) generates
guidance *before* summaries, letting the LLM decide what matters.
The guidance is editable via edit_guidance() and refreshable via
regenerate_guidance(). Combines well with review=True for full
human-in-the-loop control over both the plan and the output.

Demonstrates: compress(review=True), PendingCompress, edit_summary(),
              edit_guidance(), guidance_source, approve(), reject(),
              CompressResult, t.on("compress", handler), pprint(),
              validate(), ValidationResult, auto_retry(), HookRejection,
              two_stage=True, regenerate_guidance()
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.retry import auto_retry
from tract.hooks.validation import HookRejection, ValidationResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _seed_conversation(t):
    """Add a system prompt + several turns to give compress something to work with."""
    sys_ci = t.system("You are a concise science tutor.")
    t.annotate(sys_ci.commit_hash, Priority.PINNED)

    t.chat("What is photosynthesis?")
    t.chat("How do plants convert sunlight to energy?")
    t.chat("What role does chlorophyll play?")
    t.chat("Explain the light-dependent reactions.")


# ---------------------------------------------------------------------------
# Part 1: PendingCompress Lifecycle (review=True)
# ---------------------------------------------------------------------------

def part1_pending_lifecycle():
    print("=" * 60)
    print("PART 1 — PendingCompress Lifecycle")
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
        ctx_before = t.compile()
        print(f"    {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")
        ctx_before.pprint(style="compact")

        # --- Get the pending (not committed yet) ---
        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        print(f"\n  PendingCompress returned:")
        print(f"    status:           {pending.status}")
        print(f"    summaries:        {len(pending.summaries)} draft(s)")
        print(f"    source_commits:   {len(pending.source_commits)}")
        print(f"    preserved:        {len(pending.preserved_commits)} (PINNED)")
        print(f"    original_tokens:  {pending.original_tokens}")
        print(f"    estimated_tokens: {pending.estimated_tokens}")

        # Show each draft
        for i, summary in enumerate(pending.summaries):
            print(f"\n    Draft [{i}]: {summary[:100]}...")

        # --- Edit a summary before committing ---
        original = pending.summaries[0]
        pending.edit_summary(0, original.rstrip(".") + " — reviewed and approved.")
        print(f"\n  After edit_summary(0, ...):")
        print(f"    Draft [0]: {pending.summaries[0][:100]}...")

        # Context hasn't changed yet (nothing committed)
        ctx_still = t.compile()
        print(f"\n  Context unchanged: {ctx_still.token_count} tokens (nothing committed yet)")

        # --- Approve: NOW it commits ---
        result = pending.approve()
        print(f"\n  Approved! CompressResult:")
        print(f"    compression_ratio: {result.compression_ratio:.1%}")
        print(f"    new_head:          {result.new_head[:12]}")
        print(f"    status:            {pending.status}")

        print("\n  AFTER compression:")
        ctx_after = t.compile()
        print(f"    {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
        ctx_after.pprint(style="compact")


# ---------------------------------------------------------------------------
# Part 2: Hook Handler Patterns
# ---------------------------------------------------------------------------

def part2_hook_handler_patterns():
    print("\n" + "=" * 60)
    print("PART 2 — Hook Handler Patterns")
    print("=" * 60)

    # --- Pattern A: Auto-edit and approve ---
    print("\n  Pattern A: Auto-edit handler (enforce word limit)")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        def enforce_word_limit(pending: PendingCompress):
            """Truncate any summary over 30 words, then approve."""
            max_words = 30
            for i, summary in enumerate(pending.summaries):
                words = summary.split()
                if len(words) > max_words:
                    truncated = " ".join(words[:max_words]) + "..."
                    pending.edit_summary(i, truncated)
                    print(f"    Truncated summary [{i}]: {len(words)} → {max_words} words")
            pending.approve()

        t.on("compress", enforce_word_limit)
        _seed_conversation(t)

        # compress() fires the hook automatically (no review=True needed)
        result = t.compress(target_tokens=150)
        print(f"    CompressResult: ratio={result.compression_ratio:.1%}")

    # --- Pattern B: Quality gate (reject) ---
    print("\n  Pattern B: Quality gate (reject if too long)")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        def quality_gate(pending: PendingCompress):
            """Reject if any summary exceeds token budget."""
            max_chars = 50  # Very strict — will likely reject
            for i, summary in enumerate(pending.summaries):
                if len(summary) > max_chars:
                    pending.reject(
                        f"Summary [{i}] is {len(summary)} chars (limit: {max_chars})"
                    )
                    print(f"    Rejected: summary [{i}] too long")
                    return
            pending.approve()

        t.on("compress", quality_gate)
        _seed_conversation(t)

        result = t.compress(target_tokens=150)

        # result is PendingCompress (rejected) — not a CompressResult
        print(f"    Returned: {type(result).__name__}")
        print(f"    Status: {result.status}")
        print(f"    Reason: {result.rejection_reason}")


# ---------------------------------------------------------------------------
# Part 3: Guidance (edit_guidance, guidance_source)
# ---------------------------------------------------------------------------

def part3_guidance():
    print("\n" + "=" * 60)
    print("PART 3 — Guidance (Steering the Summarizer)")
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

        print(f"\n  Initial state:")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # User provides guidance
        pending.edit_guidance("Focus on the chemical equations. Skip analogies.")
        print(f"\n  After edit_guidance('Focus on the chemical equations...'):")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # Edit again — source stays "user"
        pending.edit_guidance("Emphasize chlorophyll's role specifically.")
        print(f"\n  After second edit_guidance(...):")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # Approve with the user's guidance in place
        result = pending.approve()
        print(f"\n  Approved with guidance. Compression ratio: {result.compression_ratio:.1%}")

        # --- Hook handler that uses guidance ---
        print("\n  Hook handler that auto-sets guidance:")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        def guided_compressor(pending: PendingCompress):
            """Set guidance before approving to steer the output."""
            pending.edit_guidance("Keep only factual statements. Remove explanations.")
            print(f"    Handler set guidance: '{pending.guidance}'")
            print(f"    guidance_source: {pending.guidance_source}")
            pending.approve()

        t.on("compress", guided_compressor)
        _seed_conversation(t)

        result = t.compress(target_tokens=150)
        print(f"    Result: {type(result).__name__}")


# ---------------------------------------------------------------------------
# Part 4: Retry and Validate
# ---------------------------------------------------------------------------

def part4_retry_and_validate():
    print("\n" + "=" * 60)
    print("PART 4 — Retry and Validate")
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

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # Validate the initial summaries
        result: ValidationResult = pending.validate()
        print(f"\n  Initial validate():")
        print(f"    passed:    {result.passed}")
        print(f"    diagnosis: {result.diagnosis}")
        print(f"    index:     {result.index}")

        # Force a bad summary to demonstrate failure + fix
        original = pending.summaries[0]
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

        # Approve after passing validation
        compress_result = pending.approve()
        print(f"\n  Approved after validation: ratio={compress_result.compression_ratio:.1%}")

    # --- 4b: auto_retry() — automated validate->retry loop ---
    print("\n  --- 4b: auto_retry() — automated loop ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        # auto_retry validates, retries failing summaries with diagnosis
        # as guidance, then approves if all pass.
        result = auto_retry(pending, max_retries=3)

        print(f"\n  auto_retry() result: {type(result).__name__}")
        if isinstance(result, HookRejection):
            print(f"    Rejected: {result.reason}")
            print(f"    Source:   {result.rejection_source}")
        else:
            print(f"    Compression ratio: {result.compression_ratio:.1%}")
            print(f"    Status: {pending.status}")

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
        result = auto_retry(pending, max_retries=1)

        print(f"\n  auto_retry(max_retries=1) result: {type(result).__name__}")
        if isinstance(result, HookRejection):
            print(f"    reason:           {result.reason}")
            print(f"    rejection_source: {result.rejection_source}")
            print(f"    metadata:         {result.metadata}")
            print(f"    pending.status:   {pending.status}")
        else:
            # LLM may have produced a valid summary on retry
            print(f"    Succeeded anyway: ratio={result.compression_ratio:.1%}")


# ---------------------------------------------------------------------------
# Part 5: Two-Stage Compression (Guidance Generation)
# ---------------------------------------------------------------------------

def part5_two_stage():
    print("\n" + "=" * 60)
    print("PART 5 — Two-Stage Compression (Guidance Generation)")
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

        pending: PendingCompress = t.compress(
            target_tokens=150, two_stage=True, review=True,
        )

        print(f"\n  After compress(two_stage=True, review=True):")
        print(f"    guidance:        {pending.guidance[:100]}..." if pending.guidance else "    guidance:        None")
        print(f"    guidance_source: {pending.guidance_source}")
        print(f"    summaries:       {len(pending.summaries)} draft(s)")

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

        original_guidance = pending.guidance
        print(f"\n  Original guidance (from two_stage):")
        print(f"    {original_guidance[:100]}..." if original_guidance else "    None")

        # Regenerate guidance from LLM (gets a fresh take)
        new_guidance = pending.regenerate_guidance()
        print(f"\n  After regenerate_guidance():")
        print(f"    guidance:        {new_guidance[:100]}...")
        print(f"    guidance_source: {pending.guidance_source}")

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
        pending.edit_guidance("Focus only on the chemical processes. Skip analogies.")
        print(f"\n  After edit_guidance(...):")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")

        # Approve with user-edited guidance
        result = pending.approve()
        print(f"\n  Approved with user guidance: ratio={result.compression_ratio:.1%}")

    # --- 5d: two_stage=False (default) skips guidance ---
    print("\n  --- 5d: two_stage=False (default) skips guidance ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        _seed_conversation(t)

        pending: PendingCompress = t.compress(target_tokens=150, review=True)

        print(f"\n  After compress(review=True) — no two_stage:")
        print(f"    guidance:        {pending.guidance}")
        print(f"    guidance_source: {pending.guidance_source}")
        print(f"    (None because two_stage defaults to False)")

        result = pending.approve()
        print(f"\n  Approved without guidance: ratio={result.compression_ratio:.1%}")


# ---------------------------------------------------------------------------

def main():
    part1_pending_lifecycle()
    part2_hook_handler_patterns()
    part3_guidance()
    part4_retry_and_validate()
    part5_two_stage()


if __name__ == "__main__":
    main()
