"""Hook Handler Patterns

Register handlers that auto-edit summaries (e.g. enforcing a word limit)
and approve, or reject poor quality. Shows the handler contract: call
approve() or reject() before returning.
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


def hook_handler_patterns() -> None:
    print("\n" + "=" * 60)
    print("PART 2 -- Hook Handler Patterns")
    print("=" * 60)

    # --- Pattern A: Auto-edit and approve ---
    print("\n  Pattern A: Auto-edit handler (enforce word limit)")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        def enforce_word_limit(pending: PendingCompress) -> None:
            """Truncate any summary over 30 words, then approve."""
            max_words = 30
            for i, summary in enumerate(pending.summaries):
                words = summary.split()
                if len(words) > max_words:
                    pending.edit_summary(i, " ".join(words[:max_words]) + "...")
            pending.approve()

        t.on("compress", enforce_word_limit, name="word-limiter")
        _seed_conversation(t)

        print(f"\n    BEFORE compression:")
        ctx_before = t.compile()
        print(f"      {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")
        ctx_before.pprint(style="compact")

        # compress() fires the hook automatically (no review=True needed)
        result: CompressResult | PendingCompress = t.compress(target_tokens=150)

        if isinstance(result, CompressResult):
            print(f"\n    Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        print(f"\n    AFTER compression:")
        ctx_after = t.compile()
        print(f"      {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")
        ctx_after.pprint(style="compact")

        # print_hooks() shows registered handlers and recent hook log
        t.print_hooks()

    # --- Pattern B: Quality gate (reject) ---
    print("\n  Pattern B: Quality gate (reject if too long)")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        def quality_gate(pending: PendingCompress) -> None:
            """Reject if any summary exceeds character budget."""
            max_chars = 50  # Very strict -- will likely reject
            for i, summary in enumerate(pending.summaries):
                if len(summary) > max_chars:
                    pending.reject(
                        f"Summary [{i}] is {len(summary)} chars (limit: {max_chars})"
                    )
                    return
            pending.approve()

        t.on("compress", quality_gate, name="quality-gate")
        _seed_conversation(t)

        print(f"\n    BEFORE compression attempt:")
        ctx_before = t.compile()
        print(f"      {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")
        ctx_before.pprint(style="compact")

        result: CompressResult | PendingCompress = t.compress(target_tokens=150)

        # result is PendingCompress (rejected) -- pprint shows status + reason
        if isinstance(result, CompressResult):
            print(f"\n    Compressed: ratio={result.compression_ratio:.1%}")
        else:
            result.pprint()

        print(f"\n    AFTER rejection (context unchanged):")
        ctx_after = t.compile()
        print(f"      {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")

        # hook_log shows the reject event
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"    [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    hook_handler_patterns()
