"""Annotations and Edit in Place

Control what the LLM sees without deleting history. Part 1 covers
priority annotations: PINNED (survives compression), SKIP (hidden from
compile), and NORMAL (default). Part 2 shows edit-in-place — correct a
mistake in a past commit while skipping the stale responses that were
based on the wrong content.

Part 1 — Annotations (no LLM):
  Demonstrates: default PINNED on system(), annotate(NORMAL) to unpin,
                annotate(SKIP), annotate(NORMAL) to reset, compile()
                reflects annotations, Priority enum values

Part 2 — Edit in Place (uses LLM):
  Demonstrates: system(edit=hash), annotate(SKIP) for stale responses,
                chat() before/after edit, compile() serves corrected
                content, log() preserves both versions
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Annotations — Pin, Skip, and Reset
# =============================================================================

def part1_annotations():
    print("=" * 60)
    print("Part 1: ANNOTATIONS — Pin, Skip, Reset")
    print("=" * 60)
    print()
    print("  System prompts are PINNED by default (they survive compression).")
    print("  Unpin one, skip a noisy tool output, then reset when you change")
    print("  your mind. All operations are non-destructive.")
    print()

    t = Tract.open()

    # --- Build a conversation with a tool output in the middle ---

    sys_ci = t.system("You are a research assistant.")
    print(f"System prompt: {sys_ci.commit_hash[:8]}  (PINNED by default)")

    t.user("Find recent papers on transformer efficiency.")

    # Simulate a tool output — useful once, noisy after
    tool_ci = t.assistant(
        "[search_results]\n"
        "1. FlashAttention-2: Faster Attention with Better Parallelism (2023)\n"
        "2. Efficient Transformers: A Survey (2022)\n"
        "3. Mamba: Linear-Time Sequence Modeling (2023)\n"
        "... 47 more results ...",
    )

    t.user("Great, summarize the top 3.")
    t.assistant(
        "Here are the key papers:\n"
        "1. FlashAttention-2 reduces memory usage via tiling.\n"
        "2. The Efficient Transformers survey covers sparse/low-rank methods.\n"
        "3. Mamba replaces attention with selective state spaces."
    )

    # --- Unpin the system prompt ---
    # System instructions (InstructionContent) are PINNED by default,
    # meaning they survive compression verbatim. In rare cases — e.g. a
    # temporary persona you plan to replace — you may want the compressor
    # to summarize it like any other message.

    t.annotate(sys_ci.commit_hash, Priority.NORMAL, reason="temporary persona, ok to compress")
    print(f"Unpinned: {sys_ci.commit_hash[:8]}  (now NORMAL — compressor may summarize)")

    # --- Skip the tool output — it's been summarized already ---

    t.annotate(tool_ci.commit_hash, Priority.SKIP, reason="already summarized")
    print(f"Skipped:  {tool_ci.commit_hash[:8]}  (tool output hidden from compile)")

    # --- Compile: tool output is gone, everything else is visible ---

    ctx = t.compile()
    print(f"\n=== After SKIP: {len(ctx.messages)} messages (tool output hidden) ===\n")
    ctx.pprint()

    # --- Change your mind: un-skip the tool output ---

    t.annotate(tool_ci.commit_hash, Priority.NORMAL)
    print("Reset tool output back to NORMAL\n")

    ctx = t.compile()
    print(f"=== After reset: {len(ctx.messages)} messages (tool output restored) ===\n")
    ctx.pprint()

    t.close()


# =============================================================================
# Part 2: Edit in Place
# =============================================================================

def part2_edit_in_place():
    print("=" * 60)
    print("Part 2: EDIT IN PLACE")
    print("=" * 60)
    print()
    print("  A support agent's system prompt says '60-day return policy' —")
    print("  but it's actually 30 days. Chat with the LLM, see it parrot the")
    print("  wrong info, then edit the system prompt in place and ask again.")
    print("  Both versions stay in history for audit.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- System prompt with a mistake baked in ---

        bad_prompt = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 60 days.",
            message="Initial system prompt with incorrect return policy",
        )
        print(f"System prompt committed: {bad_prompt.commit_hash[:8]}\n")

        # --- Ask about returns — the LLM will cite the wrong policy ---

        print("=== Before edit ===\n")
        stale_q = t.user("What's your return policy?")
        stale_a = t.generate()
        stale_a.pprint()

        # --- Fix the system prompt: 60 days -> 30 days ---
        # edit= replaces the target commit in compiled context.
        # Same shorthand, one extra param.

        fix = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 30 days.",
            edit=bad_prompt.commit_hash,
            message="Fix return policy from 60 days to 30 days",
        )
        fix.pprint()

        # --- Skip the stale Q&A — it was based on the wrong prompt ---

        t.annotate(stale_q.commit_hash, Priority.SKIP, reason="based on wrong prompt")
        t.annotate(stale_a.commit_info.commit_hash, Priority.SKIP, reason="based on wrong prompt")
        print("Skipped stale Q&A (based on the old 60-day prompt)\n")

        # --- Ask again — the LLM now sees the corrected prompt ---

        print("=== After edit ===\n")
        response = t.chat("What's your return policy?")
        response.pprint()

        # --- Compiled context: only the corrected version appears ---

        print("\n=== Compiled context (what the LLM sees now) ===\n")
        t.compile().pprint()

        # --- Full history: both system prompts preserved for audit ---

        print("=== Full history (both versions preserved) ===\n")
        for entry in reversed(t.log()):
            print(f"  {entry}")


def main():
    part1_annotations()
    print()
    part2_edit_in_place()


if __name__ == "__main__":
    main()
