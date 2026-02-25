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
    print("  Unpin one, skip a verbose response the user moved past, then")
    print("  reset when you change your mind. All operations are non-destructive.")
    print()

    t = Tract.open()

    # --- Build a conversation where an early response becomes noise ---

    sys_ci = t.system("You are a research assistant.")
    print(f"System prompt: {sys_ci.commit_hash[:8]}  (PINNED by default)")

    t.user("What are the main approaches to reducing transformer inference cost?")

    # A detailed first response — useful initially, but verbose
    verbose_ci = t.assistant(
        "There are several major approaches:\n"
        "1. Quantization — INT8/INT4 reduces memory and compute by 2-4×\n"
        "2. Pruning — remove redundant attention heads or FFN neurons\n"
        "3. Knowledge distillation — train a smaller model to mimic the large one\n"
        "4. FlashAttention — IO-aware attention reduces memory from O(N²) to O(N)\n"
        "5. Speculative decoding — draft with a small model, verify with the large one\n"
        "6. KV-cache compression — evict or merge old key-value pairs\n"
        "7. Mixture of Experts — activate only a subset of parameters per token",
    )

    t.user("I'm most interested in quantization. Let's focus there.")
    t.assistant(
        "Quantization reduces model weights from FP16/FP32 to lower precision "
        "(INT8, INT4, even binary). Post-training quantization (PTQ) is fastest — "
        "no retraining needed. GPTQ, AWQ, and SmoothQuant are the leading methods. "
        "INT4 can cut memory 4× with under 1% accuracy loss on most benchmarks."
    )

    # --- Unpin the system prompt ---
    # System instructions (InstructionContent) are PINNED by default,
    # meaning they survive compression verbatim. In rare cases — e.g. a
    # temporary persona you plan to replace — you may want the compressor
    # to summarize it like any other message.

    t.annotate(sys_ci.commit_hash, Priority.NORMAL, reason="temporary persona, ok to compress")
    print(f"Unpinned: {sys_ci.commit_hash[:8]}  (now NORMAL — compressor may summarize)")

    # --- Skip the verbose overview — the user narrowed their focus ---
    # The broad list was useful to pick a topic but wastes context now
    # that we're drilling into quantization specifically.

    t.annotate(verbose_ci.commit_hash, Priority.SKIP, reason="user narrowed focus to quantization")
    print(f"Skipped:  {verbose_ci.commit_hash[:8]}  (verbose overview hidden from compile)")

    # --- Compile: overview is gone, focused content remains ---

    ctx = t.compile()
    print(f"\n=== After SKIP: {len(ctx.messages)} messages (overview hidden) ===\n")
    ctx.pprint()

    # --- Change your mind: un-skip the overview ---

    t.annotate(verbose_ci.commit_hash, Priority.NORMAL)
    print("Reset verbose overview back to NORMAL\n")

    ctx = t.compile()
    print(f"=== After reset: {len(ctx.messages)} messages (overview restored) ===\n")
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
