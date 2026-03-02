"""Annotations — Pin, Skip, and Reset

Control what the LLM sees without deleting history. Two tiers:
manual API calls and interactive prompts.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides

Demonstrates: default PINNED on system(), annotate(NORMAL) to unpin,
              annotate(SKIP), annotate(NORMAL) to reset, compile()
              reflects annotations, Priority enum values, click prompts
"""

import click

from tract import Priority, Tract


# =============================================================================
# Part 1: Annotations — Pin, Skip, and Reset
# =============================================================================

def part1_annotations():
    print("=" * 60)
    print("Part 1: ANNOTATIONS — Pin, Skip, Reset  [Manual Tier]")
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
        "INT4 can cut memory 4x with under 1% accuracy loss on most benchmarks."
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
# Part 2: Interactive Priority Management  (PART 2 — Interactive)
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("Part 2: INTERACTIVE PRIORITY MANAGEMENT  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  Walk commits with t.log(), pick one by number, and change")
    print("  its priority with confirmation.")
    print()

    t = Tract.open()

    t.system("You are a coding assistant.")
    t.user("Write a factorial function.")
    t.assistant("def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)")
    t.user("Now write it iteratively.")
    t.assistant("def factorial(n):\n    result = 1\n    for i in range(2, n+1):\n        result *= i\n    return result")

    # Show numbered commit list with current priorities
    entries = list(reversed(t.log()))
    print("  Commits:")
    for i, entry in enumerate(entries):
        content_preview = (entry.content_text or "")[:40].replace("\n", " ")
        print(f"    [{i}] {entry.commit_hash[:8]}  {entry.role:9s}  {content_preview}")
    print()

    idx = click.prompt("  Change priority for which commit? (number)", type=int)
    if 0 <= idx < len(entries):
        choice = click.prompt(
            "  Priority (PINNED/SKIP/NORMAL)",
            type=click.Choice(["PINNED", "SKIP", "NORMAL"], case_sensitive=False),
        )
        target = entries[idx]
        if click.confirm(f"  Set {target.commit_hash[:8]} to {choice}?"):
            t.annotate(target.commit_hash, Priority[choice])
            print(f"  Done. Priority updated.\n")

            ctx = t.compile()
            print(f"  Compiled context: {ctx.commit_count} messages, "
                  f"{ctx.token_count} tokens")
    else:
        print("  (invalid index, skipping)")

    print()
    t.close()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_annotations()
    part2_interactive()
    print("=" * 60)
    print("Done -- both tiers of priority management demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
