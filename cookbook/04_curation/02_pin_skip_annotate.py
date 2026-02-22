"""Unpin, Skip, and Reset Annotations

Control what the LLM sees without deleting history. System prompts are
PINNED by default (they survive compression). Here we unpin one for a
special case, skip noisy tool output, then reset an annotation when we
change our mind. All operations are non-destructive — commits stay in
history.

Demonstrates: default PINNED on system(), annotate(NORMAL) to unpin,
              annotate(SKIP), annotate(NORMAL) to reset, compile() reflects
              annotations, Priority enum values
"""

from tract import Priority, Tract


def main():
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


if __name__ == "__main__":
    main()
