"""Batch Rollback

Group multiple commits into a single atomic unit — they all land or none do.
Useful for RAG pipelines where retrieval + question + setup must arrive
together, or any multi-step operation where partial state is worse than
no state.

Demonstrates: batch() context manager, success path, rollback on failure,
              compile() before/after batch
"""

from tract import Tract


def main():
    t = Tract.open()

    t.system("You are a financial analyst assistant.")

    # --- Successful batch: all commits land together ---

    print("=== Batch 1: Atomic RAG retrieval ===\n")

    with t.batch():
        # These three commits are one atomic unit
        t.user("What were last quarter's revenue figures?")
        t.assistant(
            "[Retrieved from knowledge base]\n"
            "Q4 2025 Revenue: $2.3B (+12% YoY)\n"
            "Operating margin: 28.4%\n"
            "Net income: $487M"
        )
        t.user("How does that compare to guidance?")

    ctx = t.compile()
    print(f"After batch: {len(ctx.messages)} messages (system + 3 from batch)\n")
    ctx.pprint()

    # --- Failed batch: everything rolls back ---

    print("=== Batch 2: Rollback on error ===\n")

    count_before = len(t.compile().messages)

    try:
        with t.batch():
            t.user("What about next quarter projections?")
            t.assistant("Projected revenue: $2.5B")
            # Simulate a pipeline failure mid-batch
            raise RuntimeError("Data source unavailable")
    except RuntimeError as e:
        print(f"Caught: {e}")

    count_after = len(t.compile().messages)
    print(f"Messages before: {count_before}, after: {count_after} (unchanged — rolled back)\n")

    # --- Conversation continues cleanly after rollback ---

    t.user("Let's move on to operating expenses instead.")
    ctx = t.compile()
    print(f"=== After recovery: {len(ctx.messages)} messages ===\n")
    ctx.pprint()

    t.close()


if __name__ == "__main__":
    main()
