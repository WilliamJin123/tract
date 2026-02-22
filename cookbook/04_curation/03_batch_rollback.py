"""Batch Rollback

Group multiple commits into a single atomic unit — they all land or none do.
Useful for RAG pipelines where retrieval + question + setup must arrive
together, or any multi-step operation where partial state is worse than
no state. Rollback leaves state clean for a safe retry.

Demonstrates: batch() context manager, rollback on failure, clean retry
              after rollback, compile() before/after batch
"""

from tract import Tract


def fetch_financials(quarter: str) -> str:
    """Simulate a flaky data source that fails on the first call."""
    if not hasattr(fetch_financials, "_called"):
        fetch_financials._called = True
        raise ConnectionError(f"Data source timeout fetching {quarter}")
    return (
        f"{quarter} Revenue: $2.3B (+12% YoY)\n"
        f"Operating margin: 28.4%\n"
        f"Net income: $487M"
    )


def run_rag_batch(t: Tract, quarter: str) -> None:
    """Commit a RAG retrieval as one atomic batch."""
    with t.batch():
        t.user(f"What were {quarter}'s revenue figures?")
        data = fetch_financials(quarter)
        t.assistant(f"[Retrieved from knowledge base]\n{data}")
        t.user("How does that compare to guidance?")


def main():
    t = Tract.open()

    t.system("You are a financial analyst assistant.")

    # --- Attempt 1: data source fails mid-batch, everything rolls back ---

    print("=== Attempt 1: data source timeout ===\n")

    count_before = len(t.compile().messages)

    try:
        run_rag_batch(t, "Q4 2025")
    except ConnectionError as e:
        print(f"Caught: {e}")

    count_after = len(t.compile().messages)
    print(f"Messages before: {count_before}, after: {count_after} (unchanged — rolled back)\n")

    # --- Attempt 2: same batch, clean retry succeeds ---

    print("=== Attempt 2: retry succeeds ===\n")

    run_rag_batch(t, "Q4 2025")

    ctx = t.compile()
    print(f"After retry: {len(ctx.messages)} messages (system + 3 from batch)\n")
    ctx.pprint()

    t.close()


if __name__ == "__main__":
    main()
