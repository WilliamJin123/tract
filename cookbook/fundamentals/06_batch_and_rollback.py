"""Batch Rollback

Group multiple commits into a single atomic unit — they all land or none do.
Useful for RAG pipelines where retrieval + question + setup must arrive
together, or any multi-step operation where partial state is worse than
no state. Rollback leaves state clean for a safe retry. After a successful
batch, chat with the LLM to verify the batched context is usable.

Demonstrates: batch() context manager, rollback on failure, clean retry
              after rollback, compile() before/after batch, chat() verification,
              set_tools(), tool_result(), ToolCall for realistic RAG pattern
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.protocols import ToolCall

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"

# Tool definition — the LLM knows about this tool, but we execute it
# deterministically (no LLM decision needed for the retrieval step).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_financials",
            "description": "Fetch quarterly financial data from the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {
                    "quarter": {
                        "type": "string",
                        "description": "The fiscal quarter, e.g. 'Q4 2025'.",
                    },
                },
                "required": ["quarter"],
            },
        },
    },
]


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
    """Commit a RAG retrieval as one atomic batch.

    The batch contains a realistic tool-call flow:
      1. User asks a question
      2. Assistant "decides" to call the retrieval tool (deterministic here)
      3. Tool result is committed via tool_result()
      4. User asks a follow-up

    All four commits land atomically — or none do.
    """
    with t.batch():
        t.user(f"What were {quarter}'s revenue figures?")

        # Simulate the assistant deciding to call the retrieval tool.
        # In a real agentic loop the LLM would return this; here we
        # construct it deterministically so the batch stays predictable.
        tc = ToolCall(
            id=f"call_{quarter.lower().replace(' ', '_')}",
            name="fetch_financials",
            arguments={"quarter": quarter},
        )
        t.assistant("", metadata={"tool_calls": [tc.to_dict()]})

        # Execute the tool — this is the flaky part that can fail mid-batch
        data = fetch_financials(quarter)
        t.tool_result(tc.id, tc.name, data)

        t.user("How does that compare to guidance?")


def main():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a financial analyst assistant.")
        t.set_tools(TOOLS)

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
        print(f"After retry: {len(ctx.messages)} messages (system + 4 from batch)\n")
        ctx.pprint()

        # --- Verify: chat with the batched context ---

        print("\n=== Verify: chat with the batched context ===\n")

        r = t.chat(
            "Based on the Q4 2025 data, how does the operating margin "
            "compare to typical tech companies?"
        )
        r.pprint()

        print("\nThe LLM used all batched context to answer.")


if __name__ == "__main__":
    main()
