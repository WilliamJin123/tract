"""Streaming -- See tokens as they arrive

Tract supports streaming with any LLM client that has a stream() method.
Pass on_token= to t.run() and text arrives chunk-by-chunk instead of
waiting for the full response.

Demonstrates: on_token callback, stream=True, LoopResult after streaming

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.small


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        t.system("You are a helpful assistant. Answer concisely.")

        # ---------------------------------------------------------
        # Streaming with on_token callback
        # ---------------------------------------------------------
        # Each text chunk from the LLM hits the callback as it arrives.
        # The full response is still committed to the tract and returned
        # in the LoopResult as usual.

        print("=== Streaming Response ===\n")

        token_count = 0

        def on_token(text: str) -> None:
            """Print each token as it arrives."""
            nonlocal token_count
            token_count += 1
            print(text, end="", flush=True)

        result = t.run(
            "Explain what a hash table is in 3 sentences.",
            max_steps=3,
            tools=[],
            on_token=on_token,
        )

        # Newline after streamed output
        print(f"\n\n--- Streamed {token_count} chunks ---\n")

        # The LoopResult is the same regardless of streaming
        print(f"Status: {result.status}")
        print(f"Steps: {result.steps}")
        print(f"Final response length: {len(result.final_response or '')}")

        # ---------------------------------------------------------
        # Streaming with tools
        # ---------------------------------------------------------
        # Streaming also works when the LLM calls tools. Text chunks
        # stream to the callback; tool calls are accumulated and
        # executed normally after the stream completes.

        print("\n=== Streaming with Tools ===\n")

        result2 = t.run(
            "Check your status and then summarize your current context.",
            max_steps=5,
            tool_names=["status", "log"],
            on_token=lambda text: print(text, end="", flush=True),
        )
        print(f"\n\nStatus: {result2.status}, tool calls: {result2.tool_calls}")


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:       getting_started/01_quick_start.py
# Config:            getting_started/02_config_and_directives.py
# Custom tools:      getting_started/03_custom_tools.py
