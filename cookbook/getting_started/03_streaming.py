"""Streaming -- See tokens as they arrive

Tract supports streaming with any LLM client that has a stream() method.
Pass on_token= to t.llm.run() and text arrives chunk-by-chunk instead of
waiting for the full response.

Two approaches:
  1. Raw callback -- print(text, end="", flush=True) for plain text output
  2. StreamPrinter -- Rich-formatted markdown panel that re-renders live

Demonstrates: on_token callback, StreamPrinter, stream=True, LoopResult

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract
from tract.formatting import StreamPrinter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _logging import StepLogger
from _providers import groq as llm

MODEL_ID = llm.small


def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
    ) as t:

        t.system("You are a helpful assistant. Answer concisely.")

        # ---------------------------------------------------------
        # 1. Rich streaming with StreamPrinter
        # ---------------------------------------------------------
        # StreamPrinter renders streamed tokens as a live Rich Markdown
        # panel.  It batches re-renders (every N chunks / min interval)
        # to avoid flicker.  Use as a context manager for auto-finish.

        print("=== Rich Streaming (StreamPrinter) ===\n")

        with StreamPrinter(title="Hash Tables") as printer:
            result = t.llm.run(
                "Explain what a hash table is in 3 sentences.",
                max_steps=3,
                tools=[],
                on_token=printer,
            )

        print(f"Streamed {printer.chunk_count} chunks")
        result.pprint()

        # ---------------------------------------------------------
        # 2. Raw streaming with plain callback
        # ---------------------------------------------------------
        # For maximum control, pass any callable as on_token.

        print("\n=== Raw Streaming ===\n")

        token_count = 0

        def on_token(text: str) -> None:
            """Print each token as it arrives."""
            nonlocal token_count
            token_count += 1
            print(text, end="", flush=True)

        result2 = t.llm.run(
            "What is a linked list? One sentence.",
            max_steps=3,
            tools=[],
            on_token=on_token,
        )

        print(f"\n\n--- Streamed {token_count} chunks ---")
        print(f"Status: {result2.status}")

        # ---------------------------------------------------------
        # 3. Streaming with tools
        # ---------------------------------------------------------
        # Streaming also works when the LLM calls tools. Text chunks
        # stream to the callback; tool calls are accumulated and
        # executed normally after the stream completes.

        print("\n=== Streaming with Tools ===\n")

        log = StepLogger()

        with StreamPrinter(title="Tool + Stream") as printer:
            result3 = t.llm.run(
                "Check your status and then summarize your current context.",
                max_steps=5,
                profile="full",
                tool_names=["status", "log"],
                on_token=printer,
                on_tool_result=log.on_tool_result,
            )
        print(f"Status: {result3.status}, tool calls: {result3.tool_calls}")


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:       getting_started/01_quick_start.py
# Config:            getting_started/02_config_and_directives.py
# Custom tools:      getting_started/03_custom_tools.py
