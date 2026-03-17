"""Async -- Non-blocking LLM calls with achat(), arun(), and acompress()

Every LLM-facing method has an async counterpart (a-prefix). DAG
operations (commit, compile, branch) stay sync -- they hit local
SQLite and are instant. Async is for the network-bound LLM calls.

Demonstrates: achat(), arun() with async tool handlers, acompress()

Requires: LLM API key (uses Groq provider)
"""

import asyncio
import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _logging import StepLogger
from _providers import groq as llm

MODEL_ID = llm.small


# --- Async tool handler: arun() dispatches these without blocking ---

async def async_lookup(topic: str) -> str:
    """Simulate an async I/O operation (e.g. HTTP fetch, DB query)."""
    await asyncio.sleep(0.01)  # pretend network call
    data = {
        "python": "Python is a high-level, interpreted programming language.",
        "rust": "Rust is a systems programming language focused on safety.",
        "javascript": "JavaScript is the language of the web platform.",
    }
    return data.get(topic.lower(), f"No info on {topic}")


CUSTOM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "async_lookup",
            "description": "Look up information about a programming language.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Language name."},
                },
                "required": ["topic"],
            },
        },
    },
]

CUSTOM_HANDLERS = {
    "async_lookup": async_lookup,  # arun() auto-detects async handlers
}


async def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
    ) as t:

        t.system("You are a helpful assistant. Be concise.")
        t.config.set(temperature=0.7)

        # --- 1. achat() -- async single-turn Q&A ---

        print("=== achat() ===")
        response = await t.llm.achat("What is the capital of France?")
        print(f"  Response: {(response.text or '(no response)')[:200]}")
        print(f"  Tokens: {response.usage}")

        # --- 2. arun() -- async agent loop with async tool handlers ---
        # arun_loop auto-detects async handlers via inspect.iscoroutinefunction
        # and awaits them directly; sync handlers run in asyncio.to_thread()

        print("\n=== arun() with async tools ===")
        log = StepLogger()

        result = await t.llm.arun(
            "Look up Python and Rust, then compare them briefly.",
            max_steps=8,
            tools=CUSTOM_TOOLS,
            tool_handlers=CUSTOM_HANDLERS,
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        print(f"\n  Final: {(result.final_response or '(no response)')[:200]}")
        print(f"  Steps: {result.steps}")

        # --- 3. acompress() -- async LLM-powered summarization ---
        # Useful in long-running agents to keep context within token budget

        print("\n=== acompress() ===")
        before = t.compile()
        print(f"  Before: {before.token_count} tokens, {before.commit_count} commits")

        compressed = await t.compression.acompress(
            content="User asked about Python and Rust, agent compared them.",
        )
        after = t.compile()
        saved = compressed.original_tokens - compressed.compressed_tokens
        print(f"  After:  {after.token_count} tokens, {after.commit_count} commits")
        print(f"  Saved:  {saved} tokens")


if __name__ == "__main__":
    asyncio.run(main())


# --- See also ---
# Sync equivalents:  getting_started/01_quick_start.py
# Custom tools:      getting_started/03_custom_tools.py
# Agent patterns:    agent/
