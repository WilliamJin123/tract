"""Quick Start -- From zero to working agent in 5 minutes

The minimum setup: open a Tract, configure rules, run an agent loop, done.
One t.run() call handles compile -> LLM -> tools -> repeat automatically.

Demonstrates: Tract.open(), system(), rules, t.run(), LoopResult

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

        # Rules configure behavior as data -- they travel with the conversation
        t.rule("temperature", trigger="active",
               action={"type": "set_config", "key": "temperature", "value": 0.7})
        t.rule("compile-strategy", trigger="active",
               action={"type": "set_config", "key": "compile_strategy", "value": "full"})

        t.system(
            "You are a helpful assistant. Answer questions concisely. "
            "You have tools for managing your own context history."
        )

        # One call: compiles context, calls LLM with tools, repeats until done
        result = t.run(
            "What are the three pillars of object-oriented programming? "
            "Explain each in one sentence.",
            max_steps=5,
            on_step=lambda step, _resp: print(f"  step {step}..."),
        )

        print(f"\nStatus:  {result.status}")
        print(f"Steps:   {result.steps}")
        if result.final_response:
            print(f"\n{result.final_response}")


if __name__ == "__main__":
    main()


# --- See also ---
# Rules in depth: getting_started/02_rules.py
# Custom tools:   getting_started/03_custom_tools.py
# Agent patterns: agentic/tool_use/
