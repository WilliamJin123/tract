"""Quick Start -- From zero to working agent in 5 minutes

The minimum setup: open a Tract, configure settings, run an agent loop, done.
One t.llm.run() call handles compile -> LLM -> tools -> repeat automatically.

Demonstrates: Tract.open(), system(), configure(), t.llm.run(), LoopResult

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract

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

        # Config settings travel with the conversation in the DAG
        t.config.set(temperature=0.7, compile_strategy="full")

        t.system(
            "You are a helpful assistant. Answer questions concisely."
        )

        # One call: compiles context, calls LLM with tools, repeats until done
        # tools=[] since this is pure Q&A -- no context-management tools needed
        log = StepLogger()

        result = t.llm.run(
            "What are the three pillars of object-oriented programming? "
            "Explain each in one sentence.",
            max_steps=5,
            tools=[],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        result.pprint()

        print(f"\nFinal answer:\n  {result.final_response}")


if __name__ == "__main__":
    main()


# --- See also ---
# Config & directives: getting_started/02_config_and_directives.py
# Custom tools:        getting_started/03_custom_tools.py
# Agent patterns:      agent/
