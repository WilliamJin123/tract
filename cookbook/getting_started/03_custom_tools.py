"""Custom Tools: Adding domain-specific tools to the agent loop

t.run() gives the agent tract's built-in context tools by default. This
example shows how to add your own tools alongside them -- the agent gets
both tract tools (commit, log, status...) and your custom functions.

Two techniques:
  1. tool_names -- select which built-in tract tools the agent gets
  2. tool_handlers + custom tool dicts -- add domain-specific functions

Demonstrates: tool_names, tool_handlers, custom tool dicts, t.run()

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _logging import StepLogger
from _providers import groq as llm

MODEL_ID = llm.small


# --- Custom tools: plain functions the agent can call ---

def calculator(expression: str) -> str:
    """Evaluate a math expression safely."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions allowed"
    try:
        result = eval(expression)  # noqa: S307
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error: {e}"


def lookup_constant(name: str) -> str:
    """Look up a named mathematical or physical constant."""
    constants = {
        "pi": "3.14159265358979",
        "e": "2.71828182845905",
        "c": "299792458 m/s (speed of light)",
        "g": "9.80665 m/s^2 (gravitational acceleration)",
        "avogadro": "6.022e23 (Avogadro's number)",
    }
    return constants.get(name.lower(), f"Unknown constant: {name}")


# OpenAI function-calling format for custom tools
CUSTOM_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression. Supports +, -, *, /, parentheses.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression to evaluate."},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_constant",
            "description": "Look up a named constant (pi, e, c, g, avogadro).",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Constant name."},
                },
                "required": ["name"],
            },
        },
    },
]

# Handlers: map tool names to the Python functions that execute them
CUSTOM_HANDLERS = {
    "calculator": calculator,
    "lookup_constant": lookup_constant,
}


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=True,
    ) as t:

        # --- 1. Select a slim subset of built-in tract tools ---

        tract_tools = t.as_tools(
            profile="full",
            tool_names=["commit", "status", "log", "compile"],
            format="openai",
        )
        print(f"Tract tools ({len(tract_tools)}): "
              f"{[td['function']['name'] for td in tract_tools]}")
        print(f"Custom tools ({len(CUSTOM_TOOLS)}): "
              f"{[td['function']['name'] for td in CUSTOM_TOOLS]}")

        # --- 2. Combine tract tools + custom tool schemas ---

        all_tools = tract_tools + CUSTOM_TOOLS

        t.system(
            "You are a math assistant. You have tools for calculations "
            "(calculator, lookup_constant) and for managing your context "
            "history (commit, status, log, compile). Use the calculator "
            "for any arithmetic."
        )

        # --- 3. Run with combined tools + handlers ---

        log = StepLogger()

        result = t.run(
            "What is the circumference of a circle with radius 10? "
            "Look up pi, then calculate 2 * pi * 10.",
            max_steps=8,
            tools=all_tools,
            tool_handlers=CUSTOM_HANDLERS,
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        result.pprint(style="chat")

        print(f"\nFinal answer:\n  {result.final_response}")


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:           getting_started/01_quick_start.py
# Config & directives:   getting_started/02_config_and_directives.py
# Agent patterns:        agent/
