"""Custom Tools: Adding domain-specific tools to the agent loop

run_loop() gives the agent tract's built-in context tools by default. This
example shows how to add your own tools alongside them -- the agent gets
both tract tools (commit, log, status...) and your custom functions.

Two techniques:
  1. ToolProfile -- select which built-in tract tools the agent gets
  2. Custom tool dicts -- add domain-specific functions in OpenAI format

Demonstrates: ToolProfile, ToolConfig, as_tools(), custom tool dicts, run_loop()

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract, ToolProfile, ToolConfig, LoopConfig, run_loop

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
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


# OpenAI function-calling format
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


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # --- 1. Select a subset of tract tools via ToolProfile ---

        slim_profile = ToolProfile(
            name="slim",
            tool_configs={
                "commit": ToolConfig(enabled=True),
                "status": ToolConfig(enabled=True),
                "log": ToolConfig(enabled=True),
                "compile": ToolConfig(enabled=True),
            },
        )

        tract_tools = t.as_tools(profile=slim_profile, format="openai")
        print(f"Tract tools ({len(tract_tools)}): "
              f"{[td['function']['name'] for td in tract_tools]}")
        print(f"Custom tools ({len(CUSTOM_TOOLS)}): "
              f"{[td['function']['name'] for td in CUSTOM_TOOLS]}")

        # --- 2. Combine tract tools + custom tools ---

        all_tools = tract_tools + CUSTOM_TOOLS

        t.system(
            "You are a math assistant. You have tools for calculations "
            "(calculator, lookup_constant) and for managing your context "
            "history (commit, status, log, compile). Use the calculator "
            "for any arithmetic."
        )

        # --- 3. Run with combined tools ---

        config = LoopConfig(max_steps=8, stop_on_no_tool_call=True)

        result = run_loop(
            t,
            task=(
                "What is the circumference of a circle with radius 10? "
                "Look up pi, then calculate 2 * pi * 10."
            ),
            config=config,
            tools=all_tools,
            on_step=lambda step, _resp: print(f"  step {step}..."),
        )

        result.pprint()


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:      getting_started/01_quick_start.py
# Rules:            getting_started/02_rules.py
# Agent patterns:   agent/
