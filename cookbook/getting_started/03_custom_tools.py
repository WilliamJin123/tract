"""Custom Tools: Adding domain-specific tools to the agent loop

Two approaches for giving the agent your own tools alongside tract built-ins:

  1. @t.toolkit.tool decorator  -- recommended; infers schema from type hints
  2. Manual tool dicts           -- full control over the OpenAI function schema

Demonstrates: @t.toolkit.tool, manual tool dicts, tool_handlers, t.llm.run()
Requires: LLM API key (uses Groq provider)
"""

import contextlib
import io
import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.small


# ---------------------------------------------------------------------------
# Tool functions (shared by both approaches)
# ---------------------------------------------------------------------------

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, parentheses."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: only numeric expressions allowed"
    try:
        return f"{expression} = {eval(expression)}"  # noqa: S307
    except Exception as e:
        return f"Error: {e}"


def python_repl(code: str) -> str:
    """Execute a Python snippet and return its stdout. Use print() for output."""
    buf = io.StringIO()
    namespace: dict = {}
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__}, namespace)  # noqa: S102
        output = buf.getvalue()
        return output if output else "(executed, no output)"
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}"


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


# ===================================================================
# Approach 1: @t.toolkit.tool decorator (recommended)
# ===================================================================

def demo_decorator() -> None:
    """Register tools via the decorator -- schema is inferred from type hints."""
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # Register existing functions (no parentheses needed)
        t.toolkit.tool(calculator)
        t.toolkit.tool(python_repl)

        # Or with overrides
        t.toolkit.tool(lookup_constant, name="constants", description="Look up math/physics constants")

        # Inspect registered tools
        print("=== Decorator-registered tools ===\n")
        for name, td in t.toolkit.custom_tools.items():
            print(f"  {name:15s}  {td.description[:60]}")

        # Custom tools merge into as_tools() automatically
        t.system(
            "You are a math assistant. Use the calculator for arithmetic, "
            "constants for lookups, and python_repl for complex calculations."
        )

        print("\n=== Running agent ===\n")

        result = t.llm.run(
            "What is the circumference of a circle with radius 10? "
            "Look up pi with the constants tool, then calculate 2 * pi * 10.",
            max_steps=8,
            profile="full",
            tool_names=["commit", "status", "calculator", "python_repl", "constants"],
        )

        result.pprint(style="chat")
        print(f"\n  Status: {result.status}  |  Steps: {result.steps}  |  Tool calls: {result.tool_calls}")


# ===================================================================
# Approach 2: Manual Tool Schemas
# ===================================================================

# OpenAI function-calling dicts -- useful when you need exact control
# over parameter descriptions, enums, or complex nested schemas.
MANUAL_TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression.",
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

MANUAL_HANDLERS = {
    "calculator": calculator,
    "lookup_constant": lookup_constant,
}


def demo_manual() -> None:
    """Register tools via explicit dicts + handler map."""
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=True,
    ) as t:

        # Get a subset of tract built-ins, then append manual tool defs
        tract_tools = t.toolkit.as_tools(
            profile="full",
            tool_names=["commit", "status", "log"],
            format="openai",
        )
        all_tools = tract_tools + MANUAL_TOOL_DEFS

        print(f"Tract tools:  {[td['function']['name'] for td in tract_tools]}")
        print(f"Custom tools: {[td['function']['name'] for td in MANUAL_TOOL_DEFS]}")

        t.system("You are a math assistant with calculator and constant-lookup tools.")
        result = t.llm.run(
            "What is 2 * pi * 10? Look up pi first, then calculate.",
            max_steps=8,
            tools=all_tools,
            tool_handlers=MANUAL_HANDLERS,
        )

        result.pprint(style="chat")
        print(f"\n  Status: {result.status}  |  Steps: {result.steps}  |  Tool calls: {result.tool_calls}")


def main() -> None:
    print("=" * 60)
    print("PART 1: @t.toolkit.tool decorator (recommended)")
    print("=" * 60, "\n")
    demo_decorator()
    print("\n\n" + "=" * 60)
    print("PART 2: Manual tool schemas + tool_handlers")
    print("=" * 60, "\n")
    demo_manual()


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:           getting_started/01_quick_start.py
# Config & directives:   getting_started/02_config_and_directives.py
# Streaming:             getting_started/04_streaming.py
