"""Tool Provenance (Typed ToolCall API)

Register real tools, let the LLM call them, execute locally, and track
everything in Tract. Every commit records which tools were available.
After the conversation, audit tool provenance for any past commit.

Uses the typed ToolCall API:
  - ChatResponse.tool_calls: list[ToolCall] | None
  - ToolCall fields: tc.id, tc.name, tc.arguments (parsed dict)
  - t.tool_result(tool_call_id, name, content) convenience method
  - ToolCall.from_openai() for manual parsing

Demonstrates: set_tools(), tool_calls detection, tool_result(),
              ToolCall.from_openai(), get_commit_tools(), manual tool loop
"""

import json
import os
import subprocess

from dotenv import load_dotenv

from tract import Tract, ToolCall

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"

# --- Tool definitions (OpenAI function calling format) ---

PYTHON_EVAL_TOOL = {
    "type": "function",
    "function": {
        "name": "python_eval",
        "description": "Evaluate a Python expression and return the result. "
                       "Use for math, string operations, list comprehensions, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A Python expression to evaluate (e.g. '2**10 + 3**7')",
                },
            },
            "required": ["expression"],
        },
    },
}

SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_shell",
        "description": "Run a shell command and return stdout. "
                       "Use for echo, file listing, system info, etc. "
                       "Commands must be cross-platform (avoid OS-specific syntax).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "A shell command to execute (e.g. 'echo hello', 'python --version')",
                },
            },
            "required": ["command"],
        },
    },
}


# --- Tool execution ---

def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name and return the result string."""
    if name == "python_eval":
        expr = args["expression"]
        try:
            # Try eval first (handles expressions like '2**10 + 3**7')
            return str(eval(expr))  # noqa: S307
        except SyntaxError:
            # Fallback: exec statements, eval the last part.
            # Handles 'import math; math.factorial(7)' by splitting on ';'
            try:
                parts = [p.strip() for p in expr.split(";")]
                ns: dict = {}
                for part in parts[:-1]:
                    exec(part, ns)  # noqa: S102
                return str(eval(parts[-1], ns))  # noqa: S307
            except Exception as e:
                return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "run_shell":
        try:
            result = subprocess.run(
                args["command"],
                shell=True,  # noqa: S602
                capture_output=True,
                text=True,
                timeout=10,
                stdin=subprocess.DEVNULL,  # prevent interactive prompts
            )
            return result.stdout.strip() or result.stderr.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out"
        except Exception as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"


def main():
    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        # ============================================================
        # Phase 1: Simple chat() with tool call detection
        # ============================================================
        # Register tools, send a message, check r.tool_calls.
        # If the LLM wants to call tools, execute them with
        # t.tool_result() and then t.generate() for the final answer.

        print("=== Phase 1: chat() with typed tool calls ===\n")

        t.set_tools([PYTHON_EVAL_TOOL, SHELL_TOOL])
        t.system(
            "You are a helpful assistant with access to tools. "
            "Use python_eval for math and run_shell for system commands. "
            "Always use tools when appropriate -- don't compute in your head."
        )

        r = t.chat("What is 2**20 + 3**13? Use the python_eval tool.")

        if r.tool_calls:
            # The LLM requested tool calls -- typed access via ToolCall
            print(f"  LLM requested {len(r.tool_calls)} tool call(s):\n")

            for tc in r.tool_calls:
                # tc.name, tc.arguments (dict), tc.id -- all typed, no JSON parsing
                print(f"    Tool: {tc.name}")
                print(f"    Args: {tc.arguments}")
                print(f"    ID:   {tc.id}")

                result = execute_tool(tc.name, tc.arguments)
                print(f"    Result: {result}\n")

                # Commit the tool result using the convenience method
                t.tool_result(tc.id, tc.name, result)

            # Generate the final answer (LLM sees tool results in context)
            r_final = t.generate()
            print("  Final answer:")
            r_final.pprint()
        else:
            # LLM responded directly (no tool calls)
            print("  Direct response (no tool calls):")
            r.pprint()

        # ============================================================
        # Phase 2: Manual tool loop (educational)
        # ============================================================
        # Lower-level approach: commit user message, compile, call
        # LLM directly, parse response with ToolCall.from_openai(),
        # commit results with t.tool_result(), loop until done.

        print("\n=== Phase 2: manual tool loop ===\n")

        t.user("Use run_shell to run 'echo hello world', and python_eval to compute 7! (factorial of 7).")

        # Loop until the LLM stops requesting tool calls (cap at 5 iterations)
        max_iterations = 5
        iteration = 0
        while iteration < max_iterations:
            iteration += 1
            print(f"  --- Iteration {iteration} ---")

            # Compile and inspect params
            compiled = t.compile()
            params = compiled.to_openai_params()
            print(f"  Sending {len(params['messages'])} messages + {len(params.get('tools', []))} tools")

            # Call the LLM client directly
            client = t._resolve_llm_client("chat")
            llm_kwargs = t._resolve_llm_config("chat")
            if compiled.tools:
                llm_kwargs["tools"] = compiled.tools
            response = client.chat(compiled.to_dicts(), **llm_kwargs)

            # Parse tool_calls from the raw response using ToolCall.from_openai()
            message = response["choices"][0]["message"]
            raw_tool_calls = message.get("tool_calls")

            if raw_tool_calls:
                # Parse with the typed API -- from_openai handles JSON string -> dict
                tool_calls = [ToolCall.from_openai(tc) for tc in raw_tool_calls]
                print(f"  LLM requested {len(tool_calls)} tool call(s):\n")

                # Commit the assistant's tool-calling message (include tool_calls
                # in metadata so compile() produces correct round-trip format)
                assistant_content = message.get("content") or ""
                t.assistant(
                    assistant_content,
                    message="tool call request",
                    metadata={"tool_calls": [tc.to_dict() for tc in tool_calls]},
                )

                # Execute each tool and commit results
                for tc in tool_calls:
                    print(f"    Tool: {tc.name}({tc.arguments})")
                    result = execute_tool(tc.name, tc.arguments)
                    print(f"    Result: {result}\n")

                    # Use the convenience method -- handles role, metadata, formatting
                    t.tool_result(tc.id, tc.name, result)
            else:
                # No more tool calls -- commit the final text and break
                final_text = message.get("content", "")
                t.assistant(final_text, message="final answer")
                print(f"  Final answer: {final_text[:200]}")
                break

        # ============================================================
        # Phase 3: Swap tools mid-session
        # ============================================================
        # set_tools() changes available tools for subsequent commits.
        # set_tools(None) clears all tools.

        print("\n\n=== Phase 3: swap tools ===\n")

        # Keep only python_eval
        t.set_tools([PYTHON_EVAL_TOOL])
        r3 = t.chat("What's the square root of 144? Use python_eval.")

        if r3.tool_calls:
            for tc in r3.tool_calls:
                result = execute_tool(tc.name, tc.arguments)
                t.tool_result(tc.id, tc.name, result)
            r3 = t.generate()

        r3.pprint()

        # Clear all tools
        t.set_tools(None)
        r4 = t.chat("Summarize what we computed today.")
        r4.pprint()

        # ============================================================
        # Phase 4: Audit tool provenance
        # ============================================================
        # Every commit records which tools were AVAILABLE at the time.
        # get_commit_tools() retrieves the tool definitions for any commit.

        print("\n=== Tool provenance audit ===\n")

        for entry in t.log(limit=50):
            tools = t.get_commit_tools(entry.commit_hash)
            tool_names = [d["function"]["name"] for d in tools] if tools else ["(none)"]
            print(f"  {entry.commit_hash[:8]}  {entry.content_type:<12}  tools: {', '.join(tool_names)}")

        # --- Show full session ---

        print("\n=== Full session ===\n")
        t.compile().pprint(style="chat")


if __name__ == "__main__":
    main()
