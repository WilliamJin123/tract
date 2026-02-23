"""Tool Provenance

Register real tools, let the LLM call them, execute locally, and track
everything in Tract. Every commit records which tools were available.
After the conversation, audit tool provenance for any past commit.

The tool call loop is intentionally manual here — frameworks like Agno
or LangChain automate this. Tract's job is to version and track the
context, not to own the execution loop.

Demonstrates: set_tools(), tool call loop, get_commit_tools(),
              compile().tools, to_openai_params(), content-addressed storage
"""

import json
import os
import subprocess

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "llama-4-scout-17b-16e-instruct"

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

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a bash command and return stdout. "
                       "Use for file listing, date/time, system info, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute (e.g. 'date', 'ls', 'echo hello')",
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
        try:
            result = eval(args["expression"])  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    elif name == "run_bash":
        try:
            result = subprocess.run(
                args["command"],
                shell=True,  # noqa: S602
                capture_output=True,
                text=True,
                timeout=10,
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

        # --- Phase 1: register tools and start conversation ---

        t.set_tools([PYTHON_EVAL_TOOL, BASH_TOOL])
        t.system(
            "You are a helpful assistant with access to tools. "
            "Use python_eval for math and run_bash for system commands. "
            "Always use tools when appropriate — don't compute in your head."
        )

        print("=== Phase 1: tool-assisted conversation ===\n")

        # Ask something that should trigger a tool call
        user_msg = "What is 2**20 + 3**13? Use the python_eval tool."
        r1 = t.chat(user_msg)

        # Check if the LLM made tool calls
        # chat() commits the text response, but the raw API response
        # might contain tool_calls. For a real tool loop we need to
        # go lower-level. Let's demonstrate the manual approach:

        print(f"Response: {r1.text}\n")

        # --- Phase 2: manual tool loop ---
        # This is how frameworks do it under the hood:
        # 1. Compile context with tools
        # 2. Call LLM
        # 3. If tool_calls: execute, commit results, call LLM again
        # 4. Repeat until no more tool_calls

        print("=== Phase 2: manual tool loop ===\n")

        # Commit a new user message
        t.user("Now use run_bash to show today's date, and python_eval to compute 7! (factorial of 7).")

        # Compile and call LLM directly (to see the raw response)
        compiled = t.compile()
        params = compiled.to_openai_params()
        print(f"  Sending {len(params['messages'])} messages + {len(params.get('tools', []))} tools\n")

        # Call the LLM client directly
        client = t._resolve_llm_client("chat")
        llm_kwargs = t._resolve_llm_config("chat")
        response = client.chat(compiled.to_dicts(), tools=compiled.tools, **llm_kwargs)

        message = response["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])

        if tool_calls:
            print(f"  LLM requested {len(tool_calls)} tool call(s):\n")

            # Commit the assistant's tool-calling message (may have null content)
            assistant_content = message.get("content") or ""
            t.assistant(assistant_content, message="tool call request")

            # Execute each tool and commit results
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = json.loads(tc["function"]["arguments"])
                tool_call_id = tc["id"]

                print(f"    Tool: {func_name}({func_args})")
                result = execute_tool(func_name, func_args)
                print(f"    Result: {result}\n")

                # Commit the tool result back into the conversation
                # Using the OpenAI tool result format
                t.commit(
                    role="tool",
                    content=result,
                    message=f"tool result: {func_name}",
                    metadata={"tool_call_id": tool_call_id, "name": func_name},
                )

            # Now generate the final answer (LLM sees tool results in context)
            r2 = t.generate()
            print(f"  Final answer:")
            r2.pprint()
        else:
            # LLM responded directly without tools
            t.assistant(message.get("content", ""))
            print(f"  LLM responded without tool calls: {message.get('content', '')[:100]}")

        # --- Phase 3: change tools mid-session ---

        print("\n=== Phase 3: swap tools ===\n")

        # Remove bash, keep only python_eval
        t.set_tools([PYTHON_EVAL_TOOL])
        r3 = t.chat("What's the square root of 144? Use python_eval.")
        r3.pprint()

        # Clear all tools
        t.set_tools(None)
        r4 = t.chat("Summarize what we computed today.")
        r4.pprint()

        # --- Phase 4: audit tool provenance ---

        print("\n=== Tool provenance audit ===\n")

        for entry in t.log():
            tools = t.get_commit_tools(entry.commit_hash)
            tool_names = [d["function"]["name"] for d in tools] if tools else ["(none)"]
            role = entry.content_type if hasattr(entry, "content_type") else "?"
            print(f"  {entry.commit_hash[:8]}  {role:<12}  tools: {', '.join(tool_names)}")

        # --- Show full session ---

        print("\n=== Full session ===\n")
        t.compile().pprint(style="chat")


if __name__ == "__main__":
    main()
