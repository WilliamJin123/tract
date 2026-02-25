"""Provenance Patterns

Three patterns for auditing what happened in a tract: which LLM config
produced each response, which tools were available at each commit, and
the full edit history of any message over time.

Part 1: Config provenance — query_by_config(), generation_config on commits
Part 2: Tool provenance — set_tools(), tool_calls, get_commit_tools()
Part 3: Edit history — edit_history(), restore(), get_content()

Each part uses its own Tract instance for clarity.

Demonstrates: generation_config, query_by_config() patterns (single-field,
              multi-field AND, comparison operators, IN operator),
              set_tools(), ChatResponse.tool_calls, ToolCall API,
              tool_result(), ToolCall.from_openai(), get_commit_tools(),
              edit_history(), restore(), get_content(), t.assistant(edit=),
              Priority.SKIP for cleaning up intermediate commits,
              response.pprint(), pprint(style="chat"), pprint(style="compact"),
              pprint(style="table")
"""

import json
import os
import subprocess

from dotenv import load_dotenv

from tract import Priority, Tract, ToolCall

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"
MODEL_ID_SMALL = "llama3.1-8b"


# =============================================================================
# Part 1: Config Provenance
# =============================================================================
# Chat with different settings across calls, then query: "what temperature
# was used for this output?" Every assistant commit auto-captures the
# fully-resolved generation_config. query_by_config() searches by single
# field, multi-field AND, or comparison operators.

def part1_config_provenance():
    print("=" * 60)
    print("Part 1: CONFIG PROVENANCE")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a creative writing assistant. Keep responses under 2 sentences.")

        # --- Turn 1: default settings ---

        print("=== Turn 1: default config ===\n")
        r1 = t.chat("Write a one-sentence opening for a mystery novel.")
        r1.pprint()

        # --- Turn 2: high temperature for more creativity ---

        print("\n=== Turn 2: temperature=1.5 ===\n")
        r2 = t.chat(
            "Now write a wilder, more surreal version.",
            temperature=1.5,
        )
        r2.pprint()

        # --- Turn 3: low temperature + limited tokens ---

        print("\n=== Turn 3: temperature=0.0, max_tokens=200 ===\n")
        r3 = t.chat(
            "Write a final version — precise, clinical, no embellishment.",
            temperature=0.0,
            max_tokens=200,
        )
        r3.pprint()

        # --- Full session view ---

        print("\n=== Full session ===\n")
        t.compile().pprint(style="chat")

        # --- Query: single field with comparison operator ---
        # "Which calls used a high temperature?"

        print("\n=== Query: temperature > 1.0 ===\n")
        hot = t.query_by_config("temperature", ">", 1.0)
        print(f"  {len(hot)} commit(s) with temperature > 1.0:")
        for c in hot:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

        # --- Query: equality match ---

        print("\n=== Query: max_tokens = 200 ===\n")
        limited = t.query_by_config("max_tokens", "=", 200)
        print(f"  {len(limited)} commit(s) with max_tokens=200:")
        for c in limited:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  max_tokens={cfg.max_tokens}")

        # --- Query: inclusive range with "between" ---
        # "Which calls used a temperature between 0.0 and 1.0 (inclusive)?"

        print("\n=== Query: temperature between [0.0, 1.0] ===\n")
        moderate = t.query_by_config("temperature", "between", [0.0, 1.0])
        print(f"  {len(moderate)} commit(s) with temperature between [0.0, 1.0]:")
        for c in moderate:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

        # --- Query: multi-field AND ---
        # "Which calls used THIS model AND temperature=0.0?"

        print(f"\n=== Query: model={MODEL_ID} AND temperature=0.0 ===\n")
        specific = t.query_by_config(conditions=[
            ("model", "=", MODEL_ID),
            ("temperature", "=", 0.0),
        ])
        print(f"  {len(specific)} commit(s) match:")
        for c in specific:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  model={cfg.model}, temp={cfg.temperature}")

        # --- Query: IN operator (set membership) ---
        # "Which calls used temperature 0.0 or 1.5?"

        print("\n=== Query: temperature in list [0.0, 1.5] ===\n")
        extremes = t.query_by_config("temperature", "in", [0.0, 1.5])
        print(f"  {len(extremes)} commit(s) at extreme temperatures:")
        for c in extremes:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

        # --- All commits: config provenance summary ---

        print("\n=== All assistant commits — config provenance ===\n")
        for entry in reversed(t.log()):
            if entry.generation_config:
                fields = entry.generation_config.non_none_fields()
                parts = [f"{k}={v}" for k, v in fields.items()]
                print(f"  {entry.commit_hash[:8]}: {', '.join(parts)}")


# =============================================================================
# Part 2: Tool Provenance
# =============================================================================
# Register real tools, let the LLM call them, execute locally, and track
# everything in Tract. Every commit records which tools were available.
# After the conversation, audit tool provenance for any past commit.

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


def part2_tool_provenance():
    print(f"\n{'=' * 60}")
    print("Part 2: TOOL PROVENANCE")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
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
            # pprint shows the tool-calling response with magenta "Tool Call" panel
            r.pprint()

            print(f"\n  LLM requested {len(r.tool_calls)} tool call(s):\n")

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

        # Show all three pprint styles to verify tool call rendering
        print("\n  --- Compact view (tool calls as one-liners) ---\n")
        t.compile().pprint(style="compact")

        print("\n  --- Table view (tool calls in table format) ---\n")
        t.compile().pprint(style="table")

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


# =============================================================================
# Part 3: Edit History
# =============================================================================
# Chat with an LLM, then iteratively refine a response via edits.
# Use edit_history() to see every version of a commit, and restore()
# to roll back when the edits go too far.

def part3_edit_history():
    print(f"\n{'=' * 60}")
    print("Part 3: EDIT HISTORY")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID_SMALL,
    ) as t:

        t.system("You are a concise writing assistant. Keep answers under 2 sentences.")

        # --- Initial conversation ---

        print("=== Initial question ===\n")
        r1 = t.chat("Explain what a black hole is.")
        r1.pprint()
        original_hash = r1.commit_info.commit_hash

        # --- Ask a follow-up so we have surrounding context ---

        print("=== Follow-up ===\n")
        r2 = t.chat("How are they detected?")
        r2.pprint()

        # --- Edit the first response to add more detail ---
        # t.assistant(edit=...) replaces the content of a previous response.
        # All edits point to the ORIGINAL commit (flat design, not chained).

        print("=== Edit 1: ask LLM to improve the first answer ===\n")
        improve = t.chat(
            "Please rewrite your first answer about black holes to also "
            "mention the event horizon. Keep it to 2 sentences."
        )
        # Use the LLM's improved text as an edit of the original
        e1 = t.assistant(
            improve.text,
            edit=original_hash,
            message="Add event horizon detail",
        )
        print(f"  Edit commit: {e1.commit_hash[:8]}")
        print(f"  Content: {t.get_content(e1)}\n")

        # The t.chat() call above created intermediate commits (user prompt +
        # LLM response) that would clutter the compiled context. SKIP them
        # so only the edit itself survives in the conversation view.
        t.annotate(improve.commit_info.parent_hash, Priority.SKIP)
        t.annotate(improve.commit_info.commit_hash, Priority.SKIP)

        # --- Edit again: further refinement ---

        print("=== Edit 2: manual refinement ===\n")
        e2 = t.assistant(
            "A black hole is a region of spacetime where gravity is so "
            "extreme that nothing, not even light, can escape past its "
            "event horizon. They form when massive stars collapse at the "
            "end of their life cycle.",
            edit=original_hash,
            message="Manual rewrite for clarity",
        )
        print(f"  Edit commit: {e2.commit_hash[:8]}")

        # --- View the full edit history ---
        # edit_history() returns [original, edit1, edit2, ...] in order.
        # This is a lightweight query -- no full context compilation needed.

        print("\n=== Edit history for the first answer ===\n")
        history = t.edit_history(original_hash)
        for i, version in enumerate(history):
            label = "ORIGINAL" if i == 0 else f"EDIT {i}"
            content = t.get_content(version)
            print(f"  v{i} ({label}) [{version.commit_hash[:8]}]")
            print(f"     {content}")
            print()

        print(f"  Total versions: {len(history)}")

        # --- The compiled context uses the latest edit automatically ---

        print("\n=== Compiled context (latest edit wins) ===\n")
        t.compile().pprint(style="chat")

        # --- Restore: the manual edit was too verbose, go back to v1 ---
        # restore() creates a NEW edit pointing to the original, with the
        # content from the specified version. The full history is preserved.

        print("\n=== Restore to v1 (LLM-improved version) ===\n")
        restored = t.restore(original_hash, version=1)
        print(f"  Restore commit: {restored.commit_hash[:8]}")
        print(f"  edit_target: {restored.edit_target[:8]} (points to original)")
        print(f"  Content: {t.get_content(restored)}\n")

        # --- Verify the restore is tracked in history ---

        print("=== Updated edit history (restore is itself an edit) ===\n")
        updated_history = t.edit_history(original_hash)
        for i, version in enumerate(updated_history):
            msg = version.message or "(no message)"
            if len(msg) > 60:
                msg = msg[:57] + "..."
            print(f"  v{i} [{version.commit_hash[:8]}] {msg}")
        print(f"\n  Total versions: {len(updated_history)} "
              f"(was {len(history)}, +1 from restore)")

        # --- Surrounding context is unaffected ---

        print("\n=== Full compiled context after restore ===\n")
        ctx = t.compile()
        ctx.pprint(style="chat")
        print(f"\n  The follow-up answer about detection is still intact.")
        print(f"  Only the black hole definition was rolled back to v1.")


def main():
    part1_config_provenance()
    part2_tool_provenance()
    part3_edit_history()


if __name__ == "__main__":
    main()
