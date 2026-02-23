"""Tool Schema Provenance

Track which tool definitions were available at each point in the
conversation. set_tools() registers schemas that auto-link to every
subsequent commit. Change the tool set mid-session and audit exactly
what tools any past commit had access to.

Demonstrates: set_tools(), get_commit_tools(), get_tools(),
              compile().tools, to_openai_params(), content-addressed storage
"""

from tract import Tract


def main():
    t = Tract.open()

    # --- Define two tool sets ---

    search_tool = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    }

    calculator_tool = {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression"},
                },
                "required": ["expression"],
            },
        },
    }

    code_tool = {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute Python code in a sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to run"},
                },
                "required": ["code"],
            },
        },
    }

    # --- Phase 1: research tools (search + calculator) ---

    print("=== Phase 1: research tools ===\n")
    t.set_tools([search_tool, calculator_tool])

    sys_ci = t.system("You are a research assistant with access to tools.")
    q1 = t.user("What is the GDP of France?")
    a1 = t.assistant("Let me search for that. France's GDP is approximately $3.05 trillion.")

    print(f"Active tools: {len(t.get_tools())} (search, calculator)")
    print(f"Tools on system commit: {len(t.get_commit_tools(sys_ci.commit_hash))}")
    print(f"Tools on Q1: {len(t.get_commit_tools(q1.commit_hash))}")
    print(f"Tools on A1: {len(t.get_commit_tools(a1.commit_hash))}")

    # --- Phase 2: switch to coding tools (search + code) ---

    print("\n=== Phase 2: coding tools ===\n")
    t.set_tools([search_tool, code_tool])

    q2 = t.user("Write a Python function to calculate compound interest.")
    a2 = t.assistant(
        "Here's a compound interest function:\n"
        "def compound_interest(principal, rate, years):\n"
        "    return principal * (1 + rate) ** years"
    )

    print(f"Active tools: {len(t.get_tools())} (search, code)")
    print(f"Tools on Q2: {len(t.get_commit_tools(q2.commit_hash))}")
    print(f"Tools on A2: {len(t.get_commit_tools(a2.commit_hash))}")

    # --- Phase 3: clear tools ---

    print("\n=== Phase 3: no tools ===\n")
    t.set_tools(None)

    q3 = t.user("Summarize what we've discussed.")
    a3 = t.assistant("We looked up France's GDP and wrote a compound interest calculator.")

    print(f"Active tools: {t.get_tools()}")
    print(f"Tools on Q3: {len(t.get_commit_tools(q3.commit_hash))}")

    # --- Audit: reconstruct tool availability at any point ---

    print("\n=== Audit: tool provenance ===\n")

    for label, ci in [("System", sys_ci), ("Q1", q1), ("A1", a1),
                       ("Q2", q2), ("A2", a2), ("Q3", q3), ("A3", a3)]:
        tools = t.get_commit_tools(ci.commit_hash)
        names = [d["function"]["name"] for d in tools] if tools else ["(none)"]
        print(f"  {label} ({ci.commit_hash[:8]}): {', '.join(names)}")

    # --- Compiled context includes latest tools ---

    print("\n=== Compiled context ===\n")
    ctx = t.compile()
    print(f"Messages: {len(ctx.messages)}")
    print(f"Tools in compiled context: {len(ctx.tools)}")

    # to_openai_params() includes tools alongside messages
    params = ctx.to_openai_params()
    print(f"OpenAI params keys: {list(params.keys())}")

    t.close()


if __name__ == "__main__":
    main()
