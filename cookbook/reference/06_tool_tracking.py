"""Tool tracking reference: commits, errors, summarization, reasoning, provenance.

Covers: tool_call/tool_result commits, is_error + tools.drop_failed_turns,
tool summarization, reasoning commits, query_by_config, tool provenance.
"""

from tract import Priority, Tract
from tract.formatting import pprint_log


def main() -> None:
    # =================================================================
    # 1. Tool result commits (no LLM needed)
    # =================================================================

    t = Tract.open()
    t.system("You are a file search agent.")
    t.user("Find Python files.")

    # Assistant requests a tool call (tool_calls in metadata)
    t.assistant("I'll search.", metadata={"tool_calls": [
        {"id": "call_001", "name": "list_directory", "arguments": {"path": "."}},
    ]})
    ci = t.tool_result("call_001", "list_directory", "main.py\nutils.py")
    t.assistant("Found 2 files.")
    print(f"tool commit: {ci.commit_hash[:8]}, {t.compile().token_count} tokens")

    ctx = t.compile()
    ctx.pprint(style="compact")

    t.close()

    # =================================================================
    # 2. Error handling: is_error + tools.drop_failed_turns
    # =================================================================

    t = Tract.open()
    t.system("You are a deployment agent.")
    t.user("Deploy to staging.")

    t.assistant("Checking...", metadata={"tool_calls": [
        {"id": "c1", "name": "health_check", "arguments": {}},
    ]})
    t.tool_result("c1", "health_check", "Server healthy.")

    t.assistant("Deploying...", metadata={"tool_calls": [
        {"id": "c2", "name": "deploy", "arguments": {"env": "staging"}},
    ]})
    t.tool_result("c2", "deploy", "Error: Connection refused", is_error=True)

    t.assistant("Retrying...", metadata={"tool_calls": [
        {"id": "c3", "name": "deploy", "arguments": {"env": "backup"}},
    ]})
    t.tool_result("c3", "deploy", "Deployed. Build #1847.")

    # Drop failed turns (SKIP-annotates error turn)
    drop = t.tools.drop_failed_turns()
    drop.pprint()

    print(f"\n  After dropping failed turns:")
    ctx = t.compile()
    ctx.pprint(style="compact")

    # Query tools: tools.find_turns, tools.find_results
    turns = t.tools.find_turns()
    print(f"{len(turns)} tool turns remaining")
    grep_turns = t.tools.find_turns(name="deploy")  # filter by name
    results = t.tools.find_results(name="health_check")
    t.close()

    # =================================================================
    # 3. Tool summarization (requires LLM for auto mode)
    # =================================================================
    # Manual: use tool_result(edit=) to trim results offline.

    t = Tract.open()
    t.system("You are an auditor.")
    t.assistant("Reading...", metadata={"tool_calls": [
        {"id": "c1", "name": "read_file", "arguments": {"path": "config.yaml"}},
    ]})
    orig = t.tool_result("c1", "read_file",
        "APP_NAME=svc\nDB_CONN=postgres://db:5432/app\nCACHE=3600")
    edited = t.tool_result("c1", "read_file",
        "DB_CONN=postgres://db:5432/app", edit=orig.commit_hash)
    print(f"\nedit: {orig.token_count} -> {edited.token_count} tokens")

    # Auto-summarization (requires LLM):
    # t.config.configure_tool_summarization(
    #     auto_threshold=50,                                # requires LLM
    #     default_instructions="Keep ONLY relevant facts.", # requires LLM
    #     context=True,                                      # requires LLM
    # )
    t.close()

    # =================================================================
    # 4. Reasoning commits
    # =================================================================

    t = Tract.open()
    t.system("You are a math tutor.")
    t.user("What is 17 * 23?")
    r = t.reasoning("17*20=340, 17*3=51, total=391", format="parsed")
    t.assistant("17 x 23 = 391")

    ctx = t.compile()  # reasoning excluded by default
    ctx_with = t.compile(include_reasoning=True)  # include it
    print(f"\nwithout reasoning: {ctx.commit_count}, with: {ctx_with.commit_count}")
    print(f"  Reasoning text: '{r.message}'")

    t.annotations.set(r.commit_hash, Priority.PINNED)  # force inclusion
    # format= and metadata= also accepted:
    # t.reasoning("...", format="think_tags", metadata={"source": "deepseek"})

    # LLM reasoning (requires LLM):
    # resp = t.llm.generate(reasoning_effort="high")
    # resp.reasoning         # extracted text
    # resp.reasoning_commit  # CommitInfo or None
    # t.llm.generate(reasoning=False)        # extract but don't commit
    # Tract.open(commit_reasoning=False) # global opt-out
    t.close()

    # =================================================================
    # 5. Config provenance: query_by_config (requires LLM commits)
    # =================================================================
    # Every assistant commit auto-captures generation_config.

    # t.llm.chat("Write opener.", temperature=0.7)
    # t.llm.chat("Wilder version.", temperature=1.5)
    # t.search.query_by_config("temperature", ">", 1.0)           # single field
    # t.search.query_by_config("temperature", "between", [0, 1])   # range
    # t.search.query_by_config("temperature", "in", [0.0, 1.5])    # set membership
    # t.search.query_by_config(conditions=[("model","=","gpt-4"),   # multi-field AND
    #                               ("temperature","=",0.0)])
    print("\nquery_by_config: requires commits with generation_config")

    # =================================================================
    # 6. Tool provenance: set_tools + get_commit_tools
    # =================================================================

    t = Tract.open()
    tools = [{"type": "function", "function": {
        "name": "python_eval", "description": "Eval Python",
        "parameters": {"type": "object", "properties": {
            "expr": {"type": "string"}}, "required": ["expr"]},
    }}]
    t.tools.set(tools)
    ci = t.system("Calculator mode.")
    recorded = t.tools.get_for_commit(ci.commit_hash)
    print(f"tool provenance: {len(recorded)} tools at {ci.commit_hash[:8]}")

    t.tools.set(None)  # clear tools mid-session
    ci2 = t.user("No tools now.")
    print(f"after clear: {len(t.tools.get_for_commit(ci2.commit_hash) or [])} tools")
    t.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
