"""Query API

Find, filter, and inspect tool-related commits across a conversation.
Use find_tool_turns(), find_tool_results(), and find_tool_calls() to
answer questions like "which tools were called?" and "how many tokens
did grep results consume?"

Also demonstrates token_checkpoints() for API-calibrated token tracking
that persists across sessions.

No LLM required — all query operations work offline.

Demonstrates: find_tool_results(name=, after=), find_tool_calls(name=),
              find_tool_turns(name=), ToolTurn (all_hashes, result_hashes,
              total_tokens, tool_names), token budget analysis,
              record_usage(), token_checkpoints()
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

# Allow importing _helpers from the same directory when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import build_agent_session  

MODEL_ID = llm.large


def part1_query_api():
    print("=" * 60)
    print("PART 1 -- Manual: QUERY API")
    print("=" * 60)
    print()

    t = Tract.open()
    build_agent_session(t)

    # ----- Section A: Full commit log -----
    print("  Section A: Full commit log\n")
    log = t.log(limit=10000)
    print(f"  {'type':12s} {'tokens':>6s}  {'hash':8s}  message")
    print(f"  {'-'*12} {'-'*6}  {'-'*8}  {'-'*30}")
    for ci in log:
        ctype = ci.content_type
        tokens = ci.token_count
        h = ci.commit_hash[:8]
        msg = (ci.message or "")[:40]
        print(f"  {ctype:12s} {tokens:6d}  {h}  {msg}")
    print()

    # ----- Section B: Tool turns -----
    print("  Section B: Tool turns\n")

    all_turns = t.find_tool_turns()
    print(f"  {len(all_turns)} tool turn(s) total\n")

    total_tokens = 0
    for i, turn in enumerate(all_turns):
        names = ", ".join(turn.tool_names)
        print(f"    Turn {i+1}: {names}")
        print(f"      results:      {len(turn.results)}")
        print(f"      total_tokens: {turn.total_tokens}")
        print(f"      all_hashes:   {len(turn.all_hashes)} commits")
        total_tokens += turn.total_tokens

        # Show full content for each result
        for result in turn.results:
            content = t.get_content(result)
            name = result.metadata.get("name", "?")
            print(f"      [{name}] {result.token_count} tokens:")
            # Show first 3 lines of content
            lines = (content or "").split("\n")
            for line in lines[:3]:
                print(f"        {line}")
            if len(lines) > 3:
                print(f"        ... ({len(lines) - 3} more lines)")

    print(f"\n  Total tool tokens: {total_tokens}")

    # Filter by tool name
    print(f"\n  find_tool_turns(name='grep') — filter by tool:\n")
    grep_turns = t.find_tool_turns(name="grep")
    print(f"  {len(grep_turns)} grep turn(s)")
    for turn in grep_turns:
        print(f"    {turn.total_tokens} tokens, {len(turn.results)} result(s)")

    # find_tool_results(name=) for specific tools
    print(f"\n  find_tool_results(name='read_file'):\n")
    read_results = t.find_tool_results(name="read_file")
    for r in read_results:
        print(f"    {r.token_count} tokens  {r.commit_hash[:8]}")

    # Token budget analysis
    ctx = t.compile()
    tool_pct = (total_tokens / ctx.token_count * 100) if ctx.token_count else 0
    print(f"\n  Token budget analysis:")
    print(f"    Total context:  {ctx.token_count} tokens")
    print(f"    Tool content:   {total_tokens} tokens ({tool_pct:.0f}%)")
    print(f"    Non-tool:       {ctx.token_count - total_tokens} tokens")

    # ----- Section C: Token checkpoints -----
    print()
    print("  Section C: Token checkpoints\n")

    # compile() gives the tiktoken estimate
    print(f"  Tiktoken estimate: {ctx.token_count} tokens ({ctx.token_source})")

    # Simulate an API response with actual usage
    t.record_usage({"prompt_tokens": ctx.token_count + 15, "completion_tokens": 42})
    print(f"  Recorded API usage: prompt={ctx.token_count + 15}, completion=42")

    # Query persisted checkpoints
    checkpoints = t.token_checkpoints()
    print(f"\n  token_checkpoints() — {len(checkpoints)} API checkpoint(s):\n")
    for cp in checkpoints:
        print(f"    record_id:    {cp.record_id[:12]}...")
        print(f"    head_hash:    {cp.head_hash[:8]}")
        print(f"    token_count:  {cp.token_count}")
        print(f"    token_source: {cp.token_source}")
        print(f"    commit_count: {cp.commit_count}")
        effectives = t.compile_record_commits(cp.record_id)
        print(f"    effectives:   {len(effectives)} commits")
        print()

    t.close()


def main():
    part1_query_api()


if __name__ == "__main__":
    main()
