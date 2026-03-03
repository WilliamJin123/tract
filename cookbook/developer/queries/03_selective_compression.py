"""Selective Compression

compress_tool_calls(name=) compresses only specific tool types,
leaving others untouched. Useful when grep results are verbose but
bash output is already concise.

Requires an LLM for compression.

Demonstrates: compress_tool_calls(name=) for selective compression,
              find_tool_results(name=), get_content(), token accounting,
              pprint(style="compact"), before/after visualization
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


def part3_selective_compression():
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 3 -- Manual: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 3 -- Manual: SELECTIVE COMPRESSION (compress_tool_calls(name=))")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        refs = build_agent_session(t)

        ctx_before = t.compile()
        print(f"  BEFORE: {ctx_before.token_count} tokens\n")

        # --- Compress only grep results ---
        # bash and read_file results stay untouched

        print("  compress_tool_calls(name='grep') — only grep turns:\n")
        grep_result = t.compress_tool_calls(
            name="grep",
            instructions="One line per file: 'filename: relevant finding'",
        )
        print(f"    original_tokens:  {grep_result.original_tokens}")
        print(f"    compacted_tokens: {grep_result.compacted_tokens}")
        print(f"    turn_count:       {grep_result.turn_count}")
        print(f"    tool_names:       {grep_result.tool_names}")

        ctx_after_grep = t.compile()
        print(f"\n  After grep compression: {ctx_after_grep.token_count} tokens")

        # --- Compress read_file results too ---

        print(f"\n  compress_tool_calls(name='read_file'):\n")
        read_result = t.compress_tool_calls(
            name="read_file",
            instructions="Summarize the file's purpose and key findings in 1-2 lines.",
        )
        print(f"    original_tokens:  {read_result.original_tokens}")
        print(f"    compacted_tokens: {read_result.compacted_tokens}")
        print(f"    turn_count:       {read_result.turn_count}")

        ctx_after_both = t.compile()
        total_saved = ctx_before.token_count - ctx_after_both.token_count
        print(f"\n  AFTER all selective compressions: {ctx_after_both.token_count} tokens")
        print(f"  Total saved: {total_saved} tokens\n")

        # --- bash results were untouched ---

        print("  bash results were never compressed (already concise):")
        bash_results = t.find_tool_results(name="bash")
        for r in bash_results:
            content = t.get_content(r.commit_hash)
            preview = content[:60] if isinstance(content, str) else str(content)[:60]
            print(f"    {r.token_count} tokens: {preview}...")

        # --- Final context ---

        print(f"\n  Final context:\n")
        ctx_after_both.pprint(style="compact")


def main():
    part3_selective_compression()


if __name__ == "__main__":
    main()
