"""Selective Compression

compress_tool_calls(name=) compresses only specific tool types,
leaving others untouched. Useful when grep results are verbose but
bash output is already concise.

Requires an LLM for compression.

Demonstrates: compress_tool_calls(name=) for selective compression,
              find_tool_results(name=), get_content(), token accounting,
              pprint(style="compact"), before/after visualization
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from tract import Tract

# Allow importing _helpers from the same directory when run as a script.
sys.path.insert(0, str(Path(__file__).parent))
from _helpers import build_agent_session  # noqa: E402

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


def part3_selective_compression():
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("PART 3 -- Manual: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 3 -- Manual: SELECTIVE COMPRESSION (compress_tool_calls(name=))")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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


# =============================================================================
# Part 2 -- Interactive: Selective Compression
# =============================================================================
# Show tool turns grouped by name. Let the human choose which tool types
# to compress interactively.

def part2_interactive():
    import click

    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("PART 2 -- Interactive: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Interactive: SELECTIVE COMPRESSION")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        build_agent_session(t)

        # Show tool turns grouped by name
        all_turns = t.find_tool_turns()
        tool_names = set()
        for turn in all_turns:
            tool_names.update(turn.tool_names)

        print(f"  Tool types found: {sorted(tool_names)}\n")
        for name in sorted(tool_names):
            name_turns = t.find_tool_turns(name=name)
            total = sum(turn.total_tokens for turn in name_turns)
            print(f"    {name}: {len(name_turns)} turn(s), {total} tokens")

        # Let human choose which to compress
        for name in sorted(tool_names):
            if click.confirm(f"\n  Compress all '{name}' tool turns?", default=False):
                result = t.compress_tool_calls(
                    name=name,
                    instructions=f"Summarize {name} output in 1-2 lines.",
                )
                print(f"    Compressed: {result.original_tokens} -> "
                      f"{result.compacted_tokens} tokens")

        ctx = t.compile()
        print(f"\n  Final: {ctx.token_count} tokens")


def main():
    part2_interactive()
    part3_selective_compression()


if __name__ == "__main__":
    main()
