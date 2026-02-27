"""Declarative config: configure_tool_summarization() with per-tool
instructions, auto_threshold, and default_instructions. The sugar layer
that replaces manual hook registration.
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def _safe(text: str) -> str:
    """Sanitize text for Windows console (cp1252) -- replace non-ASCII."""
    return text.encode("ascii", "replace").decode("ascii")


def declarative_config():
    print("\n" + "=" * 60)
    print("PART 3 -- configure_tool_summarization()")
    print("=" * 60)
    print()
    print("  Sugar over t.on('tool_result', handler).")
    print("  Set per-tool instructions and auto-thresholds declaratively.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a project analyst.")
        t.user("Give me an overview of the project structure.")

        # Configure: per-tool instructions + auto-threshold for everything else
        t.configure_tool_summarization(
            instructions={
                "list_directory": "Summarize to a bullet list of top-level directories only.",
                "read_file": "Keep the first 5 lines. Summarize the rest.",
            },
            auto_threshold=50,  # Anything over 50 tokens gets summarized
            default_instructions="Keep only the most relevant information.",
            include_context=True,
        )

        t.print_hooks()

        # --- Tool 1: list_directory (has specific instructions) ---
        t.assistant("Let me list the project.", metadata={
            "tool_calls": [{"id": "d1", "name": "list_directory",
                            "arguments": {"path": "."}}],
        })
        big_listing = "\n".join(
            ["src/", "tests/", "docs/", "config/", "README.md", "setup.py"]
            + [f"module_{i:02d}.py" for i in range(25)]
        )
        ci1 = t.tool_result("d1", "list_directory", big_listing)
        stored1 = t.get_content(ci1)
        print(f"\n  list_directory:")
        print(f"    Original: {len(big_listing)} chars")
        print(f"    Stored:   {_safe(stored1[:100])}")

        # --- Tool 2: read_file (has specific instructions) ---
        t.assistant("Reading the main module.", metadata={
            "tool_calls": [{"id": "d2", "name": "read_file",
                            "arguments": {"path": "src/main.py"}}],
        })
        big_file = "\n".join([f"# Line {i}: some code here" for i in range(50)])
        ci2 = t.tool_result("d2", "read_file", big_file)
        stored2 = t.get_content(ci2)
        print(f"\n  read_file:")
        print(f"    Original: {len(big_file)} chars")
        print(f"    Stored:   {_safe(stored2[:100])}")

        # --- Tool 3: search_code (no specific instructions, uses default) ---
        t.assistant("Searching for imports.", metadata={
            "tool_calls": [{"id": "d3", "name": "search_code",
                            "arguments": {"query": "import"}}],
        })
        big_search = "\n".join(
            [f"module_{i:02d}.py:1: import os" for i in range(30)]
        )
        ci3 = t.tool_result("d3", "search_code", big_search)
        stored3 = t.get_content(ci3)
        print(f"\n  search_code (default instructions, over threshold):")
        print(f"    Original: {len(big_search)} chars")
        print(f"    Stored:   {_safe(stored3[:100])}")

        # --- Tool 4: small result (under threshold, passes through) ---
        t.assistant("Quick check.", metadata={
            "tool_calls": [{"id": "d4", "name": "check_version",
                            "arguments": {}}],
        })
        ci4 = t.tool_result("d4", "check_version", "v2.4.1")
        stored4 = t.get_content(ci4)
        print(f"\n  check_version (under threshold, pass-through):")
        print(f"    Original: v2.4.1")
        print(f"    Stored:   {_safe(stored4)}")

        # Show all hook events for the session
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    declarative_config()
