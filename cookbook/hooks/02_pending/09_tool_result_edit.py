"""Edit and summarize tool results: pending.edit_result() for manual
replacement, pending.summarize() for LLM-driven summarization.
original_content preservation for provenance.
"""

import sys
from pathlib import Path

from tract import Tract
from tract.hooks.tool_result import PendingToolResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm  

MODEL_ID = llm.large


def edit_and_summarize() -> None:
    print("\n" + "=" * 60)
    print("PART 2 -- Edit and Summarize")
    print("=" * 60)

    # --- Without hooks (baseline) ---
    # Without review=True, tool_result() commits content as-is (tier 3).
    # With review=True, edit_result() and summarize() let you transform
    # the content before committing (tier 1).

    # --- edit_result(): manual replacement ---
    print("\n  edit_result(): replace content before commit")

    with Tract.open() as t:
        t.system("You are a security auditor.")
        t.assistant("Reading config.", metadata={
            "tool_calls": [{"id": "tc4", "name": "read_config",
                            "arguments": {"path": "app.env"}}],
        })

        pending: PendingToolResult = t.tool_result(
            "tc4", "read_config",
            "APP_NAME=myapp\nSECRET_KEY=abc123\nDB_URL=postgres://user:pass@host/db",
            review=True,
        )

        print("  Before edit:")
        pending.pprint()

        # Redact secrets
        redacted = pending.content.replace("abc123", "***").replace("user:pass", "***:***")
        pending.edit_result(redacted)

        print("\n  After edit_result() -- original_content preserved for provenance:")
        pending.pprint()

        pending.approve()
        print("  Committed with redacted content.")

        print("\n  Context after committing redacted result:")
        t.compile().pprint(style="chat")

    # --- summarize(): LLM-driven summarization ---
    print(f"\n  summarize(): LLM compresses verbose tool output")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a code auditor. Answer precisely.")
        t.user("What database is this project using?")
        t.assistant("Let me check the config files.", metadata={
            "tool_calls": [{"id": "tc5", "name": "read_file",
                            "arguments": {"path": "docker-compose.yaml"}}],
        })

        # Verbose tool output -- much more than we need
        verbose_config = "\n".join([
            "version: '3.8'",
            "services:",
            "  web:",
            "    build: .",
            "    ports:",
            "      - '8080:8080'",
            "    environment:",
            "      - DEBUG=true",
            "      - LOG_LEVEL=info",
            "  db:",
            "    image: postgres:15",
            "    environment:",
            "      - POSTGRES_DB=myapp",
            "      - POSTGRES_USER=admin",
            "      - POSTGRES_PASSWORD=secret",
            "    ports:",
            "      - '5432:5432'",
            "    volumes:",
            "      - pgdata:/var/lib/postgresql/data",
            "  redis:",
            "    image: redis:7",
            "    ports:",
            "      - '6379:6379'",
            "volumes:",
            "  pgdata:",
        ])

        pending: PendingToolResult = t.tool_result(
            "tc5", "read_file", verbose_config, review=True,
        )

        print("  Before summarize:")
        pending.pprint()

        # Summarize with instructions
        pending.summarize(
            instructions="Extract only database-related configuration.",
            include_context=True,  # LLM sees the user's question
        )

        print("\n  After summarize():")
        pending.pprint()

        pending.approve()

        print("\n  Context after committing summarized result:")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    edit_and_summarize()
