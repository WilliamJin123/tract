"""Chat Validation (chat(validator=), hide_retries)

chat() and generate() accept a validator= parameter that routes the LLM
response through a PendingGeneration hook. On failure, a steering message
is committed as a user message so the LLM sees its own mistake in context,
then a new generation is attempted. hide_retries=True (the default)
SKIP-annotates failed attempts and steering messages after approval, so
they remain in the commit chain for audit but are excluded from compile().
Retry metadata (attempt count, history) is auto-attached to the final
commit when retries occur.

Demonstrates: chat(validator=, max_retries=, hide_retries=),
              auto-attached retry metadata

Parts:
  1. Manual validator (pure Python, no LLM in the validator)
  2. chat(validator=) with steering in context
  3. hide_retries=False — keep retry artifacts visible in compile()
  4. Auto-attached retry metadata
"""

import json
import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.small


# =============================================================================
# Part 1: Manual Validator (pure Python, no LLM needed for the validator)
# =============================================================================
# The simplest usage: a deterministic Python function validates the LLM output.
# No LLM in the validator itself — just json.loads() and type checks.

def part1_manual_validator():
    print("=" * 60)
    print("Part 1: MANUAL VALIDATOR (pure Python)")
    print("=" * 60)
    print()

    def json_validator(text: str) -> tuple[bool, str | None]:
        """Validate that the response is valid JSON with required fields."""
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return (False, f"Not valid JSON: {e}")
        if not isinstance(data, dict):
            return (False, f"Expected a JSON object, got {type(data).__name__}")
        if "result" not in data:
            return (False, "Missing required key: 'result'")
        return (True, None)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a data assistant. Always respond with a JSON object "
            "containing a 'result' key. No markdown, no explanation."
        )

        response = t.chat(
            "What is 2 + 2? Return as JSON.",
            validator=json_validator,
            max_retries=3,
        )

        print(f"  Validated response: {response.text}")
        parsed = json.loads(response.text)
        print(f"  Parsed result: {parsed['result']}")

        print("\n  Commit chain:")
        for entry in reversed(t.log()):
            print(f"    {entry}")


# =============================================================================
# Part 2: Basic chat(validator=) — steering in context
# =============================================================================
# On failure, chat() auto-commits a steering user message so the LLM sees
# its own mistake. The validator signature is (str) -> (bool, str | None).

def part2_basic_validation():
    def validate_json_list(text: str) -> tuple[bool, str | None]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return (False, f"Not valid JSON: {e}")
        if not isinstance(data, list):
            return (False, f"Expected a JSON list, got {type(data).__name__}")
        if len(data) != 3:
            return (False, f"Expected exactly 3 items, got {len(data)}")
        return (True, None)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a data assistant. When asked for a list, respond with "
            "ONLY a JSON array — no markdown, no explanation."
        )

        response = t.chat(
            "Give me a JSON list of 3 programming languages.",
            validator=validate_json_list,
            max_retries=3,
        )

        response.pprint()

        # hide_retries=True (default): failed attempts + steering are SKIP-annotated
        # after approval. They're in the log but excluded from compile().
        print("Commit chain (SKIP-annotated intermediates if retries occurred):")
        for entry in reversed(t.log()):
            print(f"  {entry}")


# =============================================================================
# Part 3: hide_retries=False — keep retry artifacts visible in compile()
# =============================================================================
# hide_retries=True (default) SKIP-annotates failed attempts + steering so
# compile() excludes them. With hide_retries=False, everything stays visible
# in compiled context — the LLM sees the full retry history.

def part3_hide_retries():
    def must_contain_scala(text: str) -> tuple[bool, str | None]:
        if "scala" not in text.lower():
            return (False, "Response must mention scala")
        return (True, None)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        response = t.chat(
            "Name your favorite programming language and explain why in one sentence.",
            validator=must_contain_scala,
            max_retries=3,
            hide_retries=False,  # Keep retry artifacts visible in compile()
        )

        # With hide_retries=False, steering messages appear in compiled context
        print("Full chain (retry artifacts visible in compiled context):")
        compiled = t.compile()
        compiled.pprint()
        for msg in compiled.messages:
            preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
            print(f"  [{msg.role}] {preview}")


# =============================================================================
# Part 4: Auto-attached retry metadata
# =============================================================================
# When retries occur, the hook layer automatically attaches retry_attempts
# and retry_history to the successful commit's metadata. No explicit
# parameter needed — just use validator= and the metadata appears.
# (The failed responses themselves are committed as SKIP'd commits in the
# chain, retrievable via t.log() — metadata only stores the short
# diagnosis strings, not the full failed outputs.)

def part4_retry_metadata():
    call_count = 0

    def flaky_validator(text: str) -> tuple[bool, str | None]:
        """Fails the first call, passes the second."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (False, "Response too vague — include a specific year")
        return (True, None)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a history assistant. Be specific with dates.")

        response = t.chat(
            "When was the first computer invented?",
            validator=flaky_validator,
            max_retries=3,
        )

        response.pprint()

        # Retry metadata is auto-attached to the commit by the hook layer
        commit = response.commit_info
        print(f"Commit metadata: {commit.metadata}")

        print("\nCommit chain (SKIP'd intermediates visible in log):")
        for entry in reversed(t.log()):
            print(f"  {entry}")


def main():
    print("=== Part 1: Manual validator (pure Python) ===\n")
    part1_manual_validator()

    print(f"\n=== Part 2: chat(validator=) ===\n")
    part2_basic_validation()

    print(f"\n=== Part 3: hide_retries=False ===\n")
    part3_hide_retries()

    print(f"\n=== Part 4: Auto-attached retry metadata ===\n")
    part4_retry_metadata()


if __name__ == "__main__":
    main()
