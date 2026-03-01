"""Chat Validation (chat(validator=), purify, provenance_note)

chat() and generate() accept a validator= parameter that wraps the LLM call
in retry_with_steering automatically. On failure, a steering message is
committed as a user message so the LLM sees its own mistake in context.
purify=True resets HEAD after success and re-commits only the clean result.
provenance_note=True records the retry count.

Demonstrates: chat(validator=, max_retries=, purify=, provenance_note=,
              retry_prompt=), generate(validator=), RetryExhaustedError
"""

import json
import os

from dotenv import load_dotenv

from tract import Tract
from tract.exceptions import RetryExhaustedError

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "llama3.1-8b"


# =============================================================================
# Part 1: Basic chat(validator=) — steering in context
# =============================================================================
# On failure, chat() auto-commits a steering user message so the LLM sees
# its own mistake. The validator signature is (str) -> (bool, str | None).

def part1_basic_validation():
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
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
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

        # If retries happened, the log shows steering messages between attempts
        print("Commit chain (steering messages visible if retries occurred):")
        for entry in reversed(t.log()):
            print(f"  {entry}")


# =============================================================================
# Part 2: purify=True — clean history after retries
# =============================================================================
# Without purify, failed responses + steering messages stay in the chain.
# With purify=True, HEAD resets and only the clean result is re-committed.

def part2_purify():
    def must_contain_scala(text: str) -> tuple[bool, str | None]:
        if "scala" not in text.lower():
            return (False, "Response must mention scala")
        return (True, None)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")

        response = t.chat(
            "Name your favorite programming language and explain why in one sentence.",
            validator=must_contain_scala,
            max_retries=3,
            purify=True,
        )

        response.pprint()

        # With purify, chain is clean — system + user + assistant, no retries
        print("Purified chain (no retry artifacts):")
        for entry in reversed(t.log()):
            print(f"  {entry}")


# =============================================================================
# Part 3: provenance_note=True — record retry metadata
# =============================================================================
# Auto-commits "[retry] Succeeded on attempt 2/3. Previous failures: ..."
# after success. Survives in the log for auditing.

def part3_provenance():
    call_count = 0

    def flaky_validator(text: str) -> tuple[bool, str | None]:
        """Fails the first call, passes the second."""
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return (False, "Response too vague — include a specific year")
        return (True, None)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a history assistant. Be specific with dates.")

        response = t.chat(
            "When was the first computer invented?",
            validator=flaky_validator,
            max_retries=3,
            provenance_note=True,
        )

        response.pprint()

        # The "[retry]" provenance note appears as a user commit in the log
        print("Log (look for the [retry] provenance note):")
        for entry in reversed(t.log()):
            print(f"  {entry}")


# =============================================================================
# Part 4: retry_prompt= and generate(validator=)
# =============================================================================
# retry_prompt= customizes the steering message. generate(validator=) works
# for two-step flows where user() was already called.

def part4_custom_steering():
    def validate_haiku(text: str) -> tuple[bool, str | None]:
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        if len(lines) != 3:
            return (False, f"A haiku must have exactly 3 lines, got {len(lines)}")
        return (True, None)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a poet. Respond with ONLY the poem — no title, no explanation.")

        # Two-step: commit user message, then generate with validation
        t.user("Write a haiku about programming.")

        response = t.generate(
            validator=validate_haiku,
            max_retries=3,
            retry_prompt="Your poem did not meet the format requirements.",
        )

        response.pprint()


# =============================================================================
# Part 5: RetryExhaustedError from chat()
# =============================================================================
# When all retries fail, chat() raises RetryExhaustedError with last_result.

def part5_exhaustion():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")

        try:
            t.chat(
                "Say hello.",
                validator=lambda text: (False, "Must be exactly 1000 characters"),
                max_retries=2,
            )
        except RetryExhaustedError as e:
            print(f"RetryExhaustedError after {e.attempts} attempts")
            print(f"  last_diagnosis: {e.last_diagnosis}")
            print(f"  last_result:    {e.last_result!r:.80}")


def main():
    print("=== Part 1: chat(validator=) ===\n")
    part1_basic_validation()

    print(f"\n=== Part 2: purify=True ===\n")
    part2_purify()

    print(f"\n=== Part 3: provenance_note=True ===\n")
    part3_provenance()

    print(f"\n=== Part 4: generate(validator=) + retry_prompt= ===\n")
    part4_custom_steering()

    print(f"\n=== Part 5: RetryExhaustedError ===\n")
    part5_exhaustion()


if __name__ == "__main__":
    main()
