"""Core Retry Primitive (retry_with_steering)

The retry protocol is a standalone loop: attempt -> validate -> steer -> retry.
It has no Tract dependency — it works with any callable. This file demonstrates
it with a live LLM call: the LLM produces JSON, a validator checks the schema,
a steering message corrects the LLM, and the loop retries.

Demonstrates: retry_with_steering(), RetryResult (value, attempts, history),
              RetryExhaustedError (attempts, last_diagnosis, last_result)
"""

import json
import os

from dotenv import load_dotenv

from tract import Tract
from tract.exceptions import RetryExhaustedError
from tract.retry import RetryResult, retry_with_steering

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: retry_with_steering with a live LLM
# =============================================================================
# The LLM must return a JSON object with specific fields. The validator checks
# structure, the steerer injects a correction, and the loop retries.
# Tract's commit chain gives the LLM memory of its own mistakes.

def part1_retry_with_llm():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a data generator. When asked to produce JSON, respond "
            "with ONLY the JSON object — no markdown fences, no explanation."
        )
        t.user(
            "Generate a JSON object for a fictional person. "
            "Include their full name, their age, and a list of hobbies. "
            "Return raw JSON only."
        )

        def validate_person_json(text: str) -> tuple[bool, str | None]:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return (False, f"Invalid JSON: {e}")
            if not isinstance(data, dict):
                return (False, f"Expected a JSON object, got {type(data).__name__}")
            missing = [k for k in ("name", "age", "hobbies") if k not in data]
            if missing:
                return (False, f"Missing required keys: {missing}")
            if not isinstance(data["hobbies"], list):
                return (False, "'hobbies' must be a list")
            if not isinstance(data["age"], int):
                return (False, "'age' must be an integer")
            return (True, None)

        # Wire retry_with_steering: five callbacks + config
        result: RetryResult[str] = retry_with_steering(
            attempt=lambda: t.generate().text,
            validate=validate_person_json,
            steer=lambda diag: t.user(
                f"Validation failed: {diag}\nFix the issue. Return raw JSON only."
            ),
            head_fn=lambda: t.head or "",
            reset_fn=lambda h: t.reset(h) if h else None,
            max_retries=3,
        )

        # RetryResult: value + metadata about the retry loop
        parsed = json.loads(result.value)
        print(f"RetryResult: attempts={result.attempts}, history={result.history}")
        print(f"Parsed: {json.dumps(parsed, indent=2)}")

        # The commit chain shows the full conversation including any retries
        print()
        t.compile(include_reasoning=True).pprint(style="chat")


# =============================================================================
# Part 2: RetryExhaustedError — when all attempts fail
# =============================================================================
# An impossible validator ensures exhaustion. RetryExhaustedError carries the
# attempt count, last diagnosis, and last result for recovery.

def part2_exhaustion():
    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Be concise.")
        t.user("Name a color.")

        try:
            retry_with_steering(
                attempt=lambda: t.generate().text,
                validate=lambda text: (False, "Must contain the word 'xylophone'"),
                steer=lambda diag: t.user(f"Try again. {diag}"),
                head_fn=lambda: t.head or "",
                reset_fn=lambda h: t.reset(h) if h else None,
                max_retries=2,
            )
        except RetryExhaustedError as e:
            print(f"RetryExhaustedError after {e.attempts} attempts")
            print(f"  last_diagnosis: {e.last_diagnosis}")
            print(f"  last_result:    {e.last_result!r:.80}")
            print("  (last_result is still usable for fallback logic)")


def main():
    print("=== Part 1: retry_with_steering + live LLM ===\n")
    part1_retry_with_llm()
    print(f"\n=== Part 2: RetryExhaustedError ===\n")
    part2_exhaustion()


if __name__ == "__main__":
    main()
