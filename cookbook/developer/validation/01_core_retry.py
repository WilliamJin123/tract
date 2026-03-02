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
# Part 1 -- Manual: retry_with_steering with a live LLM
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
# Part 2 -- Manual: RetryExhaustedError — when all attempts fail
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


# =============================================================================
# Part 2b -- Interactive: Human is the Validator
# =============================================================================
# The human reviews each LLM response and decides whether it's acceptable.
# If not, the human provides a correction that steers the next attempt.

def part2b_interactive():
    import click

    print(f"\n{'=' * 60}")
    print("PART 2b -- Interactive: HUMAN IS THE VALIDATOR")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a data generator. Produce JSON objects as requested. "
            "Respond with ONLY the JSON — no markdown, no explanation."
        )
        t.user("Generate a JSON object for a fictional person with name, age, and hobbies.")

        max_attempts = 5
        for attempt in range(1, max_attempts + 1):
            print(f"\n  --- Attempt {attempt} ---")
            result = t.generate()
            print(f"  Response:\n{result.text}\n")

            if click.confirm("  Acceptable?", default=True):
                print("  Accepted!")
                break
            else:
                correction = click.prompt("  Correction")
                t.user(correction)
        else:
            print(f"  Gave up after {max_attempts} attempts.")

        print("\n  Final commit chain:")
        for entry in reversed(t.log()):
            print(f"    {entry}")


# =============================================================================
# Part 3 -- Agent: Validation via Toolkit + retry_with_steering
# =============================================================================
# Agents use retry_with_steering as the engine behind chat(validator=, max_retries=).
# The toolkit exposes this as a single-call pipeline: the agent sends a prompt,
# a validator function checks the output, and the loop auto-steers on failure.

def part3_agent():
    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: VALIDATION VIA TOOLKIT + RETRY")
    print("=" * 60)
    print()

    from tract.toolkit import ToolExecutor

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a data generator. When asked to produce JSON, respond "
            "with ONLY the JSON object — no markdown fences, no explanation."
        )
        executor = ToolExecutor(t)

        # Agent uses retry_with_steering directly — the same primitive
        # that powers chat(validator=, max_retries=).
        t.user("Generate a JSON object with keys: name (str), age (int), hobbies (list).")

        def validate(text: str) -> tuple[bool, str | None]:
            try:
                data = json.loads(text)
            except json.JSONDecodeError as e:
                return (False, f"Invalid JSON: {e}")
            missing = [k for k in ("name", "age", "hobbies") if k not in data]
            if missing:
                return (False, f"Missing keys: {missing}")
            return (True, None)

        from tract.retry import retry_with_steering

        result = retry_with_steering(
            attempt=lambda: t.generate().text,
            validate=validate,
            steer=lambda diag: t.user(f"Fix: {diag}. Return raw JSON only."),
            head_fn=lambda: t.head or "",
            reset_fn=lambda h: t.reset(h) if h else None,
            max_retries=3,
        )
        print(f"  Agent got valid JSON in {result.attempts} attempt(s)")
        print(f"  Result: {result.value[:80]}...")

    # Note: The agent-friendly API is chat(validator=fn, max_retries=3),
    # which wraps retry_with_steering internally. Use retry_with_steering
    # directly when you need custom attempt/steer/reset logic.


def main():
    print("=== Part 1 -- Manual: retry_with_steering + live LLM ===\n")
    part1_retry_with_llm()
    print(f"\n=== Part 2 -- Manual: RetryExhaustedError ===\n")
    part2_exhaustion()
    part2b_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
