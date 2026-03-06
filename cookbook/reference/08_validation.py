"""Validation reference: chat(validator=), compress(validator=), retain_match.

Covers: validator signature, chat/generate validation with hide_retries,
compress validation with retain_match, RetryExhaustedError.
All validation features require an LLM -- patterns shown with comments.
"""

import json

from tract import Priority, Tract
from tract.exceptions import RetryExhaustedError


def main():
    # =================================================================
    # 1. Validator signature: (str) -> (bool, str | None)
    # =================================================================

    def validate_person(text: str) -> tuple[bool, str | None]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return (False, f"Invalid JSON: {e}")
        if not isinstance(data, dict):
            return (False, f"Expected object, got {type(data).__name__}")
        missing = [k for k in ("name", "age", "hobbies") if k not in data]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, None)

    # Validators are pure Python -- testable offline
    assert validate_person('{"name":"Jo","age":30,"hobbies":["chess"]}') == (True, None)
    assert validate_person('not json')[0] is False
    assert validate_person('{"name":"Jo"}')[0] is False
    print("1. Validators are pure Python, testable offline")

    # =================================================================
    # 2. chat(validator=) -- validate LLM responses (requires LLM)
    # =================================================================
    # On failure: diagnosis committed as steering message, LLM retried.
    # hide_retries=True (default): SKIP-annotates failed attempts.

    # response = t.chat(
    #     "Give me 3 languages as JSON.",
    #     validator=validate_json_list,
    #     max_retries=3,        # retry up to 3 times
    #     hide_retries=True,    # default: SKIP failed attempts in compile()
    # )
    # response.text                  # validated output
    # response.commit_info.metadata  # has retry_attempts if retries occurred
    #
    # # Keep retry artifacts visible:
    # t.chat("...", validator=fn, hide_retries=False)
    #
    # # generate() has the same interface:
    # t.generate(validator=validate_person, max_retries=3)

    print("2. chat/generate(validator=, max_retries=, hide_retries=)")

    # =================================================================
    # 3. RetryExhaustedError -- all retries failed
    # =================================================================

    err = RetryExhaustedError(3, "too short", "some text")
    assert err.attempts == 3
    assert err.last_diagnosis == "too short"
    assert err.last_result == "some text"  # usable for fallback

    # try:
    #     t.chat("...", validator=impossible_fn, max_retries=2)
    # except RetryExhaustedError as e:
    #     fallback = e.last_result  # use last output as fallback

    print("3. RetryExhaustedError: attempts, last_diagnosis, last_result")

    # =================================================================
    # 4. compress(validator=) -- validate summaries (requires LLM)
    # =================================================================

    required = ["B-tree", "LSM", "write-ahead"]

    def validate_db(text: str) -> tuple[bool, str | None]:
        missing = [t for t in required if t.lower() not in text.lower()]
        if missing:
            return (False, f"Must mention: {missing}")
        if len(text.split()) < 20:
            return (False, "Too short")
        return (True, None)

    # result = t.compress(
    #     target_tokens=200,
    #     validator=validate_db,
    #     max_retries=3,
    # )

    print("4. compress(validator=, max_retries=)")

    # =================================================================
    # 5. retain_match= -- regex retention layer (requires LLM)
    # =================================================================
    # Two-layer safety: regex (hard) + validator (soft).

    # t.annotate(report.commit_hash, Priority.IMPORTANT,
    #     retain="Preserve dollar amounts and percentages",
    #     retain_match=[
    #         r"\$4[,.]?2[,.]?M",   # revenue pattern
    #         r"23\.5\s*%",          # margin pattern
    #     ],
    #     retain_match_mode="regex",
    # )
    #
    # result = t.compress(
    #     target_tokens=200,
    #     validator=validate_actionable,  # semantic check on top
    #     max_retries=4,
    # )

    print("5. retain_match: regex patterns for compression safety")

    # =================================================================
    # 6. instructions= + validator= -- guided + validated (requires LLM)
    # =================================================================

    def validate_security(text: str) -> tuple[bool, str | None]:
        categories = {"auth": ["jwt", "token"], "crypto": ["aes", "tls"],
                       "access": ["rbac", "role"]}
        missing = [c for c, kws in categories.items()
                   if not any(kw in text.lower() for kw in kws)]
        if missing:
            return (False, f"Missing: {missing}")
        return (True, None)

    # result = t.compress(
    #     target_tokens=150,
    #     instructions="Organize by: auth, encryption, access control.",
    #     validator=validate_security,
    #     max_retries=3,
    # )

    print("6. instructions + validator: guided compression with validation")
    print("\nDone.")


if __name__ == "__main__":
    main()
