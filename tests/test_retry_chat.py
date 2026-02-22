"""Tests for retry protocol integration with chat() and generate().

Verifies backward compatibility (no validator = unchanged behavior),
retry flow (steering commits, successful retry), exhaustion, custom
retry prompts, purification, and provenance notes.
"""

from __future__ import annotations

import pytest

from tract import ChatResponse, Tract
from tract.exceptions import RetryExhaustedError


# ---------------------------------------------------------------------------
# MockLLMClient -- predictable LLM responses for testing
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Minimal mock conforming to the LLMClient protocol."""

    def __init__(self, responses=None, model="mock-model"):
        self.responses = responses or ["Mock response"]
        self._call_count = 0
        self.last_messages = None
        self.last_kwargs: dict = {}
        self._model = model

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model": kwargs.get("model", self._model),
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChatNoValidator:
    """Backward compatibility: no validator = unchanged behavior."""

    def test_chat_no_validator_unchanged(self):
        """chat() without validator produces normal ChatResponse."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Hello there!"])
        t.configure_llm(mock)
        t.system("You are helpful.")

        resp = t.chat("Hi")

        assert isinstance(resp, ChatResponse)
        assert resp.text == "Hello there!"
        assert mock._call_count == 1

    def test_generate_no_validator_unchanged(self):
        """generate() without validator produces normal ChatResponse."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Generated text"])
        t.configure_llm(mock)
        t.system("Be concise.")
        t.user("What is 2+2?")

        resp = t.generate()

        assert isinstance(resp, ChatResponse)
        assert resp.text == "Generated text"
        assert mock._call_count == 1


class TestChatValidatorPassesFirstTry:
    """Validator passes on first attempt -- no retries."""

    def test_chat_validator_passes_first_try(self):
        """chat() with validator that passes immediately returns normal response."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Valid answer"])
        t.configure_llm(mock)
        t.system("Answer questions.")

        def always_valid(text):
            return (True, None)

        resp = t.chat("Tell me something", validator=always_valid)

        assert isinstance(resp, ChatResponse)
        assert resp.text == "Valid answer"
        assert mock._call_count == 1

    def test_generate_validator_passes_first_try(self):
        """generate() with validator that passes immediately returns normally."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Correct output"])
        t.configure_llm(mock)
        t.system("System prompt.")
        t.user("User input.")

        resp = t.generate(validator=lambda text: (True, None))

        assert resp.text == "Correct output"
        assert mock._call_count == 1


class TestChatValidatorFailsThenPasses:
    """Validator fails, steering committed, then succeeds."""

    def test_chat_validator_fails_then_passes(self):
        """chat() retries when validator fails, succeeds on second attempt."""
        t = Tract.open()
        # First response fails validation, second passes
        mock = MockLLMClient(responses=["bad answer", "good answer"])
        t.configure_llm(mock)
        t.system("You are helpful.")

        call_count = 0

        def validate_response(text):
            nonlocal call_count
            call_count += 1
            if "good" in text:
                return (True, None)
            return (False, "response must contain 'good'")

        resp = t.chat("Please say something good", validator=validate_response)

        assert resp.text == "good answer"
        assert mock._call_count == 2
        assert call_count == 2

    def test_generate_validator_fails_then_passes(self):
        """generate() retries on validation failure."""
        t = Tract.open()
        mock = MockLLMClient(responses=["wrong", "right"])
        t.configure_llm(mock)
        t.system("System.")
        t.user("Question.")

        def check(text):
            return (True, None) if text == "right" else (False, "not right")

        resp = t.generate(validator=check)

        assert resp.text == "right"
        assert mock._call_count == 2


class TestChatAllRetriesFail:
    """All retries exhausted -- RetryExhaustedError raised."""

    def test_chat_all_retries_fail(self):
        """chat() raises RetryExhaustedError when all attempts fail."""
        t = Tract.open()
        mock = MockLLMClient(responses=["bad"] * 5)
        t.configure_llm(mock)
        t.system("System prompt.")

        def always_fail(text):
            return (False, "always invalid")

        with pytest.raises(RetryExhaustedError) as exc_info:
            t.chat("Test", validator=always_fail, max_retries=3)

        err = exc_info.value
        assert err.attempts == 3
        assert "always invalid" in err.last_diagnosis

    def test_generate_all_retries_fail(self):
        """generate() raises RetryExhaustedError when all attempts fail."""
        t = Tract.open()
        mock = MockLLMClient(responses=["nope"] * 5)
        t.configure_llm(mock)
        t.system("System.")
        t.user("Input.")

        with pytest.raises(RetryExhaustedError):
            t.generate(validator=lambda t: (False, "bad"), max_retries=2)


class TestChatCustomRetryPrompt:
    """Custom retry_prompt is used in steering messages."""

    def test_chat_custom_retry_prompt(self):
        """Custom retry_prompt appears in the steering user message."""
        t = Tract.open()
        mock = MockLLMClient(responses=["bad", "good"])
        t.configure_llm(mock)
        t.system("System.")

        call_count = 0

        def validate(text):
            nonlocal call_count
            call_count += 1
            return (True, None) if call_count > 1 else (False, "too short")

        resp = t.chat(
            "Hello",
            validator=validate,
            retry_prompt="Please provide a longer response.",
        )

        assert resp.text == "good"
        # Verify that the steering message was sent containing the custom prompt
        # The mock client should have received messages containing the custom text
        # on the second call
        last_messages = mock.last_messages
        # The steering user message should be in the compiled context
        steering_found = any(
            "Please provide a longer response." in m.get("content", "")
            for m in last_messages
        )
        assert steering_found, (
            f"Custom retry prompt not found in messages: {last_messages}"
        )


class TestChatProvenanceNote:
    """Provenance note committed after successful retry."""

    def test_chat_provenance_note(self):
        """provenance_note=True commits a meta user message after retries."""
        t = Tract.open()
        mock = MockLLMClient(responses=["bad", "good"])
        t.configure_llm(mock)
        t.system("System.")

        call_count = 0

        def validate(text):
            nonlocal call_count
            call_count += 1
            return (True, None) if call_count > 1 else (False, "format error")

        resp = t.chat(
            "Hello",
            validator=validate,
            provenance_note=True,
        )

        assert resp.text == "good"

        # Check that a provenance note was committed as a user message
        # The compiled context should include a "[retry]" message
        compiled = t.compile()
        messages_text = [m.content for m in compiled.messages]
        retry_note_found = any("[retry]" in msg for msg in messages_text)
        assert retry_note_found, (
            f"Provenance note not found in messages: {messages_text}"
        )


class TestChatPurify:
    """purify=True resets history and re-commits clean result."""

    def test_chat_purify_cleans_history(self):
        """purify=True removes steering artifacts from commit history."""
        t = Tract.open()
        mock = MockLLMClient(responses=["bad", "good"])
        t.configure_llm(mock)
        t.system("System.")

        call_count = 0

        def validate(text):
            nonlocal call_count
            call_count += 1
            return (True, None) if call_count > 1 else (False, "wrong")

        resp = t.chat(
            "Hello",
            validator=validate,
            purify=True,
        )

        assert resp.text == "good"

        # After purification, the compiled context should NOT contain
        # steering messages (the "wrong" diagnosis user message)
        compiled = t.compile()
        messages_text = [m.content for m in compiled.messages]
        diagnosis_found = any("wrong" in msg and "Diagnosis:" in msg for msg in messages_text)
        assert not diagnosis_found, (
            f"Steering artifacts should be purified but found: {messages_text}"
        )


class TestChatMaxRetries:
    """max_retries parameter controls attempt count."""

    def test_chat_max_retries_respected(self):
        """max_retries=1 means only one attempt, no retries."""
        t = Tract.open()
        mock = MockLLMClient(responses=["bad"])
        t.configure_llm(mock)
        t.system("System.")

        with pytest.raises(RetryExhaustedError) as exc_info:
            t.chat("Test", validator=lambda t: (False, "bad"), max_retries=1)

        assert exc_info.value.attempts == 1
        assert mock._call_count == 1
