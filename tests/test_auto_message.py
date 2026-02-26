"""Tests for LLM-based auto commit message generation.

Tests the _auto_message instance method on Tract, the _fallback_message
module-level function, prompt construction, and config wiring.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tract import Tract
from tract.models.config import LLMConfig, OperationClients, OperationConfigs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_client(response_text: str = "Ask about first computer") -> MagicMock:
    """Create a mock LLM client that returns a fixed response."""
    client = MagicMock()
    client.chat.return_value = {
        "choices": [{"message": {"content": response_text}}],
    }
    # Remove auto-created extract_content so _extract_content falls through
    # to OpenAI-format extraction from the response dict.
    del client.extract_content
    del client.extract_usage
    del client.extract_reasoning
    return client


# ---------------------------------------------------------------------------
# Fallback tests (no LLM path)
# ---------------------------------------------------------------------------

class TestFallbackMessage:
    """_fallback_message (module-level) produces truncated previews."""

    def test_no_llm_uses_truncation(self):
        """Without an LLM client, commit messages are truncated."""
        with Tract.open() as t:
            info = t.system("A" * 600)
            assert len(info.message) <= 500
            assert info.message.endswith("...")

    def test_auto_summarize_default_is_off(self):
        """auto_summarize defaults to False; LLM client alone doesn't trigger it."""
        client = _make_mock_client()
        with Tract.open() as t:
            t.configure_llm(client)
            info = t.system("Hello world")
            client.chat.assert_not_called()
            assert info.message == "Hello world"

    def test_batch_mode_uses_fallback(self):
        """Inside batch(), auto-summarize is skipped."""
        client = _make_mock_client()
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            with t.batch():
                info = t.system("Hello world")
            client.chat.assert_not_called()
            assert info.message == "Hello world"

    def test_llm_error_falls_back(self):
        """If LLM call fails, falls back to truncation."""
        client = MagicMock()
        client.chat.side_effect = RuntimeError("API error")
        del client.extract_content
        del client.extract_usage
        del client.extract_reasoning
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            info = t.system("Hello world")
            assert info.message == "Hello world"

    def test_auto_summarize_true_without_llm(self):
        """auto_summarize=True but no LLM client → fallback."""
        with Tract.open(auto_summarize=True) as t:
            info = t.system("Hello world")
            assert info.message == "Hello world"

    def test_empty_text_returns_content_type(self):
        """Empty text produces content_type as the message."""
        from tract.tract import _fallback_message
        assert _fallback_message("instruction", "") == "instruction"

    def test_short_text_not_truncated(self):
        """Short text is returned as-is."""
        from tract.tract import _fallback_message
        assert _fallback_message("dialogue", "Hello") == "Hello"


# ---------------------------------------------------------------------------
# LLM summarization tests
# ---------------------------------------------------------------------------

class TestLLMSummarization:
    """When LLM is available and auto_summarize enabled, uses LLM for messages."""

    def test_auto_summarize_true(self):
        """auto_summarize=True uses the default LLM client."""
        client = _make_mock_client("Ask about first computer")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            info = t.system("When was the first computer invented?")
            assert info.message == "Ask about first computer"
            client.chat.assert_called_once()

    def test_auto_summarize_model_string(self):
        """auto_summarize='model-name' sets operation config with that model."""
        client = _make_mock_client("Summary from cheap model")
        with Tract.open(auto_summarize="llama3.1-8b") as t:
            t.configure_llm(client)
            info = t.system("Some content")
            assert info.message == "Summary from cheap model"
            call_kwargs = client.chat.call_args[1]
            assert call_kwargs.get("model") == "llama3.1-8b"
            assert call_kwargs.get("temperature") == 0.0

    def test_auto_summarize_llmconfig(self):
        """auto_summarize=LLMConfig(...) gives full control."""
        client = _make_mock_client("Custom config summary")
        cfg = LLMConfig(model="gpt-4o-mini", temperature=0.1, max_tokens=50)
        with Tract.open(auto_summarize=cfg) as t:
            t.configure_llm(client)
            info = t.system("Some content")
            assert info.message == "Custom config summary"
            call_kwargs = client.chat.call_args[1]
            assert call_kwargs.get("model") == "gpt-4o-mini"
            # LLMConfig temperature wins over _auto_message's default 0.0
            # because operation config (level 3) < sugar param (level 1)
            # and _auto_message passes temperature=0.0 as sugar
            assert call_kwargs.get("temperature") == 0.0

    def test_long_output_capped(self):
        """LLM output >100 chars is capped."""
        long_summary = "A" * 150
        client = _make_mock_client(long_summary)
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            info = t.system("Some content")
            assert len(info.message) <= 100
            assert info.message.endswith("...")

    def test_empty_llm_output_falls_back(self):
        """Empty LLM output triggers fallback."""
        client = _make_mock_client("")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            info = t.system("Hello world")
            assert info.message == "Hello world"

    def test_llm_called_with_system_prompt(self):
        """Verify system prompt is passed to the LLM."""
        from tract.prompts.commit_message import COMMIT_MESSAGE_SYSTEM

        client = _make_mock_client("Summary")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            t.system("Test content")
            call_args = client.chat.call_args
            messages = call_args[0][0]
            assert messages[0]["role"] == "system"
            assert messages[0]["content"] == COMMIT_MESSAGE_SYSTEM

    def test_reasoning_uses_llm_summary(self):
        """reasoning() also uses LLM auto-summarize."""
        client = _make_mock_client("Analyze code structure")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            info = t.reasoning("Let me think about the code structure...")
            assert info.message == "Analyze code structure"

    def test_explicit_message_skips_llm(self):
        """Providing message= explicitly skips LLM summarization."""
        client = _make_mock_client("LLM summary")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            info = t.system("Some content", message="My custom message")
            assert info.message == "My custom message"
            client.chat.assert_not_called()


# ---------------------------------------------------------------------------
# Config wiring tests
# ---------------------------------------------------------------------------

class TestConfigWiring:
    """Verify summarize config integrates with operation config system."""

    def test_configure_operations_summarize(self):
        """configure_operations accepts 'summarize' key."""
        with Tract.open() as t:
            t.configure_operations(summarize=LLMConfig(model="gpt-4o-mini"))
            assert t.operation_configs.summarize == LLMConfig(model="gpt-4o-mini")

    def test_configure_operations_dataclass(self):
        """OperationConfigs accepts summarize field."""
        with Tract.open() as t:
            t.configure_operations(OperationConfigs(
                summarize=LLMConfig(model="gpt-4o-mini", temperature=0.0),
            ))
            assert t.operation_configs.summarize is not None
            assert t.operation_configs.summarize.model == "gpt-4o-mini"

    def test_configure_clients_summarize(self):
        """configure_clients accepts 'summarize' key."""
        mock_client = MagicMock()
        with Tract.open() as t:
            t.configure_clients(summarize=mock_client)
            assert t.operation_clients.summarize is mock_client

    def test_configure_clients_dataclass(self):
        """OperationClients accepts summarize field."""
        mock_client = MagicMock()
        with Tract.open() as t:
            t.configure_clients(OperationClients(summarize=mock_client))
            assert t.operation_clients.summarize is mock_client

    def test_per_operation_client_used(self):
        """Summarize-specific client is used over default."""
        default_client = _make_mock_client("Default summary")
        summarize_client = _make_mock_client("Summarize-specific")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(default_client)
            t.configure_clients(summarize=summarize_client)
            info = t.system("Some content")
            assert info.message == "Summarize-specific"
            summarize_client.chat.assert_called_once()
            default_client.chat.assert_not_called()

    def test_auto_summarize_string_sets_operation_config(self):
        """auto_summarize='model' populates operation_configs.summarize."""
        with Tract.open(auto_summarize="llama3.1-8b") as t:
            assert t.operation_configs.summarize is not None
            assert t.operation_configs.summarize.model == "llama3.1-8b"
            assert t.operation_configs.summarize.temperature == 0.0

    def test_auto_summarize_llmconfig_sets_operation_config(self):
        """auto_summarize=LLMConfig populates operation_configs.summarize."""
        cfg = LLMConfig(model="gpt-4o-mini", temperature=0.2)
        with Tract.open(auto_summarize=cfg) as t:
            assert t.operation_configs.summarize is cfg

    def test_auto_summarize_true_no_operation_config(self):
        """auto_summarize=True doesn't set operation config (uses default)."""
        with Tract.open(auto_summarize=True) as t:
            assert t.operation_configs.summarize is None

    def test_post_hoc_configure_operations_with_auto_summarize(self):
        """configure_operations(summarize=) works alongside auto_summarize=True."""
        client = _make_mock_client("Summary")
        with Tract.open(auto_summarize=True) as t:
            t.configure_llm(client)
            t.configure_operations(summarize=LLMConfig(model="gpt-4o-mini"))
            t.system("Test")
            call_kwargs = client.chat.call_args[1]
            assert call_kwargs.get("model") == "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Prompt module tests
# ---------------------------------------------------------------------------

class TestPromptModule:
    """Tests for the commit_message prompt module."""

    def test_system_prompt_exists(self):
        from tract.prompts.commit_message import COMMIT_MESSAGE_SYSTEM
        assert isinstance(COMMIT_MESSAGE_SYSTEM, str)
        assert len(COMMIT_MESSAGE_SYSTEM) > 0

    def test_builder_includes_content_type(self):
        from tract.prompts.commit_message import build_commit_message_prompt
        prompt = build_commit_message_prompt("dialogue", "Hello world")
        assert "dialogue" in prompt

    def test_builder_truncates_long_input(self):
        from tract.prompts.commit_message import build_commit_message_prompt, _MAX_INPUT_CHARS
        long_text = "A" * 5000
        prompt = build_commit_message_prompt("instruction", long_text)
        assert len(prompt) < 5000
        assert "..." in prompt

    def test_builder_short_input_not_truncated(self):
        from tract.prompts.commit_message import build_commit_message_prompt
        prompt = build_commit_message_prompt("instruction", "Be helpful")
        assert "Be helpful" in prompt
        assert prompt.count("...") == 0
