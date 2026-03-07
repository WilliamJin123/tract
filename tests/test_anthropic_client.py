"""Tests for the Anthropic Messages API client.

Tests message translation, response normalization, tool format translation,
reasoning extraction, and streaming (all with mocked httpx).
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from tract.llm.anthropic_client import (
    AnthropicClient,
    MessageDone,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallStart,
    UsageEvent,
    _to_content_blocks,
)
from tract.llm.errors import LLMConfigError


# -----------------------------------------------------------------------
# Construction
# -----------------------------------------------------------------------


class TestConstruction:
    def test_requires_api_key(self):
        with pytest.raises(LLMConfigError, match="No API key"):
            AnthropicClient(api_key="")

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("TRACT_ANTHROPIC_API_KEY", "sk-test")
        client = AnthropicClient()
        assert client._api_key == "sk-test"
        client.close()

    def test_custom_base_url(self):
        client = AnthropicClient(api_key="sk-test", base_url="https://custom.api/")
        assert client._base_url == "https://custom.api"
        client.close()

    def test_default_model(self):
        client = AnthropicClient(api_key="sk-test")
        assert "claude" in client._default_model
        client.close()


# -----------------------------------------------------------------------
# Message translation
# -----------------------------------------------------------------------


class TestMessageTranslation:
    """Tests for _translate_messages (OpenAI → Anthropic)."""

    def setup_method(self):
        self.client = AnthropicClient(api_key="sk-test")

    def teardown_method(self):
        self.client.close()

    def test_system_messages_extracted(self):
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hello"},
        ]
        system, msgs = self.client._translate_messages(messages)
        assert system == "Be helpful"
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"

    def test_multiple_system_messages_joined(self):
        messages = [
            {"role": "system", "content": "Rule 1"},
            {"role": "system", "content": "Rule 2"},
            {"role": "user", "content": "Hi"},
        ]
        system, msgs = self.client._translate_messages(messages)
        assert system == "Rule 1\n\nRule 2"

    def test_tool_results_converted(self):
        messages = [
            {"role": "user", "content": "Call a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }],
            },
            {
                "role": "tool",
                "content": "result data",
                "tool_call_id": "call_1",
            },
        ]
        system, msgs = self.client._translate_messages(messages)
        assert system is None
        assert len(msgs) == 3
        # Tool result becomes user message with tool_result block
        tool_msg = msgs[2]
        assert tool_msg["role"] == "user"
        assert isinstance(tool_msg["content"], list)
        assert tool_msg["content"][0]["type"] == "tool_result"
        assert tool_msg["content"][0]["tool_use_id"] == "call_1"

    def test_assistant_tool_calls_to_content_blocks(self):
        messages = [
            {"role": "user", "content": "Call a tool"},
            {
                "role": "assistant",
                "content": "I'll search for that",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "test"}'},
                }],
            },
        ]
        _system, msgs = self.client._translate_messages(messages)
        asst = msgs[1]
        assert asst["role"] == "assistant"
        assert isinstance(asst["content"], list)
        assert asst["content"][0] == {"type": "text", "text": "I'll search for that"}
        assert asst["content"][1]["type"] == "tool_use"
        assert asst["content"][1]["name"] == "search"
        assert asst["content"][1]["input"] == {"q": "test"}

    def test_consecutive_same_role_merged(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "World"},
        ]
        _system, msgs = self.client._translate_messages(messages)
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        # Content merged into blocks
        blocks = msgs[0]["content"]
        assert isinstance(blocks, list)
        assert blocks[0] == {"type": "text", "text": "Hello"}
        assert blocks[1] == {"type": "text", "text": "World"}

    def test_tool_result_after_user_merged(self):
        """Tool results (→user) after user messages merge correctly."""
        messages = [
            {"role": "user", "content": "Call a tool"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }],
            },
            {
                "role": "tool",
                "content": "result 1",
                "tool_call_id": "call_1",
            },
        ]
        _system, msgs = self.client._translate_messages(messages)
        # user, assistant, user(tool_result) -- 3 messages, alternating
        assert len(msgs) == 3


# -----------------------------------------------------------------------
# Tool definition translation
# -----------------------------------------------------------------------


class TestToolTranslation:
    def test_openai_to_anthropic_format(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                },
            },
        }]
        result = AnthropicClient._translate_tools(tools)
        assert len(result) == 1
        assert result[0]["name"] == "search"
        assert result[0]["description"] == "Search the web"
        assert "input_schema" in result[0]
        assert result[0]["input_schema"]["type"] == "object"

    def test_already_anthropic_format(self):
        tools = [{
            "name": "search",
            "description": "Search",
            "input_schema": {"type": "object"},
        }]
        result = AnthropicClient._translate_tools(tools)
        assert result[0]["name"] == "search"
        assert result[0]["input_schema"] == {"type": "object"}


# -----------------------------------------------------------------------
# Response normalization
# -----------------------------------------------------------------------


class TestResponseNormalization:
    def setup_method(self):
        self.client = AnthropicClient(api_key="sk-test")

    def teardown_method(self):
        self.client.close()

    def test_text_response(self):
        anthropic_resp = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello there!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        result = self.client._normalize_response(anthropic_resp)

        assert result["choices"][0]["message"]["content"] == "Hello there!"
        assert result["choices"][0]["message"]["role"] == "assistant"
        assert result["choices"][0]["finish_reason"] == "stop"
        assert result["usage"]["prompt_tokens"] == 10
        assert result["usage"]["completion_tokens"] == 5
        assert result["usage"]["total_tokens"] == 15
        assert result["model"] == "claude-sonnet-4-20250514"

    def test_tool_use_response(self):
        anthropic_resp = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "text", "text": "I'll search for that"},
                {
                    "type": "tool_use",
                    "id": "toolu_abc",
                    "name": "search",
                    "input": {"q": "test"},
                },
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }
        result = self.client._normalize_response(anthropic_resp)

        msg = result["choices"][0]["message"]
        assert msg["content"] == "I'll search for that"
        assert len(msg["tool_calls"]) == 1
        tc = msg["tool_calls"][0]
        assert tc["id"] == "toolu_abc"
        assert tc["function"]["name"] == "search"
        assert json.loads(tc["function"]["arguments"]) == {"q": "test"}
        assert result["choices"][0]["finish_reason"] == "tool_calls"

    def test_thinking_preserved_for_extraction(self):
        anthropic_resp = {
            "id": "msg_789",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [
                {"type": "thinking", "thinking": "Let me reason about this..."},
                {"type": "text", "text": "The answer is 42."},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 30, "output_tokens": 50},
        }
        result = self.client._normalize_response(anthropic_resp)

        # Text content excludes thinking
        assert result["choices"][0]["message"]["content"] == "The answer is 42."
        # Raw blocks preserved for reasoning extraction
        assert "_anthropic_content" in result
        assert result["_anthropic_content"][0]["type"] == "thinking"

    def test_stop_reason_mapping(self):
        for stop_reason, expected in [
            ("end_turn", "stop"),
            ("tool_use", "tool_calls"),
            ("max_tokens", "length"),
            ("stop_sequence", "stop"),
        ]:
            resp = {
                "content": [{"type": "text", "text": "x"}],
                "stop_reason": stop_reason,
                "usage": {},
            }
            result = self.client._normalize_response(resp)
            assert result["choices"][0]["finish_reason"] == expected


# -----------------------------------------------------------------------
# Extract methods
# -----------------------------------------------------------------------


class TestExtractMethods:
    def test_extract_content(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        assert AnthropicClient.extract_content(resp) == "Hello"

    def test_extract_content_none(self):
        resp = {"choices": [{"message": {"content": None}}]}
        assert AnthropicClient.extract_content(resp) == ""

    def test_extract_usage(self):
        resp = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        assert AnthropicClient.extract_usage(resp) == resp["usage"]

    def test_extract_tool_calls(self):
        resp = {
            "choices": [{
                "message": {
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "search",
                            "arguments": '{"q": "test"}',
                        },
                    }],
                },
            }],
        }
        tcs = AnthropicClient.extract_tool_calls(resp)
        assert len(tcs) == 1
        assert tcs[0]["name"] == "search"
        assert tcs[0]["arguments"] == {"q": "test"}

    def test_extract_tool_calls_empty(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        assert AnthropicClient.extract_tool_calls(resp) == []

    def test_extract_reasoning_from_anthropic_blocks(self):
        resp = {
            "_anthropic_content": [
                {"type": "thinking", "thinking": "I should reason..."},
                {"type": "text", "text": "answer"},
            ],
        }
        result = AnthropicClient.extract_reasoning(resp)
        assert result is not None
        text, fmt = result
        assert text == "I should reason..."
        assert fmt == "anthropic"

    def test_extract_reasoning_none(self):
        resp = {"choices": [{"message": {"content": "Hello"}}]}
        assert AnthropicClient.extract_reasoning(resp) is None


# -----------------------------------------------------------------------
# Content block helpers
# -----------------------------------------------------------------------


class TestContentBlocks:
    def test_string_to_blocks(self):
        assert _to_content_blocks("hello") == [{"type": "text", "text": "hello"}]

    def test_empty_string_to_blocks(self):
        assert _to_content_blocks("") == []

    def test_list_passthrough(self):
        blocks = [{"type": "text", "text": "a"}]
        assert _to_content_blocks(blocks) == blocks


# -----------------------------------------------------------------------
# Tract.open() auto-detection
# -----------------------------------------------------------------------


class TestAutoDetection:
    """Tests that Tract.open() picks the right client based on model/base_url."""

    def test_detect_from_model_name(self):
        from tract.tract import _detect_provider
        assert _detect_provider(None, "claude-sonnet-4-20250514") == "anthropic"
        assert _detect_provider(None, "claude-3-opus") == "anthropic"
        assert _detect_provider(None, "gpt-4o") == "openai"
        assert _detect_provider(None, None) == "openai"

    def test_detect_from_base_url(self):
        from tract.tract import _detect_provider
        assert _detect_provider("https://api.anthropic.com", None) == "anthropic"
        assert _detect_provider("https://api.openai.com/v1", None) == "openai"

    def test_open_with_provider_anthropic(self):
        """Tract.open(provider='anthropic') creates an AnthropicClient."""
        from tract.llm.anthropic_client import AnthropicClient

        t = Tract.open(
            api_key="sk-test",
            provider="anthropic",
        )
        assert isinstance(t._llm_client, AnthropicClient)
        t.close()

    def test_open_with_claude_model(self):
        """Tract.open(model='claude-...') auto-detects Anthropic."""
        from tract.llm.anthropic_client import AnthropicClient

        t = Tract.open(
            api_key="sk-test",
            model="claude-sonnet-4-20250514",
        )
        assert isinstance(t._llm_client, AnthropicClient)
        t.close()

    def test_open_default_is_openai(self):
        """Tract.open() without provider hint uses OpenAI."""
        from tract.llm.client import OpenAIClient

        t = Tract.open(api_key="sk-test")
        assert isinstance(t._llm_client, OpenAIClient)
        t.close()


# -----------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------


class TestImports:
    def test_import_from_tract(self):
        from tract import AnthropicClient, StreamEvent, TextDelta, ThinkingDelta, MessageDone
        assert AnthropicClient is not None
        assert issubclass(TextDelta, StreamEvent)
        assert issubclass(ThinkingDelta, StreamEvent)
        assert issubclass(MessageDone, StreamEvent)

    def test_import_from_llm(self):
        from tract.llm import AnthropicClient, TextDelta, ToolCallStart
        assert AnthropicClient is not None


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

# Bring Tract into scope for auto-detection tests
from tract import Tract
