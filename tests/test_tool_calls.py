"""Tests for typed tool call support."""

import json
import pytest
from tract.protocols import ToolCall, Message, CompiledContext, ChatResponse, TokenUsage
from tract.models.content import DialogueContent
from tract.models.config import LLMConfig


class TestToolCall:
    """Tests for ToolCall dataclass."""

    def test_from_openai_basic(self):
        """Parse a standard OpenAI tool call."""
        raw = {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "NYC", "unit": "celsius"}',
            },
        }
        tc = ToolCall.from_openai(raw)
        assert tc.id == "call_abc123"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "NYC", "unit": "celsius"}
        assert tc.type == "function"

    def test_from_openai_malformed_json(self):
        """Malformed JSON arguments get stored as _raw."""
        raw = {
            "id": "call_1",
            "type": "function",
            "function": {"name": "foo", "arguments": "not json{"},
        }
        tc = ToolCall.from_openai(raw)
        assert tc.arguments == {"_raw": "not json{"}

    def test_from_openai_dict_arguments(self):
        """Some providers send arguments as a dict, not JSON string."""
        raw = {
            "id": "call_2",
            "type": "function",
            "function": {"name": "bar", "arguments": {"x": 1}},
        }
        tc = ToolCall.from_openai(raw)
        assert tc.arguments == {"x": 1}

    def test_from_anthropic(self):
        """Parse an Anthropic tool_use content block."""
        block = {
            "type": "tool_use",
            "id": "toolu_abc",
            "name": "get_weather",
            "input": {"location": "SF"},
        }
        tc = ToolCall.from_anthropic(block)
        assert tc.id == "toolu_abc"
        assert tc.name == "get_weather"
        assert tc.arguments == {"location": "SF"}
        assert tc.type == "function"  # default

    def test_from_dict_round_trip(self):
        """to_dict() -> from_dict() round-trips correctly."""
        original = ToolCall(id="call_1", name="eval", arguments={"expr": "2+2"})
        d = original.to_dict()
        restored = ToolCall.from_dict(d)
        assert restored == original

    def test_to_openai(self):
        """to_openai() serializes arguments as JSON string."""
        tc = ToolCall(id="call_1", name="eval", arguments={"expr": "2+2"})
        result = tc.to_openai()
        assert result["id"] == "call_1"
        assert result["type"] == "function"
        assert result["function"]["name"] == "eval"
        assert result["function"]["arguments"] == '{"expr": "2+2"}'
        # Verify it's valid JSON
        assert json.loads(result["function"]["arguments"]) == {"expr": "2+2"}

    def test_to_dict(self):
        """to_dict() stores arguments as a dict (not JSON string)."""
        tc = ToolCall(id="call_1", name="eval", arguments={"x": 1}, type="function")
        d = tc.to_dict()
        assert d == {"id": "call_1", "name": "eval", "arguments": {"x": 1}, "type": "function"}
        assert isinstance(d["arguments"], dict)

    def test_frozen(self):
        """ToolCall is immutable."""
        tc = ToolCall(id="call_1", name="eval", arguments={})
        with pytest.raises(AttributeError):
            tc.id = "new"  # type: ignore[misc]


class TestMessageToolFields:
    """Tests for Message tool_calls and tool_call_id fields."""

    def test_message_defaults(self):
        """New fields default to None."""
        m = Message(role="assistant", content="hello")
        assert m.tool_calls is None
        assert m.tool_call_id is None

    def test_message_with_tool_calls(self):
        """Message can carry tool_calls."""
        tcs = [ToolCall(id="call_1", name="eval", arguments={"x": 1})]
        m = Message(role="assistant", content="", tool_calls=tcs)
        assert m.tool_calls == tcs
        assert len(m.tool_calls) == 1

    def test_message_with_tool_call_id(self):
        """Message can carry tool_call_id for tool results."""
        m = Message(role="tool", content="result", tool_call_id="call_1")
        assert m.tool_call_id == "call_1"


class TestCompiledContextToolDicts:
    """Tests for to_dicts() with tool data."""

    def test_to_dicts_with_tool_calls(self):
        """to_dicts() includes tool_calls in OpenAI format."""
        tcs = [ToolCall(id="call_1", name="eval", arguments={"expr": "2+2"})]
        messages = [
            Message(role="user", content="calculate 2+2"),
            Message(role="assistant", content="", tool_calls=tcs),
        ]
        ctx = CompiledContext(messages=messages, commit_count=2)
        dicts = ctx.to_dicts()

        assert len(dicts) == 2
        assert dicts[0] == {"role": "user", "content": "calculate 2+2"}

        assistant_dict = dicts[1]
        assert assistant_dict["role"] == "assistant"
        assert assistant_dict["content"] == ""
        assert "tool_calls" in assistant_dict
        assert len(assistant_dict["tool_calls"]) == 1
        tc_dict = assistant_dict["tool_calls"][0]
        assert tc_dict["id"] == "call_1"
        assert tc_dict["function"]["name"] == "eval"
        assert tc_dict["function"]["arguments"] == '{"expr": "2+2"}'

    def test_to_dicts_with_tool_call_id(self):
        """to_dicts() includes tool_call_id on tool result messages."""
        messages = [
            Message(role="tool", content="4", tool_call_id="call_1", name="eval"),
        ]
        ctx = CompiledContext(messages=messages, commit_count=1)
        dicts = ctx.to_dicts()

        assert len(dicts) == 1
        assert dicts[0]["role"] == "tool"
        assert dicts[0]["content"] == "4"
        assert dicts[0]["tool_call_id"] == "call_1"
        assert dicts[0]["name"] == "eval"

    def test_to_dicts_no_tool_data_unchanged(self):
        """to_dicts() output is unchanged for messages without tool data."""
        messages = [
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ]
        ctx = CompiledContext(messages=messages, commit_count=2)
        dicts = ctx.to_dicts()

        assert dicts == [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        # Verify no extra keys
        assert "tool_calls" not in dicts[0]
        assert "tool_call_id" not in dicts[0]

    def test_to_openai_params_with_tools_and_calls(self):
        """to_openai_params() includes both tool definitions and tool calls."""
        tcs = [ToolCall(id="call_1", name="eval", arguments={})]
        messages = [
            Message(role="assistant", content="", tool_calls=tcs),
        ]
        tool_defs = [{"type": "function", "function": {"name": "eval", "parameters": {}}}]
        ctx = CompiledContext(messages=messages, commit_count=1, tools=tool_defs)
        params = ctx.to_openai_params()

        assert "messages" in params
        assert "tools" in params
        assert params["messages"][0]["tool_calls"][0]["id"] == "call_1"
        assert params["tools"][0]["function"]["name"] == "eval"

    def test_to_anthropic_includes_tool_data(self):
        """to_anthropic() includes tool fields on non-system messages."""
        tcs = [ToolCall(id="call_1", name="eval", arguments={})]
        messages = [
            Message(role="system", content="You are helpful"),
            Message(role="assistant", content="", tool_calls=tcs),
            Message(role="tool", content="result", tool_call_id="call_1"),
        ]
        ctx = CompiledContext(messages=messages, commit_count=3)
        result = ctx.to_anthropic()

        assert result["system"] == "You are helpful"
        assert len(result["messages"]) == 2
        assert "tool_calls" in result["messages"][0]
        assert result["messages"][1]["tool_call_id"] == "call_1"


class TestDialogueContentToolRole:
    """Tests for DialogueContent with role='tool'."""

    def test_tool_role_valid(self):
        """DialogueContent accepts role='tool'."""
        dc = DialogueContent(role="tool", text="result text")
        assert dc.role == "tool"
        assert dc.text == "result text"
        assert dc.content_type == "dialogue"

    def test_tool_role_rejects_invalid(self):
        """DialogueContent still rejects invalid roles."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            DialogueContent(role="invalid", text="test")


class TestChatResponseToolFields:
    """Tests for ChatResponse with tool_calls and raw_response."""

    def test_defaults_none(self):
        """New fields default to None."""
        # Use a minimal CommitInfo-like object
        from unittest.mock import MagicMock
        mock_commit = MagicMock()
        r = ChatResponse(
            text="hello",
            usage=None,
            commit_info=mock_commit,
            generation_config=LLMConfig(),
        )
        assert r.tool_calls is None
        assert r.raw_response is None

    def test_with_tool_calls(self):
        """ChatResponse can carry tool_calls."""
        from unittest.mock import MagicMock
        mock_commit = MagicMock()
        tcs = [ToolCall(id="call_1", name="eval", arguments={"x": 1})]
        r = ChatResponse(
            text="",
            usage=None,
            commit_info=mock_commit,
            generation_config=LLMConfig(),
            tool_calls=tcs,
            raw_response={"choices": [{"message": {"tool_calls": []}}]},
        )
        assert r.tool_calls == tcs
        assert r.raw_response is not None
        assert len(r.tool_calls) == 1

    def test_str_returns_text(self):
        """str() still returns text even with tool_calls."""
        from unittest.mock import MagicMock
        mock_commit = MagicMock()
        r = ChatResponse(
            text="",
            usage=None,
            commit_info=mock_commit,
            generation_config=LLMConfig(),
            tool_calls=[ToolCall(id="c1", name="f", arguments={})],
        )
        assert str(r) == ""


class TestExtractContentToolCalls:
    """Tests for extract_content handling tool_calls responses."""

    def test_missing_content_key(self):
        """Returns empty string when content key is absent."""
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {"tool_calls": [{"id": "c1"}]}}]
        }
        assert OpenAIClient.extract_content(response) == ""

    def test_null_content(self):
        """Returns empty string when content is None."""
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {"content": None, "tool_calls": []}}]
        }
        assert OpenAIClient.extract_content(response) == ""

    def test_normal_content_unchanged(self):
        """Normal text content still works."""
        from tract.llm.client import OpenAIClient
        response = {
            "choices": [{"message": {"content": "Hello world"}}]
        }
        assert OpenAIClient.extract_content(response) == "Hello world"


class TestToolResultConvenience:
    """Tests for Tract.tool_result() method."""

    def test_tool_result_commits(self, tmp_path):
        """tool_result() creates a commit with correct metadata."""
        from tract import Tract

        with Tract.open(":memory:") as t:
            t.system("You are helpful.")
            t.user("test")
            t.assistant("I'll use a tool.")

            info = t.tool_result("call_123", "eval", "42")

            assert info is not None
            assert info.content_type == "dialogue"
            assert info.metadata is not None
            assert info.metadata["tool_call_id"] == "call_123"
            assert info.metadata["name"] == "eval"
            assert info.message == "tool result: eval"

    def test_tool_result_custom_message(self, tmp_path):
        """tool_result() accepts custom commit message."""
        from tract import Tract

        with Tract.open(":memory:") as t:
            t.system("sys")
            t.user("q")
            t.assistant("a")
            info = t.tool_result("call_1", "bash", "output", message="custom msg")
            assert info.message == "custom msg"

    def test_tool_result_compiles_correctly(self, tmp_path):
        """tool_result commits appear in compiled context with correct role."""
        from tract import Tract

        with Tract.open(":memory:") as t:
            t.system("sys")
            t.user("run eval")
            t.assistant("")  # tool-calling assistant message
            t.tool_result("call_1", "eval", "42")

            compiled = t.compile()
            dicts = compiled.to_dicts()

            # Find the tool message
            tool_msgs = [d for d in dicts if d["role"] == "tool"]
            assert len(tool_msgs) == 1
            assert tool_msgs[0]["content"] == "42"
            assert tool_msgs[0]["tool_call_id"] == "call_1"
            assert tool_msgs[0]["name"] == "eval"


class TestCompilerToolRoundTrip:
    """Tests for compiler reading tool metadata from commits."""

    def test_tool_calls_in_metadata_compile(self, tmp_path):
        """Tool calls stored in metadata appear in compiled Messages."""
        from tract import Tract

        with Tract.open(":memory:") as t:
            t.system("sys")
            t.user("calculate 2+2")

            # Simulate what _generate_once does: commit assistant with tool_calls in metadata
            tool_calls_data = [
                {"id": "call_1", "name": "eval", "arguments": {"expr": "2+2"}, "type": "function"},
            ]
            t.assistant("", metadata={"tool_calls": tool_calls_data})

            compiled = t.compile()
            dicts = compiled.to_dicts()

            assistant_dict = [d for d in dicts if d["role"] == "assistant"][-1]
            assert "tool_calls" in assistant_dict
            assert len(assistant_dict["tool_calls"]) == 1
            tc = assistant_dict["tool_calls"][0]
            assert tc["id"] == "call_1"
            assert tc["function"]["name"] == "eval"
            assert tc["function"]["arguments"] == '{"expr": "2+2"}'

    def test_full_tool_conversation_round_trip(self, tmp_path):
        """Full tool conversation: user -> assistant(tool_calls) -> tool(result) -> compile."""
        from tract import Tract

        with Tract.open(":memory:") as t:
            t.system("You have tools.")
            t.user("What's 2+2?")

            # Assistant requests tool call
            tool_calls_data = [
                {"id": "call_1", "name": "python_eval", "arguments": {"expression": "2+2"}, "type": "function"},
            ]
            t.assistant("", metadata={"tool_calls": tool_calls_data})

            # Tool result
            t.tool_result("call_1", "python_eval", "4")

            # Assistant final answer
            t.assistant("2+2 equals 4.")

            compiled = t.compile()
            dicts = compiled.to_dicts()

            assert len(dicts) == 5  # system, user, assistant(tool_calls), tool, assistant
            assert dicts[0]["role"] == "system"
            assert dicts[1]["role"] == "user"
            assert dicts[2]["role"] == "assistant"
            assert "tool_calls" in dicts[2]
            assert dicts[3]["role"] == "tool"
            assert dicts[3]["tool_call_id"] == "call_1"
            assert dicts[3]["content"] == "4"
            assert dicts[4]["role"] == "assistant"
            assert dicts[4]["content"] == "2+2 equals 4."
