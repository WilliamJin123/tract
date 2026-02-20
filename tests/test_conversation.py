"""Tests for the conversation layer: ChatResponse, chat(), generate(), Tract.open() LLM config."""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from tract import ChatResponse, LLMConfig, Tract
from tract.exceptions import TraceError
from tract.models.commit import CommitInfo, CommitOperation
from tract.protocols import TokenUsage


def _make_commit_info(**overrides):
    """Create a minimal valid CommitInfo for testing."""
    defaults = {
        "commit_hash": "abc123",
        "tract_id": "t1",
        "parent_hash": None,
        "content_hash": "blob123",
        "content_type": "dialogue",
        "operation": CommitOperation.APPEND,
        "message": "test",
        "token_count": 10,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return CommitInfo(**defaults)


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
        self.closed = False

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
        self.closed = True


# ---------------------------------------------------------------------------
# ChatResponse model tests
# ---------------------------------------------------------------------------

class TestChatResponse:
    """Tests for the ChatResponse frozen dataclass."""

    def test_create_with_all_fields(self):
        info = _make_commit_info()
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        resp = ChatResponse(
            text="Hello",
            usage=usage,
            commit_info=info,
            generation_config=LLMConfig(model="gpt-4o"),
        )
        assert resp.text == "Hello"
        assert resp.usage is usage
        assert resp.commit_info is info
        assert resp.generation_config == LLMConfig(model="gpt-4o")

    def test_usage_none_is_valid(self):
        info = _make_commit_info()
        resp = ChatResponse(
            text="Hello",
            usage=None,
            commit_info=info,
            generation_config=LLMConfig(),
        )
        assert resp.usage is None

    def test_frozen_immutability(self):
        info = _make_commit_info()
        resp = ChatResponse(
            text="Hello",
            usage=None,
            commit_info=info,
            generation_config=LLMConfig(),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            resp.text = "Changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Tract.open() LLM config tests
# ---------------------------------------------------------------------------

class TestOpenLLMConfig:
    """Tests for LLM auto-config via Tract.open() params."""

    def test_open_with_api_key(self, monkeypatch):
        mock_client = MockLLMClient()
        monkeypatch.setattr(
            "tract.llm.client.OpenAIClient",
            lambda **kwargs: mock_client,
        )
        t = Tract.open(api_key="test-key")
        assert hasattr(t, "_llm_client")
        assert t._owns_llm_client is True
        t.close()

    def test_open_with_api_key_and_model(self, monkeypatch):
        captured = {}

        def mock_init(**kwargs):
            captured.update(kwargs)
            return MockLLMClient()

        monkeypatch.setattr("tract.llm.client.OpenAIClient", mock_init)
        t = Tract.open(api_key="test-key", model="gpt-4o")
        assert t._default_config is not None
        assert t._default_config.model == "gpt-4o"
        assert captured.get("default_model") == "gpt-4o"
        t.close()

    def test_open_with_base_url(self, monkeypatch):
        captured = {}

        def mock_init(**kwargs):
            captured.update(kwargs)
            return MockLLMClient()

        monkeypatch.setattr("tract.llm.client.OpenAIClient", mock_init)
        t = Tract.open(api_key="test-key", base_url="http://localhost:8080/v1")
        assert captured.get("base_url") == "http://localhost:8080/v1"
        t.close()

    def test_open_without_api_key_no_llm(self):
        t = Tract.open()
        assert not hasattr(t, "_llm_client")
        assert t._owns_llm_client is False
        t.close()

    def test_open_default_model_is_gpt4o_mini(self, monkeypatch):
        captured = {}

        def mock_init(**kwargs):
            captured.update(kwargs)
            return MockLLMClient()

        monkeypatch.setattr("tract.llm.client.OpenAIClient", mock_init)
        t = Tract.open(api_key="test-key")
        assert captured.get("default_model") == "gpt-4o-mini"
        t.close()


# ---------------------------------------------------------------------------
# close() lifecycle tests
# ---------------------------------------------------------------------------

class TestCloseLLMLifecycle:
    """Tests for LLM client lifecycle on Tract.close()."""

    def test_internally_created_client_closed(self, monkeypatch):
        mock_client = MockLLMClient()
        monkeypatch.setattr(
            "tract.llm.client.OpenAIClient",
            lambda **kwargs: mock_client,
        )
        t = Tract.open(api_key="test-key")
        assert not mock_client.closed
        t.close()
        assert mock_client.closed

    def test_externally_provided_client_not_closed(self):
        mock_client = MockLLMClient()
        t = Tract.open()
        t.configure_llm(mock_client)
        assert not mock_client.closed
        t.close()
        assert not mock_client.closed

    def test_close_without_llm_client(self):
        """close() works fine when no LLM was configured."""
        t = Tract.open()
        t.close()  # No error


# ---------------------------------------------------------------------------
# generate() tests
# ---------------------------------------------------------------------------

class TestGenerate:
    """Tests for Tract.generate()."""

    def test_generate_happy_path(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["I am helpful!"])
        t.configure_llm(mock)
        t.system("You are helpful.")
        t.user("Hello")

        resp = t.generate()

        assert isinstance(resp, ChatResponse)
        assert resp.text == "I am helpful!"
        assert isinstance(resp.usage, TokenUsage)
        assert resp.usage.prompt_tokens == 10
        assert resp.usage.completion_tokens == 5
        assert isinstance(resp.commit_info, CommitInfo)
        assert resp.generation_config.model == "mock-model"
        t.close()

    def test_generate_creates_assistant_commit(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["response text"])
        t.configure_llm(mock)
        t.system("System")
        t.user("Question")

        resp = t.generate()

        # Check that assistant commit exists in log
        log = t.log(limit=10)
        assert len(log) == 3  # system + user + assistant
        assert "response text" in log[0].message or log[0].content_type == "dialogue"
        t.close()

    def test_generate_records_usage(self):
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.system("System")
        t.user("Question")

        t.generate()

        # Verify usage was recorded: compile should show api: token source
        compiled = t.compile()
        assert compiled.token_source.startswith("api:")
        t.close()

    def test_generate_with_explicit_params(self):
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.system("System")
        t.user("Question")

        resp = t.generate(model="gpt-4o", temperature=0.7, max_tokens=100)

        assert resp.generation_config.model == "gpt-4o"
        assert resp.generation_config.temperature == 0.7
        assert resp.generation_config.max_tokens == 100
        # Verify mock received the kwargs
        assert mock.last_kwargs.get("model") == "gpt-4o"
        assert mock.last_kwargs.get("temperature") == 0.7
        assert mock.last_kwargs.get("max_tokens") == 100
        t.close()

    def test_generate_with_message_and_metadata(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["answer"])
        t.configure_llm(mock)
        t.system("System")
        t.user("Question")

        resp = t.generate(message="custom msg", metadata={"source": "test"})

        assert resp.commit_info.message == "custom msg"
        assert resp.commit_info.metadata == {"source": "test"}
        t.close()

    def test_generate_without_llm_raises(self):
        t = Tract.open()
        t.system("System")
        t.user("Question")

        from tract.llm.errors import LLMConfigError

        with pytest.raises(LLMConfigError, match="No LLM client configured"):
            t.generate()
        t.close()

    def test_generate_inside_batch_raises(self):
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.system("System")

        with pytest.raises(TraceError, match="batch"):
            with t.batch():
                t.user("Question")
                t.generate()
        t.close()


# ---------------------------------------------------------------------------
# chat() tests
# ---------------------------------------------------------------------------

class TestChat:
    """Tests for Tract.chat()."""

    def test_chat_happy_path(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["Hi there!"])
        t.configure_llm(mock)
        t.system("You are helpful.")

        resp = t.chat("Hello")

        assert isinstance(resp, ChatResponse)
        assert resp.text == "Hi there!"
        assert resp.usage is not None
        assert resp.commit_info is not None
        # Should have system + user + assistant = 3 commits
        log = t.log(limit=10)
        assert len(log) == 3
        t.close()

    def test_chat_compiled_context_correct(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["response"])
        t.configure_llm(mock)
        t.system("Be helpful.")

        t.chat("What is 2+2?")

        compiled = t.compile()
        assert len(compiled.messages) == 3
        assert compiled.messages[0].role == "system"
        assert compiled.messages[1].role == "user"
        assert compiled.messages[2].role == "assistant"
        assert compiled.messages[1].content == "What is 2+2?"
        assert compiled.messages[2].content == "response"
        t.close()

    def test_multi_turn_chat(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["answer1", "answer2"])
        t.configure_llm(mock)
        t.system("System prompt")

        resp1 = t.chat("q1")
        resp2 = t.chat("q2")

        assert resp1.text == "answer1"
        assert resp2.text == "answer2"
        # 5 commits: system + user1 + asst1 + user2 + asst2
        log = t.log(limit=10)
        assert len(log) == 5
        t.close()

    def test_chat_with_name_param(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["Hello Alice"])
        t.configure_llm(mock)
        t.system("System")

        t.chat("Hi", name="Alice")

        compiled = t.compile()
        user_msg = compiled.messages[1]
        assert user_msg.name == "Alice"
        t.close()

    def test_chat_without_llm_raises(self):
        t = Tract.open()
        t.system("System")

        from tract.llm.errors import LLMConfigError

        with pytest.raises(LLMConfigError, match="No LLM client configured"):
            t.chat("Hello")
        t.close()

    def test_chat_inside_batch_raises(self):
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.system("System")

        with pytest.raises(TraceError, match="batch"):
            with t.batch():
                t.chat("Hello")
        t.close()


# ---------------------------------------------------------------------------
# _build_generation_config() tests
# ---------------------------------------------------------------------------

class TestBuildGenerationConfig:
    """Tests for generation config building from resolved kwargs + response."""

    def test_response_model_authoritative(self):
        """Response model takes precedence over resolved model."""
        t = Tract.open()
        response = {"model": "gpt-4o-2024-01-01"}

        config = t._build_generation_config(
            response, resolved={"model": "gpt-4o"}
        )

        assert config["model"] == "gpt-4o-2024-01-01"
        t.close()

    def test_resolved_model_used_when_no_response(self):
        """When response has no model, use resolved model."""
        t = Tract.open()
        response = {}

        config = t._build_generation_config(response, resolved={"model": "gpt-4o"})

        assert config["model"] == "gpt-4o"
        t.close()

    def test_full_resolved_captured(self):
        """All resolved fields are captured in generation_config."""
        t = Tract.open()
        response = {"model": "gpt-4o"}

        config = t._build_generation_config(
            response, resolved={"model": "gpt-4o", "temperature": 0.5, "max_tokens": 200, "top_p": 0.9}
        )

        assert config["model"] == "gpt-4o"
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 200
        assert config["top_p"] == 0.9
        t.close()

    def test_no_model_anywhere_omitted(self):
        """When no model info anywhere, config has no model key."""
        t = Tract.open()
        response = {}

        config = t._build_generation_config(response, resolved={})

        assert "model" not in config
        t.close()


# ---------------------------------------------------------------------------
# Integration: LLM messages forwarded correctly
# ---------------------------------------------------------------------------

class TestLLMMessageForwarding:
    """Verify that compile -> LLM receives the right messages."""

    def test_messages_sent_to_llm(self):
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.system("Be concise.")
        t.user("What is Python?")

        t.generate()

        assert mock.last_messages is not None
        assert len(mock.last_messages) == 2
        assert mock.last_messages[0]["role"] == "system"
        assert mock.last_messages[0]["content"] == "Be concise."
        assert mock.last_messages[1]["role"] == "user"
        assert mock.last_messages[1]["content"] == "What is Python?"
        t.close()

    def test_multi_turn_messages_accumulate(self):
        t = Tract.open()
        mock = MockLLMClient(responses=["r1", "r2"])
        t.configure_llm(mock)
        t.system("System")

        t.chat("q1")
        t.chat("q2")

        # After second chat, LLM should see: system + user1 + asst1 + user2
        assert mock.last_messages is not None
        assert len(mock.last_messages) == 4
        assert mock.last_messages[0]["role"] == "system"
        assert mock.last_messages[1]["role"] == "user"
        assert mock.last_messages[1]["content"] == "q1"
        assert mock.last_messages[2]["role"] == "assistant"
        assert mock.last_messages[2]["content"] == "r1"
        assert mock.last_messages[3]["role"] == "user"
        assert mock.last_messages[3]["content"] == "q2"
        t.close()
