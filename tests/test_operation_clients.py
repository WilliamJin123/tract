"""Tests for per-operation LLM client support.

Covers: OperationClients dataclass, configure_clients(), _resolve_llm_client(),
per-operation wiring for chat/merge/compress, and lifecycle (close).
"""

import dataclasses

import pytest

from tract import (
    LLMConfig,
    OperationClients,
    Tract,
)


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


class MockLLMClient:
    """Minimal mock conforming to the LLMClient protocol."""

    def __init__(self, name="default", response_text="Mock response"):
        self.name = name
        self._response_text = response_text
        self._call_count = 0
        self.last_messages = None
        self.last_kwargs: dict = {}
        self.closed = False

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        self._call_count += 1
        return {
            "choices": [{"message": {"content": self._response_text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model": kwargs.get("model", "mock-model"),
        }

    def close(self):
        self.closed = True

    @property
    def call_count(self):
        return self._call_count


# ---------------------------------------------------------------------------
# OperationClients dataclass tests
# ---------------------------------------------------------------------------


class TestOperationClients:
    """Tests for the OperationClients frozen dataclass."""

    def test_default_all_none(self):
        oc = OperationClients()
        assert oc.chat is None
        assert oc.merge is None
        assert oc.compress is None
        assert oc.orchestrate is None

    def test_set_fields(self):
        client = MockLLMClient()
        oc = OperationClients(chat=client, compress=client)
        assert oc.chat is client
        assert oc.merge is None
        assert oc.compress is client

    def test_frozen(self):
        oc = OperationClients()
        with pytest.raises(dataclasses.FrozenInstanceError):
            oc.chat = MockLLMClient()  # type: ignore[misc]

    def test_replace(self):
        client_a = MockLLMClient(name="a")
        client_b = MockLLMClient(name="b")
        oc = OperationClients(chat=client_a)
        oc2 = dataclasses.replace(oc, compress=client_b)
        assert oc2.chat is client_a
        assert oc2.compress is client_b


# ---------------------------------------------------------------------------
# configure_clients() tests
# ---------------------------------------------------------------------------


class TestConfigureClients:
    """Tests for Tract.configure_clients()."""

    def test_keyword_args(self):
        t = Tract.open()
        client = MockLLMClient()
        t.configure_clients(chat=client)
        assert t.operation_clients.chat is client
        assert t.operation_clients.merge is None
        t.close()

    def test_positional_operation_clients(self):
        t = Tract.open()
        client = MockLLMClient()
        oc = OperationClients(merge=client)
        t.configure_clients(oc)
        assert t.operation_clients.merge is client
        t.close()

    def test_mixed_raises_type_error(self):
        t = Tract.open()
        with pytest.raises(TypeError, match="Cannot mix"):
            t.configure_clients(OperationClients(), chat=MockLLMClient())
        t.close()

    def test_invalid_operation_raises_value_error(self):
        t = Tract.open()
        with pytest.raises(ValueError, match="Unknown operation 'bogus'"):
            t.configure_clients(bogus=MockLLMClient())
        t.close()

    def test_wrong_positional_type_raises_type_error(self):
        t = Tract.open()
        with pytest.raises(TypeError, match="Expected OperationClients"):
            t.configure_clients("not a client")  # type: ignore[arg-type]
        t.close()

    def test_merge_with_existing(self):
        """configure_clients() merges, doesn't replace."""
        t = Tract.open()
        client_a = MockLLMClient(name="a")
        client_b = MockLLMClient(name="b")
        t.configure_clients(chat=client_a)
        t.configure_clients(compress=client_b)
        assert t.operation_clients.chat is client_a  # preserved
        assert t.operation_clients.compress is client_b  # added
        t.close()

    def test_overwrite_existing(self):
        t = Tract.open()
        client_a = MockLLMClient(name="a")
        client_b = MockLLMClient(name="b")
        t.configure_clients(chat=client_a)
        t.configure_clients(chat=client_b)
        assert t.operation_clients.chat is client_b
        t.close()


# ---------------------------------------------------------------------------
# _resolve_llm_client() tests
# ---------------------------------------------------------------------------


class TestResolveLLMClient:
    """Tests for Tract._resolve_llm_client()."""

    def test_returns_operation_client_when_set(self):
        t = Tract.open()
        default = MockLLMClient(name="default")
        chat_client = MockLLMClient(name="chat")
        t.configure_llm(default)
        t.configure_clients(chat=chat_client)
        assert t._resolve_llm_client("chat") is chat_client
        t.close()

    def test_falls_back_to_default(self):
        t = Tract.open()
        default = MockLLMClient(name="default")
        t.configure_llm(default)
        assert t._resolve_llm_client("chat") is default
        assert t._resolve_llm_client("merge") is default
        t.close()

    def test_raises_when_no_client_at_all(self):
        t = Tract.open()
        with pytest.raises(AttributeError):
            t._resolve_llm_client("chat")
        t.close()

    def test_operation_client_without_default(self):
        """Per-operation client works even without a tract-level default."""
        t = Tract.open()
        chat_client = MockLLMClient(name="chat")
        t.configure_clients(chat=chat_client)
        assert t._resolve_llm_client("chat") is chat_client
        # Other operations still raise
        with pytest.raises(AttributeError):
            t._resolve_llm_client("merge")
        t.close()


# ---------------------------------------------------------------------------
# _has_llm_client() tests
# ---------------------------------------------------------------------------


class TestHasLLMClient:
    """Tests for Tract._has_llm_client()."""

    def test_no_client_at_all(self):
        t = Tract.open()
        assert t._has_llm_client() is False
        assert t._has_llm_client("chat") is False
        t.close()

    def test_default_only(self):
        t = Tract.open()
        t.configure_llm(MockLLMClient())
        assert t._has_llm_client() is True
        assert t._has_llm_client("chat") is True
        t.close()

    def test_operation_only(self):
        t = Tract.open()
        t.configure_clients(chat=MockLLMClient())
        assert t._has_llm_client() is False  # no default
        assert t._has_llm_client("chat") is True
        assert t._has_llm_client("merge") is False
        t.close()


# ---------------------------------------------------------------------------
# Per-operation wiring: chat/generate
# ---------------------------------------------------------------------------


class TestChatUsesOperationClient:
    """chat() and generate() use the per-operation client when set."""

    def test_chat_uses_operation_client(self):
        t = Tract.open()
        default = MockLLMClient(name="default")
        chat_client = MockLLMClient(name="chat", response_text="Chat response")
        t.configure_llm(default)
        t.configure_clients(chat=chat_client)

        t.system("Hello")
        response = t.chat("Question")

        assert response.text == "Chat response"
        assert chat_client.call_count == 1
        assert default.call_count == 0
        t.close()

    def test_generate_uses_operation_client(self):
        t = Tract.open()
        default = MockLLMClient(name="default")
        chat_client = MockLLMClient(name="chat", response_text="Gen response")
        t.configure_llm(default)
        t.configure_clients(chat=chat_client)

        t.system("Hello")
        t.user("Question")
        response = t.generate()

        assert response.text == "Gen response"
        assert chat_client.call_count == 1
        assert default.call_count == 0
        t.close()

    def test_chat_falls_back_to_default(self):
        t = Tract.open()
        default = MockLLMClient(name="default", response_text="Default response")
        t.configure_llm(default)

        t.system("Hello")
        response = t.chat("Question")

        assert response.text == "Default response"
        assert default.call_count == 1
        t.close()

    def test_chat_with_only_operation_client(self):
        """chat() works with only an operation client, no default."""
        t = Tract.open()
        chat_client = MockLLMClient(name="chat", response_text="Op only")
        t.configure_clients(chat=chat_client)

        t.system("Hello")
        response = t.chat("Question")

        assert response.text == "Op only"
        assert chat_client.call_count == 1
        t.close()


# ---------------------------------------------------------------------------
# Per-operation wiring: compress
# ---------------------------------------------------------------------------


class TestCompressUsesOperationClient:
    """compress() uses the per-operation client when set."""

    def test_compress_uses_operation_client(self):
        t = Tract.open()
        default = MockLLMClient(name="default")
        compress_client = MockLLMClient(
            name="compress", response_text="Compressed summary"
        )
        t.configure_llm(default)
        t.configure_clients(compress=compress_client)

        # Build up some commits to compress
        t.system("System prompt")
        t.user("Question 1")
        t.assistant("Answer 1")
        t.user("Question 2")
        t.assistant("Answer 2")

        log = t.log(limit=10)
        first_hash = log[-1].commit_hash
        third_hash = log[-3].commit_hash

        result = t.compress(
            from_commit=first_hash,
            to_commit=third_hash,
            target_tokens=100,
        )

        assert result is not None
        assert compress_client.call_count >= 1
        assert default.call_count == 0
        t.close()

    def test_manual_compress_no_client_needed(self):
        """Manual compression (content=) doesn't need any client."""
        t = Tract.open()

        t.system("System prompt")
        t.user("Question")
        t.assistant("Answer")

        log = t.log(limit=10)
        first_hash = log[-1].commit_hash
        last_hash = log[0].commit_hash

        result = t.compress(
            from_commit=first_hash,
            to_commit=last_hash,
            content="Summary of everything",
        )

        assert result is not None
        t.close()


# ---------------------------------------------------------------------------
# Lifecycle: close()
# ---------------------------------------------------------------------------


class TestCloseDoesNotCloseOperationClients:
    """close() only closes the owned default client, not operation clients."""

    def test_operation_clients_not_closed(self):
        t = Tract.open()
        default = MockLLMClient(name="default")
        chat_client = MockLLMClient(name="chat")
        compress_client = MockLLMClient(name="compress")

        t.configure_llm(default)
        t.configure_clients(chat=chat_client, compress=compress_client)

        t.close()

        # Operation clients are user-provided — not closed by tract
        assert not chat_client.closed
        assert not compress_client.closed

    def test_default_client_closed_when_owned(self):
        """Internally-created clients (via open(api_key=)) ARE closed."""
        t = Tract.open()
        default = MockLLMClient(name="default")
        t.configure_llm(default)
        # Simulate ownership (normally set by Tract.open when it creates the client)
        t._owns_llm_client = True

        t.close()
        assert default.closed


# ---------------------------------------------------------------------------
# Mixed operation: different clients for different operations
# ---------------------------------------------------------------------------


class TestMixedOperationClients:
    """Different operations route to different clients."""

    def test_chat_and_compress_use_different_clients(self):
        t = Tract.open()
        chat_client = MockLLMClient(name="chat", response_text="Chat reply")
        compress_client = MockLLMClient(
            name="compress", response_text="Compressed"
        )
        t.configure_clients(chat=chat_client, compress=compress_client)

        # Chat uses chat_client
        t.system("Hello")
        response = t.chat("Question")
        assert response.text == "Chat reply"
        assert chat_client.call_count == 1
        assert compress_client.call_count == 0

        t.close()

    def test_unset_operation_falls_through(self):
        """Operations without per-op client use default."""
        t = Tract.open()
        default = MockLLMClient(name="default", response_text="Default")
        chat_client = MockLLMClient(name="chat", response_text="Chat only")
        t.configure_llm(default)
        t.configure_clients(chat=chat_client)

        # Chat uses operation client
        t.system("Hello")
        response = t.chat("Question 1")
        assert response.text == "Chat only"

        # Generate (also uses "chat" operation) uses operation client too
        t.user("Question 2")
        response = t.generate()
        assert response.text == "Chat only"
        assert chat_client.call_count == 2
        assert default.call_count == 0
        t.close()
