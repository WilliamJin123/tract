"""Tests for tract.gate -- SemanticGate and GateResult."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from tract.exceptions import BlockedError
from tract.gate import GateResult, SemanticGate, _GATE_SYSTEM_PROMPT
from tract.middleware import MiddlewareContext
from tract.models.commit import CommitInfo, CommitOperation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_commit_info(
    *,
    commit_hash: str = "a" * 40,
    content_type: str = "assistant",
    token_count: int = 100,
    message: str | None = "Test commit",
    tags: list[str] | None = None,
    effective_priority: str = "NORMAL",
) -> CommitInfo:
    return CommitInfo(
        commit_hash=commit_hash,
        tract_id="test-tract",
        parent_hash=None,
        content_hash="h" * 40,
        content_type=content_type,
        operation=CommitOperation.APPEND,
        message=message,
        token_count=token_count,
        tags=tags or [],
        effective_priority=effective_priority,
        created_at=datetime.now(),
    )


class FakeLLMClient:
    """Minimal LLM client for testing gate calls."""

    def __init__(self, response_text: str = '{"result": "pass", "reason": "ok"}'):
        self.response_text = response_text
        self.last_messages: list[dict] | None = None
        self.last_kwargs: dict[str, Any] = {}

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        self.last_messages = messages
        self.last_kwargs = kwargs
        return {
            "choices": [{"message": {"content": self.response_text}}],
            "usage": {"total_tokens": 42},
        }

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict | None:
        return response.get("usage")

    def close(self) -> None:
        pass


def _make_ctx(
    tract_mock: MagicMock,
    event: str = "pre_transition",
) -> MiddlewareContext:
    """Build a MiddlewareContext with the given mock tract."""
    return MiddlewareContext(
        event=event,  # type: ignore[arg-type]
        commit=None,
        tract=tract_mock,
        branch="main",
        head="a" * 40,
    )


def _make_tract_mock(
    commits: list[CommitInfo] | None = None,
    config: dict[str, Any] | None = None,
    client: Any | None = None,
) -> MagicMock:
    """Create a mock Tract with search/config/llm methods wired up."""
    mock = MagicMock()
    mock.log.return_value = commits or []
    mock.config.get_all.return_value = config or {}
    mock.current_branch = "main"
    mock.head = "a" * 40

    mock.config.get_prompt.return_value = None

    if client is not None:
        mock.config._resolve_llm_client.return_value = client
    else:
        mock.config._resolve_llm_client.side_effect = RuntimeError("No LLM client")

    return mock


# ---------------------------------------------------------------------------
# GateResult tests
# ---------------------------------------------------------------------------

class TestGateResult:
    def test_frozen(self):
        r = GateResult(gate_name="g", passed=True, reason="ok", tokens_used=10)
        with pytest.raises(AttributeError):
            r.passed = False  # type: ignore[misc]

    def test_fields(self):
        r = GateResult(gate_name="x", passed=False, reason="bad", tokens_used=99)
        assert r.gate_name == "x"
        assert r.passed is False
        assert r.reason == "bad"
        assert r.tokens_used == 99


# ---------------------------------------------------------------------------
# SemanticGate construction tests
# ---------------------------------------------------------------------------

class TestSemanticGateConstruction:
    def test_defaults(self):
        g = SemanticGate(name="g1", check="criterion")
        assert g.name == "g1"
        assert g.check == "criterion"
        assert g.model is None
        assert g.condition is None
        assert g.temperature == 0.1
        assert g.max_log_entries == 30
        assert g.last_result is None

    def test_custom_values(self):
        cond = lambda ctx: True
        g = SemanticGate(
            name="g2",
            check="check",
            model="gpt-4o",
            condition=cond,
            temperature=0.5,
            max_log_entries=10,
        )
        assert g.model == "gpt-4o"
        assert g.condition is cond
        assert g.temperature == 0.5
        assert g.max_log_entries == 10


# ---------------------------------------------------------------------------
# Condition (deterministic pre-check) tests
# ---------------------------------------------------------------------------

class TestCondition:
    def test_condition_false_skips_gate(self):
        """When condition returns False, gate auto-passes without LLM call."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(
            name="skip-gate",
            check="anything",
            condition=lambda c: False,
        )
        gate(ctx)  # Should not raise

        assert gate.last_result is not None
        assert gate.last_result.passed is True
        assert "skipped" in gate.last_result.reason.lower()
        # LLM should NOT have been called
        assert client.last_messages is None

    def test_condition_true_proceeds(self):
        """When condition returns True, gate proceeds with LLM check."""
        client = FakeLLMClient('{"result": "pass", "reason": "looks good"}')
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(
            name="proceed-gate",
            check="criterion",
            condition=lambda c: True,
        )
        gate(ctx)

        assert gate.last_result is not None
        assert gate.last_result.passed is True
        assert client.last_messages is not None

    def test_condition_exception_passes(self):
        """If condition callback raises, gate passes (fail-open)."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        def bad_condition(c):
            raise ValueError("boom")

        gate = SemanticGate(name="err-gate", check="x", condition=bad_condition)
        gate(ctx)  # Should not raise

        assert gate.last_result is not None
        assert gate.last_result.passed is True
        assert client.last_messages is None


# ---------------------------------------------------------------------------
# LLM client resolution tests
# ---------------------------------------------------------------------------

class TestLLMClientResolution:
    def test_no_client_raises_runtime_error(self):
        """If no LLM client configured, RuntimeError is raised."""
        tract_mock = _make_tract_mock()  # No client -> side_effect RuntimeError
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="no-client", check="anything")
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            gate(ctx)

    def test_no_client_sets_last_result(self):
        """last_result is populated even when client resolution fails."""
        tract_mock = _make_tract_mock()
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="no-client", check="anything")
        with pytest.raises(RuntimeError):
            gate(ctx)

        assert gate.last_result is not None
        assert gate.last_result.passed is False  # error, not a pass
        assert gate.last_result.tokens_used == 0
        assert "No LLM client" in gate.last_result.reason

    def test_client_resolved_for_gate_operation(self):
        """Client is resolved with 'gate' operation key."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="op-check", check="x")
        gate(ctx)

        tract_mock.config._resolve_llm_client.assert_called_once_with("gate")


# ---------------------------------------------------------------------------
# Full __call__ integration tests
# ---------------------------------------------------------------------------

class TestGateCall:
    def test_pass_scenario(self):
        client = FakeLLMClient('{"result": "pass", "reason": "All good"}')
        tract_mock = _make_tract_mock(
            commits=[_make_commit_info(tags=["finding"])],
            config={"stage": "research"},
            client=client,
        )
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="pass-gate", check="Has findings")
        gate(ctx)  # Should not raise

        assert gate.last_result is not None
        assert gate.last_result.passed is True
        assert gate.last_result.reason == "All good"
        assert gate.last_result.tokens_used == 42

    def test_fail_scenario_raises_blocked_error(self):
        client = FakeLLMClient('{"result": "fail", "reason": "No findings"}')
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="fail-gate", check="Has findings")
        with pytest.raises(BlockedError) as exc_info:
            gate(ctx)

        assert "fail-gate" in str(exc_info.value)
        assert exc_info.value.event == "pre_transition"
        assert gate.last_result is not None
        assert gate.last_result.passed is False

    def test_llm_exception_fails_open(self):
        """If the LLM call throws, gate passes (fail-open)."""
        client = MagicMock()
        client.chat.side_effect = ConnectionError("network error")
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="err-gate", check="x")
        gate(ctx)  # Should not raise

        assert gate.last_result is not None
        assert gate.last_result.passed is True
        assert "fail" in gate.last_result.reason.lower()

    def test_model_override_passed_to_chat(self):
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="model-gate", check="x", model="gpt-4o")
        gate(ctx)

        assert client.last_kwargs.get("model") == "gpt-4o"

    def test_temperature_passed_to_chat(self):
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="temp-gate", check="x", temperature=0.3)
        gate(ctx)

        assert client.last_kwargs.get("temperature") == 0.3

    def test_no_model_override_omits_model_kwarg(self):
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="no-model", check="x")
        gate(ctx)

        assert "model" not in client.last_kwargs

    def test_last_result_updated_on_each_call(self):
        client = FakeLLMClient('{"result": "pass", "reason": "r1"}')
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="multi", check="x")
        gate(ctx)
        assert gate.last_result.reason == "r1"

        client.response_text = '{"result": "pass", "reason": "r2"}'
        gate(ctx)
        assert gate.last_result.reason == "r2"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_get_all_configs_exception_handled(self):
        """If get_all_configs raises, manifest still builds."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        tract_mock.config.get_all.side_effect = Exception("config error")
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="cfg-err", check="x")
        gate(ctx)  # Should not raise

        assert gate.last_result.passed is True

    def test_extract_usage_missing(self):
        """If client lacks extract_usage, tokens_used is 0."""

        class NoUsageClient:
            def chat(self, messages, **kwargs):
                return {"choices": [{"message": {"content": '{"result":"pass","reason":"ok"}'}}]}
            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]
            def close(self):
                pass

        client = NoUsageClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="no-usage", check="x")
        gate(ctx)

        assert gate.last_result.tokens_used == 0

    def test_extract_usage_returns_none(self):
        """If extract_usage returns None, tokens_used is 0."""
        client = FakeLLMClient()
        client.extract_usage = lambda r: None
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        gate = SemanticGate(name="none-usage", check="x")
        gate(ctx)

        assert gate.last_result.tokens_used == 0

    def test_to_spec_and_from_spec(self):
        """Serialization round-trip preserves gate configuration."""
        gate = SemanticGate(
            name="test-gate",
            check="Has findings",
            model="gpt-4o",
            condition=lambda c: True,
            temperature=0.5,
            max_log_entries=20,
        )
        spec = gate.to_spec()

        assert spec["name"] == "test-gate"
        assert spec["check"] == "Has findings"
        assert spec["model"] == "gpt-4o"
        assert spec["has_condition"] is True
        assert spec["temperature"] == 0.5
        assert spec["max_log_entries"] == 20

        restored = SemanticGate.from_spec(spec)
        assert restored.name == "test-gate"
        assert restored.check == "Has findings"
        assert restored.model == "gpt-4o"
        assert restored.condition is None  # not restorable
        assert restored.temperature == 0.5
        assert restored.max_log_entries == 20
