"""Tests for tract.maintain -- SemanticMaintainer and MaintainResult."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest

from tract.maintain import (
    MaintainResult,
    SemanticMaintainer,
    _MAINTAINER_SYSTEM_PROMPT,
    build_manifest,
)
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
    """Minimal LLM client for testing maintainer calls."""

    def __init__(self, response_text: str = '{"reasoning": "ok", "actions": []}'):
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
    event: str = "post_commit",
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
    """Create a mock Tract with log/config/llm methods wired up."""
    mock = MagicMock()
    mock.log.return_value = commits or []
    mock.get_all_configs.return_value = config or {}
    mock.current_branch = "main"
    mock.head = "a" * 40

    if client is not None:
        mock._resolve_llm_client.return_value = client
    else:
        mock._resolve_llm_client.side_effect = RuntimeError("No LLM client")

    return mock


def _action_response(reasoning: str = "maintenance needed", actions: list[dict] | None = None) -> str:
    return json.dumps({
        "reasoning": reasoning,
        "actions": actions or [],
    })


# ---------------------------------------------------------------------------
# MaintainResult tests
# ---------------------------------------------------------------------------

class TestMaintainResult:
    def test_frozen(self):
        r = MaintainResult(
            maintainer_name="m",
            actions_requested=1,
            actions_executed=1,
            actions_failed=0,
            tokens_used=10,
            reasoning="ok",
            errors=[],
        )
        with pytest.raises(AttributeError):
            r.actions_executed = 5  # type: ignore[misc]

    def test_fields(self):
        r = MaintainResult(
            maintainer_name="x",
            actions_requested=3,
            actions_executed=2,
            actions_failed=1,
            tokens_used=99,
            reasoning="did stuff",
            errors=["boom"],
        )
        assert r.maintainer_name == "x"
        assert r.actions_requested == 3
        assert r.actions_executed == 2
        assert r.actions_failed == 1
        assert r.tokens_used == 99
        assert r.reasoning == "did stuff"
        assert r.errors == ["boom"]


# ---------------------------------------------------------------------------
# SemanticMaintainer construction tests
# ---------------------------------------------------------------------------

class TestSemanticMaintainerConstruction:
    def test_defaults(self):
        m = SemanticMaintainer(
            name="m1",
            instructions="Keep it clean",
            actions=["annotate", "gc"],
        )
        assert m.name == "m1"
        assert m.instructions == "Keep it clean"
        assert m.actions == ["annotate", "gc"]
        assert m.model is None
        assert m.condition is None
        assert m.temperature == 0.1
        assert m.max_log_entries == 30
        assert m.last_result is None

    def test_custom_values(self):
        cond = lambda ctx: True
        m = SemanticMaintainer(
            name="m2",
            instructions="Do things",
            actions=["configure", "directive"],
            model="gpt-4o",
            condition=cond,
            temperature=0.5,
            max_log_entries=10,
        )
        assert m.model == "gpt-4o"
        assert m.condition is cond
        assert m.temperature == 0.5
        assert m.max_log_entries == 10

    def test_invalid_action_types_rejected(self):
        with pytest.raises(ValueError, match="Invalid action types"):
            SemanticMaintainer(
                name="bad",
                instructions="x",
                actions=["annotate", "delete_everything"],
            )

    def test_empty_actions_rejected(self):
        with pytest.raises(ValueError, match="At least one action type"):
            SemanticMaintainer(
                name="empty",
                instructions="x",
                actions=[],
            )

    def test_all_valid_action_types(self):
        m = SemanticMaintainer(
            name="all",
            instructions="x",
            actions=["annotate", "compress", "configure", "directive", "tag", "gc"],
        )
        assert len(m.actions) == 6


# ---------------------------------------------------------------------------
# Condition (deterministic pre-check) tests
# ---------------------------------------------------------------------------

class TestCondition:
    def test_condition_false_skips_maintainer(self):
        """When condition returns False, maintainer does nothing."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="skip-m",
            instructions="anything",
            actions=["gc"],
            condition=lambda c: False,
        )
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0
        assert "skipped" in m.last_result.reasoning.lower()
        # LLM should NOT have been called
        assert client.last_messages is None

    def test_condition_true_proceeds(self):
        """When condition returns True, maintainer proceeds with LLM call."""
        client = FakeLLMClient(_action_response("nothing to do"))
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="proceed-m",
            instructions="do stuff",
            actions=["gc"],
            condition=lambda c: True,
        )
        m(ctx)

        assert m.last_result is not None
        assert client.last_messages is not None

    def test_condition_exception_skips(self):
        """If condition callback raises, maintainer skips (fail-open)."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        def bad_condition(c):
            raise ValueError("boom")

        m = SemanticMaintainer(
            name="err-m",
            instructions="x",
            actions=["gc"],
            condition=bad_condition,
        )
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0
        assert client.last_messages is None


# ---------------------------------------------------------------------------
# LLM client resolution tests
# ---------------------------------------------------------------------------

class TestLLMClientResolution:
    def test_no_client_raises_runtime_error(self):
        """If no LLM client configured, RuntimeError is raised."""
        tract_mock = _make_tract_mock()  # No client -> side_effect RuntimeError
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="no-client", instructions="x", actions=["gc"])
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            m(ctx)

    def test_client_resolved_for_maintain_operation(self):
        """Client is resolved with 'maintain' operation key."""
        client = FakeLLMClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="op-check", instructions="x", actions=["gc"])
        m(ctx)

        tract_mock._resolve_llm_client.assert_called_once_with("maintain")


# ---------------------------------------------------------------------------
# Manifest building tests
# ---------------------------------------------------------------------------

class TestBuildManifest:
    def test_empty_log(self):
        tract_mock = _make_tract_mock(commits=[], config={})
        manifest = build_manifest(tract_mock, max_log_entries=30)

        assert "CONTEXT MANIFEST" in manifest
        assert "Branch: main" in manifest
        assert "Commits shown: 0" in manifest
        assert "COMMIT LOG" not in manifest

    def test_with_commits(self):
        commits = [
            _make_commit_info(
                commit_hash="a1b2c3d4" + "0" * 32,
                content_type="assistant",
                token_count=847,
                message="Analysis of B-tree indexing strategies",
                tags=["research"],
                effective_priority="NORMAL",
            ),
        ]
        tract_mock = _make_tract_mock(commits=commits, config={"stage": "research"})
        manifest = build_manifest(tract_mock, max_log_entries=30)

        assert "COMMIT LOG" in manifest
        assert "a1b2c3d4" in manifest
        assert "assistant" in manifest
        assert "847" in manifest
        assert "research" in manifest
        assert "ACTIVE CONFIG" in manifest


# ---------------------------------------------------------------------------
# Message building tests
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_message_structure(self):
        m = SemanticMaintainer(
            name="msg-test",
            instructions="Clean up old tool_io",
            actions=["annotate", "gc"],
        )
        manifest = "=== CONTEXT MANIFEST ===\nBranch: main"
        ctx = _make_ctx(_make_tract_mock())
        messages = m._build_messages(manifest, ctx)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == _MAINTAINER_SYSTEM_PROMPT
        assert messages[1]["role"] == "user"
        assert "Clean up old tool_io" in messages[1]["content"]
        assert "MAINTENANCE INSTRUCTIONS" in messages[1]["content"]
        assert "ALLOWED ACTIONS" in messages[1]["content"]
        assert "annotate" in messages[1]["content"]
        assert "gc" in messages[1]["content"]
        assert "CONTEXT MANIFEST" in messages[1]["content"]
        assert "post_commit" in messages[1]["content"]


# ---------------------------------------------------------------------------
# Response parsing tests
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_json_with_actions(self):
        text = json.dumps({
            "reasoning": "Cleaning up",
            "actions": [
                {"type": "gc"},
                {"type": "annotate", "target": "abc123", "priority": "skip"},
            ],
        })
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert reasoning == "Cleaning up"
        assert len(actions) == 2
        assert actions[0]["type"] == "gc"
        assert actions[1]["type"] == "annotate"

    def test_json_empty_actions(self):
        text = '{"reasoning": "Nothing to do", "actions": []}'
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert reasoning == "Nothing to do"
        assert actions == []

    def test_json_with_code_fence(self):
        text = '```json\n{"reasoning": "fence", "actions": [{"type": "gc"}]}\n```'
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert reasoning == "fence"
        assert len(actions) == 1

    def test_malformed_json_returns_empty(self):
        text = "This is not valid JSON at all."
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert actions == []
        assert "could not parse" in reasoning.lower()

    def test_json_no_reasoning(self):
        text = '{"actions": [{"type": "gc"}]}'
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert reasoning == "(no reasoning given)"
        assert len(actions) == 1

    def test_invalid_action_entries_filtered(self):
        """Actions without a 'type' key are silently dropped."""
        text = json.dumps({
            "reasoning": "mixed",
            "actions": [
                {"type": "gc"},
                {"not_type": "annotate"},  # missing "type"
                "just a string",  # not a dict
            ],
        })
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert len(actions) == 1
        assert actions[0]["type"] == "gc"

    def test_actions_not_list_treated_as_empty(self):
        text = '{"reasoning": "oops", "actions": "not a list"}'
        reasoning, actions = SemanticMaintainer._parse_response(text)
        assert actions == []


# ---------------------------------------------------------------------------
# Action execution tests
# ---------------------------------------------------------------------------

class TestActionExecution:
    def test_annotate_action(self):
        """annotate action calls t.annotate with resolved hash and priority."""
        from tract.models.annotations import Priority

        tract_mock = MagicMock()
        tract_mock.resolve_commit.return_value = "a" * 40

        SemanticMaintainer._exec_annotate(tract_mock, {
            "type": "annotate",
            "target": "a1b2c3d4",
            "priority": "skip",
        })

        tract_mock.resolve_commit.assert_called_once_with("a1b2c3d4")
        tract_mock.annotate.assert_called_once_with(
            "a" * 40, Priority.SKIP, reason=None,
        )

    def test_annotate_with_reason(self):
        from tract.models.annotations import Priority

        tract_mock = MagicMock()
        tract_mock.resolve_commit.return_value = "b" * 40

        SemanticMaintainer._exec_annotate(tract_mock, {
            "type": "annotate",
            "target": "b1b2c3d4",
            "priority": "important",
            "reason": "This is crucial",
        })

        tract_mock.annotate.assert_called_once_with(
            "b" * 40, Priority.IMPORTANT, reason="This is crucial",
        )

    def test_annotate_invalid_priority_raises(self):
        tract_mock = MagicMock()
        tract_mock.resolve_commit.return_value = "a" * 40

        with pytest.raises(ValueError, match="Invalid priority"):
            SemanticMaintainer._exec_annotate(tract_mock, {
                "type": "annotate",
                "target": "abc",
                "priority": "ultra_high",
            })

    def test_compress_action(self):
        tract_mock = MagicMock()
        tract_mock.resolve_commit.side_effect = lambda x: x + "0" * (40 - len(x))

        SemanticMaintainer._exec_compress(tract_mock, {
            "type": "compress",
            "commits": ["abc", "def"],
            "instructions": "Summarize these findings",
        })

        assert tract_mock.resolve_commit.call_count == 2
        tract_mock.compress.assert_called_once()
        call_kwargs = tract_mock.compress.call_args
        assert call_kwargs[1]["instructions"] == "Summarize these findings"

    def test_compress_no_commits_raises(self):
        tract_mock = MagicMock()
        with pytest.raises(ValueError, match="requires 'commits'"):
            SemanticMaintainer._exec_compress(tract_mock, {
                "type": "compress",
                "commits": [],
            })

    def test_configure_action(self):
        tract_mock = MagicMock()

        SemanticMaintainer._exec_configure(tract_mock, {
            "type": "configure",
            "key": "stage",
            "value": "implementation",
        })

        tract_mock.configure.assert_called_once_with(stage="implementation")

    def test_configure_no_key_raises(self):
        tract_mock = MagicMock()
        with pytest.raises(ValueError, match="requires a 'key'"):
            SemanticMaintainer._exec_configure(tract_mock, {
                "type": "configure",
                "key": "",
                "value": "x",
            })

    def test_directive_action(self):
        tract_mock = MagicMock()

        SemanticMaintainer._exec_directive(tract_mock, {
            "type": "directive",
            "name": "focus",
            "text": "Focus on testing",
        })

        tract_mock.directive.assert_called_once_with("focus", "Focus on testing")

    def test_directive_no_name_raises(self):
        tract_mock = MagicMock()
        with pytest.raises(ValueError, match="requires a 'name'"):
            SemanticMaintainer._exec_directive(tract_mock, {
                "type": "directive",
                "name": "",
                "text": "some text",
            })

    def test_directive_no_text_raises(self):
        tract_mock = MagicMock()
        with pytest.raises(ValueError, match="requires 'text'"):
            SemanticMaintainer._exec_directive(tract_mock, {
                "type": "directive",
                "name": "focus",
                "text": "",
            })

    def test_tag_action(self):
        tract_mock = MagicMock()
        tract_mock.resolve_commit.return_value = "c" * 40

        m = SemanticMaintainer(name="tagger", instructions="x", actions=["tag"])
        m._exec_tag(tract_mock, {
            "type": "tag",
            "target": "c1c2c3",
            "tag": "key-finding",
        })

        tract_mock.resolve_commit.assert_called_once_with("c1c2c3")
        tract_mock.register_tag.assert_called_once_with(
            "key-finding", "Auto-registered by maintainer 'tagger'"
        )
        tract_mock.tag.assert_called_once_with("c" * 40, "key-finding")

    def test_tag_no_target_raises(self):
        tract_mock = MagicMock()
        m = SemanticMaintainer(name="t", instructions="x", actions=["tag"])
        with pytest.raises(ValueError, match="requires a 'target'"):
            m._exec_tag(tract_mock, {
                "type": "tag",
                "target": "",
                "tag": "x",
            })

    def test_tag_no_tag_name_raises(self):
        tract_mock = MagicMock()
        m = SemanticMaintainer(name="t", instructions="x", actions=["tag"])
        with pytest.raises(ValueError, match="requires a 'tag'"):
            m._exec_tag(tract_mock, {
                "type": "tag",
                "target": "abc",
                "tag": "",
            })

    def test_gc_action(self):
        tract_mock = MagicMock()

        SemanticMaintainer._exec_gc(tract_mock, {"type": "gc"})

        tract_mock.gc.assert_called_once()


# ---------------------------------------------------------------------------
# Full __call__ tests
# ---------------------------------------------------------------------------

class TestMaintainerCall:
    def test_no_actions_scenario(self):
        client = FakeLLMClient(_action_response("Nothing to do"))
        tract_mock = _make_tract_mock(
            commits=[_make_commit_info()],
            config={"stage": "research"},
            client=client,
        )
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="noop",
            instructions="Maintain context",
            actions=["gc"],
        )
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0
        assert m.last_result.actions_executed == 0
        assert m.last_result.reasoning == "Nothing to do"
        assert m.last_result.tokens_used == 42

    def test_gc_action_scenario(self):
        response = _action_response("Running gc", [{"type": "gc"}])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="gc-maintainer",
            instructions="Clean up",
            actions=["gc"],
        )
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 1
        assert m.last_result.actions_executed == 1
        assert m.last_result.actions_failed == 0
        tract_mock.gc.assert_called_once()

    def test_disallowed_action_filtered(self):
        """Actions not in the allowed list are silently skipped."""
        response = _action_response("Doing things", [
            {"type": "gc"},
            {"type": "annotate", "target": "abc", "priority": "skip"},
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        # Only allow gc, not annotate
        m = SemanticMaintainer(
            name="filtered",
            instructions="x",
            actions=["gc"],
        )
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 1  # only gc counted
        assert m.last_result.actions_executed == 1
        tract_mock.gc.assert_called_once()
        # annotate should NOT have been called
        tract_mock.annotate.assert_not_called()

    def test_action_failure_collected(self):
        """If an action raises, it's captured in errors but doesn't crash."""
        response = _action_response("Doing things", [
            {"type": "gc"},
            {"type": "configure", "key": "stage", "value": "next"},
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        # Make gc() raise
        tract_mock.gc.side_effect = RuntimeError("GC not available")
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="error-m",
            instructions="x",
            actions=["gc", "configure"],
        )
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 2
        assert m.last_result.actions_executed == 1  # configure succeeded
        assert m.last_result.actions_failed == 1  # gc failed
        assert len(m.last_result.errors) == 1
        assert "gc" in m.last_result.errors[0].lower()

    def test_llm_exception_fails_open(self):
        """If the LLM call throws, maintainer skips (fail-open)."""
        client = MagicMock()
        client.chat.side_effect = ConnectionError("network error")
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="err-m", instructions="x", actions=["gc"])
        m(ctx)  # Should not raise

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0
        assert "fail-open" in m.last_result.reasoning.lower()

    def test_extract_content_failure_fails_open(self):
        """If extract_content raises, maintainer skips (fail-open)."""
        client = MagicMock()
        client.chat.return_value = {"bad": "response"}
        client.extract_content.side_effect = KeyError("no content")
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="extract-err", instructions="x", actions=["gc"])
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0
        assert "fail-open" in m.last_result.reasoning.lower()

    def test_model_override_passed_to_chat(self):
        client = FakeLLMClient(_action_response())
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="model-m",
            instructions="x",
            actions=["gc"],
            model="gpt-4o",
        )
        m(ctx)

        assert client.last_kwargs.get("model") == "gpt-4o"

    def test_temperature_passed_to_chat(self):
        client = FakeLLMClient(_action_response())
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="temp-m",
            instructions="x",
            actions=["gc"],
            temperature=0.3,
        )
        m(ctx)

        assert client.last_kwargs.get("temperature") == 0.3

    def test_no_model_override_omits_model_kwarg(self):
        client = FakeLLMClient(_action_response())
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="no-model", instructions="x", actions=["gc"])
        m(ctx)

        assert "model" not in client.last_kwargs

    def test_last_result_updated_on_each_call(self):
        client = FakeLLMClient(_action_response("first"))
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="multi", instructions="x", actions=["gc"])
        m(ctx)
        assert m.last_result.reasoning == "first"

        client.response_text = _action_response("second")
        m(ctx)
        assert m.last_result.reasoning == "second"

    def test_tokens_used_tracked(self):
        client = FakeLLMClient(_action_response())
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="tokens", instructions="x", actions=["gc"])
        m(ctx)

        assert m.last_result.tokens_used == 42


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_get_all_configs_exception_handled(self):
        """If get_all_configs raises, manifest still builds."""
        client = FakeLLMClient(_action_response())
        tract_mock = _make_tract_mock(client=client)
        tract_mock.get_all_configs.side_effect = Exception("config error")
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="cfg-err", instructions="x", actions=["gc"])
        m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0

    def test_extract_usage_missing(self):
        """If client lacks extract_usage, tokens_used is 0."""

        class NoUsageClient:
            def chat(self, messages, **kwargs):
                return {"choices": [{"message": {"content": _action_response()}}]}
            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]
            def close(self):
                pass

        client = NoUsageClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="no-usage", instructions="x", actions=["gc"])
        m(ctx)

        assert m.last_result.tokens_used == 0

    def test_extract_usage_returns_none(self):
        """If extract_usage returns None, tokens_used is 0."""
        client = FakeLLMClient(_action_response())
        client.extract_usage = lambda r: None
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(name="none-usage", instructions="x", actions=["gc"])
        m(ctx)

        assert m.last_result.tokens_used == 0

    def test_annotate_all_priority_values(self):
        """All four priority strings map correctly."""
        from tract.models.annotations import Priority

        for prio_str, prio_enum in [
            ("skip", Priority.SKIP),
            ("normal", Priority.NORMAL),
            ("important", Priority.IMPORTANT),
            ("pinned", Priority.PINNED),
        ]:
            tract_mock = MagicMock()
            tract_mock.resolve_commit.return_value = "x" * 40
            SemanticMaintainer._exec_annotate(tract_mock, {
                "type": "annotate",
                "target": "xabc",
                "priority": prio_str,
            })
            tract_mock.annotate.assert_called_once_with(
                "x" * 40, prio_enum, reason=None,
            )


# ---------------------------------------------------------------------------
# Block action tests
# ---------------------------------------------------------------------------

class TestBlockAction:
    def test_block_in_valid_actions(self):
        """'block' is a valid action type."""
        m = SemanticMaintainer(
            name="block-test",
            instructions="x",
            actions=["block"],
        )
        assert "block" in m.actions

    def test_block_raises_blocked_error(self):
        """Block action raises BlockedError after executing other actions."""
        from tract.exceptions import BlockedError

        response = _action_response("Bad state detected", [
            {"type": "configure", "key": "stage", "value": "paused"},
            {"type": "block", "reason": "Bad state"},
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="block-m",
            instructions="Block if bad state",
            actions=["configure", "block"],
        )
        with pytest.raises(BlockedError, match="Bad state"):
            m(ctx)

        # configure should have been executed first
        tract_mock.configure.assert_called_once_with(stage="paused")

    def test_block_without_reason(self):
        """Block action with no reason uses default text."""
        from tract.exceptions import BlockedError

        response = _action_response("Blocking", [
            {"type": "block"},  # no "reason" key
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="no-reason-block",
            instructions="x",
            actions=["block"],
        )
        with pytest.raises(BlockedError) as exc_info:
            m(ctx)

        assert "(no reason given)" in exc_info.value.reasons[0]

    def test_multiple_block_actions(self):
        """Multiple block actions collect all reasons in BlockedError."""
        from tract.exceptions import BlockedError

        response = _action_response("Multiple blocks", [
            {"type": "block", "reason": "Reason A"},
            {"type": "block", "reason": "Reason B"},
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="multi-block",
            instructions="x",
            actions=["block"],
        )
        with pytest.raises(BlockedError) as exc_info:
            m(ctx)

        assert len(exc_info.value.reasons) == 2
        assert "Reason A" in exc_info.value.reasons[0]
        assert "Reason B" in exc_info.value.reasons[1]

    def test_block_only_action(self):
        """Block as the only action still raises BlockedError."""
        from tract.exceptions import BlockedError

        response = _action_response("Just blocking", [
            {"type": "block", "reason": "Stop everything"},
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="block-only",
            instructions="x",
            actions=["block"],
        )
        with pytest.raises(BlockedError, match="Stop everything"):
            m(ctx)

        # last_result should still be set before the raise
        assert m.last_result is not None
        assert m.last_result.actions_requested == 1
        assert m.last_result.actions_executed == 0  # block raises, not counted as executed

    def test_block_result_tracks_execution(self):
        """MaintainResult counts non-block executed actions; block raises separately."""
        from tract.exceptions import BlockedError

        response = _action_response("Block after gc", [
            {"type": "gc"},
            {"type": "block", "reason": "Done"},
        ])
        client = FakeLLMClient(response)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="track-block",
            instructions="x",
            actions=["gc", "block"],
        )
        with pytest.raises(BlockedError):
            m(ctx)

        assert m.last_result is not None
        assert m.last_result.actions_requested == 2
        assert m.last_result.actions_executed == 1  # only gc; block raises
        assert m.last_result.actions_failed == 0


# ---------------------------------------------------------------------------
# Peek parsing tests
# ---------------------------------------------------------------------------

class TestPeekParsing:
    def test_parse_peek_request(self):
        """Parse a JSON peek request."""
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(
            '{"peek": ["abc123", "def456"]}'
        )
        assert hashes == ["abc123", "def456"]
        assert actions is None

    def test_parse_direct_actions(self):
        """Parse direct actions (no peeking needed)."""
        text = '{"reasoning": "No peek needed", "actions": [{"type": "gc"}]}'
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(text)
        assert hashes == []
        assert actions is not None
        assert actions[0] == "No peek needed"
        assert len(actions[1]) == 1
        assert actions[1][0]["type"] == "gc"

    def test_parse_empty_peek_list(self):
        """Empty peek list means no peeking."""
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(
            '{"peek": []}'
        )
        assert hashes == []
        assert actions is None

    def test_parse_malformed_json(self):
        """Malformed JSON returns empty peeks, no actions."""
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(
            "this is not valid json at all"
        )
        assert hashes == []
        assert actions is None

    def test_parse_with_code_fences(self):
        """Code-fenced peek request is handled correctly."""
        text = '```json\n{"peek": ["abc123"]}\n```'
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(text)
        assert hashes == ["abc123"]
        assert actions is None

    def test_parse_peek_non_list(self):
        """Non-list peek value returns empty list."""
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(
            '{"peek": "not-a-list"}'
        )
        assert hashes == []
        assert actions is None

    def test_parse_peek_filters_empty_values(self):
        """Empty/falsy values in peek list are filtered out."""
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(
            '{"peek": ["abc", "", null, "def"]}'
        )
        assert hashes == ["abc", "def"]
        assert actions is None

    def test_parse_direct_actions_no_reasoning(self):
        """Direct actions with no reasoning use default."""
        text = '{"actions": [{"type": "gc"}]}'
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(text)
        assert hashes == []
        assert actions is not None
        assert actions[0] == "(no reasoning given)"

    def test_parse_direct_actions_filters_invalid(self):
        """Direct actions without 'type' key are filtered."""
        text = json.dumps({
            "reasoning": "mixed",
            "actions": [{"type": "gc"}, {"no_type": "bad"}, "string-entry"],
        })
        hashes, actions = SemanticMaintainer._parse_peek_or_actions(text)
        assert hashes == []
        assert actions is not None
        assert len(actions[1]) == 1
        assert actions[1][0]["type"] == "gc"


# ---------------------------------------------------------------------------
# Peeking flow tests
# ---------------------------------------------------------------------------

class PeekThenActionMockClient:
    """Returns peek request on first call, actions on second."""

    def __init__(self, peek_hashes: list[str], actions_response: str):
        self.call_count = 0
        self.peek_hashes = peek_hashes
        self.actions_response = actions_response
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        self.calls.append((messages, kwargs))
        self.call_count += 1
        if self.call_count == 1:
            text = json.dumps({"peek": self.peek_hashes})
        else:
            text = self.actions_response
        return {"choices": [{"message": {"content": text}}]}

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict:
        return {"total_tokens": 100}

    def close(self) -> None:
        pass


class DirectActionMockClient:
    """Returns direct actions on the first call (no peeking)."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        self.calls.append((messages, kwargs))
        return {"choices": [{"message": {"content": self.response_text}}]}

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict:
        return {"total_tokens": 100}

    def close(self) -> None:
        pass


class TestPeekingFlow:
    def test_peeking_disabled_by_default(self):
        """max_peeks=0 uses single-pass flow (no peek messages)."""
        client = FakeLLMClient(_action_response("single pass"))
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="no-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=0,
        )
        m(ctx)

        # Only one LLM call
        assert client.last_messages is not None
        # System prompt should be the action prompt, not the peek prompt
        assert "maintenance agent" in client.last_messages[0]["content"].lower()
        assert m.last_result.peeks_requested == 0
        assert m.last_result.peeks_performed == 0

    def test_peeking_two_pass(self):
        """max_peeks>0 triggers two-pass flow when LLM requests peeks."""
        hash1 = "a" * 40
        actions_resp = _action_response("After peeking", [{"type": "gc"}])
        client = PeekThenActionMockClient(
            peek_hashes=["a" * 8],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        # Set up resolve_commit and get_content for peeking
        tract_mock.resolve_commit.return_value = hash1
        tract_mock.get_content.return_value = "This is the content"
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="peek-m",
            instructions="Inspect commits",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        # Two LLM calls: peek pass + action pass
        assert client.call_count == 2
        assert m.last_result is not None
        assert m.last_result.peeks_requested == 1
        assert m.last_result.peeks_performed == 1
        assert m.last_result.actions_executed == 1
        tract_mock.gc.assert_called_once()

    def test_peeking_direct_actions(self):
        """LLM can skip peeking and return actions directly in pass 1."""
        direct_resp = _action_response("No peek needed", [{"type": "gc"}])
        client = DirectActionMockClient(direct_resp)
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="direct-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        # Only one LLM call (skipped peeking)
        assert len(client.calls) == 1
        assert m.last_result.peeks_requested == 0
        assert m.last_result.peeks_performed == 0
        assert m.last_result.actions_executed == 1
        tract_mock.gc.assert_called_once()

    def test_peeking_capped_at_max(self):
        """Peek requests exceeding max_peeks are capped."""
        actions_resp = _action_response("Done", [{"type": "gc"}])
        # Request 5 peeks but max_peeks=2
        client = PeekThenActionMockClient(
            peek_hashes=["a1", "b2", "c3", "d4", "e5"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.side_effect = lambda x: x + "0" * (40 - len(x))
        tract_mock.get_content.return_value = "content"
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="cap-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=2,
        )
        m(ctx)

        assert m.last_result.peeks_requested == 5
        assert m.last_result.peeks_performed == 2
        # Only 2 resolve_commit calls (capped)
        assert tract_mock.resolve_commit.call_count == 2

    def test_peeking_content_retrieval(self):
        """Peeked content appears in pass-2 messages."""
        actions_resp = _action_response("Reviewed", [])
        client = PeekThenActionMockClient(
            peek_hashes=["abc123"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.return_value = "abc123" + "0" * 34
        tract_mock.get_content.return_value = "Interesting commit content"
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="content-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        # Second call should contain the peeked content
        assert len(client.calls) == 2
        pass2_messages = client.calls[1][0]
        user_msg = pass2_messages[1]["content"]
        assert "PEEKED COMMIT CONTENTS" in user_msg
        assert "abc123" in user_msg
        assert "Interesting commit content" in user_msg

    def test_peeking_content_truncated(self):
        """Peeked content is truncated to 2000 chars."""
        long_content = "x" * 5000
        actions_resp = _action_response("Reviewed", [])
        client = PeekThenActionMockClient(
            peek_hashes=["abc"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.return_value = "abc" + "0" * 37
        tract_mock.get_content.return_value = long_content
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="truncate-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        # Verify truncation: pass-2 content should not have the full 5000 chars
        pass2_messages = client.calls[1][0]
        user_msg = pass2_messages[1]["content"]
        # The peeked content section should have at most 2000 x's
        assert "x" * 2000 in user_msg
        assert "x" * 2001 not in user_msg

    def test_peeking_dict_content_serialized(self):
        """Dict content is JSON-serialized and truncated."""
        actions_resp = _action_response("Done", [])
        client = PeekThenActionMockClient(
            peek_hashes=["abc"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.return_value = "abc" + "0" * 37
        tract_mock.get_content.return_value = {"key": "value", "nested": [1, 2, 3]}
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="dict-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        pass2_messages = client.calls[1][0]
        user_msg = pass2_messages[1]["content"]
        assert '"key"' in user_msg
        assert '"value"' in user_msg

    def test_peeking_none_content(self):
        """None content shows '(content not found)' message."""
        actions_resp = _action_response("Done", [])
        client = PeekThenActionMockClient(
            peek_hashes=["abc"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.return_value = "abc" + "0" * 37
        tract_mock.get_content.return_value = None
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="none-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        pass2_messages = client.calls[1][0]
        user_msg = pass2_messages[1]["content"]
        assert "(content not found)" in user_msg

    def test_peeking_failed_retrieval(self):
        """Failed content retrieval doesn't crash -- shows error message."""
        actions_resp = _action_response("Done", [])
        client = PeekThenActionMockClient(
            peek_hashes=["abc"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.side_effect = RuntimeError("No such commit")
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="fail-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        # Should not crash; should still make the second LLM call
        assert client.call_count == 2
        pass2_messages = client.calls[1][0]
        user_msg = pass2_messages[1]["content"]
        assert "could not retrieve" in user_msg

    def test_peek_result_tracking(self):
        """MaintainResult tracks peeks_requested and peeks_performed."""
        actions_resp = _action_response("Done", [{"type": "gc"}])
        client = PeekThenActionMockClient(
            peek_hashes=["h1", "h2", "h3"],
            actions_response=actions_resp,
        )
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.side_effect = lambda x: x + "0" * (40 - len(x))
        tract_mock.get_content.return_value = "content"
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="track-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=5,
        )
        m(ctx)

        assert m.last_result.peeks_requested == 3
        assert m.last_result.peeks_performed == 3
        assert m.last_result.tokens_used == 200  # 100 per call * 2 calls

    def test_peeking_llm_fail_open_pass1(self):
        """LLM failure during peek pass is fail-open."""
        client = MagicMock()
        client.chat.side_effect = ConnectionError("network error")
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="fail-peek-pass1",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)  # Should not raise

        assert m.last_result is not None
        assert m.last_result.actions_requested == 0
        assert "fail-open" in m.last_result.reasoning.lower()

    def test_peeking_llm_fail_open_pass2(self):
        """LLM failure during action pass (after successful peek) is fail-open."""
        class FailOnSecondCallClient:
            def __init__(self):
                self.call_count = 0
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                self.call_count += 1
                if self.call_count == 1:
                    return {"choices": [{"message": {"content": '{"peek": ["abc"]}'}}]}
                raise ConnectionError("second call fails")

            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]

            def extract_usage(self, response):
                return {"total_tokens": 100}

            def close(self):
                pass

        client = FailOnSecondCallClient()
        tract_mock = _make_tract_mock(client=client)
        tract_mock.resolve_commit.return_value = "abc" + "0" * 37
        tract_mock.get_content.return_value = "content"
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="fail-pass2",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)  # Should not raise

        assert m.last_result is not None
        assert "fail-open" in m.last_result.reasoning.lower()

    def test_peeking_empty_peek_list_falls_through(self):
        """Empty peek list from LLM triggers a normal action call."""
        class EmptyPeekThenActionClient:
            def __init__(self):
                self.call_count = 0
                self.calls = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                self.call_count += 1
                if self.call_count == 1:
                    return {"choices": [{"message": {"content": '{"peek": []}'}}]}
                return {"choices": [{"message": {"content": _action_response("Fallthrough", [{"type": "gc"}])}}]}

            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]

            def extract_usage(self, response):
                return {"total_tokens": 50}

            def close(self):
                pass

        client = EmptyPeekThenActionClient()
        tract_mock = _make_tract_mock(client=client)
        ctx = _make_ctx(tract_mock)

        m = SemanticMaintainer(
            name="empty-peek",
            instructions="x",
            actions=["gc"],
            max_peeks=3,
        )
        m(ctx)

        # Two LLM calls: empty peek + fallback action call
        assert client.call_count == 2
        assert m.last_result.peeks_requested == 0
        assert m.last_result.peeks_performed == 0
        assert m.last_result.actions_executed == 1
        tract_mock.gc.assert_called_once()


# ---------------------------------------------------------------------------
# _strip_fences tests
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_strip_json_fences(self):
        """Strips ```json ... ``` fences."""
        text = '```json\n{"key": "value"}\n```'
        assert SemanticMaintainer._strip_fences(text) == '{"key": "value"}'

    def test_strip_plain_fences(self):
        """Strips ``` ... ``` fences."""
        text = '```\n{"key": "value"}\n```'
        assert SemanticMaintainer._strip_fences(text) == '{"key": "value"}'

    def test_no_fences(self):
        """No fences -- returns text as-is (stripped)."""
        text = '  {"key": "value"}  '
        assert SemanticMaintainer._strip_fences(text) == '{"key": "value"}'

    def test_strip_fences_with_language_tag(self):
        """Strips fences with other language tags like ```python."""
        text = '```python\nprint("hello")\n```'
        assert SemanticMaintainer._strip_fences(text) == 'print("hello")'

    def test_strip_fences_preserves_inner_content(self):
        """Inner newlines and formatting are preserved."""
        text = '```json\n{\n  "a": 1,\n  "b": 2\n}\n```'
        result = SemanticMaintainer._strip_fences(text)
        assert '"a": 1' in result
        assert '"b": 2' in result


# ---------------------------------------------------------------------------
# MaintainResult peek fields tests
# ---------------------------------------------------------------------------

class TestMaintainResultPeekFields:
    def test_default_peek_fields(self):
        """MaintainResult defaults peeks_requested and peeks_performed to 0."""
        r = MaintainResult(
            maintainer_name="m",
            actions_requested=0,
            actions_executed=0,
            actions_failed=0,
            tokens_used=0,
            reasoning="ok",
            errors=[],
        )
        assert r.peeks_requested == 0
        assert r.peeks_performed == 0

    def test_custom_peek_fields(self):
        """MaintainResult accepts custom peek field values."""
        r = MaintainResult(
            maintainer_name="m",
            actions_requested=1,
            actions_executed=1,
            actions_failed=0,
            tokens_used=200,
            reasoning="peeked",
            errors=[],
            peeks_requested=5,
            peeks_performed=3,
        )
        assert r.peeks_requested == 5
        assert r.peeks_performed == 3
