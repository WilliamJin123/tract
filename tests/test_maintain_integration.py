"""Integration tests for SemanticMaintainer through the Tract API.

Tests maintainer registration, removal, listing, and action execution
using ``t.middleware.maintain()``, ``t.middleware.remove_maintainer()``, ``t.middleware.list_maintainers()``
with a mock LLM client (no real LLM calls).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import MaintainResult, Priority, SemanticMaintainer, Tract


# ---------------------------------------------------------------------------
# Mock LLM clients
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Fake LLM client that returns a canned response."""

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


class ErrorLLMClient:
    """LLM client that raises on every chat() call."""

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        raise ConnectionError("LLM service unavailable")

    def extract_content(self, response: dict) -> str:
        return ""

    def extract_usage(self, response: dict) -> dict:
        return {"total_tokens": 0}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_commits(t: Tract, n: int = 3, tag: str = "research") -> list[str]:
    """Add *n* commits tagged with *tag*. Returns list of commit hashes."""
    t.register_tag(tag, f"Auto-registered for test")
    hashes = []
    for i in range(n):
        info = t.commit(
            {"content_type": "dialogue", "role": "user", "text": f"Finding {i + 1}"},
            tags=[tag],
            message=f"research finding {i + 1}",
        )
        hashes.append(info.commit_hash)
    return hashes


def _action_response(reasoning: str = "maintenance", actions: list[dict] | None = None) -> str:
    return json.dumps({
        "reasoning": reasoning,
        "actions": actions or [],
    })


def _noop_response() -> str:
    return _action_response("Nothing to do", [])


# ---------------------------------------------------------------------------
# 1. Registration, listing, removal
# ---------------------------------------------------------------------------

class TestMaintainerRegistrationAndRemoval:
    def test_register_list_remove(self):
        """Register a maintainer via t.middleware.maintain(), verify in list, remove, verify gone."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            assert t.middleware.list_maintainers() == []

            t.middleware.maintain(
                "cleanup",
                event="post_commit",
                instructions="Mark old commits as SKIP",
                actions=["annotate"],
            )

            assert "cleanup" in t.middleware.list_maintainers()
            assert len(t.middleware.list_maintainers()) == 1

            t.middleware.remove_maintainer("cleanup")

            assert t.middleware.list_maintainers() == []

    def test_remove_nonexistent_raises(self):
        """Removing a maintainer that doesn't exist raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError, match="not found"):
                t.middleware.remove_maintainer("no-such-maintainer")


# ---------------------------------------------------------------------------
# 2. Duplicate name rejected
# ---------------------------------------------------------------------------

class TestDuplicateNameRejected:
    def test_duplicate_name_raises_value_error(self):
        """Registering two maintainers with the same name raises ValueError."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            t.middleware.maintain("unique-m", event="post_commit", instructions="a", actions=["gc"])

            with pytest.raises(ValueError, match="already registered"):
                t.middleware.maintain("unique-m", event="post_transition", instructions="b", actions=["gc"])

    def test_reregister_after_removal_succeeds(self):
        """After removing a maintainer, the same name can be reused."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            t.middleware.maintain("reusable", event="post_commit", instructions="v1", actions=["gc"])
            t.middleware.remove_maintainer("reusable")
            t.middleware.maintain("reusable", event="post_commit", instructions="v2", actions=["gc"])
            assert "reusable" in t.middleware.list_maintainers()


# ---------------------------------------------------------------------------
# 3. Maintainer executes annotate action
# ---------------------------------------------------------------------------

class TestMaintainerAnnotateAction:
    def test_annotate_on_post_commit(self):
        """Maintainer triggered by post_commit can annotate a commit."""
        with Tract.open() as t:
            # Seed a commit first
            hashes = _seed_commits(t, n=2)
            target_hash = hashes[0]
            prefix = target_hash[:8]

            # Set up mock that returns annotate action
            response = _action_response("Marking old commit as skip", [
                {"type": "annotate", "target": prefix, "priority": "skip"},
            ])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "auto-skip",
                event="post_commit",
                instructions="Mark old tool_io as SKIP",
                actions=["annotate"],
            )

            # Trigger the maintainer by committing
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger commit",
            )

            # Verify the LLM was called
            assert len(mock.calls) == 1

            # Verify the annotation was applied
            annotations = t.get_annotation(target_hash)
            assert any(a.priority == Priority.SKIP for a in annotations)


# ---------------------------------------------------------------------------
# 4. Maintainer executes configure action
# ---------------------------------------------------------------------------

class TestMaintainerConfigureAction:
    def test_configure_on_post_commit(self):
        """Maintainer can set config values."""
        with Tract.open() as t:
            _seed_commits(t, n=1)

            response = _action_response("Updating stage", [
                {"type": "configure", "key": "stage", "value": "implementation"},
            ])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "stage-setter",
                event="post_commit",
                instructions="Set stage based on content",
                actions=["configure"],
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            assert len(mock.calls) == 1
            # Verify config was set
            assert t.config.get("stage") == "implementation"


# ---------------------------------------------------------------------------
# 5. Maintainer executes directive action
# ---------------------------------------------------------------------------

class TestMaintainerDirectiveAction:
    def test_directive_on_post_commit(self):
        """Maintainer can create directives."""
        with Tract.open() as t:
            _seed_commits(t, n=1)

            response = _action_response("Adding focus directive", [
                {"type": "directive", "name": "current-focus", "text": "Focus on testing"},
            ])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "directive-setter",
                event="post_commit",
                instructions="Add focus directives",
                actions=["directive"],
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            assert len(mock.calls) == 1
            # Verify directive was created by checking log
            entries = t.log(limit=1)
            assert entries[0].content_type == "instruction"


# ---------------------------------------------------------------------------
# 6. Maintainer executes gc action
# ---------------------------------------------------------------------------

class TestMaintainerGCAction:
    def test_gc_on_post_commit(self):
        """Maintainer can trigger gc. (GC on fresh tract is a no-op but should not error.)"""
        with Tract.open() as t:
            _seed_commits(t, n=1)

            response = _action_response("Running gc", [{"type": "gc"}])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "auto-gc",
                event="post_commit",
                instructions="Run gc periodically",
                actions=["gc"],
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            assert len(mock.calls) == 1


# ---------------------------------------------------------------------------
# 7. Maintainer skips disallowed actions
# ---------------------------------------------------------------------------

class TestMaintainerSkipsDisallowed:
    def test_disallowed_action_skipped(self):
        """LLM returns an action type not in allowed list -- it's silently skipped."""
        with Tract.open() as t:
            _seed_commits(t, n=1)
            hashes = [e.commit_hash for e in t.log(limit=1)]

            response = _action_response("Trying everything", [
                {"type": "gc"},  # allowed
                {"type": "annotate", "target": hashes[0][:8], "priority": "skip"},  # NOT allowed
            ])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "restricted",
                event="post_commit",
                instructions="Do everything",
                actions=["gc"],  # only gc allowed
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            assert len(mock.calls) == 1
            # Only gc should have been requested, annotate filtered out
            # (we don't have direct access to last_result through the API,
            #  but the fact that no error was raised and annotation was not
            #  applied confirms filtering worked)
            annotations = t.get_annotation(hashes[0])
            # Should have no SKIP annotation
            assert not any(a.priority == Priority.SKIP for a in annotations)


# ---------------------------------------------------------------------------
# 8. Condition callback skips maintainer
# ---------------------------------------------------------------------------

class TestMaintainerCondition:
    def test_condition_false_skips(self):
        """Maintainer is skipped when condition returns False."""
        with Tract.open() as t:
            _seed_commits(t, n=1)

            response = _action_response("Would do stuff", [{"type": "gc"}])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "conditional",
                event="post_commit",
                instructions="Do things",
                actions=["gc"],
                condition=lambda ctx: False,  # always skip
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            # LLM should NOT have been called
            assert len(mock.calls) == 0


# ---------------------------------------------------------------------------
# 9. Fail-open on LLM error
# ---------------------------------------------------------------------------

class TestMaintainerFailOpen:
    def test_llm_exception_skips_gracefully(self):
        """When LLM raises, maintainer skips without crashing."""
        with Tract.open() as t:
            _seed_commits(t, n=1)

            error_client = ErrorLLMClient()
            t.config.configure_llm(error_client)

            t.middleware.maintain(
                "fragile",
                event="post_commit",
                instructions="Do maintenance",
                actions=["gc"],
            )

            # Should NOT raise
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

    def test_fail_open_emits_log_warning(self, caplog):
        """Fail-open on LLM error emits a log warning."""
        import logging

        with Tract.open() as t:
            _seed_commits(t, n=1)

            error_client = ErrorLLMClient()
            t.config.configure_llm(error_client)

            t.middleware.maintain(
                "warn-m",
                event="post_commit",
                instructions="Do things",
                actions=["gc"],
            )

            with caplog.at_level(logging.WARNING, logger="tract.maintain"):
                t.commit(
                    {"content_type": "dialogue", "role": "user", "text": "trigger"},
                    message="trigger",
                )

            assert any("fail-open" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 10. Multiple maintainers on same event
# ---------------------------------------------------------------------------

class TestMultipleMaintainers:
    def test_multiple_maintainers_all_fire(self):
        """When multiple maintainers are on the same event, all execute."""
        with Tract.open() as t:
            _seed_commits(t, n=1)

            # Both return no-op responses
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            t.middleware.maintain("m-alpha", event="post_commit", instructions="a", actions=["gc"])
            t.middleware.maintain("m-beta", event="post_commit", instructions="b", actions=["gc"])

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            # Both maintainers should have been called
            assert len(mock.calls) == 2

    def test_list_maintainers_returns_all(self):
        """list_maintainers() returns names of all registered maintainers."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            t.middleware.maintain("alpha", event="post_commit", instructions="a", actions=["gc"])
            t.middleware.maintain("beta", event="post_transition", instructions="b", actions=["gc"])
            t.middleware.maintain("gamma", event="pre_compile", instructions="c", actions=["annotate"])

            names = t.middleware.list_maintainers()
            assert set(names) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# 11. Stale _maintainers cleanup via remove_middleware()
# ---------------------------------------------------------------------------

class TestStaleMaintainersCleanup:
    def test_remove_middleware_cleans_maintainers_dict(self):
        """Calling remove_middleware() directly on a maintainer's handler ID
        also removes it from the _maintainers dict."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            handler_id = t.middleware.maintain(
                "my-m", event="post_commit", instructions="x", actions=["gc"],
            )
            assert "my-m" in t.middleware.list_maintainers()

            # Remove via remove_middleware (not remove_maintainer)
            t.middleware.remove(handler_id)

            # _maintainers should be cleaned up
            assert "my-m" not in t.middleware.list_maintainers()

    def test_reregister_after_direct_removal(self):
        """After remove_middleware() removes a maintainer, re-registering
        the same name should succeed."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            handler_id = t.middleware.maintain(
                "reuse-me", event="post_commit", instructions="v1", actions=["gc"],
            )
            t.middleware.remove(handler_id)

            # Should not raise "already registered"
            t.middleware.maintain("reuse-me", event="post_commit", instructions="v2", actions=["gc"])
            assert "reuse-me" in t.middleware.list_maintainers()


# ---------------------------------------------------------------------------
# 12. Invalid event and default actions
# ---------------------------------------------------------------------------

class TestMaintainerEdgeCases:
    def test_invalid_event_raises_value_error(self):
        """t.middleware.maintain() with a bad event string raises ValueError."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            with pytest.raises(ValueError, match="Unknown middleware event"):
                t.middleware.maintain(
                    "bad",
                    event="not_a_real_event",  # type: ignore[arg-type]
                    instructions="test",
                    actions=["gc"],
                )

    def test_maintain_return_value_is_handler_id(self):
        """t.middleware.maintain() returns a handler ID string."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            handler_id = t.middleware.maintain(
                "test-m", event="post_commit", instructions="x", actions=["gc"],
            )
            assert isinstance(handler_id, str)
            assert len(handler_id) == 12  # uuid hex prefix

    def test_none_actions_defaults_to_all(self):
        """When actions=None, all valid action types are allowed."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "all-actions",
                event="post_commit",
                instructions="Do everything",
                actions=None,  # should default to all
            )

            # The handler should have all actions
            # We can verify by checking the middleware was registered
            assert "all-actions" in t.middleware.list_maintainers()

    def test_invalid_action_type_raises(self):
        """Passing an invalid action type raises ValueError."""
        with Tract.open() as t:
            mock = MockLLMClient(_noop_response())
            t.config.configure_llm(mock)

            with pytest.raises(ValueError, match="Invalid action types"):
                t.middleware.maintain(
                    "bad-actions",
                    event="post_commit",
                    instructions="x",
                    actions=["gc", "destroy_everything"],
                )


# ---------------------------------------------------------------------------
# 13. Coexistence of gates and maintainers
# ---------------------------------------------------------------------------

class TestGateAndMaintainerCoexistence:
    def test_gate_and_maintainer_on_different_events(self):
        """A gate and a maintainer can coexist on different events."""
        with Tract.open() as t:
            mock = MockLLMClient(json.dumps({"result": "pass", "reason": "ok"}))
            t.config.configure_llm(mock)

            t.middleware.gate("quality", event="pre_commit", check="Content is good")
            t.middleware.maintain("cleanup", event="post_commit", instructions="Clean up", actions=["gc"])

            assert "quality" in t.middleware.list_gates()
            assert "cleanup" in t.middleware.list_maintainers()

            # Commit should trigger both (gate then maintainer)
            # Override mock to return appropriate responses
            call_count = [0]
            original_response = mock.response_text

            class DualMockClient:
                def __init__(self):
                    self.calls = []

                def chat(self, messages, **kwargs):
                    self.calls.append((messages, kwargs))
                    # First call = gate (pass), second call = maintainer (noop)
                    if len(self.calls) == 1:
                        text = json.dumps({"result": "pass", "reason": "ok"})
                    else:
                        text = _noop_response()
                    return {"choices": [{"message": {"content": text}}]}

                def extract_content(self, response):
                    return response["choices"][0]["message"]["content"]

                def extract_usage(self, response):
                    return {"total_tokens": 50}

                def close(self):
                    pass

            dual = DualMockClient()
            t.config.configure_llm(dual)

            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "test"},
                message="test",
            )

            # Both should have fired
            assert len(dual.calls) == 2


# ---------------------------------------------------------------------------
# 14. Block action through Tract API
# ---------------------------------------------------------------------------

class TestMaintainerBlockAction:
    def test_block_action_raises_blocked_error_through_tract(self):
        """A maintainer with block action raises BlockedError through t.commit()."""
        from tract.exceptions import BlockedError

        with Tract.open() as t:
            _seed_commits(t, n=1)

            response = _action_response("Blocking", [
                {"type": "block", "reason": "Context is in bad state"},
            ])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "blocker",
                event="post_commit",
                instructions="Block if bad state",
                actions=["block"],
            )

            with pytest.raises(BlockedError, match="bad state"):
                t.commit(
                    {"content_type": "dialogue", "role": "user", "text": "trigger"},
                    message="trigger",
                )

    def test_block_after_maintenance(self):
        """Non-block actions execute before the block raises."""
        from tract.exceptions import BlockedError

        with Tract.open() as t:
            _seed_commits(t, n=1)

            response = _action_response("Configure then block", [
                {"type": "configure", "key": "stage", "value": "blocked"},
                {"type": "block", "reason": "Pausing after config"},
            ])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "config-then-block",
                event="post_commit",
                instructions="Configure then block",
                actions=["configure", "block"],
            )

            with pytest.raises(BlockedError, match="Pausing"):
                t.commit(
                    {"content_type": "dialogue", "role": "user", "text": "trigger"},
                    message="trigger",
                )

            # The configure action should have executed before the block
            assert t.config.get("stage") == "blocked"


# ---------------------------------------------------------------------------
# 15. Peeking through Tract API
# ---------------------------------------------------------------------------

class SequenceMockClient:
    """Returns different responses on successive calls."""

    def __init__(self, responses: list[str]):
        self.responses = responses
        self.call_count = 0
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs) -> dict:
        self.calls.append((messages, kwargs))
        idx = min(self.call_count, len(self.responses) - 1)
        text = self.responses[idx]
        self.call_count += 1
        return {"choices": [{"message": {"content": text}}]}

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict:
        return {"total_tokens": 100}

    def close(self) -> None:
        pass


class TestMaintainerPeeking:
    def test_peeking_through_tract_api(self):
        """t.middleware.maintain() with max_peeks passes through to handler."""
        with Tract.open() as t:
            _seed_commits(t, n=2)

            # LLM skips peeking and returns actions directly
            response = _action_response("No peek needed", [])
            mock = MockLLMClient(response)
            t.config.configure_llm(mock)

            t.middleware.maintain(
                "peek-maintainer",
                event="post_commit",
                instructions="Check content quality",
                actions=["annotate"],
                max_peeks=3,
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            # LLM was called (first pass -- peek selection)
            assert len(mock.calls) == 1
            # The system prompt should reference peeking
            system_msg = mock.calls[0][0][0]["content"]
            assert "peek" in system_msg.lower()

    def test_peeking_fetches_real_content(self):
        """Peeking retrieves actual commit content from the tract."""
        with Tract.open() as t:
            hashes = _seed_commits(t, n=2)
            target_prefix = hashes[0][:8]

            # Pass 1: request to peek at the first commit
            # Pass 2: return no actions after peeking
            pass1_resp = json.dumps({"peek": [target_prefix]})
            pass2_resp = _action_response("Reviewed content", [])

            client = SequenceMockClient([pass1_resp, pass2_resp])
            t.config.configure_llm(client)

            t.middleware.maintain(
                "content-peek",
                event="post_commit",
                instructions="Inspect commit content",
                actions=["annotate"],
                max_peeks=3,
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            # Two LLM calls: peek + action
            assert client.call_count == 2

            # The second call should contain the peeked content
            pass2_messages = client.calls[1][0]
            user_msg = pass2_messages[1]["content"]
            assert "PEEKED COMMIT CONTENTS" in user_msg
            assert target_prefix in user_msg
            # The actual content should be present (Finding 1)
            assert "Finding 1" in user_msg

    def test_peeking_caps_at_max_peeks(self):
        """Peeking caps at max_peeks even if LLM requests more."""
        with Tract.open() as t:
            hashes = _seed_commits(t, n=5)
            all_prefixes = [h[:8] for h in hashes]

            # Pass 1: request to peek at all 5 commits (but max_peeks=2)
            pass1_resp = json.dumps({"peek": all_prefixes})
            pass2_resp = _action_response("Done", [])

            client = SequenceMockClient([pass1_resp, pass2_resp])
            t.config.configure_llm(client)

            t.middleware.maintain(
                "capped-peek",
                event="post_commit",
                instructions="x",
                actions=["annotate"],
                max_peeks=2,
            )

            # Trigger
            t.commit(
                {"content_type": "dialogue", "role": "user", "text": "trigger"},
                message="trigger",
            )

            # Second call should only contain 2 peeked commits
            pass2_messages = client.calls[1][0]
            user_msg = pass2_messages[1]["content"]
            # Count occurrences of the delimiter pattern
            peek_sections = user_msg.count("--- [")
            assert peek_sections == 2
