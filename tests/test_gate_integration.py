"""Integration tests for SemanticGate through the Tract API.

Tests gate registration, removal, listing, blocking, and pass-through
using ``t.gate()``, ``t.remove_gate()``, ``t.list_gates()`` with a
mock LLM client (no real LLM calls).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import BlockedError, GateResult, SemanticGate, Tract


# ---------------------------------------------------------------------------
# Mock LLM client
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

def _seed_commits(t: Tract, n: int = 3, tag: str = "research") -> None:
    """Add *n* commits tagged with *tag*."""
    t.register_tag(tag, f"Auto-registered for test")
    for i in range(n):
        t.commit(
            {"content_type": "dialogue", "role": "user", "text": f"Finding {i + 1}"},
            tags=[tag],
            message=f"research finding {i + 1}",
        )


def _fail_response(reason: str = "Criterion not met") -> str:
    return json.dumps({"result": "fail", "reason": reason})


def _pass_response(reason: str = "Criterion met") -> str:
    return json.dumps({"result": "pass", "reason": reason})


# ---------------------------------------------------------------------------
# 1. Registration and removal
# ---------------------------------------------------------------------------

class TestGateRegistrationAndRemoval:
    def test_register_list_remove(self):
        """Register a gate via t.gate(), verify in list_gates(), remove, verify gone."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            assert t.list_gates() == []

            t.gate(
                "quality-check",
                event="pre_transition",
                check="At least 3 commits tagged 'research'",
            )

            assert "quality-check" in t.list_gates()
            assert len(t.list_gates()) == 1

            t.remove_gate("quality-check")

            assert t.list_gates() == []

    def test_remove_nonexistent_gate_raises(self):
        """Removing a gate that doesn't exist raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError, match="not found"):
                t.remove_gate("no-such-gate")


# ---------------------------------------------------------------------------
# 2. Duplicate name rejected
# ---------------------------------------------------------------------------

class TestGateDuplicateNameRejected:
    def test_duplicate_name_raises_value_error(self):
        """Registering two gates with the same name raises ValueError."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            t.gate("unique-gate", event="pre_transition", check="check A")

            with pytest.raises(ValueError, match="already registered"):
                t.gate("unique-gate", event="pre_commit", check="check B")

    def test_reregister_after_removal_succeeds(self):
        """After removing a gate, the same name can be reused."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            t.gate("reusable", event="pre_transition", check="check A")
            t.remove_gate("reusable")
            # Should not raise
            t.gate("reusable", event="pre_commit", check="check B")
            assert "reusable" in t.list_gates()


# ---------------------------------------------------------------------------
# 3. Gate blocks transition (LLM returns FAIL)
# ---------------------------------------------------------------------------

class TestGateBlocksTransition:
    def test_fail_response_blocks_transition(self):
        """When LLM returns FAIL, transition raises BlockedError."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Not enough research commits"))
            t.configure_llm(mock)

            # Seed some commits
            _seed_commits(t, n=2, tag="research")

            # Create the target branch before gating
            t.branch("synthesis", switch=False)

            # Register the gate
            t.gate(
                "research-complete",
                event="pre_transition",
                check="At least 3 commits tagged 'research'",
            )

            with pytest.raises(BlockedError) as exc_info:
                t.transition("synthesis")

            assert exc_info.value.event == "pre_transition"
            assert "research-complete" in str(exc_info.value)

            # LLM was actually called
            assert len(mock.calls) == 1

    def test_blocked_error_contains_gate_name(self):
        """BlockedError reasons include the gate name for debuggability."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Missing analysis"))
            t.configure_llm(mock)

            _seed_commits(t, n=1)
            t.branch("next-stage", switch=False)

            t.gate(
                "analysis-gate",
                event="pre_transition",
                check="Must contain analysis",
            )

            with pytest.raises(BlockedError) as exc_info:
                t.transition("next-stage")

            assert "analysis-gate" in str(exc_info.value)


# ---------------------------------------------------------------------------
# 4. Gate passes transition (LLM returns PASS)
# ---------------------------------------------------------------------------

class TestGatePassesTransition:
    def test_pass_response_allows_transition(self):
        """When LLM returns PASS, transition completes successfully."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response("All criteria satisfied"))
            t.configure_llm(mock)

            _seed_commits(t, n=3, tag="research")

            # Create the target branch
            t.branch("synthesis", switch=False)

            t.gate(
                "research-complete",
                event="pre_transition",
                check="At least 3 commits tagged 'research'",
            )

            # Should NOT raise
            t.transition("synthesis")

            # Verify we actually transitioned
            assert t.current_branch == "synthesis"

            # LLM was called
            assert len(mock.calls) == 1

    def test_manifest_includes_commits(self):
        """The LLM receives a manifest that includes commit metadata."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            _seed_commits(t, n=2, tag="finding")
            t.branch("analysis", switch=False)

            t.gate("check", event="pre_transition", check="Has findings")
            t.transition("analysis")

            # Inspect what the LLM was sent
            messages, _kwargs = mock.calls[0]
            user_msg = messages[1]["content"]
            assert "CONTEXT MANIFEST" in user_msg
            assert "finding" in user_msg  # tag should appear


# ---------------------------------------------------------------------------
# 5. Gate with condition — skips when condition is False
# ---------------------------------------------------------------------------

class TestGateWithCondition:
    def test_condition_false_skips_gate(self):
        """Gate does not fire when condition returns False for the target."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Would block if called"))
            t.configure_llm(mock)

            _seed_commits(t, n=1)
            t.branch("other_branch", switch=False)

            # Gate only fires when target is "synthesis"
            t.gate(
                "synthesis-gate",
                event="pre_transition",
                check="Requires deep research",
                condition=lambda ctx: ctx.target == "synthesis",
            )

            # Transition to "other_branch" — condition returns False, gate skips
            t.transition("other_branch")
            assert t.current_branch == "other_branch"

            # LLM should NOT have been called
            assert len(mock.calls) == 0

    def test_condition_true_fires_gate(self):
        """Gate fires when condition returns True for the target."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Blocked"))
            t.configure_llm(mock)

            _seed_commits(t, n=1)
            t.branch("synthesis", switch=False)

            t.gate(
                "synthesis-gate",
                event="pre_transition",
                check="Requires deep research",
                condition=lambda ctx: ctx.target == "synthesis",
            )

            # Transition to "synthesis" — condition returns True, gate fires
            with pytest.raises(BlockedError):
                t.transition("synthesis")

            # LLM WAS called
            assert len(mock.calls) == 1

    def test_condition_mixed_scenario(self):
        """Same gate: skips for one branch, blocks for another."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Not ready"))
            t.configure_llm(mock)

            _seed_commits(t, n=1)
            t.branch("drafts", switch=False)
            t.branch("synthesis", switch=False)

            t.gate(
                "synth-only",
                event="pre_transition",
                check="Needs at least 5 findings",
                condition=lambda ctx: ctx.target == "synthesis",
            )

            # Drafts: passes (condition=False, gate skipped)
            t.transition("drafts")
            assert t.current_branch == "drafts"
            assert len(mock.calls) == 0

            # Switch back
            t.switch("main")

            # Synthesis: blocks (condition=True, LLM returns FAIL)
            with pytest.raises(BlockedError):
                t.transition("synthesis")
            assert len(mock.calls) == 1


# ---------------------------------------------------------------------------
# 6. Gate fail-open on LLM error
# ---------------------------------------------------------------------------

class TestGateFailOpenOnLLMError:
    def test_llm_exception_allows_transition(self):
        """When LLM raises, gate fails open and transition succeeds."""
        with Tract.open() as t:
            error_client = ErrorLLMClient()
            t.configure_llm(error_client)

            _seed_commits(t, n=2)
            t.branch("next", switch=False)

            t.gate(
                "fragile-gate",
                event="pre_transition",
                check="Some criterion",
            )

            # Should NOT raise — fail-open behavior
            t.transition("next")
            assert t.current_branch == "next"

    def test_fail_open_emits_log_warning(self, caplog):
        """Fail-open on LLM error emits a log warning."""
        import logging

        with Tract.open() as t:
            error_client = ErrorLLMClient()
            t.configure_llm(error_client)

            _seed_commits(t, n=1)
            t.branch("dest", switch=False)

            t.gate("warn-gate", event="pre_transition", check="criterion")

            with caplog.at_level(logging.WARNING, logger="tract.gate"):
                t.transition("dest")

            assert any("fail-open" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. Gate on pre_commit
# ---------------------------------------------------------------------------

class TestGateOnPreCommit:
    def test_pre_commit_gate_blocks_commit(self):
        """A gate on pre_commit blocks t.commit() when LLM returns FAIL."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Content does not meet standards"))
            t.configure_llm(mock)

            t.gate(
                "content-quality",
                event="pre_commit",
                check="Content must be substantive and well-structured",
            )

            with pytest.raises(BlockedError) as exc_info:
                t.commit(
                    {"content_type": "dialogue", "role": "user", "text": "Hello"},
                    message="test commit",
                )

            assert exc_info.value.event == "pre_commit"
            assert "content-quality" in str(exc_info.value)
            assert len(mock.calls) == 1

    def test_pre_commit_gate_allows_after_removal(self):
        """After removing a pre_commit gate, commits succeed normally."""
        with Tract.open() as t:
            mock = MockLLMClient(_fail_response("Blocked"))
            t.configure_llm(mock)

            t.gate(
                "blocker",
                event="pre_commit",
                check="Must be formatted correctly",
            )

            # Blocked
            with pytest.raises(BlockedError):
                t.commit(
                    {"content_type": "dialogue", "role": "user", "text": "test"},
                    message="attempt 1",
                )

            # Remove the gate
            t.remove_gate("blocker")
            assert t.list_gates() == []

            # Now commit should succeed (no gate to block)
            info = t.commit(
                {"content_type": "dialogue", "role": "user", "text": "test"},
                message="attempt 2",
            )
            assert info is not None
            assert info.message == "attempt 2"

    def test_pre_commit_gate_passes_allows_commit(self):
        """A pre_commit gate that returns PASS allows the commit through."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response("Content looks good"))
            t.configure_llm(mock)

            t.gate(
                "quality-pass",
                event="pre_commit",
                check="Content must be substantive",
            )

            info = t.commit(
                {"content_type": "dialogue", "role": "user", "text": "Detailed analysis of findings"},
                message="good commit",
            )
            assert info is not None
            assert len(mock.calls) == 1


# ---------------------------------------------------------------------------
# Multiple gates on same event
# ---------------------------------------------------------------------------

class TestMultipleGates:
    def test_multiple_gates_all_must_pass(self):
        """When multiple gates are on the same event, all must pass."""

        class SequenceMockClient:
            """Returns PASS on first call, FAIL on second."""

            def __init__(self):
                self.calls: list[tuple[list[dict], dict]] = []

            def chat(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                if len(self.calls) == 1:
                    return {"choices": [{"message": {"content": _pass_response()}}]}
                return {"choices": [{"message": {"content": _fail_response("Check B failed")}}]}

            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]

            def extract_usage(self, response):
                return {"total_tokens": 50}

            def close(self):
                pass

        with Tract.open() as t:
            seq_mock = SequenceMockClient()
            t.configure_llm(seq_mock)

            _seed_commits(t, n=2)
            t.branch("target", switch=False)

            t.gate("gate-a", event="pre_transition", check="Check A")
            t.gate("gate-b", event="pre_transition", check="Check B")

            # Gate-a passes, gate-b fails
            with pytest.raises(BlockedError) as exc_info:
                t.transition("target")

            assert "gate-b" in str(exc_info.value)
            assert len(seq_mock.calls) == 2

    def test_list_gates_returns_all(self):
        """list_gates() returns names of all registered gates."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            t.gate("alpha", event="pre_commit", check="a")
            t.gate("beta", event="pre_transition", check="b")
            t.gate("gamma", event="pre_compile", check="c")

            names = t.list_gates()
            assert set(names) == {"alpha", "beta", "gamma"}


# ---------------------------------------------------------------------------
# Stale _gates cleanup when remove_middleware() is called directly
# ---------------------------------------------------------------------------

class TestStaleGatesCleanup:
    def test_remove_middleware_cleans_gates_dict(self):
        """Calling remove_middleware() directly on a gate's handler ID also
        removes it from the _gates dict."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            handler_id = t.gate("my-gate", event="pre_commit", check="test")
            assert "my-gate" in t.list_gates()

            # Remove via remove_middleware (not remove_gate)
            t.remove_middleware(handler_id)

            # _gates should be cleaned up too
            assert "my-gate" not in t.list_gates()

    def test_reregister_after_direct_removal(self):
        """After remove_middleware() removes a gate, re-registering the
        same name should succeed."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            handler_id = t.gate("reuse-me", event="pre_commit", check="v1")
            t.remove_middleware(handler_id)

            # Should not raise "already registered"
            t.gate("reuse-me", event="pre_commit", check="v2")
            assert "reuse-me" in t.list_gates()


# ---------------------------------------------------------------------------
# Invalid event via t.gate()
# ---------------------------------------------------------------------------

class TestGateInvalidEvent:
    def test_invalid_event_raises_value_error(self):
        """t.gate() with a bad event string raises ValueError."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            with pytest.raises(ValueError, match="Unknown middleware event"):
                t.gate("bad", event="not_a_real_event", check="test")  # type: ignore[arg-type]

    def test_gate_return_value_is_handler_id(self):
        """t.gate() returns a handler ID string usable with remove_middleware()."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            handler_id = t.gate("test-gate", event="pre_commit", check="test")
            assert isinstance(handler_id, str)
            assert len(handler_id) == 12  # uuid hex prefix


# ---------------------------------------------------------------------------
# Fail-then-pass lifecycle
# ---------------------------------------------------------------------------

class TestGateFailThenPass:
    def test_gate_blocks_then_passes_after_more_work(self):
        """Gate blocks initial transition, then passes after more commits."""

        class ToggleMockClient:
            """Returns FAIL on first call, PASS on second."""

            def __init__(self):
                self.call_count = 0

            def chat(self, messages, **kwargs):
                self.call_count += 1
                if self.call_count == 1:
                    text = _fail_response("Not enough research depth")
                else:
                    text = _pass_response("Research now has sufficient depth")
                return {"choices": [{"message": {"content": text}}]}

            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]

            def extract_usage(self, response):
                return {"total_tokens": 80}

            def close(self):
                pass

        with Tract.open() as t:
            toggle = ToggleMockClient()
            t.configure_llm(toggle)

            _seed_commits(t, n=2)
            t.branch("synthesis", switch=False)

            t.gate(
                "depth-check",
                event="pre_transition",
                check="Research must have sufficient depth",
            )

            # First attempt: blocked
            with pytest.raises(BlockedError):
                t.transition("synthesis")
            assert toggle.call_count == 1

            # Add more work
            _seed_commits(t, n=3, tag="extra")

            # Second attempt: passes
            t.transition("synthesis")
            assert t.current_branch == "synthesis"
            assert toggle.call_count == 2


# ---------------------------------------------------------------------------
# Parse response edge cases (null reason)
# ---------------------------------------------------------------------------

class TestParseResponseNullReason:
    def test_null_reason_in_json(self):
        """JSON response with null reason should produce '(no reason given)'."""
        passed, reason = SemanticGate._parse_response(
            '{"result": "pass", "reason": null}'
        )
        assert passed is True
        assert reason == "(no reason given)"

    def test_substring_keyword_no_false_match(self):
        """Words like 'passive' or 'rainfall' should not trigger keyword fallback."""
        # "passive" contains "pass" as substring but not as word
        passed, reason = SemanticGate._parse_response("This is about passive income")
        # Should be ambiguous (no word-boundary match), defaulting to pass
        assert passed is True
        assert "defaulting to pass" in reason.lower() or "passive" in reason.lower()


# ---------------------------------------------------------------------------
# Post_* event validation
# ---------------------------------------------------------------------------

class TestGatePostEventValidation:
    """Gates block via BlockedError. Post_* events fire after the operation
    is already complete, making blocking misleading.  t.gate() should reject
    post_* events at registration time."""

    @pytest.mark.parametrize("event", [
        "post_commit",
        "post_transition",
        "post_generate",
        "post_tool_execute",
    ])
    def test_post_event_raises_value_error(self, event):
        """Registering a gate on a post_* event raises ValueError."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            with pytest.raises(ValueError, match="post_\\*"):
                t.gate("bad-gate", event=event, check="test")  # type: ignore[arg-type]

    @pytest.mark.parametrize("event", [
        "pre_commit",
        "pre_compile",
        "pre_compress",
        "pre_merge",
        "pre_gc",
        "pre_transition",
        "pre_generate",
        "pre_tool_execute",
    ])
    def test_pre_events_are_allowed(self, event):
        """All pre_* events should be valid for gate registration."""
        with Tract.open() as t:
            mock = MockLLMClient(_pass_response())
            t.configure_llm(mock)

            handler_id = t.gate(f"gate-{event}", event=event, check="test")  # type: ignore[arg-type]
            assert isinstance(handler_id, str)
            assert f"gate-{event}" in t.list_gates()
