"""Tests for orchestrator config, models, hook templates, and prompts."""

from __future__ import annotations

import dataclasses
import logging

import pytest

from tract.exceptions import OrchestratorError, TraceError
from tract.orchestrator import (
    AutonomyLevel,
    OrchestratorConfig,
    OrchestratorResult,
    OrchestratorState,
    StepResult,
    ToolCall,
    ToolCallDecision,
    ToolCallReview,
    TriggerConfig,
    auto_approve_tool_call,
    log_and_approve_tool_call,
    reject_all_tool_call,
)
from tract.prompts.orchestrator import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_assessment_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_tool_call() -> ToolCall:
    """A simple ToolCall for use in review tests."""
    return ToolCall(id="tc-1", name="compress", arguments={"threshold": 0.8})


# ---------------------------------------------------------------------------
# Enum value tests
# ---------------------------------------------------------------------------


class TestAutonomyLevel:
    def test_autonomy_level_values(self) -> None:
        assert AutonomyLevel.MANUAL == "manual"
        assert AutonomyLevel.COLLABORATIVE == "collaborative"
        assert AutonomyLevel.AUTONOMOUS == "autonomous"

    def test_autonomy_level_is_str(self) -> None:
        assert isinstance(AutonomyLevel.MANUAL, str)


class TestOrchestratorState:
    def test_orchestrator_state_values(self) -> None:
        assert OrchestratorState.IDLE == "idle"
        assert OrchestratorState.RUNNING == "running"
        assert OrchestratorState.PAUSING == "pausing"
        assert OrchestratorState.STOPPED == "stopped"


class TestToolCallDecision:
    def test_tool_call_decision_enum_values(self) -> None:
        assert ToolCallDecision.APPROVED == "approved"
        assert ToolCallDecision.REJECTED == "rejected"
        assert ToolCallDecision.MODIFIED == "modified"


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestOrchestratorConfig:
    def test_orchestrator_config_defaults(self) -> None:
        config = OrchestratorConfig()
        assert config.autonomy_ceiling == AutonomyLevel.COLLABORATIVE
        assert config.max_steps == 10
        assert config.profile == "self"
        assert config.temperature == 0.0
        assert config.system_prompt is None
        assert config.task_context is None
        assert config.triggers is None
        assert config.model is None
        assert config.on_tool_call is None
        assert config.on_step is None

    def test_orchestrator_config_mutable(self) -> None:
        config = OrchestratorConfig()
        config.autonomy_ceiling = AutonomyLevel.AUTONOMOUS
        assert config.autonomy_ceiling == AutonomyLevel.AUTONOMOUS


class TestTriggerConfig:
    def test_trigger_config_defaults(self) -> None:
        tc = TriggerConfig()
        assert tc.on_commit_count is None
        assert tc.on_token_threshold is None
        assert tc.on_compile is False

    def test_trigger_config_custom(self) -> None:
        tc = TriggerConfig(on_commit_count=5, on_token_threshold=0.8)
        assert tc.on_commit_count == 5
        assert tc.on_token_threshold == 0.8
        assert tc.on_compile is False

    def test_trigger_config_frozen(self) -> None:
        tc = TriggerConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.on_compile = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_tool_call_creation(self) -> None:
        tc = ToolCall(id="tc-1", name="compress", arguments={"k": "v"})
        assert tc.id == "tc-1"
        assert tc.name == "compress"
        assert tc.arguments == {"k": "v"}

    def test_tool_call_frozen(self) -> None:
        tc = ToolCall(id="tc-1", name="compress")
        with pytest.raises(dataclasses.FrozenInstanceError):
            tc.name = "other"  # type: ignore[misc]

    def test_tool_call_default_arguments(self) -> None:
        tc = ToolCall(id="tc-1", name="pin")
        assert tc.arguments == {}


class TestToolCallReview:
    def test_tool_call_review_frozen(self) -> None:
        resp = ToolCallReview(decision=ToolCallDecision.APPROVED)
        with pytest.raises(dataclasses.FrozenInstanceError):
            resp.decision = ToolCallDecision.REJECTED  # type: ignore[misc]

    def test_tool_call_review_defaults(self) -> None:
        resp = ToolCallReview(decision=ToolCallDecision.REJECTED)
        assert resp.modified_action is None
        assert resp.reason == ""


class TestStepResult:
    def test_step_result_frozen(self) -> None:
        tc = ToolCall(id="tc-1", name="compress")
        sr = StepResult(step=1, tool_call=tc)
        with pytest.raises(dataclasses.FrozenInstanceError):
            sr.step = 2  # type: ignore[misc]

    def test_step_result_defaults(self) -> None:
        tc = ToolCall(id="tc-1", name="compress")
        sr = StepResult(step=1, tool_call=tc)
        assert sr.result_output == ""
        assert sr.result_error == ""
        assert sr.success is True
        assert sr.review_decision == ""


class TestOrchestratorResult:
    def test_orchestrator_result_succeeded(self) -> None:
        tc1 = ToolCall(id="tc-1", name="compress")
        tc2 = ToolCall(id="tc-2", name="pin")
        tc3 = ToolCall(id="tc-3", name="branch")
        steps = [
            StepResult(step=1, tool_call=tc1, success=True),
            StepResult(step=2, tool_call=tc2, success=False, result_error="fail"),
            StepResult(step=3, tool_call=tc3, success=True),
        ]
        result = OrchestratorResult(
            steps=steps,
            state=OrchestratorState.STOPPED,
            total_tool_calls=3,
        )
        succeeded = result.succeeded
        assert len(succeeded) == 2
        assert succeeded[0].step == 1
        assert succeeded[1].step == 3

    def test_orchestrator_result_defaults(self) -> None:
        result = OrchestratorResult()
        assert result.steps == []
        assert result.state == OrchestratorState.IDLE
        assert result.assessment == ""
        assert result.total_tool_calls == 0

    def test_orchestrator_result_succeeded_empty(self) -> None:
        result = OrchestratorResult()
        assert result.succeeded == []


# ---------------------------------------------------------------------------
# Tool-call review callback tests
# ---------------------------------------------------------------------------


class TestToolCallReviewCallbacks:
    def test_auto_approve_tool_call_callback(
        self, sample_tool_call: ToolCall
    ) -> None:
        response = auto_approve_tool_call(sample_tool_call)
        assert response.decision == ToolCallDecision.APPROVED

    def test_log_and_approve_tool_call_callback(
        self, sample_tool_call: ToolCall, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="tract.hooks.templates.orchestrator"):
            response = log_and_approve_tool_call(sample_tool_call)
        assert response.decision == ToolCallDecision.APPROVED
        assert "compress" in caplog.text

    def test_reject_all_tool_call_callback(
        self, sample_tool_call: ToolCall
    ) -> None:
        response = reject_all_tool_call(sample_tool_call)
        assert response.decision == ToolCallDecision.REJECTED
        assert response.reason == "Auto-rejected"


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------


class TestOrchestratorSystemPrompt:
    def test_orchestrator_system_prompt_exists(self) -> None:
        assert isinstance(ORCHESTRATOR_SYSTEM_PROMPT, str)
        assert len(ORCHESTRATOR_SYSTEM_PROMPT) > 0

    def test_system_prompt_content(self) -> None:
        assert "context management assistant" in ORCHESTRATOR_SYSTEM_PROMPT
        assert "compression" in ORCHESTRATOR_SYSTEM_PROMPT.lower()
        assert "pinned" in ORCHESTRATOR_SYSTEM_PROMPT.lower()


class TestBuildAssessmentPrompt:
    def test_build_assessment_prompt(self) -> None:
        prompt = build_assessment_prompt(
            token_count=8000,
            max_tokens=10000,
            commit_count=15,
            branch_name="main",
            recent_commits=["commit A", "commit B"],
        )
        assert "8000/10000" in prompt
        assert "80%" in prompt
        assert "15 total" in prompt
        assert "main" in prompt
        assert "commit A" in prompt
        assert "commit B" in prompt
        assert "maintenance actions" in prompt

    def test_build_assessment_prompt_with_task_context(self) -> None:
        prompt = build_assessment_prompt(
            token_count=5000,
            max_tokens=10000,
            commit_count=10,
            branch_name="feature",
            recent_commits=["c1"],
            task_context="Building a REST API for user authentication",
        )
        assert "Building a REST API" in prompt
        assert "user authentication" in prompt

    def test_build_assessment_prompt_with_annotations(self) -> None:
        prompt = build_assessment_prompt(
            token_count=3000,
            max_tokens=10000,
            commit_count=8,
            branch_name="main",
            recent_commits=[],
            pinned_count=3,
            skip_count=2,
        )
        assert "3 pinned" in prompt
        assert "2 skipped" in prompt

    def test_build_assessment_prompt_with_branches(self) -> None:
        prompt = build_assessment_prompt(
            token_count=3000,
            max_tokens=10000,
            commit_count=8,
            branch_name="main",
            recent_commits=[],
            branch_count=4,
        )
        assert "4 total" in prompt
        assert "Branches: 4" in prompt

    def test_build_assessment_prompt_truncates_recent(self) -> None:
        commits = [f"commit {i}" for i in range(20)]
        prompt = build_assessment_prompt(
            token_count=1000,
            max_tokens=10000,
            commit_count=20,
            branch_name="main",
            recent_commits=commits,
        )
        assert "commit 9" in prompt
        assert "commit 10" not in prompt


# ---------------------------------------------------------------------------
# Exception tests
# ---------------------------------------------------------------------------


class TestOrchestratorError:
    def test_orchestrator_error_exception(self) -> None:
        assert issubclass(OrchestratorError, TraceError)

    def test_orchestrator_error_message(self) -> None:
        err = OrchestratorError("something went wrong")
        assert str(err) == "something went wrong"

    def test_orchestrator_error_catchable_as_trace_error(self) -> None:
        with pytest.raises(TraceError):
            raise OrchestratorError("test")
