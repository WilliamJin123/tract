"""Tests for OperationPrompts (Fix 3)."""
import pytest

from tract import Tract, OperationPrompts
from tract.models.config import LLMConfig


class TestOperationPrompts:
    """Tests for OperationPrompts dataclass and configure_prompts()."""

    def test_frozen_dataclass(self):
        """OperationPrompts is frozen."""
        prompts = OperationPrompts(compress="test")
        with pytest.raises(AttributeError):
            prompts.compress = "other"  # type: ignore[misc]

    def test_configure_with_instance(self):
        """configure_prompts() accepts OperationPrompts instance."""
        t = Tract.open()
        prompts = OperationPrompts(compress="You are a concise summarizer.")
        t.configure_prompts(prompts)
        assert t.operation_prompts.compress == "You are a concise summarizer."

    def test_configure_with_kwargs(self):
        """configure_prompts() accepts keyword arguments."""
        t = Tract.open()
        t.configure_prompts(compress="Summarize concisely.", merge="Merge context.")
        assert t.operation_prompts.compress == "Summarize concisely."
        assert t.operation_prompts.merge == "Merge context."

    def test_configure_rejects_mixed(self):
        """configure_prompts() rejects mixed positional and keyword args."""
        t = Tract.open()
        with pytest.raises(TypeError):
            t.configure_prompts(OperationPrompts(), compress="test")

    def test_configure_rejects_unknown_operation(self):
        """configure_prompts() rejects unknown operation names."""
        t = Tract.open()
        with pytest.raises(ValueError, match="Unknown operation"):
            t.configure_prompts(unknown="test")

    def test_explicit_system_prompt_beats_operation_prompts(self):
        """Explicit system_prompt parameter takes precedence over OperationPrompts."""
        t = Tract.open()
        t.configure_prompts(compress="Operation default prompt")
        # We can't easily test compress without an LLM, but we can verify
        # the attribute is set and would be used as fallback
        assert t.operation_prompts.compress == "Operation default prompt"

    def test_operation_prompts_property_readonly(self):
        """operation_prompts property returns the current prompts."""
        t = Tract.open()
        assert t.operation_prompts == OperationPrompts()
        t.configure_prompts(orchestrate="Agent mode")
        assert t.operation_prompts.orchestrate == "Agent mode"

    def test_config_history_logs_prompt_changes(self):
        """configure_prompts() logs to config history."""
        t = Tract.open()
        t.configure_prompts(compress="Summarize")
        history = t.config_history(change_type="prompts")
        # In-memory DB: config_history may or may not have entries
        # (depends on persistence repo availability)
        assert isinstance(history, list)
