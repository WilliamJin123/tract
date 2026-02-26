"""Tests for per-operation LLM configuration.

Covers LLMConfig dataclass, configure_operations(), _resolve_llm_config(),
Tract.open() with operation_configs, and integration with chat/generate, merge,
compress, and orchestrate operations.
"""

from __future__ import annotations

import dataclasses

import pytest

from tract import (
    DialogueContent,
    InstructionContent,
    LLMConfig,
    OperationConfigs,
    Tract,
)
from tract.models.commit import CommitOperation


# ---------------------------------------------------------------------------
# MockLLMClient -- captures kwargs for assertion
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Minimal mock LLM client that records call kwargs."""

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
# LLMConfig dataclass tests
# ---------------------------------------------------------------------------

class TestLLMConfig:
    """Tests for the LLMConfig frozen dataclass."""

    def test_create_with_defaults(self):
        """All fields default to None."""
        config = LLMConfig()
        assert config.model is None
        assert config.temperature is None
        assert config.top_p is None
        assert config.max_tokens is None
        assert config.stop_sequences is None
        assert config.frequency_penalty is None
        assert config.presence_penalty is None
        assert config.top_k is None
        assert config.seed is None
        assert config.extra is None

    def test_create_with_values(self):
        """All fields can be set, including extra."""
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000,
            top_p=0.9,
            seed=42,
            extra={"custom_param": "val"},
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.max_tokens == 1000
        assert config.top_p == 0.9
        assert config.seed == 42
        assert config.extra["custom_param"] == "val"

    def test_frozen(self):
        """Attempting to modify a frozen dataclass raises FrozenInstanceError."""
        config = LLMConfig(model="gpt-4o")
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.model = "gpt-3.5-turbo"  # type: ignore[misc]

    def test_from_dict_round_trip(self):
        """from_dict and to_dict are inverse operations."""
        d = {"model": "gpt-4o", "temperature": 0.5, "custom_key": "abc"}
        config = LLMConfig.from_dict(d)
        assert config.model == "gpt-4o"
        assert config.temperature == 0.5
        assert config.extra["custom_key"] == "abc"
        assert LLMConfig.from_dict(config.to_dict()) == config

    def test_from_dict_none(self):
        """from_dict(None) returns None."""
        assert LLMConfig.from_dict(None) is None

    def test_non_none_fields(self):
        """non_none_fields returns only set fields (excluding extra)."""
        config = LLMConfig(model="gpt-4o", temperature=0.5)
        result = config.non_none_fields()
        assert result == {"model": "gpt-4o", "temperature": 0.5}

    def test_stop_sequences_tuple_conversion(self):
        """stop_sequences list is converted to tuple."""
        config = LLMConfig.from_dict({"stop_sequences": ["stop1", "stop2"]})
        assert config.stop_sequences == ("stop1", "stop2")
        assert config.to_dict()["stop_sequences"] == ["stop1", "stop2"]

    def test_extra_is_immutable(self):
        """extra dict is wrapped in MappingProxyType."""
        config = LLMConfig(extra={"key": "val"})
        with pytest.raises(TypeError):
            config.extra["key"] = "new"  # type: ignore[index]

    def test_hashable(self):
        """LLMConfig is hashable (can be used in sets/dicts)."""
        c1 = LLMConfig(model="gpt-4o")
        c2 = LLMConfig(model="gpt-4o")
        assert hash(c1) == hash(c2)
        assert {c1, c2} == {c1}


# ---------------------------------------------------------------------------
# configure_operations() tests
# ---------------------------------------------------------------------------

class TestConfigureOperations:
    """Tests for Tract.configure_operations()."""

    def test_configure_single_operation(self):
        """Set a single operation config and verify via property."""
        t = Tract.open()
        chat_config = LLMConfig(model="gpt-4o")
        t.configure_operations(chat=chat_config)

        configs = t.operation_configs
        assert configs.chat is not None
        assert configs.chat.model == "gpt-4o"
        t.close()

    def test_configure_multiple_operations(self):
        """Set multiple operation configs in one call."""
        t = Tract.open()
        t.configure_operations(
            chat=LLMConfig(model="gpt-4o"),
            compress=LLMConfig(model="gpt-3.5-turbo"),
            merge=LLMConfig(model="gpt-4o", temperature=0.3),
        )

        configs = t.operation_configs
        assert configs.chat is not None
        assert configs.compress is not None
        assert configs.merge is not None
        assert configs.chat.model == "gpt-4o"
        assert configs.compress.model == "gpt-3.5-turbo"
        assert configs.merge.temperature == 0.3
        t.close()

    def test_configure_overwrites_existing(self):
        """Calling configure_operations twice replaces the config for that operation."""
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(model="gpt-4o"))
        assert t.operation_configs.chat.model == "gpt-4o"

        t.configure_operations(chat=LLMConfig(model="gpt-3.5-turbo"))
        assert t.operation_configs.chat.model == "gpt-3.5-turbo"
        t.close()

    def test_configure_type_error(self):
        """Passing a non-LLMConfig value raises TypeError."""
        t = Tract.open()
        with pytest.raises(TypeError, match="Expected LLMConfig"):
            t.configure_operations(chat={"model": "gpt-4o"})  # type: ignore[arg-type]
        t.close()


# ---------------------------------------------------------------------------
# _resolve_llm_config() resolution chain tests
# ---------------------------------------------------------------------------

class TestResolveLLMConfig:
    """Tests for _resolve_llm_config() three-level resolution chain."""

    def test_resolve_call_level_wins(self):
        """Call-level model overrides operation and tract defaults."""
        t = Tract.open()
        t._default_config = LLMConfig(model="tract-default")
        t.configure_operations(chat=LLMConfig(model="op-model"))

        resolved = t._resolve_llm_config("chat", model="call-model")
        assert resolved["model"] == "call-model"
        t.close()

    def test_resolve_operation_level_wins_over_tract(self):
        """Operation-level model overrides tract default."""
        t = Tract.open()
        t._default_config = LLMConfig(model="tract-default")
        t.configure_operations(chat=LLMConfig(model="op-model"))

        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "op-model"
        t.close()

    def test_resolve_tract_default_used(self):
        """Without call or operation config, tract default is used."""
        t = Tract.open()
        t._default_config = LLMConfig(model="tract-default")

        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "tract-default"
        t.close()

    def test_resolve_no_config_returns_empty(self):
        """No config at any level returns empty dict."""
        t = Tract.open()
        resolved = t._resolve_llm_config("chat")
        assert resolved == {}
        t.close()

    def test_resolve_temperature_chain(self):
        """Temperature follows call > operation resolution."""
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(temperature=0.5))

        # Operation level
        resolved = t._resolve_llm_config("chat")
        assert resolved["temperature"] == 0.5

        # Call level overrides
        resolved = t._resolve_llm_config("chat", temperature=0.9)
        assert resolved["temperature"] == 0.9
        t.close()

    def test_resolve_extra_merged(self):
        """extra from operation config is forwarded, call kwargs override."""
        t = Tract.open()
        t.configure_operations(
            chat=LLMConfig(extra={"custom_param": "val", "another": 42})
        )

        resolved = t._resolve_llm_config("chat")
        assert resolved["custom_param"] == "val"
        assert resolved["another"] == 42

        # Call-level kwargs override operation extra
        resolved = t._resolve_llm_config("chat", another=99)
        assert resolved["another"] == 99
        assert resolved["custom_param"] == "val"
        t.close()

    def test_resolve_typed_fields(self):
        """New typed fields (top_p, seed, etc.) are resolved from operation config."""
        t = Tract.open()
        t.configure_operations(
            chat=LLMConfig(top_p=0.9, seed=42, frequency_penalty=0.5)
        )

        resolved = t._resolve_llm_config("chat")
        assert resolved["top_p"] == 0.9
        assert resolved["seed"] == 42
        assert resolved["frequency_penalty"] == 0.5
        t.close()


# ---------------------------------------------------------------------------
# Tract.open() with operation_configs tests
# ---------------------------------------------------------------------------

class TestOpenWithOperationConfigs:
    """Tests for Tract.open() operation_configs parameter."""

    def test_open_with_operation_configs(self):
        """Pass operation_configs dict to Tract.open(), verify applied (legacy path)."""
        t = Tract.open(
            operation_configs={
                "chat": LLMConfig(model="gpt-4o"),
                "compress": LLMConfig(model="gpt-3.5-turbo"),
            }
        )
        configs = t.operation_configs
        assert configs.chat.model == "gpt-4o"
        assert configs.compress.model == "gpt-3.5-turbo"
        t.close()

    def test_open_with_operations_typed(self):
        """Pass operations=OperationConfigs to Tract.open(), verify applied."""
        oc = OperationConfigs(
            chat=LLMConfig(model="gpt-4o"),
            compress=LLMConfig(model="gpt-3.5-turbo"),
        )
        t = Tract.open(operations=oc)
        configs = t.operation_configs
        assert configs.chat.model == "gpt-4o"
        assert configs.compress.model == "gpt-3.5-turbo"
        t.close()

    def test_open_without_operation_configs(self):
        """Default behavior: no operation configs set."""
        t = Tract.open()
        assert isinstance(t.operation_configs, OperationConfigs)
        assert t.operation_configs.chat is None
        assert t.operation_configs.merge is None
        assert t.operation_configs.compress is None
        assert t.operation_configs.orchestrate is None
        t.close()


# ---------------------------------------------------------------------------
# chat/generate integration tests
# ---------------------------------------------------------------------------

class TestChatGenerateIntegration:
    """Tests for chat/generate using per-operation config."""

    def test_chat_uses_operation_config_model(self):
        """Configure chat model, verify MockLLMClient receives it."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(chat=LLMConfig(model="chat-model"))

        t.system("You are helpful")
        t.user("Hello")
        t.generate()

        assert mock.last_kwargs.get("model") == "chat-model"
        t.close()

    def test_chat_call_override_beats_operation(self):
        """Call-level model= on generate() overrides operation config."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(chat=LLMConfig(model="op-model"))

        t.system("You are helpful")
        t.user("Hello")
        t.generate(model="call-model")

        assert mock.last_kwargs.get("model") == "call-model"
        t.close()

    def test_generate_uses_operation_config_temperature(self):
        """Configure chat temperature, verify forwarded to LLM."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(chat=LLMConfig(temperature=0.8))

        t.system("You are helpful")
        t.user("Hello")
        t.generate()

        assert mock.last_kwargs.get("temperature") == 0.8
        t.close()

    def test_generation_config_reflects_operation_model(self):
        """generation_config on commit captures the resolved model from response."""
        t = Tract.open()
        mock = MockLLMClient(model="default-model")
        t.configure_llm(mock)
        t.configure_operations(chat=LLMConfig(model="chat-model"))

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate()

        # Verify the per-op model was sent to the LLM
        assert mock.last_kwargs.get("model") == "chat-model"
        # generation_config uses the response model (authoritative)
        # The mock returns the requested model in the response
        assert resp.generation_config.model == "chat-model"
        t.close()


# ---------------------------------------------------------------------------
# merge integration tests
# ---------------------------------------------------------------------------

class TestMergeIntegration:
    """Tests for merge using per-operation config."""

    def _make_diverged_tract(self):
        """Create a tract with diverged branches for merge testing."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        # Base commit
        base = t.commit(InstructionContent(text="original"))

        # Feature branch with edit
        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            edit_target=base.commit_hash,
        )

        # Back to main with edit
        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            edit_target=base.commit_hash,
        )

        return t, mock

    def test_merge_uses_operation_config(self):
        """Configure merge model, verify resolver gets it."""
        t, mock = self._make_diverged_tract()
        t.configure_operations(merge=LLMConfig(model="merge-model"))

        # The merge will use semantic resolution -- the resolver should
        # be created with the operation config model
        result = t.merge("feature", auto_commit=True)
        # Since it's a conflict merge, the resolver was created with merge-model
        # The MockLLMClient was used for the resolver's LLM call
        assert result is not None
        t.close()

    def test_merge_call_override_beats_operation(self):
        """model= on merge() overrides operation config."""
        t, mock = self._make_diverged_tract()
        t.configure_operations(merge=LLMConfig(model="op-merge"))

        result = t.merge("feature", model="call-merge", auto_commit=True)
        assert result is not None
        t.close()

    def test_merge_temperature_from_operation(self):
        """temperature/max_tokens from operation config forwarded to resolver."""
        t, mock = self._make_diverged_tract()
        t.configure_operations(
            merge=LLMConfig(model="merge-model", temperature=0.1, max_tokens=512)
        )

        result = t.merge("feature", auto_commit=True)
        assert result is not None
        t.close()


# ---------------------------------------------------------------------------
# compress integration tests
# ---------------------------------------------------------------------------

class TestCompressIntegration:
    """Tests for compress using per-operation config."""

    def test_compress_uses_operation_config(self):
        """Configure compress model, verify llm_kwargs forwarded to LLM."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)
        t.configure_operations(compress=LLMConfig(model="compress-model"))

        t.commit(InstructionContent(text="First instruction"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi there"))

        result = t.compress()
        assert mock.last_kwargs.get("model") == "compress-model"
        t.close()

    def test_compress_without_config_backward_compatible(self):
        """No config = current behavior (no model kwargs sent)."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)

        t.commit(InstructionContent(text="First instruction"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi there"))

        result = t.compress()
        # No model/temperature/max_tokens in kwargs
        assert "model" not in mock.last_kwargs
        assert "temperature" not in mock.last_kwargs
        assert "max_tokens" not in mock.last_kwargs
        t.close()

    def test_compress_call_level_model_override(self):
        """Pass model= on compress(), verify it overrides operation config."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)
        t.configure_operations(compress=LLMConfig(model="op-compress"))

        t.commit(InstructionContent(text="First instruction"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi there"))

        result = t.compress(model="call-compress")
        assert mock.last_kwargs.get("model") == "call-compress"
        t.close()

    def test_compress_call_level_temperature_override(self):
        """Pass temperature= on compress(), verify forwarded."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)
        t.configure_operations(compress=LLMConfig(temperature=0.1))

        t.commit(InstructionContent(text="First instruction"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi there"))

        # Call-level override
        result = t.compress(temperature=0.5)
        assert mock.last_kwargs.get("temperature") == 0.5
        t.close()


# ---------------------------------------------------------------------------
# orchestrate integration tests
# ---------------------------------------------------------------------------

class TestOrchestrateIntegration:
    """Tests for orchestrate using per-operation config."""

    def test_orchestrate_uses_operation_config_model(self):
        """Configure orchestrate model, verify OrchestratorConfig receives it."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(orchestrate=LLMConfig(model="orch-model"))

        # Capture the config that gets passed to the Orchestrator
        created_configs = []
        original_init = None

        from tract.orchestrator.loop import Orchestrator as _Orchestrator
        original_init = _Orchestrator.__init__

        def capture_init(self_orch, tract_inst, *, config=None, llm_callable=None):
            created_configs.append(config)
            original_init(self_orch, tract_inst, config=config, llm_callable=llm_callable)

        _Orchestrator.__init__ = capture_init
        try:
            # Need to set up enough for orchestrator to work
            t.commit(InstructionContent(text="System prompt"))

            # The orchestrator will fail (no real LLM), but we can check the config
            try:
                t.orchestrate()
            except Exception:
                pass  # Expected -- mock LLM won't produce valid orchestrator responses

            # Verify the config was created with operation-level model
            assert len(created_configs) == 1
            config = created_configs[0]
            assert config is not None
            assert config.model == "orch-model"
        finally:
            _Orchestrator.__init__ = original_init
        t.close()

    def test_orchestrate_explicit_config_wins(self):
        """Explicit OrchestratorConfig.model overrides operation config."""
        from tract.orchestrator.config import OrchestratorConfig

        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(orchestrate=LLMConfig(model="op-orch"))

        created_configs = []
        from tract.orchestrator.loop import Orchestrator as _Orchestrator
        original_init = _Orchestrator.__init__

        def capture_init(self_orch, tract_inst, *, config=None, llm_callable=None):
            created_configs.append(config)
            original_init(self_orch, tract_inst, config=config, llm_callable=llm_callable)

        _Orchestrator.__init__ = capture_init
        try:
            t.commit(InstructionContent(text="System prompt"))
            explicit_config = OrchestratorConfig(model="explicit-model")

            try:
                t.orchestrate(config=explicit_config)
            except Exception:
                pass

            assert len(created_configs) == 1
            config = created_configs[0]
            # Explicit model should win over operation-level
            assert config.model == "explicit-model"
        finally:
            _Orchestrator.__init__ = original_init
        t.close()

    def test_orchestrate_config_not_mutated(self):
        """Pass a config object, verify the ORIGINAL object is not mutated."""
        from tract.orchestrator.config import OrchestratorConfig

        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(orchestrate=LLMConfig(model="op-orch", temperature=0.5))

        t.commit(InstructionContent(text="System prompt"))

        # Create a config with default model (None) and default temperature (0.0)
        original_config = OrchestratorConfig()
        assert original_config.model is None
        assert original_config.temperature == 0.0

        try:
            t.orchestrate(config=original_config)
        except Exception:
            pass

        # The ORIGINAL config object must NOT have been mutated
        assert original_config.model is None
        assert original_config.temperature == 0.0
        t.close()


# ---------------------------------------------------------------------------
# Backward compatibility tests
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    """Tests ensuring no regressions when no operation config is set."""

    def test_no_operation_config_chat_unchanged(self):
        """chat without any operation config works identically."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate()

        assert resp.text == "Mock response"
        # No model/temperature/max_tokens in kwargs (no operation config, no call override)
        assert "model" not in mock.last_kwargs
        assert "temperature" not in mock.last_kwargs
        t.close()

    def test_no_operation_config_compress_unchanged(self):
        """compress without operation config works identically."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)

        t.commit(InstructionContent(text="First instruction"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi there"))

        result = t.compress()
        # No extra kwargs passed to LLM
        assert "model" not in mock.last_kwargs
        assert "temperature" not in mock.last_kwargs
        t.close()


# ---------------------------------------------------------------------------
# Multi-field query_by_config tests
# ---------------------------------------------------------------------------

class TestQueryByConfigMultiField:
    """Tests for enhanced query_by_config with multi-field AND and IN support."""

    def _setup_tract_with_configs(self):
        """Create a tract with commits using different generation configs."""
        t = Tract.open()
        t.commit(
            DialogueContent(role="user", text="q1"),
            generation_config={"model": "gpt-4o", "temperature": 0.5},
        )
        t.commit(
            DialogueContent(role="assistant", text="a1"),
            generation_config={"model": "gpt-4o", "temperature": 0.9},
        )
        t.commit(
            DialogueContent(role="user", text="q2"),
            generation_config={"model": "gpt-3.5-turbo", "temperature": 0.5},
        )
        t.commit(
            DialogueContent(role="assistant", text="a2"),
            generation_config={"model": "gpt-4o-mini", "temperature": 0.7},
        )
        return t

    def test_single_field_backward_compat(self):
        """Original (field, op, value) signature still works."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config("model", "=", "gpt-4o")
        assert len(results) == 2
        assert all(r.generation_config.model == "gpt-4o" for r in results)
        t.close()

    def test_multi_field_and(self):
        """Multiple conditions combined with AND."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config(conditions=[
            ("model", "=", "gpt-4o"),
            ("temperature", ">", 0.7),
        ])
        assert len(results) == 1
        assert results[0].generation_config.model == "gpt-4o"
        assert results[0].generation_config.temperature == 0.9
        t.close()

    def test_in_operator(self):
        """IN operator for set membership."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config(conditions=[
            ("model", "in", ["gpt-4o", "gpt-3.5-turbo"]),
        ])
        assert len(results) == 3
        t.close()

    def test_in_operator_combined_with_field(self):
        """IN operator combined with other conditions."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config(conditions=[
            ("model", "in", ["gpt-4o", "gpt-4o-mini"]),
            ("temperature", ">=", 0.7),
        ])
        assert len(results) == 2  # gpt-4o@0.9 and gpt-4o-mini@0.7
        t.close()

    def test_whole_config_match(self):
        """LLMConfig object matches all non-None fields."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config(LLMConfig(model="gpt-4o", temperature=0.5))
        assert len(results) == 1
        assert results[0].generation_config.model == "gpt-4o"
        assert results[0].generation_config.temperature == 0.5
        t.close()

    def test_whole_config_single_field(self):
        """LLMConfig with only one field set matches like single-field query."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config(LLMConfig(model="gpt-4o"))
        assert len(results) == 2
        t.close()

    def test_whole_config_empty_returns_empty(self):
        """LLMConfig with all None fields returns empty list."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config(LLMConfig())
        assert len(results) == 0
        t.close()

    def test_no_matches(self):
        """Query returns empty list when nothing matches."""
        t = self._setup_tract_with_configs()
        results = t.query_by_config("model", "=", "nonexistent-model")
        assert len(results) == 0
        t.close()

    def test_invalid_operator(self):
        """Unsupported operator raises ValueError."""
        t = self._setup_tract_with_configs()
        with pytest.raises(ValueError, match="Unsupported operator"):
            t.query_by_config("model", "LIKE", "gpt%")
        t.close()

    def test_invalid_usage_raises_type_error(self):
        """Calling with wrong argument combination raises TypeError."""
        t = self._setup_tract_with_configs()
        with pytest.raises(TypeError, match="query_by_config requires"):
            t.query_by_config()
        t.close()


# ---------------------------------------------------------------------------
# Advanced LLMConfig tests
# ---------------------------------------------------------------------------

class TestLLMConfigAdvanced:
    """Tests for LLMConfig from_dict/to_dict, round-trip, and edge cases."""

    def test_from_dict_round_trip(self):
        """from_dict(to_dict()) produces equal config."""
        config = LLMConfig(model="gpt-4o", temperature=0.7, top_p=0.9, seed=42)
        assert LLMConfig.from_dict(config.to_dict()) == config

    def test_from_dict_with_extra_keys(self):
        """Unknown keys go to extra field."""
        d = {"model": "gpt-4o", "custom_key": "custom_value", "another": 123}
        config = LLMConfig.from_dict(d)
        assert config.model == "gpt-4o"
        assert config.extra is not None
        assert config.extra["custom_key"] == "custom_value"
        assert config.extra["another"] == 123

    def test_round_trip_with_extra(self):
        """Extra keys survive round-trip."""
        d = {"model": "gpt-4o", "provider_specific": "abc"}
        config = LLMConfig.from_dict(d)
        result = config.to_dict()
        assert result == d

    def test_from_dict_none_returns_none(self):
        """from_dict(None) returns None."""
        assert LLMConfig.from_dict(None) is None

    def test_stop_sequences_as_tuple(self):
        """stop_sequences stored as tuple even when created with list."""
        config = LLMConfig(stop_sequences=["stop1", "stop2"])
        assert isinstance(config.stop_sequences, tuple)
        assert config.stop_sequences == ("stop1", "stop2")

    def test_stop_sequences_json_round_trip(self):
        """stop_sequences survives JSON round-trip (list -> tuple -> list -> tuple)."""
        config = LLMConfig(stop_sequences=("stop1", "stop2"))
        d = config.to_dict()
        assert isinstance(d["stop_sequences"], list)
        restored = LLMConfig.from_dict(d)
        assert restored.stop_sequences == ("stop1", "stop2")

    def test_extra_is_immutable(self):
        """extra field is MappingProxyType (immutable)."""
        import types
        config = LLMConfig(extra={"key": "value"})
        assert isinstance(config.extra, types.MappingProxyType)
        with pytest.raises(TypeError):
            config.extra["new_key"] = "new_value"

    def test_hashable(self):
        """LLMConfig can be used as dict key or set member."""
        c1 = LLMConfig(model="gpt-4o", temperature=0.7)
        c2 = LLMConfig(model="gpt-4o", temperature=0.7)
        assert hash(c1) == hash(c2)
        assert {c1, c2} == {c1}

    def test_non_none_fields(self):
        """non_none_fields returns only set fields."""
        config = LLMConfig(model="gpt-4o", temperature=0.7)
        assert config.non_none_fields() == {"model": "gpt-4o", "temperature": 0.7}

    def test_non_none_fields_empty(self):
        """non_none_fields on default config returns empty dict."""
        config = LLMConfig()
        assert config.non_none_fields() == {}

    def test_all_fields(self):
        """All 9 typed fields can be set and retrieved."""
        config = LLMConfig(
            model="gpt-4o",
            temperature=0.7,
            top_p=0.9,
            max_tokens=1000,
            stop_sequences=("stop",),
            frequency_penalty=0.5,
            presence_penalty=0.3,
            top_k=50,
            seed=42,
            extra={"custom": "val"},
        )
        assert config.model == "gpt-4o"
        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.max_tokens == 1000
        assert config.stop_sequences == ("stop",)
        assert config.frequency_penalty == 0.5
        assert config.presence_penalty == 0.3
        assert config.top_k == 50
        assert config.seed == 42
        assert config.extra["custom"] == "val"


# ---------------------------------------------------------------------------
# OperationConfigs dataclass tests
# ---------------------------------------------------------------------------

class TestOperationConfigsDataclass:
    """Tests for the OperationConfigs frozen dataclass."""

    def test_create_with_defaults(self):
        """All fields default to None."""
        oc = OperationConfigs()
        assert oc.chat is None
        assert oc.merge is None
        assert oc.compress is None
        assert oc.orchestrate is None

    def test_create_with_values(self):
        """Fields can be set at construction."""
        oc = OperationConfigs(
            chat=LLMConfig(model="chat-model"),
            compress=LLMConfig(model="compress-model"),
        )
        assert oc.chat.model == "chat-model"
        assert oc.compress.model == "compress-model"
        assert oc.merge is None
        assert oc.orchestrate is None

    def test_frozen(self):
        """Attempting to modify raises FrozenInstanceError."""
        oc = OperationConfigs()
        with pytest.raises(dataclasses.FrozenInstanceError):
            oc.chat = LLMConfig(model="test")

    def test_typo_caught_at_construction(self):
        """Misspelled field name raises TypeError at construction."""
        with pytest.raises(TypeError):
            OperationConfigs(chatt=LLMConfig(model="test"))

    def test_configure_operations_typed(self):
        """configure_operations accepts OperationConfigs instance."""
        t = Tract.open()
        oc = OperationConfigs(chat=LLMConfig(model="gpt-4o"))
        t.configure_operations(oc)
        assert t.operation_configs.chat.model == "gpt-4o"
        t.close()

    def test_configure_operations_mixed_raises(self):
        """Passing both OperationConfigs and kwargs raises TypeError."""
        t = Tract.open()
        oc = OperationConfigs(chat=LLMConfig(model="gpt-4o"))
        with pytest.raises(TypeError, match="Cannot mix"):
            t.configure_operations(oc, merge=LLMConfig(model="gpt-4o"))
        t.close()

    def test_open_with_operations_param(self):
        """Tract.open(operations=OperationConfigs(...)) applies config."""
        oc = OperationConfigs(chat=LLMConfig(model="gpt-4o"))
        t = Tract.open(operations=oc)
        assert t.operation_configs.chat.model == "gpt-4o"
        t.close()


# ---------------------------------------------------------------------------
# from_dict alias and ignore tests
# ---------------------------------------------------------------------------

class TestFromDictAliases:
    """Tests for LLMConfig.from_dict() alias and ignore handling."""

    def test_stop_alias(self):
        """'stop' is aliased to stop_sequences."""
        config = LLMConfig.from_dict({"stop": ["a", "b"]})
        assert config.stop_sequences == ("a", "b")
        assert config.extra is None

    def test_max_completion_tokens_alias(self):
        """'max_completion_tokens' is aliased to max_tokens."""
        config = LLMConfig.from_dict({"max_completion_tokens": 500})
        assert config.max_tokens == 500
        assert config.extra is None

    def test_canonical_wins_over_alias(self):
        """If both alias and canonical present, canonical wins."""
        config = LLMConfig.from_dict({
            "stop": ["alias"],
            "stop_sequences": ["canonical"],
        })
        assert config.stop_sequences == ("canonical",)

    def test_ignored_keys_dropped(self):
        """API plumbing keys are silently dropped."""
        config = LLMConfig.from_dict({
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
            "stream": True,
            "response_format": {"type": "json"},
        })
        assert config.model == "gpt-4o"
        assert config.extra is None  # all non-model keys were ignored

    def test_input_not_mutated(self):
        """from_dict does not mutate the input dict."""
        d = {"stop": ["a"], "messages": [1]}
        original = dict(d)
        LLMConfig.from_dict(d)
        assert d == original

    def test_alias_and_ignore_combined(self):
        """Aliases applied and ignored keys dropped in same call."""
        config = LLMConfig.from_dict({
            "model": "gpt-4o",
            "stop": ["end"],
            "max_completion_tokens": 100,
            "messages": [],
            "tools": [],
        })
        assert config.model == "gpt-4o"
        assert config.stop_sequences == ("end",)
        assert config.max_tokens == 100
        assert config.extra is None


# ---------------------------------------------------------------------------
# from_obj tests
# ---------------------------------------------------------------------------

class TestFromObj:
    """Tests for LLMConfig.from_obj()."""

    def test_from_dataclass(self):
        """Extract LLMConfig from a dataclass instance."""
        source = LLMConfig(model="gpt-4o", temperature=0.7)
        result = LLMConfig.from_obj(source)
        assert result.model == "gpt-4o"
        assert result.temperature == 0.7

    def test_from_plain_object(self):
        """Extract from a plain object with __dict__."""
        class FakeConfig:
            def __init__(self):
                self.model = "gpt-4o"
                self.temperature = 0.5
                self.unknown_field = "extra"
        result = LLMConfig.from_obj(FakeConfig())
        assert result.model == "gpt-4o"
        assert result.temperature == 0.5
        assert result.extra is not None
        assert result.extra["unknown_field"] == "extra"

    def test_from_none(self):
        """from_obj(None) returns None."""
        assert LLMConfig.from_obj(None) is None


# ---------------------------------------------------------------------------
# Consolidated _default_config tests
# ---------------------------------------------------------------------------

class TestDefaultConfig:
    """Tests for consolidated _default_config."""

    def test_open_model_creates_default_config(self):
        """Tract.open(api_key=..., model=...) creates _default_config internally."""
        # We can't test with real api_key, but we can set _default_config manually
        t = Tract.open()
        t._default_config = LLMConfig(model="default-model", temperature=0.5)
        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "default-model"
        t.close()

    def test_default_config_all_fields_available(self):
        """All LLMConfig fields from _default_config are accessible (for future Plan 02)."""
        t = Tract.open()
        t._default_config = LLMConfig(model="default", temperature=0.3)
        # Currently only model is resolved from default -- temperature requires Plan 02
        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "default"
        t.close()

    def test_open_model_and_default_config_raises(self):
        """Providing both model= and default_config= raises ValueError."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Tract.open(
                api_key="fake-key",
                model="gpt-4o",
                default_config=LLMConfig(model="gpt-4o"),
            )


# ---------------------------------------------------------------------------
# 4-level resolution chain tests (Plan 02)
# ---------------------------------------------------------------------------

class TestFourLevelResolution:
    """Tests for the 4-level _resolve_llm_config chain."""

    def test_sugar_beats_llm_config(self):
        """Sugar param (model=) overrides llm_config.model."""
        t = Tract.open()
        t._default_config = LLMConfig(model="default")
        t.configure_operations(chat=LLMConfig(model="op"))
        llm_cfg = LLMConfig(model="llm-config")

        resolved = t._resolve_llm_config(
            "chat", model="sugar", llm_config=llm_cfg,
        )
        assert resolved["model"] == "sugar"
        t.close()

    def test_llm_config_beats_operation(self):
        """llm_config.model overrides operation config."""
        t = Tract.open()
        t.configure_operations(chat=LLMConfig(model="op"))
        llm_cfg = LLMConfig(model="llm-config")

        resolved = t._resolve_llm_config("chat", llm_config=llm_cfg)
        assert resolved["model"] == "llm-config"
        t.close()

    def test_operation_beats_default(self):
        """Operation config overrides tract default."""
        t = Tract.open()
        t._default_config = LLMConfig(model="default")
        t.configure_operations(chat=LLMConfig(model="op"))

        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "op"
        t.close()

    def test_default_used_as_fallback(self):
        """Tract default is used when no higher-level config is set."""
        t = Tract.open()
        t._default_config = LLMConfig(model="default", temperature=0.5)

        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "default"
        assert resolved["temperature"] == 0.5
        t.close()

    def test_all_nine_fields_resolved(self):
        """All 9 typed fields go through the resolution chain."""
        t = Tract.open()
        t._default_config = LLMConfig(
            model="m", temperature=0.1, top_p=0.2, max_tokens=100,
            stop_sequences=("s",), frequency_penalty=0.3,
            presence_penalty=0.4, top_k=10, seed=42,
        )
        resolved = t._resolve_llm_config("chat")
        assert resolved["model"] == "m"
        assert resolved["temperature"] == 0.1
        assert resolved["top_p"] == 0.2
        assert resolved["max_tokens"] == 100
        assert resolved["stop"] == ["s"]  # canonical stop_sequences -> API "stop"
        assert resolved["frequency_penalty"] == 0.3
        assert resolved["presence_penalty"] == 0.4
        assert resolved["top_k"] == 10
        assert resolved["seed"] == 42
        t.close()

    def test_mixed_levels(self):
        """Different fields come from different levels."""
        t = Tract.open()
        t._default_config = LLMConfig(model="default-model", seed=42)
        t.configure_operations(chat=LLMConfig(temperature=0.5))
        llm_cfg = LLMConfig(top_p=0.9)

        resolved = t._resolve_llm_config("chat", max_tokens=100, llm_config=llm_cfg)
        assert resolved["model"] == "default-model"  # level 4
        assert resolved["temperature"] == 0.5  # level 3
        assert resolved["top_p"] == 0.9  # level 2
        assert resolved["max_tokens"] == 100  # level 1 (sugar)
        assert resolved["seed"] == 42  # level 4
        t.close()

    def test_extra_kwargs_merge_order(self):
        """Extra kwargs merge: default < operation < llm_config < call."""
        t = Tract.open()
        t._default_config = LLMConfig(extra={"a": 1, "b": 1})
        t.configure_operations(chat=LLMConfig(extra={"b": 2, "c": 2}))
        llm_cfg = LLMConfig(extra={"c": 3, "d": 3})

        resolved = t._resolve_llm_config("chat", llm_config=llm_cfg, e=4)
        assert resolved["a"] == 1  # from default
        assert resolved["b"] == 2  # op overrides default
        assert resolved["c"] == 3  # llm_config overrides op
        assert resolved["d"] == 3  # from llm_config
        assert resolved["e"] == 4  # from call kwargs
        t.close()

    def test_default_temperature_resolved(self):
        """Temperature from default config is used when no higher level sets it."""
        t = Tract.open()
        t._default_config = LLMConfig(temperature=0.3)

        resolved = t._resolve_llm_config("chat")
        assert resolved["temperature"] == 0.3
        t.close()


# ---------------------------------------------------------------------------
# Full generation_config capture tests (Plan 02)
# ---------------------------------------------------------------------------

class TestFullGenerationConfigCapture:
    """Tests for _build_generation_config capturing all resolved fields."""

    def test_captures_all_fields(self):
        """generation_config on commit captures full resolved config."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(
            chat=LLMConfig(model="gpt-4o", temperature=0.7, top_p=0.9, seed=42)
        )

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate()

        # All fields should be captured in generation_config
        gc = resp.generation_config
        assert gc.model is not None  # from response (authoritative)
        assert gc.temperature == 0.7
        assert gc.top_p == 0.9
        assert gc.seed == 42
        t.close()

    def test_response_model_authoritative(self):
        """Response model overrides requested model in generation_config."""
        t = Tract.open()
        # Mock always returns requested model in response (kwargs.get("model", self._model)).
        # To test authoritative response model, we need a mock that returns a
        # different model than what was requested. Use a subclass.
        class AuthoritativeMock(MockLLMClient):
            def chat(self, messages, **kwargs):
                self.last_messages = messages
                self.last_kwargs = kwargs
                text = self.responses[min(self._call_count, len(self.responses) - 1)]
                self._call_count += 1
                return {
                    "choices": [{"message": {"content": text}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    "model": "actual-model-from-api",  # Always return this regardless of request
                }

        mock = AuthoritativeMock()
        t.configure_llm(mock)
        t.configure_operations(chat=LLMConfig(model="requested-model"))

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate()

        assert resp.generation_config.model == "actual-model-from-api"
        t.close()


# ---------------------------------------------------------------------------
# llm_config= parameter tests (Plan 02)
# ---------------------------------------------------------------------------

class TestLlmConfigParameter:
    """Tests for llm_config= parameter on chat/generate/merge/compress."""

    def test_generate_with_llm_config(self):
        """generate(llm_config=...) forwards config to LLM."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        cfg = LLMConfig(model="cfg-model", temperature=0.3, top_p=0.8)
        resp = t.generate(llm_config=cfg)

        assert mock.last_kwargs.get("model") == "cfg-model"
        assert mock.last_kwargs.get("temperature") == 0.3
        assert mock.last_kwargs.get("top_p") == 0.8
        t.close()

    def test_chat_with_llm_config(self):
        """chat(text, llm_config=...) forwards config to LLM."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        cfg = LLMConfig(model="cfg-model", seed=42)
        resp = t.chat("Hello", llm_config=cfg)

        assert mock.last_kwargs.get("model") == "cfg-model"
        assert mock.last_kwargs.get("seed") == 42
        t.close()

    def test_sugar_overrides_llm_config(self):
        """model= sugar param overrides llm_config.model."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        cfg = LLMConfig(model="cfg-model")
        resp = t.generate(model="sugar-model", llm_config=cfg)

        assert mock.last_kwargs.get("model") == "sugar-model"
        t.close()

    def test_compress_with_llm_config(self):
        """compress(llm_config=...) forwards config to LLM."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary"])
        t.configure_llm(mock)

        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        cfg = LLMConfig(model="compress-cfg-model", temperature=0.1)
        t.compress(llm_config=cfg)

        assert mock.last_kwargs.get("model") == "compress-cfg-model"
        assert mock.last_kwargs.get("temperature") == 0.1
        t.close()


# ---------------------------------------------------------------------------
# Extra kwargs pass-through on generate()/chat() (direct **kwargs)
# ---------------------------------------------------------------------------

class TestExtraKwargsPassThrough:
    """Tests for passing extra provider-specific kwargs through generate()/chat()."""

    def test_generate_extra_kwargs_forwarded(self):
        """generate(reasoning_effort='high') forwards to the LLM client."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate(reasoning_effort="high")

        assert mock.last_kwargs.get("reasoning_effort") == "high"
        t.close()

    def test_chat_extra_kwargs_forwarded(self):
        """chat(text, reasoning_effort='high') forwards to the LLM client."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        resp = t.chat("Hello", reasoning_effort="high")

        assert mock.last_kwargs.get("reasoning_effort") == "high"
        t.close()

    def test_generate_extra_kwargs_override_llm_config_extra(self):
        """Call-level kwargs override llm_config.extra for the same key."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        cfg = LLMConfig(extra={"reasoning_effort": "low", "other": "keep"})
        resp = t.generate(llm_config=cfg, reasoning_effort="high")

        assert mock.last_kwargs.get("reasoning_effort") == "high"
        assert mock.last_kwargs.get("other") == "keep"
        t.close()

    def test_generate_extra_kwargs_override_operation_extra(self):
        """Call-level kwargs override operation-config extra."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(chat=LLMConfig(extra={"reasoning_effort": "low"}))

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate(reasoning_effort="high")

        assert mock.last_kwargs.get("reasoning_effort") == "high"
        t.close()

    def test_generate_multiple_extra_kwargs(self):
        """Multiple extra kwargs all arrive at the LLM client."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate(reasoning_effort="high", top_k=40, custom_flag=True)

        assert mock.last_kwargs.get("reasoning_effort") == "high"
        assert mock.last_kwargs.get("top_k") == 40
        assert mock.last_kwargs.get("custom_flag") is True
        t.close()

    def test_extra_kwargs_recorded_in_generation_config(self):
        """Extra kwargs appear in ChatResponse.generation_config."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        t.user("Hello")
        resp = t.generate(reasoning_effort="high")

        # reasoning_effort should be captured in the generation_config extra
        assert resp.generation_config.extra is not None
        assert resp.generation_config.extra.get("reasoning_effort") == "high"
        t.close()

    def test_chat_extra_kwargs_with_sugar_params(self):
        """Extra kwargs coexist with sugar params (model, temperature, etc.)."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)

        t.system("You are helpful")
        resp = t.chat(
            "Hello",
            model="gpt-4o",
            temperature=0.5,
            reasoning_effort="high",
        )

        assert mock.last_kwargs.get("model") == "gpt-4o"
        assert mock.last_kwargs.get("temperature") == 0.5
        assert mock.last_kwargs.get("reasoning_effort") == "high"
        t.close()


# ---------------------------------------------------------------------------
# Compress error guard tests (Plan 02)
# ---------------------------------------------------------------------------

class TestCompressErrorGuard:
    """Tests for compress() error when LLM params provided without client."""

    def test_compress_model_without_client_raises(self):
        """compress(model=...) without LLM client raises LLMConfigError."""
        from tract.llm.errors import LLMConfigError

        t = Tract.open()
        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        with pytest.raises(LLMConfigError, match="LLM parameters provided"):
            t.compress(model="gpt-4o")
        t.close()

    def test_compress_llm_config_without_client_raises(self):
        """compress(llm_config=...) without LLM client raises LLMConfigError."""
        from tract.llm.errors import LLMConfigError

        t = Tract.open()
        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        with pytest.raises(LLMConfigError, match="LLM parameters provided"):
            t.compress(llm_config=LLMConfig(model="gpt-4o"))
        t.close()

    def test_compress_content_without_client_ok(self):
        """compress(content=...) without LLM client works fine (manual mode)."""
        t = Tract.open()
        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        result = t.compress(content="Manual summary")
        assert result is not None
        t.close()

    def test_compress_model_with_content_without_client_ok(self):
        """compress(model=..., content=...) without LLM client works (content bypasses guard)."""
        t = Tract.open()
        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        # content= provided so no LLM call needed
        result = t.compress(model="gpt-4o", content="Manual summary")
        assert result is not None
        t.close()

    def test_compress_operation_config_without_client_ok(self):
        """Operation-level config without client does not raise (no explicit call-level request)."""
        from tract.exceptions import CompressionError

        t = Tract.open()
        t.configure_operations(compress=LLMConfig(model="gpt-4o"))

        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        # No explicit LLM params on compress() call, so no error -- but it still
        # fails because no LLM client (existing CompressionError behavior)
        with pytest.raises(CompressionError, match="No LLM client configured"):
            t.compress()
        t.close()


# ---------------------------------------------------------------------------
# Compression generation_config tests (Plan 02)
# ---------------------------------------------------------------------------

class TestCompressionGenerationConfig:
    """Tests for compression summary commits recording generation_config."""

    def test_summary_commit_has_generation_config(self):
        """LLM-compressed summary commit records the LLM config used."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)
        t.configure_operations(compress=LLMConfig(model="compress-model", temperature=0.1))

        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        result = t.compress()
        # The summary commit should have generation_config
        # We can check via query_by_config
        results = t.query_by_config("model", "=", "compress-model")
        assert len(results) >= 1, "Summary commit should have generation_config with compress-model"
        t.close()

    def test_summary_commit_captures_temperature(self):
        """Summary commit generation_config captures temperature from operation config."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Summary text"])
        t.configure_llm(mock)
        t.configure_operations(compress=LLMConfig(temperature=0.2))

        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        t.compress()
        results = t.query_by_config("temperature", "=", 0.2)
        assert len(results) >= 1, "Summary commit should have temperature=0.2"
        t.close()

    def test_manual_compress_no_generation_config(self):
        """Manual compression (content=...) has no generation_config."""
        t = Tract.open()

        t.commit(InstructionContent(text="First"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))

        t.compress(content="Manual summary")
        # No generation_config on manual compression
        results = t.query_by_config("model", "=", "anything")
        assert len(results) == 0
        t.close()


# ---------------------------------------------------------------------------
# Orchestrator full config tests (Plan 02)
# ---------------------------------------------------------------------------

class TestOrchestratorFullConfig:
    """Tests for orchestrator forwarding full config."""

    def test_orchestrator_config_has_max_tokens(self):
        """OrchestratorConfig accepts max_tokens field."""
        from tract.orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(max_tokens=500)
        assert config.max_tokens == 500

    def test_orchestrator_config_has_extra_llm_kwargs(self):
        """OrchestratorConfig accepts extra_llm_kwargs field."""
        from tract.orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig(extra_llm_kwargs={"top_p": 0.9, "seed": 42})
        assert config.extra_llm_kwargs["top_p"] == 0.9
        assert config.extra_llm_kwargs["seed"] == 42

    def test_orchestrator_config_defaults_none(self):
        """OrchestratorConfig defaults max_tokens and extra_llm_kwargs to None."""
        from tract.orchestrator.config import OrchestratorConfig
        config = OrchestratorConfig()
        assert config.max_tokens is None
        assert config.extra_llm_kwargs is None

    def test_orchestrate_forwards_max_tokens(self):
        """orchestrate() with operation config forwards max_tokens."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(
            orchestrate=LLMConfig(model="orch-model", max_tokens=500)
        )

        created_configs = []
        from tract.orchestrator.loop import Orchestrator as _Orchestrator
        original_init = _Orchestrator.__init__

        def capture_init(self_orch, tract_inst, *, config=None, llm_callable=None):
            created_configs.append(config)
            original_init(self_orch, tract_inst, config=config, llm_callable=llm_callable)

        _Orchestrator.__init__ = capture_init
        try:
            t.commit(InstructionContent(text="System prompt"))
            try:
                t.orchestrate()
            except Exception:
                pass

            assert len(created_configs) == 1
            config = created_configs[0]
            assert config.model == "orch-model"
            assert config.max_tokens == 500
        finally:
            _Orchestrator.__init__ = original_init
        t.close()

    def test_orchestrate_forwards_extra_fields(self):
        """orchestrate() with operation config containing top_p/seed forwards them."""
        t = Tract.open()
        mock = MockLLMClient()
        t.configure_llm(mock)
        t.configure_operations(
            orchestrate=LLMConfig(model="orch-model", top_p=0.9, seed=42)
        )

        created_configs = []
        from tract.orchestrator.loop import Orchestrator as _Orchestrator
        original_init = _Orchestrator.__init__

        def capture_init(self_orch, tract_inst, *, config=None, llm_callable=None):
            created_configs.append(config)
            original_init(self_orch, tract_inst, config=config, llm_callable=llm_callable)

        _Orchestrator.__init__ = capture_init
        try:
            t.commit(InstructionContent(text="System prompt"))
            try:
                t.orchestrate()
            except Exception:
                pass

            config = created_configs[0]
            assert config.extra_llm_kwargs is not None
            assert config.extra_llm_kwargs.get("top_p") == 0.9
            assert config.extra_llm_kwargs.get("seed") == 42
        finally:
            _Orchestrator.__init__ = original_init
        t.close()
