"""Tests for ConfigContent model, validate_content integration, and ConfigIndex.

Covers:
- ConfigContent model creation, validation, immutability
- ConfigContent through validate_content()
- ConfigContent has compilable=False (not in compiled context)
- ConfigIndex.build() with single and multiple config commits
- ConfigIndex.get() and get_all()
- ConfigIndex DAG precedence (closer to HEAD wins)
- ConfigIndex invalidation
- None values are "unset" semantics
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tract import (
    BUILTIN_TYPE_HINTS,
    ConfigContent,
    ContentValidationError,
    DialogueContent,
    InstructionContent,
    Tract,
    validate_content,
)
from tract.operations.config_index import ConfigIndex


# ---------------------------------------------------------------------------
# ConfigContent model tests
# ---------------------------------------------------------------------------


class TestConfigContentModel:
    """ConfigContent construction, validation, and immutability."""

    def test_create_basic(self):
        """ConfigContent accepts a settings dict."""
        c = ConfigContent(settings={"model": "gpt-4o"})
        assert c.content_type == "config"
        assert c.settings == {"model": "gpt-4o"}

    def test_create_empty_settings(self):
        """ConfigContent with empty settings is valid."""
        c = ConfigContent(settings={})
        assert c.settings == {}

    def test_create_multiple_keys(self):
        """ConfigContent supports arbitrary keys."""
        c = ConfigContent(settings={
            "model": "gpt-4o",
            "temperature": 0.7,
            "max_tokens": 1000,
            "custom_key": [1, 2, 3],
        })
        assert c.settings["temperature"] == 0.7
        assert c.settings["custom_key"] == [1, 2, 3]

    def test_content_type_literal(self):
        """content_type is always 'config'."""
        c = ConfigContent(settings={"x": 1})
        assert c.content_type == "config"

    def test_frozen_model(self):
        """ConfigContent is frozen (immutable)."""
        c = ConfigContent(settings={"model": "gpt-4o"})
        with pytest.raises(ValidationError):
            c.content_type = "other"

    def test_settings_with_none_value(self):
        """ConfigContent accepts None values (unset semantics)."""
        c = ConfigContent(settings={"model": None})
        assert c.settings["model"] is None

    def test_settings_with_nested_dict(self):
        """ConfigContent accepts nested dict values."""
        c = ConfigContent(settings={"compact_tools": {"search": 500}})
        assert c.settings["compact_tools"]["search"] == 500


# ---------------------------------------------------------------------------
# validate_content() integration
# ---------------------------------------------------------------------------


class TestConfigContentValidation:
    """ConfigContent through the validate_content function."""

    def test_validate_config_content(self):
        """validate_content recognizes content_type='config'."""
        result = validate_content({
            "content_type": "config",
            "settings": {"model": "gpt-4o"},
        })
        assert isinstance(result, ConfigContent)
        assert result.settings["model"] == "gpt-4o"

    def test_validate_config_missing_settings(self):
        """validate_content rejects config without settings field."""
        with pytest.raises(ContentValidationError):
            validate_content({
                "content_type": "config",
            })

    def test_config_in_builtin_types(self):
        """'config' is listed in BUILTIN_CONTENT_TYPES."""
        from tract.models.content import BUILTIN_CONTENT_TYPES
        assert "config" in BUILTIN_CONTENT_TYPES


# ---------------------------------------------------------------------------
# compilable=False behavior
# ---------------------------------------------------------------------------


class TestConfigNotCompilable:
    """ConfigContent has compilable=False and is excluded from compiled context."""

    def test_type_hints_compilable_false(self):
        """ContentTypeHints for 'config' has compilable=False."""
        hints = BUILTIN_TYPE_HINTS["config"]
        assert hints.compilable is False

    def test_config_excluded_from_compile(self):
        """Config commits are not present in compiled messages."""
        with Tract.open() as t:
            t.system("You are helpful.")
            t.configure(model="gpt-4o")
            t.user("Hello")
            compiled = t.compile()
            # Only instruction + user messages should appear
            texts = [m.content for m in compiled.messages]
            assert any("helpful" in txt for txt in texts)
            assert any("Hello" in txt for txt in texts)
            # No config content should leak into messages
            assert not any("gpt-4o" in txt for txt in texts)

    def test_config_does_not_increase_message_count(self):
        """Config commits do not add to compiled messages."""
        with Tract.open() as t:
            t.user("Hello")
            count_before = len(t.compile().messages)
            t.configure(temperature=0.5)
            count_after = len(t.compile().messages)
            # Config commit should not add a compiled message
            assert count_after == count_before


# ---------------------------------------------------------------------------
# ConfigIndex.build() tests
# ---------------------------------------------------------------------------


class TestConfigIndexBuild:
    """ConfigIndex.build() from DAG ancestry."""

    def test_build_single_config(self):
        """ConfigIndex.build resolves a single config commit."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            idx = t.config_index
            assert idx.get("model") == "gpt-4o"

    def test_build_multiple_configs(self):
        """ConfigIndex accumulates settings across multiple config commits."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            t.configure(temperature=0.7)
            idx = t.config_index
            assert idx.get("model") == "gpt-4o"
            assert idx.get("temperature") == 0.7

    def test_build_empty_dag(self):
        """ConfigIndex from empty DAG is empty."""
        with Tract.open() as t:
            idx = t.config_index
            assert idx.get("model") is None
            assert idx.get_all() == {}

    def test_build_no_config_commits(self):
        """ConfigIndex with only non-config commits is empty."""
        with Tract.open() as t:
            t.user("Hello")
            t.assistant("Hi")
            idx = t.config_index
            assert idx.get("model") is None
            assert len(idx) == 0


# ---------------------------------------------------------------------------
# ConfigIndex.get() and get_all()
# ---------------------------------------------------------------------------


class TestConfigIndexGetters:
    """ConfigIndex.get() and get_all() methods."""

    def test_get_existing_key(self):
        """get() returns value for existing key."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            assert t.get_config("model") == "gpt-4o"

    def test_get_missing_key_default(self):
        """get() returns default for missing key."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            assert t.get_config("temperature") is None
            assert t.get_config("temperature", default=0.5) == 0.5

    def test_get_all_returns_dict(self):
        """get_all() returns all resolved settings as a dict."""
        with Tract.open() as t:
            t.configure(model="gpt-4o", temperature=0.7)
            result = t.get_all_configs()
            assert result == {"model": "gpt-4o", "temperature": 0.7}

    def test_get_all_excludes_none(self):
        """get_all() excludes keys with None values."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            t.configure(model=None)
            result = t.get_all_configs()
            assert "model" not in result


# ---------------------------------------------------------------------------
# DAG precedence (closer to HEAD wins)
# ---------------------------------------------------------------------------


class TestConfigIndexPrecedence:
    """Config resolution with DAG precedence."""

    def test_later_config_overrides_earlier(self):
        """Closer-to-HEAD config commit wins for the same key."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5")
            t.configure(model="gpt-4o")
            assert t.get_config("model") == "gpt-4o"

    def test_partial_override(self):
        """Later config only overrides specified keys."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5", temperature=0.7)
            t.configure(model="gpt-4o")
            assert t.get_config("model") == "gpt-4o"
            assert t.get_config("temperature") == 0.7

    def test_multiple_overrides(self):
        """Multiple overrides in sequence, last one wins."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5")
            t.configure(model="gpt-4")
            t.configure(model="gpt-4o")
            assert t.get_config("model") == "gpt-4o"

    def test_interleaved_with_non_config(self):
        """Config resolution works with interleaved non-config commits."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5")
            t.user("Hello")
            t.assistant("Hi")
            t.configure(model="gpt-4o")
            t.user("Another message")
            assert t.get_config("model") == "gpt-4o"


# ---------------------------------------------------------------------------
# ConfigIndex invalidation
# ---------------------------------------------------------------------------


class TestConfigIndexInvalidation:
    """ConfigIndex staleness and rebuild."""

    def test_invalidate_marks_stale(self):
        """invalidate() marks the index as stale."""
        idx = ConfigIndex()
        assert not idx.is_stale
        idx.invalidate()
        assert idx.is_stale

    def test_configure_invalidates_index(self):
        """t.configure() invalidates the config index for rebuild."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5")
            _ = t.config_index  # Force build
            assert not t._config_mgr._config_index.is_stale
            t.configure(model="gpt-4o")
            # After configure, index should be stale
            assert t._config_mgr._config_index.is_stale

    def test_stale_index_rebuilds_on_access(self):
        """Accessing config_index when stale triggers rebuild."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5")
            _ = t.config_index  # Build
            t.configure(model="gpt-4o")
            # Access should rebuild with new value
            assert t.get_config("model") == "gpt-4o"


# ---------------------------------------------------------------------------
# None / unset semantics
# ---------------------------------------------------------------------------


class TestConfigUnsetSemantics:
    """None values as unset semantics."""

    def test_none_value_unsets_key(self):
        """Setting a key to None means 'unset'."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            assert t.get_config("model") == "gpt-4o"
            t.configure(model=None)
            assert t.get_config("model") is None

    def test_none_unset_returns_default(self):
        """After unsetting, get_config returns the default."""
        with Tract.open() as t:
            t.configure(model="gpt-4o")
            t.configure(model=None)
            assert t.get_config("model", default="fallback") == "fallback"

    def test_none_excluded_from_get_all(self):
        """get_all() does not include keys set to None."""
        with Tract.open() as t:
            t.configure(model="gpt-4o", temperature=0.7)
            t.configure(model=None)
            all_cfg = t.get_all_configs()
            assert "model" not in all_cfg
            assert all_cfg["temperature"] == 0.7

    def test_unset_then_reset(self):
        """Can set, unset, then re-set a key."""
        with Tract.open() as t:
            t.configure(model="gpt-3.5")
            t.configure(model=None)
            assert t.get_config("model") is None
            t.configure(model="gpt-4o")
            assert t.get_config("model") == "gpt-4o"
