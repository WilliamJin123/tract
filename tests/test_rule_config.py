"""Tests for config resolution from active rules."""

import pytest

from tract import Tract
from tract.rules.config import resolve_all_configs, resolve_config
from tract.rules.index import RuleIndex
from tract.rules.models import RuleEntry


# ---------------------------------------------------------------------------
# Unit tests for resolve_config / resolve_all_configs (RuleIndex level)
# ---------------------------------------------------------------------------


class TestResolveConfigUnit:
    """Unit-level tests using RuleIndex directly."""

    def test_resolve_config_basic(self):
        """Single active rule resolves its value."""
        idx = RuleIndex()
        idx.add_rule(RuleEntry(
            name="set_temp",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.7},
            commit_hash="aaa",
            dag_distance=0,
        ))
        assert resolve_config(idx, "temperature") == 0.7

    def test_resolve_config_default(self):
        """Missing key returns default (None or explicit)."""
        idx = RuleIndex()
        assert resolve_config(idx, "nonexistent") is None
        assert resolve_config(idx, "nonexistent", default="fallback") == "fallback"

    def test_resolve_config_override(self):
        """Closer rule (lower dag_distance) wins when two rules set same key."""
        idx = RuleIndex()
        idx.add_rule(RuleEntry(
            name="temp_v1",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.7},
            commit_hash="aaa",
            dag_distance=2,
        ))
        idx.add_rule(RuleEntry(
            name="temp_v2",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.3},
            commit_hash="bbb",
            dag_distance=0,
        ))
        assert resolve_config(idx, "temperature") == 0.3

    def test_resolve_all_configs(self):
        """Multiple config keys resolved simultaneously."""
        idx = RuleIndex()
        idx.add_rule(RuleEntry(
            name="set_temp",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.7},
            commit_hash="aaa",
            dag_distance=0,
        ))
        idx.add_rule(RuleEntry(
            name="set_model",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "model", "value": "gpt-4"},
            commit_hash="bbb",
            dag_distance=1,
        ))
        result = resolve_all_configs(idx)
        assert result == {"temperature": 0.7, "model": "gpt-4"}

    def test_resolve_all_configs_picks_closest(self):
        """resolve_all_configs respects dag_distance for each key independently."""
        idx = RuleIndex()
        idx.add_rule(RuleEntry(
            name="temp_old",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.9},
            commit_hash="aaa",
            dag_distance=5,
        ))
        idx.add_rule(RuleEntry(
            name="temp_new",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.2},
            commit_hash="bbb",
            dag_distance=1,
        ))
        idx.add_rule(RuleEntry(
            name="set_model",
            trigger="active",
            condition=None,
            action={"type": "set_config", "key": "model", "value": "gpt-4"},
            commit_hash="ccc",
            dag_distance=3,
        ))
        result = resolve_all_configs(idx)
        assert result["temperature"] == 0.2
        assert result["model"] == "gpt-4"

    def test_non_config_rules_ignored(self):
        """Rules with trigger != 'active' or action != 'set_config' are ignored."""
        idx = RuleIndex()
        # commit-triggered rule (not active)
        idx.add_rule(RuleEntry(
            name="on_commit",
            trigger="commit",
            condition=None,
            action={"type": "set_config", "key": "temperature", "value": 0.5},
            commit_hash="aaa",
            dag_distance=0,
        ))
        # active trigger but different action type
        idx.add_rule(RuleEntry(
            name="compress_rule",
            trigger="active",
            condition=None,
            action={"type": "compress", "threshold": 1000},
            commit_hash="bbb",
            dag_distance=0,
        ))
        assert resolve_config(idx, "temperature") is None
        assert resolve_all_configs(idx) == {}


# ---------------------------------------------------------------------------
# Integration tests through Tract facade
# ---------------------------------------------------------------------------


class TestConfigThroughFacade:
    """Integration tests using Tract.open() -> .rule() -> .get_config()."""

    def test_config_through_facade(self):
        """t.get_config() resolves a value set by t.rule()."""
        t = Tract.open()
        t.rule(
            "set_temp",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.7},
        )
        assert t.get_config("temperature") == 0.7
        assert t.get_config("nonexistent") is None
        assert t.get_config("nonexistent", "default_val") == "default_val"

    def test_config_after_rule_commit(self):
        """New rule commit updates config resolution."""
        t = Tract.open()
        t.rule(
            "temp_v1",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.7},
        )
        assert t.get_config("temperature") == 0.7

        t.user("some work in between")

        t.rule(
            "temp_v2",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.3},
        )
        assert t.get_config("temperature") == 0.3

    def test_config_on_branch(self):
        """Branch-local rule is visible on that branch."""
        t = Tract.open()
        t.user("initial commit")
        t.branch("feature")
        t.rule(
            "feature_temp",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.1},
        )
        assert t.get_config("temperature") == 0.1

    def test_config_inherited(self):
        """Child branch inherits config from parent branch, and can override."""
        t = Tract.open()
        t.rule(
            "base_model",
            trigger="active",
            action={"type": "set_config", "key": "model", "value": "gpt-4"},
        )
        assert t.get_config("model") == "gpt-4"

        t.branch("feature")
        # Still on feature branch -- should inherit config from main
        assert t.get_config("model") == "gpt-4"

        # Override on feature branch
        t.rule(
            "feature_model",
            trigger="active",
            action={"type": "set_config", "key": "model", "value": "gpt-3.5"},
        )
        assert t.get_config("model") == "gpt-3.5"
