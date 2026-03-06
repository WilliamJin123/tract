"""Tests for the RuleIndex class.

Covers: build, get_by_trigger, get_config, get_all_configs, add_rule,
invalidate, is_stale, __len__, __contains__, branch inheritance,
invalidation on rule commit / switch, and compile exclusion.
"""

import pytest

from tract import Tract, RuleContent, RuleIndex
from tract.rules.models import RuleEntry


class TestBuildEmpty:
    """Build from empty tract (no rules)."""

    def test_empty_tract_returns_empty_index(self):
        t = Tract.open()
        idx = t.rule_index
        assert len(idx) == 0

    def test_empty_tract_is_not_stale(self):
        t = Tract.open()
        idx = t.rule_index
        assert not idx.is_stale


class TestBuildSingleRule:
    """Build with a single rule."""

    def test_single_rule_len_is_one(self):
        t = Tract.open()
        t.rule("auto_compress", trigger="commit", action={"type": "compress"})
        idx = t.rule_index
        assert len(idx) == 1

    def test_single_rule_contains_key(self):
        t = Tract.open()
        t.rule("auto_compress", trigger="commit", action={"type": "compress"})
        idx = t.rule_index
        assert ("commit", "auto_compress") in idx

    def test_single_rule_get_by_trigger(self):
        t = Tract.open()
        t.rule("auto_compress", trigger="commit", action={"type": "compress"})
        idx = t.rule_index
        entries = idx.get_by_trigger("commit")
        assert len(entries) == 1
        assert entries[0].name == "auto_compress"
        assert entries[0].action == {"type": "compress"}


class TestBuildMultipleRules:
    """Build with multiple rules on different triggers."""

    def test_multiple_triggers(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        t.rule("r2", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        t.rule("r3", trigger="compile", action={"type": "log"})
        idx = t.rule_index
        assert len(idx) == 3

    def test_each_trigger_returns_own_rules(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        t.rule("r2", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        t.rule("r3", trigger="compile", action={"type": "log"})
        idx = t.rule_index
        assert len(idx.get_by_trigger("commit")) == 1
        assert len(idx.get_by_trigger("active")) == 1
        assert len(idx.get_by_trigger("compile")) == 1


class TestPrecedence:
    """Closer to HEAD wins when same (trigger, name)."""

    def test_later_rule_overrides_earlier(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.3})
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.9})
        idx = t.rule_index
        # Only one entry for the key (trigger, name)
        assert len(idx) == 1
        val = idx.get_config("temperature")
        assert val == 0.9

    def test_closer_has_lower_dag_distance(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.3})
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.9})
        idx = t.rule_index
        entries = idx.get_by_trigger("active")
        assert len(entries) == 1
        assert entries[0].dag_distance == 0  # closest to HEAD


class TestDifferentNamesCoexist:
    """Different names coexist on same trigger."""

    def test_same_trigger_different_names(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        t.rule("model", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        idx = t.rule_index
        entries = idx.get_by_trigger("active")
        assert len(entries) == 2
        names = {e.name for e in entries}
        assert names == {"temp", "model"}


class TestGetByTrigger:
    """get_by_trigger returns correct subset."""

    def test_returns_only_matching_trigger(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        t.rule("r2", trigger="commit", action={"type": "log"})
        t.rule("r3", trigger="active", action={"type": "set_config", "key": "x", "value": 1})
        idx = t.rule_index
        commit_rules = idx.get_by_trigger("commit")
        assert len(commit_rules) == 2
        assert all(e.trigger == "commit" for e in commit_rules)

    def test_empty_trigger_returns_empty_list(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        idx = t.rule_index
        assert idx.get_by_trigger("nonexistent") == []

    def test_sorted_by_dag_distance_ascending(self):
        t = Tract.open()
        t.rule("a", trigger="commit", action={"type": "a"})
        t.rule("b", trigger="commit", action={"type": "b"})
        idx = t.rule_index
        entries = idx.get_by_trigger("commit")
        # Should be sorted by dag_distance ascending (closest first)
        distances = [e.dag_distance for e in entries]
        assert distances == sorted(distances)


class TestGetConfig:
    """get_config for active rule with set_config action."""

    def test_get_config_returns_value(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        idx = t.rule_index
        assert idx.get_config("temperature") == 0.7

    def test_get_config_override_closer_wins(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.3})
        t.user("some dialogue in between")
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.9})
        idx = t.rule_index
        assert idx.get_config("temperature") == 0.9

    def test_get_config_missing_returns_none(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        idx = t.rule_index
        assert idx.get_config("nonexistent") is None

    def test_get_config_empty_index_returns_none(self):
        t = Tract.open()
        idx = t.rule_index
        assert idx.get_config("anything") is None


class TestGetAllConfigs:
    """get_all_configs with multiple keys."""

    def test_multiple_config_keys(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        t.rule("model", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        t.rule("top_p", trigger="active", action={"type": "set_config", "key": "top_p", "value": 0.95})
        idx = t.rule_index
        configs = idx.get_all_configs()
        assert configs == {"temperature": 0.7, "model": "gpt-4", "top_p": 0.95}

    def test_empty_index_returns_empty_dict(self):
        t = Tract.open()
        idx = t.rule_index
        assert idx.get_all_configs() == {}

    def test_non_config_rules_excluded(self):
        t = Tract.open()
        t.rule("compress_rule", trigger="commit", action={"type": "compress"})
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.5})
        idx = t.rule_index
        configs = idx.get_all_configs()
        assert configs == {"temperature": 0.5}


class TestIncrementalAddRule:
    """Incremental add_rule."""

    def test_add_rule_increases_len(self):
        idx = RuleIndex()
        assert len(idx) == 0
        entry = RuleEntry(
            name="r1", trigger="commit", condition=None,
            action={"type": "compress"}, commit_hash="abc123",
            dag_distance=0,
        )
        idx.add_rule(entry)
        assert len(idx) == 1

    def test_add_rule_closer_overrides(self):
        idx = RuleIndex()
        far = RuleEntry(
            name="r1", trigger="commit", condition=None,
            action={"type": "old"}, commit_hash="aaa",
            dag_distance=5,
        )
        close = RuleEntry(
            name="r1", trigger="commit", condition=None,
            action={"type": "new"}, commit_hash="bbb",
            dag_distance=1,
        )
        idx.add_rule(far)
        idx.add_rule(close)
        assert len(idx) == 1
        entries = idx.get_by_trigger("commit")
        assert entries[0].action == {"type": "new"}

    def test_add_rule_farther_does_not_override(self):
        idx = RuleIndex()
        close = RuleEntry(
            name="r1", trigger="commit", condition=None,
            action={"type": "close"}, commit_hash="aaa",
            dag_distance=1,
        )
        far = RuleEntry(
            name="r1", trigger="commit", condition=None,
            action={"type": "far"}, commit_hash="bbb",
            dag_distance=5,
        )
        idx.add_rule(close)
        idx.add_rule(far)
        entries = idx.get_by_trigger("commit")
        assert entries[0].action == {"type": "close"}


class TestInvalidateAndStale:
    """invalidate and is_stale."""

    def test_fresh_index_not_stale(self):
        idx = RuleIndex()
        assert not idx.is_stale

    def test_invalidate_marks_stale(self):
        idx = RuleIndex()
        idx.invalidate()
        assert idx.is_stale

    def test_stale_index_triggers_rebuild_via_property(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        idx1 = t.rule_index
        assert len(idx1) == 1
        # Manually invalidate
        idx1.invalidate()
        assert idx1.is_stale
        # Accessing rule_index again should rebuild
        idx2 = t.rule_index
        assert not idx2.is_stale
        assert len(idx2) == 1


class TestLenAndContains:
    """__len__ and __contains__."""

    def test_len_zero_empty(self):
        idx = RuleIndex()
        assert len(idx) == 0

    def test_len_matches_unique_keys(self):
        t = Tract.open()
        t.rule("a", trigger="commit", action={"type": "x"})
        t.rule("b", trigger="active", action={"type": "y"})
        idx = t.rule_index
        assert len(idx) == 2

    def test_contains_tuple_key(self):
        t = Tract.open()
        t.rule("a", trigger="commit", action={"type": "x"})
        idx = t.rule_index
        assert ("commit", "a") in idx
        assert ("active", "a") not in idx

    def test_contains_after_override(self):
        t = Tract.open()
        t.rule("a", trigger="commit", action={"type": "old"})
        t.rule("a", trigger="commit", action={"type": "new"})
        idx = t.rule_index
        assert ("commit", "a") in idx
        # Still only one entry
        assert len(idx) == 1


class TestSkipsNonRuleCommits:
    """Build skips non-rule commits (dialogue)."""

    def test_dialogue_not_in_index(self):
        t = Tract.open()
        t.user("Hello, world!")
        t.assistant("Hi there!")
        idx = t.rule_index
        assert len(idx) == 0

    def test_mixed_commits_only_rules_indexed(self):
        t = Tract.open()
        t.user("Hello")
        t.rule("r1", trigger="commit", action={"type": "compress"})
        t.assistant("Response")
        t.rule("r2", trigger="active", action={"type": "set_config", "key": "k", "value": "v"})
        t.user("More dialogue")
        idx = t.rule_index
        assert len(idx) == 2


class TestBranchInheritance:
    """Rules inherited from parent branch."""

    def test_child_branch_inherits_rules(self):
        t = Tract.open()
        t.user("base")
        t.rule("base_rule", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        t.branch("feature")
        # On feature branch now, should inherit base_rule
        idx = t.rule_index
        assert len(idx) == 1
        assert ("active", "base_rule") in idx
        assert idx.get_config("model") == "gpt-4"

    def test_child_branch_can_override_inherited_rule(self):
        t = Tract.open()
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.3})
        t.branch("feature")
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.9})
        idx = t.rule_index
        assert idx.get_config("temperature") == 0.9
        assert len(idx) == 1

    def test_child_branch_adds_new_rule(self):
        t = Tract.open()
        t.rule("base", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        t.branch("feature")
        t.rule("extra", trigger="commit", action={"type": "log"})
        idx = t.rule_index
        assert len(idx) == 2
        assert ("active", "base") in idx
        assert ("commit", "extra") in idx


class TestInvalidationOnRuleCommit:
    """Rule index invalidated after rule commit."""

    def test_index_stale_after_rule_commit(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        idx = t.rule_index
        assert not idx.is_stale
        # Commit another rule
        t.rule("r2", trigger="active", action={"type": "set_config", "key": "k", "value": "v"})
        # The previously-fetched index should be stale
        assert idx.is_stale

    def test_rebuilt_index_includes_new_rule(self):
        t = Tract.open()
        t.rule("r1", trigger="commit", action={"type": "compress"})
        idx1 = t.rule_index
        assert len(idx1) == 1
        t.rule("r2", trigger="active", action={"type": "set_config", "key": "k", "value": "v"})
        # Fresh access rebuilds
        idx2 = t.rule_index
        assert len(idx2) == 2


class TestInvalidationOnSwitch:
    """Rule index invalidated after switch."""

    def test_switch_invalidates_index(self):
        t = Tract.open()
        t.user("base commit")
        t.rule("main_rule", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        t.branch("feature", switch=False)
        # Still on main, build index
        idx = t.rule_index
        assert not idx.is_stale
        # Switch to feature
        t.switch("feature")
        assert idx.is_stale

    def test_switch_rebuilds_correct_rules(self):
        t = Tract.open()
        t.user("base commit")
        t.branch("feature", switch=False)
        # Add a rule only on main
        t.rule("main_only", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        main_idx = t.rule_index
        assert len(main_idx) == 1
        # Switch to feature (branched before the rule was added)
        t.switch("feature")
        feature_idx = t.rule_index
        assert len(feature_idx) == 0


class TestSkipAnnotationDisablesRule:
    """SKIP annotation on a rule commit excludes it from the index."""

    def test_skip_annotation_removes_rule(self):
        from tract.models.annotations import Priority
        t = Tract.open()
        info = t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        assert len(t.rule_index) == 1
        t.annotate(info.commit_hash, Priority.SKIP)
        # Invalidate so next access rebuilds
        t._rule_index.invalidate()
        idx = t.rule_index
        assert len(idx) == 0
        assert ("active", "temp") not in idx

    def test_skip_only_affects_annotated_rule(self):
        from tract.models.annotations import Priority
        t = Tract.open()
        info1 = t.rule("r1", trigger="commit", action={"type": "compress"})
        t.rule("r2", trigger="active", action={"type": "set_config", "key": "k", "value": "v"})
        # Build index so it exists before annotate
        _ = t.rule_index
        t.annotate(info1.commit_hash, Priority.SKIP)
        t._rule_index.invalidate()
        idx = t.rule_index
        assert len(idx) == 1
        assert ("commit", "r1") not in idx
        assert ("active", "r2") in idx

    def test_skip_config_no_longer_resolves(self):
        from tract.models.annotations import Priority
        t = Tract.open()
        info = t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        assert t.get_config("temperature") == 0.7
        t.annotate(info.commit_hash, Priority.SKIP)
        t._rule_index.invalidate()
        assert t.get_config("temperature") is None


class TestBuildWithMerge:
    """Rules from both parents collected after merge."""

    def test_rules_from_merged_branch_visible_after_merge(self):
        t = Tract.open()
        t.user("base commit")
        # Add a rule on main
        t.rule("main_rule", trigger="active", action={"type": "set_config", "key": "model", "value": "gpt-4"})
        # Create feature branch and add a different rule
        t.branch("feature")
        t.rule("feature_rule", trigger="commit", action={"type": "log"})
        # Switch back to main and merge
        t.switch("main")
        t.merge("feature")
        # After merge, both rules should be visible
        idx = t.rule_index
        assert ("active", "main_rule") in idx
        assert ("commit", "feature_rule") in idx
        assert len(idx) >= 2

    def test_merged_rule_config_accessible(self):
        t = Tract.open()
        t.user("base")
        t.branch("feature")
        t.rule("feat_temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.2})
        t.switch("main")
        t.user("diverge")
        t.merge("feature")
        assert t.get_config("temperature") == 0.2


class TestRuleNotInCompileOutput:
    """Rule not in compile output."""

    def test_rule_excluded_from_compiled_messages(self):
        t = Tract.open()
        t.user("Hello")
        t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        t.assistant("Hi there")
        ctx = t.compile()
        messages = ctx.to_dicts()
        # Only dialogue commits should appear, not the rule
        for msg in messages:
            # No message should contain rule content
            text = msg.get("content", "")
            assert "set_config" not in text
            assert "temperature" not in text or "temperature" in "Hi there"
        # Should have exactly 2 messages (user + assistant)
        assert len(messages) == 2

    def test_rule_commit_hash_not_in_compiled_hashes(self):
        t = Tract.open()
        t.user("Hello")
        rule_info = t.rule("temp", trigger="active", action={"type": "set_config", "key": "temperature", "value": 0.7})
        t.assistant("Hi")
        ctx = t.compile()
        assert rule_info.commit_hash not in ctx.commit_hashes
