"""Integration tests for the rule engine with Tract operations.

Tests the full integration of the rule engine with Tract's commit/compile/
compress/merge/gc/transition operations. Exercises _fire_rules(),
_fire_transition_rules(), transition(), get_config(), and create_rule actions
through the public Tract API.
"""

from __future__ import annotations

import pytest

from tract import Tract
from tract.rules.models import EvalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tract_with_content() -> Tract:
    """Create an in-memory Tract with a few dialogue commits."""
    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Hello")
    t.assistant("Hi there!")
    return t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def t():
    """In-memory tract, cleaned up after test."""
    tract = Tract.open()
    yield tract
    tract.close()


@pytest.fixture()
def t_with_content():
    """In-memory tract with dialogue commits, cleaned up after test."""
    tract = _make_tract_with_content()
    yield tract
    tract.close()


# ===========================================================================
# Direct _fire_rules tests
# ===========================================================================


class TestFireRules:
    """Tests for Tract._fire_rules() called directly."""

    def test_fire_rules_no_rules(self, t_with_content: Tract):
        """No rules defined -> returns empty EvalResult immediately."""
        result = t_with_content._fire_rules("commit")
        assert isinstance(result, EvalResult)
        assert not result.blocked
        assert result.rules_evaluated == 0
        assert result.rules_fired == 0
        assert result.action_results == []

    def test_fire_rules_block_action(self, t_with_content: Tract):
        """A block rule prevents the event."""
        t_with_content.rule(
            "no_commits",
            trigger="commit",
            action={"type": "block", "reason": "Commits are forbidden"},
        )
        result = t_with_content._fire_rules("commit")
        assert result.blocked
        assert len(result.block_reasons) >= 1
        assert "forbidden" in result.block_reasons[0].lower()

    def test_fire_rules_require_met(self, t_with_content: Tract):
        """A require rule with a met condition does not block."""
        # commit_count will be >= 3 (system + user + assistant + rule itself)
        t_with_content.rule(
            "need_commits",
            trigger="commit",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">=",
                    "value": 1,
                },
            },
        )
        result = t_with_content._fire_rules("commit")
        assert not result.blocked
        assert result.rules_fired >= 1

    def test_fire_rules_require_not_met(self, t_with_content: Tract):
        """A require rule with an unmet condition blocks."""
        t_with_content.rule(
            "need_many_commits",
            trigger="commit",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">",
                    "value": 10000,
                },
            },
        )
        result = t_with_content._fire_rules("commit")
        assert result.blocked
        assert len(result.block_reasons) >= 1

    def test_fire_rules_set_config(self, t_with_content: Tract):
        """set_config action produces an ActionResult with config data."""
        t_with_content.rule(
            "set_temp",
            trigger="commit",
            action={"type": "set_config", "key": "temperature", "value": 0.9},
        )
        result = t_with_content._fire_rules("commit")
        assert not result.blocked
        # Find the set_config action result
        config_results = [
            ar for ar in result.action_results if ar.action_type == "set_config"
        ]
        assert len(config_results) == 1
        assert config_results[0].data["key"] == "temperature"
        assert config_results[0].data["value"] == 0.9

    def test_fire_rules_create_rule_deferred(self, t_with_content: Tract):
        """create_rule action defers and commits a new rule after pipeline."""
        t_with_content.rule(
            "spawner",
            trigger="commit",
            action={
                "type": "create_rule",
                "template": {
                    "name": "spawned_rule",
                    "trigger": "active",
                    "action": {"type": "set_config", "key": "spawned", "value": True},
                },
            },
        )
        # Fire the event -- this should create a deferred rule
        result = t_with_content._fire_rules("commit")
        assert not result.blocked

        # The deferred create_rule should have committed the spawned rule
        create_results = [
            ar for ar in result.action_results if ar.action_type == "create_rule"
        ]
        assert len(create_results) == 1
        assert create_results[0].success
        assert create_results[0].data.get("deferred") is True

        # Verify the spawned rule is now in the index
        # Force index rebuild by invalidating
        if t_with_content._rule_index is not None:
            t_with_content._rule_index.invalidate()
        spawned_config = t_with_content.get_config("spawned")
        assert spawned_config is True

    def test_fire_rules_recursion_guard(self, t_with_content: Tract):
        """When depth exceeds max_depth (3), rule eval is skipped."""
        t_with_content.rule(
            "some_rule",
            trigger="commit",
            action={"type": "set_config", "key": "x", "value": 1},
        )
        # Manually push depth beyond the guard threshold
        t_with_content._rule_eval_depth = 4
        try:
            result = t_with_content._fire_rules("commit")
            # When depth >= max_depth, process_event returns empty EvalResult
            assert not result.blocked
            assert result.rules_evaluated == 0
            assert result.rules_fired == 0
        finally:
            t_with_content._rule_eval_depth = 0


# ===========================================================================
# Config from rules
# ===========================================================================


class TestConfigFromRules:
    """Tests for config resolution via active rules."""

    def test_config_from_active_rule(self, t: Tract):
        """Active rule provides config through get_config()."""
        t.rule(
            "temp",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.7},
        )
        assert t.get_config("temperature") == 0.7

    def test_config_override_on_branch(self, t: Tract):
        """A branch rule overrides a root-level rule (closest to HEAD wins)."""
        # Create root-level config
        t.user("initial commit")
        t.rule(
            "root_temp",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.5},
        )
        assert t.get_config("temperature") == 0.5

        # Create a branch and add an overriding rule
        from tract.operations.branch import create_branch

        create_branch("feature", t._tract_id, t._ref_repo, t._commit_repo)
        t._session.commit()
        t.switch("feature")

        t.rule(
            "branch_temp",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.9},
        )
        # Branch rule is closer to HEAD, should override root
        assert t.get_config("temperature") == 0.9


# ===========================================================================
# Transition tests
# ===========================================================================


class TestTransition:
    """Tests for Tract.transition() integration."""

    def test_transition_no_rules(self, t_with_content: Tract):
        """No rules: transition creates branch, switches, and commits handoff."""
        result = t_with_content.transition("ads")
        assert result is not None
        assert result.content_type == "dialogue"
        assert "transition handoff" in (result.message or "").lower()
        assert "main" in (result.message or "")
        assert t_with_content.current_branch == "ads"

    def test_transition_blocked_by_block(self, t_with_content: Tract):
        """Block rule prevents transition; returns None."""
        t_with_content.rule(
            "no_ads",
            trigger="transition:ads",
            action={"type": "block", "reason": "ads transitions not allowed"},
        )
        result = t_with_content.transition("ads")
        assert result is None
        # Should remain on original branch
        assert t_with_content.current_branch != "ads"

    def test_transition_blocked_by_require(self, t_with_content: Tract):
        """Unmet require prevents transition."""
        t_with_content.rule(
            "require_many",
            trigger="transition",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">",
                    "value": 99999,
                },
            },
        )
        result = t_with_content.transition("target_branch")
        assert result is None

    def test_transition_with_compile_filter_same_context(
        self, t_with_content: Tract
    ):
        """same_context mode returns None (no handoff needed)."""
        t_with_content.rule(
            "same_ctx",
            trigger="transition:shared",
            action={"type": "compile_filter", "mode": "same_context"},
        )
        result = t_with_content.transition("shared")
        assert result is None

    def test_transition_creates_branch_and_commits(self, t_with_content: Tract):
        """Verifies handoff commit exists on the target branch after transition."""
        original_branch = t_with_content.current_branch
        original_head = t_with_content.head

        result = t_with_content.transition("target")
        assert result is not None

        # Now on target branch
        assert t_with_content.current_branch == "target"

        # Target branch should have the handoff commit (identifiable via message)
        target_log = t_with_content.log()
        handoff_commits = [
            c for c in target_log
            if "transition handoff" in (c.message or "").lower()
        ]
        assert len(handoff_commits) >= 1

        # The handoff commit's message should reference the source branch
        assert original_branch is not None
        assert original_branch in (result.message or "")


# ===========================================================================
# Multiple events in sequence
# ===========================================================================


class TestMultipleEvents:
    """Test firing different events sequentially."""

    def test_multiple_events_sequence(self, t_with_content: Tract):
        """Fire different events; each works independently with its own rules."""
        t_with_content.rule(
            "on_commit",
            trigger="commit",
            action={"type": "set_config", "key": "commit_fired", "value": True},
        )
        t_with_content.rule(
            "on_compile",
            trigger="compile",
            action={"type": "set_config", "key": "compile_fired", "value": True},
        )

        r1 = t_with_content._fire_rules("commit")
        assert not r1.blocked
        config1 = [
            ar for ar in r1.action_results if ar.action_type == "set_config"
        ]
        assert len(config1) == 1
        assert config1[0].data["key"] == "commit_fired"

        r2 = t_with_content._fire_rules("compile")
        assert not r2.blocked
        config2 = [
            ar for ar in r2.action_results if ar.action_type == "set_config"
        ]
        assert len(config2) == 1
        assert config2[0].data["key"] == "compile_fired"

        # An event with no matching rules should be empty
        r3 = t_with_content._fire_rules("gc")
        assert not r3.blocked
        assert r3.rules_evaluated == 0


# ===========================================================================
# Condition types in integration
# ===========================================================================


class TestConditionIntegration:
    """Test condition types through the full rule engine pipeline."""

    def test_tag_condition_integration(self, t_with_content: Tract):
        """Tag condition fires when the triggering commit has a matching tag."""
        t_with_content.register_tag("important", "Important commit")
        t_with_content.rule(
            "on_important",
            trigger="commit",
            condition={"type": "tag", "tag": "important", "present": True},
            action={"type": "set_config", "key": "saw_important", "value": True},
        )

        # Fire with a commit that has the tag
        tagged_commit = t_with_content.user("tagged msg", tags=["important"])
        result = t_with_content._fire_rules("commit", commit=tagged_commit)
        config_results = [
            ar for ar in result.action_results if ar.action_type == "set_config"
        ]
        assert len(config_results) == 1
        assert config_results[0].data["value"] is True

    def test_threshold_condition_integration(self, t_with_content: Tract):
        """Threshold condition on commit_count correctly evaluates."""
        t_with_content.rule(
            "count_check",
            trigger="commit",
            condition={
                "type": "threshold",
                "metric": "commit_count",
                "op": ">=",
                "value": 1,
            },
            action={"type": "set_config", "key": "count_ok", "value": True},
        )
        result = t_with_content._fire_rules("commit")
        config_results = [
            ar for ar in result.action_results if ar.action_type == "set_config"
        ]
        assert len(config_results) == 1

    def test_pattern_condition_integration(self, t_with_content: Tract):
        """Pattern condition matches against commit content."""
        t_with_content.rule(
            "hello_pattern",
            trigger="commit",
            condition={"type": "pattern", "regex": "secret keyword"},
            action={"type": "set_config", "key": "matched", "value": True},
        )

        # Commit with matching content
        matching_commit = t_with_content.user("This has the secret keyword in it")
        result = t_with_content._fire_rules("commit", commit=matching_commit)
        config_results = [
            ar for ar in result.action_results if ar.action_type == "set_config"
        ]
        assert len(config_results) == 1
        assert config_results[0].data["value"] is True

        # Commit without matching content should not fire
        non_matching = t_with_content.user("Nothing special here")
        result2 = t_with_content._fire_rules("commit", commit=non_matching)
        config_results2 = [
            ar for ar in result2.action_results if ar.action_type == "set_config"
        ]
        assert len(config_results2) == 0


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and performance short-circuits."""

    def test_empty_rule_index_short_circuits(self, t: Tract):
        """No rules = no processing cost; returns empty EvalResult immediately."""
        assert len(t.rule_index) == 0
        result = t._fire_rules("commit")
        assert isinstance(result, EvalResult)
        assert not result.blocked
        assert result.rules_evaluated == 0
        assert result.action_results == []

    def test_rule_index_rebuilt_after_rule_commit(self, t: Tract):
        """Rule index is invalidated and rebuilt after a new rule is committed."""
        t.user("initial")
        assert len(t.rule_index) == 0

        t.rule(
            "new_rule",
            trigger="active",
            action={"type": "set_config", "key": "x", "value": 42},
        )

        # After rule commit, index should be stale and rebuilt on next access
        # The rule() method invalidates the index, so next access rebuilds it
        idx = t.rule_index
        assert len(idx) == 1
        assert not idx.is_stale

        # Config resolution should work
        assert t.get_config("x") == 42
