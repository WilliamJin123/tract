"""Tests for rule engine condition evaluators."""

import pytest

from tract import Tract, EvalContext
from tract.rules.conditions import (
    evaluate_condition,
    TagCondition,
    PatternCondition,
    ThresholdCondition,
    AllCondition,
    AnyCondition,
    NotCondition,
    BUILTIN_CONDITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ctx(t, info, *, metrics=None):
    """Build an EvalContext from a Tract and a CommitInfo."""
    return EvalContext(
        event="commit",
        commit=info,
        branch="main",
        head=info.commit_hash,
        tract=t,
        metrics=metrics,
    )


def _open_with_tags(path, tags):
    """Open a Tract and register the given tag names."""
    t = Tract.open(str(path))
    for tag in tags:
        t.register_tag(tag, description=f"test tag: {tag}")
    return t


# ---------------------------------------------------------------------------
# 1. None condition always True
# ---------------------------------------------------------------------------


class TestNoneCondition:
    def test_none_condition_always_true(self, tmp_path):
        t = Tract.open(str(tmp_path / "t1"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info)
        assert evaluate_condition(None, ctx) is True


# ---------------------------------------------------------------------------
# 2-4. Tag conditions
# ---------------------------------------------------------------------------


class TestTagCondition:
    def test_tag_present_true(self, tmp_path):
        t = _open_with_tags(tmp_path / "t2", ["important"])
        info = t.user("Hi", tags=["important"])
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "tag", "tag": "important", "present": True}, ctx
        )
        assert result is True

    def test_tag_present_false_when_missing(self, tmp_path):
        t = _open_with_tags(tmp_path / "t3", ["other"])
        info = t.user("Hi", tags=["other"])
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "tag", "tag": "important", "present": True}, ctx
        )
        assert result is False

    def test_tag_absent_check(self, tmp_path):
        """present=False should return True when the tag is NOT on the commit."""
        t = _open_with_tags(tmp_path / "t4", ["other"])
        info = t.user("Hi", tags=["other"])
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "tag", "tag": "important", "present": False}, ctx
        )
        assert result is True

    def test_tag_absent_check_when_present(self, tmp_path):
        """present=False should return False when the tag IS on the commit."""
        t = _open_with_tags(tmp_path / "t4b", ["important"])
        info = t.user("Hi", tags=["important"])
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "tag", "tag": "important", "present": False}, ctx
        )
        assert result is False


# ---------------------------------------------------------------------------
# 5-7. Pattern conditions
# ---------------------------------------------------------------------------


class TestPatternCondition:
    def test_pattern_match(self, tmp_path):
        t = Tract.open(str(tmp_path / "t5"))
        info = t.user("Please summarize this document")
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "pattern", "regex": "summarize"}, ctx
        )
        assert result is True

    def test_pattern_no_match(self, tmp_path):
        t = Tract.open(str(tmp_path / "t6"))
        info = t.user("Hello world")
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "pattern", "regex": "summarize"}, ctx
        )
        assert result is False

    def test_pattern_complex_regex(self, tmp_path):
        t = Tract.open(str(tmp_path / "t7"))
        info = t.user("Error code: ERR-4502 detected")
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "pattern", "regex": r"ERR-\d{4}"}, ctx
        )
        assert result is True


# ---------------------------------------------------------------------------
# 8-18. Threshold conditions
# ---------------------------------------------------------------------------


class TestThresholdCondition:
    def test_greater_than(self, tmp_path):
        t = Tract.open(str(tmp_path / "t8"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 5000})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": ">", "value": 4000},
            ctx,
        )
        assert result is True

    def test_less_than(self, tmp_path):
        t = Tract.open(str(tmp_path / "t9"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 1000})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": "<", "value": 2000},
            ctx,
        )
        assert result is True

    def test_equal(self, tmp_path):
        t = Tract.open(str(tmp_path / "t10"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 3000})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": "==", "value": 3000},
            ctx,
        )
        assert result is True

    def test_greater_than_or_equal(self, tmp_path):
        t = Tract.open(str(tmp_path / "t11"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 4000})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": ">=", "value": 4000},
            ctx,
        )
        assert result is True

    def test_less_than_or_equal(self, tmp_path):
        t = Tract.open(str(tmp_path / "t12"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 2000})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": "<=", "value": 2000},
            ctx,
        )
        assert result is True

    def test_not_equal(self, tmp_path):
        t = Tract.open(str(tmp_path / "t13"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 5000})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": "!=", "value": 3000},
            ctx,
        )
        assert result is True

    def test_token_count_from_commit(self, tmp_path):
        """token_count metric should fall through to commit.token_count."""
        t = Tract.open(str(tmp_path / "t14"))
        info = t.user("Hello world")
        ctx = _make_ctx(t, info)
        # token_count should be > 0 for a non-empty commit
        result = evaluate_condition(
            {"type": "threshold", "metric": "token_count", "op": ">", "value": 0},
            ctx,
        )
        assert result is True

    def test_total_tokens_from_metrics(self, tmp_path):
        """total_tokens should come from ctx.metrics."""
        t = Tract.open(str(tmp_path / "t15"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 7500})
        result = evaluate_condition(
            {"type": "threshold", "metric": "total_tokens", "op": ">", "value": 7000},
            ctx,
        )
        assert result is True

    def test_commit_count_from_log(self, tmp_path):
        """commit_count should use len(tract.log())."""
        t = Tract.open(str(tmp_path / "t16"))
        t.user("One")
        t.user("Two")
        info = t.user("Three")
        ctx = _make_ctx(t, info)
        result = evaluate_condition(
            {"type": "threshold", "metric": "commit_count", "op": "==", "value": 3},
            ctx,
        )
        assert result is True

    def test_unknown_metric_raises(self, tmp_path):
        t = Tract.open(str(tmp_path / "t17"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info)
        with pytest.raises(ValueError, match="Unknown metric"):
            evaluate_condition(
                {"type": "threshold", "metric": "nonexistent_metric", "op": ">", "value": 0},
                ctx,
            )

    def test_invalid_operator_raises(self, tmp_path):
        t = Tract.open(str(tmp_path / "t18"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 100})
        with pytest.raises(ValueError, match="Invalid threshold operator"):
            evaluate_condition(
                {"type": "threshold", "metric": "total_tokens", "op": "~", "value": 0},
                ctx,
            )


# ---------------------------------------------------------------------------
# 19-20. All combinator
# ---------------------------------------------------------------------------


class TestAllCondition:
    def test_all_true(self, tmp_path):
        t = _open_with_tags(tmp_path / "t19", ["important"])
        info = t.user("Hello", tags=["important"])
        ctx = _make_ctx(t, info, metrics={"total_tokens": 5000})
        cond = {
            "type": "all",
            "conditions": [
                {"type": "tag", "tag": "important", "present": True},
                {"type": "threshold", "metric": "total_tokens", "op": ">", "value": 1000},
            ],
        }
        assert evaluate_condition(cond, ctx) is True

    def test_all_one_false(self, tmp_path):
        t = _open_with_tags(tmp_path / "t20", ["important"])
        info = t.user("Hello", tags=["important"])
        ctx = _make_ctx(t, info, metrics={"total_tokens": 500})
        cond = {
            "type": "all",
            "conditions": [
                {"type": "tag", "tag": "important", "present": True},
                {"type": "threshold", "metric": "total_tokens", "op": ">", "value": 1000},
            ],
        }
        assert evaluate_condition(cond, ctx) is False


# ---------------------------------------------------------------------------
# 21-22. Any combinator
# ---------------------------------------------------------------------------


class TestAnyCondition:
    def test_any_one_true(self, tmp_path):
        t = Tract.open(str(tmp_path / "t21"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 500})
        cond = {
            "type": "any",
            "conditions": [
                {"type": "tag", "tag": "important", "present": True},  # False
                {"type": "threshold", "metric": "total_tokens", "op": "<", "value": 1000},  # True
            ],
        }
        assert evaluate_condition(cond, ctx) is True

    def test_any_all_false(self, tmp_path):
        t = Tract.open(str(tmp_path / "t22"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info, metrics={"total_tokens": 5000})
        cond = {
            "type": "any",
            "conditions": [
                {"type": "tag", "tag": "important", "present": True},  # False
                {"type": "threshold", "metric": "total_tokens", "op": "<", "value": 1000},  # False
            ],
        }
        assert evaluate_condition(cond, ctx) is False


# ---------------------------------------------------------------------------
# 23. Not combinator
# ---------------------------------------------------------------------------


class TestNotCondition:
    def test_not_inverts_result(self, tmp_path):
        t = _open_with_tags(tmp_path / "t23", ["draft"])
        info = t.user("Hello", tags=["draft"])
        ctx = _make_ctx(t, info)
        # tag "draft" is present, so tag check returns True; NOT inverts to False
        cond = {
            "type": "not",
            "condition": {"type": "tag", "tag": "draft", "present": True},
        }
        assert evaluate_condition(cond, ctx) is False

    def test_not_inverts_false_to_true(self, tmp_path):
        t = Tract.open(str(tmp_path / "t23b"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info)
        # tag "draft" is NOT present, so tag check returns False; NOT inverts to True
        cond = {
            "type": "not",
            "condition": {"type": "tag", "tag": "draft", "present": True},
        }
        assert evaluate_condition(cond, ctx) is True


# ---------------------------------------------------------------------------
# 24. Nested combinator
# ---------------------------------------------------------------------------


class TestNestedConditions:
    def test_all_with_nested_any(self, tmp_path):
        """all(tag, any(pattern, threshold)) -- tag True, pattern True -> True."""
        t = _open_with_tags(tmp_path / "t24", ["important"])
        info = t.user("Please summarize", tags=["important"])
        ctx = _make_ctx(t, info, metrics={"total_tokens": 500})
        cond = {
            "type": "all",
            "conditions": [
                {"type": "tag", "tag": "important", "present": True},
                {
                    "type": "any",
                    "conditions": [
                        {"type": "pattern", "regex": "summarize"},  # True
                        {"type": "threshold", "metric": "total_tokens", "op": ">", "value": 9000},  # False
                    ],
                },
            ],
        }
        assert evaluate_condition(cond, ctx) is True

    def test_nested_all_false_when_inner_any_false(self, tmp_path):
        """all(tag, any(pattern, threshold)) -- tag True, both inner False -> False."""
        t = _open_with_tags(tmp_path / "t24b", ["important"])
        info = t.user("Hello world", tags=["important"])
        ctx = _make_ctx(t, info, metrics={"total_tokens": 500})
        cond = {
            "type": "all",
            "conditions": [
                {"type": "tag", "tag": "important", "present": True},
                {
                    "type": "any",
                    "conditions": [
                        {"type": "pattern", "regex": "summarize"},  # False
                        {"type": "threshold", "metric": "total_tokens", "op": ">", "value": 9000},  # False
                    ],
                },
            ],
        }
        assert evaluate_condition(cond, ctx) is False


# ---------------------------------------------------------------------------
# 25. Unknown condition type raises ValueError
# ---------------------------------------------------------------------------


class TestUnknownConditionType:
    def test_unknown_type_raises(self, tmp_path):
        t = Tract.open(str(tmp_path / "t25"))
        info = t.user("Hello")
        ctx = _make_ctx(t, info)
        with pytest.raises(ValueError, match="Unknown condition type"):
            evaluate_condition({"type": "bogus_type"}, ctx)
