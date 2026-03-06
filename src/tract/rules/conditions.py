"""Condition evaluators for the rule engine."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from tract.rules.models import EvalContext


class ConditionEvaluator(Protocol):
    """Protocol for condition type evaluators."""

    def evaluate(self, params: dict, ctx: EvalContext) -> bool: ...


class TagCondition:
    """Check if a tag exists on the triggering commit or in scope."""

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        tag = params.get("tag", "")
        present = params.get("present", True)
        if ctx.commit is None:
            return not present
        commit_tags = set(ctx.commit.tags) if ctx.commit.tags else set()
        has_tag = tag in commit_tags
        return has_tag if present else not has_tag


class PatternCondition:
    """Regex match on commit content."""

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        regex = params.get("regex", "")
        if ctx.commit is None:
            return False
        # Get commit content text from blob
        content = _get_commit_text(ctx)
        return bool(re.search(regex, content))


class ThresholdCondition:
    """Numeric comparison on a metric."""

    VALID_OPS = {">", "<", "==", ">=", "<=", "!="}

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        metric_name = params.get("metric", "")
        op = params.get("op", "")
        if op not in self.VALID_OPS:
            raise ValueError(
                f"Invalid threshold operator: {op!r}. Valid: {sorted(self.VALID_OPS)}"
            )
        value = params.get("value", 0)
        actual = _get_metric(metric_name, ctx)
        return _compare(actual, op, value)


class AllCondition:
    """AND combinator."""

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        conditions = params.get("conditions", [])
        return all(evaluate_condition(c, ctx) for c in conditions)


class AnyCondition:
    """OR combinator."""

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        conditions = params.get("conditions", [])
        return any(evaluate_condition(c, ctx) for c in conditions)


class NotCondition:
    """Negation."""

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        condition = params.get("condition")
        return not evaluate_condition(condition, ctx)


class LLMCondition:
    """LLM-evaluated condition. Expensive, evaluated last.

    Placeholder: passes through (True) when no LLM client is available.
    Real implementation deferred to R4.
    """

    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        llm = getattr(ctx.tract, "_llm_client", None)
        if llm is None:
            return True
        # Real implementation: send instruction + context to LLM, parse bool
        return True


# Registry of built-in condition evaluators
BUILTIN_CONDITIONS: dict[str, ConditionEvaluator] = {
    "tag": TagCondition(),
    "pattern": PatternCondition(),
    "threshold": ThresholdCondition(),
    "all": AllCondition(),
    "any": AnyCondition(),
    "not": NotCondition(),
    "llm": LLMCondition(),
}


def evaluate_condition(
    condition: dict | None,
    ctx: EvalContext,
    *,
    custom_conditions: dict[str, ConditionEvaluator] | None = None,
) -> bool:
    """Evaluate a condition dict. None = always True."""
    if condition is None:
        return True
    ctype = condition.get("type")
    evaluator = (custom_conditions or {}).get(ctype) or BUILTIN_CONDITIONS.get(ctype)
    if evaluator is None:
        raise ValueError(f"Unknown condition type: {ctype!r}")
    return evaluator.evaluate(condition, ctx)


# ---------------------------------------------------------------------------
# Built-in metric resolution
# ---------------------------------------------------------------------------


def _get_metric(name: str, ctx: EvalContext) -> float:
    """Resolve a built-in metric value.

    IMPORTANT: Never call ctx.tract.compile() here -- this function may be
    called during a "compile" event, which would cause infinite recursion.
    All metrics must use pre-computed values from ctx.metrics or lightweight
    queries (log length, commit fields).

    NOTE (cold start): total_tokens defaults to 0 when no compile cache exists
    (first compile in a session). This means token-based threshold rules will
    not fire until after the first compile populates the cache. This is by
    design to avoid expensive traversals on every event fire.
    """
    # Check pre-computed metrics first (set by _fire_rules in tract.py)
    if ctx.metrics and name in ctx.metrics:
        return ctx.metrics[name]
    if name == "token_count" and ctx.commit:
        return ctx.commit.token_count or 0
    if name == "total_tokens":
        return (ctx.metrics or {}).get("total_tokens", 0)
    if name == "commit_count":
        return len(ctx.tract.log())
    if name == "age_hours" and ctx.commit:
        delta = datetime.now(timezone.utc) - ctx.commit.created_at
        return delta.total_seconds() / 3600
    if name == "branch_depth":
        return len(ctx.tract.log())
    raise ValueError(f"Unknown metric: {name!r}")


def _compare(actual: float, op: str, value: float) -> bool:
    """Execute a comparison operation."""
    if op == ">":
        return actual > value
    if op == "<":
        return actual < value
    if op == "==":
        return actual == value
    if op == ">=":
        return actual >= value
    if op == "<=":
        return actual <= value
    if op == "!=":
        return actual != value
    raise ValueError(f"Invalid operator: {op!r}")


def _get_commit_text(ctx: EvalContext) -> str:
    """Extract text content from a commit for pattern matching."""
    if ctx.commit is None:
        return ""
    # Load blob content via tract's internal repos
    blob = ctx.tract._blob_repo.get(ctx.commit.content_hash)
    if blob is None:
        return ""
    import json

    try:
        payload = json.loads(blob.payload_json)
    except (json.JSONDecodeError, TypeError):
        return ""
    # Extract text from common content fields
    return payload.get("text", "") or payload.get("content", "") or str(payload.get("payload", ""))
