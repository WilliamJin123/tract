"""Rule engine for Tract.

Provides the rule index, condition evaluators, and config resolution.
Rules are first-class commits (RuleContent) that configure behavior
via DAG-scoped precedence.
"""

from tract.rules.conditions import (
    BUILTIN_CONDITIONS,
    AllCondition,
    AnyCondition,
    ConditionEvaluator,
    NotCondition,
    PatternCondition,
    TagCondition,
    ThresholdCondition,
    evaluate_condition,
)
from tract.rules.config import resolve_all_configs, resolve_config
from tract.rules.index import RuleIndex
from tract.rules.models import ActionResult, EvalContext, EvalResult, RuleEntry

__all__ = [
    "RuleIndex",
    "RuleEntry",
    "EvalContext",
    "ActionResult",
    "EvalResult",
    "evaluate_condition",
    "resolve_config",
    "resolve_all_configs",
    "BUILTIN_CONDITIONS",
    "ConditionEvaluator",
    "TagCondition",
    "PatternCondition",
    "ThresholdCondition",
    "AllCondition",
    "AnyCondition",
    "NotCondition",
]
