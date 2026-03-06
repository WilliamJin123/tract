"""Rule engine for Tract.

Provides the rule index, condition evaluators, action handlers,
event processing engine, and config resolution.
Rules are first-class commits (RuleContent) that configure behavior
via DAG-scoped precedence.
"""

from tract.rules.actions import (
    ACTION_CATEGORIES,
    ACTION_SEMANTICS,
    BUILTIN_ACTIONS,
    ActionHandler,
    BlockAction,
    CompileFilterAction,
    CreateRuleAction,
    LLMAction,
    OperationAction,
    RequireAction,
    SetConfigAction,
)
from tract.rules.conditions import (
    BUILTIN_CONDITIONS,
    AllCondition,
    AnyCondition,
    ConditionEvaluator,
    LLMCondition,
    NotCondition,
    PatternCondition,
    TagCondition,
    ThresholdCondition,
    evaluate_condition,
)
from tract.rules.config import resolve_all_configs, resolve_config
from tract.rules.engine import RuleEngine
from tract.rules.index import RuleIndex
from tract.rules.models import ActionResult, EvalContext, EvalResult, RuleEntry

__all__ = [
    "RuleIndex",
    "RuleEntry",
    "EvalContext",
    "ActionResult",
    "EvalResult",
    "RuleEngine",
    "evaluate_condition",
    "resolve_config",
    "resolve_all_configs",
    "BUILTIN_CONDITIONS",
    "BUILTIN_ACTIONS",
    "ACTION_CATEGORIES",
    "ACTION_SEMANTICS",
    "ConditionEvaluator",
    "ActionHandler",
    "TagCondition",
    "PatternCondition",
    "ThresholdCondition",
    "AllCondition",
    "AnyCondition",
    "NotCondition",
    "LLMCondition",
    "SetConfigAction",
    "OperationAction",
    "BlockAction",
    "RequireAction",
    "CompileFilterAction",
    "LLMAction",
    "CreateRuleAction",
]
