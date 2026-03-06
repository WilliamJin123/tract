"""Config resolution from active rules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tract.rules.index import RuleIndex


def resolve_config(
    rule_index: RuleIndex,
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Resolve a config value from active rules.

    Looks through rules with trigger="active" for set_config actions
    matching the given key. Returns the value from the closest rule
    (lowest dag_distance). If no rule matches, returns default.
    """
    value = rule_index.get_config(key)
    if value is None:
        return default
    return value


def resolve_all_configs(rule_index: RuleIndex) -> dict[str, Any]:
    """Resolve all active config key-value pairs."""
    return rule_index.get_all_configs()
