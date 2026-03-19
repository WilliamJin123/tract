"""Policy -- unified primitive for context management decisions.

A Policy combines a condition (when to fire) with a strategy (what to do).
Either side can be deterministic (a plain callable) or semantic (a Judgment
that queries an LLM). This gives a 2x2 matrix:

    +----------------------+-------------------------+
    | Det. condition        | Semantic condition       |
    | + Det. strategy       | + Det. strategy          |
    +----------------------+-------------------------+
    | Det. condition        | Semantic condition       |
    | + Semantic strategy   | + Semantic strategy      |
    +----------------------+-------------------------+

Policies replace the current separate systems: SemanticGate (blocking policy),
SemanticMaintainer (action policy), middleware handlers (deterministic policy),
and standalone intelligence/autonomous/routing functions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from tract.middleware import VALID_EVENTS

if TYPE_CHECKING:
    from tract.tract import Tract

__all__: list[str] = [
    "PolicyContext",
    "PolicyOutcome",
    "Evaluable",
    "Policy",
    "PolicyEngine",
    "always",
    "never",
    "token_ratio_above",
    "commit_count_above",
    "block_with_reason",
    "pass_through",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyContext:
    """Immutable snapshot passed to policy conditions and strategies."""

    tract: Any  # Tract instance (Any to avoid circular import)
    event: str | None = None  # MiddlewareEvent that triggered this, or None for manual
    trigger_data: Any = None  # Event-specific data (pending content for pre_commit, etc.)
    branch: str = ""  # Current branch name
    head: str = ""  # Current HEAD hash


@dataclass(frozen=True)
class PolicyOutcome:
    """Result of policy strategy evaluation."""

    triggered: bool  # Whether the condition was met
    actions: tuple[dict, ...] = ()  # Actions to execute (same schema as MaintenanceAction)
    block: bool = False  # Whether to raise BlockedError
    block_reason: str = ""  # Reason for blocking
    reasoning: str = ""  # Human-readable explanation
    tokens_used: int = 0  # LLM tokens consumed (0 for deterministic)


# ---------------------------------------------------------------------------
# Duck-type protocol for Judgment interop
# ---------------------------------------------------------------------------


@runtime_checkable
class Evaluable(Protocol):
    """Anything with an evaluate() method -- matches Judgment.

    Used for duck-typing: ``isinstance(obj, Evaluable)`` returns True
    for any object exposing ``evaluate(tract, *, llm_client=...)``.
    The Judgment class (built in parallel) satisfies this protocol.
    """

    def evaluate(self, tract: Any, *, llm_client: Any | None = None) -> Any: ...


# ---------------------------------------------------------------------------
# Judgment result -> PolicyOutcome conversion
# ---------------------------------------------------------------------------


def _judgment_result_to_outcome(result: Any) -> PolicyOutcome:
    """Convert a JudgmentResult to a PolicyOutcome.

    Handles multiple response model shapes via duck typing:
    - GateVerdict-style (has ``result`` field): "fail" means block
    - MaintenancePlan-style (has ``actions`` list): actions are extracted,
      ``block`` actions cause blocking
    - Any object with ``reasoning``: used for human-readable explanation
    """
    output = result.output
    if output is None:
        return PolicyOutcome(
            triggered=True,
            reasoning=getattr(result, "reasoning", ""),
            tokens_used=getattr(result, "tokens_used", 0),
        )

    # Defaults
    block = False
    block_reason = ""
    actions: tuple[dict, ...] = ()
    reasoning = getattr(result, "reasoning", "")

    # GateVerdict-style: ``result`` field determines block
    if hasattr(output, "result"):
        block = str(output.result).lower() == "fail"
        block_reason = getattr(output, "reason", "") or reasoning

    # MaintenancePlan-style: ``actions`` list
    if hasattr(output, "actions"):
        raw_actions = output.actions
        action_dicts: list[dict] = []
        for a in raw_actions:
            if isinstance(a, dict):
                action_dicts.append(a)
            elif hasattr(a, "model_dump"):
                action_dicts.append(a.model_dump(exclude_none=True))
            else:
                action_dicts.append(vars(a))

        non_block = [a for a in action_dicts if a.get("type") != "block"]
        block_actions = [a for a in action_dicts if a.get("type") == "block"]
        if block_actions:
            block = True
            block_reason = "; ".join(
                a.get("reason", "Blocked by policy") for a in block_actions
            )
        actions = tuple(non_block)

    # Override reasoning from output if available
    if hasattr(output, "reasoning"):
        reasoning = output.reasoning

    return PolicyOutcome(
        triggered=True,
        actions=actions,
        block=block,
        block_reason=block_reason,
        reasoning=reasoning,
        tokens_used=getattr(result, "tokens_used", 0),
    )


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class Policy:
    """A context management policy.

    Combines a condition (when to fire) with a strategy (what to do).
    Either can be a plain callable (deterministic) or an Evaluable/Judgment
    (semantic).

    Attributes:
        name: Unique identifier for this policy.
        condition: Callable(PolicyContext)->bool or Evaluable for semantic check.
        strategy: Callable(PolicyContext)->PolicyOutcome or Evaluable for semantic action.
        priority: Higher priority = evaluated first when multiple policies match.
        fail_open: On error, skip rather than crash.
        enabled: Can be toggled without removing from the engine.
    """

    name: str
    condition: Callable[[PolicyContext], bool] | Evaluable
    strategy: Callable[[PolicyContext], PolicyOutcome] | Evaluable
    priority: int = 0
    fail_open: bool = True
    enabled: bool = True

    def evaluate(self, ctx: PolicyContext) -> PolicyOutcome:
        """Evaluate this policy: check condition, then run strategy if triggered.

        Returns a PolicyOutcome. If the policy is disabled, returns
        ``PolicyOutcome(triggered=False)`` immediately. Errors are handled
        according to ``fail_open``: if True, the policy is skipped on error;
        if False, the exception propagates.
        """
        if not self.enabled:
            return PolicyOutcome(triggered=False)

        # --- 1. Evaluate condition -------------------------------------------
        try:
            if isinstance(self.condition, Evaluable):
                # Semantic condition: call evaluate(), interpret result
                result = self.condition.evaluate(ctx.tract)
                output = result.output
                if output is None:
                    condition_met = False  # fail-open: don't trigger
                elif hasattr(output, "result"):
                    # GateVerdict-style: "fail" means condition IS met
                    condition_met = str(output.result).lower() != "pass"
                elif hasattr(output, "decision"):
                    condition_met = bool(output.decision)
                else:
                    condition_met = bool(output)
            else:
                # Deterministic condition
                condition_met = bool(self.condition(ctx))
        except Exception:
            if self.fail_open:
                logger.warning(
                    "Policy '%s' condition failed; skipping (fail-open)",
                    self.name,
                    exc_info=True,
                )
                return PolicyOutcome(
                    triggered=False,
                    reasoning="Condition evaluation failed (fail-open)",
                )
            raise

        if not condition_met:
            return PolicyOutcome(triggered=False)

        # --- 2. Evaluate strategy --------------------------------------------
        try:
            if isinstance(self.strategy, Evaluable):
                # Semantic strategy: call evaluate(), convert result
                result = self.strategy.evaluate(ctx.tract)
                if result.output is None:
                    return PolicyOutcome(
                        triggered=True,
                        reasoning="Strategy evaluation failed (fail-open)",
                        tokens_used=getattr(result, "tokens_used", 0),
                    )
                return _judgment_result_to_outcome(result)
            else:
                # Deterministic strategy
                outcome = self.strategy(ctx)
                if not isinstance(outcome, PolicyOutcome):
                    # Allow bare callables that return bool (simple block/pass)
                    return PolicyOutcome(triggered=True, block=bool(outcome))
                return outcome
        except Exception:
            if self.fail_open:
                logger.warning(
                    "Policy '%s' strategy failed; skipping (fail-open)",
                    self.name,
                    exc_info=True,
                )
                return PolicyOutcome(
                    triggered=True,
                    reasoning="Strategy evaluation failed (fail-open)",
                )
            raise


# ---------------------------------------------------------------------------
# PolicyEngine
# ---------------------------------------------------------------------------


class PolicyEngine:
    """Manages policies and dispatches them on events.

    Replaces MiddlewareManager. Policies are registered globally or on
    specific events. When an event fires, matching policies are evaluated
    in priority order (highest first).
    """

    def __init__(self) -> None:
        self._policies: dict[str, Policy] = {}  # name -> Policy
        self._event_bindings: dict[str, list[str]] = {}  # event -> [policy_names]
        self._global_policies: list[str] = []  # policies that fire on all events
        self._recursion_guard: set[str] = set()  # prevent re-entrant firing

    # -- Registration --------------------------------------------------------

    def add(self, policy: Policy, *, event: str | None = None) -> None:
        """Register a policy.

        Args:
            policy: The policy to register.
            event: If provided, policy fires only on this event.
                   If None, policy is manual-only (call evaluate_one directly).

        Raises:
            ValueError: If a policy with the same name is already registered,
                or if the event name is not in VALID_EVENTS.
        """
        if policy.name in self._policies:
            raise ValueError(
                f"Policy '{policy.name}' already registered. Remove it first."
            )
        self._policies[policy.name] = policy
        if event is not None:
            if event not in VALID_EVENTS:
                raise ValueError(
                    f"Unknown event '{event}'. Valid: {sorted(VALID_EVENTS)}"
                )
            self._event_bindings.setdefault(event, []).append(policy.name)

    def remove(self, name: str) -> None:
        """Remove a policy by name.

        Raises:
            KeyError: If no policy with that name exists.
        """
        if name not in self._policies:
            raise KeyError(f"No policy named '{name}'")
        del self._policies[name]
        for event_names in self._event_bindings.values():
            if name in event_names:
                event_names.remove(name)
        if name in self._global_policies:
            self._global_policies.remove(name)

    # -- Introspection -------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        """List all registered policies with their event bindings.

        Returns a list of dicts sorted by descending priority.
        """
        result: list[dict[str, Any]] = []
        for name, policy in self._policies.items():
            events = [
                ev for ev, names in self._event_bindings.items() if name in names
            ]
            result.append(
                {
                    "name": name,
                    "priority": policy.priority,
                    "enabled": policy.enabled,
                    "events": events,
                    "fail_open": policy.fail_open,
                }
            )
        return sorted(result, key=lambda x: -x["priority"])

    def get(self, name: str) -> Policy | None:
        """Get a policy by name, or None if not found."""
        return self._policies.get(name)

    # -- Dispatching ---------------------------------------------------------

    def fire(self, event: str, ctx: PolicyContext) -> list[PolicyOutcome]:
        """Fire all policies bound to an event. Returns outcomes.

        Policies are evaluated in priority order (highest first).
        If any policy blocks, :class:`~tract.exceptions.BlockedError` is
        raised AFTER all policies have been evaluated (so all outcomes are
        collected before raising).

        Re-entrant calls for the same event are silently skipped to prevent
        infinite recursion (e.g., a policy that commits triggering pre_commit).
        """
        if event in self._recursion_guard:
            return []  # Prevent re-entrant firing

        self._recursion_guard.add(event)
        try:
            return self._fire_impl(event, ctx)
        finally:
            self._recursion_guard.discard(event)

    def _fire_impl(self, event: str, ctx: PolicyContext) -> list[PolicyOutcome]:
        """Internal: evaluate policies and collect outcomes."""
        policy_names = self._event_bindings.get(event, [])
        if not policy_names:
            return []

        # Resolve to Policy objects, filter disabled, sort by priority (desc)
        policies = [
            self._policies[n]
            for n in policy_names
            if n in self._policies and self._policies[n].enabled
        ]
        policies.sort(key=lambda p: -p.priority)

        outcomes: list[PolicyOutcome] = []
        block_reasons: list[str] = []

        for policy in policies:
            outcome = policy.evaluate(ctx)
            outcomes.append(outcome)
            if outcome.block:
                block_reasons.append(
                    f"Policy '{policy.name}': {outcome.block_reason}"
                )

        # Raise BlockedError if any policy blocked
        if block_reasons:
            from tract.exceptions import BlockedError

            raise BlockedError(event, block_reasons)

        return outcomes

    def evaluate_one(self, name: str, ctx: PolicyContext) -> PolicyOutcome:
        """Evaluate a single named policy manually.

        Useful for policies registered without an event binding, or for
        testing a policy outside the event dispatch flow.

        Raises:
            KeyError: If no policy with that name exists.
        """
        policy = self._policies.get(name)
        if policy is None:
            raise KeyError(f"No policy named '{name}'")
        return policy.evaluate(ctx)


# ---------------------------------------------------------------------------
# Convenience condition factories
# ---------------------------------------------------------------------------


def always(_ctx: PolicyContext) -> bool:
    """Condition that always returns True. Use for policies that fire every time."""
    return True


def never(_ctx: PolicyContext) -> bool:
    """Condition that always returns False. Use for disabled policies."""
    return False


def token_ratio_above(threshold: float) -> Callable[[PolicyContext], bool]:
    """Create a condition that fires when token usage ratio exceeds *threshold*.

    The ratio is ``current_tokens / max_tokens`` from ``t.search.status()``.
    Returns False if the ratio cannot be determined.
    """

    def check(ctx: PolicyContext) -> bool:
        try:
            status = ctx.tract.search.status()
            if hasattr(status, "token_ratio") and status.token_ratio is not None:
                return status.token_ratio > threshold
        except Exception:
            pass
        return False

    return check


def commit_count_above(threshold: int) -> Callable[[PolicyContext], bool]:
    """Create a condition that fires when the commit count exceeds *threshold*."""

    def check(ctx: PolicyContext) -> bool:
        try:
            status = ctx.tract.search.status()
            return status.commit_count > threshold
        except Exception:
            return False

    return check


# ---------------------------------------------------------------------------
# Convenience strategy factories
# ---------------------------------------------------------------------------


def block_with_reason(reason: str) -> Callable[[PolicyContext], PolicyOutcome]:
    """Create a strategy that always blocks with the given reason."""

    def strategy(_ctx: PolicyContext) -> PolicyOutcome:
        return PolicyOutcome(triggered=True, block=True, block_reason=reason)

    return strategy


def pass_through(_ctx: PolicyContext) -> PolicyOutcome:
    """Strategy that does nothing (triggered but no action)."""
    return PolicyOutcome(triggered=True)
