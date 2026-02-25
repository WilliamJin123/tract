"""PolicyEvaluator -- sidecar class that evaluates policies against a Tract.

The evaluator iterates registered policies sorted by priority, dispatches
actions based on autonomy level, and logs audit entries for every evaluation.

Recursion guard: if a policy's action triggers compile() or commit(),
the evaluator will not re-enter evaluate() (same pattern as Tract._in_batch).

Phase 2 integration: collaborative mode now routes through the hook system
via PendingPolicy. The old PolicyProposal / on_proposal callback pattern
is replaced. Three-tier handler precedence:
    1. User hook (t.on("policy", handler)) -- highest priority
    2. Policy.default_handler() -- policy-specific review logic
    3. Auto-approve -- if no hook and no default_handler override
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from tract.models.annotations import Priority
from tract.models.policy import EvaluationResult, PolicyAction
from tract.policy.protocols import Policy

if TYPE_CHECKING:
    from collections.abc import Callable

    from tract.hooks.policy import PendingPolicy
    from tract.hooks.validation import HookRejection
    from tract.storage.schema import PolicyProposalRow
    from tract.storage.sqlite import SqlitePolicyRepository
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class PolicyEvaluator:
    """Evaluates policies against a Tract instance.

    Supports the full autonomy spectrum:
    - **autonomous**: executes actions immediately
    - **collaborative**: creates PendingPolicy, routes through hook system
    - **manual/supervised**: skips execution, logs only

    Features:
    - Priority-sorted evaluation (lower priority runs first)
    - Recursion guard (prevents infinite loops from policies triggering compile/commit)
    - Cooldown tracking (prevents rapid re-firing)
    - Audit logging (every evaluation logged to DB if policy_repo available)
    """

    def __init__(
        self,
        tract: Tract,
        policies: list[Policy] | None = None,
        policy_repo: SqlitePolicyRepository | None = None,
        on_proposal: Callable | None = None,
        cooldown_seconds: float = 0,
    ) -> None:
        self._tract = tract
        self._policies: list[Policy] = sorted(
            policies or [], key=lambda p: p.priority
        )
        self._policy_repo = policy_repo
        self._on_proposal = on_proposal  # Kept for backward compatibility
        self._paused: bool = False
        self._evaluating: bool = False
        self._last_fired: dict[str, datetime] = {}
        self._cooldown_seconds: float = cooldown_seconds

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, policy: Policy) -> None:
        """Add a policy to the evaluator, maintaining priority sort order."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority)

    def unregister(self, policy_name: str) -> None:
        """Remove a policy by name."""
        self._policies = [p for p in self._policies if p.name != policy_name]

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause all policy evaluation (emergency kill switch)."""
        self._paused = True

    def resume(self) -> None:
        """Resume policy evaluation."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Whether the evaluator is currently paused."""
        return self._paused

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, trigger: str = "compile") -> list[EvaluationResult]:
        """Evaluate all registered policies matching the given trigger.

        Args:
            trigger: The evaluation trigger -- "compile" or "commit".

        Returns:
            List of EvaluationResult for each policy that matched the trigger.
        """
        if self._paused or self._evaluating:
            return []

        self._evaluating = True
        try:
            return self._do_evaluate(trigger)
        finally:
            self._evaluating = False

    def _do_evaluate(self, trigger: str) -> list[EvaluationResult]:
        """Internal evaluation loop (called inside recursion guard)."""
        results: list[EvaluationResult] = []

        # Filter policies matching this trigger
        matching = [p for p in self._policies if p.trigger == trigger]

        # Get pending proposal policy names (for dedup) -- backward compat
        pending_names: set[str] = set()
        if self._policy_repo is not None:
            pending_rows = self._policy_repo.get_pending_proposals(
                self._tract.tract_id
            )
            pending_names = {row.policy_name for row in pending_rows}

        for policy in matching:
            # Check cooldown
            if self._cooldown_seconds > 0 and policy.name in self._last_fired:
                elapsed = (
                    datetime.now() - self._last_fired[policy.name]
                ).total_seconds()
                if elapsed < self._cooldown_seconds:
                    result = EvaluationResult(
                        policy_name=policy.name,
                        triggered=False,
                        outcome="skipped",
                    )
                    results.append(result)
                    continue

            # Check pending proposal dedup
            if policy.name in pending_names:
                result = EvaluationResult(
                    policy_name=policy.name,
                    triggered=False,
                    outcome="skipped",
                )
                results.append(result)
                continue

            # Evaluate the policy
            try:
                action = policy.evaluate(self._tract)
            except Exception as exc:
                logger.error(
                    "Policy '%s' raised %s: %s",
                    policy.name,
                    type(exc).__name__,
                    exc,
                )
                result = EvaluationResult(
                    policy_name=policy.name,
                    triggered=True,
                    outcome="error",
                    error=str(exc),
                )
                self._log_evaluation(policy, trigger, result)
                results.append(result)
                continue

            if action is None:
                # Policy didn't fire
                result = EvaluationResult(
                    policy_name=policy.name,
                    triggered=False,
                    outcome="skipped",
                )
                results.append(result)
                continue

            # Policy wants to fire -- dispatch based on autonomy
            self._last_fired[policy.name] = datetime.now()

            if action.autonomy == "autonomous":
                result = self._execute_action(policy, action)
            elif action.autonomy == "collaborative":
                result = self._handle_collaborative(policy, action)
            else:
                # manual / supervised -- skip execution
                result = EvaluationResult(
                    policy_name=policy.name,
                    triggered=True,
                    action=action,
                    outcome="skipped",
                )
                logger.debug(
                    "Policy '%s' action skipped (autonomy=%s)",
                    policy.name,
                    action.autonomy,
                )

            self._log_evaluation(policy, trigger, result)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self, policy: Policy, action: PolicyAction
    ) -> EvaluationResult:
        """Execute a policy action immediately (autonomous mode).

        Dispatches to the appropriate Tract method based on action_type.
        """
        try:
            commit_hash = self._dispatch_action(action)
            logger.debug(
                "Policy '%s' executed: %s -> %s",
                policy.name,
                action.action_type,
                commit_hash,
            )
            # Notify policy of success
            try:
                policy.on_success(commit_hash)
            except Exception:
                pass  # Don't let on_success errors break the flow
            return EvaluationResult(
                policy_name=policy.name,
                triggered=True,
                action=action,
                outcome="executed",
                commit_hash=commit_hash,
            )
        except Exception as exc:
            logger.error(
                "Policy '%s' action failed: %s", policy.name, exc
            )
            return EvaluationResult(
                policy_name=policy.name,
                triggered=True,
                action=action,
                outcome="error",
                error=str(exc),
            )

    def _dispatch_action(self, action: PolicyAction) -> str | None:
        """Dispatch an action to the appropriate Tract method.

        Returns the resulting commit_hash if applicable.
        """
        if action.action_type == "compress":
            result = self._tract.compress(**action.params)
            # CompressResult has new_commits with hashes
            if hasattr(result, "new_commits") and result.new_commits:
                return result.new_commits[-1].commit_hash
            return None

        if action.action_type == "annotate":
            params = dict(action.params)
            # Convert string priority to Priority enum
            priority_str = params.pop("priority", "normal")
            priority_map = {
                "pinned": Priority.PINNED,
                "normal": Priority.NORMAL,
                "skip": Priority.SKIP,
            }
            priority = priority_map.get(priority_str.lower(), Priority.NORMAL)
            target_hash = params.pop("target_hash", None)
            if target_hash is None:
                raise ValueError("annotate action missing 'target_hash' in params")
            reason = params.pop("reason", None)
            self._tract.annotate(target_hash, priority, reason=reason)
            return None

        if action.action_type == "branch":
            return self._tract.branch(**action.params)

        if action.action_type == "archive":
            # Archive: create branch with archive name
            archive_name = action.params.get(
                "archive_name", f"archive/{datetime.now().strftime('%Y%m%d')}"
            )
            source = action.params.get("source")
            return self._tract.branch(archive_name, source=source, switch=False)

        raise ValueError(f"Unknown action_type: {action.action_type}")

    # ------------------------------------------------------------------
    # Collaborative mode -- hook system integration
    # ------------------------------------------------------------------

    def _handle_collaborative(
        self, policy: Policy, action: PolicyAction
    ) -> EvaluationResult:
        """Handle collaborative mode via PendingPolicy + hook system.

        Three-tier handler precedence:
        1. User hook (t.on("policy", handler))
        2. Policy.default_handler()
        3. Auto-approve
        """
        from tract.hooks.policy import PendingPolicy

        pending = PendingPolicy(
            operation="policy",
            tract=self._tract,
            policy_name=policy.name,
            action_type=action.action_type,
            action_params=dict(action.params),
            reason=action.reason,
            triggered_by=f"policy:{policy.name}",
        )

        # Set execute function to dispatch the action
        def _execute_fn(p: PendingPolicy) -> object:
            result = self._dispatch_action(
                PolicyAction(
                    action_type=p.action_type,
                    params=dict(p.action_params),
                    reason=p.reason,
                    autonomy="collaborative",
                )
            )
            return result

        pending._execute_fn = _execute_fn

        # Also persist proposal to DB for backward compatibility
        if self._policy_repo is not None:
            from tract.storage.schema import PolicyProposalRow

            row = PolicyProposalRow(
                proposal_id=pending.pending_id,
                tract_id=self._tract.tract_id,
                policy_name=policy.name,
                action_type=action.action_type,
                action_params_json=action.params,
                reason=action.reason,
                status="pending",
                created_at=datetime.now(),
            )
            self._policy_repo.save_proposal(row)
            self._tract._session.commit()

        # Three-tier routing for policy:
        # 1. User hook (t.on("policy", handler))
        has_user_hook = (
            "policy" in self._tract._hooks or "*" in self._tract._hooks
        )

        if has_user_hook and not self._tract._in_hook:
            # User hook takes precedence
            self._tract._fire_hook(pending)
        else:
            # 2. Policy.default_handler()
            try:
                policy.default_handler(pending)
            except Exception as exc:
                logger.error(
                    "Policy '%s' default_handler raised: %s", policy.name, exc
                )
                if pending.status == "pending":
                    pending.reject(f"default_handler error: {exc}")

        # Backward compatibility: call on_proposal callback if set
        if self._on_proposal is not None and pending.status == "pending":
            try:
                self._on_proposal(pending)
            except Exception:
                pass

        # Route feedback based on outcome
        if pending.status == "approved":
            # Update DB if persisted
            if self._policy_repo is not None:
                self._policy_repo.update_proposal_status(
                    pending.pending_id, "executed", datetime.now()
                )
                self._tract._session.commit()
            # Notify policy of success
            try:
                policy.on_success(getattr(pending, "_result", None))
            except Exception:
                pass
            return EvaluationResult(
                policy_name=policy.name,
                triggered=True,
                action=action,
                outcome="executed",
            )

        if pending.status == "rejected":
            # Update DB if persisted
            if self._policy_repo is not None:
                self._policy_repo.update_proposal_status(
                    pending.pending_id, "rejected", datetime.now()
                )
                self._tract._session.commit()
            # Notify policy of rejection
            from tract.hooks.validation import HookRejection

            rejection = HookRejection(
                reason=pending.rejection_reason or "",
                pending=pending,
                rejection_source="hook",
            )
            try:
                policy.on_rejection(rejection)
            except Exception:
                pass
            return EvaluationResult(
                policy_name=policy.name,
                triggered=True,
                action=action,
                outcome="proposed",  # Still "proposed" for backward compat
            )

        # Still pending (handler didn't resolve) -- treat as proposed
        return EvaluationResult(
            policy_name=policy.name,
            triggered=True,
            action=action,
            outcome="proposed",
        )

    # ------------------------------------------------------------------
    # Legacy proposal management (backward compatibility)
    # ------------------------------------------------------------------

    def approve_proposal(self, proposal_id: str) -> object:
        """Approve and execute a pending proposal.

        Args:
            proposal_id: The ID of the proposal to approve.

        Returns:
            Result of executing the action.

        Raises:
            PolicyExecutionError: If proposal not found.
        """
        from tract.exceptions import PolicyExecutionError

        if self._policy_repo is not None:
            row = self._policy_repo.get_proposal(proposal_id)
            if row is None:
                raise PolicyExecutionError(
                    f"Proposal not found: {proposal_id}"
                )
            if row.status != "pending":
                raise PolicyExecutionError(
                    f"Proposal {proposal_id} is not pending (status={row.status})"
                )

            # Reconstruct and execute action
            action = PolicyAction(
                action_type=row.action_type,
                params=row.action_params_json or {},
                reason=row.reason or "",
            )
            result = self._dispatch_action(action)

            # Update proposal status
            self._policy_repo.update_proposal_status(
                proposal_id, "executed", datetime.now()
            )
            self._tract._session.commit()
            return result

        raise PolicyExecutionError("No policy repository configured")

    def reject_proposal(self, proposal_id: str, reason: str = "") -> None:
        """Reject a pending proposal.

        Args:
            proposal_id: The ID of the proposal to reject.
            reason: Optional reason for rejection.

        Raises:
            PolicyExecutionError: If proposal not found.
        """
        from tract.exceptions import PolicyExecutionError

        if self._policy_repo is not None:
            row = self._policy_repo.get_proposal(proposal_id)
            if row is None:
                raise PolicyExecutionError(
                    f"Proposal not found: {proposal_id}"
                )
            if row.status != "pending":
                raise PolicyExecutionError(
                    f"Proposal {proposal_id} is not pending (status={row.status})"
                )

            self._policy_repo.update_proposal_status(
                proposal_id, "rejected", datetime.now()
            )
            self._tract._session.commit()
            return

        raise PolicyExecutionError("No policy repository configured")

    def get_pending_proposals(self) -> list:
        """Get all pending proposals.

        Returns:
            List of PolicyProposal objects with status="pending".
            Kept for backward compatibility.
        """
        from tract.models.policy import PolicyProposal

        if self._policy_repo is None:
            return []

        rows = self._policy_repo.get_pending_proposals(self._tract.tract_id)
        proposals: list[PolicyProposal] = []
        for row in rows:
            action = PolicyAction(
                action_type=row.action_type,
                params=row.action_params_json or {},
                reason=row.reason or "",
            )
            proposal = PolicyProposal(
                proposal_id=row.proposal_id,
                policy_name=row.policy_name,
                action=action,
                created_at=row.created_at,
                status=row.status,
            )
            # Reconstruct _execute_fn and _reject_fn so proposals remain
            # approvable/rejectable after restart
            proposal._execute_fn = self._reconstruct_proposal_fn(row)
            proposal._reject_fn = self._reconstruct_reject_fn(row)
            proposals.append(proposal)
        return proposals

    def _reconstruct_proposal_fn(
        self, row: PolicyProposalRow
    ) -> Callable:
        """Build a closure from a stored proposal row for deferred execution."""
        action = PolicyAction(
            action_type=row.action_type,
            params=row.action_params_json or {},
            reason=row.reason or "",
        )

        def _execute_fn(p: object) -> object:
            result = self._dispatch_action(action)
            p.status = "executed"  # type: ignore[attr-defined]
            if self._policy_repo is not None:
                self._policy_repo.update_proposal_status(
                    p.proposal_id, "executed", datetime.now()  # type: ignore[attr-defined]
                )
                self._tract._session.commit()
            return result

        return _execute_fn

    def _reconstruct_reject_fn(
        self, row: PolicyProposalRow
    ) -> Callable:
        """Build a reject closure from a stored proposal row."""

        def _reject_fn(p: object, reason: str = "") -> None:
            p.status = "rejected"  # type: ignore[attr-defined]
            if self._policy_repo is not None:
                self._policy_repo.update_proposal_status(
                    p.proposal_id, "rejected", datetime.now()  # type: ignore[attr-defined]
                )
                self._tract._session.commit()

        return _reject_fn

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _log_evaluation(
        self, policy: Policy, trigger: str, result: EvaluationResult
    ) -> None:
        """Log a policy evaluation to the audit log."""
        logger.debug(
            "Policy '%s' [%s]: outcome=%s",
            policy.name,
            trigger,
            result.outcome,
        )

        if self._policy_repo is not None:
            from tract.storage.schema import PolicyLogRow

            entry = PolicyLogRow(
                tract_id=self._tract.tract_id,
                policy_name=policy.name,
                trigger=trigger,
                action_type=result.action.action_type if result.action else None,
                reason=result.action.reason if result.action else None,
                outcome=result.outcome,
                commit_hash=result.commit_hash,
                error_message=result.error,
                created_at=datetime.now(),
            )
            self._policy_repo.save_log_entry(entry)
            self._tract._session.commit()
