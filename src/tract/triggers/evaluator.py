"""TriggerEvaluator -- sidecar class that evaluates triggers against a Tract.

The evaluator iterates registered triggers sorted by priority, dispatches
actions based on autonomy level, and logs audit entries for every evaluation.

Recursion guard: if a trigger's action triggers compile() or commit(),
the evaluator will not re-enter evaluate() (same pattern as Tract._in_batch).

Collaborative mode routes through the hook system via PendingTrigger.
Three-tier handler precedence:
    1. User hook (t.on("trigger", handler)) -- highest priority
    2. Trigger.default_handler() -- trigger-specific review logic
    3. Auto-approve -- if no hook and no default_handler override
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import TYPE_CHECKING

from tract.models.annotations import Priority
from tract.models.trigger import EvaluationResult, TriggerAction
from tract.triggers.protocols import Trigger

if TYPE_CHECKING:
    from tract.hooks.trigger import PendingTrigger
    from tract.hooks.validation import HookRejection
    from tract.storage.sqlite import SqliteTriggerRepository
    from tract.tract import Tract

logger = logging.getLogger(__name__)


class TriggerEvaluator:
    """Evaluates triggers against a Tract instance.

    Supports the full autonomy spectrum:
    - **autonomous**: executes actions immediately
    - **collaborative**: creates PendingTrigger, routes through hook system
    - **manual/supervised**: skips execution, logs only

    Features:
    - Priority-sorted evaluation (lower priority runs first)
    - Recursion guard (prevents infinite loops from triggers triggering compile/commit)
    - Cooldown tracking (prevents rapid re-firing)
    - Audit logging (every evaluation logged to DB if trigger_repo available)
    """

    def __init__(
        self,
        tract: Tract,
        triggers: list[Trigger] | None = None,
        trigger_repo: SqliteTriggerRepository | None = None,
        cooldown_seconds: float = 0,
    ) -> None:
        self._tract = tract
        self._triggers: list[Trigger] = sorted(
            triggers or [], key=lambda p: p.priority
        )
        self._trigger_repo = trigger_repo
        self._paused: bool = False
        self._evaluating: bool = False
        self._last_fired: dict[str, datetime] = {}
        self._cooldown_seconds: float = cooldown_seconds

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, trigger_obj: Trigger) -> None:
        """Add a trigger to the evaluator, maintaining priority sort order."""
        self._triggers.append(trigger_obj)
        self._triggers.sort(key=lambda p: p.priority)

    def unregister(self, trigger_name: str) -> None:
        """Remove a trigger by name."""
        self._triggers = [p for p in self._triggers if p.name != trigger_name]

    # ------------------------------------------------------------------
    # Pause / Resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        """Pause all trigger evaluation (emergency kill switch)."""
        self._paused = True

    def resume(self) -> None:
        """Resume trigger evaluation."""
        self._paused = False

    @property
    def is_paused(self) -> bool:
        """Whether the evaluator is currently paused."""
        return self._paused

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def evaluate(self, trigger: str = "compile") -> list[EvaluationResult]:
        """Evaluate all registered triggers matching the given trigger.

        Args:
            trigger: The evaluation trigger -- "compile" or "commit".

        Returns:
            List of EvaluationResult for each trigger that matched the trigger.
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

        # Filter triggers matching this trigger
        matching = [p for p in self._triggers if p.fires_on == trigger]

        for trigger_obj in matching:
            # Check cooldown
            if self._cooldown_seconds > 0 and trigger_obj.name in self._last_fired:
                elapsed = (
                    datetime.now() - self._last_fired[trigger_obj.name]
                ).total_seconds()
                if elapsed < self._cooldown_seconds:
                    result = EvaluationResult(
                        trigger_name=trigger_obj.name,
                        triggered=False,
                        outcome="skipped",
                    )
                    results.append(result)
                    continue

            # Evaluate the trigger
            try:
                action = trigger_obj.evaluate(self._tract)
            except Exception as exc:
                logger.error(
                    "Trigger '%s' raised %s: %s",
                    trigger_obj.name,
                    type(exc).__name__,
                    exc,
                )
                result = EvaluationResult(
                    trigger_name=trigger_obj.name,
                    triggered=True,
                    outcome="error",
                    error=str(exc),
                )
                self._log_evaluation(trigger_obj, trigger, result)
                results.append(result)
                continue

            if action is None:
                # Trigger didn't fire
                result = EvaluationResult(
                    trigger_name=trigger_obj.name,
                    triggered=False,
                    outcome="skipped",
                )
                results.append(result)
                continue

            # Trigger wants to fire -- dispatch based on autonomy
            self._last_fired[trigger_obj.name] = datetime.now()

            if action.autonomy == "autonomous":
                result = self._execute_action(trigger_obj, action)
            elif action.autonomy == "collaborative":
                result = self._handle_collaborative(trigger_obj, action)
            else:
                # manual / supervised -- skip execution
                result = EvaluationResult(
                    trigger_name=trigger_obj.name,
                    triggered=True,
                    action=action,
                    outcome="skipped",
                )
                logger.debug(
                    "Trigger '%s' action skipped (autonomy=%s)",
                    trigger_obj.name,
                    action.autonomy,
                )

            self._log_evaluation(trigger_obj, trigger, result)
            results.append(result)

        return results

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------

    def _execute_action(
        self, trigger_obj: Trigger, action: TriggerAction
    ) -> EvaluationResult:
        """Execute a trigger action immediately (autonomous mode).

        Dispatches to the appropriate Tract method based on action_type.
        """
        try:
            commit_hash = self._dispatch_action(action)
            logger.debug(
                "Trigger '%s' executed: %s -> %s",
                trigger_obj.name,
                action.action_type,
                commit_hash,
            )
            # Notify trigger of success
            try:
                trigger_obj.on_success(commit_hash)
            except Exception:
                pass  # Don't let on_success errors break the flow
            return EvaluationResult(
                trigger_name=trigger_obj.name,
                triggered=True,
                action=action,
                outcome="executed",
                commit_hash=commit_hash,
            )
        except Exception as exc:
            logger.error(
                "Trigger '%s' action failed: %s", trigger_obj.name, exc
            )
            return EvaluationResult(
                trigger_name=trigger_obj.name,
                triggered=True,
                action=action,
                outcome="error",
                error=str(exc),
            )

    def _dispatch_action(self, action: TriggerAction) -> str | None:
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

        if action.action_type == "rebase":
            target = action.params.get("target", "main")
            result = self._tract.rebase(target)
            if hasattr(result, "new_head"):
                return result.new_head
            return None

        if action.action_type == "gc":
            result = self._tract.gc()
            if hasattr(result, "commits_removed"):
                return None  # GC doesn't produce a commit hash
            return None

        if action.action_type == "merge":
            source = action.params.get("source")
            target = action.params.get("target")
            if source is None:
                raise ValueError("merge action missing 'source' in params")
            # If we need to switch to the target branch first
            if target is not None and self._tract.current_branch != target:
                self._tract.switch(target)
            result = self._tract.merge(source)
            if hasattr(result, "merge_commit_hash"):
                return result.merge_commit_hash
            return None

        raise ValueError(f"Unknown action_type: {action.action_type}")

    # ------------------------------------------------------------------
    # Collaborative mode -- hook system integration
    # ------------------------------------------------------------------

    def _handle_collaborative(
        self, trigger_obj: Trigger, action: TriggerAction
    ) -> EvaluationResult:
        """Handle collaborative mode via PendingTrigger + hook system.

        Three-tier handler precedence:
        1. User hook (t.on("trigger", handler))
        2. Trigger.default_handler()
        3. Auto-approve
        """
        from tract.hooks.trigger import PendingTrigger

        pending = PendingTrigger(
            operation="trigger",
            tract=self._tract,
            trigger_name=trigger_obj.name,
            action_type=action.action_type,
            action_params=dict(action.params),
            reason=action.reason,
            triggered_by=f"trigger:{trigger_obj.name}",
        )

        # Set execute function to dispatch the action
        def _execute_fn(p: PendingTrigger) -> object:
            result = self._dispatch_action(
                TriggerAction(
                    action_type=p.action_type,
                    params=dict(p.action_params),
                    reason=p.reason,
                    autonomy="collaborative",
                )
            )
            return result

        pending._execute_fn = _execute_fn

        # Three-tier routing for trigger:
        # 1. User hook (t.on("trigger", handler))
        has_user_hook = (
            "trigger" in self._tract._hooks or "*" in self._tract._hooks
        )

        if has_user_hook and not self._tract._in_hook:
            # User hook takes precedence
            self._tract._fire_hook(pending)
        else:
            # 2. Trigger.default_handler()
            try:
                trigger_obj.default_handler(pending)
            except Exception as exc:
                logger.error(
                    "Trigger '%s' default_handler raised: %s", trigger_obj.name, exc
                )
                if pending.status == "pending":
                    pending.reject(f"default_handler error: {exc}")

        # Route feedback based on outcome
        if pending.status == "approved":
            # Notify trigger of success
            try:
                trigger_obj.on_success(getattr(pending, "_result", None))
            except Exception:
                pass
            return EvaluationResult(
                trigger_name=trigger_obj.name,
                triggered=True,
                action=action,
                outcome="executed",
            )

        if pending.status == "rejected":
            # Notify trigger of rejection
            from tract.hooks.validation import HookRejection

            rejection = HookRejection(
                reason=pending.rejection_reason or "",
                pending=pending,
                rejection_source="hook",
            )
            try:
                trigger_obj.on_rejection(rejection)
            except Exception:
                pass
            return EvaluationResult(
                trigger_name=trigger_obj.name,
                triggered=True,
                action=action,
                outcome="proposed",  # Still "proposed" for backward compat
            )

        # Still pending (handler didn't resolve) -- treat as proposed
        return EvaluationResult(
            trigger_name=trigger_obj.name,
            triggered=True,
            action=action,
            outcome="proposed",
        )

    # ------------------------------------------------------------------
    # Audit logging
    # ------------------------------------------------------------------

    def _log_evaluation(
        self, trigger_obj: Trigger, trigger: str, result: EvaluationResult
    ) -> None:
        """Log a trigger evaluation to the audit log."""
        logger.debug(
            "Trigger '%s' [%s]: outcome=%s",
            trigger_obj.name,
            trigger,
            result.outcome,
        )

        if self._trigger_repo is not None:
            from tract.storage.schema import TriggerLogRow

            entry = TriggerLogRow(
                tract_id=self._tract.tract_id,
                trigger_name=trigger_obj.name,
                trigger=trigger,
                action_type=result.action.action_type if result.action else None,
                reason=result.action.reason if result.action else None,
                outcome=result.outcome,
                commit_hash=result.commit_hash,
                error_message=result.error,
                created_at=datetime.now(),
            )
            self._trigger_repo.save_log_entry(entry)
            self._tract._session.commit()
