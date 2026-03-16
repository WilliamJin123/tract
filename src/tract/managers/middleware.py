"""Middleware manager for Tract.

Extracted from MiddlewareMixin (_middleware_mix.py) into a standalone class
with explicit constructor dependencies.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.middleware import MiddlewareEvent
    from tract.models.commit import CommitInfo


class MiddlewareManager:
    """Middleware: add/remove, gate, maintain, _run, transition."""

    def __init__(
        self,
        check_open: Callable[[], None],
        persist_behavioral_spec: Callable,  # for auto-persist
        remove_behavioral_spec: Callable,
        get_current_branch: Callable[[], str | None],
        get_head: Callable[[], str | None],
        tract_ref: Callable,  # returns the Tract instance
    ) -> None:
        self._check_open = check_open
        self._persist_behavioral_spec = persist_behavioral_spec
        self._remove_behavioral_spec = remove_behavioral_spec
        self._get_current_branch = get_current_branch
        self._get_head = get_head
        self._tract_ref = tract_ref

        # Owned state
        self._middleware: dict[str, list[tuple[str, Callable]]] = {}
        self._in_middleware_events: set[str] = set()
        self._gates: dict[str, str] = {}
        self._maintainers: dict[str, str] = {}

    def add(self, event: MiddlewareEvent, handler: Callable) -> str:
        """Register middleware. Returns handler ID for removal."""
        from tract.middleware import VALID_EVENTS

        if event not in VALID_EVENTS:
            raise ValueError(
                f"Unknown middleware event '{event}'. "
                f"Valid events: {sorted(VALID_EVENTS)}"
            )
        handler_id = uuid.uuid4().hex[:12]
        self._middleware.setdefault(event, []).append((handler_id, handler))
        return handler_id

    # Backward-compatible alias
    use = add

    def remove(self, handler_id: str) -> None:
        """Remove a registered middleware handler."""
        for event, handlers in self._middleware.items():
            for i, (hid, _fn) in enumerate(handlers):
                if hid == handler_id:
                    handlers.pop(i)
                    # Clean up _gates if this was a gate handler
                    stale = [n for n, gid in self._gates.items() if gid == handler_id]
                    for n in stale:
                        del self._gates[n]
                    # Clean up _maintainers if this was a maintainer handler
                    stale_m = [n for n, mid in self._maintainers.items() if mid == handler_id]
                    for n in stale_m:
                        del self._maintainers[n]
                    return
        raise ValueError(f"Middleware handler '{handler_id}' not found")

    def gate(
        self,
        name: str,
        *,
        event: MiddlewareEvent,
        check: str,
        model: str | None = None,
        condition: Callable | None = None,
        temperature: float = 0.1,
        max_log_entries: int = 30,
    ) -> str:
        """Register a semantic gate (LLM-powered quality check).

        A semantic gate is a middleware handler that uses an LLM to evaluate
        whether the current context meets a natural-language criterion.
        If the criterion is not met, the gate blocks by raising BlockedError.

        Args:
            name: Unique name for this gate.
            event: Middleware event to fire on (e.g., "pre_transition", "pre_commit").
            check: Natural language description of the criterion to evaluate.
            model: LLM model override. Uses tract's default if None.
            condition: Optional deterministic pre-check.
            temperature: LLM temperature (default 0.1).
            max_log_entries: Maximum commits to include in the manifest.

        Returns:
            Handler ID (can be used with remove() or remove_gate()).

        Raises:
            ValueError: If a gate with this name already exists, or event is invalid.
        """
        if name in self._gates:
            raise ValueError(f"Gate '{name}' already registered. Remove it first.")

        if event.startswith("post_"):
            raise ValueError(
                f"Gates cannot be registered on post_* events (got '{event}'). "
                f"Gates block operations via BlockedError, but post_* events fire "
                f"after the operation is already complete. Use t.use() with a "
                f"regular middleware handler for post_* auditing."
            )

        from tract.gate import SemanticGate

        handler = SemanticGate(
            name=name,
            check=check,
            model=model,
            condition=condition,
            temperature=temperature,
            max_log_entries=max_log_entries,
        )
        handler_id = self.add(event, handler)
        self._gates[name] = handler_id
        # Auto-persist gate spec
        try:
            spec_data = handler.to_spec()
            spec_data["event"] = event
            self._persist_behavioral_spec("gate", name, spec_data)
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "Failed to auto-persist gate spec '%s'", name, exc_info=True
            )
        return handler_id

    def remove_gate(self, name: str) -> None:
        """Remove a named semantic gate.

        Args:
            name: The gate name passed to gate().

        Raises:
            ValueError: If no gate with this name exists.
        """
        handler_id = self._gates.pop(name, None)
        if handler_id is None:
            raise ValueError(f"Gate '{name}' not found")
        self.remove(handler_id)
        # Remove persisted spec
        self._remove_behavioral_spec("gate", name)

    def list_gates(self) -> list[str]:
        """Return names of all registered semantic gates."""
        return list(self._gates.keys())

    def maintain(
        self,
        name: str,
        *,
        event: MiddlewareEvent,
        instructions: str,
        actions: list[str] | None = None,
        model: str | None = None,
        condition: Callable | None = None,
        temperature: float = 0.1,
        max_log_entries: int = 30,
        max_peeks: int = 0,
    ) -> str:
        """Register a semantic maintainer (LLM-powered context maintenance).

        A semantic maintainer is a middleware handler that uses an LLM to decide
        what maintenance actions to take on the context.

        Args:
            name: Unique name for this maintainer.
            event: Middleware event to fire on.
            instructions: Natural language description of what maintenance to perform.
            actions: List of allowed action types. If None, all actions are allowed.
            model: LLM model override.
            condition: Optional deterministic pre-check.
            temperature: LLM temperature (default 0.1).
            max_log_entries: Maximum commits to include in the manifest.
            max_peeks: Maximum commits the LLM may inspect for full content.

        Returns:
            Handler ID.

        Raises:
            ValueError: If a maintainer with this name already exists.
        """
        if name in self._maintainers:
            raise ValueError(f"Maintainer '{name}' already registered. Remove it first.")

        from tract.maintain import SemanticMaintainer

        # Default to all valid actions if none specified
        if actions is None:
            actions = sorted(SemanticMaintainer.VALID_ACTIONS)

        handler = SemanticMaintainer(
            name=name,
            instructions=instructions,
            actions=actions,
            model=model,
            condition=condition,
            temperature=temperature,
            max_log_entries=max_log_entries,
            max_peeks=max_peeks,
        )
        handler_id = self.add(event, handler)
        self._maintainers[name] = handler_id
        # Auto-persist maintainer spec
        try:
            spec_data = handler.to_spec()
            spec_data["event"] = event
            self._persist_behavioral_spec("maintainer", name, spec_data)
        except Exception:
            import logging
            logging.getLogger(__name__).debug(
                "Failed to auto-persist maintainer spec '%s'", name, exc_info=True
            )
        return handler_id

    def remove_maintainer(self, name: str) -> None:
        """Remove a named semantic maintainer.

        Args:
            name: The maintainer name passed to maintain().

        Raises:
            ValueError: If no maintainer with this name exists.
        """
        handler_id = self._maintainers.pop(name, None)
        if handler_id is None:
            raise ValueError(f"Maintainer '{name}' not found")
        self.remove(handler_id)
        # Remove persisted spec
        self._remove_behavioral_spec("maintainer", name)

    def list_maintainers(self) -> list[str]:
        """Return names of all registered semantic maintainers."""
        return list(self._maintainers.keys())

    def _run(self, event: str, **kwargs: Any) -> None:
        """Run middleware handlers for an event.

        Raises BlockedError if a handler blocks (pre_* events only).
        """
        if event in self._in_middleware_events:
            return  # recursion guard
        handlers = self._middleware.get(event, [])
        if not handlers:
            return
        self._in_middleware_events.add(event)
        try:
            from tract.middleware import MiddlewareContext

            ctx = MiddlewareContext(
                event=event,
                commit=kwargs.get("commit"),
                tract=self._tract_ref(),
                branch=self._get_current_branch() or "",
                head=self._get_head() or "",
                target=kwargs.get("target"),
                pending=kwargs.get("pending"),
            )
            for _id, fn in list(handlers):
                fn(ctx)
        finally:
            self._in_middleware_events.discard(event)
