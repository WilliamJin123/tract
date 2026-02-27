"""Base Pending class for the hook system.

Every hookable operation produces a Pending -- a mutable container
with methods to approve, reject, modify, or retry the planned operation.
Subclasses add operation-specific fields and methods.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from tract.tract import Tract


class PendingStatus(str, Enum):
    """Status of a pending operation.

    Uses ``str, Enum`` dual inheritance for Python 3.10+ compatibility
    while keeping string comparison and serialization working.
    """

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"

    def __str__(self) -> str:
        return self.value


def _format_value_for_display(value: Any) -> str:
    """Format a value for Rich table display, truncating long content."""
    if value is None:
        return "[dim]None[/dim]"
    if isinstance(value, str):
        if len(value) > 80:
            return repr(value[:77] + "...")
        return repr(value)
    if isinstance(value, list):
        if len(value) == 0:
            return "[]"
        if len(value) > 5:
            return f"[{len(value)} items]"
        items = [repr(v) if isinstance(v, str) and len(str(v)) < 40 else str(v) for v in value]
        return "[" + ", ".join(items) + "]"
    if isinstance(value, dict):
        if len(value) == 0:
            return "{}"
        if len(value) > 5:
            return f"{{{len(value)} entries}}"
        return repr(value)
    if isinstance(value, set):
        if len(value) == 0:
            return "set()"
        return repr(value)
    return repr(value)


@dataclass(repr=False)
class Pending:
    """Base class for all hookable pending operations.

    Fields:
        operation: Name of the hookable operation (e.g. "compress", "gc").
        pending_id: Unique identifier for this pending instance (auto-generated).
        created_at: When this pending was created (UTC).
        tract: The Tract instance that created this pending (full SDK access).
        status: Current status -- "pending", "approved", or "rejected".
        triggered_by: Optional provenance string (e.g. "policy:auto_compress").
        rejection_reason: Human-readable reason if status is "rejected".

    Internal:
        _execute_fn: Closure set by the creating operation to finalize the work.
        _public_actions: Whitelist of method names allowed via agent dispatch.
    """

    operation: str
    tract: Tract
    pending_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: PendingStatus = PendingStatus.PENDING
    triggered_by: str | None = None
    rejection_reason: str | None = None

    # Internal -- set by the creating operation, not by users
    _execute_fn: Callable[..., Any] | None = field(default=None, repr=False)

    # Result stored by approve() for uniform retrieval
    _result: Any = field(default=None, repr=False)

    # Subclasses override this with their allowed action names.
    # Frozen by default; use register_action() for dynamic extension.
    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({"approve", "reject"}), repr=False
    )

    # -- Status guards --------------------------------------------------

    def _require_pending(self) -> None:
        """Raise if this pending has already been resolved."""
        if self.status != PendingStatus.PENDING:
            raise RuntimeError(
                f"Cannot modify a {self.operation} pending with status "
                f"{self.status!r}. Only 'pending' items can be approved or rejected."
            )

    def register_action(self, name: str) -> None:
        """Register an additional public action for agent dispatch.

        Creates a new frozenset with the added action name. This is the
        safe way to extend the whitelist at runtime.

        Args:
            name: Method name to add to the allowed actions.

        Raises:
            ValueError: If name starts with '_'.
            AttributeError: If no method with this name exists.
        """
        if name.startswith("_"):
            raise ValueError(f"Cannot register private method {name!r}.")
        if not hasattr(self, name):
            raise AttributeError(
                f"{type(self).__name__} has no method {name!r}."
            )
        self._public_actions = self._public_actions | frozenset({name})

    # -- Core methods (subclasses should override) -----------------------

    def approve(self) -> Any:
        """Approve and execute the pending operation.

        Subclasses should override this to add operation-specific logic.
        The base implementation sets status, calls _execute_fn, and stores
        the result in ``_result`` for uniform retrieval.

        Returns:
            The result of executing the operation.

        Raises:
            RuntimeError: If status is not "pending".
            RuntimeError: If no _execute_fn has been set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                f"Cannot approve: no execute function set. "
                f"This {type(self).__name__} was not created by a Tract operation."
            )
        self.status = PendingStatus.APPROVED
        self._result = self._execute_fn(self)
        return self._result

    def reject(self, reason: str = "") -> None:
        """Reject the pending operation.

        Subclasses should override this to add operation-specific logic.
        The base implementation sets status and stores the reason.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = PendingStatus.REJECTED
        self.rejection_reason = reason

    # -- Agent interface (auto-generated from subclass methods) ----------

    def to_dict(self) -> dict:
        """Serialize this Pending to a structured dict for LLM consumption.

        Returns a dict with keys: operation, pending_id, status, fields,
        available_actions. Fields are all public (non-underscore) dataclass
        fields excluding identity/status metadata.

        Returns:
            A JSON-serializable dict describing this Pending.
        """
        from tract.hooks.introspection import pending_to_dict

        return pending_to_dict(self)

    def to_tools(self) -> list[dict]:
        """Generate JSON Schema tool definitions for available actions.

        Produces a list of tool definitions compatible with OpenAI/Anthropic
        function calling format, one per method in _public_actions.

        Returns:
            List of tool definition dicts.
        """
        from tract.hooks.introspection import pending_to_tools

        return pending_to_tools(self)

    def describe_api(self) -> str:
        """Generate human/LLM-readable API description.

        Returns a markdown-formatted string listing the Pending's fields
        and available actions with their signatures and docstrings.

        Returns:
            Markdown string describing the API.
        """
        from tract.hooks.introspection import pending_describe_api

        return pending_describe_api(self)

    # -- Dispatch methods -----------------------------------------------

    def apply_decision(self, decision: dict) -> Any:
        """Apply a structured decision dict from an LLM.

        The decision dict must have an "action" key naming the method
        to call, and optionally an "args" key with a dict of arguments.

        Example::

            pending.apply_decision({"action": "approve"})
            pending.apply_decision({"action": "reject", "args": {"reason": "bad quality"}})

        Args:
            decision: Dict with "action" (str) and optional "args" (dict).

        Returns:
            Whatever the dispatched method returns.

        Raises:
            ValueError: If action is not in _public_actions or starts with '_'.
            KeyError: If "action" key is missing from decision.
        """
        action = decision["action"]
        args = decision.get("args", {})
        return self.execute_tool(action, args)

    def execute_tool(self, name: str, args: dict | None = None) -> Any:
        """Execute a named action on this pending, guarded by whitelist.

        Args:
            name: Method name to call.
            args: Keyword arguments to pass to the method.

        Returns:
            Whatever the method returns.

        Raises:
            ValueError: If name starts with '_' or is not in _public_actions.
            AttributeError: If the method does not exist.
        """
        if args is None:
            args = {}

        if name.startswith("_"):
            raise ValueError(
                f"Cannot execute private method {name!r}. "
                f"Allowed actions: {sorted(self._public_actions)}"
            )
        if name not in self._public_actions:
            raise ValueError(
                f"Action {name!r} is not in the allowed actions for "
                f"{type(self).__name__}. "
                f"Allowed: {sorted(self._public_actions)}"
            )

        method = getattr(self, name)
        return method(**args)

    # -- Display --------------------------------------------------------

    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        return f"<{type(self).__name__}: {self.operation}, {status}, id={self.pending_id[:8]}>"

    def pprint(self, *, verbose: bool = False) -> None:
        """Pretty-print this Pending using Rich.

        Shows operation type, status, pending_id, all public fields
        in a table, and available actions.

        Args:
            verbose: If True, show additional details via _pprint_details().
        """
        import dataclasses

        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Header
        status_color = {
            PendingStatus.PENDING: "yellow",
            PendingStatus.APPROVED: "green",
            PendingStatus.REJECTED: "red",
        }.get(self.status, "white")

        console.print(
            f"[bold]{type(self).__name__}[/bold] "
            f"[dim]id={self.pending_id}[/dim]"
        )
        console.print(
            f"  operation: [bold]{self.operation}[/bold]  "
            f"status: [{status_color}]{self.status}[/{status_color}]"
        )
        if self.triggered_by:
            console.print(f"  triggered_by: {self.triggered_by}")
        if self.rejection_reason:
            console.print(f"  rejection_reason: [red]{self.rejection_reason}[/red]")

        # Fields table
        skip_fields = {
            "operation", "pending_id", "status", "tract",
            "triggered_by", "rejection_reason", "created_at",
        }
        table = Table(title="Fields", show_header=True, header_style="bold")
        table.add_column("Field", style="cyan")
        table.add_column("Value")

        for f in dataclasses.fields(self):
            if f.name.startswith("_"):
                continue
            if f.name in skip_fields:
                continue
            value = getattr(self, f.name)
            table.add_row(f.name, _format_value_for_display(value))

        console.print(table)

        # Available actions
        actions = sorted(self._public_actions)
        console.print(f"  [bold]Available actions:[/bold] {', '.join(actions)}")

        # Subclass-specific details
        self._pprint_details(console, verbose=verbose)

    def _pprint_details(self, console, *, verbose: bool = False) -> None:
        """Hook for subclasses to add operation-specific display details.

        Called by pprint() after the base fields table and available actions.
        Override in subclasses to show additional information.

        Args:
            console: The Rich Console instance to print to.
            verbose: If True, show more detailed information.
        """
        pass

    def review(self, *, prompt_fn: Callable[[str], str] | None = None) -> None:
        """Interactive review flow: pprint then prompt for approve/reject.

        Convenience method for CLI usage. Displays the Pending state
        and waits for user input. Subclasses can override for
        operation-specific flows.

        Args:
            prompt_fn: Optional callback for reading user input.
                Receives a prompt string, returns the user's response.
                Defaults to :func:`input` for standard CLI usage.
                Pass a custom function for testing or non-TTY contexts.
        """
        _prompt = prompt_fn or input
        self.pprint()
        # Interactive prompt
        while self.status == PendingStatus.PENDING:
            choice = _prompt("\n[approve/reject/skip] > ").strip().lower()
            if choice == "approve":
                self.approve()
                print(f"Approved {self.operation}.")
            elif choice == "reject":
                reason = _prompt("Reason: ").strip()
                self.reject(reason)
                print(f"Rejected {self.operation}.")
            elif choice == "skip":
                print("Skipped (still pending).")
                break
            else:
                print("Enter 'approve', 'reject', or 'skip'.")
