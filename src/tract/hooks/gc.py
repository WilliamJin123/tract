"""PendingGC -- hook object for garbage collection operations.

Wired to Tract.gc() in Phase 2. Plan phase determines commits to
remove, execute phase actually deletes them. Handlers can exclude
specific commits from the removal plan before approving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending, PendingStatus

if TYPE_CHECKING:
    from tract.storage.schema import CommitRow
    from tract.tract import Tract


@dataclass(repr=False)
class PendingGC(Pending):
    """A garbage collection operation that has been planned but not yet executed.

    Mutable: handlers can exclude specific commits from removal before
    approving or rejecting.

    Fields:
        commits_to_remove: Hashes of commits scheduled for removal.
        tokens_to_free: Estimated tokens that will be freed by this GC.
    """

    commits_to_remove: list[str] = field(default_factory=list)
    """Hashes of commits scheduled for removal."""

    tokens_to_free: int = 0
    """Estimated tokens freed by removing these commits."""

    # -- Internal state for execute phase --------------------------------

    _commit_rows: list[CommitRow] | None = field(default=None, repr=False)
    """Full CommitRow objects for the execute phase (set by Tract.gc())."""

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({"approve", "reject", "exclude"}),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "gc"

    # -- Core methods ---------------------------------------------------

    def approve(self) -> Any:
        """Execute the garbage collection, permanently removing commits.

        Returns:
            GCResult with details of what was removed.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingGC was not created by Tract.gc()."
            )
        self.status = PendingStatus.APPROVED
        self._result = self._execute_fn(self)
        return self._result

    def reject(self, reason: str = "") -> None:
        """Reject the garbage collection, keeping all commits.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = PendingStatus.REJECTED
        self.rejection_reason = reason

    # -- Editing methods ------------------------------------------------

    def exclude(self, commit_hash: str) -> None:
        """Exclude a commit from removal, keeping it in the repository.

        Args:
            commit_hash: Hash of the commit to exclude from removal.

        Raises:
            ValueError: If commit_hash is not in commits_to_remove.
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        if commit_hash not in self.commits_to_remove:
            raise ValueError(
                f"Commit {commit_hash!r} is not in the removal list."
            )
        self.commits_to_remove.remove(commit_hash)

    # -- Display --------------------------------------------------------

    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        return f"<PendingGC: {len(self.commits_to_remove)} commits, ~{self.tokens_to_free} tokens, {status}>"

    def _compact_detail(self) -> str:
        return f"{len(self.commits_to_remove)} commits, ~{self.tokens_to_free} tokens"

    def _pprint_details(self, console, *, verbose: bool = False) -> None:
        """Show GC-specific details: commit count, token estimate, commit list."""
        console.print(
            f"  GC targets: [bold]{len(self.commits_to_remove)}[/bold] commits, "
            f"~[bold]{self.tokens_to_free}[/bold] tokens to free"
        )
        if verbose and self.commits_to_remove:
            console.print("  [bold]Commits to remove:[/bold]")
            for h in self.commits_to_remove:
                console.print(f"    [bright_cyan]{h[:8]}[/bright_cyan]  {h}")
