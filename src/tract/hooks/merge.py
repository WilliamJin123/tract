"""PendingMerge -- hook object for merge operations with conflicts.

Wired to Tract.merge() conflict path in Phase 2. Only merges with
conflicts get hooked -- fast-forward and clean merges proceed without
interception. Handlers can edit conflict resolutions before approving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.guidance import GuidanceMixin
from tract.hooks.pending import Pending

if TYPE_CHECKING:
    from tract.models.merge import MergeResult
    from tract.tract import Tract


@dataclass
class PendingMerge(GuidanceMixin, Pending):
    """A merge operation with conflicts that has been planned but not yet executed.

    Mutable: handlers can edit conflict resolutions, retry LLM-generated
    resolutions, or reject the merge entirely.

    Fields:
        resolutions: Dict mapping conflict keys to their resolved content.
        source_branch: Name of the branch being merged in.
        target_branch: Name of the branch being merged into.
        conflicts: List of conflict descriptions.
        guidance: Guidance text for conflict resolution.
        guidance_source: Where guidance came from.
    """

    resolutions: dict = field(default_factory=dict)
    """Mapping of conflict keys to resolved content strings."""

    source_branch: str = ""
    """Name of the branch being merged in."""

    target_branch: str = ""
    """Name of the branch being merged into."""

    conflicts: list[Any] = field(default_factory=list)
    """List of conflict descriptions (e.g. ConflictInfo instances)."""

    guidance: str | None = None
    """Guidance text for resolution (from user instructions or LLM)."""

    guidance_source: str | None = None
    """Where guidance came from: None, "user", "llm", or "user+llm"."""

    # -- Internal state for execute phase --------------------------------

    _merge_result: MergeResult | None = field(default=None, repr=False)
    """The MergeResult from merge_branches() (set by Tract.merge())."""

    _message: str | None = field(default=None, repr=False)
    """Optional merge commit message (set by Tract.merge())."""

    _delete_branch: bool = field(default=False, repr=False)
    """Whether to delete source branch after merge (set by Tract.merge())."""

    # -- Whitelist for agent dispatch -----------------------------------

    _public_actions: set[str] = field(
        default_factory=lambda: {
            "approve",
            "reject",
            "edit_resolution",
            "set_resolution",
            "edit_interactive",
            "edit_guidance",
        },
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "merge"

    # -- Core methods ---------------------------------------------------

    def approve(self) -> Any:
        """Execute the merge using the current conflict resolutions.

        Returns:
            MergeResult with the merge commit hash and branch update details.

        Raises:
            RuntimeError: If status is not "pending" or no execute function is set.
        """
        self._require_pending()
        if self._execute_fn is None:
            raise RuntimeError(
                "Cannot approve: no execute function set. "
                "This PendingMerge was not created by Tract.merge()."
            )
        self.status = "approved"
        return self._execute_fn(self)

    def reject(self, reason: str = "") -> None:
        """Reject the merge, leaving both branches unchanged.

        Args:
            reason: Human-readable explanation for the rejection.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.status = "rejected"
        self.rejection_reason = reason

    # -- Editing methods ------------------------------------------------

    def edit_resolution(self, key: str, new_content: str) -> None:
        """Replace the resolved content for a specific conflict key.

        Args:
            key: The conflict key in the resolutions dict.
            new_content: Replacement content for the resolution.

        Raises:
            KeyError: If key is not in resolutions.
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        if key not in self.resolutions:
            raise KeyError(
                f"Conflict key {key!r} is not in the resolutions dict. "
                f"Available keys: {sorted(self.resolutions.keys())}"
            )
        self.resolutions[key] = new_content

    def retry(self, *, guidance: str = "", **llm_overrides: Any) -> None:
        """Re-run LLM conflict resolution with updated guidance.

        Args:
            guidance: Feedback text to inject into the retry prompt.
            **llm_overrides: Override LLM parameters for this retry.

        Raises:
            NotImplementedError: Until Phase 2 wiring is complete.
        """
        raise NotImplementedError(
            "retry() is not yet implemented. Use edit_resolution() for manual corrections."
        )

    def set_resolution(self, key: str, content: str) -> None:
        """Set or replace a conflict resolution (does not require key to exist).

        Unlike :meth:`edit_resolution`, this does **not** raise if ``key``
        is not already in :attr:`resolutions`.  Use this when building
        resolutions from scratch (e.g. ``review=True`` without a resolver).

        Args:
            key: The conflict key (typically a ``target_hash``).
            content: The resolved content text.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.resolutions[key] = content

    def edit_interactive(self) -> None:
        """Interactive conflict resolution with a quick-pick menu.

        For each conflict, displays both versions and offers:

        1. **Accept current** — use the target branch version
        2. **Accept incoming** — use the source branch version
        3. **Accept both** — concatenate current + incoming
        4. **Edit in $EDITOR** — open marker text for manual editing
        s. **Skip** — leave unresolved

        Already-resolved conflicts show the existing resolution and
        offer to re-edit.

        Raises:
            RuntimeError: If status is not "pending".
        """
        import click

        from tract.models.merge import ConflictInfo

        self._require_pending()

        for i, conflict in enumerate(self.conflicts):
            if not isinstance(conflict, ConflictInfo):
                continue

            key = conflict.target_hash
            if key is None:
                continue

            # Already resolved? Offer to re-edit
            if key in self.resolutions:
                click.echo(f"\nConflict [{i}] — already resolved:")
                click.echo(f'  "{self.resolutions[key][:80]}"')
                if not click.confirm("  Re-edit this conflict?", default=False):
                    continue

            # Display the two versions
            target = self.target_branch or "current"
            source = self.source_branch or "incoming"
            click.echo(f"\nConflict [{i}]: {conflict.conflict_type} on {key[:8]}")
            click.echo(f"\n  CURRENT ({target}):")
            click.echo(f'    "{conflict.content_a_text}"')
            click.echo(f"\n  INCOMING ({source}):")
            click.echo(f'    "{conflict.content_b_text}"')

            click.echo(f"\n  [1] Accept current")
            click.echo(f"  [2] Accept incoming")
            click.echo(f"  [3] Accept both (current + incoming)")
            click.echo(f"  [4] Edit in $EDITOR")
            click.echo(f"  [s] Skip")

            choice = click.prompt(
                "  Choice", type=click.Choice(["1", "2", "3", "4", "s"])
            )

            if choice == "1":
                self.resolutions[key] = conflict.content_a_text
            elif choice == "2":
                self.resolutions[key] = conflict.content_b_text
            elif choice == "3":
                self.resolutions[key] = (
                    conflict.content_a_text + "\n" + conflict.content_b_text
                )
            elif choice == "4":
                header = (
                    "# MERGE CONFLICT — edit below, then save and close.\n"
                    "# Delete the marker lines (<<<, ===, >>>) and keep "
                    "your resolution.\n"
                    "# Lines starting with # at the top will be stripped.\n"
                    "#\n"
                )
                initial = header + conflict.to_marker_text()
                edited = click.edit(initial)
                if edited is None:
                    click.echo("  Skipped (editor closed without saving).")
                    continue

                # Strip leading comment lines
                lines = edited.split("\n")
                while lines and lines[0].startswith("#"):
                    lines.pop(0)
                cleaned = "\n".join(lines)

                parsed = ConflictInfo.parse_conflict_markers(cleaned)
                if parsed is None:
                    click.echo("  Markers still present — skipped.")
                    continue
                self.resolutions[key] = parsed
            # choice == "s": skip

    # -- Display --------------------------------------------------------
    # Inherits Rich-based pprint() from Pending base class.
