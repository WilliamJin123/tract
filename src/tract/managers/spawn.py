"""Spawn relationship manager extracted from SpawnMixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable

    from tract.models.session import SpawnInfo
    from tract.storage.sqlite import SqliteSpawnPointerRepository


class SpawnManager:
    """Spawn relationship helpers: parent, children, send_to_child, spawn_repo."""

    def __init__(
        self,
        tract_id: str,
        spawn_repo: SqliteSpawnPointerRepository | None = None,
        check_open: Callable | None = None,  # Callable
        session_owner: object | None = None,  # Session back-reference
    ) -> None:
        self._tract_id = tract_id
        self._spawn_repo = spawn_repo
        self._check_open_fn = check_open or (lambda: None)
        self.session_owner = session_owner

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def spawn_repo(self) -> SqliteSpawnPointerRepository | None:
        """Expose spawn repo for internal use by Session."""
        return self._spawn_repo

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def parent(self) -> SpawnInfo | None:
        """Get the spawn info for this tract's parent.

        Returns:
            SpawnInfo if this tract was spawned from a parent, None for
            root tracts or tracts without a spawn_repo.
        """
        if self._spawn_repo is None:
            return None
        row = self._spawn_repo.get_by_child(self._tract_id)
        if row is None:
            return None
        from tract.operations.spawn import _row_to_spawn_info

        return _row_to_spawn_info(row)

    def children(self) -> list[SpawnInfo]:
        """Get spawn info for all child tracts spawned from this tract.

        Returns:
            List of SpawnInfo for each child, in chronological order.
            Empty list if no children or no spawn_repo.
        """
        if self._spawn_repo is None:
            return []
        rows = self._spawn_repo.get_children(self._tract_id)
        from tract.operations.spawn import _row_to_spawn_info

        return [_row_to_spawn_info(row) for row in rows]

    def send_to_child(self, child_tract_id: str, content: str, **kwargs: Any) -> str:
        """Send a message to a child tract.

        Delegates to :meth:`Session.send_message`.  Requires a Session context.

        Args:
            child_tract_id: The child tract identifier.
            content: Message text to send.
            **kwargs: Additional keyword arguments passed through to
                :meth:`Session.send_message` (e.g. ``tags``).

        Returns:
            Commit hash of the message commit in the child tract.

        Raises:
            SessionError: If not called within a Session context.
        """
        self._check_open_fn()
        from tract.exceptions import SessionError

        if self.session_owner is None:
            raise SessionError(
                "send_to_child() requires a Session context. "
                "Use session.send_message(from_id, to_id, content) instead."
            )

        return self.session_owner.send_message(
            self._tract_id, child_tract_id, content, **kwargs
        )
