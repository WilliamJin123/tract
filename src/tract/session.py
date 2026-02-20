"""Session -- multi-agent entry point for Trace.

All tracts in a session share one SQLite file and engine.
Each tract gets its own SQLAlchemy session for thread safety.

Usage::

    with Session.open("project.db") as session:
        parent = session.create_tract(display_name="orchestrator")
        child = session.spawn(parent, purpose="research task")
        child.commit(DialogueContent(role="user", text="Research X"))
        result = session.collapse(child, into=parent, content="X is Y")
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from tract.exceptions import SessionError, SpawnError
from tract.models.config import TractConfig
from tract.models.session import CollapseResult, SpawnInfo
from tract.operations.spawn import (
    collapse_tract,
    spawn_tract,
)
from tract.storage.engine import create_session_factory, create_trace_engine, init_db
from tract.storage.sqlite import SqliteSpawnPointerRepository

if TYPE_CHECKING:
    from sqlalchemy import Engine
    from sqlalchemy.orm import sessionmaker

    from tract.protocols import CompiledContext
    from tract.models.commit import CommitInfo
    from tract.tract import Tract


class Session:
    """Multi-agent entry point backed by a single shared SQLite DB.

    All tracts in a session share one SQLite file and engine.
    Each tract gets its own SQLAlchemy session for thread safety.
    """

    def __init__(
        self,
        engine: Engine,
        session_factory: sessionmaker,
        spawn_repo: SqliteSpawnPointerRepository,
        db_path: str,
        *,
        autonomy: str = "collaborative",
    ) -> None:
        self._engine = engine
        self._session_factory = session_factory
        self._spawn_repo = spawn_repo
        self._db_path = db_path
        self._autonomy = autonomy
        self._tracts: dict[str, Tract] = {}
        self._closed = False
        # Keep the session used by spawn_repo alive
        self._spawn_session = spawn_repo._session

    @classmethod
    def open(
        cls,
        path: str = ":memory:",
        *,
        autonomy: str = "collaborative",
    ) -> Session:
        """Open (or create) a multi-agent session.

        Args:
            path: SQLite path. ``":memory:"`` for in-memory (default).
            autonomy: Default autonomy level for collapse operations.
                "manual", "collaborative" (default), or "autonomous".

        Returns:
            A ready-to-use Session instance.
        """
        valid_autonomy = {"manual", "collaborative", "autonomous"}
        if autonomy not in valid_autonomy:
            raise ValueError(
                f"Invalid autonomy level: {autonomy!r}. "
                f"Must be one of {valid_autonomy}"
            )

        engine = create_trace_engine(path)
        init_db(engine)
        session_factory = create_session_factory(engine)
        session = session_factory()
        spawn_repo = SqliteSpawnPointerRepository(session)

        return cls(
            engine=engine,
            session_factory=session_factory,
            spawn_repo=spawn_repo,
            db_path=path,
            autonomy=autonomy,
        )

    # ------------------------------------------------------------------
    # Tract management
    # ------------------------------------------------------------------

    def create_tract(
        self,
        *,
        display_name: str | None = None,
        tract_id: str | None = None,
        config: TractConfig | None = None,
    ) -> Tract:
        """Create a fresh tract within this session.

        Args:
            display_name: Optional human-readable name.
            tract_id: Optional tract identifier. Generated if not provided.
            config: Optional tract configuration.

        Returns:
            A new Tract instance sharing the session's engine.
        """
        from tract.engine.cache import CacheManager
        from tract.engine.commit import CommitEngine
        from tract.engine.compiler import DefaultContextCompiler
        from tract.engine.tokens import TiktokenCounter
        from tract.storage.sqlite import (
            SqliteAnnotationRepository,
            SqliteBlobRepository,
            SqliteCommitParentRepository,
            SqliteCommitRepository,
            SqliteOperationEventRepository,
            SqliteRefRepository,
        )
        from tract.tract import Tract

        if tract_id is None:
            tract_id = uuid.uuid4().hex

        if config is None:
            config = TractConfig(db_path=self._db_path)

        # Create a new session from the factory for this tract
        session = self._session_factory()

        # Build repositories
        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annotation_repo = SqliteAnnotationRepository(session)
        parent_repo = SqliteCommitParentRepository(session)
        event_repo = SqliteOperationEventRepository(session)

        # Token counter
        token_counter = TiktokenCounter(
            encoding_name=config.tokenizer_encoding,
        )

        # Commit engine
        commit_engine = CommitEngine(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            tract_id=tract_id,
            token_budget=config.token_budget,
            parent_repo=parent_repo,
        )

        # Context compiler
        compiler = DefaultContextCompiler(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            parent_repo=parent_repo,
        )

        tract = Tract(
            engine=None,  # Engine owned by Session
            session=session,
            commit_engine=commit_engine,
            compiler=compiler,
            tract_id=tract_id,
            config=config,
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=token_counter,
            parent_repo=parent_repo,
            event_repo=event_repo,
        )
        tract._spawn_repo = self._spawn_repo

        self._tracts[tract_id] = tract
        return tract

    def get_tract(self, tract_id: str) -> Tract:
        """Retrieve a tract by its ID.

        Returns from cache if available, otherwise reconstructs from DB.

        Note:
            Reconstructed tracts use default ``TractConfig``. If the original
            tract was created with a custom config, those settings are not
            persisted and will be lost on reconstruction.

        Args:
            tract_id: The tract identifier.

        Returns:
            The Tract instance.

        Raises:
            SessionError: If tract_id not found in the database.
        """
        if tract_id in self._tracts:
            return self._tracts[tract_id]

        # Try to reconstruct from DB: check if any commits exist for this tract_id
        from sqlalchemy import select
        from tract.storage.schema import CommitRow

        session = self._session_factory()
        stmt = select(CommitRow).where(CommitRow.tract_id == tract_id).limit(1)
        result = session.execute(stmt).first()

        if result is None:
            # Also check if it's a known child tract with no commits yet
            pointer = self._spawn_repo.get_by_child(tract_id)
            if pointer is None:
                raise SessionError(f"Tract not found: {tract_id}")

        # Reconstruct the tract
        session.close()  # Close the probe session
        tract = self.create_tract(tract_id=tract_id)
        return tract

    def release_tract(self, tract_id: str) -> None:
        """Release a tract, freeing its session and compile cache.

        Use this to reclaim memory for tracts that are no longer needed
        (e.g. after collapsing a child).

        Args:
            tract_id: The tract identifier to release.

        Raises:
            SessionError: If tract_id is not currently held.
        """
        if tract_id not in self._tracts:
            raise SessionError(f"Tract not held in session: {tract_id}")
        tract = self._tracts.pop(tract_id)
        tract._cache.clear()
        try:
            tract._session.close()
        except Exception:
            pass

    def list_tracts(self) -> list[dict]:
        """List all tracts in the session.

        Returns:
            List of dicts with tract metadata.
        """
        from tract.operations.session_ops import list_tracts as _list_tracts

        return _list_tracts(self._spawn_session, self._spawn_repo)

    # ------------------------------------------------------------------
    # Spawn / Collapse
    # ------------------------------------------------------------------

    def spawn(
        self,
        parent: Tract,
        *,
        purpose: str,
        inheritance: str = "head_snapshot",
        display_name: str | None = None,
        max_tokens: int | None = None,
    ) -> Tract:
        """Spawn a child tract from a parent.

        Args:
            parent: The parent Tract.
            purpose: Description of the child's task.
            inheritance: "head_snapshot" (default) or "full_clone".
            display_name: Optional human-readable name for the child.
            max_tokens: Max tokens for head_snapshot truncation.

        Returns:
            The new child Tract instance.
        """
        child = spawn_tract(
            session_factory=self._session_factory,
            engine=self._engine,
            spawn_repo=self._spawn_repo,
            parent_tract=parent,
            purpose=purpose,
            inheritance=inheritance,
            display_name=display_name,
            max_tokens=max_tokens,
        )
        self._tracts[child.tract_id] = child
        return child

    def collapse(
        self,
        child: Tract,
        into: Tract,
        *,
        content: str | None = None,
        instructions: str | None = None,
        auto_commit: bool | None = None,
        target_tokens: int | None = None,
    ) -> CollapseResult:
        """Collapse a child tract's history into a summary commit in the parent.

        Args:
            child: The child Tract to summarize.
            into: The parent Tract to receive the summary.
            content: Manual summary text (bypasses LLM).
            instructions: Additional LLM instructions.
            auto_commit: If True, commit immediately. Defaults based on autonomy.
            target_tokens: Target token count for LLM summary.

        Returns:
            CollapseResult with summary details.
        """
        # Determine auto_commit from autonomy if not specified
        if auto_commit is None:
            if self._autonomy == "autonomous":
                auto_commit = True
            elif self._autonomy == "manual":
                auto_commit = True  # Manual mode expects content to be provided
            else:
                auto_commit = False  # Collaborative: review before commit

        llm_client = getattr(into, "_llm_client", None)

        return collapse_tract(
            parent_tract=into,
            child_tract=child,
            spawn_repo=self._spawn_repo,
            content=content,
            instructions=instructions,
            auto_commit=auto_commit,
            target_tokens=target_tokens,
            llm_client=llm_client,
        )

    def get_child_tract(self, collapse_commit_hash: str) -> Tract:
        """Navigate from a collapse commit back to the child tract.

        Enables "expand for debugging" by retrieving the child tract
        that produced a collapse summary.

        Args:
            collapse_commit_hash: Hash of the collapse commit in the parent.

        Returns:
            The child Tract that was collapsed.

        Raises:
            SessionError: If commit has no collapse metadata or child not found.
        """
        from sqlalchemy import select
        from tract.storage.schema import CommitRow

        # Query the database directly instead of searching cached tracts
        stmt = select(CommitRow).where(
            CommitRow.commit_hash == collapse_commit_hash
        )
        row = self._spawn_session.execute(stmt).scalar_one_or_none()

        if row is None:
            raise SessionError(
                f"Commit not found: {collapse_commit_hash}"
            )

        metadata = row.metadata_json
        if metadata is None or "collapse_source_tract_id" not in metadata:
            raise SessionError(
                f"Commit {collapse_commit_hash} is not a collapse commit "
                "(no collapse_source_tract_id in metadata)"
            )

        child_tract_id = metadata["collapse_source_tract_id"]
        return self.get_tract(child_tract_id)

    # ------------------------------------------------------------------
    # Cross-repo queries
    # ------------------------------------------------------------------

    def timeline(self, *, limit: int | None = None) -> list[CommitInfo]:
        """Get all commits across all tracts in chronological order.

        Args:
            limit: Maximum number of commits to return.

        Returns:
            List of CommitInfo in chronological order.
        """
        from tract.operations.session_ops import timeline as _timeline

        return _timeline(self._spawn_session, limit=limit)

    def search(
        self, term: str, *, tract_id: str | None = None
    ) -> list[CommitInfo]:
        """Search for commits matching a term across tracts.

        Args:
            term: Search term (matched via LIKE on blob content).
            tract_id: Optional filter to a specific tract.

        Returns:
            List of matching CommitInfo.
        """
        from tract.operations.session_ops import search as _search

        return _search(self._spawn_session, term, tract_id=tract_id)

    def compile_at(
        self,
        tract_id: str,
        *,
        at_time: datetime | None = None,
        at_commit: str | None = None,
    ) -> CompiledContext:
        """Compile any tract at a historical point-in-time.

        Args:
            tract_id: The tract to compile.
            at_time: Compile as of this datetime.
            at_commit: Compile up to this commit hash.

        Returns:
            CompiledContext for the tract at the specified point.
        """
        from tract.operations.session_ops import compile_at as _compile_at

        return _compile_at(
            self._session_factory,
            self._engine,
            tract_id,
            at_time=at_time,
            at_commit=at_commit,
        )

    def resume(self) -> Tract | None:
        """Find the most recent active tract for crash recovery.

        Returns the most recent active root tract (no session_type="end"
        commit), or None if no active tracts.

        Returns:
            The most recent active Tract, or None.
        """
        from tract.operations.session_ops import resume as _resume

        result = _resume(self._spawn_session, self._spawn_repo)
        if result is None:
            return None

        tract_id = result["tract_id"]
        return self.get_tract(tract_id)

    # ------------------------------------------------------------------
    # Context manager / lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all tract sessions and dispose the engine."""
        if self._closed:
            return
        self._closed = True

        # Close all tract sessions
        for tract in self._tracts.values():
            try:
                tract._session.close()
            except Exception:
                pass

        # Close spawn repo session
        try:
            self._spawn_session.close()
        except Exception:
            pass

        # Dispose engine
        if self._engine is not None:
            self._engine.dispose()

    def __enter__(self) -> Session:
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        tract_count = len(self._tracts)
        return f"Session(db='{self._db_path}', tracts={tract_count})"
