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

from tract.exceptions import CurationError, SessionError, SpawnError
from tract.storage.repositories import RefRepository
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


class _BranchScopedRefProxy(RefRepository):
    """Proxy that gives a Tract its own HEAD position on a fixed branch.

    Parent and child share the same tract_id (and thus the same branch refs),
    but each needs an independent HEAD pointer.  This proxy intercepts
    HEAD-related reads so that they resolve via the pinned branch ref
    instead of the shared HEAD symbolic ref.

    Branch mutations (set_branch, delete_ref, etc.) and commit-data
    reads pass through to the real repository unchanged.
    """

    def __init__(self, real: RefRepository, branch_name: str) -> None:
        self._real = real
        self._branch_name = branch_name

    # -- HEAD reads: resolve via the pinned branch ref --

    def get_head(self, tract_id: str) -> str | None:
        return self._real.get_branch(tract_id, self._branch_name)

    def get_current_branch(self, tract_id: str) -> str | None:
        return self._branch_name

    def is_detached(self, tract_id: str) -> bool:
        return False

    # -- HEAD writes: update the branch ref, not the shared HEAD --

    def update_head(self, tract_id: str, commit_hash: str) -> None:
        self._real.set_branch(tract_id, self._branch_name, commit_hash)

    def attach_head(self, tract_id: str, branch_name: str) -> None:
        # Update internal branch tracking
        self._branch_name = branch_name

    def detach_head(self, tract_id: str, commit_hash: str) -> None:
        # Not supported for branch-scoped proxy, but pass through
        self._real.detach_head(tract_id, commit_hash)

    # -- Everything else passes through --

    def get_branch(self, tract_id: str, branch_name: str) -> str | None:
        return self._real.get_branch(tract_id, branch_name)

    def set_branch(self, tract_id: str, branch_name: str, commit_hash: str) -> None:
        self._real.set_branch(tract_id, branch_name, commit_hash)

    def list_branches(self, tract_id: str) -> list[str]:
        return self._real.list_branches(tract_id)

    def get_ref(self, tract_id: str, ref_name: str) -> str | None:
        return self._real.get_ref(tract_id, ref_name)

    def set_ref(self, tract_id: str, ref_name: str, commit_hash: str) -> None:
        self._real.set_ref(tract_id, ref_name, commit_hash)

    def delete_ref(self, tract_id: str, ref_name: str) -> None:
        self._real.delete_ref(tract_id, ref_name)

    def set_symbolic_ref(self, tract_id: str, ref_name: str, symbolic_target: str) -> None:
        self._real.set_symbolic_ref(tract_id, ref_name, symbolic_target)

    def get_symbolic_ref(self, tract_id: str, ref_name: str) -> str | None:
        return self._real.get_symbolic_ref(tract_id, ref_name)


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

    def deploy(
        self,
        parent: Tract,
        *,
        purpose: str,
        branch_name: str,
        curate: dict | None = None,
    ) -> Tract:
        """Deploy a sub-agent with curated context on a new branch.

        Creates a branch from parent's HEAD, applies curation operations,
        and returns a new Tract instance on the child branch.  Parent stays
        on its original branch.

        Args:
            parent: The parent Tract instance.
            purpose: Description of the child's task.
            branch_name: Name for the child's branch.
            curate: Optional curation config dict with keys:
                - ``"keep_tags"``: list[str] -- keep only commits with these tags
                - ``"drop"``: list[str] -- drop these commit hashes
                - ``"compact_before"``: str -- compress commits before this hash
                - ``"reorder"``: list[str] -- new commit hash order

        Returns:
            A new Tract instance on the child branch (same DB, same tract_id).

        Raises:
            BranchExistsError: If *branch_name* already exists.
            CurationError: If curation operations fail.
        """
        from datetime import timezone
        from tract.models.annotations import Priority
        from tract.storage.sqlite import (
            SqliteAnnotationRepository,
            SqliteBlobRepository,
            SqliteCommitParentRepository,
            SqliteCommitRepository,
            SqliteOperationEventRepository,
            SqliteRefRepository,
            SqliteTagAnnotationRepository,
            SqliteTagRegistryRepository,
        )
        from tract.engine.cache import CacheManager
        from tract.engine.commit import CommitEngine
        from tract.engine.compiler import DefaultContextCompiler
        from tract.engine.tokens import TiktokenCounter
        from tract.tract import Tract as _Tract

        # 1. Record parent's current branch
        original_branch = parent.current_branch

        # 2. Create new branch from parent HEAD (without switching)
        parent.branch(branch_name, switch=False)

        # 3. Build a child Tract instance sharing the same tract_id + DB
        #    but with its own HEAD tracking via _BranchScopedRefProxy.
        child_session = self._session_factory()
        child_config = parent.config

        child_commit_repo = SqliteCommitRepository(child_session)
        child_blob_repo = SqliteBlobRepository(child_session)
        real_ref_repo = SqliteRefRepository(child_session)
        child_annotation_repo = SqliteAnnotationRepository(child_session)
        child_parent_repo = SqliteCommitParentRepository(child_session)
        child_event_repo = SqliteOperationEventRepository(child_session)

        # Wrap the ref repo so the child's HEAD resolves via its own branch,
        # not the shared HEAD symbolic ref.
        child_ref_repo = _BranchScopedRefProxy(real_ref_repo, branch_name)

        child_token_counter = TiktokenCounter(
            encoding_name=child_config.tokenizer_encoding,
        )

        child_commit_engine = CommitEngine(
            commit_repo=child_commit_repo,
            blob_repo=child_blob_repo,
            ref_repo=child_ref_repo,
            annotation_repo=child_annotation_repo,
            token_counter=child_token_counter,
            tract_id=parent.tract_id,
            token_budget=child_config.token_budget,
            parent_repo=child_parent_repo,
        )

        child_compiler = DefaultContextCompiler(
            commit_repo=child_commit_repo,
            blob_repo=child_blob_repo,
            annotation_repo=child_annotation_repo,
            token_counter=child_token_counter,
            parent_repo=child_parent_repo,
        )

        child = _Tract(
            engine=None,  # Engine owned by Session
            session=child_session,
            commit_engine=child_commit_engine,
            compiler=child_compiler,
            tract_id=parent.tract_id,
            config=child_config,
            commit_repo=child_commit_repo,
            blob_repo=child_blob_repo,
            ref_repo=child_ref_repo,
            annotation_repo=child_annotation_repo,
            token_counter=child_token_counter,
            parent_repo=child_parent_repo,
            event_repo=child_event_repo,
        )
        child._spawn_repo = self._spawn_repo

        # Set up tag repos on child so get_tags() works
        child._tag_annotation_repo = SqliteTagAnnotationRepository(child_session)
        child._tag_registry_repo = SqliteTagRegistryRepository(child_session)

        # 4. Apply curation pipeline (child is already on its branch via proxy)
        if curate:
            _apply_curation(child, curate)

        # 5. Record spawn pointer with "branch" inheritance mode.
        #    Use "{tract_id}:{branch_name}" as the child_tract_id because
        #    spawn_pointers has a UNIQUE constraint on child_tract_id, and
        #    multiple deploys from the same parent need distinct entries.
        now = datetime.now(timezone.utc)
        child_pointer_id = f"{parent.tract_id}:{branch_name}"
        self._spawn_repo.save(
            parent_tract_id=parent.tract_id,
            parent_commit_hash=parent.head,
            child_tract_id=child_pointer_id,
            purpose=purpose,
            inheritance_mode="branch",
            display_name=branch_name,
            created_at=now,
        )
        self._spawn_repo._session.commit()

        # 6. Track child in session (use branch_name as key to avoid collision
        #    with parent which uses the same tract_id)
        child_key = f"{parent.tract_id}:{branch_name}"
        self._tracts[child_key] = child

        return child

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


# ======================================================================
# Curation pipeline (module-level helpers for deploy())
# ======================================================================


def _apply_curation(child: Tract, curate: dict) -> None:
    """Apply the curation pipeline in fixed order on the child branch.

    Order: keep_tags -> drop -> compact_before -> reorder.

    Args:
        child: The child Tract instance (already on the child branch).
        curate: Curation config dict.

    Raises:
        CurationError: If any curation step fails.
    """
    if "keep_tags" in curate:
        _curate_keep_tags(child, curate["keep_tags"])

    if "drop" in curate:
        _curate_drop(child, curate["drop"])

    if "compact_before" in curate:
        _curate_compact_before(child, curate["compact_before"])

    if "reorder" in curate:
        _curate_reorder(child, curate["reorder"])


def _get_commit_chain(child: Tract) -> list:
    """Get the commit chain from HEAD back to root (newest first).

    Returns list of CommitRow objects.
    """
    head = child.head
    if head is None:
        return []
    return list(child._commit_repo.get_ancestors(head))


def _curate_keep_tags(child: Tract, keep_tags: list[str]) -> None:
    """Keep only commits that have at least one of the specified tags.

    Commits without matching tags get annotated with SKIP priority.

    Args:
        child: The child Tract on its branch.
        keep_tags: List of tag names to keep.
    """
    from tract.models.annotations import Priority

    tag_set = set(keep_tags)
    chain = _get_commit_chain(child)

    for row in chain:
        commit_tags = set(child.get_tags(row.commit_hash))
        if not (commit_tags & tag_set):
            child.annotate(row.commit_hash, Priority.SKIP, reason="curation:keep_tags")


def _curate_drop(child: Tract, drop_hashes: list[str]) -> None:
    """Drop explicitly listed commit hashes by annotating with SKIP.

    Validates that no dropped commit is the edit_target of a non-skipped
    remaining commit.

    Args:
        child: The child Tract on its branch.
        drop_hashes: List of commit hashes to drop.

    Raises:
        CurationError: If a dropped hash is an edit_target of another commit.
    """
    from tract.models.annotations import Priority
    from tract.models.commit import CommitOperation

    drop_set = set(drop_hashes)
    chain = _get_commit_chain(child)
    chain_hashes = {row.commit_hash for row in chain}

    # Verify all drop hashes exist on the branch
    for h in drop_hashes:
        if h not in chain_hashes:
            raise CurationError(
                f"Cannot drop commit {h}: not found on the current branch"
            )

    # Check no non-dropped EDIT commit targets a dropped commit
    for row in chain:
        if row.commit_hash in drop_set:
            continue  # This commit is being dropped, skip it
        if row.operation == CommitOperation.EDIT and row.edit_target:
            if row.edit_target in drop_set:
                raise CurationError(
                    f"Cannot drop commit {row.edit_target[:12]}: "
                    f"it is the edit_target of commit {row.commit_hash[:12]}"
                )

    # Apply SKIP annotations
    for h in drop_hashes:
        child.annotate(h, Priority.SKIP, reason="curation:drop")


def _curate_compact_before(child: Tract, marker_hash: str) -> None:
    """Compress all non-SKIPPED commits before the marker hash.

    If an LLM client is available on the tract, uses it.
    Otherwise, creates a simple concatenation summary.

    Args:
        child: The child Tract on its branch.
        marker_hash: Compress all commits before this one.

    Raises:
        CurationError: If marker_hash is not found on the branch.
    """
    from tract.models.annotations import Priority

    chain = _get_commit_chain(child)  # newest first
    chain_reversed = list(reversed(chain))  # root first

    # Find marker position (in root-first order)
    marker_idx = None
    for i, row in enumerate(chain_reversed):
        if row.commit_hash == marker_hash:
            marker_idx = i
            break

    if marker_idx is None:
        raise CurationError(
            f"compact_before marker {marker_hash} not found on the branch"
        )

    if marker_idx == 0:
        return  # Nothing before the marker

    # Collect non-SKIPPED commits before the marker
    before_commits = chain_reversed[:marker_idx]
    non_skipped = []
    for row in before_commits:
        annotations = child._annotation_repo.get_history(row.commit_hash)
        is_skipped = False
        if annotations:
            # Last annotation is the effective one
            if annotations[-1].priority == Priority.SKIP:
                is_skipped = True
        if not is_skipped:
            non_skipped.append(row)

    if not non_skipped:
        return  # All commits before marker are already skipped

    non_skipped_hashes = [row.commit_hash for row in non_skipped]

    # Try to use compress_range if an LLM client is available
    # Check if LLM is available (via child which shares parent's config)
    has_llm = getattr(child, "_has_llm_client", lambda op: False)("compress")

    if has_llm:
        # Use the Tract.compress() API which handles all wiring
        child.compress(
            commits=non_skipped_hashes,
            triggered_by="curation:compact_before",
        )
    else:
        # No LLM: build a concatenation summary and use manual content mode
        import json
        lines = []
        for row in non_skipped:
            blob = child._blob_repo.get(row.content_hash)
            if blob is not None:
                try:
                    data = json.loads(blob.payload_json)
                    text = data.get("text", "") or data.get("content", "") or str(data)
                except (json.JSONDecodeError, TypeError):
                    text = str(blob.payload_json)
                role = getattr(row, "content_type", "unknown")
                lines.append(f"[{role}]: {text}")

        summary = "\n".join(lines)
        if not summary:
            summary = "(empty summary)"

        # Use Tract.compress() with manual content
        child.compress(
            commits=non_skipped_hashes,
            content=summary,
            triggered_by="curation:compact_before",
        )


def _curate_reorder(child: Tract, order: list[str]) -> None:
    """Reorder remaining commits on the child branch.

    Uses check_reorder_safety for validation (warnings only, not errors).
    Replays commits in the new order using replay_commit.

    Args:
        child: The child Tract on its branch.
        order: List of commit hashes in the desired order.

    Raises:
        CurationError: If any hash in order is not found on the branch.
    """
    from tract.operations.compression import check_reorder_safety
    from tract.operations.rebase import replay_commit

    chain = _get_commit_chain(child)  # newest first
    chain_hashes = {row.commit_hash for row in chain}

    # Validate all hashes exist
    for h in order:
        if h not in chain_hashes:
            raise CurationError(
                f"Cannot reorder: commit {h} not found on the branch"
            )

    # Safety check (warnings only, don't error)
    _warnings = check_reorder_safety(
        order, child._commit_repo, child._blob_repo,
    )

    # Build hash-to-row lookup
    row_map = {row.commit_hash: row for row in chain}

    # Get the current branch name
    branch_name = child.current_branch

    # Find the parent of the first commit to reorder
    # (we need the commit that comes just before the reorder range)
    chain_reversed = list(reversed(chain))  # root-first
    all_ordered = set(order)

    # Find what's before the reorder range
    # Everything in `order` gets replayed; everything NOT in `order` stays
    # We rebuild the entire chain: non-reordered commits in original order,
    # then inject reordered ones at their positions.
    # Actually per the spec: replay the chain in the new order.

    # Strategy: clear the branch to the commit before the first reorder target,
    # then replay in the specified order, then replay any remaining commits
    # not in the order list.

    # Simplification: find the earliest (closest to root) commit in the order set
    earliest_idx = None
    for i, row in enumerate(chain_reversed):
        if row.commit_hash in all_ordered:
            earliest_idx = i
            break

    if earliest_idx is None:
        return  # Nothing to reorder

    # The parent of the earliest reorder target is what we reset the branch to
    if earliest_idx > 0:
        reset_parent = chain_reversed[earliest_idx - 1].commit_hash
    else:
        reset_parent = None

    # Collect commits not in the order list that come after the earliest
    remaining_after = [
        row for row in chain_reversed[earliest_idx:]
        if row.commit_hash not in all_ordered
    ]

    # Reset the branch to the parent of the earliest commit
    if reset_parent is not None:
        if branch_name:
            child._ref_repo.set_branch(child.tract_id, branch_name, reset_parent)
        child._session.commit()
    else:
        # Reset to no commits -- clear the branch ref
        if branch_name:
            child._ref_repo.delete_ref(child.tract_id, f"refs/heads/{branch_name}")
        child._session.commit()

    # Replay in the specified order
    for h in order:
        row = row_map[h]
        replay_commit(
            row,
            child.head,
            child._commit_engine,
            child._blob_repo,
        )
        child._session.commit()

    # Replay remaining commits not in the order list
    for row in remaining_after:
        replay_commit(
            row,
            child.head,
            child._commit_engine,
            child._blob_repo,
        )
        child._session.commit()
