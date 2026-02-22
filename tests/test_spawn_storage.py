"""Tests for spawn storage: schema, repository, session models, prompts, and exceptions.

Covers:
- SpawnPointerRow table creation and migration
- Schema migration v3->v4 and v2->v4 chain
- SqliteSpawnPointerRepository CRUD and ancestry operations
- SessionContent, SpawnInfo, CollapseResult models
- Collapse prompt constants and builder
- SpawnError and SessionError exceptions
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from tract.models.commit import CommitOperation
from tract.storage.engine import create_trace_engine, init_db
from tract.storage.schema import (
    BlobRow,
    CommitRow,
    SpawnPointerRow,
    TraceMetaRow,
)
from tract.storage.sqlite import SqliteSpawnPointerRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_blob(session: Session, content_hash: str) -> BlobRow:
    """Create and persist a minimal blob for FK satisfaction."""
    blob = BlobRow(
        content_hash=content_hash,
        payload_json='{"text": "test"}',
        byte_size=16,
        token_count=2,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    session.add(blob)
    session.flush()
    return blob


def _make_commit(
    session: Session,
    commit_hash: str,
    tract_id: str = "test-tract",
    parent_hash: str | None = None,
    content_hash: str | None = None,
) -> CommitRow:
    """Create and persist a minimal commit for FK satisfaction."""
    if content_hash is None:
        content_hash = f"blob-{commit_hash}"
    _make_blob(session, content_hash)
    commit = CommitRow(
        commit_hash=commit_hash,
        tract_id=tract_id,
        parent_hash=parent_hash,
        content_hash=content_hash,
        content_type="dialogue",
        operation=CommitOperation.APPEND,
        message="test commit",
        token_count=10,
        created_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
    session.add(commit)
    session.flush()
    return commit


# ===========================================================================
# Schema Tests
# ===========================================================================


class TestSpawnPointerSchema:
    """Tests for spawn_pointers table creation and migration."""

    def test_spawn_pointers_table_created(self, engine):
        """init_db creates spawn_pointers table."""
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "spawn_pointers" in table_names

    def test_migration_v3_to_v6(self):
        """Start with schema_version=3, call init_db, verify table + version=6."""
        engine = create_trace_engine(":memory:")

        from tract.storage.schema import Base

        # Create all tables, then drop spawn_pointers + policy + v6 tables to simulate v3
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS policy_log"))
            conn.execute(text("DROP TABLE IF EXISTS policy_proposals"))
            conn.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.commit()

        # Set schema version to 3
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            session.add(TraceMetaRow(key="schema_version", value="3"))
            session.commit()

        # Now call init_db -- should migrate v3->v4->v5->v6
        init_db(engine)

        # Verify spawn_pointers table exists
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "spawn_pointers" in table_names

        # Verify schema version is now 6
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert int(row.value) >= 7

        engine.dispose()

    def test_migration_v2_to_v6(self):
        """Start with schema_version=2, verify migration chain v2->v3->v4->v5->v6."""
        engine = create_trace_engine(":memory:")

        from tract.storage.schema import Base

        # Create all tables, then drop compression + spawn + policy + v6 tables to simulate v2
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS policy_log"))
            conn.execute(text("DROP TABLE IF EXISTS policy_proposals"))
            conn.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS commit_parents"))
            conn.commit()

        # Set schema version to 2
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            session.add(TraceMetaRow(key="schema_version", value="2"))
            session.commit()

        # Now call init_db -- should migrate v2->v3->v4->v5->v6
        init_db(engine)

        # Verify v6 tables exist (old compression tables dropped by v5->v6)
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "spawn_pointers" in table_names
        assert "operation_events" in table_names
        assert "operation_commits" in table_names
        assert "compile_records" in table_names
        # Old compression tables should be gone
        assert "compressions" not in table_names
        assert "compression_sources" not in table_names
        assert "compression_results" not in table_names

        # Verify schema version is now 6
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert int(row.value) >= 7

        engine.dispose()

    def test_new_db_starts_at_v6(self):
        """Fresh database gets schema_version=6."""
        engine = create_trace_engine(":memory:")
        init_db(engine)

        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert int(row.value) >= 7

        engine.dispose()

    def test_spawn_pointer_row_roundtrip(self, session):
        """Create SpawnPointerRow, save, retrieve."""
        _make_commit(session, "parent-commit-1", tract_id="parent-tract")
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        row = SpawnPointerRow(
            parent_tract_id="parent-tract",
            parent_commit_hash="parent-commit-1",
            child_tract_id="child-tract",
            purpose="research subtask",
            inheritance_mode="head_snapshot",
            display_name="Research Agent",
            created_at=now,
        )
        session.add(row)
        session.flush()

        fetched = session.execute(
            select(SpawnPointerRow).where(SpawnPointerRow.id == row.id)
        ).scalar_one()

        assert fetched.parent_tract_id == "parent-tract"
        assert fetched.parent_commit_hash == "parent-commit-1"
        assert fetched.child_tract_id == "child-tract"
        assert fetched.purpose == "research subtask"
        assert fetched.inheritance_mode == "head_snapshot"
        assert fetched.display_name == "Research Agent"


# ===========================================================================
# Repository Tests
# ===========================================================================


class TestSpawnPointerRepository:
    """Tests for SqliteSpawnPointerRepository."""

    @pytest.fixture
    def repo(self, session: Session) -> SqliteSpawnPointerRepository:
        return SqliteSpawnPointerRepository(session)

    @pytest.fixture
    def setup_commits(self, session: Session) -> list[str]:
        """Create commits for FK satisfaction."""
        hashes = ["commit-a", "commit-b", "commit-c"]
        prev = None
        for h in hashes:
            _make_commit(session, h, parent_hash=prev)
            prev = h
        return hashes

    def test_save_and_get(
        self, repo: SqliteSpawnPointerRepository, setup_commits: list[str]
    ):
        """save() then get() returns matching data."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        row = repo.save(
            parent_tract_id="test-tract",
            parent_commit_hash="commit-a",
            child_tract_id="child-1",
            purpose="analyze data",
            inheritance_mode="full_clone",
            display_name="Analyzer",
            created_at=now,
        )

        fetched = repo.get(row.id)
        assert fetched is not None
        assert fetched.parent_tract_id == "test-tract"
        assert fetched.parent_commit_hash == "commit-a"
        assert fetched.child_tract_id == "child-1"
        assert fetched.purpose == "analyze data"
        assert fetched.inheritance_mode == "full_clone"
        assert fetched.display_name == "Analyzer"

    def test_get_by_child(
        self, repo: SqliteSpawnPointerRepository, setup_commits: list[str]
    ):
        """get_by_child() returns the spawn pointer for a child tract."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save(
            parent_tract_id="test-tract",
            parent_commit_hash="commit-a",
            child_tract_id="child-1",
            purpose="task-a",
            inheritance_mode="head_snapshot",
            display_name=None,
            created_at=now,
        )

        result = repo.get_by_child("child-1")
        assert result is not None
        assert result.parent_tract_id == "test-tract"
        assert result.purpose == "task-a"

    def test_get_by_child_not_found(self, repo: SqliteSpawnPointerRepository):
        """get_by_child() returns None for non-child tract."""
        result = repo.get_by_child("nonexistent-tract")
        assert result is None

    def test_get_children(
        self, repo: SqliteSpawnPointerRepository, setup_commits: list[str]
    ):
        """get_children() returns all children, ordered by created_at."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Create children in specific time order
        repo.save(
            parent_tract_id="test-tract",
            parent_commit_hash="commit-a",
            child_tract_id="child-2",
            purpose="second task",
            inheritance_mode="selective",
            display_name=None,
            created_at=now + timedelta(seconds=1),
        )
        repo.save(
            parent_tract_id="test-tract",
            parent_commit_hash="commit-a",
            child_tract_id="child-1",
            purpose="first task",
            inheritance_mode="full_clone",
            display_name=None,
            created_at=now,
        )

        children = repo.get_children("test-tract")
        assert len(children) == 2
        assert children[0].child_tract_id == "child-1"  # earlier created_at
        assert children[1].child_tract_id == "child-2"

    def test_get_children_empty(self, repo: SqliteSpawnPointerRepository):
        """get_children() returns empty list for tract with no children."""
        children = repo.get_children("no-children-tract")
        assert children == []

    def test_get_all(
        self, repo: SqliteSpawnPointerRepository, setup_commits: list[str]
    ):
        """get_all() returns pointers where tract is parent OR child."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # test-tract is parent of child-1
        repo.save(
            parent_tract_id="test-tract",
            parent_commit_hash="commit-a",
            child_tract_id="child-1",
            purpose="task-a",
            inheritance_mode="full_clone",
            display_name=None,
            created_at=now,
        )

        # other-parent spawned test-tract as child
        repo.save(
            parent_tract_id="other-parent",
            parent_commit_hash=None,
            child_tract_id="test-tract",
            purpose="main task",
            inheritance_mode="head_snapshot",
            display_name=None,
            created_at=now,
        )

        all_pointers = repo.get_all("test-tract")
        assert len(all_pointers) == 2

    def test_has_ancestor_true(
        self, repo: SqliteSpawnPointerRepository, setup_commits: list[str]
    ):
        """tract A spawns B spawns C; has_ancestor(C, A) is True."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # A spawns B
        repo.save(
            parent_tract_id="tract-A",
            parent_commit_hash=None,
            child_tract_id="tract-B",
            purpose="subtask-1",
            inheritance_mode="full_clone",
            display_name=None,
            created_at=now,
        )

        # B spawns C
        repo.save(
            parent_tract_id="tract-B",
            parent_commit_hash=None,
            child_tract_id="tract-C",
            purpose="subtask-2",
            inheritance_mode="full_clone",
            display_name=None,
            created_at=now,
        )

        assert repo.has_ancestor("tract-C", "tract-A") is True
        assert repo.has_ancestor("tract-C", "tract-B") is True
        assert repo.has_ancestor("tract-B", "tract-A") is True

    def test_has_ancestor_false(self, repo: SqliteSpawnPointerRepository):
        """Unrelated tracts; has_ancestor() returns False."""
        assert repo.has_ancestor("tract-X", "tract-Y") is False


# ===========================================================================
# Model Tests
# ===========================================================================


class TestSessionModels:
    """Tests for SessionContent, SpawnInfo, and CollapseResult."""

    def test_session_content_valid(self):
        """SessionContent validates with all fields."""
        from tract.models.session import SessionContent

        content = SessionContent(
            session_type="end",
            summary="Completed data analysis",
            decisions=["Use pandas", "Filter outliers"],
            failed_approaches=["Tried numpy first"],
            next_steps=["Generate report"],
        )
        assert content.content_type == "session"
        assert content.session_type == "end"
        assert len(content.decisions) == 2
        assert len(content.failed_approaches) == 1
        assert len(content.next_steps) == 1

    def test_session_content_in_union(self):
        """validate_content({content_type: 'session', ...}) returns SessionContent."""
        from tract.models.content import validate_content
        from tract.models.session import SessionContent

        result = validate_content({
            "content_type": "session",
            "session_type": "start",
            "summary": "Beginning research phase",
        })
        assert isinstance(result, SessionContent)
        assert result.session_type == "start"
        assert result.summary == "Beginning research phase"
        assert result.decisions == []
        assert result.next_steps == []

    def test_spawn_info_frozen(self):
        """SpawnInfo is immutable."""
        from tract.models.session import SpawnInfo

        now = datetime.now(timezone.utc)
        info = SpawnInfo(
            spawn_id=1,
            parent_tract_id="parent",
            parent_commit_hash="abc123",
            child_tract_id="child",
            purpose="research",
            inheritance_mode="full_clone",
            display_name="Researcher",
            created_at=now,
        )
        with pytest.raises(AttributeError):
            info.spawn_id = 2  # type: ignore[misc]

    def test_collapse_result_frozen(self):
        """CollapseResult is immutable."""
        from tract.models.session import CollapseResult

        result = CollapseResult(
            parent_commit_hash="abc123",
            child_tract_id="child-1",
            summary_text="Subagent completed: analysis done",
            summary_tokens=50,
            source_tokens=500,
            purpose="analyze data",
        )
        with pytest.raises(AttributeError):
            result.summary_text = "changed"  # type: ignore[misc]

    def test_session_content_type_hints(self):
        """'session' is in BUILTIN_TYPE_HINTS with correct values."""
        from tract.models.content import BUILTIN_TYPE_HINTS

        assert "session" in BUILTIN_TYPE_HINTS
        hints = BUILTIN_TYPE_HINTS["session"]
        assert hints.default_priority == "pinned"
        assert hints.default_role == "system"
        assert hints.compression_priority == 95

    def test_session_in_builtin_content_types(self):
        """'session' is in BUILTIN_CONTENT_TYPES."""
        from tract.models.content import BUILTIN_CONTENT_TYPES

        assert "session" in BUILTIN_CONTENT_TYPES


# ===========================================================================
# Prompt Tests
# ===========================================================================


class TestCollapsePrompt:
    """Tests for the collapse summarization prompt."""

    def test_collapse_system_prompt_not_empty(self):
        """DEFAULT_COLLAPSE_SYSTEM is a non-empty string."""
        from tract.prompts.summarize import DEFAULT_COLLAPSE_SYSTEM

        assert isinstance(DEFAULT_COLLAPSE_SYSTEM, str)
        assert len(DEFAULT_COLLAPSE_SYSTEM) > 50
        assert "subagent" in DEFAULT_COLLAPSE_SYSTEM.lower()

    def test_build_collapse_prompt_basic(self):
        """Includes purpose and messages_text."""
        from tract.prompts.summarize import build_collapse_prompt

        prompt = build_collapse_prompt(
            messages_text="Agent found 3 bugs in the codebase.",
            purpose="code review",
        )
        assert "code review" in prompt
        assert "Agent found 3 bugs" in prompt
        assert "Purpose:" in prompt

    def test_build_collapse_prompt_with_options(self):
        """Includes target_tokens and instructions when provided."""
        from tract.prompts.summarize import build_collapse_prompt

        prompt = build_collapse_prompt(
            messages_text="Some work output",
            purpose="data analysis",
            target_tokens=200,
            instructions="Focus on findings only",
        )
        assert "200 tokens" in prompt
        assert "Focus on findings only" in prompt
        assert "data analysis" in prompt


# ===========================================================================
# Exception Tests
# ===========================================================================


class TestSpawnExceptions:
    """Tests for SpawnError and SessionError."""

    def test_spawn_error_inherits_trace_error(self):
        """SpawnError is a TraceError."""
        from tract.exceptions import SpawnError, TraceError

        err = SpawnError("spawn failed")
        assert isinstance(err, TraceError)
        assert str(err) == "spawn failed"

    def test_session_error_inherits_trace_error(self):
        """SessionError is a TraceError."""
        from tract.exceptions import SessionError, TraceError

        err = SessionError("session failed")
        assert isinstance(err, TraceError)
        assert str(err) == "session failed"
