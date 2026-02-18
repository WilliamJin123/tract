"""Tests for compression storage: schema, repository, domain models, and prompts.

Covers:
- Compression tables created by init_db
- Schema migration v2->v3
- SqliteCompressionRepository CRUD operations
- CompressResult, PendingCompression, GCResult, ReorderWarning models
- Default summarization prompt and builder
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from tract.models.commit import CommitOperation
from tract.storage.engine import create_trace_engine, init_db
from tract.storage.schema import (
    BlobRow,
    CommitRow,
    CompressionResultRow,
    CompressionRow,
    CompressionSourceRow,
    TraceMetaRow,
)
from tract.storage.sqlite import SqliteCompressionRepository


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


class TestCompressionSchema:
    """Tests for compression table creation and migration."""

    def test_compression_tables_created(self, engine):
        """init_db creates all 3 compression tables."""
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "compressions" in table_names
        assert "compression_sources" in table_names
        assert "compression_results" in table_names

    def test_migration_v2_to_v3_and_v4(self):
        """Start with schema_version=2, call init_db, verify tables + version=5.

        Migration chain: v2 -> v3 (compression tables) -> v4 (spawn_pointers) -> v5 (policy).
        """
        # Create a v2 database manually (without compression tables)
        engine = create_trace_engine(":memory:")

        # Create only the base tables that existed in v2
        from tract.storage.schema import Base

        # Create all tables first, then drop compression + spawn + policy to simulate v2
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS policy_log"))
            conn.execute(text("DROP TABLE IF EXISTS policy_proposals"))
            conn.execute(text("DROP TABLE IF EXISTS compression_results"))
            conn.execute(text("DROP TABLE IF EXISTS compression_sources"))
            conn.execute(text("DROP TABLE IF EXISTS compressions"))
            conn.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            conn.commit()

        # Set schema version to 2
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            session.add(TraceMetaRow(key="schema_version", value="2"))
            session.commit()

        # Now call init_db -- should migrate v2->v3->v4
        init_db(engine)

        # Verify compression tables exist
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "compressions" in table_names
        assert "compression_sources" in table_names
        assert "compression_results" in table_names
        assert "spawn_pointers" in table_names

        # Verify schema version is now 4
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert row.value == "5"

        engine.dispose()

    def test_new_db_starts_at_v4(self):
        """Fresh database gets schema_version=5."""
        engine = create_trace_engine(":memory:")
        init_db(engine)

        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert row.value == "5"

        engine.dispose()

    def test_compression_row_roundtrip(self, session):
        """Create CompressionRow, save, retrieve."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        row = CompressionRow(
            compression_id="comp-001",
            tract_id="test-tract",
            branch_name="main",
            created_at=now,
            original_tokens=1000,
            compressed_tokens=200,
            target_tokens=250,
            instructions="Focus on decisions",
        )
        session.add(row)
        session.flush()

        fetched = session.execute(
            select(CompressionRow).where(
                CompressionRow.compression_id == "comp-001"
            )
        ).scalar_one()

        assert fetched.tract_id == "test-tract"
        assert fetched.branch_name == "main"
        assert fetched.original_tokens == 1000
        assert fetched.compressed_tokens == 200
        assert fetched.target_tokens == 250
        assert fetched.instructions == "Focus on decisions"

    def test_compression_source_result_roundtrip(self, session):
        """Create source + result rows, retrieve."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Create prerequisite compression record
        session.add(
            CompressionRow(
                compression_id="comp-002",
                tract_id="test-tract",
                created_at=now,
                original_tokens=500,
                compressed_tokens=100,
            )
        )

        # Create prerequisite commits
        _make_commit(session, "src-hash-1")
        _make_commit(session, "src-hash-2")
        _make_commit(session, "res-hash-1")

        # Add source and result rows
        session.add(
            CompressionSourceRow(
                compression_id="comp-002", commit_hash="src-hash-1", position=0
            )
        )
        session.add(
            CompressionSourceRow(
                compression_id="comp-002", commit_hash="src-hash-2", position=1
            )
        )
        session.add(
            CompressionResultRow(
                compression_id="comp-002", commit_hash="res-hash-1", position=0
            )
        )
        session.flush()

        # Retrieve sources
        sources = (
            session.execute(
                select(CompressionSourceRow)
                .where(CompressionSourceRow.compression_id == "comp-002")
                .order_by(CompressionSourceRow.position)
            )
            .scalars()
            .all()
        )
        assert len(sources) == 2
        assert sources[0].commit_hash == "src-hash-1"
        assert sources[1].commit_hash == "src-hash-2"

        # Retrieve results
        results = (
            session.execute(
                select(CompressionResultRow)
                .where(CompressionResultRow.compression_id == "comp-002")
                .order_by(CompressionResultRow.position)
            )
            .scalars()
            .all()
        )
        assert len(results) == 1
        assert results[0].commit_hash == "res-hash-1"


# ===========================================================================
# Repository Tests
# ===========================================================================


class TestCompressionRepository:
    """Tests for SqliteCompressionRepository."""

    @pytest.fixture
    def repo(self, session: Session) -> SqliteCompressionRepository:
        return SqliteCompressionRepository(session)

    @pytest.fixture
    def setup_commits(self, session: Session) -> list[str]:
        """Create a set of commits for use in compression tests."""
        hashes = ["commit-a", "commit-b", "commit-c", "commit-d", "commit-e"]
        for h in hashes:
            _make_commit(session, h)
        return hashes

    def test_save_and_get_record(
        self, repo: SqliteCompressionRepository, session: Session
    ):
        """save_record() then get_record() returns matching data."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-r1",
            tract_id="test-tract",
            branch_name="main",
            created_at=now,
            original_tokens=800,
            compressed_tokens=150,
            target_tokens=200,
            instructions="Be concise",
        )

        record = repo.get_record("comp-r1")
        assert record is not None
        assert record.tract_id == "test-tract"
        assert record.branch_name == "main"
        assert record.original_tokens == 800
        assert record.compressed_tokens == 150
        assert record.target_tokens == 200
        assert record.instructions == "Be concise"

    def test_add_and_get_sources(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """add_source() x3, get_sources() returns in position order."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-s1",
            tract_id="test-tract",
            branch_name=None,
            created_at=now,
            original_tokens=600,
            compressed_tokens=100,
            target_tokens=None,
            instructions=None,
        )

        # Add sources in non-sequential order
        repo.add_source("comp-s1", "commit-c", position=2)
        repo.add_source("comp-s1", "commit-a", position=0)
        repo.add_source("comp-s1", "commit-b", position=1)

        sources = repo.get_sources("comp-s1")
        assert len(sources) == 3
        assert sources[0].commit_hash == "commit-a"
        assert sources[1].commit_hash == "commit-b"
        assert sources[2].commit_hash == "commit-c"

    def test_add_and_get_results(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """add_result() x2, get_results() returns in position order."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-res1",
            tract_id="test-tract",
            branch_name=None,
            created_at=now,
            original_tokens=500,
            compressed_tokens=80,
            target_tokens=None,
            instructions=None,
        )

        repo.add_result("comp-res1", "commit-e", position=1)
        repo.add_result("comp-res1", "commit-d", position=0)

        results = repo.get_results("comp-res1")
        assert len(results) == 2
        assert results[0].commit_hash == "commit-d"
        assert results[1].commit_hash == "commit-e"

    def test_is_source_of_true(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """Commit that is a compression source returns True."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-iso1",
            tract_id="test-tract",
            branch_name=None,
            created_at=now,
            original_tokens=300,
            compressed_tokens=50,
            target_tokens=None,
            instructions=None,
        )
        repo.add_source("comp-iso1", "commit-a", position=0)

        assert repo.is_source_of("commit-a") is True

    def test_is_source_of_false(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """Commit that is not a compression source returns False."""
        assert repo.is_source_of("commit-a") is False

    def test_get_all_source_hashes(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """Returns set of all source hashes for a tract."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Compression 1
        repo.save_record(
            compression_id="comp-gash1",
            tract_id="test-tract",
            branch_name=None,
            created_at=now,
            original_tokens=300,
            compressed_tokens=50,
            target_tokens=None,
            instructions=None,
        )
        repo.add_source("comp-gash1", "commit-a", position=0)
        repo.add_source("comp-gash1", "commit-b", position=1)

        # Compression 2
        repo.save_record(
            compression_id="comp-gash2",
            tract_id="test-tract",
            branch_name=None,
            created_at=now,
            original_tokens=200,
            compressed_tokens=40,
            target_tokens=None,
            instructions=None,
        )
        repo.add_source("comp-gash2", "commit-c", position=0)

        hashes = repo.get_all_source_hashes("test-tract")
        assert hashes == {"commit-a", "commit-b", "commit-c"}

    def test_get_record_not_found(self, repo: SqliteCompressionRepository):
        """get_record() returns None for nonexistent compression_id."""
        assert repo.get_record("nonexistent-id") is None

    def test_get_sources_empty(self, repo: SqliteCompressionRepository):
        """get_sources() returns empty list for nonexistent compression_id."""
        assert repo.get_sources("nonexistent-id") == []

    def test_delete_source(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """delete_source removes CompressionSourceRow entries."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-ds1",
            tract_id="test-tract",
            branch_name="main",
            created_at=now,
            original_tokens=100,
            compressed_tokens=50,
            target_tokens=None,
            instructions=None,
        )
        repo.add_source("comp-ds1", "commit-a", 0)
        repo.add_source("comp-ds1", "commit-b", 1)

        repo.delete_source("commit-a")

        sources = repo.get_sources("comp-ds1")
        assert len(sources) == 1
        assert sources[0].commit_hash == "commit-b"

    def test_delete_result(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """delete_result removes CompressionResultRow entries."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-dr1",
            tract_id="test-tract",
            branch_name="main",
            created_at=now,
            original_tokens=100,
            compressed_tokens=50,
            target_tokens=None,
            instructions=None,
        )
        repo.add_result("comp-dr1", "commit-d", 0)
        repo.add_result("comp-dr1", "commit-e", 1)

        repo.delete_result("commit-d")

        results = repo.get_results("comp-dr1")
        assert len(results) == 1
        assert results[0].commit_hash == "commit-e"

    def test_delete_record(
        self,
        repo: SqliteCompressionRepository,
        setup_commits: list[str],
    ):
        """delete_record removes the CompressionRow and all associations."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            compression_id="comp-drc1",
            tract_id="test-tract",
            branch_name="main",
            created_at=now,
            original_tokens=100,
            compressed_tokens=50,
            target_tokens=None,
            instructions=None,
        )
        repo.add_source("comp-drc1", "commit-a", 0)
        repo.add_result("comp-drc1", "commit-d", 0)

        repo.delete_record("comp-drc1")

        assert repo.get_record("comp-drc1") is None
        assert repo.get_sources("comp-drc1") == []
        assert repo.get_results("comp-drc1") == []


# ===========================================================================
# Model Tests
# ===========================================================================


class TestCompressionModels:
    """Tests for compression domain models."""

    def test_compress_result_frozen(self):
        """CompressResult is immutable."""
        from tract.models.compression import CompressResult

        result = CompressResult(
            compression_id="comp-1",
            original_tokens=1000,
            compressed_tokens=200,
            source_commits=("a", "b"),
            summary_commits=("c",),
            preserved_commits=("d",),
            compression_ratio=0.2,
            new_head="c",
        )
        with pytest.raises(AttributeError):
            result.compression_id = "changed"  # type: ignore[misc]

    def test_pending_compression_edit_summary(self):
        """edit_summary() replaces text at index."""
        from tract.models.compression import PendingCompression

        pending = PendingCompression(
            summaries=["draft 1", "draft 2"],
            source_commits=["a", "b"],
            preserved_commits=[],
            original_tokens=500,
            estimated_tokens=100,
        )
        pending.edit_summary(0, "revised draft 1")
        assert pending.summaries[0] == "revised draft 1"
        assert pending.summaries[1] == "draft 2"

    def test_pending_compression_approve_no_fn(self):
        """approve() without _commit_fn raises CompressionError."""
        from tract.exceptions import CompressionError
        from tract.models.compression import PendingCompression

        pending = PendingCompression(
            summaries=["draft"],
            source_commits=["a"],
            preserved_commits=[],
            original_tokens=500,
            estimated_tokens=100,
        )
        with pytest.raises(CompressionError, match="no commit function"):
            pending.approve()

    def test_gc_result_frozen(self):
        """GCResult is immutable."""
        from tract.models.compression import GCResult

        result = GCResult(
            commits_removed=5,
            blobs_removed=3,
            tokens_freed=2000,
            archives_removed=1,
            duration_seconds=0.5,
        )
        with pytest.raises(AttributeError):
            result.commits_removed = 10  # type: ignore[misc]

    def test_reorder_warning_frozen(self):
        """ReorderWarning is immutable."""
        from tract.models.compression import ReorderWarning

        warning = ReorderWarning(
            warning_type="edit_before_target",
            commit_hash="abc123",
            description="Edit commit appears before its target",
            severity="structural",
        )
        with pytest.raises(AttributeError):
            warning.severity = "semantic"  # type: ignore[misc]


# ===========================================================================
# Prompt Tests
# ===========================================================================


class TestSummarizationPrompt:
    """Tests for the default summarization prompt module."""

    def test_default_system_prompt_not_empty(self):
        """DEFAULT_SUMMARIZE_SYSTEM is a non-empty string."""
        from tract.prompts.summarize import DEFAULT_SUMMARIZE_SYSTEM

        assert isinstance(DEFAULT_SUMMARIZE_SYSTEM, str)
        assert len(DEFAULT_SUMMARIZE_SYSTEM) > 50

    def test_build_prompt_basic(self):
        """build_summarize_prompt("text") includes the text."""
        from tract.prompts.summarize import build_summarize_prompt

        prompt = build_summarize_prompt("Hello world conversation")
        assert "Hello world conversation" in prompt
        assert "Summarize the following conversation segment" in prompt

    def test_build_prompt_with_target_and_instructions(self):
        """Includes target tokens and instructions when provided."""
        from tract.prompts.summarize import build_summarize_prompt

        prompt = build_summarize_prompt(
            "Some conversation text",
            target_tokens=500,
            instructions="Focus on code decisions",
        )
        assert "500 tokens" in prompt
        assert "Focus on code decisions" in prompt
        assert "Some conversation text" in prompt
