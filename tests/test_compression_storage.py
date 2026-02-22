"""Tests for unified operation event and compile record storage.

Covers:
- OperationEventRow, OperationCommitRow table creation and CRUD
- CompileRecordRow, CompileEffectiveRow table creation and CRUD
- SqliteOperationEventRepository all 9 methods
- SqliteCompileRecordRepository all 5 methods
- Schema migration v2->v6 and v5->v6
- CompressResult, PendingCompression, GCResult, ReorderWarning models
- Default summarization prompt and builder
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from tract.models.commit import CommitOperation
from tract.storage.engine import create_trace_engine, init_db
from tract.storage.schema import (
    BlobRow,
    CommitRow,
    CompileEffectiveRow,
    CompileRecordRow,
    OperationCommitRow,
    OperationEventRow,
    TraceMetaRow,
)
from tract.storage.sqlite import (
    SqliteCompileRecordRepository,
    SqliteOperationEventRepository,
)


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


class TestOperationEventSchema:
    """Tests for operation event and compile record table creation."""

    def test_init_db_creates_latest_schema(self):
        """Fresh database gets the latest schema version."""
        engine = create_trace_engine(":memory:")
        init_db(engine)

        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            # Schema version should be >= 7 (v7 added retention_json)
            assert int(row.value) >= 7

        engine.dispose()

    def test_operation_event_table_exists(self, engine):
        """init_db creates the operation_events table."""
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "operation_events" in table_names
        assert "operation_commits" in table_names

    def test_compile_record_table_exists(self, engine):
        """init_db creates the compile_records table."""
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "compile_records" in table_names
        assert "compile_effectives" in table_names

    def test_migrate_v2_to_latest(self):
        """Start with schema_version=2, call init_db, verify latest and new tables exist."""
        engine = create_trace_engine(":memory:")

        # Create all tables first, then drop to simulate v2
        from tract.storage.schema import Base

        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS policy_log"))
            conn.execute(text("DROP TABLE IF EXISTS policy_proposals"))
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))
            conn.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            conn.commit()

        # Set schema version to 2
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            session.add(TraceMetaRow(key="schema_version", value="2"))
            session.commit()

        # Call init_db -- should migrate v2 to latest
        init_db(engine)

        # Verify version is >= 7 (v7 added retention_json)
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert int(row.value) >= 7

        # Verify new tables exist, old tables dropped
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "operation_events" in table_names
        assert "operation_commits" in table_names
        assert "compile_records" in table_names
        assert "compile_effectives" in table_names
        assert "spawn_pointers" in table_names
        assert "policy_proposals" in table_names
        # Old tables should be dropped
        assert "compressions" not in table_names
        assert "compression_sources" not in table_names
        assert "compression_results" not in table_names

        engine.dispose()

    def test_migrate_v5_to_v6(self):
        """Start with schema_version=5 with compression data, verify migration to v6."""
        engine = create_trace_engine(":memory:")

        # Create all tables first, then set up v5 state
        from tract.storage.schema import Base

        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            # Drop new v6 tables to simulate v5
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))

            # Create old compression tables
            conn.execute(text("""
                CREATE TABLE compressions (
                    compression_id VARCHAR(64) PRIMARY KEY,
                    tract_id VARCHAR(64) NOT NULL,
                    branch_name VARCHAR(255),
                    created_at DATETIME NOT NULL,
                    original_tokens INTEGER NOT NULL DEFAULT 0,
                    compressed_tokens INTEGER NOT NULL DEFAULT 0,
                    target_tokens INTEGER,
                    instructions TEXT
                )
            """))
            conn.execute(text("""
                CREATE TABLE compression_sources (
                    compression_id VARCHAR(64) NOT NULL
                        REFERENCES compressions(compression_id),
                    commit_hash VARCHAR(64) NOT NULL
                        REFERENCES commits(commit_hash),
                    position INTEGER NOT NULL,
                    PRIMARY KEY (compression_id, commit_hash)
                )
            """))
            conn.execute(text("""
                CREATE TABLE compression_results (
                    compression_id VARCHAR(64) NOT NULL
                        REFERENCES compressions(compression_id),
                    commit_hash VARCHAR(64) NOT NULL
                        REFERENCES commits(commit_hash),
                    position INTEGER NOT NULL,
                    PRIMARY KEY (compression_id, commit_hash)
                )
            """))

            # Create test blobs and commits for FK satisfaction
            now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            conn.execute(text(
                "INSERT INTO blobs (content_hash, payload_json, byte_size, token_count, created_at) "
                "VALUES ('blob-src1', '{\"text\":\"test\"}', 16, 2, :now)"
            ), {"now": now})
            conn.execute(text(
                "INSERT INTO blobs (content_hash, payload_json, byte_size, token_count, created_at) "
                "VALUES ('blob-src2', '{\"text\":\"test\"}', 16, 2, :now)"
            ), {"now": now})
            conn.execute(text(
                "INSERT INTO blobs (content_hash, payload_json, byte_size, token_count, created_at) "
                "VALUES ('blob-res1', '{\"text\":\"test\"}', 16, 2, :now)"
            ), {"now": now})
            conn.execute(text(
                "INSERT INTO commits (commit_hash, tract_id, content_hash, content_type, operation, token_count, created_at) "
                "VALUES ('src-hash-1', 'test-tract', 'blob-src1', 'dialogue', 'APPEND', 10, :now)"
            ), {"now": now})
            conn.execute(text(
                "INSERT INTO commits (commit_hash, tract_id, content_hash, content_type, operation, token_count, created_at) "
                "VALUES ('src-hash-2', 'test-tract', 'blob-src2', 'dialogue', 'APPEND', 10, :now)"
            ), {"now": now})
            conn.execute(text(
                "INSERT INTO commits (commit_hash, tract_id, content_hash, content_type, operation, token_count, created_at) "
                "VALUES ('res-hash-1', 'test-tract', 'blob-res1', 'dialogue', 'APPEND', 10, :now)"
            ), {"now": now})

            # Insert test compression data
            conn.execute(text(
                "INSERT INTO compressions "
                "(compression_id, tract_id, branch_name, created_at, original_tokens, compressed_tokens, target_tokens, instructions) "
                "VALUES ('comp-001', 'test-tract', 'main', :now, 1000, 200, 250, 'Focus on decisions')"
            ), {"now": now})
            conn.execute(text(
                "INSERT INTO compression_sources (compression_id, commit_hash, position) "
                "VALUES ('comp-001', 'src-hash-1', 0)"
            ))
            conn.execute(text(
                "INSERT INTO compression_sources (compression_id, commit_hash, position) "
                "VALUES ('comp-001', 'src-hash-2', 1)"
            ))
            conn.execute(text(
                "INSERT INTO compression_results (compression_id, commit_hash, position) "
                "VALUES ('comp-001', 'res-hash-1', 0)"
            ))
            conn.commit()

        # Set schema version to 5
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            meta = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one_or_none()
            if meta is None:
                session.add(TraceMetaRow(key="schema_version", value="5"))
            else:
                meta.value = "5"
            session.commit()

        # Run migration
        init_db(engine)

        # Verify version is >= 7 (v7 added retention_json)
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert int(row.value) >= 7

        # Verify new tables exist
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "operation_events" in table_names
        assert "operation_commits" in table_names

        # Verify old tables dropped
        assert "compressions" not in table_names
        assert "compression_sources" not in table_names
        assert "compression_results" not in table_names

        # Verify data was migrated
        with engine.connect() as conn:
            # Check operation_events
            events = conn.execute(text("SELECT * FROM operation_events")).fetchall()
            assert len(events) == 1
            event = events[0]
            assert event[0] == "comp-001"  # event_id
            assert event[1] == "test-tract"  # tract_id
            assert event[2] == "compress"  # event_type
            assert event[3] == "main"  # branch_name
            assert event[5] == 1000  # original_tokens
            assert event[6] == 200  # compressed_tokens
            # Check params_json
            params = json.loads(event[7]) if event[7] else {}
            assert params.get("target_tokens") == 250
            assert params.get("instructions") == "Focus on decisions"

            # Check operation_commits
            commits = conn.execute(
                text("SELECT * FROM operation_commits ORDER BY role, position")
            ).fetchall()
            assert len(commits) == 3
            # Results come first alphabetically
            result_commits = [c for c in commits if c[2] == "result"]
            source_commits = [c for c in commits if c[2] == "source"]
            assert len(source_commits) == 2
            assert len(result_commits) == 1
            assert source_commits[0][1] == "src-hash-1"  # commit_hash
            assert source_commits[1][1] == "src-hash-2"
            assert result_commits[0][1] == "res-hash-1"

        engine.dispose()


# ===========================================================================
# OperationEventRepository Tests
# ===========================================================================


class TestOperationEventRepository:
    """Tests for SqliteOperationEventRepository."""

    @pytest.fixture
    def repo(self, session: Session) -> SqliteOperationEventRepository:
        return SqliteOperationEventRepository(session)

    @pytest.fixture
    def setup_commits(self, session: Session) -> list[str]:
        """Create a set of commits for use in operation event tests."""
        hashes = ["commit-a", "commit-b", "commit-c", "commit-d", "commit-e"]
        for h in hashes:
            _make_commit(session, h)
        return hashes

    def test_save_and_get_event(
        self, repo: SqliteOperationEventRepository, session: Session
    ):
        """save_event() then get_event() returns matching data."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_event(
            event_id="evt-001",
            tract_id="test-tract",
            event_type="compress",
            branch_name="main",
            created_at=now,
            original_tokens=800,
            compressed_tokens=150,
            params_json={"target_tokens": 200},
        )

        event = repo.get_event("evt-001")
        assert event is not None
        assert event.tract_id == "test-tract"
        assert event.event_type == "compress"
        assert event.branch_name == "main"
        assert event.original_tokens == 800
        assert event.compressed_tokens == 150
        assert event.params_json == {"target_tokens": 200}

    def test_add_and_get_commits(
        self,
        repo: SqliteOperationEventRepository,
        setup_commits: list[str],
    ):
        """add_commit() for sources and results, get_commits() returns filtered."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_event(
            event_id="evt-002",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=600,
            compressed_tokens=100,
            params_json=None,
        )

        # Add source commits in non-sequential order
        repo.add_commit("evt-002", "commit-c", "source", position=2)
        repo.add_commit("evt-002", "commit-a", "source", position=0)
        repo.add_commit("evt-002", "commit-b", "source", position=1)
        # Add result commit
        repo.add_commit("evt-002", "commit-d", "result", position=0)

        # Get all commits
        all_commits = repo.get_commits("evt-002")
        assert len(all_commits) == 4

        # Get by role
        sources = repo.get_commits("evt-002", role="source")
        assert len(sources) == 3
        assert sources[0].commit_hash == "commit-a"
        assert sources[1].commit_hash == "commit-b"
        assert sources[2].commit_hash == "commit-c"

        results = repo.get_commits("evt-002", role="result")
        assert len(results) == 1
        assert results[0].commit_hash == "commit-d"

    def test_is_source_of(
        self,
        repo: SqliteOperationEventRepository,
        setup_commits: list[str],
    ):
        """is_source_of returns True for source commits, False for others."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_event(
            event_id="evt-iso",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=300,
            compressed_tokens=50,
            params_json=None,
        )
        repo.add_commit("evt-iso", "commit-a", "source", position=0)
        repo.add_commit("evt-iso", "commit-d", "result", position=0)

        assert repo.is_source_of("commit-a") is True
        assert repo.is_source_of("commit-d") is False  # result, not source
        assert repo.is_source_of("commit-e") is False  # not in any event

    def test_get_all_source_hashes(
        self,
        repo: SqliteOperationEventRepository,
        setup_commits: list[str],
    ):
        """Returns set of all source hashes for a tract."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        # Event 1
        repo.save_event(
            event_id="evt-gash1",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=300,
            compressed_tokens=50,
            params_json=None,
        )
        repo.add_commit("evt-gash1", "commit-a", "source", position=0)
        repo.add_commit("evt-gash1", "commit-b", "source", position=1)

        # Event 2
        repo.save_event(
            event_id="evt-gash2",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=200,
            compressed_tokens=40,
            params_json=None,
        )
        repo.add_commit("evt-gash2", "commit-c", "source", position=0)

        hashes = repo.get_all_source_hashes("test-tract")
        assert hashes == {"commit-a", "commit-b", "commit-c"}

    def test_get_all_ids(
        self,
        repo: SqliteOperationEventRepository,
        setup_commits: list[str],
    ):
        """Returns list of all event IDs for a tract."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)

        repo.save_event(
            event_id="evt-id1",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=100,
            compressed_tokens=20,
            params_json=None,
        )
        repo.save_event(
            event_id="evt-id2",
            tract_id="test-tract",
            event_type="reorganize",
            branch_name=None,
            created_at=now,
            original_tokens=200,
            compressed_tokens=40,
            params_json=None,
        )

        ids = repo.get_all_ids("test-tract")
        assert set(ids) == {"evt-id1", "evt-id2"}

    def test_delete_commit(
        self,
        repo: SqliteOperationEventRepository,
        setup_commits: list[str],
    ):
        """delete_commit removes OperationCommitRow entries for a hash."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_event(
            event_id="evt-dc1",
            tract_id="test-tract",
            event_type="compress",
            branch_name="main",
            created_at=now,
            original_tokens=100,
            compressed_tokens=50,
            params_json=None,
        )
        repo.add_commit("evt-dc1", "commit-a", "source", 0)
        repo.add_commit("evt-dc1", "commit-b", "source", 1)

        repo.delete_commit("commit-a")

        sources = repo.get_commits("evt-dc1", role="source")
        assert len(sources) == 1
        assert sources[0].commit_hash == "commit-b"

    def test_delete_event(
        self,
        repo: SqliteOperationEventRepository,
        setup_commits: list[str],
    ):
        """delete_event removes the event and all its commit associations."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_event(
            event_id="evt-de1",
            tract_id="test-tract",
            event_type="compress",
            branch_name="main",
            created_at=now,
            original_tokens=100,
            compressed_tokens=50,
            params_json=None,
        )
        repo.add_commit("evt-de1", "commit-a", "source", 0)
        repo.add_commit("evt-de1", "commit-d", "result", 0)

        repo.delete_event("evt-de1")

        assert repo.get_event("evt-de1") is None
        assert repo.get_commits("evt-de1") == []

    def test_event_types(
        self, repo: SqliteOperationEventRepository
    ):
        """All three event types (compress, reorganize, import) work."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for event_type in ["compress", "reorganize", "import"]:
            repo.save_event(
                event_id=f"evt-type-{event_type}",
                tract_id="test-tract",
                event_type=event_type,
                branch_name=None,
                created_at=now,
                original_tokens=100,
                compressed_tokens=50,
                params_json=None,
            )
            event = repo.get_event(f"evt-type-{event_type}")
            assert event is not None
            assert event.event_type == event_type

    def test_indexed_token_columns(
        self, repo: SqliteOperationEventRepository
    ):
        """original_tokens and compressed_tokens stored and queryable."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_event(
            event_id="evt-tok1",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=5000,
            compressed_tokens=1200,
            params_json=None,
        )

        event = repo.get_event("evt-tok1")
        assert event is not None
        assert event.original_tokens == 5000
        assert event.compressed_tokens == 1200

    def test_params_json_roundtrip(
        self, repo: SqliteOperationEventRepository
    ):
        """JSON params survive save/load."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        params = {
            "target_tokens": 500,
            "instructions": "Focus on code decisions",
            "nested": {"key": "value"},
        }
        repo.save_event(
            event_id="evt-json1",
            tract_id="test-tract",
            event_type="compress",
            branch_name=None,
            created_at=now,
            original_tokens=100,
            compressed_tokens=50,
            params_json=params,
        )

        event = repo.get_event("evt-json1")
        assert event is not None
        assert event.params_json == params
        assert event.params_json["nested"]["key"] == "value"

    def test_get_event_not_found(self, repo: SqliteOperationEventRepository):
        """get_event() returns None for nonexistent event_id."""
        assert repo.get_event("nonexistent-id") is None

    def test_get_commits_empty(self, repo: SqliteOperationEventRepository):
        """get_commits() returns empty list for nonexistent event_id."""
        assert repo.get_commits("nonexistent-id") == []


# ===========================================================================
# CompileRecordRepository Tests
# ===========================================================================


class TestCompileRecordRepository:
    """Tests for SqliteCompileRecordRepository."""

    @pytest.fixture
    def repo(self, session: Session) -> SqliteCompileRecordRepository:
        return SqliteCompileRecordRepository(session)

    @pytest.fixture
    def setup_commits(self, session: Session) -> list[str]:
        """Create a set of commits for use in compile record tests."""
        hashes = ["commit-a", "commit-b", "commit-c"]
        for h in hashes:
            _make_commit(session, h)
        return hashes

    def test_save_and_get_record(
        self, repo: SqliteCompileRecordRepository, session: Session
    ):
        """save_record() then get_record() returns matching data."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            record_id="rec-001",
            tract_id="test-tract",
            head_hash="abc123",
            token_count=500,
            commit_count=5,
            token_source="tiktoken:cl100k_base",
            params_json={"max_tokens": 4096},
            created_at=now,
        )

        record = repo.get_record("rec-001")
        assert record is not None
        assert record.tract_id == "test-tract"
        assert record.head_hash == "abc123"
        assert record.token_count == 500
        assert record.commit_count == 5
        assert record.token_source == "tiktoken:cl100k_base"
        assert record.params_json == {"max_tokens": 4096}

    def test_add_and_get_effectives(
        self,
        repo: SqliteCompileRecordRepository,
        setup_commits: list[str],
    ):
        """add_effective() x3, get_effectives() returns in position order."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            record_id="rec-eff1",
            tract_id="test-tract",
            head_hash="abc123",
            token_count=300,
            commit_count=3,
            token_source="tiktoken:cl100k_base",
            params_json=None,
            created_at=now,
        )

        # Add in non-sequential order
        repo.add_effective("rec-eff1", "commit-c", position=2)
        repo.add_effective("rec-eff1", "commit-a", position=0)
        repo.add_effective("rec-eff1", "commit-b", position=1)

        effectives = repo.get_effectives("rec-eff1")
        assert len(effectives) == 3
        assert effectives[0].commit_hash == "commit-a"
        assert effectives[1].commit_hash == "commit-b"
        assert effectives[2].commit_hash == "commit-c"

    def test_get_all(
        self, repo: SqliteCompileRecordRepository
    ):
        """get_all() returns records in chronological order."""
        from datetime import timedelta

        now = datetime.now(timezone.utc).replace(tzinfo=None)
        repo.save_record(
            record_id="rec-all2",
            tract_id="test-tract",
            head_hash="hash2",
            token_count=200,
            commit_count=2,
            token_source="tiktoken:cl100k_base",
            params_json=None,
            created_at=now + timedelta(seconds=1),
        )
        repo.save_record(
            record_id="rec-all1",
            tract_id="test-tract",
            head_hash="hash1",
            token_count=100,
            commit_count=1,
            token_source="tiktoken:cl100k_base",
            params_json=None,
            created_at=now,
        )

        records = repo.get_all("test-tract")
        assert len(records) == 2
        assert records[0].record_id == "rec-all1"  # earlier created_at first
        assert records[1].record_id == "rec-all2"

    def test_params_json_roundtrip(
        self, repo: SqliteCompileRecordRepository
    ):
        """JSON params survive save/load."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        params = {"max_tokens": 4096, "branch": "main", "order": "chronological"}
        repo.save_record(
            record_id="rec-json1",
            tract_id="test-tract",
            head_hash="abc123",
            token_count=500,
            commit_count=5,
            token_source="tiktoken:cl100k_base",
            params_json=params,
            created_at=now,
        )

        record = repo.get_record("rec-json1")
        assert record is not None
        assert record.params_json == params

    def test_get_record_not_found(self, repo: SqliteCompileRecordRepository):
        """get_record() returns None for nonexistent record_id."""
        assert repo.get_record("nonexistent-id") is None

    def test_get_effectives_empty(self, repo: SqliteCompileRecordRepository):
        """get_effectives() returns empty list for nonexistent record_id."""
        assert repo.get_effectives("nonexistent-id") == []


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
            source_commits_removed=1,
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
