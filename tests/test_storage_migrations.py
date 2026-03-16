"""Tests for storage engine migration logic and SQLite pragma configuration.

Covers:
- v5 to v6 migration: compression tables -> operation_events/operation_commits
- Full migration chain: every version step from v1 through v12
- Schema evolution: all expected tables and columns exist after init_db
- Idempotent migration: calling init_db twice produces identical results
- Edge cases: empty database, migration with no data to migrate
- set_sqlite_pragma: WAL mode, busy_timeout, synchronous, foreign_keys
- create_trace_engine: memory, file path, and url modes
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.orm import sessionmaker

from tract.storage.engine import (
    _migrate_compressions_v5_to_v6,
    create_session_factory,
    create_trace_engine,
    init_db,
)
from tract.storage.schema import Base, TraceMetaRow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_tables(engine) -> set[str]:
    """Return set of table names in the database."""
    inspector = inspect(engine)
    return set(inspector.get_table_names())


def _get_columns(engine, table_name: str) -> list[str]:
    """Return list of column names for a table."""
    with engine.connect() as conn:
        rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
    return [r[1] for r in rows]


def _get_schema_version(engine) -> str:
    """Read the current schema_version from _trace_meta."""
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        from sqlalchemy import select
        row = session.execute(
            select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        ).scalar_one()
        return row.value


def _create_v5_engine_with_compression_data():
    """Create an in-memory engine at schema v5 with old compression tables and test data.

    Returns the engine ready for migration.
    """
    engine = create_trace_engine(":memory:")
    # Create all current ORM tables first (gives us blobs, commits, refs, etc.)
    Base.metadata.create_all(engine)

    with engine.connect() as conn:
        # Drop v6+ tables that shouldn't exist in v5
        conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
        conn.execute(text("DROP TABLE IF EXISTS operation_events"))
        conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
        conn.execute(text("DROP TABLE IF EXISTS compile_records"))

        # Create old compression tables (v3-v5 schema)
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

        # Insert test blobs and commits for FK satisfaction
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        for blob_id in ["blob-s1", "blob-s2", "blob-s3", "blob-r1", "blob-r2"]:
            conn.execute(text(
                "INSERT INTO blobs (content_hash, payload_json, byte_size, token_count, created_at) "
                "VALUES (:bh, '{\"text\":\"test\"}', 16, 2, :now)"
            ), {"bh": blob_id, "now": now})
        for ch, bh in [
            ("src-1", "blob-s1"), ("src-2", "blob-s2"), ("src-3", "blob-s3"),
            ("res-1", "blob-r1"), ("res-2", "blob-r2"),
        ]:
            conn.execute(text(
                "INSERT INTO commits (commit_hash, tract_id, content_hash, content_type, "
                "operation, token_count, created_at) "
                "VALUES (:ch, 'test-tract', :bh, 'dialogue', 'APPEND', 10, :now)"
            ), {"ch": ch, "bh": bh, "now": now})

        # Insert two compression records with sources and results
        conn.execute(text(
            "INSERT INTO compressions "
            "(compression_id, tract_id, branch_name, created_at, "
            "original_tokens, compressed_tokens, target_tokens, instructions) "
            "VALUES ('comp-A', 'test-tract', 'main', :now, 1000, 200, 250, 'Keep decisions')"
        ), {"now": now})
        conn.execute(text(
            "INSERT INTO compressions "
            "(compression_id, tract_id, branch_name, created_at, "
            "original_tokens, compressed_tokens) "
            "VALUES ('comp-B', 'test-tract', 'feature', :now, 500, 100)"
        ), {"now": now})

        # comp-A: 2 sources, 1 result
        conn.execute(text(
            "INSERT INTO compression_sources VALUES ('comp-A', 'src-1', 0)"
        ))
        conn.execute(text(
            "INSERT INTO compression_sources VALUES ('comp-A', 'src-2', 1)"
        ))
        conn.execute(text(
            "INSERT INTO compression_results VALUES ('comp-A', 'res-1', 0)"
        ))

        # comp-B: 1 source, 1 result
        conn.execute(text(
            "INSERT INTO compression_sources VALUES ('comp-B', 'src-3', 0)"
        ))
        conn.execute(text(
            "INSERT INTO compression_results VALUES ('comp-B', 'res-2', 0)"
        ))

        conn.commit()

    # Set schema version to 5
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    with Session() as session:
        from sqlalchemy import select
        meta = session.execute(
            select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        ).scalar_one_or_none()
        if meta is None:
            session.add(TraceMetaRow(key="schema_version", value="5"))
        else:
            meta.value = "5"
        session.commit()

    return engine


# ===========================================================================
# v5 -> v6 Migration Tests
# ===========================================================================


class TestMigrateCompressionsV5ToV6:
    """Tests for the _migrate_compressions_v5_to_v6 function."""

    def test_data_migrated_to_operation_events(self):
        """Compression rows become operation_events with event_type='compress'."""
        engine = _create_v5_engine_with_compression_data()

        # Manually create the target tables (init_db does this, but we test
        # the migration function in isolation first)
        for table_name in [
            "operation_events", "operation_commits",
            "compile_records", "compile_effectives",
        ]:
            Base.metadata.tables[table_name].create(engine, checkfirst=True)

        _migrate_compressions_v5_to_v6(engine)

        with engine.connect() as conn:
            events = conn.execute(
                text("SELECT * FROM operation_events ORDER BY event_id")
            ).fetchall()
            assert len(events) == 2

            # comp-A: has target_tokens and instructions
            evt_a = [e for e in events if e[0] == "comp-A"][0]
            assert evt_a[1] == "test-tract"       # tract_id
            assert evt_a[2] == "compress"          # event_type
            assert evt_a[3] == "main"              # branch_name
            assert evt_a[5] == 1000                # original_tokens
            assert evt_a[6] == 200                 # compressed_tokens
            params_a = json.loads(evt_a[7]) if evt_a[7] else {}
            assert params_a["target_tokens"] == 250
            assert params_a["instructions"] == "Keep decisions"

            # comp-B: no target_tokens, no instructions -> params_json is null
            evt_b = [e for e in events if e[0] == "comp-B"][0]
            assert evt_b[1] == "test-tract"
            assert evt_b[3] == "feature"
            assert evt_b[5] == 500
            assert evt_b[6] == 100
            # No params since target_tokens and instructions are both absent
            assert evt_b[7] is None

        engine.dispose()

    def test_sources_migrated_to_operation_commits(self):
        """compression_sources become operation_commits with role='source'."""
        engine = _create_v5_engine_with_compression_data()
        for table_name in ["operation_events", "operation_commits",
                           "compile_records", "compile_effectives"]:
            Base.metadata.tables[table_name].create(engine, checkfirst=True)

        _migrate_compressions_v5_to_v6(engine)

        with engine.connect() as conn:
            sources = conn.execute(text(
                "SELECT event_id, commit_hash, role, position "
                "FROM operation_commits WHERE role='source' ORDER BY event_id, position"
            )).fetchall()
            assert len(sources) == 3
            # comp-A sources
            assert sources[0] == ("comp-A", "src-1", "source", 0)
            assert sources[1] == ("comp-A", "src-2", "source", 1)
            # comp-B source
            assert sources[2] == ("comp-B", "src-3", "source", 0)

        engine.dispose()

    def test_results_migrated_to_operation_commits(self):
        """compression_results become operation_commits with role='result'."""
        engine = _create_v5_engine_with_compression_data()
        for table_name in ["operation_events", "operation_commits",
                           "compile_records", "compile_effectives"]:
            Base.metadata.tables[table_name].create(engine, checkfirst=True)

        _migrate_compressions_v5_to_v6(engine)

        with engine.connect() as conn:
            results = conn.execute(text(
                "SELECT event_id, commit_hash, role, position "
                "FROM operation_commits WHERE role='result' ORDER BY event_id, position"
            )).fetchall()
            assert len(results) == 2
            assert results[0] == ("comp-A", "res-1", "result", 0)
            assert results[1] == ("comp-B", "res-2", "result", 0)

        engine.dispose()

    def test_no_compressions_table_is_noop(self):
        """If compressions table doesn't exist, function returns silently."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)
        # No old tables exist -- should not raise
        _migrate_compressions_v5_to_v6(engine)
        engine.dispose()

    def test_empty_compressions_table(self):
        """If compressions table exists but is empty, no rows are migrated."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
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
                    compression_id VARCHAR(64) NOT NULL,
                    commit_hash VARCHAR(64) NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (compression_id, commit_hash)
                )
            """))
            conn.execute(text("""
                CREATE TABLE compression_results (
                    compression_id VARCHAR(64) NOT NULL,
                    commit_hash VARCHAR(64) NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (compression_id, commit_hash)
                )
            """))
            conn.commit()

        _migrate_compressions_v5_to_v6(engine)

        with engine.connect() as conn:
            events = conn.execute(text("SELECT * FROM operation_events")).fetchall()
            assert events == []
            commits = conn.execute(text("SELECT * FROM operation_commits")).fetchall()
            assert commits == []

        engine.dispose()

    def test_compression_without_optional_fields(self):
        """Compression with no target_tokens/instructions gets null params_json."""
        engine = _create_v5_engine_with_compression_data()
        for table_name in ["operation_events", "operation_commits",
                           "compile_records", "compile_effectives"]:
            Base.metadata.tables[table_name].create(engine, checkfirst=True)

        _migrate_compressions_v5_to_v6(engine)

        with engine.connect() as conn:
            # comp-B had no target_tokens or instructions
            evt = conn.execute(text(
                "SELECT params_json FROM operation_events WHERE event_id='comp-B'"
            )).fetchone()
            assert evt[0] is None

        engine.dispose()


# ===========================================================================
# Full Migration Chain Tests
# ===========================================================================


class TestFullMigrationChain:
    """Tests for init_db migration from various starting schema versions."""

    def _set_version(self, engine, version: str):
        """Set schema_version in _trace_meta."""
        Session = sessionmaker(bind=engine, expire_on_commit=False)
        with Session() as session:
            from sqlalchemy import select
            meta = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one_or_none()
            if meta is None:
                session.add(TraceMetaRow(key="schema_version", value=version))
            else:
                meta.value = version
            session.commit()

    def test_fresh_database_gets_v12(self):
        """A brand-new database gets schema version 12."""
        engine = create_trace_engine(":memory:")
        init_db(engine)
        assert _get_schema_version(engine) == "13"
        engine.dispose()

    def test_v1_migrates_to_v12(self):
        """Starting from v1, init_db runs the full chain to v12."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        # Simulate v1: drop commit_parents and everything added after v1
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS config_change_log"))
            conn.execute(text("DROP TABLE IF EXISTS operation_configs"))
            conn.execute(text("DROP TABLE IF EXISTS tag_annotations"))
            conn.execute(text("DROP TABLE IF EXISTS tag_registry"))
            conn.execute(text("DROP TABLE IF EXISTS commit_tools"))
            conn.execute(text("DROP TABLE IF EXISTS tool_definitions"))
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))
            conn.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            conn.execute(text("DROP TABLE IF EXISTS commit_parents"))
            conn.commit()
        self._set_version(engine, "1")

        init_db(engine)

        assert _get_schema_version(engine) == "13"
        tables = _get_tables(engine)
        assert "commit_parents" in tables
        assert "spawn_pointers" in tables
        assert "operation_events" in tables
        assert "config_change_log" in tables
        engine.dispose()

    def test_v6_migrates_to_v12(self):
        """Starting from v6, init_db adds retention_json, tool tables, etc."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        # Remove v7+ columns/tables to simulate v6
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS config_change_log"))
            conn.execute(text("DROP TABLE IF EXISTS operation_configs"))
            conn.execute(text("DROP TABLE IF EXISTS tag_annotations"))
            conn.execute(text("DROP TABLE IF EXISTS tag_registry"))
            conn.execute(text("DROP TABLE IF EXISTS commit_tools"))
            conn.execute(text("DROP TABLE IF EXISTS tool_definitions"))
            conn.commit()
        self._set_version(engine, "6")

        init_db(engine)

        assert _get_schema_version(engine) == "13"
        # v7: retention_json on annotations
        assert "retention_json" in _get_columns(engine, "annotations")
        # v8: tool tables
        assert "tool_definitions" in _get_tables(engine)
        assert "commit_tools" in _get_tables(engine)
        # v9: instruction columns on operation_events
        oe_cols = _get_columns(engine, "operation_events")
        assert "original_instructions" in oe_cols
        assert "effective_instructions" in oe_cols
        # v10: tags
        assert "tags_json" in _get_columns(engine, "commits")
        assert "tag_annotations" in _get_tables(engine)
        assert "tag_registry" in _get_tables(engine)
        # v11: persistence tables
        assert "operation_configs" in _get_tables(engine)
        # v12: config_change_log
        assert "config_change_log" in _get_tables(engine)
        engine.dispose()

    def test_v9_migrates_to_v12(self):
        """Starting from v9, init_db adds tags, persistence, and config provenance."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS config_change_log"))
            conn.execute(text("DROP TABLE IF EXISTS operation_configs"))
            conn.execute(text("DROP TABLE IF EXISTS tag_annotations"))
            conn.execute(text("DROP TABLE IF EXISTS tag_registry"))
            conn.commit()
        self._set_version(engine, "9")

        init_db(engine)

        assert _get_schema_version(engine) == "13"
        assert "tags_json" in _get_columns(engine, "commits")
        assert "tag_annotations" in _get_tables(engine)
        assert "operation_configs" in _get_tables(engine)
        assert "config_change_log" in _get_tables(engine)
        engine.dispose()

    def test_v11_migrates_to_v12(self):
        """Starting from v11, init_db adds config_change_log."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS config_change_log"))
            conn.commit()
        self._set_version(engine, "11")

        init_db(engine)

        assert _get_schema_version(engine) == "13"
        assert "config_change_log" in _get_tables(engine)
        engine.dispose()


# ===========================================================================
# Schema Evolution Tests
# ===========================================================================


class TestSchemaEvolution:
    """Verify that after init_db, all expected tables and columns exist."""

    ALL_EXPECTED_TABLES = {
        "blobs", "commits", "refs", "annotations", "commit_parents",
        "_trace_meta", "operation_events", "operation_commits",
        "compile_records", "compile_effectives", "spawn_pointers",
        "tool_definitions", "commit_tools", "tag_annotations",
        "tag_registry", "operation_configs", "config_change_log",
        "behavioral_specs",
    }

    @pytest.fixture
    def fresh_engine(self):
        engine = create_trace_engine(":memory:")
        init_db(engine)
        yield engine
        engine.dispose()

    def test_all_orm_tables_exist(self, fresh_engine):
        """Every ORM-defined table must be present."""
        tables = _get_tables(fresh_engine)
        missing = self.ALL_EXPECTED_TABLES - tables
        assert not missing, f"Missing tables: {missing}"

    def test_commits_table_columns(self, fresh_engine):
        """commits table has all expected columns including tags_json."""
        cols = _get_columns(fresh_engine, "commits")
        for expected in [
            "commit_hash", "tract_id", "parent_hash", "content_hash",
            "content_type", "operation", "edit_target", "message",
            "token_count", "metadata_json", "generation_config_json",
            "tags_json", "created_at",
        ]:
            assert expected in cols, f"Missing column: {expected}"

    def test_annotations_has_retention_json(self, fresh_engine):
        """annotations table includes the v7 retention_json column."""
        cols = _get_columns(fresh_engine, "annotations")
        assert "retention_json" in cols

    def test_operation_events_has_instruction_columns(self, fresh_engine):
        """operation_events table includes the v9 instruction columns."""
        cols = _get_columns(fresh_engine, "operation_events")
        assert "original_instructions" in cols
        assert "effective_instructions" in cols

    def test_operation_events_all_columns(self, fresh_engine):
        """operation_events has every expected column."""
        cols = _get_columns(fresh_engine, "operation_events")
        for expected in [
            "event_id", "tract_id", "event_type", "branch_name", "created_at",
            "original_tokens", "compressed_tokens", "params_json",
            "original_instructions", "effective_instructions",
        ]:
            assert expected in cols, f"Missing column: {expected}"

    def test_config_change_log_columns(self, fresh_engine):
        """config_change_log (v12) has all expected columns."""
        cols = _get_columns(fresh_engine, "config_change_log")
        for expected in [
            "id", "tract_id", "change_type", "change_key",
            "config_json", "previous_json", "source", "created_at",
        ]:
            assert expected in cols, f"Missing column: {expected}"

    def test_key_indexes_exist(self, fresh_engine):
        """Critical indexes are present on high-traffic tables."""
        inspector = inspect(fresh_engine)

        commit_idx_names = {
            idx["name"] for idx in inspector.get_indexes("commits")
        }
        assert "ix_commits_tract_time" in commit_idx_names
        assert "ix_commits_tract_type" in commit_idx_names

        oe_idx_names = {
            idx["name"] for idx in inspector.get_indexes("operation_events")
        }
        assert "ix_operation_events_tract_type" in oe_idx_names

        ccl_idx_names = {
            idx["name"] for idx in inspector.get_indexes("config_change_log")
        }
        assert "ix_config_change_log_tract_time" in ccl_idx_names


# ===========================================================================
# Idempotent Migration Tests
# ===========================================================================


class TestIdempotentMigration:
    """Running init_db multiple times must not corrupt the database."""

    def test_double_init_db_on_fresh_database(self):
        """Calling init_db twice on fresh DB keeps version at 12."""
        engine = create_trace_engine(":memory:")
        init_db(engine)
        init_db(engine)
        assert _get_schema_version(engine) == "13"
        engine.dispose()

    def test_double_init_db_preserves_tables(self):
        """Second init_db does not duplicate or remove tables."""
        engine = create_trace_engine(":memory:")
        init_db(engine)
        tables_after_first = _get_tables(engine)
        init_db(engine)
        tables_after_second = _get_tables(engine)
        assert tables_after_first == tables_after_second
        engine.dispose()

    def test_double_init_db_preserves_data(self):
        """Data inserted between two init_db calls is preserved."""
        engine = create_trace_engine(":memory:")
        init_db(engine)

        # Insert some data
        with engine.connect() as conn:
            now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
            conn.execute(text(
                "INSERT INTO blobs (content_hash, payload_json, byte_size, token_count, created_at) "
                "VALUES ('test-blob-hash', '{\"text\":\"hello\"}', 16, 2, :now)"
            ), {"now": now})
            conn.commit()

        init_db(engine)

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT content_hash FROM blobs WHERE content_hash='test-blob-hash'")
            ).fetchone()
            assert row is not None
            assert row[0] == "test-blob-hash"

        engine.dispose()

    def test_double_init_db_after_v5_migration(self):
        """init_db twice on a v5 database: first migrates, second is a no-op."""
        engine = _create_v5_engine_with_compression_data()

        init_db(engine)
        assert _get_schema_version(engine) == "13"
        tables_first = _get_tables(engine)

        # Old tables should be gone
        assert "compressions" not in tables_first

        # Second call should not raise
        init_db(engine)
        assert _get_schema_version(engine) == "13"
        tables_second = _get_tables(engine)
        assert tables_first == tables_second

        # Migrated data should still be there
        with engine.connect() as conn:
            events = conn.execute(text("SELECT * FROM operation_events")).fetchall()
            assert len(events) == 2

        engine.dispose()

    def test_v7_retention_json_column_idempotent(self):
        """Adding retention_json column a second time does not fail."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        # Simulate v6 where retention_json doesn't exist yet
        # (Base.metadata.create_all already adds it, so we verify the
        # migration code checks for existence before ALTER TABLE)
        Session = sessionmaker(bind=engine, expire_on_commit=False)
        with Session() as session:
            session.add(TraceMetaRow(key="schema_version", value="6"))
            session.commit()

        # First migration adds the column (or sees it already exists)
        init_db(engine)
        assert _get_schema_version(engine) == "13"

        # Reset to v6 and try again -- the column already exists
        with Session() as session:
            from sqlalchemy import select
            meta = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            meta.value = "6"
            session.commit()

        # Should not raise "duplicate column" error
        init_db(engine)
        assert _get_schema_version(engine) == "13"
        assert "retention_json" in _get_columns(engine, "annotations")
        engine.dispose()


# ===========================================================================
# Empty Database Edge Cases
# ===========================================================================


class TestEmptyDatabaseEdgeCases:
    """Migration on databases with no user data."""

    def test_init_db_on_completely_empty_database(self):
        """init_db on a raw engine with no tables at all."""
        engine = create_trace_engine(":memory:")
        # No tables exist yet
        assert _get_tables(engine) == set()

        init_db(engine)

        assert _get_schema_version(engine) == "13"
        tables = _get_tables(engine)
        assert "blobs" in tables
        assert "commits" in tables
        assert "_trace_meta" in tables
        engine.dispose()

    def test_migrate_v5_with_no_compression_data(self):
        """v5 migration with empty compression tables doesn't error."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            # Drop v6+ tables
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))

            # Create empty old compression tables
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
                    compression_id VARCHAR(64) NOT NULL,
                    commit_hash VARCHAR(64) NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (compression_id, commit_hash)
                )
            """))
            conn.execute(text("""
                CREATE TABLE compression_results (
                    compression_id VARCHAR(64) NOT NULL,
                    commit_hash VARCHAR(64) NOT NULL,
                    position INTEGER NOT NULL,
                    PRIMARY KEY (compression_id, commit_hash)
                )
            """))
            conn.commit()

        Session = sessionmaker(bind=engine, expire_on_commit=False)
        with Session() as session:
            from sqlalchemy import select
            meta = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one_or_none()
            if meta:
                meta.value = "5"
            else:
                session.add(TraceMetaRow(key="schema_version", value="5"))
            session.commit()

        init_db(engine)

        assert _get_schema_version(engine) == "13"
        # Old tables should be dropped
        tables = _get_tables(engine)
        assert "compressions" not in tables
        # New tables should exist
        assert "operation_events" in tables
        engine.dispose()

    def test_migrate_v5_no_source_or_result_tables(self):
        """v5 migration when only compressions table exists (no sources/results)."""
        engine = create_trace_engine(":memory:")
        Base.metadata.create_all(engine)

        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))

            # Only create the compressions table, not the child tables
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
            conn.commit()

        Session = sessionmaker(bind=engine, expire_on_commit=False)
        with Session() as session:
            from sqlalchemy import select
            meta = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one_or_none()
            if meta:
                meta.value = "5"
            else:
                session.add(TraceMetaRow(key="schema_version", value="5"))
            session.commit()

        # Should not crash despite missing compression_sources/compression_results
        init_db(engine)

        assert _get_schema_version(engine) == "13"
        engine.dispose()


# ===========================================================================
# SQLite Pragma Tests
# ===========================================================================


class TestSqlitePragma:
    """Tests for set_sqlite_pragma behavior on engine connections."""

    def test_wal_journal_mode(self):
        """SQLite engine uses WAL journal mode."""
        engine = create_trace_engine(":memory:")
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            # In-memory databases may report "memory" instead of "wal"
            # since WAL is not supported for :memory:, but the pragma is still issued.
            # For file-based databases it would be "wal".
            assert result[0] in ("wal", "memory")
        engine.dispose()

    def test_wal_journal_mode_file_based(self, tmp_path):
        """File-based SQLite engine uses WAL journal mode."""
        db_path = str(tmp_path / "test_pragma.db")
        engine = create_trace_engine(db_path)
        init_db(engine)
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA journal_mode")).fetchone()
            assert result[0] == "wal"
        engine.dispose()

    def test_busy_timeout(self):
        """SQLite engine has busy_timeout set to 5000ms."""
        engine = create_trace_engine(":memory:")
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA busy_timeout")).fetchone()
            assert result[0] == 5000
        engine.dispose()

    def test_synchronous_normal(self):
        """SQLite engine uses NORMAL synchronous mode (value 1)."""
        engine = create_trace_engine(":memory:")
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA synchronous")).fetchone()
            # NORMAL = 1
            assert result[0] == 1
        engine.dispose()

    def test_foreign_keys_enabled(self):
        """SQLite engine has foreign keys enforcement enabled."""
        engine = create_trace_engine(":memory:")
        with engine.connect() as conn:
            result = conn.execute(text("PRAGMA foreign_keys")).fetchone()
            assert result[0] == 1
        engine.dispose()

    def test_pragmas_applied_on_every_connection(self):
        """Pragmas are reapplied on each new connection from the pool."""
        engine = create_trace_engine(":memory:")
        # Get two separate connections and check both have pragmas
        with engine.connect() as conn1:
            fk1 = conn1.execute(text("PRAGMA foreign_keys")).fetchone()[0]
            assert fk1 == 1
        with engine.connect() as conn2:
            fk2 = conn2.execute(text("PRAGMA foreign_keys")).fetchone()[0]
            assert fk2 == 1
        engine.dispose()


# ===========================================================================
# Engine Creation Tests
# ===========================================================================


class TestCreateTraceEngine:
    """Tests for create_trace_engine factory function."""

    def test_memory_engine(self):
        """Default :memory: creates a usable SQLite engine."""
        engine = create_trace_engine(":memory:")
        assert engine.dialect.name == "sqlite"
        engine.dispose()

    def test_file_engine(self, tmp_path):
        """File path creates a persistent SQLite database."""
        db_path = str(tmp_path / "test.db")
        engine = create_trace_engine(db_path)
        init_db(engine)
        assert engine.dialect.name == "sqlite"
        assert Path(db_path).exists()
        engine.dispose()

    def test_url_engine(self):
        """Explicit sqlite:// URL creates a working engine."""
        engine = create_trace_engine(url="sqlite://")
        assert engine.dialect.name == "sqlite"
        engine.dispose()

    def test_url_overrides_db_path(self):
        """When url is provided, db_path is ignored."""
        engine = create_trace_engine(
            db_path="/nonexistent/path.db",
            url="sqlite://",
        )
        # Should succeed because url takes precedence
        assert engine.dialect.name == "sqlite"
        engine.dispose()

    def test_session_factory_creation(self):
        """create_session_factory returns a usable sessionmaker."""
        engine = create_trace_engine(":memory:")
        init_db(engine)
        factory = create_session_factory(engine)
        session = factory()
        assert isinstance(session, type(factory()))
        session.close()
        engine.dispose()

    def test_session_factory_expire_on_commit_false(self):
        """Sessions from factory have expire_on_commit=False."""
        engine = create_trace_engine(":memory:")
        factory = create_session_factory(engine)
        session = factory()
        assert session.expire_on_commit is False
        session.close()
        engine.dispose()
