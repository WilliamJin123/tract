"""Tests for file-based persistence (.tract/ directory).

Covers:
- Schema migration v10->v11
- save_workflow writes file
- In-memory database -> no file operations
- Directory structure
- Persistence repository CRUD
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract import Tract


# ============================================================
# Helpers
# ============================================================


def _make_file_tract(tmp_path: Path, **kwargs) -> Tract:
    """Create a Tract backed by a file DB in tmp_path."""
    db_path = str(tmp_path / "test.db")
    return Tract.open(path=db_path, **kwargs)


# ============================================================
# Schema migration v10->v11
# ============================================================


class TestSchemaMigration:
    """Test that v10->v11 migration creates persistence tables."""

    def test_new_db_is_v12(self, tmp_path: Path) -> None:
        """A fresh database should be at schema version 12."""
        from sqlalchemy import select

        from tract.storage.schema import TraceMetaRow

        t = _make_file_tract(tmp_path)
        stmt = select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        row = t._session.execute(stmt).scalar_one()
        assert row.value == "13"
        t.close()

    def test_persistence_tables_exist(self, tmp_path: Path) -> None:
        """New tables should exist in the database."""
        from sqlalchemy import text

        t = _make_file_tract(tmp_path)
        with t._engine.connect() as conn:
            tables = [
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]
        assert "operation_configs" in tables
        t.close()

    def test_migration_from_v10(self, tmp_path: Path) -> None:
        """Simulate a v10 DB and verify it migrates to v11."""
        from sqlalchemy import text

        from tract.storage.engine import create_trace_engine, init_db

        db_path = str(tmp_path / "migrate.db")
        engine = create_trace_engine(db_path)

        # First init creates v11
        init_db(engine)

        # Manually set version back to 10 and drop the new tables
        with engine.connect() as conn:
            conn.execute(text("UPDATE _trace_meta SET value='10' WHERE key='schema_version'"))
            conn.execute(text("DROP TABLE IF EXISTS hook_wirings"))
            conn.execute(text("DROP TABLE IF EXISTS dynamic_op_specs"))
            conn.execute(text("DROP TABLE IF EXISTS operation_configs"))
            conn.commit()

        # Re-init should migrate v10->v11->v12
        init_db(engine)

        with engine.connect() as conn:
            version = conn.execute(
                text("SELECT value FROM _trace_meta WHERE key='schema_version'")
            ).scalar_one()
            tables = [
                r[0]
                for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]

        assert version == "13"
        assert "operation_configs" in tables
        assert "config_change_log" in tables
        assert "behavioral_specs" in tables

        engine.dispose()


# ============================================================
# tract_dir property
# ============================================================


class TestTractDir:
    """Test the tract_dir property."""

    def test_tract_dir_for_file_db(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        assert t.tract_dir is not None
        assert t.tract_dir == tmp_path / ".tract"
        t.close()

    def test_tract_dir_for_memory_db(self) -> None:
        t = Tract.open()
        assert t.tract_dir is None
        t.close()


# ============================================================
# save_workflow
# ============================================================


class TestSaveWorkflow:
    """Test save_workflow writes file."""

    def test_save_workflow_writes_file(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def run():\n    print("workflow")\n'
        path = t.save_workflow("my_workflow", code)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == code
        assert path == tmp_path / ".tract" / "workflows" / "my_workflow.py"
        t.close()

    def test_save_workflow_validates_syntax(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        bad_code = "def run(:\n"
        with pytest.raises(SyntaxError):
            t.save_workflow("bad_workflow", bad_code)
        t.close()


# ============================================================
# In-memory database
# ============================================================


class TestInMemory:
    """Test that in-memory databases handle persistence gracefully."""

    def test_memory_db_has_no_tract_dir(self) -> None:
        t = Tract.open()
        assert t.tract_dir is None
        t.close()

    def test_memory_db_quarantined_empty(self) -> None:
        t = Tract.open()
        assert t.quarantined == []
        t.close()


# ============================================================
# Repository CRUD
# ============================================================


class TestPersistenceRepository:
    """Test SqlitePersistenceRepository directly."""

    def test_operation_config_crud(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        repo = t._persistence_repo
        assert repo is not None

        from tract.storage.schema import OperationConfigRow

        config_row = OperationConfigRow(
            tract_id=t.tract_id,
            config_key="llm_config",
            config_json='{"model":"gpt-4"}',
            created_at=__import__("datetime").datetime.now(),
        )
        repo.save_operation_config(config_row)
        t._session.commit()

        configs = repo.get_operation_configs(t.tract_id)
        assert len(configs) == 1
        assert configs[0].config_key == "llm_config"

        # Get by key
        found = repo.get_operation_config(t.tract_id, "llm_config")
        assert found is not None
        assert json.loads(found.config_json) == {"model": "gpt-4"}

        # Upsert
        config_row2 = OperationConfigRow(
            tract_id=t.tract_id,
            config_key="llm_config",
            config_json='{"model":"gpt-4o"}',
            created_at=__import__("datetime").datetime.now(),
        )
        repo.save_operation_config(config_row2)
        t._session.commit()

        # Should still be 1 row (upserted)
        configs = repo.get_operation_configs(t.tract_id)
        assert len(configs) == 1
        assert json.loads(configs[0].config_json) == {"model": "gpt-4o"}

        # Delete
        assert repo.delete_operation_config(t.tract_id, "llm_config") is True
        t._session.commit()
        assert len(repo.get_operation_configs(t.tract_id)) == 0

        t.close()


# ============================================================
# Directory structure
# ============================================================


class TestDirectoryStructure:
    """Test .tract/ directory is created lazily with correct structure."""

    def test_no_tract_dir_on_open(self, tmp_path: Path) -> None:
        """Opening a tract should NOT create .tract/ directory."""
        t = _make_file_tract(tmp_path)
        assert not (tmp_path / ".tract").exists()
        t.close()

    def test_tract_dir_created_on_save_workflow(self, tmp_path: Path) -> None:
        """save_workflow creates .tract/workflows/ lazily."""
        t = _make_file_tract(tmp_path)
        code = 'def run():\n    pass\n'
        t.save_workflow("my_workflow", code)
        assert (tmp_path / ".tract" / "workflows").is_dir()
        t.close()
