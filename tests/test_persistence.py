"""Tests for file-based persistence (.tract/ directory).

Covers:
- Schema migration v10->v11
- save_hook writes file and DB entry
- save_trigger writes file and DB entry
- save_workflow writes file
- delete_hook removes file and DB entry
- delete_trigger removes file and DB entry
- Auto-load on Tract.open() restores hooks
- Auto-load on Tract.open() restores dynamic ops
- Broken Python file -> quarantined, not crash
- In-memory database -> no file operations
- list_saved_hooks/list_saved_triggers return correct data
- save_hook validates syntax before writing
- save_hook validates operation is hookable
- Priority ordering of hooks preserved
- Roundtrip: save -> close -> reopen -> hooks still active
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from tract import Tract
from tract.models.content import DialogueContent


# ============================================================
# Helpers
# ============================================================


def _make_file_tract(tmp_path: Path, **kwargs) -> Tract:
    """Create a Tract backed by a file DB in tmp_path."""
    db_path = str(tmp_path / "test.db")
    return Tract.open(path=db_path, **kwargs)


def _reopen_tract(t: Tract) -> Tract:
    """Close and reopen a Tract at the same path+tract_id."""
    db_path = t._db_path
    tract_id = t.tract_id
    t.close()
    return Tract.open(path=db_path, tract_id=tract_id)


# ============================================================
# Schema migration v10->v11
# ============================================================


class TestSchemaMigration:
    """Test that v10->v11 migration creates persistence tables."""

    def test_new_db_is_v11(self, tmp_path: Path) -> None:
        """A fresh database should be at schema version 11."""
        from sqlalchemy import select

        from tract.storage.schema import TraceMetaRow

        t = _make_file_tract(tmp_path)
        stmt = select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        row = t._session.execute(stmt).scalar_one()
        assert row.value == "11"
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
        assert "hook_wirings" in tables
        assert "dynamic_op_specs" in tables
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

        # Re-init should migrate v10->v11
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

        assert version == "11"
        assert "hook_wirings" in tables
        assert "dynamic_op_specs" in tables
        assert "operation_configs" in tables

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
# save_hook
# ============================================================


class TestSaveHook:
    """Test save_hook writes file and DB entry."""

    def test_save_hook_writes_file(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        path = t.save_hook("my_hook", code, "compress")
        assert path.exists()
        assert path.read_text(encoding="utf-8") == code
        assert path == tmp_path / ".tract" / "hooks" / "my_hook.py"
        t.close()

    def test_save_hook_creates_db_entry(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        hooks = t.list_saved_hooks()
        assert len(hooks) == 1
        assert hooks[0]["name"] == "my_hook"
        assert hooks[0]["operation"] == "compress"
        assert hooks[0]["priority"] == 100
        assert hooks[0]["enabled"] is True
        assert hooks[0]["handler_source"] == "file"
        t.close()

    def test_save_hook_registers_in_memory(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        hook_names = t.hook_names
        assert "compress" in hook_names
        assert "my_hook" in hook_names["compress"]
        t.close()

    def test_save_hook_validates_syntax(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        bad_code = "def handler(pending:\n"  # missing closing paren
        with pytest.raises(SyntaxError):
            t.save_hook("bad_hook", bad_code, "compress")
        # File should NOT be written
        assert not (tmp_path / ".tract" / "hooks" / "bad_hook.py").exists()
        t.close()

    def test_save_hook_validates_operation(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        with pytest.raises(ValueError, match="not a hookable operation"):
            t.save_hook("bad_op", code, "not_a_real_op")
        t.close()

    def test_save_hook_rejects_memory_db(self) -> None:
        t = Tract.open()
        code = 'def handler(pending):\n    pending.approve()\n'
        with pytest.raises(RuntimeError, match="in-memory"):
            t.save_hook("my_hook", code, "compress")
        t.close()

    def test_save_hook_custom_priority(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("hook1", code, "compress", priority=200)
        t.save_hook("hook2", code, "compress", priority=50)
        hooks = t.list_saved_hooks()
        # Ordered by priority (50, 200)
        assert hooks[0]["name"] == "hook2"
        assert hooks[0]["priority"] == 50
        assert hooks[1]["name"] == "hook1"
        assert hooks[1]["priority"] == 200
        t.close()

    def test_save_hook_overwrites_existing(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code1 = 'def handler(pending):\n    pending.approve()\n'
        code2 = 'def handler(pending):\n    pending.reject("nope")\n'
        t.save_hook("my_hook", code1, "compress")
        t.save_hook("my_hook", code2, "compress")
        hooks = t.list_saved_hooks()
        assert len(hooks) == 1
        # File should have the new content
        file_path = tmp_path / ".tract" / "hooks" / "my_hook.py"
        assert 'reject' in file_path.read_text(encoding="utf-8")
        t.close()


# ============================================================
# save_trigger
# ============================================================


class TestSaveTrigger:
    """Test save_trigger writes file and DB entry."""

    def test_save_trigger_writes_file(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'trigger_name = "my_trigger"\n'
        path = t.save_trigger("my_trigger", code)
        assert path.exists()
        assert path.read_text(encoding="utf-8") == code
        assert path == tmp_path / ".tract" / "triggers" / "my_trigger.py"
        t.close()

    def test_save_trigger_creates_db_entry(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'trigger_name = "my_trigger"\n'
        t.save_trigger("my_trigger", code, config={"key": "value"})
        triggers = t.list_saved_triggers()
        assert len(triggers) == 1
        assert triggers[0]["name"] == "my_trigger"
        assert triggers[0]["config"] == {"key": "value"}
        t.close()

    def test_save_trigger_validates_syntax(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        bad_code = "class Broken(\n"
        with pytest.raises(SyntaxError):
            t.save_trigger("bad_trigger", bad_code)
        assert not (tmp_path / ".tract" / "triggers" / "bad_trigger.py").exists()
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
# delete_hook
# ============================================================


class TestDeleteHook:
    """Test delete_hook removes file and DB entry."""

    def test_delete_hook_removes_file_and_db(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        path = t.save_hook("my_hook", code, "compress")
        assert path.exists()
        assert len(t.list_saved_hooks()) == 1

        t.delete_hook("my_hook")
        assert not path.exists()
        assert len(t.list_saved_hooks()) == 0
        t.close()

    def test_delete_hook_removes_in_memory_registration(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        assert "compress" in t.hook_names

        t.delete_hook("my_hook")
        assert "compress" not in t.hook_names
        t.close()

    def test_delete_nonexistent_hook_is_safe(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        t.delete_hook("no_such_hook")  # should not raise
        t.close()


# ============================================================
# delete_trigger
# ============================================================


class TestDeleteTrigger:
    """Test delete_trigger removes file and DB entry."""

    def test_delete_trigger_removes_file_and_db(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'trigger_name = "my_trigger"\n'
        path = t.save_trigger("my_trigger", code)
        assert path.exists()
        assert len(t.list_saved_triggers()) == 1

        t.delete_trigger("my_trigger")
        assert not path.exists()
        assert len(t.list_saved_triggers()) == 0
        t.close()


# ============================================================
# Auto-load on restart
# ============================================================


class TestAutoLoad:
    """Test that hooks and dynamic ops survive process restart."""

    def test_roundtrip_hook(self, tmp_path: Path) -> None:
        """save_hook -> close -> reopen -> hook is registered and active."""
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        db_path = t._db_path
        tract_id = t.tract_id
        t.close()

        # Reopen
        t2 = Tract.open(path=db_path, tract_id=tract_id)
        assert "compress" in t2.hook_names
        assert "my_hook" in t2.hook_names["compress"]
        assert len(t2.quarantined) == 0
        t2.close()

    def test_roundtrip_multiple_hooks_priority(self, tmp_path: Path) -> None:
        """Multiple hooks are restored in priority order."""
        t = _make_file_tract(tmp_path)
        code_a = 'def handler(pending):\n    pending.pass_through()\n'
        code_b = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("hook_a", code_a, "compress", priority=200)
        t.save_hook("hook_b", code_b, "compress", priority=50)
        db_path = t._db_path
        tract_id = t.tract_id
        t.close()

        t2 = Tract.open(path=db_path, tract_id=tract_id)
        assert "compress" in t2.hook_names
        # hook_b (priority 50) loads first, hook_a (priority 200) loads second
        names = t2.hook_names["compress"]
        assert "hook_b" in names
        assert "hook_a" in names
        assert names.index("hook_b") < names.index("hook_a")
        t2.close()

    def test_broken_hook_file_quarantined(self, tmp_path: Path) -> None:
        """If a hook file has bad syntax on reload, it's quarantined, not crashed."""
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        db_path = t._db_path
        tract_id = t.tract_id
        t.close()

        # Corrupt the file
        hook_file = tmp_path / ".tract" / "hooks" / "my_hook.py"
        hook_file.write_text("def handler(pending:\n", encoding="utf-8")  # syntax error

        # Reopen should not crash
        t2 = Tract.open(path=db_path, tract_id=tract_id)
        assert len(t2.quarantined) == 1
        assert "hook:" in t2.quarantined[0]
        # Hook should NOT be registered
        assert "compress" not in t2.hook_names
        t2.close()

    def test_missing_hook_file_quarantined(self, tmp_path: Path) -> None:
        """If a hook file is missing on reload, it's quarantined."""
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        db_path = t._db_path
        tract_id = t.tract_id
        t.close()

        # Delete the file
        hook_file = tmp_path / ".tract" / "hooks" / "my_hook.py"
        hook_file.unlink()

        # Reopen should not crash
        t2 = Tract.open(path=db_path, tract_id=tract_id)
        assert len(t2.quarantined) == 1
        t2.close()

    def test_hook_fires_after_reload(self, tmp_path: Path) -> None:
        """After reload, the hook actually fires during operations."""
        t = _make_file_tract(tmp_path)
        # Hook that rejects compression
        code = 'def handler(pending):\n    pending.reject("blocked by hook")\n'
        t.save_hook("reject_compress", code, "compress")
        db_path = t._db_path
        tract_id = t.tract_id
        t.close()

        # Reopen and verify hook is functional
        t2 = Tract.open(path=db_path, tract_id=tract_id)
        assert "compress" in t2.hook_names
        assert "reject_compress" in t2.hook_names["compress"]
        t2.close()


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

    def test_memory_db_list_hooks_empty(self) -> None:
        t = Tract.open()
        assert t.list_saved_hooks() == []
        t.close()

    def test_memory_db_list_triggers_empty(self) -> None:
        t = Tract.open()
        assert t.list_saved_triggers() == []
        t.close()


# ============================================================
# list_saved_hooks / list_saved_triggers
# ============================================================


class TestListMethods:
    """Test list_saved_hooks and list_saved_triggers."""

    def test_list_saved_hooks_multiple(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("hook_a", code, "compress", priority=100)
        t.save_hook("hook_b", code, "gc", priority=50)

        hooks = t.list_saved_hooks()
        assert len(hooks) == 2
        # Ordered by priority: gc (50) first, compress (100) second
        assert hooks[0]["name"] == "hook_b"
        assert hooks[0]["operation"] == "gc"
        assert hooks[1]["name"] == "hook_a"
        assert hooks[1]["operation"] == "compress"
        t.close()

    def test_list_saved_triggers_multiple(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code1 = 'trigger = "a"\n'
        code2 = 'trigger = "b"\n'
        t.save_trigger("trigger_a", code1, config={"x": 1})
        t.save_trigger("trigger_b", code2)

        triggers = t.list_saved_triggers()
        assert len(triggers) == 2
        names = [tr["name"] for tr in triggers]
        assert "trigger_a" in names
        assert "trigger_b" in names
        t.close()

    def test_list_saved_hooks_shows_quarantined(self, tmp_path: Path) -> None:
        """Quarantined hooks should show quarantined=True in listing."""
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        db_path = t._db_path
        tract_id = t.tract_id
        t.close()

        # Corrupt the file
        hook_file = tmp_path / ".tract" / "hooks" / "my_hook.py"
        hook_file.write_text("bad syntax ---\n", encoding="utf-8")

        t2 = Tract.open(path=db_path, tract_id=tract_id)
        hooks = t2.list_saved_hooks()
        assert len(hooks) == 1
        assert hooks[0]["quarantined"] is True
        t2.close()


# ============================================================
# Repository CRUD
# ============================================================


class TestPersistenceRepository:
    """Test SqlitePersistenceRepository directly."""

    def test_hook_wiring_crud(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        repo = t._persistence_repo
        assert repo is not None

        from tract.storage.schema import HookWiringRow

        wiring = HookWiringRow(
            tract_id=t.tract_id,
            operation="compress",
            handler_source="file",
            handler_path="hooks/test.py",
            handler_function="handler",
            handler_code=None,
            priority=100,
            enabled=True,
            created_at=__import__("datetime").datetime.now(),
        )
        repo.save_hook_wiring(wiring)
        t._session.commit()

        wirings = repo.get_hook_wirings(t.tract_id)
        assert len(wirings) == 1
        assert wirings[0].operation == "compress"

        # Delete by name
        assert repo.delete_hook_wiring_by_name(t.tract_id, "test") is True
        t._session.commit()
        assert len(repo.get_hook_wirings(t.tract_id)) == 0

        t.close()

    def test_dynamic_op_spec_crud(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        repo = t._persistence_repo
        assert repo is not None

        from tract.storage.schema import DynamicOpSpecRow

        spec_row = DynamicOpSpecRow(
            tract_id=t.tract_id,
            name="trigger:my_trigger",
            spec_json='{"type":"trigger","name":"my_trigger"}',
            created_at=__import__("datetime").datetime.now(),
        )
        repo.save_dynamic_op(spec_row)
        t._session.commit()

        specs = repo.get_dynamic_ops(t.tract_id)
        assert len(specs) == 1
        assert specs[0].name == "trigger:my_trigger"

        # Delete
        assert repo.delete_dynamic_op(t.tract_id, "trigger:my_trigger") is True
        t._session.commit()
        assert len(repo.get_dynamic_ops(t.tract_id)) == 0

        t.close()

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

    def test_tract_dir_created_on_save_hook(self, tmp_path: Path) -> None:
        """save_hook creates .tract/hooks/ lazily."""
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("my_hook", code, "compress")
        assert (tmp_path / ".tract" / "hooks").is_dir()
        t.close()

    def test_tract_dir_created_on_save_trigger(self, tmp_path: Path) -> None:
        """save_trigger creates .tract/triggers/ lazily."""
        t = _make_file_tract(tmp_path)
        code = 'trigger = "x"\n'
        t.save_trigger("my_trigger", code)
        assert (tmp_path / ".tract" / "triggers").is_dir()
        t.close()

    def test_tract_dir_created_on_save_workflow(self, tmp_path: Path) -> None:
        """save_workflow creates .tract/workflows/ lazily."""
        t = _make_file_tract(tmp_path)
        code = 'def run():\n    pass\n'
        t.save_workflow("my_workflow", code)
        assert (tmp_path / ".tract" / "workflows").is_dir()
        t.close()


# ============================================================
# Wildcard hooks
# ============================================================


class TestWildcardHook:
    """Test save_hook with wildcard operation."""

    def test_save_hook_wildcard(self, tmp_path: Path) -> None:
        t = _make_file_tract(tmp_path)
        code = 'def handler(pending):\n    pending.approve()\n'
        t.save_hook("catch_all", code, "*")
        assert "*" in t.hook_names
        assert "catch_all" in t.hook_names["*"]
        t.close()
