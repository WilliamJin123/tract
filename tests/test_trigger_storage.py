"""Tests for trigger storage: schema, migration, repository, domain models, and exceptions.

Covers:
- TriggerProposalRow and TriggerLogRow table creation
- Schema migration v4->v5 and full v1->v5 chain
- SqliteTriggerRepository CRUD operations
- Proposal status updates and pending filtering
- Log entry filtering by time, trigger_name, ordering, and limit
- Log entry deletion (audit GC)
- Domain models (TriggerAction, EvaluationResult, TriggerLogEntry)
- Trigger exceptions (TriggerExecutionError, TriggerConfigError)
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest
from sqlalchemy import inspect, select, text
from sqlalchemy.orm import Session, sessionmaker

from tract.storage.engine import create_trace_engine, init_db
from tract.storage.schema import (
    TriggerLogRow,
    TriggerProposalRow,
    TraceMetaRow,
)
from tract.storage.sqlite import SqliteTriggerRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """Return a naive UTC datetime for SQLite storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _make_proposal(
    proposal_id: str = "prop-001",
    tract_id: str = "test-tract",
    trigger_name: str = "token_budget",
    action_type: str = "compress",
    status: str = "pending",
    created_at: datetime | None = None,
) -> TriggerProposalRow:
    """Create a TriggerProposalRow for testing."""
    return TriggerProposalRow(
        proposal_id=proposal_id,
        tract_id=tract_id,
        trigger_name=trigger_name,
        action_type=action_type,
        action_params_json={"target_tokens": 1000},
        reason="Token budget exceeded",
        status=status,
        created_at=created_at or _now(),
    )


def _make_log_entry(
    tract_id: str = "test-tract",
    trigger_name: str = "token_budget",
    trigger: str = "commit",
    outcome: str = "executed",
    created_at: datetime | None = None,
) -> TriggerLogRow:
    """Create a TriggerLogRow for testing."""
    return TriggerLogRow(
        tract_id=tract_id,
        trigger_name=trigger_name,
        trigger=trigger,
        action_type="compress",
        reason="Budget exceeded",
        outcome=outcome,
        commit_hash=None,
        error_message=None,
        created_at=created_at or _now(),
    )


# ===========================================================================
# Schema Tests
# ===========================================================================


class TestTriggerSchema:
    """Tests for trigger table creation and migration."""

    def test_trigger_tables_created(self, engine):
        """init_db creates trigger_proposals and trigger_log tables."""
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "trigger_proposals" in table_names
        assert "trigger_log" in table_names

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

    def test_migration_v4_to_v6(self):
        """Start with schema_version=4, call init_db, verify tables + version=6."""
        engine = create_trace_engine(":memory:")

        from tract.storage.schema import Base

        # Create all tables, then drop trigger + v6 tables to simulate v4
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS trigger_log"))
            conn.execute(text("DROP TABLE IF EXISTS trigger_proposals"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.commit()

        # Set schema version to 4
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            session.add(TraceMetaRow(key="schema_version", value="4"))
            session.commit()

        # Now call init_db -- should migrate v4->v5->v6
        init_db(engine)

        # Verify trigger tables exist
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "trigger_proposals" in table_names
        assert "trigger_log" in table_names
        # Verify v6 tables exist
        assert "operation_events" in table_names
        assert "operation_commits" in table_names
        assert "compile_records" in table_names
        assert "compile_effectives" in table_names

        # Verify schema version is now 6
        with SessionLocal() as session:
            row = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert int(row.value) >= 7

        engine.dispose()

    def test_migration_v1_to_v6_full_chain(self):
        """Start with schema_version=1, verify full migration chain v1->v2->v3->v4->v5->v6."""
        engine = create_trace_engine(":memory:")

        from tract.storage.schema import Base

        # Create all tables, then drop everything added after v1 to simulate v1
        Base.metadata.create_all(engine)
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS trigger_log"))
            conn.execute(text("DROP TABLE IF EXISTS trigger_proposals"))
            conn.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            conn.execute(text("DROP TABLE IF EXISTS compile_effectives"))
            conn.execute(text("DROP TABLE IF EXISTS compile_records"))
            conn.execute(text("DROP TABLE IF EXISTS operation_commits"))
            conn.execute(text("DROP TABLE IF EXISTS operation_events"))
            conn.execute(text("DROP TABLE IF EXISTS commit_parents"))
            conn.commit()

        # Set schema version to 1
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            session.add(TraceMetaRow(key="schema_version", value="1"))
            session.commit()

        # Now call init_db -- should migrate v1->v2->v3->v4->v5->v6
        init_db(engine)

        # Verify all tables exist (old compression tables dropped by v5->v6)
        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert "commit_parents" in table_names
        assert "spawn_pointers" in table_names
        assert "trigger_proposals" in table_names
        assert "trigger_log" in table_names
        assert "operation_events" in table_names
        assert "operation_commits" in table_names
        assert "compile_records" in table_names
        assert "compile_effectives" in table_names
        # Old compression tables should be gone (dropped by v5->v6 migration)
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

    def test_proposal_row_roundtrip(self, session):
        """Create TriggerProposalRow, save, retrieve."""
        proposal = _make_proposal()
        session.add(proposal)
        session.flush()

        fetched = session.execute(
            select(TriggerProposalRow).where(
                TriggerProposalRow.proposal_id == "prop-001"
            )
        ).scalar_one()

        assert fetched.tract_id == "test-tract"
        assert fetched.trigger_name == "token_budget"
        assert fetched.action_type == "compress"
        assert fetched.action_params_json == {"target_tokens": 1000}
        assert fetched.reason == "Token budget exceeded"
        assert fetched.status == "pending"
        assert fetched.resolved_at is None

    def test_log_row_roundtrip(self, session):
        """Create TriggerLogRow, save, retrieve."""
        entry = _make_log_entry()
        session.add(entry)
        session.flush()

        fetched = session.execute(
            select(TriggerLogRow).where(TriggerLogRow.id == entry.id)
        ).scalar_one()

        assert fetched.tract_id == "test-tract"
        assert fetched.trigger_name == "token_budget"
        assert fetched.trigger == "commit"
        assert fetched.action_type == "compress"
        assert fetched.outcome == "executed"
        assert fetched.commit_hash is None


# ===========================================================================
# Repository Tests
# ===========================================================================


class TestTriggerRepository:
    """Tests for SqliteTriggerRepository."""

    @pytest.fixture
    def repo(self, session: Session) -> SqliteTriggerRepository:
        return SqliteTriggerRepository(session)

    def test_save_and_get_proposal(self, repo: SqliteTriggerRepository):
        """save_proposal() then get_proposal() returns matching data."""
        proposal = _make_proposal()
        repo.save_proposal(proposal)

        fetched = repo.get_proposal("prop-001")
        assert fetched is not None
        assert fetched.tract_id == "test-tract"
        assert fetched.trigger_name == "token_budget"
        assert fetched.status == "pending"

    def test_get_proposal_not_found(self, repo: SqliteTriggerRepository):
        """get_proposal() returns None for nonexistent ID."""
        assert repo.get_proposal("nonexistent") is None

    def test_get_pending_proposals(self, repo: SqliteTriggerRepository):
        """get_pending_proposals() returns only pending proposals."""
        base_time = _now()
        repo.save_proposal(_make_proposal(
            proposal_id="p1", status="pending",
            created_at=base_time,
        ))
        repo.save_proposal(_make_proposal(
            proposal_id="p2", status="approved",
            created_at=base_time + timedelta(seconds=1),
        ))
        repo.save_proposal(_make_proposal(
            proposal_id="p3", status="pending",
            created_at=base_time + timedelta(seconds=2),
        ))
        repo.save_proposal(_make_proposal(
            proposal_id="p4", status="rejected",
            created_at=base_time + timedelta(seconds=3),
        ))

        pending = repo.get_pending_proposals("test-tract")
        assert len(pending) == 2
        assert pending[0].proposal_id == "p1"
        assert pending[1].proposal_id == "p3"

    def test_get_pending_proposals_empty(self, repo: SqliteTriggerRepository):
        """get_pending_proposals() returns empty list when no pending."""
        repo.save_proposal(_make_proposal(proposal_id="p1", status="approved"))
        assert repo.get_pending_proposals("test-tract") == []

    def test_update_proposal_status(self, repo: SqliteTriggerRepository):
        """update_proposal_status() changes status and sets resolved_at."""
        repo.save_proposal(_make_proposal(proposal_id="p1"))
        resolved = _now() + timedelta(minutes=5)
        repo.update_proposal_status("p1", "approved", resolved)

        fetched = repo.get_proposal("p1")
        assert fetched is not None
        assert fetched.status == "approved"
        assert fetched.resolved_at == resolved

    def test_save_and_get_log_entry(self, repo: SqliteTriggerRepository):
        """save_log_entry() then get_log() returns matching data."""
        entry = _make_log_entry()
        repo.save_log_entry(entry)

        log = repo.get_log("test-tract")
        assert len(log) == 1
        assert log[0].trigger_name == "token_budget"
        assert log[0].trigger == "commit"
        assert log[0].outcome == "executed"

    def test_get_log_ordering(self, repo: SqliteTriggerRepository):
        """get_log() returns entries ordered by created_at DESC."""
        base_time = _now()
        for i in range(3):
            repo.save_log_entry(_make_log_entry(
                created_at=base_time + timedelta(seconds=i),
                trigger_name=f"trigger_{i}",
            ))

        log = repo.get_log("test-tract")
        assert len(log) == 3
        # DESC order: most recent first
        assert log[0].trigger_name == "trigger_2"
        assert log[1].trigger_name == "trigger_1"
        assert log[2].trigger_name == "trigger_0"

    def test_get_log_limit(self, repo: SqliteTriggerRepository):
        """get_log() respects limit parameter."""
        base_time = _now()
        for i in range(5):
            repo.save_log_entry(_make_log_entry(
                created_at=base_time + timedelta(seconds=i),
            ))

        log = repo.get_log("test-tract", limit=2)
        assert len(log) == 2

    def test_get_log_filter_since(self, repo: SqliteTriggerRepository):
        """get_log() filters by since parameter."""
        base_time = _now()
        for i in range(3):
            repo.save_log_entry(_make_log_entry(
                created_at=base_time + timedelta(seconds=i),
            ))

        # Only entries at or after second one
        log = repo.get_log("test-tract", since=base_time + timedelta(seconds=1))
        assert len(log) == 2

    def test_get_log_filter_until(self, repo: SqliteTriggerRepository):
        """get_log() filters by until parameter."""
        base_time = _now()
        for i in range(3):
            repo.save_log_entry(_make_log_entry(
                created_at=base_time + timedelta(seconds=i),
            ))

        # Only entries at or before second one
        log = repo.get_log("test-tract", until=base_time + timedelta(seconds=1))
        assert len(log) == 2

    def test_get_log_filter_trigger_name(self, repo: SqliteTriggerRepository):
        """get_log() filters by trigger_name parameter."""
        base_time = _now()
        repo.save_log_entry(_make_log_entry(
            trigger_name="alpha", created_at=base_time,
        ))
        repo.save_log_entry(_make_log_entry(
            trigger_name="beta", created_at=base_time + timedelta(seconds=1),
        ))
        repo.save_log_entry(_make_log_entry(
            trigger_name="alpha", created_at=base_time + timedelta(seconds=2),
        ))

        log = repo.get_log("test-tract", trigger_name="alpha")
        assert len(log) == 2
        assert all(e.trigger_name == "alpha" for e in log)

    def test_delete_log_entries(self, repo: SqliteTriggerRepository):
        """delete_log_entries() removes entries before timestamp."""
        base_time = _now()
        for i in range(5):
            repo.save_log_entry(_make_log_entry(
                created_at=base_time + timedelta(seconds=i),
            ))

        # Delete entries before t+2 (should remove t+0 and t+1)
        deleted = repo.delete_log_entries(
            "test-tract", base_time + timedelta(seconds=2)
        )
        assert deleted == 2

        remaining = repo.get_log("test-tract")
        assert len(remaining) == 3

    def test_delete_log_entries_none(self, repo: SqliteTriggerRepository):
        """delete_log_entries() returns 0 when nothing to delete."""
        deleted = repo.delete_log_entries("test-tract", _now())
        assert deleted == 0


# ===========================================================================
# Domain Model Tests
# ===========================================================================


class TestTriggerDomainModels:
    """Tests for trigger domain models."""

    def test_trigger_action_defaults(self):
        """TriggerAction has correct defaults."""
        from tract.models.trigger import TriggerAction

        action = TriggerAction(action_type="compress")
        assert action.action_type == "compress"
        assert action.params == {}
        assert action.reason == ""
        assert action.autonomy == "collaborative"

    def test_trigger_action_frozen(self):
        """TriggerAction is immutable."""
        from tract.models.trigger import TriggerAction

        action = TriggerAction(action_type="compress")
        with pytest.raises(AttributeError):
            action.action_type = "prune"  # type: ignore[misc]

    def test_evaluation_result_defaults(self):
        """EvaluationResult has correct defaults."""
        from tract.models.trigger import EvaluationResult

        result = EvaluationResult(trigger_name="budget", triggered=False)
        assert result.outcome == "skipped"
        assert result.action is None
        assert result.error is None
        assert result.commit_hash is None

    def test_evaluation_result_frozen(self):
        """EvaluationResult is immutable."""
        from tract.models.trigger import EvaluationResult

        result = EvaluationResult(trigger_name="budget", triggered=True)
        with pytest.raises(AttributeError):
            result.triggered = False  # type: ignore[misc]

    def test_trigger_log_entry(self):
        """TriggerLogEntry has all required fields."""
        from tract.models.trigger import TriggerLogEntry

        entry = TriggerLogEntry(
            id=1,
            tract_id="t1",
            trigger_name="budget",
            trigger="commit",
            action_type="compress",
            reason="Over budget",
            outcome="executed",
            commit_hash="abc123",
            error_message=None,
            created_at=_now(),
        )
        assert entry.id == 1
        assert entry.trigger == "commit"
        assert entry.outcome == "executed"


# ===========================================================================
# Exception Tests
# ===========================================================================


class TestTriggerExceptions:
    """Tests for trigger-specific exceptions."""

    def test_trigger_execution_error(self):
        """TriggerExecutionError inherits from TraceError."""
        from tract.exceptions import TriggerExecutionError, TraceError

        error = TriggerExecutionError("test")
        assert isinstance(error, TraceError)

    def test_trigger_config_error(self):
        """TriggerConfigError inherits from TraceError."""
        from tract.exceptions import TriggerConfigError, TraceError

        error = TriggerConfigError("bad config")
        assert isinstance(error, TraceError)
