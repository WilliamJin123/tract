"""End-to-end integration tests for the complete multi-agent workflow.

These tests exercise the full stack: Session -> spawn -> commit -> collapse ->
timeline -> search -> resume, using real SQLite databases (tmp_path fixtures).
No mocks except where LLM calls would be needed.
"""

from __future__ import annotations

import json
import time

import pytest

from tract import (
    CollapseResult,
    DialogueContent,
    InstructionContent,
    Session,
    SessionError,
    SpawnInfo,
    Tract,
)
from tract.models.session import SessionContent


# ---------------------------------------------------------------------------
# 1. Full multi-agent workflow
# ---------------------------------------------------------------------------


class TestFullMultiagentWorkflow:
    """Complete spawn -> commit -> collapse -> verify cycle."""

    def test_full_multiagent_workflow(self, tmp_path):
        """End-to-end: create session, spawn, commit, collapse, verify pointers."""
        db_path = str(tmp_path / "e2e.db")
        with Session.open(db_path) as session:
            # Parent tract: orchestrator
            parent = session.create_tract(display_name="orchestrator")
            parent.commit(InstructionContent(text="You are a project orchestrator."))
            parent.commit(
                DialogueContent(role="user", text="Research quantum computing trends.")
            )

            # Spawn child for research
            child = session.spawn(parent, purpose="research quantum computing")

            # Child does its work
            child.commit(
                DialogueContent(
                    role="assistant",
                    text="Quantum computing has seen advances in error correction...",
                )
            )
            child.commit(
                DialogueContent(
                    role="assistant",
                    text="Key players: IBM, Google, IonQ. Topological qubits are promising.",
                )
            )

            # Collapse child into parent (manual mode)
            result = session.collapse(
                child,
                into=parent,
                content="Quantum computing trends: error correction advances, IBM/Google/IonQ lead.",
                auto_commit=True,
            )

            # Verify collapse result
            assert result.parent_commit_hash is not None
            assert result.child_tract_id == child.tract_id
            assert result.summary_tokens > 0

            # Verify parent has spawn + collapse commits
            parent_log = parent.log(limit=10)
            messages = [c.message for c in parent_log if c.message]
            assert any("spawn:" in m for m in messages)
            assert any("collapse:" in m for m in messages)

            # Verify child has its own commits
            child_compiled = child.compile()
            assert child_compiled.commit_count >= 2

            # Timeline includes all commits from both tracts
            tl = session.timeline()
            tract_ids = {c.tract_id for c in tl}
            assert parent.tract_id in tract_ids
            assert child.tract_id in tract_ids

            # Verify spawn pointers
            children_list = parent.children()
            assert len(children_list) == 1
            assert children_list[0].child_tract_id == child.tract_id

            parent_info = child.parent()
            assert parent_info is not None
            assert parent_info.parent_tract_id == parent.tract_id


# ---------------------------------------------------------------------------
# 2. Recursive spawn
# ---------------------------------------------------------------------------


class TestRecursiveSpawn:
    """Parent -> child -> grandchild spawn chains."""

    def test_recursive_spawn(self, tmp_path):
        """Parent spawns child, child spawns grandchild. All pointers correct."""
        db_path = str(tmp_path / "recursive.db")
        with Session.open(db_path) as session:
            parent = session.create_tract(display_name="root")
            parent.commit(InstructionContent(text="Root context"))

            child = session.spawn(parent, purpose="level-1 task")
            child.commit(DialogueContent(role="assistant", text="Working on level 1"))

            grandchild = session.spawn(child, purpose="level-2 subtask")
            grandchild.commit(
                DialogueContent(role="assistant", text="Deep work at level 2")
            )

            # Verify pointer chain
            assert child.parent().parent_tract_id == parent.tract_id
            assert grandchild.parent().parent_tract_id == child.tract_id

            # Timeline includes all 3 tracts
            tl = session.timeline()
            tract_ids = {c.tract_id for c in tl}
            assert parent.tract_id in tract_ids
            assert child.tract_id in tract_ids
            assert grandchild.tract_id in tract_ids


# ---------------------------------------------------------------------------
# 3. Multiple collapse
# ---------------------------------------------------------------------------


class TestMultipleCollapse:
    """Collapse a child multiple times (interim updates)."""

    def test_multiple_collapse(self, tmp_path):
        """Spawn, work, collapse (interim), work more, collapse again."""
        db_path = str(tmp_path / "multi_collapse.db")
        with Session.open(db_path) as session:
            parent = session.create_tract()
            parent.commit(InstructionContent(text="Root"))

            child = session.spawn(parent, purpose="iterative research")
            child.commit(
                DialogueContent(role="assistant", text="Initial findings...")
            )

            # First collapse (interim)
            r1 = session.collapse(
                child,
                into=parent,
                content="Interim: found 3 relevant papers.",
                auto_commit=True,
            )
            assert r1.parent_commit_hash is not None

            # Child continues working
            child.commit(
                DialogueContent(role="assistant", text="Further analysis complete.")
            )

            # Second collapse
            r2 = session.collapse(
                child,
                into=parent,
                content="Final: analysis complete with 5 papers.",
                auto_commit=True,
            )
            assert r2.parent_commit_hash is not None

            # Parent should have 2 collapse commits
            parent_log = parent.log(limit=20)
            collapse_commits = [c for c in parent_log if c.message and "collapse:" in c.message]
            assert len(collapse_commits) == 2


# ---------------------------------------------------------------------------
# 4. Session boundary commit
# ---------------------------------------------------------------------------


class TestSessionBoundaryCommit:
    """SessionContent commit lifecycle."""

    def test_session_boundary_commit(self, tmp_path):
        """Commit SessionContent with session_type='end', verify compile and query."""
        db_path = str(tmp_path / "boundary.db")
        with Session.open(db_path) as session:
            t = session.create_tract()
            t.commit(InstructionContent(text="Working on task."))
            t.commit(
                SessionContent(
                    session_type="end",
                    summary="Completed the task.",
                    decisions=["Used approach A"],
                    next_steps=["Test approach A"],
                )
            )

            # Verify it compiles
            compiled = t.compile()
            assert compiled.commit_count == 2

            # Verify it is searchable
            results = session.search("Completed the task")
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# 5. Cross-repo search
# ---------------------------------------------------------------------------


class TestCrossRepoSearch:
    """Search across multiple tracts."""

    def test_cross_repo_search(self, tmp_path):
        """Two tracts, different content. Search finds only matching tract."""
        db_path = str(tmp_path / "cross_search.db")
        with Session.open(db_path) as session:
            t1 = session.create_tract()
            t1.commit(
                DialogueContent(role="user", text="Photosynthesis converts sunlight to energy.")
            )

            t2 = session.create_tract()
            t2.commit(
                DialogueContent(role="user", text="Quantum entanglement is spooky action.")
            )

            # Search for unique keyword
            results = session.search("Photosynthesis")
            assert len(results) >= 1
            assert all(r.tract_id == t1.tract_id for r in results)

            # Search for keyword only in t2
            results2 = session.search("entanglement")
            assert len(results2) >= 1
            assert all(r.tract_id == t2.tract_id for r in results2)


# ---------------------------------------------------------------------------
# 6. Cross-repo compile_at
# ---------------------------------------------------------------------------


class TestCrossRepoCompileAt:
    """Time-travel compilation across tracts."""

    def test_cross_repo_compile_at(self, tmp_path):
        """Commit at T1, commit at T2. compile_at(at_time=T1) returns only T1."""
        db_path = str(tmp_path / "compile_at.db")
        with Session.open(db_path) as session:
            t = session.create_tract()
            info1 = t.commit(InstructionContent(text="First instruction"))
            cutoff = info1.created_at
            time.sleep(0.01)  # Ensure different timestamps
            t.commit(DialogueContent(role="user", text="Second message"))

            compiled = session.compile_at(t.tract_id, at_time=cutoff)
            assert compiled.commit_count == 1


# ---------------------------------------------------------------------------
# 7. Crash recovery integration
# ---------------------------------------------------------------------------


class TestCrashRecoveryIntegration:
    """Full crash-recovery scenario with file-based DB."""

    def test_crash_recovery_integration(self, tmp_path):
        """Create, close, reopen, resume finds the tract."""
        db_path = str(tmp_path / "crash.db")

        # First session
        with Session.open(db_path) as s1:
            t = s1.create_tract()
            t.commit(InstructionContent(text="Important context"))
            t.commit(DialogueContent(role="user", text="Critical data"))
            tid = t.tract_id

        # Reopen after "crash"
        with Session.open(db_path) as s2:
            recovered = s2.resume()
            assert recovered is not None
            assert recovered.tract_id == tid

            # All commits accessible
            compiled = recovered.compile()
            assert compiled.commit_count == 2
            assert "Important context" in compiled.messages[0].content


# ---------------------------------------------------------------------------
# 8. Full clone inheritance
# ---------------------------------------------------------------------------


class TestFullCloneInheritance:
    """Spawn with full_clone replays all parent commits."""

    def test_full_clone_inheritance(self, tmp_path):
        """Parent has 5 commits. Child via full_clone has 5 commits with same content."""
        db_path = str(tmp_path / "clone.db")
        with Session.open(db_path) as session:
            parent = session.create_tract()
            parent.commit(InstructionContent(text="System prompt"))
            for i in range(4):
                parent.commit(
                    DialogueContent(role="user", text=f"Message {i}")
                )

            # Verify parent has 5 commits
            parent_compiled = parent.compile()
            assert parent_compiled.commit_count == 5

            child = session.spawn(
                parent, purpose="clone test", inheritance="full_clone"
            )

            # Child should have 5 commits (different hashes, same content)
            child_compiled = child.compile()
            assert child_compiled.commit_count == 5

            # Content should match
            for p_msg, c_msg in zip(
                parent_compiled.messages, child_compiled.messages
            ):
                # full_clone preserves content type and text
                assert p_msg.content == c_msg.content


# ---------------------------------------------------------------------------
# 9. Head snapshot inheritance
# ---------------------------------------------------------------------------


class TestHeadSnapshotInheritance:
    """Spawn with head_snapshot compiles parent into one initial commit."""

    def test_head_snapshot_inheritance(self, tmp_path):
        """Parent has 3 commits. Child via head_snapshot has 1 initial commit."""
        db_path = str(tmp_path / "snapshot.db")
        with Session.open(db_path) as session:
            parent = session.create_tract()
            parent.commit(InstructionContent(text="You are helpful."))
            parent.commit(DialogueContent(role="user", text="Hello"))
            parent.commit(DialogueContent(role="assistant", text="Hi there!"))

            child = session.spawn(parent, purpose="snapshot test")

            # Child should have 1 commit (compiled parent context)
            child_compiled = child.compile()
            assert child_compiled.commit_count == 1
            # The snapshot should contain parent content
            assert "You are helpful" in child_compiled.messages[0].content


# ---------------------------------------------------------------------------
# 10. Session list_tracts
# ---------------------------------------------------------------------------


class TestSessionListTracts:
    """Verify list_tracts returns all tracts with metadata."""

    def test_session_list_tracts(self, tmp_path):
        """Create 3 tracts, verify list_tracts returns all 3."""
        db_path = str(tmp_path / "list.db")
        with Session.open(db_path) as session:
            tracts = []
            for i in range(3):
                t = session.create_tract()
                t.commit(InstructionContent(text=f"Tract {i}"))
                tracts.append(t)

            result = session.list_tracts()
            listed_ids = {t["tract_id"] for t in result}
            for t in tracts:
                assert t.tract_id in listed_ids

            # Each entry has expected keys
            for entry in result:
                assert "commit_count" in entry
                assert "is_active" in entry
                assert entry["commit_count"] >= 1


# ---------------------------------------------------------------------------
# 11. Timeline ordering
# ---------------------------------------------------------------------------


class TestTimelineOrdering:
    """Verify timeline returns all commits in chronological order."""

    def test_timeline_ordering(self, tmp_path):
        """Two tracts, interleaved commits. Timeline is chronological."""
        db_path = str(tmp_path / "timeline.db")
        with Session.open(db_path) as session:
            t1 = session.create_tract()
            t2 = session.create_tract()

            t1.commit(InstructionContent(text="T1 first"))
            time.sleep(0.01)
            t2.commit(InstructionContent(text="T2 first"))
            time.sleep(0.01)
            t1.commit(DialogueContent(role="user", text="T1 second"))
            time.sleep(0.01)
            t2.commit(DialogueContent(role="user", text="T2 second"))

            tl = session.timeline()
            assert len(tl) == 4

            # Verify chronological order
            for i in range(len(tl) - 1):
                assert tl[i].created_at <= tl[i + 1].created_at


# ---------------------------------------------------------------------------
# 12. Resume skips ended sessions
# ---------------------------------------------------------------------------


class TestResumeSkipsEndedSessions:
    """Verify resume() skips tracts with session_type='end'."""

    def test_resume_skips_ended_sessions(self, tmp_path):
        """Two tracts, end one. resume() returns the other."""
        db_path = str(tmp_path / "resume_skip.db")
        with Session.open(db_path) as session:
            t1 = session.create_tract()
            t1.commit(InstructionContent(text="Agent A work"))
            t1.commit(
                SessionContent(session_type="end", summary="Done with A")
            )

            t2 = session.create_tract()
            t2.commit(InstructionContent(text="Agent B active"))

            result = session.resume()
            assert result is not None
            assert result.tract_id == t2.tract_id


# ---------------------------------------------------------------------------
# 13. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Single-agent Tract.open() still works without Session."""

    def test_backward_compatibility(self, tmp_path):
        """Tract.open() works for single-agent use. parent() and children() return empty."""
        db_path = str(tmp_path / "compat.db")
        with Tract.open(db_path) as t:
            t.commit(InstructionContent(text="Solo agent"))
            t.commit(DialogueContent(role="user", text="Hello"))

            compiled = t.compile()
            assert compiled.commit_count == 2

            # No spawn relationships
            assert t.parent() is None
            assert t.children() == []


# ---------------------------------------------------------------------------
# 14. Spawn with display_name
# ---------------------------------------------------------------------------


class TestSpawnWithDisplayName:
    """Verify display_name appears in SpawnInfo and list_tracts."""

    def test_spawn_with_display_name(self, tmp_path):
        """Spawn with display_name, verify it in SpawnInfo and list_tracts."""
        db_path = str(tmp_path / "display.db")
        with Session.open(db_path) as session:
            parent = session.create_tract(display_name="orchestrator")
            parent.commit(InstructionContent(text="Root"))

            child = session.spawn(
                parent, purpose="research", display_name="researcher"
            )
            child.commit(DialogueContent(role="assistant", text="Research data"))

            # SpawnInfo has display_name
            child_info = child.parent()
            assert child_info is not None
            assert child_info.display_name == "researcher"

            # list_tracts shows display_name
            tracts = session.list_tracts()
            child_entry = next(
                (t for t in tracts if t["tract_id"] == child.tract_id), None
            )
            assert child_entry is not None
            assert child_entry["display_name"] == "researcher"


# ---------------------------------------------------------------------------
# 15. Session schema migration (v3 -> v4)
# ---------------------------------------------------------------------------


class TestSessionSchemaMigration:
    """Verify v3 databases migrate to v4 when opened with Session."""

    def test_session_schema_migration(self, tmp_path):
        """Create a v3 database (no spawn_pointers), open with Session, verify migration."""
        from sqlalchemy import create_engine, text
        from sqlalchemy.orm import sessionmaker

        from tract.storage.schema import Base, TraceMetaRow

        db_path = str(tmp_path / "v3.db")

        # Create a v3 database manually (all tables except spawn_pointers)
        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        # Create all tables
        Base.metadata.create_all(engine)

        # Set schema version to 3 (simulating v3 state)
        SessionLocal = sessionmaker(bind=engine)
        with SessionLocal() as s:
            # Update schema_version to 3
            from sqlalchemy import select

            meta = s.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one_or_none()
            if meta:
                meta.value = "3"
            else:
                s.add(TraceMetaRow(key="schema_version", value="3"))
            s.commit()

            # Drop spawn_pointers to simulate v3
            s.execute(text("DROP TABLE IF EXISTS spawn_pointers"))
            s.commit()

        engine.dispose()

        # Now open with Session -- should auto-migrate to v4
        with Session.open(db_path) as session:
            t = session.create_tract()
            t.commit(InstructionContent(text="Post-migration commit"))
            assert t.head is not None

            # Verify schema is now v4
            from tract.storage.engine import create_trace_engine, create_session_factory
            from sqlalchemy import select as sel

            check_engine = create_trace_engine(db_path)
            sf = create_session_factory(check_engine)
            with sf() as cs:
                meta = cs.execute(
                    sel(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
                ).scalar_one()
                assert meta.value == "5"
            check_engine.dispose()


# ---------------------------------------------------------------------------
# 16. Cross-session handoff
# ---------------------------------------------------------------------------


class TestCrossSessionHandoff:
    """Agent A works, ends session. Agent B spawns from A's final state."""

    def test_cross_session_handoff(self, tmp_path):
        """Agent A ends session with decisions. Agent B inherits and can read them."""
        db_path = str(tmp_path / "handoff.db")

        # Agent A works
        with Session.open(db_path) as s1:
            agent_a = s1.create_tract(display_name="agent-a")
            agent_a.commit(InstructionContent(text="You are building a backend."))
            agent_a.commit(
                DialogueContent(role="assistant", text="Implemented user auth with JWT.")
            )
            agent_a.commit(
                SessionContent(
                    session_type="end",
                    summary="Built backend authentication.",
                    decisions=["Chose FastAPI", "Used JWT tokens"],
                    next_steps=["Add frontend", "Write tests"],
                )
            )
            agent_a_id = agent_a.tract_id

        # Agent B opens new session, spawns from Agent A
        with Session.open(db_path) as s2:
            # Get Agent A's tract
            agent_a_restored = s2.get_tract(agent_a_id)

            # Agent B spawns from Agent A's final state
            agent_b = s2.spawn(
                agent_a_restored,
                purpose="Add frontend",
                display_name="agent-b",
            )

            # Agent B has inherited context
            b_compiled = agent_b.compile()
            assert b_compiled.commit_count == 1  # head_snapshot = 1 commit

            # The snapshot should contain Agent A's session boundary info
            snapshot_text = b_compiled.messages[0].content
            assert "backend" in snapshot_text.lower() or "auth" in snapshot_text.lower()

            # Agent B can do its own work
            agent_b.commit(
                DialogueContent(role="assistant", text="Starting frontend with React.")
            )
            b_compiled2 = agent_b.compile()
            assert b_compiled2.commit_count == 2

            # Agent A's decisions are accessible from compiled parent context
            # (they were part of the session boundary commit that was compiled)
            parent_compiled = agent_a_restored.compile()
            session_messages = [
                m for m in parent_compiled.messages
                if "FastAPI" in m.content or "JWT" in m.content
            ]
            assert len(session_messages) >= 1
