"""Tests for Session class and cross-repo query operations.

Tests cover session lifecycle, cross-repo queries (timeline, search,
compile_at), resume/crash recovery, concurrent threading, and the
get_child_tract ("expand for debugging") feature.
"""

from __future__ import annotations

import concurrent.futures
import time

import pytest

from tract import (
    CollapseResult,
    DialogueContent,
    InstructionContent,
    Session,
    SessionError,
)
from tract.models.session import SessionContent


# ---------------------------------------------------------------------------
# Session lifecycle tests
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Tests for Session creation and management."""

    def test_session_open(self, tmp_path):
        """Session.open() creates engine and returns Session."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        assert session is not None
        assert repr(session).startswith("Session(")
        session.close()

    def test_session_create_tract(self, tmp_path):
        """create_tract() returns a Tract that can commit."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        tract = session.create_tract(display_name="test-agent")

        info = tract.commit(InstructionContent(text="hello"))
        assert info.commit_hash is not None
        assert info.tract_id == tract.tract_id

        session.close()

    def test_session_get_tract(self, tmp_path):
        """get_tract() retrieves a previously created tract."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        tract = session.create_tract()
        tract.commit(InstructionContent(text="hello"))
        tid = tract.tract_id

        # Clear cache to force reconstruction
        del session._tracts[tid]

        retrieved = session.get_tract(tid)
        assert retrieved.tract_id == tid

        session.close()

    def test_session_get_tract_not_found(self, tmp_path):
        """get_tract() raises SessionError for unknown tract_id."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        with pytest.raises(SessionError, match="Tract not found"):
            session.get_tract("nonexistent-id-12345")
        session.close()

    def test_session_list_tracts(self, tmp_path):
        """list_tracts() returns all tracts with metadata."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        t1 = session.create_tract()
        t1.commit(InstructionContent(text="hello"))
        t2 = session.create_tract()
        t2.commit(DialogueContent(role="user", text="world"))

        tracts = session.list_tracts()
        ids = {t["tract_id"] for t in tracts}
        assert t1.tract_id in ids
        assert t2.tract_id in ids

        for t in tracts:
            assert "commit_count" in t
            assert "is_active" in t
            assert t["commit_count"] >= 1

        session.close()

    def test_session_context_manager(self, tmp_path):
        """Session works as context manager."""
        db_path = str(tmp_path / "test.db")
        with Session.open(db_path) as session:
            tract = session.create_tract()
            tract.commit(InstructionContent(text="hello"))
            assert tract.head is not None

        # Session should be closed after exiting context
        assert session._closed is True


# ---------------------------------------------------------------------------
# Cross-repo query tests
# ---------------------------------------------------------------------------


class TestCrossRepoQueries:
    """Tests for timeline, search, and compile_at operations."""

    def test_timeline_chronological(self, tmp_path):
        """timeline() returns commits from all tracts in time order."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        t1 = session.create_tract()
        t1.commit(InstructionContent(text="first"))
        time.sleep(0.01)  # Ensure different timestamps

        t2 = session.create_tract()
        t2.commit(DialogueContent(role="user", text="second"))

        tl = session.timeline()
        assert len(tl) == 2
        # Should be chronological
        assert tl[0].created_at <= tl[1].created_at

        session.close()

    def test_timeline_with_limit(self, tmp_path):
        """timeline(limit=N) returns at most N commits."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()
        for i in range(10):
            t.commit(DialogueContent(role="user", text=f"msg {i}"))

        tl = session.timeline(limit=5)
        assert len(tl) == 5

        session.close()

    def test_search_finds_matching(self, tmp_path):
        """search("keyword") finds commits containing the keyword."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()
        t.commit(DialogueContent(role="user", text="The cat sat on the mat"))
        t.commit(DialogueContent(role="user", text="A dog chased the ball"))

        results = session.search("cat")
        assert len(results) >= 1
        # At least one result should contain "cat" in content
        assert any(r.content_type == "dialogue" for r in results)

        session.close()

    def test_search_scoped_to_tract(self, tmp_path):
        """search("keyword", tract_id=X) only returns commits from tract X."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t1 = session.create_tract()
        t1.commit(DialogueContent(role="user", text="apple pie recipe"))
        t2 = session.create_tract()
        t2.commit(DialogueContent(role="user", text="apple sauce guide"))

        results = session.search("apple", tract_id=t1.tract_id)
        assert all(r.tract_id == t1.tract_id for r in results)

        session.close()

    def test_search_no_results(self, tmp_path):
        """search("nonexistent") returns empty list."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()
        t.commit(DialogueContent(role="user", text="hello world"))

        results = session.search("zzz_nonexistent_zzz")
        assert results == []

        session.close()

    def test_compile_at_time(self, tmp_path):
        """compile_at(tract_id, at_time=T) compiles tract as of time T."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()

        info1 = t.commit(InstructionContent(text="first"))
        cutoff = info1.created_at
        time.sleep(0.01)
        t.commit(DialogueContent(role="user", text="second"))

        # Compile at time of first commit only
        compiled = session.compile_at(t.tract_id, at_time=cutoff)
        assert compiled.commit_count == 1

        session.close()

    def test_compile_at_commit(self, tmp_path):
        """compile_at(tract_id, at_commit=H) compiles tract up to commit H."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()

        info1 = t.commit(InstructionContent(text="first"))
        t.commit(DialogueContent(role="user", text="second"))

        compiled = session.compile_at(t.tract_id, at_commit=info1.commit_hash)
        assert compiled.commit_count == 1

        session.close()

    def test_session_content_queryable(self, tmp_path):
        """SessionContent commits are queryable by content_type 'session'."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()

        # Commit a session boundary
        t.commit(SessionContent(
            session_type="start",
            summary="Starting work",
            decisions=["Decision A"],
        ))
        t.commit(DialogueContent(role="user", text="hello"))

        # Search for session content
        results = session.search("Starting work")
        assert len(results) >= 1

        session.close()


# ---------------------------------------------------------------------------
# Resume / crash recovery tests
# ---------------------------------------------------------------------------


class TestResumeAndCrashRecovery:
    """Tests for resume() and crash recovery scenarios."""

    def test_resume_finds_latest_active(self, tmp_path):
        """resume() returns most recent active tract."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t1 = session.create_tract()
        t1.commit(InstructionContent(text="old"))
        time.sleep(0.01)
        t2 = session.create_tract()
        t2.commit(InstructionContent(text="new"))

        result = session.resume()
        assert result is not None
        # Should find one of the active tracts
        assert result.tract_id in {t1.tract_id, t2.tract_id}

        session.close()

    def test_resume_skips_ended_tracts(self, tmp_path):
        """Tract with session_type='end' commit is not returned."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        t1 = session.create_tract()
        t1.commit(InstructionContent(text="hello"))
        # Mark t1 as ended
        t1.commit(SessionContent(
            session_type="end",
            summary="Done",
        ))

        t2 = session.create_tract()
        t2.commit(InstructionContent(text="still active"))

        result = session.resume()
        assert result is not None
        assert result.tract_id == t2.tract_id

        session.close()

    def test_resume_prefers_root_tracts(self, tmp_path):
        """Root tract preferred over child tracts."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        parent = session.create_tract()
        parent.commit(InstructionContent(text="root"))
        child = session.spawn(parent, purpose="child work")
        child.commit(DialogueContent(role="user", text="child data"))

        result = session.resume()
        assert result is not None
        assert result.tract_id == parent.tract_id

        session.close()

    def test_resume_empty_db(self, tmp_path):
        """resume() returns None for empty session."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        result = session.resume()
        assert result is None

        session.close()

    def test_crash_recovery(self, tmp_path):
        """Create tracts, 'crash' (close session), reopen, verify all commits intact."""
        db_path = str(tmp_path / "test.db")

        # Create session and tracts
        with Session.open(db_path) as s1:
            t = s1.create_tract()
            t.commit(InstructionContent(text="before crash"))
            t.commit(DialogueContent(role="user", text="important data"))
            tid = t.tract_id

        # "Crash" happened -- reopen
        with Session.open(db_path) as s2:
            tracts = s2.list_tracts()
            assert any(t["tract_id"] == tid for t in tracts)

            recovered = s2.get_tract(tid)
            compiled = recovered.compile()
            # Should have the 2 commits
            assert compiled.commit_count == 2

    def test_resume_after_crash(self, tmp_path):
        """Create tract, commit, close, reopen, resume() finds the tract."""
        db_path = str(tmp_path / "test.db")

        with Session.open(db_path) as s1:
            t = s1.create_tract()
            t.commit(InstructionContent(text="hello"))
            tid = t.tract_id

        with Session.open(db_path) as s2:
            result = s2.resume()
            assert result is not None
            assert result.tract_id == tid


# ---------------------------------------------------------------------------
# Concurrent multi-threaded tests
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Tests for concurrent access from multiple threads."""

    def test_concurrent_commits_from_different_threads(self, tmp_path):
        """Two tracts, each in its own thread, both commit simultaneously."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t1 = session.create_tract()
        t2 = session.create_tract()

        def commit_messages(tract, count):
            for i in range(count):
                tract.commit(
                    DialogueContent(role="user", text=f"msg-{i}")
                )

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(commit_messages, t1, 10)
            f2 = pool.submit(commit_messages, t2, 10)
            f1.result(timeout=30)
            f2.result(timeout=30)

        # Verify all commits exist
        tl = session.timeline()
        assert len(tl) == 20

        session.close()

    def test_read_while_write(self, tmp_path):
        """One thread commits, another reads timeline -- no errors."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t = session.create_tract()

        errors = []

        def writer():
            for i in range(10):
                try:
                    t.commit(DialogueContent(role="user", text=f"msg-{i}"))
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(10):
                try:
                    session.timeline()
                    time.sleep(0.001)
                except Exception as e:
                    errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(writer)
            f2 = pool.submit(reader)
            f1.result(timeout=30)
            f2.result(timeout=30)

        assert errors == [], f"Errors during concurrent read/write: {errors}"

        session.close()

    def test_concurrent_spawn_and_commit(self, tmp_path):
        """One thread spawns a child from t1, another commits to t2 simultaneously."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        t1 = session.create_tract()
        t1.commit(InstructionContent(text="root for spawn"))
        t2 = session.create_tract()
        t2.commit(InstructionContent(text="root for commits"))

        errors = []
        child_holder = [None]

        def spawner():
            try:
                child_holder[0] = session.spawn(t1, purpose="concurrent spawn")
            except Exception as e:
                errors.append(e)

        def committer():
            try:
                for i in range(5):
                    t2.commit(
                        DialogueContent(role="user", text=f"concurrent-{i}")
                    )
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f1 = pool.submit(spawner)
            f2 = pool.submit(committer)
            f1.result(timeout=30)
            f2.result(timeout=30)

        assert errors == [], f"Errors during concurrent spawn+commit: {errors}"
        assert child_holder[0] is not None

        session.close()


# ---------------------------------------------------------------------------
# Expand for debugging tests
# ---------------------------------------------------------------------------


class TestExpandForDebugging:
    """Tests for get_child_tract (navigate from collapse commit to child)."""

    def test_get_child_tract(self, tmp_path):
        """Collapse a child, then get_child_tract returns it."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        parent = session.create_tract()
        parent.commit(InstructionContent(text="root"))

        child = session.spawn(parent, purpose="debug test")
        child.commit(DialogueContent(role="user", text="child work"))
        child.commit(DialogueContent(role="assistant", text="child result"))

        result = session.collapse(
            child, into=parent, content="Summary of child", auto_commit=True
        )

        # Navigate back to child
        recovered_child = session.get_child_tract(result.parent_commit_hash)
        assert recovered_child.tract_id == child.tract_id
        # Child commits are still accessible
        compiled = recovered_child.compile()
        assert compiled.commit_count >= 2

        session.close()

    def test_get_child_tract_invalid_commit(self, tmp_path):
        """get_child_tract with a non-collapse commit raises SessionError."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        parent = session.create_tract()
        info = parent.commit(InstructionContent(text="not a collapse"))

        with pytest.raises(SessionError, match="not a collapse commit"):
            session.get_child_tract(info.commit_hash)

        session.close()


# ---------------------------------------------------------------------------
# Session.release_tract tests
# ---------------------------------------------------------------------------


class TestReleaseTract:
    """Tests for Session.release_tract() memory management."""

    def test_release_tract_removes_from_dict(self, tmp_path):
        """release_tract() removes the tract from _tracts."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        tract = session.create_tract()
        tid = tract.tract_id
        assert tid in session._tracts

        session.release_tract(tid)
        assert tid not in session._tracts
        session.close()

    def test_release_tract_clears_cache(self, tmp_path):
        """release_tract() clears the tract's compile cache."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        tract = session.create_tract()
        tract.commit(InstructionContent(text="hello"))
        tract.compile()
        assert len(tract._cache._cache) > 0

        session.release_tract(tract.tract_id)
        assert len(tract._cache._cache) == 0
        session.close()

    def test_release_unknown_tract_raises(self, tmp_path):
        """release_tract() raises SessionError for unknown tract_id."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        with pytest.raises(SessionError, match="not held in session"):
            session.release_tract("nonexistent")
        session.close()

    def test_released_tract_can_be_reacquired(self, tmp_path):
        """After release, get_tract() reconstructs from DB."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        tract = session.create_tract()
        tid = tract.tract_id
        tract.commit(InstructionContent(text="persisted"))

        session.release_tract(tid)
        assert tid not in session._tracts

        # Reconstruct from DB
        recovered = session.get_tract(tid)
        assert recovered.tract_id == tid
        result = recovered.compile()
        assert result.commit_count == 1
        session.close()
