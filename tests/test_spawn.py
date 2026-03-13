"""Tests for spawn and collapse operations.

Tests cover spawn creation, inheritance modes (head_snapshot, full_clone),
collapse with manual/auto modes, Tract.parent() and children() helpers,
and edge cases.
"""

from __future__ import annotations

import pytest

from tract import (
    CollapseResult,
    DialogueContent,
    InstructionContent,
    Session,
    SessionError,
    SpawnError,
    SpawnInfo,
)
from tract.models.session import SessionContent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_session_with_parent(tmp_path, *, n_commits=3):
    """Create a session with a parent tract that has some commits."""
    db_path = str(tmp_path / "test.db")
    session = Session.open(db_path)
    parent = session.create_tract(display_name="parent")
    parent.commit(InstructionContent(text="System: you are helpful"))
    for i in range(n_commits - 1):
        parent.commit(
            DialogueContent(role="user", text=f"Message {i + 1}")
        )
    return session, parent


# ---------------------------------------------------------------------------
# Spawn tests
# ---------------------------------------------------------------------------


class TestSpawn:
    """Tests for spawn_tract and related operations."""

    def test_spawn_head_snapshot(self, tmp_path):
        """Spawn with head_snapshot inherits compiled parent context."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="research task")

        # Child should have one commit (the inherited snapshot)
        compiled = child.compile()
        assert compiled.commit_count == 1
        # The inherited text should contain parent context
        assert compiled.messages[0].role == "system"
        assert "you are helpful" in compiled.messages[0].content

        session.close()

    def test_spawn_creates_parent_commit(self, tmp_path):
        """Spawning creates a commit in the parent documenting the spawn."""
        session, parent = _create_session_with_parent(tmp_path)
        head_before = parent.head

        session.spawn(parent, purpose="data analysis")

        # Parent should have a new commit
        assert parent.head != head_before
        log = parent.log(limit=1)
        assert len(log) == 1
        assert "spawn: data analysis" in log[0].message

        session.close()

    def test_spawn_creates_pointer(self, tmp_path):
        """Spawn pointer exists in DB with correct fields."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="summarize docs", display_name="summarizer")

        # Check spawn pointer via parent.children()
        children = parent.children()
        assert len(children) == 1
        info = children[0]
        assert isinstance(info, SpawnInfo)
        assert info.parent_tract_id == parent.tract_id
        assert info.child_tract_id == child.tract_id
        assert info.purpose == "summarize docs"
        assert info.inheritance_mode == "head_snapshot"
        assert info.display_name == "summarizer"

        session.close()

    def test_spawn_full_clone(self, tmp_path):
        """Spawn with full_clone replicates all parent commits."""
        session, parent = _create_session_with_parent(tmp_path, n_commits=4)
        # Parent has 4 commits + 1 spawn commit = 5 total after spawn
        child = session.spawn(parent, purpose="clone task", inheritance="full_clone")

        # Child should have the same number of commits as parent had BEFORE spawn
        # (4 original commits get cloned, spawn commit is not cloned)
        compiled = child.compile()
        # full_clone replays all original commits (but hashes differ)
        assert compiled.commit_count == 4

        session.close()

    def test_spawn_full_clone_preserves_content(self, tmp_path):
        """Full clone preserves content from parent commits."""
        session, parent = _create_session_with_parent(tmp_path, n_commits=2)
        child = session.spawn(parent, purpose="deep clone", inheritance="full_clone")

        parent_compiled = parent.compile()
        child_compiled = child.compile()

        # Content should match (ignoring the spawn commit in parent)
        # Parent has: instruction + 1 dialogue + spawn commit = 3 messages
        # Child has: instruction + 1 dialogue = 2 messages (cloned before spawn)
        assert child_compiled.commit_count == 2
        # First message should have same content
        assert child_compiled.messages[0].content == parent_compiled.messages[0].content

        session.close()

    def test_spawn_selective_requires_filter(self, tmp_path):
        """Spawn with selective but no filter criteria raises ValueError."""
        session, parent = _create_session_with_parent(tmp_path)
        with pytest.raises(ValueError, match="selective inheritance requires"):
            session.spawn(parent, purpose="selective", inheritance="selective")

        session.close()

    def test_spawn_purpose_required(self, tmp_path):
        """Spawn without purpose raises TypeError."""
        session, parent = _create_session_with_parent(tmp_path)
        with pytest.raises(TypeError):
            session.spawn(parent)  # type: ignore[call-arg]

        session.close()

    def test_spawn_display_name_optional(self, tmp_path):
        """Display name is stored when provided, None when not."""
        session, parent = _create_session_with_parent(tmp_path)

        child1 = session.spawn(parent, purpose="task1", display_name="worker-1")
        child2 = session.spawn(parent, purpose="task2")

        children = parent.children()
        names = {c.child_tract_id: c.display_name for c in children}
        assert names[child1.tract_id] == "worker-1"
        assert names[child2.tract_id] is None

        session.close()

    def test_spawn_recursive(self, tmp_path):
        """Tract A spawns B, B spawns C -- all pointers correct."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)

        a = session.create_tract(display_name="A")
        a.commit(InstructionContent(text="root"))

        b = session.spawn(a, purpose="subtask-B")
        b.commit(DialogueContent(role="user", text="working on B"))

        c = session.spawn(b, purpose="subtask-C")

        # Check parent chain
        assert c.parent().parent_tract_id == b.tract_id
        assert b.parent().parent_tract_id == a.tract_id
        assert a.parent() is None

        # Check children
        assert len(a.children()) == 1
        assert a.children()[0].child_tract_id == b.tract_id
        assert len(b.children()) == 1
        assert b.children()[0].child_tract_id == c.tract_id
        assert len(c.children()) == 0

        session.close()

    def test_tract_parent_and_children(self, tmp_path):
        """Tract.parent() and Tract.children() return correct SpawnInfo."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="test task")

        # Parent has no parent
        assert parent.parent() is None

        # Child has parent
        info = child.parent()
        assert info is not None
        assert info.parent_tract_id == parent.tract_id
        assert info.purpose == "test task"

        # Parent has one child
        children = parent.children()
        assert len(children) == 1
        assert children[0].child_tract_id == child.tract_id

        session.close()

    # ------------------------------------------------------------------
    # Spawn-with-persona tests
    # ------------------------------------------------------------------

    def test_spawn_with_profile(self, tmp_path):
        """Spawn with profile= loads the workflow profile on the child."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="coding task",
            profile="coding",
        )

        # Profile should be loaded — active_profile set
        assert child._active_profile is not None
        assert child._active_profile.name == "coding"
        # Profile directives should be committed
        compiled = child.compile()
        # At least the inherited snapshot + profile directives
        assert compiled.commit_count >= 2

        session.close()

    def test_spawn_with_profile_and_stage(self, tmp_path):
        """Spawn with profile + stage applies stage-specific config."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="implement feature",
            profile="coding",
            stage="implement",
        )

        # Stage config applies — implement stage sets temperature=0.2
        assert child.get_config("temperature") == 0.2
        assert child.get_config("compile_strategy") == "messages"

        session.close()

    def test_spawn_with_directives(self, tmp_path):
        """Spawn with directives= commits named directives on the child."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="scoped task",
            directives={
                "role": "You are a security analyst.",
                "scope": "Only analyze the auth module.",
            },
        )

        compiled = child.compile()
        text = " ".join(m.content for m in compiled.messages)
        assert "security analyst" in text
        assert "auth module" in text

        session.close()

    def test_spawn_with_configure(self, tmp_path):
        """Spawn with configure= applies config to the child."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="configured task",
            configure={"temperature": 0.1, "analyst_role": "performance"},
        )

        assert child.get_config("temperature") == 0.1
        assert child.get_config("analyst_role") == "performance"

        session.close()

    def test_spawn_directives_override_profile(self, tmp_path):
        """Directives passed to spawn override same-named profile directives."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="override test",
            profile="coding",
            directives={"methodology": "Custom methodology override."},
        )

        # The custom directive should win (closest-to-HEAD deduplication)
        compiled = child.compile()
        text = " ".join(m.content for m in compiled.messages)
        assert "Custom methodology override" in text

        session.close()

    def test_spawn_configure_overrides_stage(self, tmp_path):
        """Explicit configure overrides stage defaults from profile."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="explicit config",
            profile="coding",
            stage="implement",
            configure={"temperature": 0.9},
        )

        # Explicit temperature (0.9) should win over implement stage default (0.2)
        assert child.get_config("temperature") == 0.9
        # compile_strategy from stage still applies (not overridden)
        assert child.get_config("compile_strategy") == "messages"

        session.close()

    def test_spawn_stage_without_profile_raises(self, tmp_path):
        """Passing stage without profile raises ValueError."""
        session, parent = _create_session_with_parent(tmp_path)
        with pytest.raises(ValueError, match="stage requires profile"):
            session.spawn(
                parent,
                purpose="bad combo",
                stage="implement",
            )

        session.close()

    def test_spawn_all_persona_params(self, tmp_path):
        """Full persona: profile + stage + directives + configure."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(
            parent,
            purpose="full persona",
            profile="research",
            stage="ingest",
            directives={"focus": "Focus only on primary sources."},
            configure={"domain": "machine-learning"},
        )

        assert child._active_profile is not None
        assert child._active_profile.name == "research"
        # ingest stage sets temperature=0.3
        assert child.get_config("temperature") == 0.3
        # explicit configure applied last
        assert child.get_config("domain") == "machine-learning"
        compiled = child.compile()
        text = " ".join(m.content for m in compiled.messages)
        assert "primary sources" in text

        session.close()


# ---------------------------------------------------------------------------
# Collapse tests
# ---------------------------------------------------------------------------


class TestCollapse:
    """Tests for collapse_tract and related operations."""

    def test_collapse_manual(self, tmp_path):
        """Collapse with user-provided content creates summary in parent."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="research")
        child.commit(DialogueContent(role="user", text="Found important data"))
        child.commit(DialogueContent(role="assistant", text="Analysis: X = Y"))

        result = session.collapse(
            child, into=parent, content="Research complete: X = Y", auto_commit=True
        )

        assert isinstance(result, CollapseResult)
        assert result.summary_text == "Research complete: X = Y"
        assert result.purpose == "research"
        assert result.parent_commit_hash is not None
        assert result.child_tract_id == child.tract_id

        session.close()

    def test_collapse_creates_commit_in_parent(self, tmp_path):
        """Summary commit exists in parent with correct message."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="analysis")
        child.commit(DialogueContent(role="user", text="data"))

        result = session.collapse(
            child, into=parent, content="Analysis done", auto_commit=True
        )

        # Check the parent's latest commit
        log = parent.log(limit=1)
        assert len(log) == 1
        assert "collapse: analysis" in log[0].message
        assert log[0].commit_hash == result.parent_commit_hash

        session.close()

    def test_collapse_metadata(self, tmp_path):
        """Collapse commit has collapse_source_tract_id and collapse_source_head metadata."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="metadata test")
        child.commit(DialogueContent(role="user", text="hello"))

        child_head = child.head

        result = session.collapse(
            child, into=parent, content="Summary", auto_commit=True
        )

        commit = parent.get_commit(result.parent_commit_hash)
        assert commit is not None
        assert commit.metadata["collapse_source_tract_id"] == child.tract_id
        assert commit.metadata["collapse_source_head"] == child_head

        session.close()

    def test_collapse_without_llm_and_no_content_raises(self, tmp_path):
        """Collaborative mode without LLM client raises SpawnError."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="no llm")
        child.commit(DialogueContent(role="user", text="hi"))

        with pytest.raises(SpawnError, match="content or LLM client"):
            session.collapse(child, into=parent, auto_commit=True)

        session.close()

    def test_collapse_auto_commit_false(self, tmp_path):
        """Collapse with auto_commit=False returns result without committing."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="draft")
        child.commit(DialogueContent(role="user", text="hello"))

        parent_head_before = parent.head

        result = session.collapse(
            child, into=parent, content="Draft summary", auto_commit=False
        )

        # No commit in parent
        assert result.parent_commit_hash is None
        assert result.summary_text == "Draft summary"
        # Parent HEAD unchanged
        assert parent.head == parent_head_before

        session.close()

    def test_collapse_multiple_times(self, tmp_path):
        """Subagent can be collapsed multiple times (interim progress)."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="ongoing work")
        child.commit(DialogueContent(role="user", text="step 1"))

        r1 = session.collapse(
            child, into=parent, content="Progress update 1", auto_commit=True
        )
        assert r1.parent_commit_hash is not None

        child.commit(DialogueContent(role="user", text="step 2"))

        r2 = session.collapse(
            child, into=parent, content="Progress update 2", auto_commit=True
        )
        assert r2.parent_commit_hash is not None
        assert r2.parent_commit_hash != r1.parent_commit_hash

        session.close()

    def test_collapse_result_fields(self, tmp_path):
        """CollapseResult has correct parent_commit_hash, tokens, purpose."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="field test")
        child.commit(DialogueContent(role="user", text="some content here"))

        result = session.collapse(
            child, into=parent, content="Summary of findings", auto_commit=True
        )

        assert result.summary_tokens > 0
        assert result.source_tokens > 0
        assert result.purpose == "field test"
        assert result.child_tract_id == child.tract_id

        session.close()

    def test_collapse_preserves_child_tract(self, tmp_path):
        """After collapse, child tract and its commits still exist."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="preserved")
        child.commit(DialogueContent(role="user", text="important data"))
        child_head = child.head

        session.collapse(
            child, into=parent, content="Summary", auto_commit=True
        )

        # Child tract still works
        assert child.head == child_head
        compiled = child.compile()
        assert compiled.commit_count >= 1

        session.close()


# ---------------------------------------------------------------------------
# Inheritance detail tests
# ---------------------------------------------------------------------------


class TestInheritanceDetails:
    """Tests for inheritance mode details."""

    def test_head_snapshot_single_commit(self, tmp_path):
        """Child gets one commit with compiled context from head_snapshot."""
        session, parent = _create_session_with_parent(tmp_path, n_commits=5)
        child = session.spawn(parent, purpose="snapshot test")

        # Head snapshot produces exactly one commit
        log = child.log(limit=10)
        assert len(log) == 1
        assert log[0].content_type == "instruction"

        session.close()

    def test_full_clone_annotation_copy(self, tmp_path):
        """Annotations are copied during full clone."""
        from tract import Priority

        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        parent = session.create_tract()
        info = parent.commit(InstructionContent(text="important"))
        parent.annotate(info.commit_hash, Priority.PINNED, reason="key")

        child = session.spawn(parent, purpose="clone", inheritance="full_clone")

        # Child should have the commit with an annotation
        child_log = child.log(limit=10)
        assert len(child_log) >= 1
        # The cloned commit should have a PINNED annotation
        child_annotations = child.get_annotations(child_log[-1].commit_hash)
        # There will be 2: the auto-annotation (instruction=pinned) + the explicit PINNED
        pinned = [a for a in child_annotations if a.priority == Priority.PINNED]
        assert len(pinned) >= 1

        session.close()

    def test_head_snapshot_empty_parent(self, tmp_path):
        """Spawning from empty parent creates child with no commits."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        parent = session.create_tract()

        # Parent has no commits
        child = session.spawn(parent, purpose="empty parent test")

        # Child should have no commits (empty parent produces nothing)
        # The spawn commit was created in parent, but snapshot produced nothing
        compiled = child.compile()
        assert compiled.commit_count == 0

        session.close()
