"""Advanced session tests -- spawn/collapse edge cases and multi-agent patterns."""
from __future__ import annotations

import pytest

from tract import (
    CollapseResult,
    DialogueContent,
    InstructionContent,
    Session,
    SessionError,
    SpawnError,
)


# ---------------------------------------------------------------------------
# Spawn and Collapse edge cases
# ---------------------------------------------------------------------------


class TestSpawnAndCollapse:
    """Tests for spawn/collapse edge cases not covered by basic tests."""

    def test_multiple_spawn_from_same_parent(self, tmp_path):
        """Multiple children can spawn from the same parent."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Base")
            parent.user("Task")

            child1 = s.spawn(parent, purpose="research")
            child2 = s.spawn(parent, purpose="analysis")

            child1.user("Research finding")
            child2.user("Analysis result")

            # Both should have independent state
            c1 = child1.compile().to_text()
            c2 = child2.compile().to_text()
            assert "Research finding" in c1
            assert "Analysis result" in c2
            assert "Analysis result" not in c1
            assert "Research finding" not in c2

    def test_collapse_with_content(self, tmp_path):
        """Collapse with manual content brings summary into parent."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Orchestrator")

            child = s.spawn(parent, purpose="task")
            child.user("Working...")
            child.assistant("Done: result is 42")

            result = s.collapse(
                child, into=parent, content="Child found: 42", auto_commit=True
            )
            assert result is not None
            assert result.parent_commit_hash is not None

            compiled = parent.compile().to_text()
            assert "42" in compiled

    def test_sequential_spawn_collapse(self, tmp_path):
        """Spawn -> work -> collapse -> spawn -> work -> collapse chain."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Pipeline")

            # Stage 1
            w1 = s.spawn(parent, purpose="stage1")
            w1.user("Process input")
            w1.assistant("Stage 1 output: processed")
            s.collapse(
                w1, into=parent, content="Stage 1: processed input",
                auto_commit=True,
            )

            # Stage 2
            w2 = s.spawn(parent, purpose="stage2")
            w2.user("Refine")
            w2.assistant("Stage 2 output: refined")
            s.collapse(
                w2, into=parent, content="Stage 2: refined output",
                auto_commit=True,
            )

            # Parent should have both stages
            compiled = parent.compile().to_text()
            assert "Stage 1" in compiled
            assert "Stage 2" in compiled

    def test_spawn_inherits_system_prompt(self, tmp_path):
        """Child inherits parent's system prompt on spawn via head_snapshot."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("You are a helpful assistant.")
            parent.user("Hello")

            child = s.spawn(parent, purpose="subtask")
            compiled = child.compile().to_text()
            assert "helpful assistant" in compiled

    def test_deep_spawn_chain(self, tmp_path):
        """Grandchild spawn (parent -> child -> grandchild)."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            root = s.create_tract()
            root.system("Root instruction")
            root.user("Root task")

            child = s.spawn(root, purpose="child")
            child.user("Child work")

            grandchild = s.spawn(child, purpose="grandchild")
            grandchild.user("Grandchild work")

            # Grandchild should have inherited context from child
            # (which itself had root's snapshot)
            gc_text = grandchild.compile().to_text()
            assert "Root" in gc_text

    def test_collapse_chain_up_hierarchy(self, tmp_path):
        """Grandchild collapses into child, child collapses into root."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            root = s.create_tract()
            root.system("Director")

            child = s.spawn(root, purpose="manager")
            child.user("Managing subtask")

            grandchild = s.spawn(child, purpose="worker")
            grandchild.user("Doing detailed work")
            grandchild.assistant("Work result: X=42")

            # Collapse grandchild into child
            s.collapse(
                grandchild, into=child,
                content="Worker found X=42",
                auto_commit=True,
            )

            # Collapse child into root
            result = s.collapse(
                child, into=root,
                content="Manager reports: worker found X=42",
                auto_commit=True,
            )

            assert result.parent_commit_hash is not None
            root_text = root.compile().to_text()
            assert "X=42" in root_text


# ---------------------------------------------------------------------------
# compile_at tests
# ---------------------------------------------------------------------------


class TestSessionCompileAt:
    """Tests for Session.compile_at() cross-tract compilation."""

    def test_compile_at_specific_tract(self, tmp_path):
        """compile_at returns context for specific tract without switching."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            t1 = s.create_tract(display_name="agent_a")
            t1.system("Agent A system")
            t1.user("A's question")

            t2 = s.create_tract(display_name="agent_b")
            t2.system("Agent B system")
            t2.user("B's question")

            # Compile each without interfering
            c1 = s.compile_at(t1.tract_id)
            c2 = s.compile_at(t2.tract_id)

            assert "Agent A" in c1.to_text()
            assert "Agent B" in c2.to_text()
            assert "Agent B" not in c1.to_text()
            assert "Agent A" not in c2.to_text()

    def test_compile_at_after_spawn(self, tmp_path):
        """compile_at works for child tracts created via spawn."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Parent system")
            parent.user("Parent question")

            child = s.spawn(parent, purpose="child task")
            child.user("Child-specific work")

            # compile_at the child tract
            compiled = s.compile_at(child.tract_id)
            text = compiled.to_text()
            assert "Child-specific work" in text


# ---------------------------------------------------------------------------
# Session edge cases
# ---------------------------------------------------------------------------


class TestSessionEdgeCases:
    """Edge cases for session management."""

    def test_collapse_empty_child(self, tmp_path):
        """Collapsing a child with no work beyond spawn."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Base")

            child = s.spawn(parent, purpose="empty task")
            # Don't add any work to child
            result = s.collapse(
                child, into=parent, content="Nothing found",
                auto_commit=True,
            )
            assert result is not None
            assert result.summary_text == "Nothing found"

    def test_many_children_parallel(self, tmp_path):
        """Spawn and collapse many children."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Coordinator")

            results = []
            for i in range(5):
                child = s.spawn(parent, purpose=f"task_{i}")
                child.user(f"Working on task {i}")
                child.assistant(f"Result {i}: done")
                r = s.collapse(
                    child, into=parent,
                    content=f"Task {i} completed",
                    auto_commit=True,
                )
                results.append(r)

            assert len(results) == 5
            compiled = parent.compile().to_text()
            for i in range(5):
                assert f"Task {i}" in compiled

    def test_session_tract_isolation(self, tmp_path):
        """Tracts in same session have independent state."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            t1 = s.create_tract()
            t2 = s.create_tract()

            t1.system("System 1")
            t2.system("System 2")
            t1.user("User 1")
            t2.user("User 2")

            text1 = t1.compile().to_text()
            text2 = t2.compile().to_text()

            assert "System 1" in text1
            assert "System 2" not in text1
            assert "System 2" in text2
            assert "System 1" not in text2

    def test_collapse_preserves_purpose(self, tmp_path):
        """CollapseResult retains the child's original purpose."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Root")

            child = s.spawn(parent, purpose="specific research task")
            child.user("Research data")

            result = s.collapse(
                child, into=parent,
                content="Research complete",
                auto_commit=True,
            )

            assert result.purpose == "specific research task"
            assert result.child_tract_id == child.tract_id

    def test_multiple_collapses_from_same_child(self, tmp_path):
        """A single child can be collapsed multiple times (interim reports)."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Manager")

            child = s.spawn(parent, purpose="long task")
            child.user("Phase 1 work")
            child.assistant("Phase 1 done")

            r1 = s.collapse(
                child, into=parent,
                content="Interim: Phase 1 complete",
                auto_commit=True,
            )

            child.user("Phase 2 work")
            child.assistant("Phase 2 done")

            r2 = s.collapse(
                child, into=parent,
                content="Final: Phase 2 complete",
                auto_commit=True,
            )

            assert r1.parent_commit_hash != r2.parent_commit_hash
            text = parent.compile().to_text()
            assert "Phase 1" in text
            assert "Phase 2" in text

    def test_child_continues_after_collapse(self, tmp_path):
        """Child can continue working after being collapsed."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Root")

            child = s.spawn(parent, purpose="continuing worker")
            child.user("First batch")
            child.assistant("First result")

            s.collapse(
                child, into=parent,
                content="First batch done",
                auto_commit=True,
            )

            # Child should still be usable
            child.user("Second batch")
            child.assistant("Second result")

            child_compiled = child.compile()
            child_text = child_compiled.to_text()
            assert "Second batch" in child_text
            assert "Second result" in child_text

    def test_spawn_with_display_name(self, tmp_path):
        """Spawn records display_name in spawn info."""
        db = str(tmp_path / "test.db")
        with Session.open(db) as s:
            parent = s.create_tract()
            parent.system("Root")

            child = s.spawn(
                parent, purpose="named task",
                display_name="research-worker",
            )

            children = parent.spawn_children()
            assert len(children) >= 1
            child_info = [
                c for c in children if c.child_tract_id == child.tract_id
            ]
            assert len(child_info) == 1
            assert child_info[0].display_name == "research-worker"

    def test_autonomy_levels(self, tmp_path):
        """Different autonomy levels affect collapse auto_commit default."""
        db_manual = str(tmp_path / "manual.db")
        db_auto = str(tmp_path / "auto.db")

        # Autonomous mode: collapse auto-commits by default
        with Session.open(db_auto, autonomy="autonomous") as s:
            parent = s.create_tract()
            parent.system("Root")
            child = s.spawn(parent, purpose="auto task")
            child.user("Work")

            result = s.collapse(
                child, into=parent, content="Auto result",
            )
            # autonomous -> auto_commit=True by default
            assert result.parent_commit_hash is not None

        # Collaborative mode: collapse does NOT auto-commit by default
        with Session.open(db_manual, autonomy="collaborative") as s:
            parent = s.create_tract()
            parent.system("Root")
            child = s.spawn(parent, purpose="collab task")
            child.user("Work")

            result = s.collapse(
                child, into=parent, content="Collab result",
            )
            # collaborative -> auto_commit=False by default
            assert result.parent_commit_hash is None
