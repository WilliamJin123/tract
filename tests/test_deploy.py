"""Tests for session.deploy() -- sub-agent deployment with curated context.

Covers:
- deploy() creates child on a new branch
- Parent branch unchanged after deploy
- Child on correct branch
- keep_tags curation: non-matching commits SKIPPED
- drop curation: specified commits SKIPPED
- drop of edit_target raises CurationError
- compact_before compresses old history
- reorder changes commit order
- Full curation pipeline in correct order
- Merge-back workflow: deploy -> work -> merge
- deploy without curation (just branch)
- SpawnInfo recorded with "branch" mode
- deploy with branch that already exists -> BranchExistsError
"""

from __future__ import annotations

import pytest

from tract import (
    BranchExistsError,
    CommitInfo,
    CurationError,
    DialogueContent,
    InstructionContent,
    Priority,
    Session,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _setup_tag_repos(tract_instance) -> None:
    """Set up tag repos on a tract instance created via Session.create_tract().

    Session.create_tract() doesn't set up tag repositories or seed base tags,
    so we add them here for tests that need tag functionality.
    """
    from tract.storage.sqlite import (
        SqliteTagAnnotationRepository,
        SqliteTagRegistryRepository,
    )
    tract_instance._tag_annotation_repo = SqliteTagAnnotationRepository(tract_instance._session)
    tract_instance._tag_registry_repo = SqliteTagRegistryRepository(tract_instance._session)
    tract_instance._seed_base_tags()


def _make_session_with_parent(n_commits: int = 3) -> tuple:
    """Create a Session with a parent tract that has n_commits.

    Returns (session, parent, commit_hashes).
    """
    session = Session.open()
    parent = session.create_tract(display_name="orchestrator")
    _setup_tag_repos(parent)

    hashes = []
    for i in range(n_commits):
        if i == 0:
            info = parent.commit(
                InstructionContent(text=f"System prompt {i}"),
                message=f"commit-{i}",
            )
        else:
            info = parent.commit(
                DialogueContent(role="user", text=f"Message {i}"),
                message=f"commit-{i}",
            )
        hashes.append(info.commit_hash)
    return session, parent, hashes


# ===========================================================================
# Basic deploy tests
# ===========================================================================


class TestDeployBasic:
    """Tests for basic deploy() functionality."""

    def test_deploy_creates_child_on_branch(self):
        """deploy() returns a Tract on the specified branch."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research",
            branch_name="research-task",
        )

        assert child.current_branch == "research-task"
        session.close()

    def test_parent_branch_unchanged_after_deploy(self):
        """Parent stays on its original branch after deploy()."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research",
            branch_name="research-task",
        )

        assert parent.current_branch == "main"
        session.close()

    def test_child_shares_same_tract_id(self):
        """Child operates on the same tract_id as parent."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research",
            branch_name="research-task",
        )

        assert child.tract_id == parent.tract_id
        session.close()

    def test_child_has_same_commits(self):
        """Child branch starts with the same commits as parent HEAD."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research",
            branch_name="research-task",
        )

        # Child should see the same HEAD as parent at deploy time
        assert child.head == hashes[-1]
        session.close()

    def test_deploy_without_curation(self):
        """deploy() without curation just creates a branch."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="simple task",
            branch_name="simple-branch",
        )

        assert child.current_branch == "simple-branch"
        assert child.head == hashes[-1]
        session.close()

    def test_deploy_duplicate_branch_raises(self):
        """deploy() with an existing branch name raises BranchExistsError."""
        session, parent, hashes = _make_session_with_parent(3)

        # First deploy succeeds
        session.deploy(
            parent,
            purpose="task1",
            branch_name="task-branch",
        )

        # Second deploy with same branch name raises
        with pytest.raises(BranchExistsError):
            session.deploy(
                parent,
                purpose="task2",
                branch_name="task-branch",
            )

        session.close()

    def test_spawn_info_recorded_with_branch_mode(self):
        """deploy() records a SpawnInfo with inheritance_mode='branch'."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research",
            branch_name="research-task",
        )

        # Check the spawn pointer was recorded
        # child_tract_id is "{tract_id}:{branch_name}" for branch mode
        child_pointer_id = f"{parent.tract_id}:research-task"
        pointer = session._spawn_repo.get_by_child(child_pointer_id)
        assert pointer is not None
        assert pointer.inheritance_mode == "branch"
        assert pointer.purpose == "research"
        assert pointer.display_name == "research-task"
        session.close()


# ===========================================================================
# Curation: keep_tags
# ===========================================================================


class TestCurateKeepTags:
    """Tests for keep_tags curation."""

    def test_keep_tags_skips_non_matching(self):
        """Commits without matching tags get SKIP annotation."""
        session, parent, hashes = _make_session_with_parent(4)

        # Register a tag and tag one commit
        parent.register_tag("important")
        parent.tag(hashes[1], "important")

        child = session.deploy(
            parent,
            purpose="focused work",
            branch_name="focused",
            curate={"keep_tags": ["important"]},
        )

        # Only hashes[1] should NOT be skipped
        for h in [hashes[0], hashes[2], hashes[3]]:
            annotations = child.get_annotations(h)
            skip_anns = [a for a in annotations if a.priority == Priority.SKIP]
            assert len(skip_anns) > 0, f"Expected SKIP annotation on {h[:12]}"

        # hashes[1] should NOT have SKIP
        annotations = child.get_annotations(hashes[1])
        skip_anns = [a for a in annotations if a.priority == Priority.SKIP]
        assert len(skip_anns) == 0, "Tagged commit should not be skipped"

        session.close()

    def test_keep_tags_with_immutable_tags(self):
        """keep_tags works with immutable commit tags too."""
        session = Session.open()
        parent = session.create_tract(display_name="parent")
        _setup_tag_repos(parent)

        # Commit with immutable tag
        h1 = parent.commit(
            DialogueContent(role="user", text="Message 1"),
            message="msg1",
            tags=["tool_call"],
        ).commit_hash

        # Commit without tag
        h2 = parent.commit(
            DialogueContent(role="user", text="Message 2"),
            message="msg2",
        ).commit_hash

        child = session.deploy(
            parent,
            purpose="focused",
            branch_name="focused",
            curate={"keep_tags": ["tool_call"]},
        )

        # h1 should not be skipped, h2 should be skipped
        anns_h2 = child.get_annotations(h2)
        skip_anns = [a for a in anns_h2 if a.priority == Priority.SKIP]
        assert len(skip_anns) > 0

        anns_h1 = child.get_annotations(h1)
        skip_anns = [a for a in anns_h1 if a.priority == Priority.SKIP]
        assert len(skip_anns) == 0

        session.close()


# ===========================================================================
# Curation: drop
# ===========================================================================


class TestCurateDrop:
    """Tests for drop curation."""

    def test_drop_marks_commit_as_skipped(self):
        """Dropped commits get SKIP annotation."""
        session, parent, hashes = _make_session_with_parent(4)

        child = session.deploy(
            parent,
            purpose="pruned",
            branch_name="pruned",
            curate={"drop": [hashes[1], hashes[2]]},
        )

        # Dropped commits should be SKIPPED
        for h in [hashes[1], hashes[2]]:
            annotations = child.get_annotations(h)
            skip_anns = [a for a in annotations if a.priority == Priority.SKIP]
            assert len(skip_anns) > 0, f"Expected SKIP on {h[:12]}"

        # Non-dropped commits should NOT be skipped
        for h in [hashes[0], hashes[3]]:
            annotations = child.get_annotations(h)
            skip_anns = [a for a in annotations if a.priority == Priority.SKIP]
            assert len(skip_anns) == 0, f"Did not expect SKIP on {h[:12]}"

        session.close()

    def test_drop_nonexistent_hash_raises(self):
        """Dropping a hash not on the branch raises CurationError."""
        session, parent, hashes = _make_session_with_parent(3)

        with pytest.raises(CurationError, match="not found"):
            session.deploy(
                parent,
                purpose="bad drop",
                branch_name="bad-drop",
                curate={"drop": ["nonexistent_hash_abc123"]},
            )

        session.close()

    def test_drop_edit_target_raises(self):
        """Dropping a commit that is the edit_target of another raises CurationError."""
        session = Session.open()
        parent = session.create_tract(display_name="parent")
        _setup_tag_repos(parent)

        # Create initial commit
        h1 = parent.commit(
            DialogueContent(role="user", text="Original message"),
            message="original",
        ).commit_hash

        # Create edit of h1
        from tract import CommitOperation
        h2 = parent.commit(
            DialogueContent(role="user", text="Edited message"),
            operation=CommitOperation.EDIT,
            message="edit",
            edit_target=h1,
        ).commit_hash

        # Try to drop h1 (which is edit_target of h2)
        with pytest.raises(CurationError, match="edit_target"):
            session.deploy(
                parent,
                purpose="bad edit drop",
                branch_name="bad-edit-drop",
                curate={"drop": [h1]},
            )

        session.close()


# ===========================================================================
# Curation: compact_before
# ===========================================================================


class TestCurateCompactBefore:
    """Tests for compact_before curation."""

    def test_compact_before_reduces_commits(self):
        """compact_before compresses commits before the marker."""
        session, parent, hashes = _make_session_with_parent(5)

        child = session.deploy(
            parent,
            purpose="compact test",
            branch_name="compact-test",
            curate={"compact_before": hashes[3]},
        )

        # After compaction, the child should still compile successfully
        compiled = child.compile()
        assert compiled is not None
        assert len(compiled.messages) > 0

        # The total number of commits should be less than the original
        # (3 commits before marker compressed into 1 summary)
        child_log = child.log(limit=100)
        # We should have fewer commits than original 5
        # (at most: 1 summary + marker + commit after marker = 3)
        assert len(child_log) < 5

        session.close()

    def test_compact_before_bad_marker_raises(self):
        """compact_before with non-existent marker raises CurationError."""
        session, parent, hashes = _make_session_with_parent(3)

        with pytest.raises(CurationError, match="not found"):
            session.deploy(
                parent,
                purpose="bad compact",
                branch_name="bad-compact",
                curate={"compact_before": "nonexistent_hash_xyz"},
            )

        session.close()

    def test_compact_before_first_commit_is_noop(self):
        """compact_before the first commit is a no-op (nothing before it)."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="compact first",
            branch_name="compact-first",
            curate={"compact_before": hashes[0]},
        )

        # All original commits should still be present
        child_log = child.log(limit=100)
        assert len(child_log) == 3

        session.close()


# ===========================================================================
# Curation: reorder
# ===========================================================================


class TestCurateReorder:
    """Tests for reorder curation."""

    def test_reorder_changes_commit_order(self):
        """Reordering commits changes their order in the log."""
        session, parent, hashes = _make_session_with_parent(3)

        # Reverse the order
        reversed_hashes = list(reversed(hashes))

        child = session.deploy(
            parent,
            purpose="reorder test",
            branch_name="reorder-test",
            curate={"reorder": reversed_hashes},
        )

        # The child should compile successfully
        compiled = child.compile()
        assert compiled is not None
        assert len(compiled.messages) == 3

        # The compiled messages should be in the reversed order's content
        # (the first commit in the new order should appear first in compile output)
        # Note: commit hashes change after replay, but content should match
        messages = compiled.messages
        assert "Message 2" in messages[0].content  # was last, now first
        assert "Message 1" in messages[1].content
        assert "System prompt 0" in messages[2].content  # was first, now last

        session.close()

    def test_reorder_nonexistent_hash_raises(self):
        """Reordering with a non-existent hash raises CurationError."""
        session, parent, hashes = _make_session_with_parent(3)

        with pytest.raises(CurationError, match="not found"):
            session.deploy(
                parent,
                purpose="bad reorder",
                branch_name="bad-reorder",
                curate={"reorder": ["nonexistent_abc", hashes[0]]},
            )

        session.close()


# ===========================================================================
# Full curation pipeline
# ===========================================================================


class TestFullCurationPipeline:
    """Tests for applying multiple curation steps in the correct order."""

    def test_keep_tags_then_drop(self):
        """keep_tags + drop applied in correct order."""
        session = Session.open()
        parent = session.create_tract(display_name="parent")
        _setup_tag_repos(parent)
        parent.register_tag("keep")

        h1 = parent.commit(
            DialogueContent(role="user", text="Msg 1"),
            tags=["keep"],
        ).commit_hash
        h2 = parent.commit(
            DialogueContent(role="user", text="Msg 2"),
            tags=["keep"],
        ).commit_hash
        h3 = parent.commit(
            DialogueContent(role="user", text="Msg 3"),
        ).commit_hash

        child = session.deploy(
            parent,
            purpose="pipeline test",
            branch_name="pipeline",
            curate={
                "keep_tags": ["keep"],  # Keeps h1, h2; skips h3
                "drop": [h2],           # Additionally drop h2
            },
        )

        # h1 should not be skipped
        anns = child.get_annotations(h1)
        assert all(a.priority != Priority.SKIP for a in anns)

        # h2 should be skipped (by drop)
        anns = child.get_annotations(h2)
        assert any(a.priority == Priority.SKIP for a in anns)

        # h3 should be skipped (by keep_tags)
        anns = child.get_annotations(h3)
        assert any(a.priority == Priority.SKIP for a in anns)

        session.close()


# ===========================================================================
# Merge-back workflow
# ===========================================================================


class TestMergeBackWorkflow:
    """Tests for merging child branch back into parent."""

    def test_deploy_work_merge(self):
        """deploy -> commit on child -> merge back to parent."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research",
            branch_name="research",
        )

        # Commit on child branch
        child_commit = child.commit(
            DialogueContent(role="assistant", text="Research result"),
            message="research finding",
        )

        # Parent should not see the child's commit yet
        parent_log = parent.log(limit=100)
        parent_hashes = [c.commit_hash for c in parent_log]
        assert child_commit.commit_hash not in parent_hashes

        # Merge child branch into parent
        result = parent.merge("research")

        # Parent should now include the child's commits
        parent_log_after = parent.log(limit=100)
        # The parent should have more commits after merge
        assert len(parent_log_after) > len(parent_log)

        session.close()

    def test_deploy_work_manual_summary(self):
        """deploy -> commit on child -> manually summarize back to parent.

        Note: collapse() uses spawn pointers with separate tract_ids.
        For branch-based deploys, the recommended merge-back is via merge().
        This test verifies that manual summarization still works by
        committing the summary directly to the parent.
        """
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="research task",
            branch_name="research-summary",
        )

        # Commit on child branch
        child.commit(
            DialogueContent(role="assistant", text="Research result A"),
            message="finding A",
        )
        child.commit(
            DialogueContent(role="assistant", text="Research result B"),
            message="finding B",
        )

        # Manually summarize child context back to parent
        child_compiled = child.compile()
        assert len(child_compiled.messages) > 0

        summary = "Summary: Findings A and B"
        parent.commit(
            DialogueContent(role="assistant", text=summary),
            message="collapse: research task",
        )

        # Parent should have the summary commit
        parent_head_info = parent.log(limit=1)[0]
        assert "collapse" in parent_head_info.message.lower()

        session.close()


# ===========================================================================
# Edge cases
# ===========================================================================


class TestDeployEdgeCases:
    """Edge case tests for deploy()."""

    def test_deploy_empty_curation_dict(self):
        """deploy() with empty curation dict is equivalent to no curation."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="empty curate",
            branch_name="empty-curate",
            curate={},
        )

        assert child.current_branch == "empty-curate"
        assert child.head == hashes[-1]
        session.close()

    def test_child_can_commit_independently(self):
        """Child commits do not appear on parent branch."""
        session, parent, hashes = _make_session_with_parent(3)

        child = session.deploy(
            parent,
            purpose="independent",
            branch_name="independent",
        )

        # Commit on child
        child.commit(
            DialogueContent(role="user", text="Child only message"),
            message="child commit",
        )

        # Parent HEAD should be unchanged
        assert parent.head == hashes[-1]

        # Child should have one more commit
        child_log = child.log(limit=100)
        assert len(child_log) == 4  # 3 original + 1 child commit

        session.close()

    def test_multiple_deploys_from_same_parent(self):
        """Multiple children can be deployed from the same parent."""
        session, parent, hashes = _make_session_with_parent(3)

        child1 = session.deploy(
            parent,
            purpose="task 1",
            branch_name="task-1",
        )
        child2 = session.deploy(
            parent,
            purpose="task 2",
            branch_name="task-2",
        )

        assert child1.current_branch == "task-1"
        assert child2.current_branch == "task-2"
        assert parent.current_branch == "main"

        # Both children start at the same HEAD
        assert child1.head == hashes[-1]
        assert child2.head == hashes[-1]

        # Independent commits
        child1.commit(DialogueContent(role="user", text="From child 1"))
        child2.commit(DialogueContent(role="user", text="From child 2"))

        assert child1.head != child2.head

        session.close()
