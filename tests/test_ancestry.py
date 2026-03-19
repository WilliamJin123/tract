"""Tests for ancestry operations -- walk_ancestry DAG traversal.

Tests the walk_ancestry function which walks commit ancestry from HEAD
to root, with optional content_type filtering and merge-parent traversal.
"""

from __future__ import annotations

from tract import (
    DialogueContent,
    InstructionContent,
    Tract,
)
from tract.operations.ancestry import walk_ancestry
from tests.conftest import make_tract, populate_tract


# ==================================================================
# Helpers
# ==================================================================

def _get_internals(t: Tract):
    """Extract internal repos from a Tract instance for direct walk_ancestry calls."""
    return t._commit_repo, t._blob_repo


# ==================================================================
# Basic walk_ancestry behavior
# ==================================================================

class TestWalkAncestryBasic:
    """Tests for basic walk_ancestry traversal."""

    def test_single_commit(self):
        """Walk from a single commit returns just that commit."""
        t = make_tract()
        hashes = populate_tract(t, 1)
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(commit_repo, blob_repo, hashes[0])
        assert len(result) == 1
        assert result[0].commit_hash == hashes[0]

    def test_linear_chain_root_first_order(self):
        """Walk returns commits in root-to-head order."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(commit_repo, blob_repo, hashes[-1])
        assert len(result) == 3
        # Root first
        assert result[0].commit_hash == hashes[0]
        assert result[1].commit_hash == hashes[1]
        assert result[2].commit_hash == hashes[2]

    def test_walk_from_middle_commit(self):
        """Walking from a middle commit excludes later commits."""
        t = make_tract()
        hashes = populate_tract(t, 5)
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(commit_repo, blob_repo, hashes[2])
        assert len(result) == 3
        assert result[0].commit_hash == hashes[0]
        assert result[-1].commit_hash == hashes[2]

    def test_deep_chain(self):
        """Walk handles a deeper chain correctly."""
        t = make_tract()
        hashes = []
        for i in range(20):
            info = t.commit(DialogueContent(role="user", text=f"Msg {i}"))
            hashes.append(info.commit_hash)
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(commit_repo, blob_repo, hashes[-1])
        assert len(result) == 20
        # Verify root-first ordering
        assert result[0].commit_hash == hashes[0]
        assert result[-1].commit_hash == hashes[-1]


# ==================================================================
# Content type filtering
# ==================================================================

class TestWalkAncestryContentTypeFilter:
    """Tests for content_type_filter parameter."""

    def test_filter_by_dialogue(self):
        """Filter to dialogue content only."""
        t = make_tract()
        # InstructionContent has content_type="instruction"
        t.commit(InstructionContent(text="System prompt"))
        h2 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h3 = t.commit(DialogueContent(role="assistant", text="Hi")).commit_hash
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(
            commit_repo, blob_repo, h3,
            content_type_filter={"dialogue"},
        )
        assert len(result) == 2
        assert all(r.content_type == "dialogue" for r in result)
        assert result[0].commit_hash == h2
        assert result[1].commit_hash == h3

    def test_filter_by_instruction(self):
        """Filter to instruction content only."""
        t = make_tract()
        h1 = t.commit(InstructionContent(text="System prompt")).commit_hash
        t.commit(DialogueContent(role="user", text="Hello"))
        t.commit(DialogueContent(role="assistant", text="Hi"))
        commit_repo, blob_repo = _get_internals(t)

        head = t.head
        result = walk_ancestry(
            commit_repo, blob_repo, head,
            content_type_filter={"instruction"},
        )
        assert len(result) == 1
        assert result[0].commit_hash == h1

    def test_filter_by_config(self):
        """Filter to config content type."""
        t = make_tract()
        populate_tract(t, 2)
        t.config.set(model="gpt-4")
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(
            commit_repo, blob_repo, t.head,
            content_type_filter={"config"},
        )
        assert len(result) == 1
        assert result[0].content_type == "config"

    def test_filter_no_matches(self):
        """Filter with no matching content type returns empty list."""
        t = make_tract()
        populate_tract(t, 3)
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(
            commit_repo, blob_repo, t.head,
            content_type_filter={"nonexistent_type"},
        )
        assert result == []

    def test_filter_multiple_types(self):
        """Filter with multiple content types returns all matching."""
        t = make_tract()
        t.commit(InstructionContent(text="System prompt"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.config.set(model="gpt-4")
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(
            commit_repo, blob_repo, t.head,
            content_type_filter={"instruction", "config"},
        )
        assert len(result) == 2
        types = {r.content_type for r in result}
        assert types == {"instruction", "config"}

    def test_no_filter_returns_all(self):
        """With content_type_filter=None, all commits returned."""
        t = make_tract()
        t.commit(InstructionContent(text="System prompt"))
        t.commit(DialogueContent(role="user", text="Hello"))
        t.config.set(model="gpt-4")
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(commit_repo, blob_repo, t.head)
        assert len(result) == 3


# ==================================================================
# Merge parent traversal
# ==================================================================

class TestWalkAncestryMergeParents:
    """Tests for parent_repo parameter (merge commit ancestry)."""

    def test_without_parent_repo_skips_merge_parents(self):
        """Without parent_repo, merge parents are not walked."""
        t = make_tract()
        populate_tract(t, 2)

        # Create a branch, add commits, then merge
        t.branch("feature")
        t.checkout("feature")
        t.commit(
            DialogueContent(role="user", text="Feature work")
        )
        t.checkout("main")
        t.merge("feature")

        commit_repo, blob_repo = _get_internals(t)

        # Without parent_repo, walk only follows the linear parent chain
        result_without = walk_ancestry(commit_repo, blob_repo, t.head)

        # With parent_repo, walk also follows merge parents
        result_with = walk_ancestry(
            commit_repo, blob_repo, t.head,
            parent_repo=t._parent_repo,
        )

        # The with-parent version should include at least as many commits
        without_hashes = {r.commit_hash for r in result_without}
        with_hashes = {r.commit_hash for r in result_with}
        assert without_hashes.issubset(with_hashes)

    def test_merge_parent_includes_branch_commits(self):
        """With parent_repo, merge ancestry includes commits from merged branch."""
        t = make_tract()
        main_h1 = t.commit(
            InstructionContent(text="System prompt")
        ).commit_hash

        t.branch("feature")
        t.checkout("feature")
        feat_h1 = t.commit(
            DialogueContent(role="user", text="Feature 1")
        ).commit_hash
        feat_h2 = t.commit(
            DialogueContent(role="user", text="Feature 2")
        ).commit_hash

        t.checkout("main")
        main_h2 = t.commit(
            DialogueContent(role="user", text="Main work")
        ).commit_hash

        t.merge("feature")
        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(
            commit_repo, blob_repo, t.head,
            parent_repo=t._parent_repo,
        )
        result_hashes = {r.commit_hash for r in result}

        # All commits should be reachable
        assert main_h1 in result_hashes
        assert main_h2 in result_hashes
        assert feat_h1 in result_hashes
        assert feat_h2 in result_hashes

    def test_merge_parent_with_content_filter(self):
        """Content type filter applies to merge parent ancestry too."""
        t = make_tract()
        t.commit(InstructionContent(text="System prompt"))

        t.branch("feature")
        t.checkout("feature")
        t.commit(DialogueContent(role="user", text="Feature dialogue"))
        t.config.set(model="gpt-4")

        t.checkout("main")
        t.commit(DialogueContent(role="user", text="Main dialogue"))
        t.merge("feature")

        commit_repo, blob_repo = _get_internals(t)

        result = walk_ancestry(
            commit_repo, blob_repo, t.head,
            content_type_filter={"config"},
            parent_repo=t._parent_repo,
        )
        assert all(r.content_type == "config" for r in result)

    def test_parent_repo_none_equivalent_to_omitted(self):
        """Passing parent_repo=None has same result as not passing it."""
        t = make_tract()
        populate_tract(t, 3)
        commit_repo, blob_repo = _get_internals(t)

        result_default = walk_ancestry(commit_repo, blob_repo, t.head)
        result_none = walk_ancestry(
            commit_repo, blob_repo, t.head, parent_repo=None,
        )
        assert len(result_default) == len(result_none)
        assert [r.commit_hash for r in result_default] == [
            r.commit_hash for r in result_none
        ]
