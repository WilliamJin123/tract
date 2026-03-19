"""Tests that find(), log(), query_by_tags(), and list_tags() traverse merge parents.

Verifies that after merging a branch, commits from the merged branch are
visible through these query methods -- not just the primary parent chain.
"""

from __future__ import annotations

import pytest

from tract import (
    CommitInfo,
    CommitOperation,
    DialogueContent,
    InstructionContent,
    Tract,
)


def _setup_merged_tract() -> tuple[Tract, str, str, str]:
    """Create a tract with diverged branches merged back together.

    Returns (tract, feature_only_hash, main_only_hash, merge_hash).

    Timeline:
        base -> main_commit (on main)
             \\-> feature_commit (on feature)
        Then merge feature into main, creating a merge commit.
    """
    t = Tract.open()
    t.register_tag("feature-tag", "Tag for feature work")
    t.register_tag("main-tag", "Tag for main work")
    t.commit(InstructionContent(text="base instruction"))

    # Create feature branch and commit
    t.branch("feature")
    feature_info = t.commit(
        DialogueContent(role="user", text="feature work"),
        tags=["feature-tag"],
    )
    feature_hash = feature_info.commit_hash

    # Switch to main and commit
    t.switch("main")
    main_info = t.commit(
        DialogueContent(role="user", text="main work"),
        tags=["main-tag"],
    )
    main_hash = main_info.commit_hash

    # Merge feature into main (no_ff to guarantee a merge commit)
    result = t.merge("feature", no_ff=True)
    assert result.committed
    merge_hash = t.head

    return t, feature_hash, main_hash, merge_hash


# ---------------------------------------------------------------------------
# find() tests
# ---------------------------------------------------------------------------


class TestFindTraversesMergeParents:
    """find() should see commits from merged branches."""

    def test_find_content_from_merged_branch(self) -> None:
        t, feature_hash, _, _ = _setup_merged_tract()
        try:
            results = t.find(content="feature work")
            hashes = [r.commit_hash for r in results]
            assert feature_hash in hashes, (
                "find(content=) did not traverse merge parents"
            )
        finally:
            t.close()

    def test_find_content_from_main_branch(self) -> None:
        t, _, main_hash, _ = _setup_merged_tract()
        try:
            results = t.find(content="main work")
            hashes = [r.commit_hash for r in results]
            assert main_hash in hashes
        finally:
            t.close()

    def test_find_tag_from_merged_branch(self) -> None:
        t, feature_hash, _, _ = _setup_merged_tract()
        try:
            results = t.find(tag="feature-tag")
            hashes = [r.commit_hash for r in results]
            assert feature_hash in hashes, (
                "find(tag=) did not traverse merge parents"
            )
        finally:
            t.close()

    def test_find_sees_both_branches(self) -> None:
        """find() without filters should return commits from both branches."""
        t, feature_hash, main_hash, _ = _setup_merged_tract()
        try:
            # Search for all user dialogue
            results = t.find(content_type="dialogue")
            hashes = [r.commit_hash for r in results]
            assert feature_hash in hashes, "Missing feature branch commit"
            assert main_hash in hashes, "Missing main branch commit"
        finally:
            t.close()

    def test_find_one_from_merged_branch(self) -> None:
        t, feature_hash, _, _ = _setup_merged_tract()
        try:
            result = t.find_one(content="feature work")
            assert result is not None
            assert result.commit_hash == feature_hash
        finally:
            t.close()


# ---------------------------------------------------------------------------
# log() tests
# ---------------------------------------------------------------------------


class TestLogTraversesMergeParents:
    """log() should include commits from merged branches."""

    def test_log_includes_merged_branch_commits(self) -> None:
        t, feature_hash, main_hash, merge_hash = _setup_merged_tract()
        try:
            entries = t.log(limit=50)
            hashes = [e.commit_hash for e in entries]
            assert feature_hash in hashes, (
                "log() did not traverse merge parents"
            )
            assert main_hash in hashes
        finally:
            t.close()

    def test_log_order_newest_first(self) -> None:
        """Merged branch commits should appear in chronological position."""
        t, feature_hash, main_hash, merge_hash = _setup_merged_tract()
        try:
            entries = t.log(limit=50)
            hashes = [e.commit_hash for e in entries]
            # Merge commit should be first (newest)
            assert hashes[0] == merge_hash
            # Both branch commits should appear somewhere after
            assert feature_hash in hashes
            assert main_hash in hashes
        finally:
            t.close()

    def test_log_with_tag_filter_finds_merged(self) -> None:
        t, feature_hash, _, _ = _setup_merged_tract()
        try:
            entries = t.log(limit=50, tags=["feature-tag"])
            hashes = [e.commit_hash for e in entries]
            assert feature_hash in hashes, (
                "log(tags=) did not traverse merge parents"
            )
        finally:
            t.close()


# ---------------------------------------------------------------------------
# query_by_tags() tests
# ---------------------------------------------------------------------------


class TestQueryByTagsTraversesMergeParents:
    """query_by_tags() should find tagged commits from merged branches."""

    def test_query_finds_tag_from_merged_branch(self) -> None:
        t, feature_hash, _, _ = _setup_merged_tract()
        try:
            results = t._tags_mgr.query(["feature-tag"])
            hashes = [r.commit_hash for r in results]
            assert feature_hash in hashes, (
                "query_by_tags() did not traverse merge parents"
            )
        finally:
            t.close()

    def test_query_finds_tags_from_both_branches(self) -> None:
        t, feature_hash, main_hash, _ = _setup_merged_tract()
        try:
            results = t._tags_mgr.query(["feature-tag", "main-tag"])
            hashes = [r.commit_hash for r in results]
            assert feature_hash in hashes, "Missing feature branch tagged commit"
            assert main_hash in hashes, "Missing main branch tagged commit"
        finally:
            t.close()

    def test_query_match_all_from_merged_branch(self) -> None:
        """Test match='all' with a commit that has multiple tags, one from merge."""
        t = Tract.open()
        try:
            t.register_tag("alpha", "Alpha tag")
            t.register_tag("beta", "Beta tag")
            t.commit(InstructionContent(text="base"))
            t.branch("feature")
            info = t.commit(
                DialogueContent(role="user", text="tagged"),
                tags=["alpha", "beta"],
            )
            tagged_hash = info.commit_hash

            t.switch("main")
            t.commit(DialogueContent(role="user", text="main"))
            t.merge("feature", no_ff=True)

            results = t._tags_mgr.query(["alpha", "beta"], match="all")
            hashes = [r.commit_hash for r in results]
            assert tagged_hash in hashes
        finally:
            t.close()


# ---------------------------------------------------------------------------
# list_tags() tests
# ---------------------------------------------------------------------------


class TestListTagsTraversesMergeParents:
    """list_tags() should count tags from merged branches."""

    def test_list_tags_counts_merged_branch(self) -> None:
        t = Tract.open()
        try:
            t.commit(InstructionContent(text="base"))

            # Register the tag first
            t.register_tag("feature-tag", "A tag for feature work")

            t.branch("feature")
            t.commit(
                DialogueContent(role="user", text="feature"),
                tags=["feature-tag"],
            )

            t.switch("main")
            t.commit(DialogueContent(role="user", text="main"))
            t.merge("feature", no_ff=True)

            tags = t.list_tags()
            feature_tag = next(
                (tg for tg in tags if tg["name"] == "feature-tag"), None,
            )
            assert feature_tag is not None
            assert feature_tag["count"] >= 1, (
                "list_tags() did not count tags from merged branch"
            )
        finally:
            t.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestMergeTraversalEdgeCases:
    """Edge cases for merge-aware traversal."""

    def test_no_parent_repo_fallback(self) -> None:
        """When _parent_repo is None, methods should still work (linear history)."""
        t = Tract.open()
        try:
            t.commit(InstructionContent(text="msg1"))
            t.commit(DialogueContent(role="user", text="msg2"))
            entries = t.log(limit=10)
            assert len(entries) == 2
            results = t.find(content="msg1")
            assert len(results) == 1
        finally:
            t.close()

    def test_fast_forward_merge_no_duplicate(self) -> None:
        """Fast-forward merge (no merge commit) should not duplicate entries."""
        t = Tract.open()
        try:
            t.commit(InstructionContent(text="base"))
            t.branch("feature")
            t.commit(DialogueContent(role="user", text="feature"))

            t.switch("main")
            result = t.merge("feature")
            assert result.merge_type == "fast_forward"

            entries = t.log(limit=50)
            hashes = [e.commit_hash for e in entries]
            # No duplicates
            assert len(hashes) == len(set(hashes))
        finally:
            t.close()

    def test_multiple_merges(self) -> None:
        """Commits from multiple merged branches should all be visible."""
        t = Tract.open()
        try:
            t.commit(InstructionContent(text="base"))

            # First feature branch
            t.branch("feature-a")
            a_info = t.commit(
                DialogueContent(role="user", text="work from branch A"),
            )
            a_hash = a_info.commit_hash

            t.switch("main")
            t.merge("feature-a", no_ff=True)

            # Second feature branch
            t.branch("feature-b")
            b_info = t.commit(
                DialogueContent(role="user", text="work from branch B"),
            )
            b_hash = b_info.commit_hash

            t.switch("main")
            t.merge("feature-b", no_ff=True)

            results = t.find(content_type="dialogue")
            hashes = [r.commit_hash for r in results]
            assert a_hash in hashes, "Missing commit from first merged branch"
            assert b_hash in hashes, "Missing commit from second merged branch"
        finally:
            t.close()
