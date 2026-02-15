"""Comprehensive tests for branch operations, DAG utilities, and Tract facade.

Tests cover:
- Branch creation (from HEAD, specific commit, with/without switch)
- Branch name validation (git-style rules)
- Branch switching (switch to branch, verify independent histories)
- Branch listing (with current flag)
- Branch deletion (current branch guard, unmerged guard, force delete)
- DAG utilities (find_merge_base, is_ancestor, get_all_ancestors, get_branch_commits)
- Compiler multi-parent support
- Integration: full branch workflow
"""

from __future__ import annotations

import pytest

from tract import (
    BranchExistsError,
    BranchInfo,
    BranchNotFoundError,
    InstructionContent,
    InvalidBranchNameError,
    Tract,
    TraceError,
    UnmergedBranchError,
)
from tract.operations.branch import validate_branch_name
from tract.operations.dag import (
    find_merge_base,
    get_all_ancestors,
    get_branch_commits,
    is_ancestor,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tract():
    """Create a Tract with one initial commit on main."""
    t = Tract.open()
    t.commit(InstructionContent(text="initial commit"), message="init")
    yield t
    t.close()


@pytest.fixture
def empty_tract():
    """Create a Tract with no commits."""
    t = Tract.open()
    yield t
    t.close()


# ---------------------------------------------------------------------------
# Branch name validation
# ---------------------------------------------------------------------------

class TestBranchNameValidation:
    def test_valid_simple_name(self):
        validate_branch_name("feature")

    def test_valid_slashed_name(self):
        validate_branch_name("feature/auth")

    def test_valid_hyphenated_name(self):
        validate_branch_name("my-feature")

    def test_empty_name_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="cannot be empty"):
            validate_branch_name("")

    def test_double_dot_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="cannot contain '\\.\\.'"):
            validate_branch_name("foo..bar")

    def test_lock_suffix_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="cannot end with '\\.lock'"):
            validate_branch_name("foo.lock")

    def test_leading_dot_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="cannot start with '\\.'"):
            validate_branch_name(".hidden")

    def test_trailing_dot_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="cannot end with '\\.'"):
            validate_branch_name("trail.")

    def test_tilde_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("feat~1")

    def test_space_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("my branch")

    def test_caret_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("feat^2")

    def test_colon_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("a:b")

    def test_question_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("what?")

    def test_asterisk_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("feat*")

    def test_bracket_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("feat[0]")

    def test_backslash_rejected(self):
        with pytest.raises(InvalidBranchNameError, match="forbidden characters"):
            validate_branch_name("a\\b")


# ---------------------------------------------------------------------------
# Branch creation
# ---------------------------------------------------------------------------

class TestBranchCreation:
    def test_create_from_head(self, tract):
        """Create a branch from HEAD, auto-switch."""
        head_before = tract.head
        result = tract.branch("feature")
        assert result == head_before
        assert tract.current_branch == "feature"

    def test_create_from_specific_commit(self, tract):
        """Create branch from a specific commit hash."""
        first_hash = tract.head
        tract.commit(InstructionContent(text="second"), message="second")
        result = tract.branch("from-first", source=first_hash, switch=False)
        assert result == first_hash
        # Should still be on main since switch=False
        assert tract.current_branch == "main"

    def test_create_with_auto_switch(self, tract):
        """Default: switch to new branch."""
        tract.branch("feature")
        assert tract.current_branch == "feature"

    def test_create_without_switch(self, tract):
        """Create branch without switching."""
        tract.branch("feature", switch=False)
        assert tract.current_branch == "main"

    def test_duplicate_name_error(self, tract):
        """Duplicate branch name raises BranchExistsError."""
        tract.branch("feature", switch=False)
        with pytest.raises(BranchExistsError, match="feature"):
            tract.branch("feature")

    def test_create_main_duplicate_error(self, tract):
        """Cannot create 'main' when it already exists."""
        with pytest.raises(BranchExistsError, match="main"):
            tract.branch("main")

    def test_create_no_commits_error(self, empty_tract):
        """Cannot create branch with no commits."""
        with pytest.raises(TraceError, match="no commits"):
            empty_tract.branch("feature")

    def test_invalid_name_error(self, tract):
        """Invalid branch name raises InvalidBranchNameError."""
        with pytest.raises(InvalidBranchNameError):
            tract.branch("feat..bar")


# ---------------------------------------------------------------------------
# Branch switching
# ---------------------------------------------------------------------------

class TestBranchSwitching:
    def test_switch_to_branch(self, tract):
        """Switch to an existing branch."""
        tract.branch("feature")
        tract.switch("main")
        assert tract.current_branch == "main"

    def test_switch_back_and_forth(self, tract):
        """Switch between branches maintains state."""
        tract.branch("feature")
        tract.switch("main")
        tract.switch("feature")
        assert tract.current_branch == "feature"

    def test_switch_nonexistent_raises(self, tract):
        """Switching to nonexistent branch raises BranchNotFoundError."""
        with pytest.raises(BranchNotFoundError, match="nonexistent"):
            tract.switch("nonexistent")

    def test_switch_rejects_commit_hash(self, tract):
        """switch() only accepts branch names, not commit hashes."""
        commit_hash = tract.head
        with pytest.raises(BranchNotFoundError):
            tract.switch(commit_hash)

    def test_independent_histories(self, tract):
        """Commits on different branches produce different HEADs."""
        initial_head = tract.head

        # Create feature branch and commit
        tract.branch("feature")
        tract.commit(InstructionContent(text="on feature"), message="feature commit")
        feature_head = tract.head

        # Switch to main and commit
        tract.switch("main")
        assert tract.head == initial_head  # Main hasn't moved
        tract.commit(InstructionContent(text="on main"), message="main commit")
        main_head = tract.head

        # Both branches have different heads
        assert main_head != feature_head
        assert main_head != initial_head

        # Switch back to feature -- head is still feature's head
        tract.switch("feature")
        assert tract.head == feature_head


# ---------------------------------------------------------------------------
# Branch listing
# ---------------------------------------------------------------------------

class TestBranchListing:
    def test_list_single_branch(self, tract):
        """Only main branch after initial commit."""
        branches = tract.list_branches()
        assert len(branches) == 1
        assert branches[0].name == "main"
        assert branches[0].is_current is True

    def test_list_multiple_branches(self, tract):
        """List all branches with current flag."""
        tract.branch("feature", switch=False)
        tract.branch("develop", switch=False)

        branches = tract.list_branches()
        names = {b.name for b in branches}
        assert names == {"main", "feature", "develop"}

        # main is current
        current = [b for b in branches if b.is_current]
        assert len(current) == 1
        assert current[0].name == "main"

    def test_list_branches_current_after_switch(self, tract):
        """Current branch flag updates after switch."""
        tract.branch("feature")  # auto-switch
        branches = tract.list_branches()
        current = [b for b in branches if b.is_current]
        assert len(current) == 1
        assert current[0].name == "feature"

    def test_list_empty_before_commits(self, empty_tract):
        """No branches before any commits."""
        branches = empty_tract.list_branches()
        assert branches == []

    def test_branch_info_has_commit_hash(self, tract):
        """BranchInfo includes the commit hash."""
        branches = tract.list_branches()
        assert branches[0].commit_hash == tract.head


# ---------------------------------------------------------------------------
# Branch deletion
# ---------------------------------------------------------------------------

class TestBranchDeletion:
    def test_delete_non_current_branch(self, tract):
        """Can delete a branch that is not current."""
        tract.branch("feature", switch=False)
        tract.delete_branch("feature")
        branches = tract.list_branches()
        names = {b.name for b in branches}
        assert "feature" not in names

    def test_delete_current_branch_blocked(self, tract):
        """Cannot delete the current branch."""
        tract.branch("feature")  # switches to feature
        with pytest.raises(TraceError, match="Cannot delete the current branch"):
            tract.delete_branch("feature")

    def test_delete_nonexistent_raises(self, tract):
        """Deleting nonexistent branch raises BranchNotFoundError."""
        with pytest.raises(BranchNotFoundError, match="ghost"):
            tract.delete_branch("ghost")

    def test_delete_merged_branch(self, tract):
        """Delete a branch whose tip is reachable from current (merged)."""
        # Branch from initial commit -- same tip as main
        tract.branch("feature", switch=False)
        # feature tip == main tip, so it's "merged"
        tract.delete_branch("feature")
        branches = tract.list_branches()
        assert len(branches) == 1

    def test_delete_unmerged_branch_blocked(self, tract):
        """Block deletion of branch with unmerged commits."""
        tract.branch("feature")  # switch to feature
        tract.commit(InstructionContent(text="unmerged work"), message="unmerged")
        tract.switch("main")

        with pytest.raises(UnmergedBranchError, match="feature"):
            tract.delete_branch("feature")

    def test_force_delete_unmerged(self, tract):
        """Force delete bypasses unmerged check."""
        tract.branch("feature")
        tract.commit(InstructionContent(text="unmerged work"), message="unmerged")
        tract.switch("main")

        tract.delete_branch("feature", force=True)
        branches = tract.list_branches()
        names = {b.name for b in branches}
        assert "feature" not in names


# ---------------------------------------------------------------------------
# DAG utilities
# ---------------------------------------------------------------------------

class TestFindMergeBase:
    def test_linear_history_merge_base(self, tract):
        """Merge base of linear commits returns the earlier one."""
        first = tract.head
        tract.commit(InstructionContent(text="second"), message="second")
        second = tract.head

        result = find_merge_base(
            tract._commit_repo, tract._parent_repo, first, second
        )
        assert result == first

    def test_merge_base_same_commit(self, tract):
        """Merge base of a commit with itself is itself."""
        head = tract.head
        result = find_merge_base(
            tract._commit_repo, tract._parent_repo, head, head
        )
        assert result == head

    def test_diverged_branches_merge_base(self, tract):
        """Merge base for diverged branches returns the fork point."""
        fork_point = tract.head

        # Commit on main
        tract.commit(InstructionContent(text="main work"), message="main work")
        main_head = tract.head

        # Create feature from fork point and commit
        tract.branch("feature", source=fork_point)
        tract.commit(InstructionContent(text="feature work"), message="feature work")
        feature_head = tract.head

        result = find_merge_base(
            tract._commit_repo, tract._parent_repo, main_head, feature_head
        )
        assert result == fork_point

    def test_merge_base_no_common_ancestor(self):
        """Two unrelated tracts have no merge base (returns None)."""
        t = Tract.open()
        t.commit(InstructionContent(text="a"), message="a")
        head_a = t.head

        # Create a completely separate commit with a different tract_id
        # This can't happen in practice within one tract, but tests the edge case
        t2 = Tract.open()
        t2.commit(InstructionContent(text="b"), message="b")
        head_b = t2.head

        # In separate tracts, merge base is None
        result = find_merge_base(t._commit_repo, t._parent_repo, head_a, head_b)
        assert result is None

        t.close()
        t2.close()


class TestIsAncestor:
    def test_direct_ancestor(self, tract):
        """Parent is ancestor of child."""
        parent = tract.head
        tract.commit(InstructionContent(text="child"), message="child")
        child = tract.head

        assert is_ancestor(
            tract._commit_repo, tract._parent_repo, parent, child
        ) is True

    def test_self_is_ancestor(self, tract):
        """A commit is its own ancestor."""
        head = tract.head
        assert is_ancestor(
            tract._commit_repo, tract._parent_repo, head, head
        ) is True

    def test_non_ancestor(self, tract):
        """Child is not ancestor of parent."""
        parent = tract.head
        tract.commit(InstructionContent(text="child"), message="child")
        child = tract.head

        assert is_ancestor(
            tract._commit_repo, tract._parent_repo, child, parent
        ) is False

    def test_diverged_not_ancestor(self, tract):
        """Sibling branch tips are not ancestors of each other."""
        fork = tract.head

        tract.commit(InstructionContent(text="main"), message="main")
        main_head = tract.head

        tract.branch("feature", source=fork)
        tract.commit(InstructionContent(text="feature"), message="feature")
        feature_head = tract.head

        assert is_ancestor(
            tract._commit_repo, tract._parent_repo, main_head, feature_head
        ) is False
        assert is_ancestor(
            tract._commit_repo, tract._parent_repo, feature_head, main_head
        ) is False


class TestGetAllAncestors:
    def test_single_commit(self, tract):
        """Single commit has itself as only ancestor."""
        head = tract.head
        ancestors = get_all_ancestors(head, tract._commit_repo, tract._parent_repo)
        assert ancestors == {head}

    def test_linear_chain(self, tract):
        """Linear chain returns all commits."""
        first = tract.head
        tract.commit(InstructionContent(text="second"), message="second")
        second = tract.head
        tract.commit(InstructionContent(text="third"), message="third")
        third = tract.head

        ancestors = get_all_ancestors(third, tract._commit_repo, tract._parent_repo)
        assert ancestors == {first, second, third}


class TestGetBranchCommits:
    def test_single_branch_commit(self, tract):
        """Get single commit between merge base and branch tip."""
        base = tract.head
        tract.commit(InstructionContent(text="on branch"), message="branch")
        tip = tract.head

        commits = get_branch_commits(
            tract._commit_repo, tract._parent_repo, tip, base
        )
        assert len(commits) == 1
        assert commits[0].commit_hash == tip

    def test_multiple_branch_commits(self, tract):
        """Get multiple commits in chronological order."""
        base = tract.head
        tract.commit(InstructionContent(text="first"), message="first")
        first = tract.head
        tract.commit(InstructionContent(text="second"), message="second")
        second = tract.head

        commits = get_branch_commits(
            tract._commit_repo, tract._parent_repo, second, base
        )
        assert len(commits) == 2
        assert commits[0].commit_hash == first  # chronological order
        assert commits[1].commit_hash == second

    def test_empty_range(self, tract):
        """Same commit as base and tip returns empty list."""
        base = tract.head
        commits = get_branch_commits(
            tract._commit_repo, tract._parent_repo, base, base
        )
        assert commits == []


# ---------------------------------------------------------------------------
# Compiler multi-parent support
# ---------------------------------------------------------------------------

class TestCompilerMultiParent:
    def test_compiler_handles_merge_commit(self, tract):
        """Compiler walks second parent's unique commits for merge commits."""
        # Set up: fork, commit on both branches, then manually create a merge commit
        fork = tract.head

        # Commit on main
        tract.commit(InstructionContent(text="main-only"), message="main work")
        main_head = tract.head

        # Create feature from fork and commit
        tract.branch("feature", source=fork)
        tract.commit(InstructionContent(text="feature-only"), message="feature work")
        feature_head = tract.head

        # Switch back to main
        tract.switch("main")

        # Manually create a merge commit that points at both parents
        # First, create the merge commit content
        merge_info = tract.commit(
            InstructionContent(text="merge commit"),
            message="merge feature into main",
        )

        # Record the multi-parent relationship
        tract._parent_repo.add_parents(
            merge_info.commit_hash,
            [main_head, feature_head],
        )
        tract._session.commit()

        # Now compile -- should include feature-only commit
        result = tract.compile()

        # Check that messages include content from both branches
        contents = [m.content for m in result.messages]
        assert "initial commit" in " ".join(contents) or any("initial" in c for c in contents)
        assert any("feature-only" in c for c in contents)
        assert any("main-only" in c for c in contents)
        assert any("merge commit" in c for c in contents)


# ---------------------------------------------------------------------------
# Integration: Full branch workflow
# ---------------------------------------------------------------------------

class TestBranchIntegration:
    def test_full_branch_workflow(self):
        """Full workflow: create, commit independently, verify fork point."""
        t = Tract.open()

        # Initial commit
        t.commit(InstructionContent(text="base"), message="base")
        base_hash = t.head

        # Create feature branch
        t.branch("feature")
        assert t.current_branch == "feature"
        t.commit(InstructionContent(text="on feature"), message="feat 1")
        feature_head = t.head

        # Switch to main and commit
        t.switch("main")
        assert t.current_branch == "main"
        assert t.head == base_hash  # Main unchanged
        t.commit(InstructionContent(text="on main"), message="main 1")
        main_head = t.head

        # Verify independent heads
        assert main_head != feature_head
        assert main_head != base_hash

        # Verify merge base is the fork point
        merge_base = find_merge_base(
            t._commit_repo, t._parent_repo, main_head, feature_head
        )
        assert merge_base == base_hash

        # List branches
        branches = t.list_branches()
        assert len(branches) == 2
        names = {b.name for b in branches}
        assert names == {"main", "feature"}

        # Current is main
        current = [b for b in branches if b.is_current]
        assert len(current) == 1
        assert current[0].name == "main"

        t.close()

    def test_multiple_branches_from_same_point(self):
        """Multiple branches from the same commit."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"), message="base")
        base = t.head

        t.branch("feature-a", switch=False)
        t.branch("feature-b", switch=False)
        t.branch("feature-c", switch=False)

        branches = t.list_branches()
        assert len(branches) == 4  # main + 3 features

        # All point at base
        for b in branches:
            assert b.commit_hash == base

        t.close()

    def test_compile_after_branch_switch(self):
        """Compile gives correct context after switching branches."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"), message="base")

        # Feature branch with its own commit
        t.branch("feature")
        t.commit(InstructionContent(text="feature content"), message="feat")

        # Compile on feature should include both
        feature_result = t.compile()
        assert len(feature_result.messages) == 2

        # Switch to main -- compile should only have base
        t.switch("main")
        main_result = t.compile()
        assert len(main_result.messages) == 1

        t.close()

    def test_delete_after_switch_back(self):
        """Delete a branch after switching away from it."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"), message="base")

        t.branch("temp")
        t.switch("main")
        t.delete_branch("temp")

        branches = t.list_branches()
        assert len(branches) == 1
        assert branches[0].name == "main"

        t.close()

    def test_smoke_test_from_plan(self):
        """The exact smoke test from the plan's verification section."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")  # Creates and switches
        t.commit(InstructionContent(text="on feature"))
        t.switch("main")
        t.commit(InstructionContent(text="on main"))
        assert t.head != t._ref_repo.get_branch(t.tract_id, "feature")
        branches = t.list_branches()
        assert len(branches) == 2
        t.close()
