"""Comprehensive tests for rebase and import-commit operations.

Tests import-commit (APPEND, EDIT, missing target, resolver), rebase
(simple, content preservation, branch pointer update, safety checks),
and integration scenarios (import-commit then compile, rebase then merge,
full branching workflow).
"""

from __future__ import annotations

import pytest

from tract import (
    ImportCommitError,
    ImportResult,
    CommitInfo,
    CommitOperation,
    DialogueContent,
    InstructionContent,
    RebaseError,
    RebaseResult,
    RebaseWarning,
    SemanticSafetyError,
    Tract,
)
from tract.llm.protocols import Resolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setup_diverged_branches(
    t: Tract,
    *,
    main_texts: list[str] | None = None,
    feature_texts: list[str] | None = None,
    branch_name: str = "feature",
) -> tuple[str, list[CommitInfo], list[CommitInfo]]:
    """Create a branch, commit on both sides, return branch info.

    Creates: base commit -> branch point
    Then: feature commits and main commits (diverged).

    Returns:
        (branch_name, main_commits, feature_commits)
    """
    main_texts = main_texts or ["main work"]
    feature_texts = feature_texts or ["feature 1", "feature 2"]

    # Create branch from current HEAD
    t.branch(branch_name)

    # Commit on feature branch
    feature_commits = []
    for text in feature_texts:
        info = t.commit(DialogueContent(role="user", text=text))
        feature_commits.append(info)

    # Switch back to main and commit
    t.switch("main")
    main_commits = []
    for text in main_texts:
        info = t.commit(DialogueContent(role="user", text=text))
        main_commits.append(info)

    return branch_name, main_commits, feature_commits


def make_approve_resolver():
    """Create a resolver that always approves (resolved action)."""
    def resolver(issue):
        return Resolution(
            action="resolved",
            content_text="resolved content",
            reasoning="auto-approved",
        )
    return resolver


def make_abort_resolver():
    """Create a resolver that always aborts."""
    def resolver(issue):
        return Resolution(
            action="abort",
            reasoning="user chose to abort",
        )
    return resolver


def make_skip_resolver():
    """Create a resolver that always skips."""
    def resolver(issue):
        return Resolution(
            action="skip",
            reasoning="user chose to skip",
        )
    return resolver


# ---------------------------------------------------------------------------
# Import-commit tests
# ---------------------------------------------------------------------------


class TestImportCommitAppend:
    """Import-commit APPEND commit tests."""

    def test_import_commit_append_commit(self):
        """Import-commit an APPEND commit from feature to main."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(t)

        # Import-commit first feature commit onto main
        feat1 = feature_commits[0]
        result = t.import_commit(feat1.commit_hash)

        assert result.new_commit is not None
        assert result.original_commit is not None
        assert result.new_commit.commit_hash != feat1.commit_hash
        assert result.new_commit.parent_hash == main_commits[-1].commit_hash
        assert len(result.issues) == 0
        t.close()

    def test_import_commit_preserves_content(self):
        """Import-commited commit has identical blob content."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(
            t, feature_texts=["unique feature text"]
        )

        feat1 = feature_commits[0]
        result = t.import_commit(feat1.commit_hash)

        # Content hash should be the same (same blob)
        assert result.new_commit is not None
        assert result.new_commit.content_hash == feat1.content_hash
        t.close()

    def test_import_commit_preserves_metadata(self):
        """Import-commit preserves message, metadata, generation_config."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")

        feat1 = t.commit(
            DialogueContent(role="user", text="feature work"),
            message="important commit",
            metadata={"source": "test"},
            generation_config={"temperature": 0.7},
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        result = t.import_commit(feat1.commit_hash)

        assert result.new_commit is not None
        assert result.new_commit.message == "important commit"
        assert result.new_commit.metadata == {"source": "test"}
        from tract.models.config import LLMConfig
        assert result.new_commit.generation_config == LLMConfig(temperature=0.7)
        t.close()

    def test_import_commit_from_commit_hash(self):
        """Import-commit by full commit hash (not just branch tip)."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(
            t, feature_texts=["feat 1", "feat 2", "feat 3"]
        )

        # Import-commit the second commit (not tip)
        feat2 = feature_commits[1]
        result = t.import_commit(feat2.commit_hash)

        assert result.new_commit is not None
        assert result.new_commit.content_hash == feat2.content_hash
        t.close()


class TestImportCommitEdit:
    """Import-commit EDIT commit tests."""

    def test_import_commit_edit_with_target_on_branch(self):
        """Import-commit EDIT commit where edit_target exists on current branch."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="base instruction"))

        t.branch("feature")
        # Edit the base instruction (which exists on both branches)
        edit = t.commit(
            InstructionContent(text="edited instruction"),
            operation=CommitOperation.EDIT,
            edit_target=base.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        # Import-commit the edit -- base commit exists on main
        result = t.import_commit(edit.commit_hash)

        assert result.new_commit is not None
        assert len(result.issues) == 0
        assert result.new_commit.operation == CommitOperation.EDIT
        assert result.new_commit.edit_target == base.commit_hash
        t.close()

    def test_import_commit_edit_missing_target(self):
        """Import-commit EDIT with missing target raises ImportCommitError."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        # Create a commit only on feature branch
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        # Edit it
        edit = t.commit(
            DialogueContent(role="user", text="edited feature only"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        # Import-commit the edit -- its target doesn't exist on main
        with pytest.raises(ImportCommitError, match="issue"):
            t.import_commit(edit.commit_hash)
        t.close()

    def test_import_commit_edit_missing_target_with_resolver(self):
        """Import-commit EDIT with missing target succeeds via resolver."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        edit = t.commit(
            DialogueContent(role="user", text="edited feature only"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        result = t.import_commit(edit.commit_hash, resolver=make_approve_resolver())

        assert result.new_commit is not None
        assert len(result.issues) == 1
        assert result.issues[0].issue_type == "edit_target_missing"
        t.close()

    def test_import_commit_edit_missing_target_skip(self):
        """Import-commit EDIT with missing target -- resolver skips."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        edit = t.commit(
            DialogueContent(role="user", text="edited feature only"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        result = t.import_commit(edit.commit_hash, resolver=make_skip_resolver())

        assert result.new_commit is None  # Skipped
        assert len(result.issues) == 1
        t.close()


# ---------------------------------------------------------------------------
# Rebase tests
# ---------------------------------------------------------------------------


class TestRebaseSimple:
    """Basic rebase functionality tests."""

    def test_rebase_simple(self):
        """Rebase feature branch with 2 commits onto main."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(t)

        # Switch to feature and rebase onto main
        t.switch("feature")
        result = t.rebase("main")

        assert len(result.replayed_commits) == 2
        assert len(result.original_commits) == 2
        assert result.new_head is not None

        # Verify new commits have different hashes
        for orig, replayed in zip(result.original_commits, result.replayed_commits):
            assert replayed.commit_hash != orig.commit_hash

        # Verify feature is now ahead of main (main commits are ancestors)
        from tract.operations.dag import is_ancestor
        assert is_ancestor(
            t._commit_repo, t._parent_repo,
            main_commits[-1].commit_hash, result.new_head
        )
        t.close()

    def test_rebase_preserves_content(self):
        """After rebase, compile() has same feature content."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(
            t, feature_texts=["feat A", "feat B"]
        )

        # Compile feature before rebase
        t.switch("feature")
        before = t.compile()
        before_texts = [m.content for m in before.messages]

        # Rebase
        result = t.rebase("main")

        # Compile after rebase
        after = t.compile()
        after_texts = [m.content for m in after.messages]

        # Feature content should still be present
        assert "feat A" in " ".join(after_texts)
        assert "feat B" in " ".join(after_texts)
        # Main content should also be present (now an ancestor)
        assert "main work" in " ".join(after_texts)
        t.close()

    def test_rebase_updates_branch_pointer(self):
        """After rebase, feature branch ref points at last replayed commit."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(t)

        t.switch("feature")
        result = t.rebase("main")

        # Feature branch should point at the new HEAD
        feature_hash = t._ref_repo.get_branch(t._tract_id, "feature")
        assert feature_hash == result.new_head
        assert feature_hash == result.replayed_commits[-1].commit_hash
        t.close()

    def test_rebase_noop_already_ahead(self):
        """Rebase onto target when current is already ahead is noop."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feature work"))

        # Feature is ahead of main (main hasn't diverged), rebase is noop
        result = t.rebase("main")

        # Should be a noop -- no replayed commits
        assert len(result.replayed_commits) == 0
        t.close()


class TestRebaseSafetyChecks:
    """Rebase semantic safety check tests."""

    def test_rebase_with_edit_missing_target(self):
        """Rebase with EDIT targeting commit not on target raises error."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        t.commit(
            DialogueContent(role="user", text="edited feature only"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        t.switch("feature")
        with pytest.raises(SemanticSafetyError, match="safety warning"):
            t.rebase("main")
        t.close()

    def test_rebase_with_resolver(self):
        """Rebase with EDIT missing target succeeds via resolver."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        t.commit(
            DialogueContent(role="user", text="edited feature only"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        t.switch("feature")
        result = t.rebase("main", resolver=make_approve_resolver())

        assert len(result.replayed_commits) > 0
        assert len(result.warnings) > 0
        assert result.warnings[0].warning_type == "edit_target_missing"
        t.close()

    def test_rebase_abort_on_safety(self):
        """Rebase with resolver that aborts raises RebaseError."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        t.commit(
            DialogueContent(role="user", text="edited feature only"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        t.switch("feature")
        with pytest.raises(RebaseError, match="abort"):
            t.rebase("main", resolver=make_abort_resolver())
        t.close()

    def test_safety_check_blocks_without_resolver(self):
        """Operation with semantic issue and no resolver raises error."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        t.commit(
            DialogueContent(role="user", text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main"))

        t.switch("feature")
        # No resolver -- should block
        with pytest.raises(SemanticSafetyError):
            t.rebase("main")
        t.close()

    def test_safety_check_passes_with_resolver(self):
        """Operation with semantic issue and resolver succeeds."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        t.branch("feature")
        feature_only = t.commit(DialogueContent(role="user", text="feature only"))
        t.commit(
            DialogueContent(role="user", text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=feature_only.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main"))

        t.switch("feature")
        result = t.rebase("main", resolver=make_approve_resolver())
        assert result.new_head is not None
        t.close()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining import-commit, rebase, merge, and compile."""

    def test_import_commit_then_compile(self):
        """Import-commit commit, verify compile() includes the new content."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(
            t, feature_texts=["cherry target"]
        )

        # Import-commit feature commit onto main
        result = t.import_commit(feature_commits[0].commit_hash)

        # Compile main should include the imported content
        compiled = t.compile()
        texts = [m.content for m in compiled.messages]
        assert any("cherry target" in text for text in texts)
        t.close()

    def test_rebase_then_merge(self):
        """Rebase feature onto main, then merge (should fast-forward)."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(t)

        # Rebase feature onto main
        t.switch("feature")
        rebase_result = t.rebase("main")
        assert len(rebase_result.replayed_commits) == 2

        # Now merge feature into main -- should fast-forward
        t.switch("main")
        merge_result = t.merge("feature")
        assert merge_result.merge_type == "fast_forward"
        t.close()

    def test_full_branching_workflow(self):
        """Create branch, diverge, import-commit, rebase, merge."""
        t = Tract.open()
        t.commit(InstructionContent(text="system prompt"))

        # Create feature branch and diverge
        t.branch("feature")
        f1 = t.commit(DialogueContent(role="user", text="feature work 1"))
        f2 = t.commit(DialogueContent(role="user", text="feature work 2"))

        t.switch("main")
        m1 = t.commit(DialogueContent(role="user", text="main work"))

        # Import-commit f1 onto main
        cp = t.import_commit(f1.commit_hash)
        assert cp.new_commit is not None

        # Rebase feature onto main (now main has base + main_work + imported f1)
        t.switch("feature")
        rebase_result = t.rebase("main")
        assert len(rebase_result.replayed_commits) >= 1

        # Merge feature into main (should fast-forward)
        t.switch("main")
        merge_result = t.merge("feature")
        assert merge_result.merge_type == "fast_forward"

        # Final compile should have all content
        compiled = t.compile()
        texts = " ".join(m.content for m in compiled.messages)
        assert "system prompt" in texts
        assert "main work" in texts
        assert "feature work 2" in texts
        t.close()

    def test_rebase_append_only_no_warnings(self):
        """Rebase with only APPEND commits produces no warnings."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(t)

        t.switch("feature")
        result = t.rebase("main")

        assert len(result.warnings) == 0
        assert len(result.replayed_commits) == 2
        t.close()

    def test_rebase_edit_target_on_shared_history(self):
        """Rebase with EDIT targeting shared commit (on target branch) succeeds."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="base"))

        t.branch("feature")
        # Edit the base commit (which is shared/on both branches)
        t.commit(
            InstructionContent(text="edited base"),
            operation=CommitOperation.EDIT,
            edit_target=base.commit_hash,
        )

        t.switch("main")
        t.commit(DialogueContent(role="user", text="main work"))

        t.switch("feature")
        # base commit is on the target branch, so no warnings expected
        result = t.rebase("main")
        assert len(result.warnings) == 0
        assert len(result.replayed_commits) == 1
        t.close()

    def test_import_commit_result_types(self):
        """Verify ImportResult fields are correct types."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(
            t, feature_texts=["feat"]
        )

        result = t.import_commit(feature_commits[0].commit_hash)

        assert isinstance(result, ImportResult)
        assert isinstance(result.original_commit, CommitInfo)
        assert isinstance(result.new_commit, CommitInfo)
        assert isinstance(result.issues, list)
        t.close()

    def test_rebase_result_types(self):
        """Verify RebaseResult fields are correct types."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        branch_name, main_commits, feature_commits = setup_diverged_branches(t)

        t.switch("feature")
        result = t.rebase("main")

        assert isinstance(result, RebaseResult)
        assert isinstance(result.replayed_commits, list)
        assert isinstance(result.original_commits, list)
        assert isinstance(result.new_head, str)
        for commit in result.replayed_commits:
            assert isinstance(commit, CommitInfo)
        t.close()

    def test_rebase_detached_head_error(self):
        """Rebase in detached HEAD state raises RebaseError."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="base"))

        # Detach HEAD
        t.checkout(base.commit_hash)

        with pytest.raises(RebaseError, match="detached"):
            t.rebase("main")
        t.close()

    def test_import_commit_nonexistent_commit(self):
        """Import-commit nonexistent commit raises error."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        with pytest.raises(Exception):  # CommitNotFoundError or ImportCommitError
            t.import_commit("0000dead" * 8)
        t.close()
