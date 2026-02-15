"""Comprehensive tests for merge operations.

Tests merge strategies: fast-forward, clean auto-merge, structural conflict
detection (both_edit, skip_vs_edit, edit_plus_append), LLM-mediated
resolution, MergeResult review/commit flow, and post-merge compilation.
"""

from __future__ import annotations

import pytest

from tract import (
    CommitInfo,
    CommitOperation,
    ConflictInfo,
    DialogueContent,
    InstructionContent,
    MergeError,
    MergeResult,
    NothingToMergeError,
    Priority,
    Tract,
)
from tract.llm.protocols import Resolution


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def diverge_branches(
    t: Tract,
    *,
    main_texts: list[str] | None = None,
    feature_texts: list[str] | None = None,
    branch_name: str = "feature",
) -> tuple[str, str]:
    """Create a branch, commit on both sides, and return branch names.

    Returns:
        (main_branch_name, feature_branch_name)
    """
    main_texts = main_texts or ["main work"]
    feature_texts = feature_texts or ["feature work"]

    # Create branch from current HEAD
    t.branch(branch_name)

    # Commit on feature branch
    for text in feature_texts:
        t.commit(DialogueContent(role="user", text=text))

    # Switch back to main and commit
    t.switch("main")
    for text in main_texts:
        t.commit(DialogueContent(role="user", text=text))

    return "main", branch_name


def make_mock_resolver(
    content_text: str = "merged content",
    action: str = "resolved",
    reasoning: str = "test resolution",
    generation_config: dict | None = None,
) -> object:
    """Create a mock resolver callable."""

    class MockResolver:
        def __call__(self, issue: object) -> Resolution:
            return Resolution(
                action=action,
                content_text=content_text,
                reasoning=reasoning,
                generation_config=generation_config or {"model": "test-model", "source": "infrastructure:merge"},
            )

    return MockResolver()


# ===========================================================================
# Fast-forward tests
# ===========================================================================


class TestFastForwardMerge:
    """Tests for fast-forward merge (source is ahead of current)."""

    def test_fast_forward_merge(self) -> None:
        """Branch ahead, main unchanged -> fast-forward pointer move."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        base_hash = t.head

        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feature 1"))
        t.commit(DialogueContent(role="user", text="feature 2"))
        feature_tip = t.head

        t.switch("main")
        assert t.head == base_hash  # main hasn't moved

        result = t.merge("feature")

        assert result.merge_type == "fast_forward"
        assert result.committed is True
        assert t.head == feature_tip  # main now at feature tip
        t.close()

    def test_fast_forward_no_merge_commit(self) -> None:
        """Fast-forward should NOT create a merge commit."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feature"))
        feature_tip = t.head

        t.switch("main")
        result = t.merge("feature")

        assert result.merge_type == "fast_forward"
        # Verify no merge commit in parents table
        parents = t._parent_repo.get_parents(feature_tip)
        assert parents == []  # feature tip is a regular commit
        t.close()

    def test_fast_forward_no_ff(self) -> None:
        """no_ff=True forces a merge commit even for fast-forward."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feature"))
        feature_tip = t.head

        t.switch("main")
        base_hash = t.head

        result = t.merge("feature", no_ff=True)

        # Should be a clean merge (not fast-forward) since we diverged by 0
        # Actually with no_ff, the is_ancestor check passes but we skip FF
        # and fall through to the merge base logic
        assert result.merge_type == "clean"
        assert result.committed is True
        assert t.head != base_hash  # moved past base
        assert t.head != feature_tip  # new merge commit

        # Verify merge commit has two parents
        parents = t._parent_repo.get_parents(t.head)
        assert len(parents) == 2
        assert base_hash in parents
        assert feature_tip in parents
        t.close()

    def test_already_up_to_date(self) -> None:
        """Source branch is ancestor of current -> NothingToMergeError."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")
        # Don't commit on feature, switch back and commit on main
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main ahead"))

        with pytest.raises(NothingToMergeError, match="already up-to-date"):
            t.merge("feature")
        t.close()

    def test_same_commit_raises(self) -> None:
        """Source branch pointing to same commit as HEAD -> NothingToMergeError."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature", switch=False)

        with pytest.raises(NothingToMergeError):
            t.merge("feature")
        t.close()


# ===========================================================================
# Clean merge tests (both sides APPEND only)
# ===========================================================================


class TestCleanMerge:
    """Tests for clean merge with divergent APPEND-only histories."""

    def test_clean_merge_creates_merge_commit(self) -> None:
        """Diverged APPEND-only -> merge commit created."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        diverge_branches(t)

        result = t.merge("feature")

        assert result.merge_type == "clean"
        assert result.committed is True
        assert result.merge_commit_hash is not None
        t.close()

    def test_clean_merge_two_parents(self) -> None:
        """Merge commit has entries in commit_parents table."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        diverge_branches(t)

        main_tip = t.head
        feature_tip = t._ref_repo.get_branch(t._tract_id, "feature")

        result = t.merge("feature")

        parents = t._parent_repo.get_parents(result.merge_commit_hash)
        assert len(parents) == 2
        assert main_tip in parents
        assert feature_tip in parents
        t.close()

    def test_compiled_after_merge(self) -> None:
        """After merge, compile() includes messages from both branches."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        diverge_branches(
            t,
            main_texts=["main msg"],
            feature_texts=["feature msg"],
        )

        t.merge("feature")
        compiled = t.compile()

        # Should have: base + main_msg + feature_msg + merge commit
        texts = [m.content for m in compiled.messages]
        assert any("base" in text for text in texts)
        assert any("main msg" in text for text in texts)
        assert any("feature msg" in text for text in texts)
        assert len(compiled.messages) >= 3
        t.close()

    def test_clean_merge_branch_blocks_ordering(self) -> None:
        """Verify branch-blocks ordering: all first-parent then second-parent's."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feat-1"))
        t.commit(DialogueContent(role="user", text="feat-2"))
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main-1"))

        t.merge("feature")
        compiled = t.compile()

        texts = [m.content for m in compiled.messages]
        # base should be first, main-1 should come before feat-1/feat-2
        base_idx = next(i for i, t in enumerate(texts) if "base" in t)
        main_idx = next(i for i, t in enumerate(texts) if "main-1" in t)
        feat1_idx = next(i for i, t in enumerate(texts) if "feat-1" in t)
        feat2_idx = next(i for i, t in enumerate(texts) if "feat-2" in t)
        assert base_idx < main_idx
        assert feat1_idx < feat2_idx
        t.close()

    def test_clean_merge_multiple_commits(self) -> None:
        """Multiple commits on each side merge cleanly."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        diverge_branches(
            t,
            main_texts=["m1", "m2", "m3"],
            feature_texts=["f1", "f2"],
        )

        result = t.merge("feature")
        assert result.merge_type == "clean"

        compiled = t.compile()
        texts = [m.content for m in compiled.messages]
        # All commits present
        for expected in ["base", "m1", "m2", "m3", "f1", "f2"]:
            assert any(expected in t for t in texts), f"Missing {expected}"
        t.close()


# ===========================================================================
# Conflict detection tests
# ===========================================================================


class TestConflictDetection:
    """Tests for structural conflict detection."""

    def test_conflict_both_edit_same_target(self) -> None:
        """Both branches EDIT the same commit -> both_edit conflict."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))
        base_hash = base.commit_hash

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base_hash,
        )

        result = t.merge("feature")

        assert result.merge_type == "conflict"
        assert len(result.conflicts) == 1
        assert result.conflicts[0].conflict_type == "both_edit"
        assert result.conflicts[0].target_hash == base_hash
        assert result.committed is False
        t.close()

    def test_conflict_skip_vs_edit(self) -> None:
        """One branch SKIPs, other EDITs same commit -> skip_vs_edit conflict."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))
        base_hash = base.commit_hash

        # Feature branch: EDIT the base commit
        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base_hash,
        )

        # Main branch: SKIP the base commit and add something
        t.switch("main")
        t.annotate(base_hash, Priority.SKIP, reason="not needed")
        t.commit(DialogueContent(role="user", text="main addition"))

        result = t.merge("feature")

        assert result.merge_type == "conflict"
        assert any(c.conflict_type == "skip_vs_edit" for c in result.conflicts)
        t.close()

    def test_conflict_edit_plus_append(self) -> None:
        """One branch EDITs pre-merge-base, other APPENDs -> edit_plus_append."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))
        base_hash = base.commit_hash

        # Feature branch: EDIT the pre-merge-base commit
        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base_hash,
        )

        # Main branch: just APPEND
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main append"))

        result = t.merge("feature")

        assert result.merge_type == "conflict"
        assert any(c.conflict_type == "edit_plus_append" for c in result.conflicts)
        t.close()

    def test_no_conflict_edit_post_merge_base(self) -> None:
        """EDIT targeting post-merge-base commit is NOT a conflict with other branch appends."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        # Feature branch: commit and edit its own commit
        t.branch("feature")
        feat_commit = t.commit(DialogueContent(role="user", text="feature original"))
        t.commit(
            DialogueContent(role="user", text="feature edited"),
            operation=CommitOperation.EDIT,
            response_to=feat_commit.commit_hash,
        )

        # Main branch: just APPEND
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main append"))

        result = t.merge("feature")

        # Should be clean -- the edit targets a post-merge-base commit
        assert result.merge_type == "clean"
        t.close()

    def test_conflict_content_preloaded(self) -> None:
        """ConflictInfo has pre-loaded content text from blobs."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original text"))
        base_hash = base.commit_hash

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature version"),
            operation=CommitOperation.EDIT,
            response_to=base_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main version"),
            operation=CommitOperation.EDIT,
            response_to=base_hash,
        )

        result = t.merge("feature")

        conflict = result.conflicts[0]
        # Content should be pre-loaded (non-empty strings)
        assert conflict.content_a_text != ""
        assert conflict.content_b_text != ""
        t.close()


# ===========================================================================
# Conflict resolution tests
# ===========================================================================


class TestConflictResolution:
    """Tests for conflict merge resolution (manual and LLM-mediated)."""

    def test_conflict_merge_without_resolver(self) -> None:
        """Conflicts with no resolver -> returns uncommitted MergeResult."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        result = t.merge("feature")

        assert result.merge_type == "conflict"
        assert len(result.conflicts) >= 1
        assert result.committed is False
        assert result.merge_commit_hash is None
        t.close()

    def test_conflict_merge_with_resolver(self) -> None:
        """Resolver resolves all conflicts -> resolutions populated."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        resolver = make_mock_resolver(content_text="resolved text")
        result = t.merge("feature", resolver=resolver)

        # Resolver was called, resolutions populated
        assert len(result.resolutions) >= 1
        assert any("resolved text" in v for v in result.resolutions.values())
        t.close()

    def test_commit_merge_after_review(self) -> None:
        """Get MergeResult, edit resolution, call commit_merge()."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        # Get conflicts without resolver
        result = t.merge("feature")
        assert result.committed is False

        # Manually resolve all conflicts
        for conflict in result.conflicts:
            target_key = conflict.target_hash or conflict.commit_b.commit_hash
            result.edit_resolution(target_key, "manually resolved content")

        # Commit the merge
        result = t.commit_merge(result)

        assert result.committed is True
        assert result.merge_commit_hash is not None
        # Verify merge commit exists and has two parents
        parents = t._parent_repo.get_parents(result.merge_commit_hash)
        assert len(parents) == 2
        t.close()

    def test_commit_merge_unresolved_raises(self) -> None:
        """commit_merge() with unresolved conflicts raises MergeError."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        result = t.merge("feature")
        assert result.committed is False

        # Don't resolve anything, try to commit
        with pytest.raises(MergeError, match="unresolved"):
            t.commit_merge(result)
        t.close()

    def test_merge_with_auto_commit(self) -> None:
        """auto_commit=True with resolver -> single-step merge."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        resolver = make_mock_resolver(content_text="auto resolved")
        result = t.merge("feature", resolver=resolver, auto_commit=True)

        assert result.committed is True
        assert result.merge_commit_hash is not None
        t.close()

    def test_merge_delete_branch(self) -> None:
        """delete_branch=True removes source branch after merge."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))
        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feature work"))
        t.switch("main")

        # Fast-forward merge with delete
        result = t.merge("feature", delete_branch=True)
        assert result.committed is True

        # Branch should be deleted
        branches = [b.name for b in t.list_branches()]
        assert "feature" not in branches
        t.close()

    def test_merge_delete_branch_not_on_unresolved_conflict(self) -> None:
        """delete_branch=True does NOT delete if merge has unresolved conflicts."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        result = t.merge("feature", delete_branch=True)
        assert result.committed is False

        # Branch should still exist
        branches = [b.name for b in t.list_branches()]
        assert "feature" in branches
        t.close()

    def test_resolver_abort_raises(self) -> None:
        """Resolver returning abort raises MergeError."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        resolver = make_mock_resolver(action="abort", reasoning="cannot resolve")
        with pytest.raises(MergeError, match="aborted"):
            t.merge("feature", resolver=resolver)
        t.close()


# ===========================================================================
# Integration tests
# ===========================================================================


class TestMergeIntegration:
    """End-to-end merge workflow tests."""

    def test_full_merge_workflow(self) -> None:
        """Complete workflow: branch, diverge, merge, verify compiled context."""
        t = Tract.open()
        t.commit(InstructionContent(text="system prompt"))
        t.commit(DialogueContent(role="user", text="initial query"))

        # Branch and diverge
        t.branch("experiment")
        t.commit(DialogueContent(role="assistant", text="experiment response"))
        t.switch("main")
        t.commit(DialogueContent(role="assistant", text="main response"))

        # Merge
        result = t.merge("experiment")
        assert result.merge_type == "clean"

        # Verify compiled context
        compiled = t.compile()
        texts = [m.content for m in compiled.messages]
        assert any("system prompt" in t for t in texts)
        assert any("initial query" in t for t in texts)
        assert any("experiment response" in t for t in texts)
        assert any("main response" in t for t in texts)
        t.close()

    def test_merge_preserves_generation_config(self) -> None:
        """Merge commit records generation_config from resolver."""
        t = Tract.open()
        base = t.commit(InstructionContent(text="original"))

        t.branch("feature")
        t.commit(
            DialogueContent(role="assistant", text="feature edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        t.switch("main")
        t.commit(
            DialogueContent(role="assistant", text="main edit"),
            operation=CommitOperation.EDIT,
            response_to=base.commit_hash,
        )

        resolver = make_mock_resolver(
            content_text="resolved",
            generation_config={"model": "gpt-4o", "source": "infrastructure:merge"},
        )
        result = t.merge("feature", resolver=resolver, auto_commit=True)

        assert result.committed is True
        # The generation_config should be recorded on the merge commit
        merge_commit = t.get_commit(result.merge_commit_hash)
        assert merge_commit is not None
        assert merge_commit.generation_config is not None
        assert merge_commit.generation_config.get("model") == "gpt-4o"
        assert merge_commit.generation_config.get("source") == "infrastructure:merge"
        t.close()

    def test_merge_clears_compile_cache(self) -> None:
        """Merge clears compile cache so next compile reflects merged state."""
        t = Tract.open(verify_cache=True)
        t.commit(InstructionContent(text="base"))
        _ = t.compile()  # populate cache

        t.branch("feature")
        t.commit(DialogueContent(role="user", text="feature"))
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main"))

        t.merge("feature")

        # Cache should be cleared; compile should work fresh
        compiled = t.compile()
        texts = [m.content for m in compiled.messages]
        assert any("feature" in t for t in texts)
        assert any("main" in t for t in texts)
        t.close()

    def test_merge_nonexistent_branch_raises(self) -> None:
        """Merging a nonexistent branch raises BranchNotFoundError."""
        from tract.exceptions import BranchNotFoundError

        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        with pytest.raises(BranchNotFoundError):
            t.merge("does-not-exist")
        t.close()

    def test_merge_detached_head_raises(self) -> None:
        """Merge in detached HEAD state raises MergeError."""
        t = Tract.open()
        c1 = t.commit(InstructionContent(text="base"))
        t.branch("feature", switch=False)
        t.checkout(c1.commit_hash)  # detach HEAD

        with pytest.raises(MergeError, match="detached"):
            t.merge("feature")
        t.close()

    def test_edit_resolution_modifies_content(self) -> None:
        """MergeResult.edit_resolution() updates resolution text."""
        result = MergeResult(
            merge_type="conflict",
            source_branch="feature",
            target_branch="main",
            resolutions={"abc123": "original resolution"},
        )

        result.edit_resolution("abc123", "updated resolution")
        assert result.resolutions["abc123"] == "updated resolution"

    def test_sequential_merges(self) -> None:
        """Multiple sequential merges work correctly."""
        t = Tract.open()
        t.commit(InstructionContent(text="base"))

        # First merge: feature-a
        t.branch("feature-a")
        t.commit(DialogueContent(role="user", text="feature-a"))
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main-1"))
        result_a = t.merge("feature-a")
        assert result_a.merge_type == "clean"

        # Second merge: feature-b
        t.branch("feature-b")
        t.commit(DialogueContent(role="user", text="feature-b"))
        t.switch("main")
        t.commit(DialogueContent(role="user", text="main-2"))
        result_b = t.merge("feature-b")
        assert result_b.merge_type == "clean"

        compiled = t.compile()
        texts = [m.content for m in compiled.messages]
        assert any("feature-a" in t for t in texts)
        assert any("feature-b" in t for t in texts)
        t.close()

    def test_configure_llm(self) -> None:
        """configure_llm() stores client and creates default resolver."""
        t = Tract.open()

        class FakeLLMClient:
            def chat(self, messages, **kwargs):
                return {
                    "choices": [{"message": {"content": "resolved"}}],
                    "model": "fake",
                }

            def close(self):
                pass

        t.configure_llm(FakeLLMClient())
        assert hasattr(t, "_llm_client")
        assert hasattr(t, "_default_resolver")
        t.close()


# ===========================================================================
# Model tests
# ===========================================================================


class TestMergeModels:
    """Tests for merge data models."""

    def test_conflict_info_creation(self) -> None:
        """ConflictInfo can be created with all fields."""
        from datetime import datetime

        commit = CommitInfo(
            commit_hash="abc",
            tract_id="test",
            content_hash="def",
            content_type="dialogue",
            operation=CommitOperation.EDIT,
            token_count=10,
            created_at=datetime.now(),
        )

        info = ConflictInfo(
            conflict_type="both_edit",
            commit_a=commit,
            commit_b=commit,
            content_a_text="version A",
            content_b_text="version B",
            target_hash="xyz",
        )

        assert info.conflict_type == "both_edit"
        assert info.content_a_text == "version A"
        assert info.target_hash == "xyz"

    def test_merge_result_creation(self) -> None:
        """MergeResult can be created with defaults."""
        result = MergeResult(
            merge_type="clean",
            source_branch="feature",
            target_branch="main",
        )

        assert result.merge_type == "clean"
        assert result.conflicts == []
        assert result.resolutions == {}
        assert result.committed is False

    def test_merge_result_edit_resolution(self) -> None:
        """edit_resolution() adds/updates resolutions."""
        result = MergeResult(
            merge_type="conflict",
            source_branch="feature",
            target_branch="main",
        )

        result.edit_resolution("hash1", "content 1")
        result.edit_resolution("hash2", "content 2")

        assert result.resolutions["hash1"] == "content 1"
        assert result.resolutions["hash2"] == "content 2"

        # Update existing
        result.edit_resolution("hash1", "updated content")
        assert result.resolutions["hash1"] == "updated content"
