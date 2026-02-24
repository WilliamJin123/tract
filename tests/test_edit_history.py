"""Tests for edit_history() and restore() on Tract.

Validates the full edit chain retrieval and version restoration features.
"""

from __future__ import annotations

import time

import pytest

from tract import (
    CommitInfo,
    CommitOperation,
    DialogueContent,
    InstructionContent,
    Tract,
)
from tract.exceptions import CommitNotFoundError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tract():
    """In-memory tract, cleaned up after test."""
    t = Tract.open()
    yield t
    t.close()


# ---------------------------------------------------------------------------
# edit_history() tests
# ---------------------------------------------------------------------------


class TestEditHistory:
    """Tests for Tract.edit_history()."""

    def test_no_edits_returns_single_element(self, tract: Tract):
        """A commit with no edits returns a one-element list."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        history = tract.edit_history(c1.commit_hash)
        assert len(history) == 1
        assert history[0].commit_hash == c1.commit_hash
        assert history[0].operation == CommitOperation.APPEND

    def test_single_edit(self, tract: Tract):
        """One edit returns [original, edit]."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        time.sleep(0.01)  # ensure distinct created_at
        e1 = tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        history = tract.edit_history(c1.commit_hash)
        assert len(history) == 2
        assert history[0].commit_hash == c1.commit_hash
        assert history[1].commit_hash == e1.commit_hash
        assert history[0].operation == CommitOperation.APPEND
        assert history[1].operation == CommitOperation.EDIT

    def test_multiple_edits_chronological(self, tract: Tract):
        """Multiple edits are returned in chronological order."""
        c1 = tract.commit(InstructionContent(text="v0"), message="orig")
        time.sleep(0.01)
        e1 = tract.commit(
            InstructionContent(text="v1"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        time.sleep(0.01)
        e2 = tract.commit(
            InstructionContent(text="v2"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 2",
        )
        history = tract.edit_history(c1.commit_hash)
        assert len(history) == 3
        assert history[0].commit_hash == c1.commit_hash
        assert history[1].commit_hash == e1.commit_hash
        assert history[2].commit_hash == e2.commit_hash
        # Verify chronological order
        assert history[0].created_at <= history[1].created_at
        assert history[1].created_at <= history[2].created_at

    def test_prefix_matching(self, tract: Tract):
        """edit_history accepts a hash prefix."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        prefix = c1.commit_hash[:8]
        history = tract.edit_history(prefix)
        assert len(history) == 2
        assert history[0].commit_hash == c1.commit_hash

    def test_lookup_by_edit_hash(self, tract: Tract):
        """edit_history works when given an edit's hash (follows to original)."""
        c1 = tract.commit(InstructionContent(text="v0"), message="orig")
        time.sleep(0.01)
        e1 = tract.commit(
            InstructionContent(text="v1"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        # Look up using the edit's hash
        history = tract.edit_history(e1.commit_hash)
        assert len(history) == 2
        assert history[0].commit_hash == c1.commit_hash
        assert history[1].commit_hash == e1.commit_hash

    def test_commit_not_found(self, tract: Tract):
        """Raises CommitNotFoundError for a nonexistent hash."""
        with pytest.raises(CommitNotFoundError):
            tract.edit_history("deadbeefdeadbeef" * 4)

    def test_returns_commit_info_models(self, tract: Tract):
        """All elements are CommitInfo instances."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        history = tract.edit_history(c1.commit_hash)
        for item in history:
            assert isinstance(item, CommitInfo)

    def test_other_commits_not_included(self, tract: Tract):
        """edit_history only returns the target commit and its edits."""
        c1 = tract.commit(InstructionContent(text="first"), message="c1")
        c2 = tract.commit(
            DialogueContent(role="user", text="hello"), message="c2"
        )
        tract.commit(
            InstructionContent(text="first edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit c1",
        )
        history = tract.edit_history(c1.commit_hash)
        hashes = {h.commit_hash for h in history}
        assert c2.commit_hash not in hashes


# ---------------------------------------------------------------------------
# restore() tests
# ---------------------------------------------------------------------------


class TestRestore:
    """Tests for Tract.restore()."""

    def test_restore_to_original(self, tract: Tract):
        """restore(version=0) creates an EDIT with the original content."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="edited v1"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        restored = tract.restore(c1.commit_hash, version=0)
        assert restored.operation == CommitOperation.EDIT
        assert restored.edit_target == c1.commit_hash
        assert restored.message == "restore to version 0"

        # Verify the content matches the original
        compiled = tract.compile()
        assert "original" in compiled.messages[0].content

    def test_restore_to_specific_version(self, tract: Tract):
        """restore(version=1) creates an EDIT with the first edit's content."""
        c1 = tract.commit(InstructionContent(text="v0"), message="orig")
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="v1"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="v2"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 2",
        )
        restored = tract.restore(c1.commit_hash, version=1)
        assert restored.operation == CommitOperation.EDIT
        assert restored.edit_target == c1.commit_hash

        # Verify the content matches v1
        compiled = tract.compile()
        assert "v1" in compiled.messages[0].content

    def test_restore_custom_message(self, tract: Tract):
        """restore() uses custom message when provided."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        restored = tract.restore(c1.commit_hash, message="rollback")
        assert restored.message == "rollback"

    def test_restore_creates_proper_edit(self, tract: Tract):
        """The restored commit is a proper EDIT pointing to the original."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        restored = tract.restore(c1.commit_hash)

        # The new edit should appear in edit_history
        history = tract.edit_history(c1.commit_hash)
        assert len(history) == 3  # original + edit + restore
        assert history[-1].commit_hash == restored.commit_hash
        assert history[-1].edit_target == c1.commit_hash

    def test_restore_version_out_of_range(self, tract: Tract):
        """Raises IndexError for invalid version number."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        with pytest.raises(IndexError, match="out of range"):
            tract.restore(c1.commit_hash, version=5)

    def test_restore_negative_version(self, tract: Tract):
        """Raises IndexError for negative version."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        with pytest.raises(IndexError, match="out of range"):
            tract.restore(c1.commit_hash, version=-1)

    def test_restore_preserves_generation_config(self, tract: Tract):
        """restore() preserves the generation_config from the source version."""
        c1 = tract.commit(
            InstructionContent(text="original"),
            message="orig",
            generation_config={"model": "gpt-4", "temperature": 0.7},
        )
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
            generation_config={"model": "gpt-3.5", "temperature": 0.9},
        )
        # Restore to version 0 (original with gpt-4 config)
        restored = tract.restore(c1.commit_hash, version=0)
        assert restored.generation_config is not None
        assert restored.generation_config.model == "gpt-4"
        assert restored.generation_config.temperature == 0.7

    def test_restore_with_prefix(self, tract: Tract):
        """restore() works with hash prefixes."""
        c1 = tract.commit(InstructionContent(text="original"), message="orig")
        time.sleep(0.01)
        tract.commit(
            InstructionContent(text="edited"),
            operation=CommitOperation.EDIT,
            edit_target=c1.commit_hash,
            message="edit 1",
        )
        prefix = c1.commit_hash[:8]
        restored = tract.restore(prefix)
        assert restored.edit_target == c1.commit_hash

    def test_restore_commit_not_found(self, tract: Tract):
        """Raises CommitNotFoundError for nonexistent hash."""
        with pytest.raises(CommitNotFoundError):
            tract.restore("deadbeefdeadbeef" * 4)
