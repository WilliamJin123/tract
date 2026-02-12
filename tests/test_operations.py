"""Tests for read operations -- log (enhanced), status, diff.

Tests the three read-oriented SDK operations that let users inspect
their linear commit history through the Tract facade.
"""

from __future__ import annotations

import pytest

from tract import (
    CommitNotFoundError,
    CommitOperation,
    DialogueContent,
    DiffResult,
    DiffStat,
    InstructionContent,
    MessageDiff,
    StatusInfo,
    Tract,
    TractConfig,
    TokenBudgetConfig,
    TraceError,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_tract(**kwargs) -> Tract:
    """Create an in-memory Tract for testing."""
    return Tract.open(":memory:", **kwargs)


def populate_tract(t: Tract, n: int = 3) -> list[str]:
    """Commit n dialogue messages and return their hashes."""
    hashes = []
    for i in range(n):
        if i == 0:
            info = t.commit(InstructionContent(text=f"System prompt {i}"))
        else:
            info = t.commit(DialogueContent(role="user", text=f"Message {i}"))
        hashes.append(info.commit_hash)
    return hashes


# ==================================================================
# Log enhancement tests
# ==================================================================

class TestLogEnhanced:
    """Tests for enhanced Tract.log() with op_filter and new default limit."""

    def test_log_default_limit_20(self):
        """Default limit is now 20, not 10."""
        t = make_tract()
        # Create 25 commits
        hashes = []
        for i in range(25):
            info = t.commit(DialogueContent(role="user", text=f"Msg {i}"))
            hashes.append(info.commit_hash)
        result = t.log()
        assert len(result) == 20

    def test_log_with_op_filter_append(self):
        """op_filter=APPEND returns only APPEND commits."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h2 = t.commit(DialogueContent(role="assistant", text="Hi")).commit_hash
        # Edit h1
        t.commit(
            DialogueContent(role="user", text="Hello edited"),
            operation=CommitOperation.EDIT,
            response_to=h1,
        )
        result = t.log(op_filter=CommitOperation.APPEND)
        ops = [c.operation for c in result]
        assert all(op == CommitOperation.APPEND for op in ops)
        assert len(result) == 2  # h1 and h2

    def test_log_with_op_filter_edit(self):
        """op_filter=EDIT returns only EDIT commits."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        t.commit(DialogueContent(role="assistant", text="Hi"))
        t.commit(
            DialogueContent(role="user", text="Hello edited"),
            operation=CommitOperation.EDIT,
            response_to=h1,
        )
        result = t.log(op_filter=CommitOperation.EDIT)
        assert len(result) == 1
        assert result[0].operation == CommitOperation.EDIT

    def test_log_empty_tract(self):
        """Log on empty tract returns empty list."""
        t = make_tract()
        assert t.log() == []

    def test_log_op_filter_no_matches(self):
        """op_filter with no matching commits returns empty list."""
        t = make_tract()
        t.commit(DialogueContent(role="user", text="Hello"))
        result = t.log(op_filter=CommitOperation.EDIT)
        assert result == []


# ==================================================================
# Status tests
# ==================================================================

class TestStatus:
    """Tests for Tract.status() method."""

    def test_status_empty_tract(self):
        """Status on empty tract returns defaults."""
        t = make_tract()
        info = t.status()
        assert isinstance(info, StatusInfo)
        assert info.head_hash is None
        assert info.commit_count == 0
        assert info.token_count == 0
        assert info.is_detached is False
        assert info.recent_commits == []

    def test_status_with_commits(self):
        """Status shows correct head, branch, and commit count."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        info = t.status()
        assert info.head_hash == hashes[-1]
        assert info.branch_name == "main"
        assert info.is_detached is False
        assert info.commit_count == 3

    def test_status_detached_head(self):
        """After checkout to specific commit, is_detached=True."""
        t = make_tract()
        hashes = populate_tract(t, 3)
        t.checkout(hashes[0])
        info = t.status()
        assert info.is_detached is True
        assert info.branch_name is None
        assert info.head_hash == hashes[0]

    def test_status_with_token_budget(self):
        """Status includes token_budget_max when configured."""
        config = TractConfig(
            token_budget=TokenBudgetConfig(max_tokens=5000),
        )
        t = Tract.open(":memory:", config=config)
        populate_tract(t, 1)
        info = t.status()
        assert info.token_budget_max == 5000

    def test_status_without_budget(self):
        """Status has token_budget_max=None when no budget configured."""
        t = make_tract()
        populate_tract(t, 1)
        info = t.status()
        assert info.token_budget_max is None

    def test_status_recent_commits(self):
        """Status returns last 3 commits in recent_commits."""
        t = make_tract()
        hashes = populate_tract(t, 5)
        info = t.status()
        assert len(info.recent_commits) == 3
        # Newest first
        assert info.recent_commits[0].commit_hash == hashes[4]
        assert info.recent_commits[1].commit_hash == hashes[3]
        assert info.recent_commits[2].commit_hash == hashes[2]

    def test_status_fewer_than_3_commits(self):
        """If only 1 commit, recent_commits has 1 entry."""
        t = make_tract()
        hashes = populate_tract(t, 1)
        info = t.status()
        assert len(info.recent_commits) == 1
        assert info.recent_commits[0].commit_hash == hashes[0]

    def test_status_token_count_is_compiled(self):
        """Token count in status matches compile() output."""
        t = make_tract()
        populate_tract(t, 3)
        compiled = t.compile()
        info = t.status()
        assert info.token_count == compiled.token_count
        assert info.token_source == compiled.token_source


# ==================================================================
# Diff tests
# ==================================================================

class TestDiff:
    """Tests for Tract.diff() method."""

    def test_diff_two_commits(self):
        """Diff between two commits shows added message."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h2 = t.commit(DialogueContent(role="assistant", text="World")).commit_hash
        result = t.diff(h1, h2)
        assert isinstance(result, DiffResult)
        assert result.commit_a == h1
        assert result.commit_b == h2
        # h2 has one more message than h1
        added = [d for d in result.message_diffs if d.status == "added"]
        assert len(added) == 1
        assert added[0].role_b == "assistant"

    def test_diff_against_head_default(self):
        """Diff with only commit_a specified, commit_b defaults to HEAD."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h2 = t.commit(DialogueContent(role="assistant", text="World")).commit_hash
        result = t.diff(commit_a=h1)
        assert result.commit_b == h2  # HEAD

    def test_diff_edit_auto_resolve(self):
        """diff(commit_b=edit_hash) auto-resolves to diff against original."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        edit_info = t.commit(
            DialogueContent(role="user", text="Hello edited"),
            operation=CommitOperation.EDIT,
            response_to=h1,
        )
        # diff with commit_b=edit commit should auto-resolve commit_a to h1's content
        result = t.diff(commit_b=edit_info.commit_hash)
        # The edit replaces h1's content, so we should see a modification
        assert result.commit_a == h1
        # There should be exactly 1 message diff (the edited message)
        assert len(result.message_diffs) >= 1

    def test_diff_first_commit_vs_empty(self):
        """Diff with no commit_a and commit_b is first commit shows all added."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        result = t.diff(commit_b=h1)
        assert result.commit_a == "(empty)"
        added = [d for d in result.message_diffs if d.status == "added"]
        assert len(added) == 1

    def test_diff_identical_commits(self):
        """Diff A vs A returns all unchanged."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        result = t.diff(h1, h1)
        unchanged = [d for d in result.message_diffs if d.status == "unchanged"]
        assert len(unchanged) == 1
        assert result.stat.messages_unchanged == 1
        assert result.stat.messages_added == 0
        assert result.stat.messages_removed == 0
        assert result.stat.messages_modified == 0

    def test_diff_modified_message(self):
        """Edit changes content, showing modified with unified diff lines."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello world")).commit_hash
        edit_info = t.commit(
            DialogueContent(role="user", text="Hello universe"),
            operation=CommitOperation.EDIT,
            response_to=h1,
        )
        # Diff the original content vs the edit
        # compile at h1 gives original; compile at edit gives the edited version
        result = t.diff(h1, edit_info.commit_hash)
        modified = [d for d in result.message_diffs if d.status == "modified"]
        assert len(modified) == 1
        assert len(modified[0].content_diff_lines) > 0  # Has unified diff

    def test_diff_role_change(self):
        """Edit changes role of a message. MessageDiff shows role_a != role_b."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        edit_info = t.commit(
            DialogueContent(role="assistant", text="Hello"),
            operation=CommitOperation.EDIT,
            response_to=h1,
        )
        result = t.diff(h1, edit_info.commit_hash)
        modified = [d for d in result.message_diffs if d.status == "modified"]
        assert len(modified) == 1
        assert modified[0].role_a == "user"
        assert modified[0].role_b == "assistant"

    def test_diff_generation_config_change(self):
        """Two commits with different generation_config show config changes."""
        t = make_tract()
        t.commit(
            DialogueContent(role="user", text="Hello"),
            generation_config={"temperature": 0.5, "model": "gpt-4"},
        )
        t.commit(
            DialogueContent(role="assistant", text="Hi"),
            generation_config={"temperature": 0.9, "model": "gpt-4"},
        )
        hashes = [c.commit_hash for c in t.log()]
        # hashes[0] is latest (assistant), hashes[1] is oldest (user)
        result = t.diff(hashes[1], hashes[0])
        assert "temperature" in result.generation_config_changes
        assert result.generation_config_changes["temperature"] == (0.5, 0.9)
        # model didn't change, so should not be in changes
        assert "model" not in result.generation_config_changes

    def test_diff_stat(self):
        """DiffStat counts are correct."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h2 = t.commit(DialogueContent(role="assistant", text="World")).commit_hash
        result = t.diff(h1, h2)
        # h1 has 1 message, h2 has 2. The first is unchanged, second is added.
        assert result.stat.messages_unchanged == 1
        assert result.stat.messages_added == 1
        assert result.stat.messages_removed == 0
        assert result.stat.messages_modified == 0

    def test_diff_by_prefix(self):
        """Use short prefix to reference commits in diff."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h2 = t.commit(DialogueContent(role="assistant", text="World")).commit_hash
        # Use first 8 chars as prefix
        result = t.diff(h1[:8], h2[:8])
        assert result.commit_a == h1
        assert result.commit_b == h2

    def test_diff_no_commits_raises(self):
        """Diff on empty tract raises TraceError."""
        t = make_tract()
        with pytest.raises(TraceError, match="No commits to diff"):
            t.diff()

    def test_diff_invalid_commit_raises(self):
        """Diff with nonexistent hash raises CommitNotFoundError."""
        t = make_tract()
        t.commit(DialogueContent(role="user", text="Hello"))
        with pytest.raises(CommitNotFoundError):
            t.diff("nonexistent_commit_hash_that_does_not_exist")

    def test_diff_parent_auto_resolve(self):
        """When commit_a is None and commit_b is not EDIT, uses parent."""
        t = make_tract()
        h1 = t.commit(DialogueContent(role="user", text="Hello")).commit_hash
        h2 = t.commit(DialogueContent(role="assistant", text="World")).commit_hash
        # diff(commit_b=h2) with no commit_a should use h2's parent (h1)
        result = t.diff(commit_b=h2)
        assert result.commit_a == h1
        assert result.commit_b == h2

    def test_diff_result_message_diff_indices(self):
        """MessageDiff indices are sequential starting from 0."""
        t = make_tract()
        t.commit(DialogueContent(role="user", text="A"))
        t.commit(DialogueContent(role="assistant", text="B"))
        h3 = t.commit(DialogueContent(role="user", text="C")).commit_hash
        # Diff first commit vs h3 (3 messages vs 1 message)
        hashes = [c.commit_hash for c in t.log()]
        result = t.diff(hashes[-1], h3)  # oldest vs newest
        indices = [d.index for d in result.message_diffs]
        assert indices == list(range(len(indices)))
