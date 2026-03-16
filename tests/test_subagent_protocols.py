"""Tests for subagent communication and delegation protocols.

Tests cover:
- send_message: inter-agent messaging
- inherit_tools: tool delegation from parent to child on spawn
- context_budget: limiting inherited tokens on spawn
- Tract-level API integration (t.send_to_child)
"""

from __future__ import annotations

import pytest

from tract import (
    DialogueContent,
    InstructionContent,
    Session,
    SessionError,
)


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
# send_message tests
# ---------------------------------------------------------------------------


class TestSendMessage:
    """Tests for Session.send_message()."""

    def test_send_message_creates_commit(self, tmp_path):
        """send_message creates a commit in the target tract."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="worker")

        commit_hash = session.send_message(
            parent.tract_id,
            child.tract_id,
            "Please focus on task A.",
        )
        assert commit_hash is not None
        assert isinstance(commit_hash, str)
        assert len(commit_hash) > 0

    def test_send_message_metadata(self, tmp_path):
        """send_message stores correct metadata on the commit."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="worker")

        session.send_message(
            parent.tract_id,
            child.tract_id,
            "Focus on task A.",
        )

        # Find the message in child's log
        log = child.log(limit=5)
        # The most recent commit should be the agent message
        found = False
        for entry in log:
            if entry.metadata and entry.metadata.get("message_type") == "agent_message":
                assert entry.metadata["from_tract_id"] == parent.tract_id
                found = True
                break
        assert found, "agent_message commit not found in child's log"

    def test_send_message_with_tags(self, tmp_path):
        """send_message applies custom tags in addition to agent_message."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="worker")

        session.send_message(
            parent.tract_id,
            child.tract_id,
            "Priority update.",
            tags=["urgent", "priority"],
        )

        # Check that the commit has the expected tags
        log = child.log(limit=1)
        assert len(log) == 1
        tags = child.get_tags(log[0].commit_hash)
        assert "agent_message" in tags
        assert "urgent" in tags
        assert "priority" in tags


# ---------------------------------------------------------------------------
# inherit_tools tests
# ---------------------------------------------------------------------------


class TestInheritTools:
    """Tests for inherit_tools parameter on spawn."""

    def test_inherit_tools_copies_parent_tools(self, tmp_path):
        """inherit_tools=True copies parent's active tools to child."""
        session, parent = _create_session_with_parent(tmp_path)

        # Set tools on parent
        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}},
            {"type": "function", "function": {"name": "calculate", "parameters": {}}},
        ]
        parent.set_tools(tools)

        child = session.spawn(
            parent, purpose="tool user", inherit_tools=True
        )

        child_tools = child.get_tools()
        assert child_tools is not None
        assert len(child_tools) == 2
        names = {t["function"]["name"] for t in child_tools}
        assert names == {"search", "calculate"}

    def test_inherit_tools_false_by_default(self, tmp_path):
        """By default, tools are not inherited."""
        session, parent = _create_session_with_parent(tmp_path)

        tools = [
            {"type": "function", "function": {"name": "search", "parameters": {}}},
        ]
        parent.set_tools(tools)

        child = session.spawn(parent, purpose="no tools")

        child_tools = child.get_tools()
        assert child_tools is None

    def test_inherit_tools_no_parent_tools(self, tmp_path):
        """inherit_tools=True with no parent tools sets child tools to None."""
        session, parent = _create_session_with_parent(tmp_path)
        # Parent has no tools set
        assert parent.get_tools() is None

        child = session.spawn(
            parent, purpose="task", inherit_tools=True
        )

        # Child should also have no tools (parent had None)
        assert child.get_tools() is None


# ---------------------------------------------------------------------------
# context_budget tests
# ---------------------------------------------------------------------------


class TestContextBudget:
    """Tests for context_budget parameter on spawn."""

    def test_context_budget_limits_head_snapshot(self, tmp_path):
        """context_budget limits tokens in head_snapshot inheritance."""
        session, parent = _create_session_with_parent(tmp_path, n_commits=5)

        # Spawn with a very small budget
        child = session.spawn(
            parent,
            purpose="budget test",
            context_budget=50,
        )

        compiled = child.compile()
        # The inherited context should be truncated
        assert compiled.token_count <= 100  # Allow some overhead

    def test_context_budget_limits_selective(self, tmp_path):
        """context_budget drops oldest non-instruction commits in selective mode."""
        db_path = str(tmp_path / "test.db")
        session = Session.open(db_path)
        parent = session.create_tract(display_name="parent")

        # Create an instruction commit (should be preserved)
        parent.commit(InstructionContent(text="System: be helpful"))

        # Create several dialogue commits with known sizes
        for i in range(10):
            parent.commit(
                DialogueContent(
                    role="user",
                    text=f"This is a longer message number {i} with some content to use tokens.",
                ),
                tags=["dialogue"],
            )

        # Get total tokens in parent
        parent_compiled = parent.compile()
        parent_tokens = parent_compiled.token_count

        # Spawn with selective mode and a tight budget
        # Use a budget that is less than total but enough for instruction + a few
        budget = parent_tokens // 3
        child = session.spawn(
            parent,
            purpose="budget selective test",
            inheritance="selective",
            include_tags=["dialogue"],
            include_instructions=True,
            context_budget=budget,
        )

        child_compiled = child.compile()
        # Child should have fewer commits than parent
        assert child_compiled.commit_count < parent_compiled.commit_count
        # Child should respect the budget (approximately)
        # The instruction commit is always included, so token count may
        # slightly exceed budget if instruction alone is large
        assert child_compiled.token_count <= budget + 200  # Allow margin for instruction

        session.close()

    def test_context_budget_none_no_effect(self, tmp_path):
        """context_budget=None (default) does not limit inheritance."""
        session, parent = _create_session_with_parent(tmp_path, n_commits=5)

        child = session.spawn(
            parent,
            purpose="no budget",
            context_budget=None,
        )

        compiled = child.compile()
        assert compiled.commit_count >= 1  # At least the inherited snapshot


# ---------------------------------------------------------------------------
# Tract-level API integration tests
# ---------------------------------------------------------------------------


class TestTractLevelAPI:
    """Tests for Tract.send_to_child()."""

    def test_tract_send_to_child(self, tmp_path):
        """t.send_to_child() creates message in child tract."""
        session, parent = _create_session_with_parent(tmp_path)
        child = session.spawn(parent, purpose="send test")

        commit_hash = parent.send_to_child(
            child.tract_id, "Instructions for you."
        )
        assert commit_hash is not None

        # Verify the message is in child via find()
        messages = child.find(tag="agent_message")
        assert len(messages) == 1
        assert messages[0].metadata["from_tract_id"] == parent.tract_id

    def test_tract_send_to_child_no_session_raises(self, tmp_path):
        """t.send_to_child() without session raises SessionError."""
        from tract import Tract

        t = Tract.open()
        t.commit(InstructionContent(text="standalone"))
        with pytest.raises(SessionError):
            t.send_to_child("some-child-id", "hello")
        t.close()
