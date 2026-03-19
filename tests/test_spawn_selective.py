"""Tests for selective inheritance mode in spawn operations.

Covers tag filtering, type filtering, custom filter_func, instruction
preservation, EDIT commit handling, and edge cases.
"""

from __future__ import annotations

import pytest

from tract import (
    DialogueContent,
    InstructionContent,
    Session,
    SpawnInfo,
)
from tract.models.content import (
    ArtifactContent,
    ConfigContent,
    ReasoningContent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session_and_parent(tmp_path, *, db_name="test.db"):
    """Create a session with a fresh parent tract."""
    db_path = str(tmp_path / db_name)
    session = Session.open(db_path)
    parent = session.create_tract(display_name="parent")
    return session, parent


# ---------------------------------------------------------------------------
# Tag filter tests
# ---------------------------------------------------------------------------


class TestSelectiveTagFilter:
    """Selective spawn with include_tags / exclude_tags."""

    def test_include_tags_filters_commits(self, tmp_path):
        """Only commits with a matching tag are inherited."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"), tags=["system"])
        parent.commit(
            DialogueContent(role="user", text="important question"),
            tags=["important"],
        )
        parent.commit(
            DialogueContent(role="assistant", text="important answer"),
            tags=["important"],
        )
        parent.commit(
            DialogueContent(role="user", text="throwaway chat"),
            tags=["casual"],
        )

        child = session.spawn(
            parent,
            purpose="important only",
            inheritance="selective",
            include_tags=["important"],
        )

        compiled = child.compile()
        # Should have: instruction (auto-included) + 2 important commits = 3
        assert compiled.commit_count == 3
        texts = [m.content for m in compiled.messages]
        assert any("important question" in t for t in texts)
        assert any("important answer" in t for t in texts)
        assert not any("throwaway" in t for t in texts)

        session.close()

    def test_exclude_tags_filters_commits(self, tmp_path):
        """Commits with excluded tags are not inherited."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"), tags=["system"])
        parent.commit(
            DialogueContent(role="user", text="keep me"),
            tags=["keep"],
        )
        parent.commit(
            DialogueContent(role="user", text="drop me"),
            tags=["temporary"],
        )

        child = session.spawn(
            parent,
            purpose="exclude temp",
            inheritance="selective",
            exclude_tags=["temporary"],
        )

        compiled = child.compile()
        texts = [m.content for m in compiled.messages]
        assert any("keep me" in t for t in texts)
        assert not any("drop me" in t for t in texts)

        session.close()

    def test_include_and_exclude_tags_combined(self, tmp_path):
        """include_tags and exclude_tags work together."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(
            DialogueContent(role="user", text="A"),
            tags=["keep", "debug"],
        )
        parent.commit(
            DialogueContent(role="user", text="B"),
            tags=["keep"],
        )
        parent.commit(
            DialogueContent(role="user", text="C"),
            tags=["other"],
        )

        child = session.spawn(
            parent,
            purpose="combined filter",
            inheritance="selective",
            include_tags=["keep"],
            exclude_tags=["debug"],
        )

        compiled = child.compile()
        texts = [m.content for m in compiled.messages]
        # A has keep but also debug -> excluded
        # B has keep, no debug -> included
        # C has other, not keep -> excluded
        assert any("B" == t for t in texts)
        assert not any("A" == t for t in texts)
        assert not any("C" == t for t in texts)

        session.close()


# ---------------------------------------------------------------------------
# Type filter tests
# ---------------------------------------------------------------------------


class TestSelectiveTypeFilter:
    """Selective spawn with include_types."""

    def test_include_types_filters_by_content_type(self, tmp_path):
        """Only commits of the specified content types are inherited."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(DialogueContent(role="user", text="hello"))
        parent.commit(
            ArtifactContent(artifact_type="code", content="print('hi')")
        )
        parent.commit(ReasoningContent(text="thinking..."))

        child = session.spawn(
            parent,
            purpose="dialogue only",
            inheritance="selective",
            include_types=["dialogue"],
        )

        compiled = child.compile()
        # instruction (auto-included) + dialogue = 2
        assert compiled.commit_count == 2
        types = {m.role for m in compiled.messages}
        # instruction -> system role, dialogue -> user role
        assert "system" in types
        assert "user" in types

        session.close()

    def test_include_types_multiple(self, tmp_path):
        """Multiple content types can be included."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(DialogueContent(role="user", text="msg"))
        parent.commit(
            ArtifactContent(artifact_type="code", content="x = 1")
        )
        parent.commit(ReasoningContent(text="thought"))

        child = session.spawn(
            parent,
            purpose="dialogue+artifact",
            inheritance="selective",
            include_types=["dialogue", "artifact"],
        )

        compiled = child.compile()
        # instruction (auto) + dialogue + artifact = 3
        assert compiled.commit_count == 3

        session.close()


# ---------------------------------------------------------------------------
# Custom filter_func tests
# ---------------------------------------------------------------------------


class TestSelectiveCustomFilter:
    """Selective spawn with a custom filter_func."""

    def test_custom_filter_func(self, tmp_path):
        """Custom filter_func controls which commits are included."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(
            DialogueContent(role="user", text="short"),
        )
        parent.commit(
            DialogueContent(role="user", text="this is a much longer message for testing"),
        )

        # Filter: only include commits where the message field is set
        # and content_type is dialogue
        def my_filter(commit_row):
            return (
                commit_row.content_type == "dialogue"
                and commit_row.message is not None
            )

        child = session.spawn(
            parent,
            purpose="custom filter",
            inheritance="selective",
            filter_func=my_filter,
        )

        compiled = child.compile()
        # instruction (auto-included) + all dialogue commits (they all have messages)
        assert compiled.commit_count >= 2

        session.close()

    def test_filter_func_receives_commit_row_attributes(self, tmp_path):
        """filter_func can access content_type, tags_json, operation, etc."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(
            DialogueContent(role="user", text="tagged"),
            tags=["special"],
            message="special commit",
        )
        parent.commit(
            DialogueContent(role="user", text="untagged"),
        )

        seen_attrs = []

        def inspector(commit_row):
            seen_attrs.append({
                "content_type": commit_row.content_type,
                "tags_json": commit_row.tags_json,
                "operation": str(commit_row.operation),
                "has_message": commit_row.message is not None,
            })
            return True  # include all

        child = session.spawn(
            parent,
            purpose="inspect",
            inheritance="selective",
            filter_func=inspector,
        )

        # The filter was called for each commit
        assert len(seen_attrs) >= 2
        # At least one should have tags
        tagged = [a for a in seen_attrs if a["tags_json"]]
        assert len(tagged) >= 1
        assert "special" in tagged[0]["tags_json"]

        session.close()


# ---------------------------------------------------------------------------
# Instruction preservation tests
# ---------------------------------------------------------------------------


class TestSelectiveInstructionPreservation:
    """Instruction and config commits bypass filter by default."""

    def test_instructions_always_included(self, tmp_path):
        """Instruction commits survive even when filter rejects them."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="You must follow rules"))
        parent.commit(
            DialogueContent(role="user", text="hello"),
            tags=["chat"],
        )

        # Filter that rejects everything
        child = session.spawn(
            parent,
            purpose="reject all",
            inheritance="selective",
            filter_func=lambda _: False,
        )

        compiled = child.compile()
        # Only instructions survive
        assert compiled.commit_count == 1
        assert "rules" in compiled.messages[0].content

        session.close()

    def test_config_commits_always_included(self, tmp_path):
        """Config commits are always included when include_instructions=True."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(ConfigContent(settings={"temperature": 0.5}))
        parent.commit(
            DialogueContent(role="user", text="chat"),
            tags=["chat"],
        )

        child = session.spawn(
            parent,
            purpose="config preserved",
            inheritance="selective",
            filter_func=lambda _: False,
        )

        # instruction + config = 2 (config is not compilable, but is committed)
        log = child.log(limit=10)
        types = {c.content_type for c in log}
        assert "instruction" in types
        assert "config" in types

        session.close()

    def test_include_instructions_false_skips_them(self, tmp_path):
        """When include_instructions=False, instructions are also filtered."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(
            DialogueContent(role="user", text="hello"),
            tags=["keep"],
        )

        child = session.spawn(
            parent,
            purpose="no auto instructions",
            inheritance="selective",
            include_tags=["keep"],
            include_instructions=False,
        )

        compiled = child.compile()
        # Only the tagged dialogue commit
        assert compiled.commit_count == 1
        assert "hello" in compiled.messages[0].content

        session.close()


# ---------------------------------------------------------------------------
# EDIT commit handling
# ---------------------------------------------------------------------------


class TestSelectiveEditHandling:
    """EDIT commits whose targets are filtered out should be skipped."""

    def test_edit_skipped_when_target_filtered(self, tmp_path):
        """An EDIT commit is dropped if its edit_target was not included."""
        from tract.models.commit import CommitOperation

        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        info_orig = parent.commit(
            DialogueContent(role="user", text="original msg"),
            tags=["v1"],
        )
        # Create an EDIT commit targeting the original
        parent.commit(
            DialogueContent(role="user", text="edited msg"),
            operation=CommitOperation.EDIT,
            edit_target=info_orig.commit_hash,
            message="fix typo",
        )

        # Filter: only include commits tagged "v1"
        # The edit has no "v1" tag, so it gets filtered by include_tags.
        # Even if we tried to include it, its edit_target points at the
        # *parent's* hash which won't exist in the child.
        child = session.spawn(
            parent,
            purpose="no edits",
            inheritance="selective",
            include_tags=["v1"],
        )

        compiled = child.compile()
        texts = [m.content for m in compiled.messages]
        # Should have instruction + original, no edit
        assert any("original msg" in t for t in texts)

        session.close()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestSelectiveEdgeCases:
    """Edge cases for selective inheritance."""

    def test_empty_filter_returns_only_instructions(self, tmp_path):
        """Filter that rejects everything yields only instruction/config commits."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="Always be helpful"))
        parent.commit(DialogueContent(role="user", text="msg1"))
        parent.commit(DialogueContent(role="user", text="msg2"))
        parent.commit(DialogueContent(role="assistant", text="reply"))

        child = session.spawn(
            parent,
            purpose="empty filter",
            inheritance="selective",
            filter_func=lambda _: False,
        )

        compiled = child.compile()
        assert compiled.commit_count == 1  # only the instruction
        assert "helpful" in compiled.messages[0].content

        session.close()

    def test_error_when_no_filter_criteria(self, tmp_path):
        """Selective mode without any filter raises ValueError."""
        session, parent = _session_and_parent(tmp_path)
        parent.commit(InstructionContent(text="System"))

        with pytest.raises(ValueError, match="selective inheritance requires"):
            session.spawn(
                parent,
                purpose="no filter",
                inheritance="selective",
            )

        session.close()

    def test_selective_from_empty_parent(self, tmp_path):
        """Selective spawn from empty parent creates empty child."""
        session, parent = _session_and_parent(tmp_path)

        child = session.spawn(
            parent,
            purpose="empty parent",
            inheritance="selective",
            filter_func=lambda _: True,
        )

        compiled = child.compile()
        assert compiled.commit_count == 0

        session.close()

    def test_selective_spawn_pointer_records_mode(self, tmp_path):
        """Spawn pointer stores 'selective' as inheritance_mode."""
        session, parent = _session_and_parent(tmp_path)
        parent.commit(InstructionContent(text="System"))
        parent.commit(
            DialogueContent(role="user", text="msg"),
            tags=["keep"],
        )

        child = session.spawn(
            parent,
            purpose="check pointer",
            inheritance="selective",
            include_tags=["keep"],
        )

        children = parent.spawn_children()
        assert len(children) == 1
        info = children[0]
        assert isinstance(info, SpawnInfo)
        assert info.inheritance_mode == "selective"
        assert info.child_tract_id == child.tract_id

        session.close()

    def test_selective_preserves_annotations(self, tmp_path):
        """Annotations on included commits are copied to the child."""
        from tract import Priority

        session, parent = _session_and_parent(tmp_path)

        info = parent.commit(
            DialogueContent(role="user", text="annotated"),
            tags=["keep"],
        )
        parent.annotate(info.commit_hash, Priority.IMPORTANT, reason="key data")

        child = session.spawn(
            parent,
            purpose="with annotations",
            inheritance="selective",
            include_tags=["keep"],
            include_instructions=False,
        )

        child_log = child.log(limit=10)
        assert len(child_log) == 1
        child_annotations = child.get_annotation(child_log[0].commit_hash)
        important = [a for a in child_annotations if a.priority == Priority.IMPORTANT]
        assert len(important) >= 1

        session.close()

    def test_selective_all_included_matches_full_clone(self, tmp_path):
        """filter_func returning True for all commits behaves like full_clone."""
        session, parent = _session_and_parent(tmp_path)

        parent.commit(InstructionContent(text="System prompt"))
        parent.commit(DialogueContent(role="user", text="msg1"))
        parent.commit(DialogueContent(role="assistant", text="reply"))

        child = session.spawn(
            parent,
            purpose="include all",
            inheritance="selective",
            filter_func=lambda _: True,
        )

        compiled = child.compile()
        # Should have all 3 commits
        assert compiled.commit_count == 3

        session.close()
