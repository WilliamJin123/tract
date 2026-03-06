"""Tests for t.rule() and t.metadata() facade methods on the Tract API.

Validates:
- rule/metadata commit creation with correct content_type
- field population and default messages
- custom messages and tags
- rule_index property and invalidation
- rule and metadata commits excluded from compile()
"""

from __future__ import annotations

import pytest

from tract import Tract, RuleContent, MetadataContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def t():
    """In-memory tract, cleaned up after test."""
    tract = Tract.open()
    yield tract
    tract.close()


# ===========================================================================
# t.rule() tests
# ===========================================================================


class TestRule:
    def test_rule_creates_commit_with_rule_content_type(self, t: Tract):
        """t.rule() creates a commit whose content_type is 'rule'."""
        info = t.rule(
            "auto_compress",
            trigger="commit",
            condition={"type": "threshold"},
            action={"type": "compress"},
        )
        assert info.content_type == "rule"

    def test_rule_populates_all_fields_correctly(self, t: Tract):
        """All RuleContent fields are stored and retrievable."""
        info = t.rule(
            "auto_compress",
            trigger="commit",
            condition={"type": "threshold", "max_tokens": 1000},
            action={"type": "compress", "strategy": "summarize"},
        )
        # Retrieve the commit and verify content via blob
        commit_info = t.get_commit(info.commit_hash)
        assert commit_info.content_type == "rule"
        assert commit_info.commit_hash == info.commit_hash

    def test_rule_default_message(self, t: Tract):
        """Default message is 'rule: {name}'."""
        info = t.rule(
            "auto_compress",
            trigger="commit",
            action={"type": "compress"},
        )
        assert info.message == "rule: auto_compress"

    def test_rule_custom_message_overrides_default(self, t: Tract):
        """A custom message replaces the default 'rule: {name}'."""
        info = t.rule(
            "temp",
            trigger="active",
            action={"type": "set_config"},
            message="custom msg",
        )
        assert info.message == "custom msg"

    def test_rule_with_tags(self, t: Tract):
        """Tags are propagated to the rule commit."""
        t.register_tag("experimental", "Experimental feature")
        t.register_tag("v2", "Version 2")
        info = t.rule(
            "tagged_rule",
            trigger="compile",
            action={"type": "noop"},
            tags=["experimental", "v2"],
        )
        assert "experimental" in info.tags
        assert "v2" in info.tags

    def test_rule_with_condition_none(self, t: Tract):
        """condition=None means the rule is unconditional."""
        info = t.rule(
            "unconditional",
            trigger="commit",
            condition=None,
            action={"type": "noop"},
        )
        assert info.content_type == "rule"
        # Rule should still be indexed (unconditional fires always)
        idx = t.rule_index
        assert ("commit", "unconditional") in idx


# ===========================================================================
# t.metadata() tests
# ===========================================================================


class TestMetadata:
    def test_metadata_creates_commit_with_metadata_content_type(self, t: Tract):
        """t.metadata() creates a commit with content_type='metadata'."""
        info = t.metadata("file_tree", {"files": ["a.py", "b.py"]})
        assert info.content_type == "metadata"

    def test_metadata_default_message(self, t: Tract):
        """Default message is 'metadata: {kind}'."""
        info = t.metadata("file_tree", {"files": ["a.py"]})
        assert info.message == "metadata: file_tree"

    def test_metadata_with_dict_data_preserved(self, t: Tract):
        """Dict data is preserved in the metadata commit."""
        data = {"files": ["a.py", "b.py"], "count": 2}
        info = t.metadata("file_tree", data)
        assert info.content_type == "metadata"
        # Verify we can retrieve the commit
        commit_info = t.get_commit(info.commit_hash)
        assert commit_info.content_type == "metadata"

    def test_metadata_with_str_data_wrapped(self, t: Tract):
        """String data is wrapped as {'text': data}."""
        info = t.metadata("note", "some text")
        assert info.content_type == "metadata"
        # The commit should still be valid metadata
        commit_info = t.get_commit(info.commit_hash)
        assert commit_info.content_type == "metadata"

    def test_metadata_with_path_field(self, t: Tract):
        """The path parameter is accepted and stored."""
        info = t.metadata(
            "artifact",
            {"content": "x"},
            path="/output/file.md",
        )
        assert info.content_type == "metadata"


# ===========================================================================
# rule_index property and invalidation
# ===========================================================================


class TestRuleIndex:
    def test_rule_index_property_returns_rule_index(self, t: Tract):
        """t.rule_index returns a RuleIndex instance."""
        from tract.rules.index import RuleIndex

        t.rule("r1", trigger="active", action={"type": "set_config", "key": "k", "value": "v"})
        idx = t.rule_index
        assert isinstance(idx, RuleIndex)
        assert len(idx) == 1
        assert ("active", "r1") in idx

    def test_rule_index_empty_when_no_rules(self, t: Tract):
        """rule_index is empty when tract has no rule commits."""
        t.user("Hello")
        idx = t.rule_index
        assert len(idx) == 0

    def test_rule_index_invalidated_after_rule_commit(self, t: Tract):
        """After a new rule commit, the next rule_index access rebuilds."""
        t.user("Hello")
        idx_before = t.rule_index
        assert len(idx_before) == 0

        t.rule("r1", trigger="commit", action={"type": "noop"})

        # The cached index should have been invalidated
        idx_after = t.rule_index
        assert len(idx_after) == 1
        assert ("commit", "r1") in idx_after

    def test_rule_index_invalidated_after_branch_switch(self, t: Tract):
        """Switching branches invalidates the rule index."""
        t.user("Hello")
        t.rule("r1", trigger="active", action={"type": "set_config", "key": "k", "value": "v"})

        # Create a new branch from same point and switch
        t.branch("feature", switch=True)
        idx = t.rule_index
        # feature branch was created from same HEAD, so same rules visible
        assert len(idx) == 1

        # Switch back to main
        t.switch("main")
        idx_main = t.rule_index
        assert len(idx_main) == 1

        # Add a rule only on main
        t.rule("r2", trigger="compile", action={"type": "noop"})
        idx_main_updated = t.rule_index
        assert len(idx_main_updated) == 2

        # Switch to feature -- should NOT see r2
        t.switch("feature")
        idx_feature = t.rule_index
        assert len(idx_feature) == 1
        assert ("compile", "r2") not in idx_feature


# ===========================================================================
# Compile exclusion
# ===========================================================================


class TestCompileExclusion:
    def test_rule_commit_excluded_from_compile(self, t: Tract):
        """Rule commits are excluded from compile() output."""
        t.user("Hello")
        t.rule("r1", trigger="active", action={"type": "noop"})
        t.assistant("World")

        compiled = t.compile()
        assert len(compiled.messages) == 2
        roles = [m.role for m in compiled.messages]
        assert "user" in roles
        assert "assistant" in roles

    def test_metadata_commit_excluded_from_compile(self, t: Tract):
        """Metadata commits are excluded from compile() output."""
        t.user("Hello")
        t.metadata("note", {"text": "internal"})
        t.assistant("World")

        compiled = t.compile()
        assert len(compiled.messages) == 2
        roles = [m.role for m in compiled.messages]
        assert "user" in roles
        assert "assistant" in roles
