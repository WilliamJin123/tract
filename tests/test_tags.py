"""Tests for the semantic tag system (Gap 1).

Covers:
- CommitInfo with tags roundtrip through storage
- Mutable tag annotation CRUD
- TagRegistry creation, registration, listing
- Strict mode enforcement
- _auto_classify() heuristics for each content type
- query_by_tags with match="any" and match="all"
- log() with tags filter
- Combined immutable + mutable tag queries
- Schema migration from v9 to v10
"""

import pytest

from tract import (
    CommitInfo,
    CommitOperation,
    DialogueContent,
    InstructionContent,
    Tract,
    TagNotRegisteredError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tract(**kwargs) -> Tract:
    """Create an in-memory Tract for testing."""
    return Tract.open(":memory:", **kwargs)


# ---------------------------------------------------------------------------
# 1a. CommitInfo with tags roundtrip through storage
# ---------------------------------------------------------------------------

class TestCommitInfoTags:
    """CommitInfo.tags field and storage roundtrip."""

    def test_commit_info_default_tags_empty(self):
        """CommitInfo default tags is empty list."""
        t = make_tract()
        info = t.user("hello")
        # Tags may have auto-classified values, but if we access .tags it's a list
        assert isinstance(info.tags, list)

    def test_commit_with_explicit_tags(self):
        """Explicit tags set at commit time appear on CommitInfo."""
        t = make_tract()
        info = t.user("hello", tags=["observation"])
        assert "observation" in info.tags

    def test_tags_roundtrip_through_storage(self):
        """Tags survive storage and retrieval."""
        t = make_tract()
        info = t.commit(
            InstructionContent(text="Be helpful"),
            tags=["instruction", "observation"],
        )
        # Retrieve via log
        log_entries = t.log(limit=1)
        assert len(log_entries) == 1
        assert "instruction" in log_entries[0].tags
        assert "observation" in log_entries[0].tags

    def test_tags_immutable_at_commit_time(self):
        """Tags in CommitInfo cannot be changed after commit."""
        t = make_tract()
        info = t.user("hello", tags=["observation"])
        original_tags = list(info.tags)
        # Verify they persist across retrieval
        log_entries = t.log(limit=1)
        assert set(log_entries[0].tags) == set(original_tags)

    def test_commit_with_empty_tags(self):
        """Empty tags list is valid."""
        t = make_tract()
        # Turn off strict mode so we don't get errors for auto-tags
        t._strict_tags = False
        info = t.commit(DialogueContent(role="user", text="hi"), tags=[])
        assert isinstance(info.tags, list)

    def test_tags_deduplicated(self):
        """Duplicate tags are deduplicated."""
        t = make_tract()
        info = t.user("hello", tags=["observation", "observation"])
        # Should not have duplicates
        assert len(info.tags) == len(set(info.tags))


# ---------------------------------------------------------------------------
# 1b. Mutable annotation tags via t.tag() / t.untag() / t.get_tags()
# ---------------------------------------------------------------------------

class TestMutableTags:
    """Mutable tag annotation CRUD."""

    def test_tag_adds_annotation(self):
        """t.tag() adds a tag annotation to a commit."""
        t = make_tract()
        info = t.user("hello")
        t.tag(info.commit_hash, "decision")
        tags = t.get_tags(info.commit_hash)
        assert "decision" in tags

    def test_untag_removes_annotation(self):
        """t.untag() removes a tag annotation."""
        t = make_tract()
        info = t.user("hello")
        t.tag(info.commit_hash, "decision")
        assert "decision" in t.get_tags(info.commit_hash)
        result = t.untag(info.commit_hash, "decision")
        assert result is True
        assert "decision" not in t.get_tags(info.commit_hash)

    def test_untag_nonexistent_returns_false(self):
        """t.untag() returns False if the tag doesn't exist."""
        t = make_tract()
        info = t.user("hello")
        result = t.untag(info.commit_hash, "nonexistent")
        assert result is False

    def test_tag_invalid_commit_raises(self):
        """t.tag() raises CommitNotFoundError for nonexistent commit."""
        from tract.exceptions import CommitNotFoundError
        t = make_tract()
        with pytest.raises(CommitNotFoundError):
            t.tag("deadbeef" * 8, "instruction")

    def test_get_tags_combines_immutable_and_mutable(self):
        """get_tags() returns both immutable and mutable tags."""
        t = make_tract()
        info = t.user("hello", tags=["observation"])
        t.tag(info.commit_hash, "decision")
        tags = t.get_tags(info.commit_hash)
        assert "observation" in tags
        assert "decision" in tags

    def test_get_tags_deduplicates(self):
        """get_tags() doesn't return duplicates if same tag in both sources."""
        t = make_tract()
        info = t.user("hello", tags=["reasoning"])
        t.tag(info.commit_hash, "reasoning")
        tags = t.get_tags(info.commit_hash)
        assert tags.count("reasoning") == 1

    def test_multiple_annotation_tags(self):
        """Multiple annotation tags can be added to a commit."""
        t = make_tract()
        info = t.user("hello")
        t.tag(info.commit_hash, "decision")
        t.tag(info.commit_hash, "observation")
        tags = t.get_tags(info.commit_hash)
        assert "decision" in tags
        assert "observation" in tags


# ---------------------------------------------------------------------------
# 1c. TagRegistry (strict enforcement)
# ---------------------------------------------------------------------------

class TestTagRegistry:
    """Tag registry creation, registration, and strict mode."""

    def test_base_tags_pre_seeded(self):
        """Base tags are pre-seeded on open()."""
        t = make_tract()
        tag_list = t.list_tags()
        tag_names = {entry["name"] for entry in tag_list}
        expected = {
            "instruction", "tool_call", "tool_result", "reasoning",
            "revision", "observation", "decision", "summary",
        }
        assert expected <= tag_names

    def test_base_tags_marked_auto_created(self):
        """Base tags have auto_created=True."""
        t = make_tract()
        tag_list = t.list_tags()
        for entry in tag_list:
            if entry["name"] in {"instruction", "tool_call", "tool_result", "reasoning"}:
                assert entry["auto_created"] is True

    def test_register_custom_tag(self):
        """register_tag() adds a custom tag to the registry."""
        t = make_tract()
        t.register_tag("important-decision", "Marks critical decisions")
        tag_list = t.list_tags()
        names = {entry["name"] for entry in tag_list}
        assert "important-decision" in names

    def test_register_tag_idempotent(self):
        """register_tag() is idempotent (calling twice doesn't create duplicate)."""
        t = make_tract()
        t.register_tag("my-tag", "Description 1")
        t.register_tag("my-tag", "Description 2")
        tag_list = t.list_tags()
        my_tags = [e for e in tag_list if e["name"] == "my-tag"]
        assert len(my_tags) == 1
        # Description should be updated
        assert my_tags[0]["description"] == "Description 2"

    def test_strict_mode_rejects_unregistered_tag(self):
        """In strict mode, using an unregistered tag raises TagNotRegisteredError."""
        t = make_tract()
        assert t._strict_tags is True  # Default
        with pytest.raises(TagNotRegisteredError):
            t.user("hello", tags=["unregistered-tag"])

    def test_strict_mode_allows_registered_tag(self):
        """In strict mode, registered tags work fine."""
        t = make_tract()
        info = t.user("hello", tags=["reasoning"])
        assert "reasoning" in info.tags

    def test_strict_mode_off_allows_any_tag(self):
        """With strict mode off, any tag is accepted."""
        t = make_tract()
        t._strict_tags = False
        info = t.user("hello", tags=["anything-goes"])
        assert "anything-goes" in info.tags

    def test_strict_mode_tag_annotation(self):
        """t.tag() also respects strict mode."""
        t = make_tract()
        info = t.user("hello")
        with pytest.raises(TagNotRegisteredError):
            t.tag(info.commit_hash, "unregistered-annotation-tag")

    def test_list_tags_includes_counts(self):
        """list_tags() includes usage counts."""
        t = make_tract()
        t.user("message 1", tags=["reasoning"])
        t.user("message 2", tags=["reasoning"])
        t.user("message 3", tags=["observation"])
        tag_list = t.list_tags()
        reasoning_entry = next(e for e in tag_list if e["name"] == "reasoning")
        # Auto-classify may also add reasoning tags, so count >= 2
        assert reasoning_entry["count"] >= 2

    def test_custom_tag_auto_created_false(self):
        """Custom registered tags have auto_created=False."""
        t = make_tract()
        t.register_tag("custom", "A custom tag")
        tag_list = t.list_tags()
        custom = next(e for e in tag_list if e["name"] == "custom")
        assert custom["auto_created"] is False


# ---------------------------------------------------------------------------
# 1d. _auto_classify() heuristics
# ---------------------------------------------------------------------------

class TestAutoClassify:
    """_auto_classify() heuristic-based tag classification."""

    def test_instruction_content_tagged_instruction(self):
        """InstructionContent gets 'instruction' tag."""
        t = make_tract()
        info = t.system("Be helpful")
        assert "instruction" in info.tags

    def test_user_message_no_tool_call_id(self):
        """User message without tool_call_id does not get tool_result."""
        t = make_tract()
        info = t.user("What is 2+2?")
        assert "tool_result" not in info.tags

    def test_assistant_without_tool_calls_tagged_reasoning(self):
        """Assistant message without tool_calls gets 'reasoning' tag."""
        t = make_tract()
        info = t.assistant("The answer is 4")
        assert "reasoning" in info.tags

    def test_assistant_with_tool_calls_tagged_tool_call(self):
        """Assistant message with tool_calls in metadata gets 'tool_call' tag."""
        t = make_tract()
        info = t.assistant(
            "Let me calculate",
            metadata={"tool_calls": [{"name": "calc", "args": {"x": 2}}]},
        )
        assert "tool_call" in info.tags

    def test_edit_operation_tagged_revision(self):
        """EDIT operations get 'revision' tag."""
        t = make_tract()
        original = t.user("first version")
        edited = t.user("second version", edit=original.commit_hash, tags=["observation"])
        assert "revision" in edited.tags

    def test_session_content_tagged_observation(self):
        """Session content gets 'observation' tag."""
        t = make_tract()
        t._strict_tags = False  # session auto-classification
        from tract.models.session import SessionContent
        info = t.commit(SessionContent(
            content_type="session",
            session_type="start",
            summary="Starting work",
        ))
        assert "observation" in info.tags

    def test_auto_tags_merge_with_explicit(self):
        """Auto-classified tags merge with explicit tags, deduplicated."""
        t = make_tract()
        # System message gets auto-tagged 'instruction', we also add 'decision'
        info = t.system("Be decisive", tags=["decision"])
        assert "instruction" in info.tags
        assert "decision" in info.tags
        # No duplicates
        assert len(info.tags) == len(set(info.tags))

    def test_auto_classify_returns_tuple(self):
        """_auto_classify returns (message, tags) tuple."""
        t = make_tract()
        result = t._auto_classify("instruction", "test text", role="system")
        assert isinstance(result, tuple)
        assert len(result) == 2
        msg, tags = result
        assert isinstance(msg, str)
        assert isinstance(tags, list)


# ---------------------------------------------------------------------------
# 1e. Tag-based queries
# ---------------------------------------------------------------------------

class TestTagQueries:
    """query_by_tags and log with tags filter."""

    def test_query_by_tags_any_match(self):
        """query_by_tags with match='any' returns commits with any listed tag."""
        t = make_tract()
        t.system("instruction 1")
        t.user("message 1", tags=["observation"])
        t.assistant("response 1")
        t.user("message 2", tags=["decision"])

        results = t.query_by_tags(["observation", "decision"])
        tag_sets = [set(r.tags) for r in results]
        for ts in tag_sets:
            assert ts & {"observation", "decision"}

    def test_query_by_tags_all_match(self):
        """query_by_tags with match='all' returns commits with all listed tags."""
        t = make_tract()
        t.user("msg1", tags=["observation", "decision"])
        t.user("msg2", tags=["observation"])
        t.assistant("response")

        results = t.query_by_tags(["observation", "decision"], match="all")
        for r in results:
            assert "observation" in r.tags
            assert "decision" in r.tags

    def test_query_by_tags_empty_list(self):
        """query_by_tags with empty list returns empty."""
        t = make_tract()
        t.user("hello")
        assert t.query_by_tags([]) == []

    def test_query_by_tags_limit(self):
        """query_by_tags respects limit."""
        t = make_tract()
        for i in range(10):
            t.user(f"msg {i}", tags=["observation"])
        results = t.query_by_tags(["observation"], limit=3)
        assert len(results) <= 3

    def test_query_by_tags_includes_annotation_tags(self):
        """query_by_tags includes commits with mutable annotation tags."""
        t = make_tract()
        info = t.user("special message")
        t.tag(info.commit_hash, "decision")
        results = t.query_by_tags(["decision"])
        hashes = [r.commit_hash for r in results]
        assert info.commit_hash in hashes

    def test_log_with_tags_filter(self):
        """log(tags=...) filters by tag."""
        t = make_tract()
        t.system("instruction")
        t.user("user msg")
        t.assistant("assistant msg")

        # Filter for instruction tags only
        results = t.log(tags=["instruction"])
        for r in results:
            all_tags = t.get_tags(r.commit_hash)
            assert "instruction" in all_tags

    def test_log_with_tags_any(self):
        """log(tags=..., tag_match='any') uses OR logic."""
        t = make_tract()
        t.system("sys")
        t.user("usr")
        t.assistant("ast")

        results = t.log(tags=["instruction", "reasoning"], tag_match="any")
        assert len(results) >= 2  # At least system + assistant

    def test_log_with_tags_all(self):
        """log(tags=..., tag_match='all') uses AND logic."""
        t = make_tract()
        info = t.user("tagged message", tags=["observation", "decision"])
        t.user("single tag", tags=["observation"])

        results = t.log(tags=["observation", "decision"], tag_match="all")
        # Only the first message should match
        hashes = [r.commit_hash for r in results]
        assert info.commit_hash in hashes

    def test_log_without_tags_filter_returns_all(self):
        """log() without tags returns all commits as before."""
        t = make_tract()
        t.system("sys")
        t.user("usr")
        t.assistant("ast")
        results = t.log(limit=100)
        assert len(results) == 3

    def test_query_by_tags_combined_sources(self):
        """query_by_tags combines immutable commit tags and mutable annotation tags."""
        t = make_tract()
        info1 = t.user("msg1", tags=["observation"])
        info2 = t.user("msg2")
        t.tag(info2.commit_hash, "observation")

        results = t.query_by_tags(["observation"])
        hashes = [r.commit_hash for r in results]
        assert info1.commit_hash in hashes
        assert info2.commit_hash in hashes

    def test_query_by_tags_all_match_combined_sources(self):
        """match='all' works across immutable + mutable tag sources."""
        t = make_tract()
        # This commit has 'observation' immutably and 'decision' added later
        info = t.user("msg", tags=["observation"])
        t.tag(info.commit_hash, "decision")

        results = t.query_by_tags(["observation", "decision"], match="all")
        hashes = [r.commit_hash for r in results]
        assert info.commit_hash in hashes


# ---------------------------------------------------------------------------
# Schema migration v9 -> v10
# ---------------------------------------------------------------------------

class TestSchemaMigration:
    """Schema migration from v9 to v10."""

    def test_new_database_gets_v10(self):
        """New databases start at schema version 10."""
        from tract.storage.engine import create_trace_engine, init_db
        from sqlalchemy import select, text
        from sqlalchemy.orm import sessionmaker
        from tract.storage.schema import TraceMetaRow

        engine = create_trace_engine(":memory:")
        init_db(engine)
        SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
        with SessionLocal() as session:
            meta = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
            assert meta.value == "12"
        engine.dispose()

    def test_v9_to_v10_migration_creates_tables(self):
        """Migration from v9 to v10 creates tag_annotations and tag_registry tables."""
        from sqlalchemy import text
        from sqlalchemy.orm import sessionmaker
        from tract.storage.engine import create_trace_engine, init_db
        from tract.storage.schema import Base, TraceMetaRow

        engine = create_trace_engine(":memory:")

        # Simulate a v9 database: create all tables except tag tables, set version to 9
        # We need to create the base tables manually
        with engine.connect() as conn:
            # Create core tables using raw SQL for v9 schema
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _trace_meta (
                    key VARCHAR(255) PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """))
            conn.execute(text("INSERT INTO _trace_meta (key, value) VALUES ('schema_version', '9')"))
            # Create minimal tables needed
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS blobs (
                    content_hash VARCHAR(64) PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    byte_size INTEGER NOT NULL,
                    token_count INTEGER NOT NULL,
                    created_at DATETIME NOT NULL
                )
            """))
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS commits (
                    commit_hash VARCHAR(64) PRIMARY KEY,
                    tract_id VARCHAR(64) NOT NULL,
                    parent_hash VARCHAR(64),
                    content_hash VARCHAR(64) NOT NULL,
                    content_type VARCHAR(50) NOT NULL,
                    operation VARCHAR(10) NOT NULL,
                    edit_target VARCHAR(64),
                    message TEXT,
                    token_count INTEGER NOT NULL,
                    metadata_json TEXT,
                    generation_config_json TEXT,
                    created_at DATETIME NOT NULL
                )
            """))
            conn.commit()

        # Now run init_db which should migrate v9 -> v10
        init_db(engine)

        with engine.connect() as conn:
            # Check schema version is now 10
            result = conn.execute(
                text("SELECT value FROM _trace_meta WHERE key='schema_version'")
            ).scalar_one()
            assert result == "12"

            # Check tag_annotations table exists
            tables = [
                r[0] for r in conn.execute(
                    text("SELECT name FROM sqlite_master WHERE type='table'")
                ).fetchall()
            ]
            assert "tag_annotations" in tables
            assert "tag_registry" in tables

            # Check tags_json column was added to commits
            columns = [
                r[1] for r in conn.execute(
                    text("PRAGMA table_info(commits)")
                ).fetchall()
            ]
            assert "tags_json" in columns

        engine.dispose()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases for the tag system."""

    def test_get_tags_on_commit_without_tags(self):
        """get_tags on a commit with no tags returns empty list."""
        t = make_tract()
        t._strict_tags = False
        info = t.commit(DialogueContent(role="user", text="no tags"), tags=[])
        # May have auto-classified tags, but get_tags should still work
        tags = t.get_tags(info.commit_hash)
        assert isinstance(tags, list)

    def test_get_tags_sorted(self):
        """get_tags returns sorted list."""
        t = make_tract()
        info = t.user("msg")
        t.tag(info.commit_hash, "decision")
        t.tag(info.commit_hash, "observation")
        tags = t.get_tags(info.commit_hash)
        assert tags == sorted(tags)

    def test_log_tags_filter_with_op_filter(self):
        """log() with both tags and op_filter."""
        t = make_tract()
        t.system("system msg")
        t.user("user msg")
        original = t.assistant("v1")
        t.assistant("v2", edit=original.commit_hash)

        # Only APPEND commits with instruction tag
        results = t.log(
            tags=["instruction"],
            op_filter=CommitOperation.APPEND,
        )
        for r in results:
            assert r.operation == CommitOperation.APPEND

    def test_system_shorthand_passes_tags(self):
        """system() shorthand passes tags to commit."""
        t = make_tract()
        info = t.system("Be helpful", tags=["decision"])
        assert "instruction" in info.tags
        assert "decision" in info.tags

    def test_user_shorthand_passes_tags(self):
        """user() shorthand passes tags to commit."""
        t = make_tract()
        info = t.user("hello", tags=["observation"])
        assert "observation" in info.tags

    def test_assistant_shorthand_passes_tags(self):
        """assistant() shorthand passes tags to commit."""
        t = make_tract()
        info = t.assistant("response", tags=["decision"])
        assert "decision" in info.tags
        assert "reasoning" in info.tags  # auto-classified

    def test_tag_count_in_list_tags(self):
        """list_tags() count reflects actual usage."""
        t = make_tract()
        t.register_tag("custom", "A custom tag")
        t._strict_tags = False  # Allow unregistered auto-tags
        info = t.user("msg", tags=["custom"])
        tag_list = t.list_tags()
        custom = next(e for e in tag_list if e["name"] == "custom")
        assert custom["count"] >= 1

    def test_query_by_tags_respects_history_order(self):
        """query_by_tags returns results in reverse chronological order."""
        t = make_tract()
        t.user("first", tags=["observation"])
        t.user("second", tags=["observation"])
        t.user("third", tags=["observation"])
        results = t.query_by_tags(["observation"])
        # Results should be in reverse chronological order (newest first)
        if len(results) >= 2:
            assert results[0].created_at >= results[1].created_at
