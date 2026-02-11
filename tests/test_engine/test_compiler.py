"""Tests for the DefaultContextCompiler.

Full-stack integration tests using in-memory SQLite with real repos,
CommitEngine, and DefaultContextCompiler. Tests edit resolution,
priority filtering, time-travel, role mapping, and aggregation.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from tract.engine.commit import CommitEngine
from tract.engine.compiler import DefaultContextCompiler
from tract.engine.tokens import NullTokenCounter, TiktokenCounter
from tract.models.annotations import Priority
from tract.models.commit import CommitOperation
from tract.models.content import (
    ArtifactContent,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
)
from tract.protocols import CompiledContext
from tract.storage.sqlite import (
    SqliteAnnotationRepository,
    SqliteBlobRepository,
    SqliteCommitRepository,
    SqliteRefRepository,
)


TRACT_ID = "compiler-test-tract"


@pytest.fixture
def stack(session):
    """Create a full engine + compiler stack sharing the same session."""
    commit_repo = SqliteCommitRepository(session)
    blob_repo = SqliteBlobRepository(session)
    ref_repo = SqliteRefRepository(session)
    annot_repo = SqliteAnnotationRepository(session)
    counter = TiktokenCounter()

    engine = CommitEngine(commit_repo, blob_repo, ref_repo, annot_repo, counter, TRACT_ID)
    compiler = DefaultContextCompiler(commit_repo, blob_repo, annot_repo, counter)

    return {
        "engine": engine,
        "compiler": compiler,
        "commit_repo": commit_repo,
        "blob_repo": blob_repo,
        "ref_repo": ref_repo,
        "annot_repo": annot_repo,
        "counter": counter,
    }


@pytest.fixture
def commit_engine(stack):
    return stack["engine"]


@pytest.fixture
def compiler(stack):
    return stack["compiler"]


# =========================================================================
# Core compilation tests
# =========================================================================


class TestCoreCompilation:
    """Basic compilation tests."""

    def test_compile_empty_chain(self, session) -> None:
        """Compiling with a nonexistent head returns empty result."""
        # Use a fresh compiler with NullTokenCounter to test empty case
        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        annot_repo = SqliteAnnotationRepository(session)
        counter = NullTokenCounter()
        compiler = DefaultContextCompiler(commit_repo, blob_repo, annot_repo, counter)

        # Nonexistent head hash - get_ancestors returns empty list
        result = compiler.compile(TRACT_ID, "nonexistent_hash_abc123")
        assert result.messages == []
        assert result.token_count == 0
        assert result.commit_count == 0

    def test_compile_single_instruction(self, commit_engine, compiler) -> None:
        """Single instruction compiles to one system message."""
        c = commit_engine.create_commit(InstructionContent(text="You are a helpful assistant."))
        result = compiler.compile(TRACT_ID, c.commit_hash)

        assert len(result.messages) == 1
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are a helpful assistant."
        assert result.commit_count == 1

    def test_compile_three_commit_chain(self, commit_engine, compiler) -> None:
        """Chain of instruction + user + assistant produces 3 messages with correct roles."""
        c1 = commit_engine.create_commit(InstructionContent(text="You are a helper."))
        c2 = commit_engine.create_commit(DialogueContent(role="user", text="Hello!"))
        c3 = commit_engine.create_commit(DialogueContent(role="assistant", text="Hi there!"))

        result = compiler.compile(TRACT_ID, c3.commit_hash)

        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[0].content == "You are a helper."
        assert result.messages[1].role == "user"
        assert result.messages[1].content == "Hello!"
        assert result.messages[2].role == "assistant"
        assert result.messages[2].content == "Hi there!"
        assert result.commit_count == 3

    def test_dialogue_uses_content_role(self, commit_engine, compiler) -> None:
        """DialogueContent uses its own role field, not a type default."""
        commit_engine.create_commit(DialogueContent(role="system", text="System via dialogue."))
        c2 = commit_engine.create_commit(DialogueContent(role="user", text="User message."))

        result = compiler.compile(TRACT_ID, c2.commit_hash)

        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"


# =========================================================================
# Role mapping tests
# =========================================================================


class TestRoleMapping:
    """Tests for content type to role mapping."""

    def test_instruction_maps_to_system(self, commit_engine, compiler) -> None:
        c = commit_engine.create_commit(InstructionContent(text="test"))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.messages[0].role == "system"

    def test_dialogue_maps_from_content(self, commit_engine, compiler) -> None:
        """Dialogue role comes from the content's role field."""
        commit_engine.create_commit(DialogueContent(role="user", text="u"))
        c2 = commit_engine.create_commit(DialogueContent(role="assistant", text="a"))
        result = compiler.compile(TRACT_ID, c2.commit_hash)
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_tool_io_maps_to_tool(self, commit_engine, compiler) -> None:
        c = commit_engine.create_commit(ToolIOContent(
            tool_name="search", direction="result",
            payload={"results": ["item1"]}, status="success",
        ))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.messages[0].role == "tool"

    def test_reasoning_maps_to_assistant(self, commit_engine, compiler) -> None:
        c = commit_engine.create_commit(ReasoningContent(text="Let me think..."))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.messages[0].role == "assistant"

    def test_artifact_maps_to_assistant(self, commit_engine, compiler) -> None:
        c = commit_engine.create_commit(ArtifactContent(
            artifact_type="code", content="print('hello')", language="python",
        ))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.messages[0].role == "assistant"

    def test_output_maps_to_assistant(self, commit_engine, compiler) -> None:
        c = commit_engine.create_commit(OutputContent(text="Final answer"))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.messages[0].role == "assistant"

    def test_freeform_maps_to_assistant(self, commit_engine, compiler) -> None:
        c = commit_engine.create_commit(FreeformContent(payload={"data": "value"}))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.messages[0].role == "assistant"


# =========================================================================
# Edit resolution tests
# =========================================================================


class TestEditResolution:
    """Tests for edit commit resolution during compilation."""

    def test_edit_replaces_original(self, commit_engine, compiler) -> None:
        """Edit commit content replaces the original in compiled output."""
        original = commit_engine.create_commit(
            DialogueContent(role="user", text="Hello"),
            message="original",
        )
        commit_engine.create_commit(
            DialogueContent(role="user", text="Hello, world!"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
            message="edit",
        )
        # Compile from current HEAD (which is the edit commit)
        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)

        # Should have 1 message (original replaced by edit, edit itself is not standalone)
        assert len(result.messages) == 1
        assert result.messages[0].content == "Hello, world!"

    def test_multiple_edits_latest_wins(self, commit_engine, compiler) -> None:
        """When multiple edits target the same commit, the latest wins."""
        original = commit_engine.create_commit(
            DialogueContent(role="user", text="Version 1"),
        )
        # First edit
        commit_engine.create_commit(
            DialogueContent(role="user", text="Version 2"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
        )
        # Second edit (latest)
        commit_engine.create_commit(
            DialogueContent(role="user", text="Version 3"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
        )

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)

        assert len(result.messages) == 1
        assert result.messages[0].content == "Version 3"

    def test_edit_not_standalone_message(self, commit_engine, compiler) -> None:
        """Edit commits do not appear as standalone messages."""
        c1 = commit_engine.create_commit(DialogueContent(role="user", text="original"))
        commit_engine.create_commit(
            DialogueContent(role="user", text="edited"),
            operation=CommitOperation.EDIT,
            response_to=c1.commit_hash,
        )

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)

        # Only one message (the original position, with edited content)
        assert len(result.messages) == 1
        # Content should be from the edit
        assert result.messages[0].content == "edited"

    def test_edit_annotation_marker(self, commit_engine, compiler) -> None:
        """include_edit_annotations appends [edited] to replaced content."""
        original = commit_engine.create_commit(DialogueContent(role="user", text="original"))
        commit_engine.create_commit(
            DialogueContent(role="user", text="new content"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
        )

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head, include_edit_annotations=True)

        assert len(result.messages) == 1
        assert result.messages[0].content.endswith(" [edited]")


# =========================================================================
# Priority filtering tests
# =========================================================================


class TestPriorityFiltering:
    """Tests for priority-based commit filtering."""

    def test_skip_annotation_excludes_commit(self, commit_engine, compiler) -> None:
        """Commit with SKIP annotation is excluded from compilation."""
        c1 = commit_engine.create_commit(DialogueContent(role="user", text="keep me"))
        c2 = commit_engine.create_commit(DialogueContent(role="assistant", text="skip me"))

        # Mark c2 as SKIP
        commit_engine.annotate(c2.commit_hash, Priority.SKIP, reason="not relevant")

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)

        assert len(result.messages) == 1
        assert result.messages[0].content == "keep me"

    def test_pinned_annotation_included(self, commit_engine, compiler) -> None:
        """Commit with PINNED annotation is included in compilation."""
        c1 = commit_engine.create_commit(DialogueContent(role="user", text="normal"))
        c2 = commit_engine.create_commit(DialogueContent(role="assistant", text="pinned"))

        commit_engine.annotate(c2.commit_hash, Priority.PINNED)

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)

        assert len(result.messages) == 2
        assert any(m.content == "pinned" for m in result.messages)

    def test_instruction_auto_pinned_included(self, commit_engine, compiler) -> None:
        """InstructionContent with auto-PINNED annotation is always included."""
        c = commit_engine.create_commit(InstructionContent(text="System prompt"))

        result = compiler.compile(TRACT_ID, c.commit_hash)

        assert len(result.messages) == 1
        assert result.messages[0].role == "system"


# =========================================================================
# Time-travel tests
# =========================================================================


class TestTimeTravel:
    """Tests for at_time and at_commit time-travel parameters."""

    def test_at_time_filters_by_datetime(self, commit_engine, compiler) -> None:
        """at_time only includes commits created at or before the given time."""
        c1 = commit_engine.create_commit(DialogueContent(role="user", text="first"))
        cutoff = datetime.now(timezone.utc)
        # Small sleep to ensure c2 has a later timestamp
        time.sleep(0.01)
        c2 = commit_engine.create_commit(DialogueContent(role="assistant", text="second"))

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head, at_time=cutoff)

        assert len(result.messages) == 1
        assert result.messages[0].content == "first"

    def test_at_commit_filters_by_commit_hash(self, commit_engine, compiler) -> None:
        """at_commit only includes commits up to and including the given hash."""
        c1 = commit_engine.create_commit(DialogueContent(role="user", text="first"))
        c2 = commit_engine.create_commit(DialogueContent(role="assistant", text="second"))
        c3 = commit_engine.create_commit(DialogueContent(role="user", text="third"))

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head, at_commit=c2.commit_hash)

        assert len(result.messages) == 2
        assert result.messages[0].content == "first"
        assert result.messages[1].content == "second"

    def test_both_at_time_and_at_commit_raises(self, commit_engine, compiler) -> None:
        """Providing both at_time and at_commit raises ValueError."""
        c = commit_engine.create_commit(DialogueContent(role="user", text="test"))
        head = commit_engine._ref_repo.get_head(TRACT_ID)

        with pytest.raises(ValueError, match="Cannot specify both"):
            compiler.compile(
                TRACT_ID,
                head,
                at_time=datetime.now(timezone.utc),
                at_commit=c.commit_hash,
            )


# =========================================================================
# Aggregation tests
# =========================================================================


class TestAggregation:
    """Tests for same-role consecutive message aggregation."""

    def test_consecutive_same_role_aggregated(self, commit_engine, compiler) -> None:
        """Two consecutive user messages are concatenated."""
        commit_engine.create_commit(DialogueContent(role="user", text="Part 1"))
        c2 = commit_engine.create_commit(DialogueContent(role="user", text="Part 2"))

        result = compiler.compile(TRACT_ID, c2.commit_hash)

        assert len(result.messages) == 1
        assert "Part 1" in result.messages[0].content
        assert "Part 2" in result.messages[0].content
        assert result.messages[0].role == "user"

    def test_different_roles_not_aggregated(self, commit_engine, compiler) -> None:
        """Messages with different roles remain separate."""
        commit_engine.create_commit(DialogueContent(role="user", text="Hello"))
        c2 = commit_engine.create_commit(DialogueContent(role="assistant", text="Hi"))

        result = compiler.compile(TRACT_ID, c2.commit_hash)

        assert len(result.messages) == 2
        assert result.messages[0].role == "user"
        assert result.messages[1].role == "assistant"

    def test_aggregation_preserves_boundaries(self, commit_engine, compiler) -> None:
        """user-user-assistant-user becomes 3 messages (agg, single, single)."""
        commit_engine.create_commit(DialogueContent(role="user", text="A"))
        commit_engine.create_commit(DialogueContent(role="user", text="B"))
        commit_engine.create_commit(DialogueContent(role="assistant", text="C"))
        c4 = commit_engine.create_commit(DialogueContent(role="user", text="D"))

        result = compiler.compile(TRACT_ID, c4.commit_hash)

        assert len(result.messages) == 3
        assert result.messages[0].role == "user"
        assert "A" in result.messages[0].content
        assert "B" in result.messages[0].content
        assert result.messages[1].role == "assistant"
        assert result.messages[1].content == "C"
        assert result.messages[2].role == "user"
        assert result.messages[2].content == "D"


# =========================================================================
# Token counting tests
# =========================================================================


class TestTokenCounting:
    """Tests for compiled output token counting."""

    def test_positive_token_count(self, commit_engine, compiler) -> None:
        """Non-empty compilation has positive token count."""
        c = commit_engine.create_commit(InstructionContent(text="You are a helpful assistant."))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.token_count > 0

    def test_token_count_reflects_compiled_output(self, commit_engine, compiler) -> None:
        """Token count reflects formatted messages, not raw content tokens."""
        c1 = commit_engine.create_commit(InstructionContent(text="System prompt."))
        c2 = commit_engine.create_commit(DialogueContent(role="user", text="Hello!"))

        result = compiler.compile(TRACT_ID, c2.commit_hash)

        # Token count should include per-message overhead and response primer
        # So it should be MORE than just the text tokens
        counter = TiktokenCounter()
        raw_text_tokens = counter.count_text("System prompt.") + counter.count_text("Hello!")
        assert result.token_count > raw_text_tokens

    def test_token_source_includes_encoding(self, commit_engine, compiler) -> None:
        """token_source field includes the tiktoken encoding name."""
        c = commit_engine.create_commit(InstructionContent(text="test"))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert result.token_source.startswith("tiktoken:")


# =========================================================================
# Content type text extraction tests
# =========================================================================


class TestContentTextExtraction:
    """Tests for text extraction from different content types in compiler."""

    def test_tool_io_formatting(self, commit_engine, compiler) -> None:
        """ToolIOContent formats as 'Tool {direction}: {name}' + payload."""
        c = commit_engine.create_commit(ToolIOContent(
            tool_name="search",
            direction="call",
            payload={"query": "test"},
        ))
        result = compiler.compile(TRACT_ID, c.commit_hash)

        assert "Tool call: search" in result.messages[0].content
        assert "test" in result.messages[0].content

    def test_freeform_as_json(self, commit_engine, compiler) -> None:
        """FreeformContent payload is rendered as JSON."""
        c = commit_engine.create_commit(FreeformContent(payload={"key": "value"}))
        result = compiler.compile(TRACT_ID, c.commit_hash)

        assert "key" in result.messages[0].content
        assert "value" in result.messages[0].content

    def test_artifact_uses_content_field(self, commit_engine, compiler) -> None:
        """ArtifactContent uses the 'content' field for text."""
        c = commit_engine.create_commit(ArtifactContent(
            artifact_type="code", content="def hello(): pass", language="python",
        ))
        result = compiler.compile(TRACT_ID, c.commit_hash)
        assert "def hello(): pass" in result.messages[0].content


# =========================================================================
# Generation Config collection tests (Phase 1.3)
# =========================================================================


class TestGenerationConfigCollection:
    """Unit tests for generation_config collection during compile."""

    def test_compile_collects_generation_configs(self, commit_engine, compiler) -> None:
        """generation_configs list parallel to effective commits."""
        config1 = {"temperature": 0.3}
        config2 = {"temperature": 0.7}
        commit_engine.create_commit(
            InstructionContent(text="System"), generation_config=config1
        )
        c2 = commit_engine.create_commit(
            DialogueContent(role="user", text="Hi"), generation_config=config2
        )
        result = compiler.compile(TRACT_ID, c2.commit_hash)
        assert len(result.generation_configs) == 2
        assert result.generation_configs[0] == config1
        assert result.generation_configs[1] == config2

    def test_compile_empty_config_for_no_config_commit(self, commit_engine, compiler) -> None:
        """Commits without generation_config get {} in the list."""
        commit_engine.create_commit(InstructionContent(text="System"))
        c2 = commit_engine.create_commit(
            DialogueContent(role="user", text="Hi"), generation_config={"temperature": 0.7}
        )
        result = compiler.compile(TRACT_ID, c2.commit_hash)
        assert result.generation_configs[0] == {}
        assert result.generation_configs[1] == {"temperature": 0.7}

    def test_edit_with_config_uses_edit_config(self, commit_engine, compiler) -> None:
        """When edit has generation_config, it is used."""
        original = commit_engine.create_commit(
            DialogueContent(role="user", text="original"),
            generation_config={"temperature": 0.3},
        )
        commit_engine.create_commit(
            DialogueContent(role="user", text="edited"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
            generation_config={"temperature": 0.9},
        )
        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)
        assert result.generation_configs[0] == {"temperature": 0.9}

    def test_edit_without_config_inherits_original(self, commit_engine, compiler) -> None:
        """When edit has no generation_config, original commit's config is inherited."""
        original = commit_engine.create_commit(
            DialogueContent(role="user", text="original"),
            generation_config={"temperature": 0.7},
        )
        commit_engine.create_commit(
            DialogueContent(role="user", text="edited"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
            # No generation_config
        )
        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)
        assert result.generation_configs[0] == {"temperature": 0.7}

    def test_skip_priority_commit_excluded_from_configs(self, commit_engine, compiler) -> None:
        """SKIP commits don't appear in generation_configs list."""
        c1 = commit_engine.create_commit(
            DialogueContent(role="user", text="keep"),
            generation_config={"temperature": 0.3},
        )
        c2 = commit_engine.create_commit(
            DialogueContent(role="assistant", text="skip me"),
            generation_config={"temperature": 0.9},
        )
        # Mark c2 as SKIP
        commit_engine.annotate(c2.commit_hash, Priority.SKIP, reason="not needed")

        head = commit_engine._ref_repo.get_head(TRACT_ID)
        result = compiler.compile(TRACT_ID, head)
        # Only c1 should be in the result
        assert len(result.generation_configs) == 1
        assert result.generation_configs[0] == {"temperature": 0.3}
