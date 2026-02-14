"""Tests for the DefaultContextCompiler.

Full-stack integration tests using in-memory SQLite with real repos,
CommitEngine, and DefaultContextCompiler. Only tests with unique behavior
not covered by the Tract facade (test_tract.py) are retained here.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tract.engine.commit import CommitEngine
from tract.engine.compiler import DefaultContextCompiler
from tract.engine.tokens import NullTokenCounter, TiktokenCounter
from tract.models.annotations import Priority
from tract.models.commit import CommitOperation
from tract.models.content import (
    DialogueContent,
    InstructionContent,
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
    """Basic compilation smoke tests and unique role-mapping edge cases."""

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
# Unique edge cases (edit, time-travel, aggregation, tokens, gen-config)
# =========================================================================


class TestUniqueEdgeCases:
    """Tests covering unique compiler behavior not exercised by the facade."""

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

    def test_mixed_roles_all_preserved(self, commit_engine, compiler) -> None:
        """user-user-assistant-user produces 4 messages, not 3."""
        commit_engine.create_commit(DialogueContent(role="user", text="A"))
        commit_engine.create_commit(DialogueContent(role="user", text="B"))
        commit_engine.create_commit(DialogueContent(role="assistant", text="C"))
        c4 = commit_engine.create_commit(DialogueContent(role="user", text="D"))

        result = compiler.compile(TRACT_ID, c4.commit_hash)

        assert len(result.messages) == 4
        assert result.messages[0].content == "A"
        assert result.messages[1].content == "B"
        assert result.messages[2].content == "C"
        assert result.messages[3].content == "D"

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
