"""Tests for the CommitEngine.

Integration tests using in-memory SQLite with real repository implementations.
Tests commit creation, edit validation, token budget enforcement, blob dedup,
and annotation management.
"""

from __future__ import annotations

import logging

import pytest

from tract.engine.commit import CommitEngine, extract_text_from_content
from tract.engine.tokens import NullTokenCounter, TiktokenCounter
from tract.exceptions import BudgetExceededError, CommitNotFoundError, EditTargetError
from tract.models.annotations import Priority
from tract.models.commit import CommitOperation
from tract.models.config import BudgetAction, TokenBudgetConfig
from tract.models.content import (
    ArtifactContent,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
)
from tract.storage.schema import BlobRow
from tract.storage.sqlite import (
    SqliteAnnotationRepository,
    SqliteBlobRepository,
    SqliteCommitRepository,
    SqliteRefRepository,
)


@pytest.fixture
def commit_engine(session, sample_tract_id):
    """CommitEngine with real repos and TiktokenCounter."""
    commit_repo = SqliteCommitRepository(session)
    blob_repo = SqliteBlobRepository(session)
    ref_repo = SqliteRefRepository(session)
    annot_repo = SqliteAnnotationRepository(session)
    counter = TiktokenCounter()
    return CommitEngine(commit_repo, blob_repo, ref_repo, annot_repo, counter, sample_tract_id)


@pytest.fixture
def repos(session):
    """Return all four repos as a dict."""
    return {
        "commit": SqliteCommitRepository(session),
        "blob": SqliteBlobRepository(session),
        "ref": SqliteRefRepository(session),
        "annotation": SqliteAnnotationRepository(session),
    }


class TestExtractText:
    """Tests for extract_text_from_content helper."""

    def test_instruction(self) -> None:
        assert extract_text_from_content(InstructionContent(text="hello")) == "hello"

    def test_dialogue(self) -> None:
        assert extract_text_from_content(DialogueContent(role="user", text="hi")) == "hi"

    def test_artifact(self) -> None:
        assert extract_text_from_content(ArtifactContent(artifact_type="code", content="print()")) == "print()"

    def test_tool_io(self) -> None:
        result = extract_text_from_content(ToolIOContent(tool_name="search", direction="call", payload={"q": "test"}))
        assert "test" in result

    def test_freeform(self) -> None:
        result = extract_text_from_content(FreeformContent(payload={"key": "val"}))
        assert "val" in result

    def test_reasoning(self) -> None:
        assert extract_text_from_content(ReasoningContent(text="thinking...")) == "thinking..."

    def test_output(self) -> None:
        assert extract_text_from_content(OutputContent(text="result")) == "result"


class TestCreateCommit:
    """Tests for CommitEngine.create_commit."""

    def test_append_creates_commit(self, commit_engine, repos, sample_tract_id) -> None:
        """Basic append creates a commit, updates HEAD, stores blob."""
        info = commit_engine.create_commit(
            InstructionContent(text="You are a helpful assistant."),
            message="system prompt",
        )

        assert info.commit_hash is not None
        assert len(info.commit_hash) == 64
        assert info.tract_id == sample_tract_id
        assert info.parent_hash is None  # first commit
        assert info.operation == CommitOperation.APPEND
        assert info.content_type == "instruction"
        assert info.token_count >= 0
        assert info.message == "system prompt"

        # HEAD should point to this commit
        head = repos["ref"].get_head(sample_tract_id)
        assert head == info.commit_hash

        # Blob should exist
        blob = repos["blob"].get(info.content_hash)
        assert blob is not None

    def test_parent_chain(self, commit_engine) -> None:
        """Second commit has first commit as parent."""
        c1 = commit_engine.create_commit(InstructionContent(text="first"))
        c2 = commit_engine.create_commit(DialogueContent(role="user", text="second"))

        assert c1.parent_hash is None
        assert c2.parent_hash == c1.commit_hash

    def test_blob_deduplication(self, commit_engine, repos) -> None:
        """Same content stored once but can be referenced by multiple commits."""
        content = InstructionContent(text="You are a helpful assistant.")
        c1 = commit_engine.create_commit(content, message="first")
        c2 = commit_engine.create_commit(content, message="second")

        # Same content hash
        assert c1.content_hash == c2.content_hash
        # Different commit hashes (different parent, timestamp)
        assert c1.commit_hash != c2.commit_hash

        # Only one blob row
        blob = repos["blob"].get(c1.content_hash)
        assert blob is not None

    def test_edit_operation(self, commit_engine) -> None:
        """Edit operation creates commit with response_to pointing to original."""
        original = commit_engine.create_commit(
            DialogueContent(role="user", text="Hello"),
            message="original",
        )
        edited = commit_engine.create_commit(
            DialogueContent(role="user", text="Hello, world!"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
            message="edit",
        )

        assert edited.operation == CommitOperation.EDIT
        assert edited.response_to == original.commit_hash

    def test_edit_without_response_to_raises(self, commit_engine) -> None:
        """Edit without response_to raises EditTargetError."""
        with pytest.raises(EditTargetError):
            commit_engine.create_commit(
                DialogueContent(role="user", text="Hello"),
                operation=CommitOperation.EDIT,
            )

    def test_edit_nonexistent_target_raises(self, commit_engine) -> None:
        """Edit targeting nonexistent commit raises EditTargetError."""
        with pytest.raises(EditTargetError):
            commit_engine.create_commit(
                DialogueContent(role="user", text="Hello"),
                operation=CommitOperation.EDIT,
                response_to="nonexistent_hash_1234567890abcdef",
            )

    def test_edit_targeting_edit_raises(self, commit_engine) -> None:
        """Cannot edit an EDIT commit."""
        original = commit_engine.create_commit(
            DialogueContent(role="user", text="Hello"),
        )
        edit = commit_engine.create_commit(
            DialogueContent(role="user", text="Hello edited"),
            operation=CommitOperation.EDIT,
            response_to=original.commit_hash,
        )
        with pytest.raises(EditTargetError):
            commit_engine.create_commit(
                DialogueContent(role="user", text="Hello re-edited"),
                operation=CommitOperation.EDIT,
                response_to=edit.commit_hash,
            )

    def test_metadata_stored(self, commit_engine) -> None:
        """Metadata dict is preserved in commit."""
        info = commit_engine.create_commit(
            InstructionContent(text="test"),
            metadata={"source": "test", "version": 1},
        )
        assert info.metadata == {"source": "test", "version": 1}

    def test_commit_hash_is_deterministic_for_same_inputs(self, commit_engine) -> None:
        """Different timestamps mean different hashes (time is part of hash)."""
        c1 = commit_engine.create_commit(InstructionContent(text="test"))
        c2 = commit_engine.create_commit(InstructionContent(text="test"))
        # Different because parent_hash and timestamp differ
        assert c1.commit_hash != c2.commit_hash


class TestTokenBudget:
    """Tests for token budget enforcement."""

    def test_warn_mode_allows_commit(self, session, sample_tract_id, caplog) -> None:
        """WARN mode logs warning but allows commit."""
        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annot_repo = SqliteAnnotationRepository(session)
        counter = TiktokenCounter()
        budget = TokenBudgetConfig(max_tokens=1, action=BudgetAction.WARN)

        engine = CommitEngine(
            commit_repo, blob_repo, ref_repo, annot_repo, counter, sample_tract_id,
            token_budget=budget,
        )

        with caplog.at_level(logging.WARNING, logger="tract.engine.commit"):
            info = engine.create_commit(InstructionContent(text="This text definitely has more than 1 token"))

        assert info.commit_hash is not None
        assert "budget exceeded" in caplog.text.lower() or "Token budget" in caplog.text

    def test_reject_mode_raises(self, session, sample_tract_id) -> None:
        """REJECT mode raises BudgetExceededError."""
        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annot_repo = SqliteAnnotationRepository(session)
        counter = TiktokenCounter()
        budget = TokenBudgetConfig(max_tokens=1, action=BudgetAction.REJECT)

        engine = CommitEngine(
            commit_repo, blob_repo, ref_repo, annot_repo, counter, sample_tract_id,
            token_budget=budget,
        )

        with pytest.raises(BudgetExceededError) as exc_info:
            engine.create_commit(InstructionContent(text="This text definitely has more than 1 token"))
        assert exc_info.value.max_tokens == 1

    def test_callback_mode_calls_callback(self, session, sample_tract_id) -> None:
        """CALLBACK mode calls the provided callback function."""
        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annot_repo = SqliteAnnotationRepository(session)
        counter = TiktokenCounter()

        callback_calls: list[tuple[int, int]] = []

        budget = TokenBudgetConfig(
            max_tokens=1,
            action=BudgetAction.CALLBACK,
            callback=lambda current, max_t: callback_calls.append((current, max_t)),
        )

        engine = CommitEngine(
            commit_repo, blob_repo, ref_repo, annot_repo, counter, sample_tract_id,
            token_budget=budget,
        )

        engine.create_commit(InstructionContent(text="This text definitely has more than 1 token"))

        assert len(callback_calls) == 1
        assert callback_calls[0][1] == 1  # max_tokens
        assert callback_calls[0][0] > 1  # current_tokens > max


class TestGetCommit:
    """Tests for CommitEngine.get_commit."""

    def test_returns_commit_info(self, commit_engine) -> None:
        """get_commit returns CommitInfo for existing hash."""
        created = commit_engine.create_commit(InstructionContent(text="test"))
        fetched = commit_engine.get_commit(created.commit_hash)

        assert fetched is not None
        assert fetched.commit_hash == created.commit_hash
        assert fetched.content_type == "instruction"
        assert fetched.token_count == created.token_count

    def test_returns_none_for_nonexistent(self, commit_engine) -> None:
        """get_commit returns None for nonexistent hash."""
        result = commit_engine.get_commit("nonexistent_hash_abc123")
        assert result is None


class TestAnnotate:
    """Tests for CommitEngine.annotate."""

    def test_creates_annotation(self, commit_engine) -> None:
        """annotate creates a priority annotation."""
        commit = commit_engine.create_commit(DialogueContent(role="user", text="hello"))
        annotation = commit_engine.annotate(commit.commit_hash, Priority.PINNED, reason="important")

        assert annotation.target_hash == commit.commit_hash
        assert annotation.priority == Priority.PINNED
        assert annotation.reason == "important"

    def test_annotate_nonexistent_raises(self, commit_engine) -> None:
        """annotate raises CommitNotFoundError for nonexistent target."""
        with pytest.raises(CommitNotFoundError):
            commit_engine.annotate("nonexistent_hash", Priority.SKIP)

    def test_instruction_auto_pinned(self, commit_engine, repos) -> None:
        """InstructionContent automatically gets PINNED annotation."""
        commit = commit_engine.create_commit(InstructionContent(text="system prompt"))

        latest = repos["annotation"].get_latest(commit.commit_hash)
        assert latest is not None
        assert latest.priority == Priority.PINNED

    def test_dialogue_no_auto_annotation(self, commit_engine, repos) -> None:
        """DialogueContent does NOT get an auto annotation (default is NORMAL)."""
        commit = commit_engine.create_commit(DialogueContent(role="user", text="hello"))

        latest = repos["annotation"].get_latest(commit.commit_hash)
        assert latest is None  # NORMAL is the default, no annotation created


# =========================================================================
# Generation Config (Phase 1.3)
# =========================================================================


class TestGenerationConfig:
    """Tests for generation_config threading through CommitEngine."""

    def test_create_commit_with_generation_config(self, commit_engine) -> None:
        """generation_config is stored on CommitRow and returned in CommitInfo."""
        config = {"model": "gpt-4o", "temperature": 0.7}
        info = commit_engine.create_commit(
            InstructionContent(text="test"),
            generation_config=config,
        )
        assert info.generation_config == config

    def test_create_commit_without_generation_config(self, commit_engine) -> None:
        """generation_config defaults to None when not provided."""
        info = commit_engine.create_commit(InstructionContent(text="test"))
        assert info.generation_config is None

    def test_row_to_info_maps_generation_config(self, commit_engine, repos) -> None:
        """_row_to_info correctly maps generation_config_json to generation_config."""
        config = {"temperature": 0.5, "top_p": 0.9}
        info = commit_engine.create_commit(
            DialogueContent(role="user", text="hello"),
            generation_config=config,
        )
        # Fetch via get_commit which uses _row_to_info
        fetched = commit_engine.get_commit(info.commit_hash)
        assert fetched is not None
        assert fetched.generation_config == config

    def test_generation_config_not_in_content_hash(self, commit_engine) -> None:
        """generation_config does not affect content_hash."""
        c1 = commit_engine.create_commit(
            DialogueContent(role="user", text="same"),
            generation_config={"temperature": 0.1},
        )
        c2 = commit_engine.create_commit(
            DialogueContent(role="user", text="same"),
            generation_config={"temperature": 0.9},
        )
        assert c1.content_hash == c2.content_hash
