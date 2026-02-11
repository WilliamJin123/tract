"""Integration tests through the public Repo API.

Validates all 5 Phase 1 success criteria end-to-end using ONLY
the public API surface (no internal imports).
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest
from pydantic import BaseModel

from tract import (
    ArtifactContent,
    BudgetAction,
    BudgetExceededError,
    CommitInfo,
    CommitOperation,
    CompiledContext,
    ContextCompiler,
    DialogueContent,
    EditTargetError,
    FreeformContent,
    InstructionContent,
    Message,
    OutputContent,
    Priority,
    ReasoningContent,
    Repo,
    RepoConfig,
    TokenBudgetConfig,
    TokenCounter,
    ToolIOContent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def repo():
    """In-memory repo, cleaned up after test."""
    r = Repo.open()
    yield r
    r.close()


@pytest.fixture()
def repo_with_commits(repo: Repo):
    """Repo pre-loaded with 3 commits (instruction + user + assistant)."""
    c1 = repo.commit(InstructionContent(text="You are helpful."), message="system")
    c2 = repo.commit(DialogueContent(role="user", text="Hi"), message="greeting")
    c3 = repo.commit(
        DialogueContent(role="assistant", text="Hello!"), message="reply"
    )
    return repo, c1, c2, c3


# ===========================================================================
# SC1: Initialization and persistence
# ===========================================================================


class TestSC1Initialization:
    """User can initialize a new trace via Repo.open() and it persists."""

    def test_open_in_memory(self):
        r = Repo.open()
        assert r.repo_id is not None
        assert r.head is None
        r.close()

    def test_open_file_backed(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        r = Repo.open(db_path)
        assert r.repo_id is not None
        r.close()
        assert (tmp_path / "test.db").exists()

    def test_open_with_custom_repo_id(self):
        r = Repo.open(repo_id="my-repo")
        assert r.repo_id == "my-repo"
        r.close()

    def test_open_as_context_manager(self):
        with Repo.open() as r:
            assert r.repo_id is not None
            r.commit(InstructionContent(text="test"))
            assert r.head is not None
        # after exit, should not raise on repr
        rep = repr(r)
        assert "Repo" in rep
        assert "closed=True" in rep

    def test_persistence_across_reopen(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        repo_id = "persist-test"

        # First session: commit content
        with Repo.open(db_path, repo_id=repo_id) as r1:
            r1.commit(InstructionContent(text="Persist me"), message="first")
            head1 = r1.head

        # Second session: re-open and verify
        with Repo.open(db_path, repo_id=repo_id) as r2:
            assert r2.head == head1
            result = r2.compile()
            assert len(result.messages) == 1
            assert "Persist me" in result.messages[0].content

    def test_from_components_uses_injected_deps(self):
        """Repo.from_components() creates a working Repo with injected components."""
        from tract.engine.compiler import DefaultContextCompiler
        from tract.engine.tokens import TiktokenCounter
        from tract.storage.engine import create_session_factory, create_trace_engine, init_db
        from tract.storage.sqlite import (
            SqliteAnnotationRepository,
            SqliteBlobRepository,
            SqliteCommitRepository,
            SqliteRefRepository,
        )

        engine = create_trace_engine(":memory:")
        init_db(engine)
        session = create_session_factory(engine)()

        commit_repo = SqliteCommitRepository(session)
        blob_repo = SqliteBlobRepository(session)
        ref_repo = SqliteRefRepository(session)
        annotation_repo = SqliteAnnotationRepository(session)
        counter = TiktokenCounter()
        compiler = DefaultContextCompiler(
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            annotation_repo=annotation_repo,
            token_counter=counter,
        )

        r = Repo.from_components(
            engine=engine,
            session=session,
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=counter,
            compiler=compiler,
            repo_id="injected",
        )

        c = r.commit(InstructionContent(text="DI test"))
        assert c.commit_hash is not None
        result = r.compile()
        assert len(result.messages) == 1
        r.close()


# ===========================================================================
# SC2: Commits, operations, and annotations
# ===========================================================================


class TestSC2CommitsAndAnnotations:
    """User can commit context, retrieve by hash, and use annotations."""

    def test_commit_append(self, repo: Repo):
        info = repo.commit(
            InstructionContent(text="System prompt"),
            message="init",
        )
        assert isinstance(info, CommitInfo)
        assert info.commit_hash is not None
        assert info.message == "init"
        assert info.created_at is not None
        assert info.token_count > 0
        assert info.operation == CommitOperation.APPEND

    def test_commit_edit(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits
        edit = repo.commit(
            InstructionContent(text="Updated system prompt"),
            operation=CommitOperation.EDIT,
            reply_to=c1.commit_hash,
            message="edit system",
        )
        assert edit.operation == CommitOperation.EDIT
        assert edit.reply_to == c1.commit_hash

        # Verify edit is reflected in compilation
        result = repo.compile()
        assert "Updated system prompt" in result.messages[0].content

    def test_get_commit_by_hash(self, repo: Repo):
        info = repo.commit(InstructionContent(text="findme"))
        retrieved = repo.get_commit(info.commit_hash)
        assert retrieved is not None
        assert retrieved.commit_hash == info.commit_hash
        assert retrieved.message is None

    def test_get_commit_nonexistent(self, repo: Repo):
        result = repo.get_commit("0000000000000000000000000000000000000000")
        assert result is None

    def test_commit_chain(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits
        assert c1.parent_hash is None
        assert c2.parent_hash == c1.commit_hash
        assert c3.parent_hash == c2.commit_hash

    def test_delete_via_skip_annotation(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits

        # Annotate c2 with SKIP
        repo.annotate(c2.commit_hash, Priority.SKIP, reason="not needed")
        result = repo.compile()
        # Should have system + assistant only (c2 user skipped)
        roles = [m.role for m in result.messages]
        assert "user" not in roles

        # Restore to NORMAL
        repo.annotate(c2.commit_hash, Priority.NORMAL, reason="restored")
        result2 = repo.compile()
        roles2 = [m.role for m in result2.messages]
        assert "user" in roles2

    def test_batch_context_manager(self, repo: Repo):
        with repo.batch():
            for i in range(10):
                repo.commit(
                    DialogueContent(role="user", text=f"Message {i}"),
                    message=f"msg-{i}",
                )
        # All 10 should be committed
        history = repo.log(limit=20)
        assert len(history) == 10


# ===========================================================================
# SC3: All content types preserved through compilation
# ===========================================================================


class TestSC3ContentTypes:
    """All 7 content types commit and compile correctly."""

    def test_instruction_content(self, repo: Repo):
        repo.commit(InstructionContent(text="Be concise."))
        result = repo.compile()
        assert len(result.messages) == 1
        assert result.messages[0].role == "system"
        assert "Be concise." in result.messages[0].content

    def test_dialogue_content_user(self, repo: Repo):
        repo.commit(DialogueContent(role="user", text="Hello"))
        result = repo.compile()
        assert result.messages[0].role == "user"
        assert "Hello" in result.messages[0].content

    def test_dialogue_content_assistant(self, repo: Repo):
        repo.commit(DialogueContent(role="assistant", text="Hi there"))
        result = repo.compile()
        assert result.messages[0].role == "assistant"
        assert "Hi there" in result.messages[0].content

    def test_tool_io_content(self, repo: Repo):
        repo.commit(
            ToolIOContent(
                tool_name="calculator",
                direction="call",
                payload={"expression": "2+2"},
            )
        )
        result = repo.compile()
        assert result.messages[0].role == "tool"
        assert "calculator" in result.messages[0].content

    def test_reasoning_content(self, repo: Repo):
        repo.commit(ReasoningContent(text="Let me think..."))
        result = repo.compile()
        assert result.messages[0].role == "assistant"
        assert "Let me think..." in result.messages[0].content

    def test_artifact_content(self, repo: Repo):
        repo.commit(
            ArtifactContent(
                artifact_type="code",
                content="print('hello')",
                language="python",
            )
        )
        result = repo.compile()
        assert "print('hello')" in result.messages[0].content

    def test_output_content(self, repo: Repo):
        repo.commit(OutputContent(text="Final answer: 42"))
        result = repo.compile()
        assert "Final answer: 42" in result.messages[0].content

    def test_freeform_content(self, repo: Repo):
        repo.commit(FreeformContent(payload={"custom": "data", "num": 42}))
        result = repo.compile()
        assert "custom" in result.messages[0].content

    def test_content_from_dict(self, repo: Repo):
        repo.commit({"content_type": "instruction", "text": "From dict"})
        result = repo.compile()
        assert "From dict" in result.messages[0].content

    def test_mixed_content_types(self, repo: Repo):
        repo.commit(InstructionContent(text="System"))
        repo.commit(DialogueContent(role="user", text="Question"))
        repo.commit(
            ToolIOContent(
                tool_name="search",
                direction="result",
                payload={"results": []},
                status="success",
            )
        )
        result = repo.compile()
        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "tool"

    def test_custom_content_type(self, repo: Repo):
        class CustomContent(BaseModel):
            content_type: str = "custom_note"
            note: str

        repo.register_content_type("custom_note", CustomContent)
        repo.commit({"content_type": "custom_note", "note": "My custom note"})
        # Should not raise -- custom type registered and validated
        info = repo.get_commit(repo.head)
        assert info is not None
        assert info.content_type == "custom_note"


# ===========================================================================
# SC4: Compilation (default and custom compiler)
# ===========================================================================


class TestSC4Compilation:
    """Compilation with default and custom compilers."""

    def test_compile_empty_repo(self, repo: Repo):
        result = repo.compile()
        assert result.messages == []
        assert result.token_count == 0

    def test_compile_default(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits
        result = repo.compile()
        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"

    def test_compile_edit_resolution(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits
        repo.commit(
            InstructionContent(text="Updated instructions"),
            operation=CommitOperation.EDIT,
            reply_to=c1.commit_hash,
        )
        result = repo.compile()
        assert "Updated instructions" in result.messages[0].content

    def test_compile_skip_annotation(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits
        repo.annotate(c2.commit_hash, Priority.SKIP)
        result = repo.compile()
        # Only system + assistant remain
        roles = [m.role for m in result.messages]
        assert "user" not in roles

    def test_compile_time_travel_datetime(self, repo: Repo):
        c1 = repo.commit(InstructionContent(text="First"))
        # Small delay so timestamps differ
        time.sleep(0.05)
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.05)
        c2 = repo.commit(DialogueContent(role="user", text="Second"))

        result = repo.compile(as_of=cutoff)
        assert len(result.messages) == 1
        assert "First" in result.messages[0].content

    def test_compile_time_travel_hash(self, repo: Repo):
        c1 = repo.commit(InstructionContent(text="First"))
        c2 = repo.commit(DialogueContent(role="user", text="Second"))
        c3 = repo.commit(DialogueContent(role="assistant", text="Third"))

        result = repo.compile(up_to=c2.commit_hash)
        assert len(result.messages) == 2

    def test_compile_aggregation(self, repo: Repo):
        """Consecutive same-role messages are aggregated."""
        repo.commit(DialogueContent(role="user", text="Part 1"))
        repo.commit(DialogueContent(role="user", text="Part 2"))
        result = repo.compile()
        assert len(result.messages) == 1
        assert "Part 1" in result.messages[0].content
        assert "Part 2" in result.messages[0].content

    def test_custom_compiler(self):
        """Custom compiler is used when provided to Repo.open()."""

        class FixedCompiler:
            """A trivial compiler that always returns a fixed result."""

            def compile(
                self,
                repo_id: str,
                head_hash: str,
                *,
                as_of=None,
                up_to=None,
                include_edit_annotations=False,
            ) -> CompiledContext:
                return CompiledContext(
                    messages=[Message(role="system", content="custom-compiled")],
                    token_count=99,
                    commit_count=1,
                    token_source="custom",
                )

        with Repo.open(compiler=FixedCompiler()) as r:
            r.commit(InstructionContent(text="anything"))
            result = r.compile()
            assert result.messages[0].content == "custom-compiled"
            assert result.token_count == 99

    def test_compile_cache_invalidated_on_commit(self, repo: Repo):
        repo.commit(InstructionContent(text="First"))
        result1 = repo.compile()
        assert len(result1.messages) == 1

        repo.commit(DialogueContent(role="user", text="Second"))
        result2 = repo.compile()
        # Cache should have been cleared; new result has 2 messages
        assert len(result2.messages) == 2


# ===========================================================================
# SC5: Token counting
# ===========================================================================


class TestSC5TokenCounting:
    """Token counting with pluggable tokenizer."""

    def test_commit_has_token_count(self, repo: Repo):
        info = repo.commit(InstructionContent(text="Some text for counting"))
        assert info.token_count > 0

    def test_compile_has_token_count(self, repo_with_commits):
        repo, *_ = repo_with_commits
        result = repo.compile()
        assert result.token_count > 0

    def test_custom_tokenizer(self):
        """Custom TokenCounter is respected."""

        class FixedCounter:
            def count_text(self, text: str) -> int:
                return 42

            def count_messages(self, messages: list[dict]) -> int:
                return 100

        with Repo.open(tokenizer=FixedCounter()) as r:
            info = r.commit(InstructionContent(text="test"))
            assert info.token_count == 42
            result = r.compile()
            assert result.token_count == 100

    def test_token_budget_warn(self, repo: Repo):
        """WARN mode: commit succeeds despite exceeding budget."""
        config = RepoConfig(
            token_budget=TokenBudgetConfig(max_tokens=5, action=BudgetAction.WARN)
        )
        with Repo.open(config=config) as r:
            # First commit is small enough, second pushes over budget
            r.commit(InstructionContent(text="a"))
            # Should NOT raise -- warn mode just logs
            r.commit(DialogueContent(role="user", text="This has many tokens in it"))

    def test_token_budget_reject(self):
        """REJECT mode: commit raises BudgetExceededError."""
        config = RepoConfig(
            token_budget=TokenBudgetConfig(max_tokens=5, action=BudgetAction.REJECT)
        )
        with Repo.open(config=config) as r:
            r.commit(InstructionContent(text="a"))
            with pytest.raises(BudgetExceededError):
                r.commit(
                    DialogueContent(
                        role="user",
                        text="This sentence definitely exceeds five tokens",
                    )
                )


# ===========================================================================
# History and logging
# ===========================================================================


class TestHistory:
    """repo.log() returns commit history."""

    def test_log_returns_commits_newest_first(self, repo_with_commits):
        repo, c1, c2, c3 = repo_with_commits
        history = repo.log()
        assert len(history) == 3
        assert history[0].commit_hash == c3.commit_hash
        assert history[1].commit_hash == c2.commit_hash
        assert history[2].commit_hash == c1.commit_hash

    def test_log_respects_limit(self, repo: Repo):
        for i in range(5):
            repo.commit(DialogueContent(role="user", text=f"msg {i}"))
        history = repo.log(limit=2)
        assert len(history) == 2

    def test_log_empty_repo_returns_empty_list(self, repo: Repo):
        assert repo.log() == []


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_edit_requires_reply_to(self, repo: Repo):
        repo.commit(InstructionContent(text="original"))
        with pytest.raises(EditTargetError):
            repo.commit(
                InstructionContent(text="edited"),
                operation=CommitOperation.EDIT,
                # reply_to intentionally omitted
            )

    def test_edit_cannot_target_edit(self, repo: Repo):
        c1 = repo.commit(InstructionContent(text="original"))
        c2 = repo.commit(
            InstructionContent(text="first edit"),
            operation=CommitOperation.EDIT,
            reply_to=c1.commit_hash,
        )
        with pytest.raises(EditTargetError):
            repo.commit(
                InstructionContent(text="edit of edit"),
                operation=CommitOperation.EDIT,
                reply_to=c2.commit_hash,
            )

    def test_annotate_priority_changes(self, repo: Repo):
        c1 = repo.commit(DialogueContent(role="user", text="test"))

        # SKIP it
        repo.annotate(c1.commit_hash, Priority.SKIP)
        result1 = repo.compile()
        assert len(result1.messages) == 0

        # PIN it
        repo.annotate(c1.commit_hash, Priority.PINNED)
        result2 = repo.compile()
        assert len(result2.messages) == 1

    def test_annotation_history(self, repo: Repo):
        c1 = repo.commit(DialogueContent(role="user", text="test"))
        repo.annotate(c1.commit_hash, Priority.SKIP, reason="hide")
        repo.annotate(c1.commit_hash, Priority.NORMAL, reason="show")

        history = repo.get_annotations(c1.commit_hash)
        # instruction auto-annotation from PINNED default not applicable here,
        # but we get the 2 manual annotations at minimum
        assert len(history) >= 2
        reasons = [a.reason for a in history if a.reason]
        assert "hide" in reasons
        assert "show" in reasons

    def test_repo_repr(self, repo: Repo):
        rep = repr(repo)
        assert "Repo(" in rep
        assert repo.repo_id in rep

    def test_repo_config_accessible(self, repo: Repo):
        assert repo.config is not None
        assert isinstance(repo.config, RepoConfig)


# ===========================================================================
# Incremental compile cache tests
# ===========================================================================


class TestIncrementalCompileCache:
    """Tests for the incremental compile cache (CompileSnapshot-based).

    Validates that APPEND-only paths use incremental extension
    while EDIT/annotate/batch/time-travel/custom-compiler correctly
    invalidate or bypass the cache.
    """

    def test_append_fast_path_matches_full_compile(self):
        """Incremental compile after APPEND produces identical output to full compile."""
        with Repo.open(":memory:", repo_id="inc-test") as repo:
            # Build up 5 commits
            for i in range(5):
                repo.commit(DialogueContent(role="user", text=f"Message {i}"))
            result_5 = repo.compile()

            # 6th APPEND triggers incremental path
            repo.commit(DialogueContent(role="assistant", text="Response"))
            result_incremental = repo.compile()

        # Verify via a fresh full compile on same DB
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with Repo.open(db_path, repo_id="full-test") as repo1:
                for i in range(5):
                    repo1.commit(DialogueContent(role="user", text=f"Message {i}"))
                repo1.commit(DialogueContent(role="assistant", text="Response"))
                head = repo1.head

            # Re-open for fresh compile (no cached snapshot)
            with Repo.open(db_path, repo_id="full-test") as repo2:
                result_full = repo2.compile()

        assert result_incremental.messages == result_full.messages
        assert result_incremental.token_count == result_full.token_count
        assert result_incremental.commit_count == result_full.commit_count

    def test_append_same_role_aggregation(self):
        """Incremental extend correctly aggregates consecutive same-role messages."""
        with Repo.open(":memory:", repo_id="agg-test") as repo:
            repo.commit(DialogueContent(role="user", text="Part 1"))
            r1 = repo.compile()
            assert len(r1.messages) == 1

            repo.commit(DialogueContent(role="user", text="Part 2"))
            r2 = repo.compile()
            assert len(r2.messages) == 1
            assert "Part 1" in r2.messages[0].content
            assert "Part 2" in r2.messages[0].content

            repo.commit(DialogueContent(role="user", text="Part 3"))
            r3 = repo.compile()
            assert len(r3.messages) == 1
            assert "Part 1" in r3.messages[0].content
            assert "Part 2" in r3.messages[0].content
            assert "Part 3" in r3.messages[0].content

        # Verify equivalence with full compile
        with Repo.open(":memory:", repo_id="agg-full") as repo2:
            repo2.commit(DialogueContent(role="user", text="Part 1"))
            repo2.commit(DialogueContent(role="user", text="Part 2"))
            repo2.commit(DialogueContent(role="user", text="Part 3"))
            full_result = repo2.compile()

        assert r3.messages == full_result.messages
        assert r3.token_count == full_result.token_count

    def test_edit_invalidates_cache(self):
        """EDIT commit invalidates the compile snapshot."""
        with Repo.open(":memory:", repo_id="edit-inv") as repo:
            c1 = repo.commit(InstructionContent(text="Original instruction"))
            repo.compile()  # Populate snapshot

            # Verify snapshot is populated
            assert repo._compile_snapshot is not None

            # EDIT commit should invalidate
            repo.commit(
                InstructionContent(text="Updated instruction"),
                operation=CommitOperation.EDIT,
                reply_to=c1.commit_hash,
            )
            assert repo._compile_snapshot is None

            # Compile should reflect the edit
            result = repo.compile()
            assert "Updated instruction" in result.messages[0].content
            assert "Original instruction" not in result.messages[0].content

    def test_annotate_invalidates_cache(self):
        """annotate() invalidates the compile snapshot."""
        with Repo.open(":memory:", repo_id="annot-inv") as repo:
            c1 = repo.commit(DialogueContent(role="user", text="Keep"))
            c2 = repo.commit(DialogueContent(role="assistant", text="Skip me"))
            c3 = repo.commit(DialogueContent(role="user", text="Also keep"))
            repo.compile()  # Populate snapshot

            assert repo._compile_snapshot is not None

            # Annotate with SKIP should invalidate snapshot
            repo.annotate(c2.commit_hash, Priority.SKIP)
            assert repo._compile_snapshot is None

            result = repo.compile()
            contents = " ".join(m.content for m in result.messages)
            assert "Skip me" not in contents
            assert "Keep" in contents

    def test_batch_invalidates_and_rebuilds(self):
        """batch() invalidates snapshot on entry; compile after batch rebuilds."""
        with Repo.open(":memory:", repo_id="batch-inv") as repo:
            # Pre-populate with a commit and compile to get a snapshot
            repo.commit(InstructionContent(text="Before batch"))
            repo.compile()
            assert repo._compile_snapshot is not None

            with repo.batch():
                # Inside batch, snapshot should have been cleared
                # (commit() sets it to None for non-incremental cases during batch)
                repo.commit(DialogueContent(role="user", text="Batch 1"))
                repo.commit(DialogueContent(role="assistant", text="Batch 2"))
                repo.commit(DialogueContent(role="user", text="Batch 3"))

            # After batch, compile should work with full rebuild
            result = repo.compile()
            assert result.commit_count == 4  # 1 before + 3 in batch
            assert len(result.messages) >= 3  # system + some aggregation

    def test_time_travel_bypasses_cache(self):
        """Time-travel params bypass cache without overwriting the snapshot."""
        with Repo.open(":memory:", repo_id="tt-bypass") as repo:
            c1 = repo.commit(InstructionContent(text="First"))
            c2 = repo.commit(DialogueContent(role="user", text="Second"))
            c3 = repo.commit(DialogueContent(role="assistant", text="Third"))

            # Compile to populate snapshot for full HEAD
            full_result = repo.compile()
            assert repo._compile_snapshot is not None
            snapshot_head = repo._compile_snapshot.head_hash

            # Time-travel compile should NOT overwrite the snapshot
            tt_result = repo.compile(up_to=c1.commit_hash)
            assert len(tt_result.messages) == 1
            assert "First" in tt_result.messages[0].content

            # Snapshot should still be for the full HEAD
            assert repo._compile_snapshot is not None
            assert repo._compile_snapshot.head_hash == snapshot_head

    def test_custom_compiler_bypasses_incremental(self):
        """Custom compiler bypasses incremental cache entirely."""
        call_count = 0

        class CountingCompiler:
            """Compiler that counts how many times compile() is called."""

            def compile(
                self,
                repo_id: str,
                head_hash: str,
                *,
                as_of=None,
                up_to=None,
                include_edit_annotations=False,
            ) -> CompiledContext:
                nonlocal call_count
                call_count += 1
                return CompiledContext(
                    messages=[Message(role="system", content=f"call-{call_count}")],
                    token_count=10,
                    commit_count=1,
                    token_source="custom",
                )

        with Repo.open(compiler=CountingCompiler()) as repo:
            repo.commit(InstructionContent(text="First"))
            r1 = repo.compile()
            assert r1.messages[0].content == "call-1"

            repo.commit(DialogueContent(role="user", text="Second"))
            r2 = repo.compile()
            assert r2.messages[0].content == "call-2"

            # Both compiles invoked the custom compiler (no caching)
            assert call_count == 2


# ===========================================================================
# Record Usage (Plan 02 -- two-tier token tracking)
# ===========================================================================


class TestRecordUsage:
    """Tests for repo.record_usage() -- post-call API token recording."""

    def test_record_usage_openai_dict(self):
        """OpenAI-format dict updates CompiledContext with API-reported counts."""
        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            repo.compile()

            result = repo.record_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            })
            assert result.token_count == 100
            assert result.token_source == "api:100+50"

    def test_record_usage_anthropic_dict(self):
        """Anthropic-format dict updates CompiledContext with API-reported counts."""
        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            repo.compile()

            result = repo.record_usage({
                "input_tokens": 200,
                "output_tokens": 80,
            })
            assert result.token_count == 200
            assert result.token_source == "api:200+80"

    def test_record_usage_token_usage_dataclass(self):
        """TokenUsage dataclass directly updates CompiledContext."""
        from tract.protocols import TokenUsage

        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            repo.compile()

            result = repo.record_usage(TokenUsage(
                prompt_tokens=300,
                completion_tokens=100,
                total_tokens=400,
            ))
            assert result.token_count == 300
            assert result.token_source == "api:300+100"

    def test_record_usage_updates_snapshot(self):
        """After record_usage, subsequent compile() (without new commits) returns API counts."""
        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            repo.compile()

            repo.record_usage({
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
            })

            # Compile again without new commits -- should return cached API counts
            cached = repo.compile()
            assert cached.token_count == 500
            assert cached.token_source == "api:500+200"

    def test_record_usage_no_commits_raises(self):
        """record_usage on empty repo raises TraceError."""
        from tract.exceptions import TraceError

        with Repo.open() as repo:
            with pytest.raises(TraceError, match="Cannot record usage: no commits exist"):
                repo.record_usage({
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                })

    def test_record_usage_unrecognized_format_raises(self):
        """Unrecognized dict format raises ContentValidationError."""
        from tract.exceptions import ContentValidationError

        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            repo.compile()

            with pytest.raises(ContentValidationError, match="Unrecognized usage dict format"):
                repo.record_usage({"foo": 42})

    def test_record_usage_no_prior_compile(self):
        """record_usage works even if compile() has not been called yet."""
        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            # Do NOT call compile()

            result = repo.record_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            })
            assert result.token_count == 100
            assert result.token_source == "api:100+50"

    def test_record_usage_with_head_hash(self):
        """record_usage with explicit head_hash works for matching head."""
        from tract.exceptions import TraceError

        with Repo.open() as repo:
            repo.commit(InstructionContent(text="A"))
            repo.commit(DialogueContent(role="user", text="B"))
            c3 = repo.commit(DialogueContent(role="assistant", text="C"))
            repo.compile()

            result = repo.record_usage(
                {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
                head_hash=c3.commit_hash,
            )
            assert result.token_count == 500

            with pytest.raises(TraceError, match="does not match current HEAD"):
                repo.record_usage(
                    {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    head_hash="nonexistent",
                )

    def test_token_source_reflects_api_after_record(self):
        """Token source transitions: tiktoken -> api -> tiktoken (after new commit)."""
        with Repo.open() as repo:
            repo.commit(InstructionContent(text="System prompt"))
            ctx1 = repo.compile()
            assert ctx1.token_source.startswith("tiktoken:")

            repo.record_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            })
            ctx2 = repo.compile()
            assert ctx2.token_source.startswith("api:")

            # New commit resets to tiktoken
            repo.commit(DialogueContent(role="user", text="New message"))
            ctx3 = repo.compile()
            assert ctx3.token_source.startswith("tiktoken:")
