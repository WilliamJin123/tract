"""Integration tests through the public Tract API.

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
    Tract,
    TractConfig,
    TokenBudgetConfig,
    TokenCounter,
    ToolIOContent,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tract():
    """In-memory tract, cleaned up after test."""
    t = Tract.open()
    yield t
    t.close()


@pytest.fixture()
def tract_with_commits(tract: Tract):
    """Tract pre-loaded with 3 commits (instruction + user + assistant)."""
    c1 = tract.commit(InstructionContent(text="You are helpful."), message="system")
    c2 = tract.commit(DialogueContent(role="user", text="Hi"), message="greeting")
    c3 = tract.commit(
        DialogueContent(role="assistant", text="Hello!"), message="reply"
    )
    return tract, c1, c2, c3


# ===========================================================================
# SC1: Initialization and persistence
# ===========================================================================


class TestSC1Initialization:
    """User can initialize a new trace via Tract.open() and it persists."""

    def test_open_in_memory(self):
        t = Tract.open()
        assert t.tract_id is not None
        assert t.head is None
        t.close()

    def test_open_file_backed(self, tmp_path):
        db_path = str(tmp_path / "test.db")
        t = Tract.open(db_path)
        assert t.tract_id is not None
        t.close()
        assert (tmp_path / "test.db").exists()

    def test_open_with_custom_tract_id(self):
        t = Tract.open(tract_id="my-tract")
        assert t.tract_id == "my-tract"
        t.close()

    def test_open_as_context_manager(self):
        with Tract.open() as t:
            assert t.tract_id is not None
            t.commit(InstructionContent(text="test"))
            assert t.head is not None
        # after exit, should not raise on repr
        rep = repr(t)
        assert "Tract" in rep
        assert "closed=True" in rep

    def test_persistence_across_reopen(self, tmp_path):
        db_path = str(tmp_path / "persist.db")
        tract_id = "persist-test"

        # First session: commit content
        with Tract.open(db_path, tract_id=tract_id) as t1:
            t1.commit(InstructionContent(text="Persist me"), message="first")
            head1 = t1.head

        # Second session: re-open and verify
        with Tract.open(db_path, tract_id=tract_id) as t2:
            assert t2.head == head1
            result = t2.compile()
            assert len(result.messages) == 1
            assert "Persist me" in result.messages[0].content

    def test_from_components_uses_injected_deps(self):
        """Tract.from_components() creates a working Tract with injected components."""
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

        t = Tract.from_components(
            engine=engine,
            session=session,
            commit_repo=commit_repo,
            blob_repo=blob_repo,
            ref_repo=ref_repo,
            annotation_repo=annotation_repo,
            token_counter=counter,
            compiler=compiler,
            tract_id="injected",
        )

        c = t.commit(InstructionContent(text="DI test"))
        assert c.commit_hash is not None
        result = t.compile()
        assert len(result.messages) == 1
        t.close()


# ===========================================================================
# SC2: Commits, operations, and annotations
# ===========================================================================


class TestSC2CommitsAndAnnotations:
    """User can commit context, retrieve by hash, and use annotations."""

    def test_commit_append(self, tract: Tract):
        info = tract.commit(
            InstructionContent(text="System prompt"),
            message="init",
        )
        assert isinstance(info, CommitInfo)
        assert info.commit_hash is not None
        assert info.message == "init"
        assert info.created_at is not None
        assert info.token_count > 0
        assert info.operation == CommitOperation.APPEND

    def test_commit_edit(self, tract_with_commits):
        tract, c1, c2, c3 = tract_with_commits
        edit = tract.commit(
            InstructionContent(text="Updated system prompt"),
            operation=CommitOperation.EDIT,
            response_to=c1.commit_hash,
            message="edit system",
        )
        assert edit.operation == CommitOperation.EDIT
        assert edit.response_to == c1.commit_hash

        # Verify edit is reflected in compilation
        result = tract.compile()
        assert "Updated system prompt" in result.messages[0].content

    def test_get_commit_by_hash(self, tract: Tract):
        info = tract.commit(InstructionContent(text="findme"))
        retrieved = tract.get_commit(info.commit_hash)
        assert retrieved is not None
        assert retrieved.commit_hash == info.commit_hash
        assert retrieved.message is None

    def test_get_commit_nonexistent(self, tract: Tract):
        result = tract.get_commit("0000000000000000000000000000000000000000")
        assert result is None

    def test_commit_chain(self, tract_with_commits):
        tract, c1, c2, c3 = tract_with_commits
        assert c1.parent_hash is None
        assert c2.parent_hash == c1.commit_hash
        assert c3.parent_hash == c2.commit_hash

    def test_delete_via_skip_annotation(self, tract_with_commits):
        tract, c1, c2, c3 = tract_with_commits

        # Annotate c2 with SKIP
        tract.annotate(c2.commit_hash, Priority.SKIP, reason="not needed")
        result = tract.compile()
        # Should have system + assistant only (c2 user skipped)
        roles = [m.role for m in result.messages]
        assert "user" not in roles

        # Restore to NORMAL
        tract.annotate(c2.commit_hash, Priority.NORMAL, reason="restored")
        result2 = tract.compile()
        roles2 = [m.role for m in result2.messages]
        assert "user" in roles2

    def test_batch_context_manager(self, tract: Tract):
        with tract.batch():
            for i in range(10):
                tract.commit(
                    DialogueContent(role="user", text=f"Message {i}"),
                    message=f"msg-{i}",
                )
        # All 10 should be committed
        history = tract.log(limit=20)
        assert len(history) == 10


# ===========================================================================
# SC3: All content types preserved through compilation
# ===========================================================================


class TestSC3ContentTypes:
    """All 7 content types commit and compile correctly."""

    @pytest.mark.parametrize("content,expected_role,expected_text", [
        (InstructionContent(text="Be concise."), "system", "Be concise."),
        (DialogueContent(role="user", text="Hello"), "user", "Hello"),
        (DialogueContent(role="assistant", text="Hi there"), "assistant", "Hi there"),
        (ToolIOContent(tool_name="calculator", direction="call", payload={"expression": "2+2"}), "tool", "calculator"),
        (ReasoningContent(text="Let me think..."), "assistant", "Let me think..."),
        (ArtifactContent(artifact_type="code", content="print('hello')", language="python"), "assistant", "print('hello')"),
        (OutputContent(text="Final answer: 42"), "assistant", "Final answer: 42"),
        (FreeformContent(payload={"custom": "data", "num": 42}), "assistant", "custom"),
    ])
    def test_content_type_commit_and_compile(self, tract: Tract, content, expected_role, expected_text):
        tract.commit(content)
        result = tract.compile()
        assert len(result.messages) == 1
        assert result.messages[0].role == expected_role
        assert expected_text in result.messages[0].content

    def test_content_from_dict(self, tract: Tract):
        tract.commit({"content_type": "instruction", "text": "From dict"})
        result = tract.compile()
        assert "From dict" in result.messages[0].content

    def test_mixed_content_types(self, tract: Tract):
        tract.commit(InstructionContent(text="System"))
        tract.commit(DialogueContent(role="user", text="Question"))
        tract.commit(
            ToolIOContent(
                tool_name="search",
                direction="result",
                payload={"results": []},
                status="success",
            )
        )
        result = tract.compile()
        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "tool"

    def test_custom_content_type(self, tract: Tract):
        class CustomContent(BaseModel):
            content_type: str = "custom_note"
            note: str

        tract.register_content_type("custom_note", CustomContent)
        tract.commit({"content_type": "custom_note", "note": "My custom note"})
        # Should not raise -- custom type registered and validated
        info = tract.get_commit(tract.head)
        assert info is not None
        assert info.content_type == "custom_note"


# ===========================================================================
# SC4: Compilation (default and custom compiler)
# ===========================================================================


class TestSC4Compilation:
    """Compilation with default and custom compilers."""

    def test_compile_empty_tract(self, tract: Tract):
        result = tract.compile()
        assert result.messages == []
        assert result.token_count == 0

    def test_compile_default(self, tract_with_commits):
        tract, c1, c2, c3 = tract_with_commits
        result = tract.compile()
        assert len(result.messages) == 3
        assert result.messages[0].role == "system"
        assert result.messages[1].role == "user"
        assert result.messages[2].role == "assistant"

    def test_compile_time_travel_datetime(self, tract: Tract):
        c1 = tract.commit(InstructionContent(text="First"))
        # Small delay so timestamps differ
        time.sleep(0.05)
        cutoff = datetime.now(timezone.utc)
        time.sleep(0.05)
        c2 = tract.commit(DialogueContent(role="user", text="Second"))

        result = tract.compile(at_time=cutoff)
        assert len(result.messages) == 1
        assert "First" in result.messages[0].content

    def test_compile_time_travel_hash(self, tract: Tract):
        c1 = tract.commit(InstructionContent(text="First"))
        c2 = tract.commit(DialogueContent(role="user", text="Second"))
        c3 = tract.commit(DialogueContent(role="assistant", text="Third"))

        result = tract.compile(at_commit=c2.commit_hash)
        assert len(result.messages) == 2

    def test_custom_compiler(self):
        """Custom compiler is used when provided to Tract.open()."""

        class FixedCompiler:
            """A trivial compiler that always returns a fixed result."""

            def compile(
                self,
                tract_id: str,
                head_hash: str,
                *,
                at_time=None,
                at_commit=None,
                include_edit_annotations=False,
            ) -> CompiledContext:
                return CompiledContext(
                    messages=[Message(role="system", content="custom-compiled")],
                    token_count=99,
                    commit_count=1,
                    token_source="custom",
                )

        with Tract.open(compiler=FixedCompiler()) as t:
            t.commit(InstructionContent(text="anything"))
            result = t.compile()
            assert result.messages[0].content == "custom-compiled"
            assert result.token_count == 99

    def test_compile_cache_invalidated_on_commit(self, tract: Tract):
        tract.commit(InstructionContent(text="First"))
        result1 = tract.compile()
        assert len(result1.messages) == 1

        tract.commit(DialogueContent(role="user", text="Second"))
        result2 = tract.compile()
        # Cache should have been cleared; new result has 2 messages
        assert len(result2.messages) == 2


# ===========================================================================
# SC5: Token counting
# ===========================================================================


class TestSC5TokenCounting:
    """Token counting with pluggable tokenizer."""

    def test_compile_has_token_count(self, tract_with_commits):
        tract, *_ = tract_with_commits
        result = tract.compile()
        assert result.token_count > 0

    def test_custom_tokenizer(self):
        """Custom TokenCounter is respected."""

        class FixedCounter:
            def count_text(self, text: str) -> int:
                return 42

            def count_messages(self, messages: list[dict]) -> int:
                return 100

        with Tract.open(tokenizer=FixedCounter()) as t:
            info = t.commit(InstructionContent(text="test"))
            assert info.token_count == 42
            result = t.compile()
            assert result.token_count == 100

    def test_token_budget_warn(self, tract: Tract):
        """WARN mode: commit succeeds despite exceeding budget."""
        config = TractConfig(
            token_budget=TokenBudgetConfig(max_tokens=5, action=BudgetAction.WARN)
        )
        with Tract.open(config=config) as t:
            # First commit is small enough, second pushes over budget
            t.commit(InstructionContent(text="a"))
            # Should NOT raise -- warn mode just logs
            t.commit(DialogueContent(role="user", text="This has many tokens in it"))

    def test_token_budget_reject(self):
        """REJECT mode: commit raises BudgetExceededError."""
        config = TractConfig(
            token_budget=TokenBudgetConfig(max_tokens=5, action=BudgetAction.REJECT)
        )
        with Tract.open(config=config) as t:
            t.commit(InstructionContent(text="a"))
            with pytest.raises(BudgetExceededError):
                t.commit(
                    DialogueContent(
                        role="user",
                        text="This sentence definitely exceeds five tokens",
                    )
                )


# ===========================================================================
# History and logging
# ===========================================================================


class TestHistory:
    """tract.log() returns commit history."""

    def test_log_returns_commits_newest_first(self, tract_with_commits):
        tract, c1, c2, c3 = tract_with_commits
        history = tract.log()
        assert len(history) == 3
        assert history[0].commit_hash == c3.commit_hash
        assert history[1].commit_hash == c2.commit_hash
        assert history[2].commit_hash == c1.commit_hash

    def test_log_respects_limit(self, tract: Tract):
        for i in range(5):
            tract.commit(DialogueContent(role="user", text=f"msg {i}"))
        history = tract.log(limit=2)
        assert len(history) == 2

    def test_log_empty_tract_returns_empty_list(self, tract: Tract):
        assert tract.log() == []


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and error paths."""

    def test_edit_requires_response_to(self, tract: Tract):
        tract.commit(InstructionContent(text="original"))
        with pytest.raises(EditTargetError):
            tract.commit(
                InstructionContent(text="edited"),
                operation=CommitOperation.EDIT,
                # response_to intentionally omitted
            )

    def test_edit_cannot_target_edit(self, tract: Tract):
        c1 = tract.commit(InstructionContent(text="original"))
        c2 = tract.commit(
            InstructionContent(text="first edit"),
            operation=CommitOperation.EDIT,
            response_to=c1.commit_hash,
        )
        with pytest.raises(EditTargetError):
            tract.commit(
                InstructionContent(text="edit of edit"),
                operation=CommitOperation.EDIT,
                response_to=c2.commit_hash,
            )

    def test_annotation_history(self, tract: Tract):
        c1 = tract.commit(DialogueContent(role="user", text="test"))
        tract.annotate(c1.commit_hash, Priority.SKIP, reason="hide")
        tract.annotate(c1.commit_hash, Priority.NORMAL, reason="show")

        history = tract.get_annotations(c1.commit_hash)
        # instruction auto-annotation from PINNED default not applicable here,
        # but we get the 2 manual annotations at minimum
        assert len(history) >= 2
        reasons = [a.reason for a in history if a.reason]
        assert "hide" in reasons
        assert "show" in reasons


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
        with Tract.open(":memory:", tract_id="inc-test") as t:
            # Build up 5 commits
            for i in range(5):
                t.commit(DialogueContent(role="user", text=f"Message {i}"))
            result_5 = t.compile()

            # 6th APPEND triggers incremental path
            t.commit(DialogueContent(role="assistant", text="Response"))
            result_incremental = t.compile()

        # Verify via a fresh full compile on same DB
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")
            with Tract.open(db_path, tract_id="full-test") as t1:
                for i in range(5):
                    t1.commit(DialogueContent(role="user", text=f"Message {i}"))
                t1.commit(DialogueContent(role="assistant", text="Response"))
                head = t1.head

            # Re-open for fresh compile (no cached snapshot)
            with Tract.open(db_path, tract_id="full-test") as t2:
                result_full = t2.compile()

        assert result_incremental.messages == result_full.messages
        assert result_incremental.token_count == result_full.token_count
        assert result_incremental.commit_count == result_full.commit_count

    def test_append_same_role_no_aggregation(self):
        """Incremental extend preserves consecutive same-role messages as separate."""
        with Tract.open(":memory:", tract_id="no-agg-test") as t:
            t.commit(DialogueContent(role="user", text="Part 1"))
            r1 = t.compile()
            assert len(r1.messages) == 1

            t.commit(DialogueContent(role="user", text="Part 2"))
            r2 = t.compile()
            assert len(r2.messages) == 2
            assert r2.messages[0].content == "Part 1"
            assert r2.messages[1].content == "Part 2"

            t.commit(DialogueContent(role="user", text="Part 3"))
            r3 = t.compile()
            assert len(r3.messages) == 3

        # Verify equivalence with full compile
        with Tract.open(":memory:", tract_id="no-agg-full") as t2:
            t2.commit(DialogueContent(role="user", text="Part 1"))
            t2.commit(DialogueContent(role="user", text="Part 2"))
            t2.commit(DialogueContent(role="user", text="Part 3"))
            full_result = t2.compile()

        assert len(r3.messages) == len(full_result.messages)
        assert r3.token_count == full_result.token_count
        assert r3.commit_count == full_result.commit_count

    def test_edit_does_not_clear_cache(self):
        """EDIT commit does not clear the LRU cache (other entries remain valid)."""
        with Tract.open(":memory:", tract_id="edit-inv") as t:
            c1 = t.commit(InstructionContent(text="Original instruction"))
            h1 = t.head
            t.compile()  # Populate snapshot

            # Verify snapshot is populated
            assert t._cache_get(h1) is not None

            # EDIT commit -- parent snapshot stays in cache (different HEAD)
            t.commit(
                InstructionContent(text="Updated instruction"),
                operation=CommitOperation.EDIT,
                response_to=c1.commit_hash,
            )

            # Compile should reflect the edit
            result = t.compile()
            assert "Updated instruction" in result.messages[0].content
            assert "Original instruction" not in result.messages[0].content

    def test_time_travel_bypasses_cache(self):
        """Time-travel params bypass cache without overwriting the snapshot."""
        with Tract.open(":memory:", tract_id="tt-bypass") as t:
            c1 = t.commit(InstructionContent(text="First"))
            c2 = t.commit(DialogueContent(role="user", text="Second"))
            c3 = t.commit(DialogueContent(role="assistant", text="Third"))

            # Compile to populate snapshot for full HEAD
            full_result = t.compile()
            head = t.head
            assert t._cache_get(head) is not None

            # Time-travel compile should NOT overwrite the snapshot
            tt_result = t.compile(at_commit=c1.commit_hash)
            assert len(tt_result.messages) == 1
            assert "First" in tt_result.messages[0].content

            # Snapshot should still be for the full HEAD
            cached = t._cache_get(head)
            assert cached is not None
            assert cached.head_hash == head

    def test_custom_compiler_bypasses_incremental(self):
        """Custom compiler bypasses incremental cache entirely."""
        call_count = 0

        class CountingCompiler:
            """Compiler that counts how many times compile() is called."""

            def compile(
                self,
                tract_id: str,
                head_hash: str,
                *,
                at_time=None,
                at_commit=None,
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

        with Tract.open(compiler=CountingCompiler()) as t:
            t.commit(InstructionContent(text="First"))
            r1 = t.compile()
            assert r1.messages[0].content == "call-1"

            t.commit(DialogueContent(role="user", text="Second"))
            r2 = t.compile()
            assert r2.messages[0].content == "call-2"

            # Both compiles invoked the custom compiler (no caching)
            assert call_count == 2


# ===========================================================================
# Record Usage (Plan 02 -- two-tier token tracking)
# ===========================================================================


class TestRecordUsage:
    """Tests for tract.record_usage() -- post-call API token recording."""

    @pytest.mark.parametrize("usage_input,expected_count,expected_source", [
        ({"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}, 100, "api:100+50"),
        ({"input_tokens": 200, "output_tokens": 80}, 200, "api:200+80"),
    ])
    def test_record_usage_formats(self, usage_input, expected_count, expected_source):
        with Tract.open() as t:
            t.commit(InstructionContent(text="System prompt"))
            t.compile()
            result = t.record_usage(usage_input)
            assert result.token_count == expected_count
            assert result.token_source == expected_source

    def test_record_usage_token_usage_dataclass(self):
        """TokenUsage dataclass directly updates CompiledContext."""
        from tract.protocols import TokenUsage

        with Tract.open() as t:
            t.commit(InstructionContent(text="System prompt"))
            t.compile()

            result = t.record_usage(TokenUsage(
                prompt_tokens=300,
                completion_tokens=100,
                total_tokens=400,
            ))
            assert result.token_count == 300
            assert result.token_source == "api:300+100"

    def test_record_usage_updates_snapshot(self):
        """After record_usage, subsequent compile() (without new commits) returns API counts."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="System prompt"))
            t.compile()

            t.record_usage({
                "prompt_tokens": 500,
                "completion_tokens": 200,
                "total_tokens": 700,
            })

            # Compile again without new commits -- should return cached API counts
            cached = t.compile()
            assert cached.token_count == 500
            assert cached.token_source == "api:500+200"

    def test_record_usage_no_commits_raises(self):
        """record_usage on empty tract raises TraceError."""
        from tract.exceptions import TraceError

        with Tract.open() as t:
            with pytest.raises(TraceError, match="Cannot record usage: no commits exist"):
                t.record_usage({
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                })

    def test_record_usage_unrecognized_format_raises(self):
        """Unrecognized dict format raises ContentValidationError."""
        from tract.exceptions import ContentValidationError

        with Tract.open() as t:
            t.commit(InstructionContent(text="System prompt"))
            t.compile()

            with pytest.raises(ContentValidationError, match="Unrecognized usage dict format"):
                t.record_usage({"foo": 42})

    def test_record_usage_no_prior_compile(self):
        """record_usage works even if compile() has not been called yet."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="System prompt"))
            # Do NOT call compile()

            result = t.record_usage({
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            })
            assert result.token_count == 100
            assert result.token_source == "api:100+50"

    def test_record_usage_with_head_hash(self):
        """record_usage with explicit head_hash works for matching head."""
        from tract.exceptions import TraceError

        with Tract.open() as t:
            t.commit(InstructionContent(text="A"))
            t.commit(DialogueContent(role="user", text="B"))
            c3 = t.commit(DialogueContent(role="assistant", text="C"))
            t.compile()

            result = t.record_usage(
                {"prompt_tokens": 500, "completion_tokens": 200, "total_tokens": 700},
                head_hash=c3.commit_hash,
            )
            assert result.token_count == 500

            with pytest.raises(TraceError, match="does not match current HEAD"):
                t.record_usage(
                    {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                    head_hash="nonexistent",
                )


# ===========================================================================
# Two-tier token tracking integration tests (Plan 02)
# ===========================================================================


class TestTwoTierTokenTracking:
    """End-to-end tests for the two-tier token tracking workflow.

    Tier 1: tiktoken estimates (pre-call budget checks)
    Tier 2: API-reported actuals (post-call truth via record_usage)
    """

    def test_full_workflow_tiktoken_then_api(self):
        """Simulates a real agent loop: compile -> API call -> record_usage."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="System prompt"))
            t.commit(DialogueContent(role="user", text="Hello"))

            # Pre-call: tiktoken estimate
            ctx = t.compile()
            assert ctx.token_source.startswith("tiktoken:")
            assert ctx.token_count > 0
            tiktoken_count = ctx.token_count

            # Simulate API call and record usage
            api_usage = {
                "prompt_tokens": tiktoken_count + 5,
                "completion_tokens": 42,
                "total_tokens": tiktoken_count + 47,
            }
            updated = t.record_usage(api_usage)
            assert updated.token_source == f"api:{tiktoken_count + 5}+42"
            assert updated.token_count == tiktoken_count + 5

            # Cached compile returns API counts
            cached = t.compile()
            assert cached.token_source == updated.token_source
            assert cached.token_count == updated.token_count

            # New commit resets to tiktoken
            t.commit(DialogueContent(role="assistant", text="Hi there!"))
            fresh = t.compile()
            assert fresh.token_source.startswith("tiktoken:")


# ===========================================================================
# Generation Config (Phase 1.3)
# ===========================================================================


class TestGenerationConfig:
    """Phase 1.3: Hyperparameter config storage tests."""

    # SC1: Attach and retrieve generation_config
    def test_commit_with_generation_config(self, tract: Tract):
        config = {"model": "gpt-4o", "temperature": 0.7, "top_p": 0.95}
        info = tract.commit(
            DialogueContent(role="user", text="Hello"),
            generation_config=config,
        )
        assert info.generation_config == config
        # Retrieve via get_commit
        fetched = tract.get_commit(info.commit_hash)
        assert fetched is not None
        assert fetched.generation_config == config

    def test_commit_without_generation_config(self, tract: Tract):
        info = tract.commit(DialogueContent(role="user", text="Hi"))
        assert info.generation_config is None
        fetched = tract.get_commit(info.commit_hash)
        assert fetched is not None
        assert fetched.generation_config is None

    def test_generation_config_in_log(self, tract: Tract):
        config = {"temperature": 0.9}
        tract.commit(
            DialogueContent(role="user", text="Hello"),
            generation_config=config,
        )
        entries = tract.log(limit=1)
        assert entries[0].generation_config == config

    # SC2: Flexible schema -- any provider params work
    def test_generation_config_arbitrary_provider_params(self, tract: Tract):
        openai_config = {"model": "gpt-4o", "temperature": 0.7, "frequency_penalty": 0.5}
        anthropic_config = {"model": "claude-3-opus", "top_k": 40, "temperature": 0.3}
        meta_config = {"model": "llama-3", "repetition_penalty": 1.2, "min_p": 0.05}

        c1 = tract.commit(DialogueContent(role="user", text="a"), generation_config=openai_config)
        c2 = tract.commit(DialogueContent(role="user", text="b"), generation_config=anthropic_config)
        c3 = tract.commit(DialogueContent(role="user", text="c"), generation_config=meta_config)

        assert c1.generation_config == openai_config
        assert c2.generation_config == anthropic_config
        assert c3.generation_config == meta_config

    # SC3: generation_configs preserved through compile
    def test_compile_exposes_generation_configs(self, tract: Tract):
        config1 = {"temperature": 0.5}
        config2 = {"temperature": 0.9}
        tract.commit(InstructionContent(text="System"), generation_config=config1)
        tract.commit(DialogueContent(role="user", text="Hi"), generation_config=config2)
        result = tract.compile()
        assert len(result.generation_configs) == 2
        assert result.generation_configs[0] == config1
        assert result.generation_configs[1] == config2

    def test_compile_empty_config_for_commits_without_config(self, tract: Tract):
        tract.commit(InstructionContent(text="System"))
        tract.commit(DialogueContent(role="user", text="Hi"), generation_config={"temperature": 0.7})
        result = tract.compile()
        assert result.generation_configs[0] == {}
        assert result.generation_configs[1] == {"temperature": 0.7}

    def test_compile_incremental_carries_generation_configs(self, tract: Tract):
        config1 = {"temperature": 0.3}
        config2 = {"temperature": 0.7}
        tract.commit(InstructionContent(text="System"), generation_config=config1)
        result1 = tract.compile()
        assert result1.generation_configs == [config1]

        tract.commit(DialogueContent(role="user", text="Hi"), generation_config=config2)
        result2 = tract.compile()
        assert result2.generation_configs == [config1, config2]

    # SC4: generation_config NOT in commit hash
    def test_generation_config_not_in_hash(self, tract: Tract):
        """Same content with different generation_configs should produce
        the same content_hash (different commit_hash due to timestamp/parent)."""
        c1 = tract.commit(
            DialogueContent(role="user", text="identical"),
            generation_config={"temperature": 0.1},
        )
        c2 = tract.commit(
            DialogueContent(role="user", text="identical"),
            generation_config={"temperature": 0.9},
        )
        # content_hash should be the same (same content)
        assert c1.content_hash == c2.content_hash
        # commit_hash differs because of parent_hash and timestamp, not config
        assert c1.commit_hash != c2.commit_hash

    # SC5: Query by config values
    @pytest.mark.parametrize("key,op,value,expected_count", [
        ("model", "=", "gpt-4o", 2),
        ("temperature", ">", 0.5, 2),
        ("temperature", ">", 0.9, 0),
    ])
    def test_query_by_config_operators(self, tract: Tract, key, op, value, expected_count):
        tract.commit(
            DialogueContent(role="user", text="a"),
            generation_config={"model": "gpt-4o", "temperature": 0.5},
        )
        tract.commit(
            DialogueContent(role="user", text="b"),
            generation_config={"model": "claude-3", "temperature": 0.9},
        )
        tract.commit(
            DialogueContent(role="user", text="c"),
            generation_config={"model": "gpt-4o", "temperature": 0.7},
        )
        results = tract.query_by_config(key, op, value)
        assert len(results) == expected_count

    def test_query_by_config_invalid_operator(self, tract: Tract):
        with pytest.raises(ValueError, match="Unsupported operator"):
            tract.query_by_config("temperature", "LIKE", 0.5)

    def test_query_by_config_commits_without_config_excluded(self, tract: Tract):
        tract.commit(DialogueContent(role="user", text="no config"))
        tract.commit(
            DialogueContent(role="user", text="with config"),
            generation_config={"temperature": 0.7},
        )
        results = tract.query_by_config("temperature", "=", 0.7)
        assert len(results) == 1

    # Cache safety: copy-on-output prevents corruption
    def test_compile_cache_not_corrupted_by_mutation(self, tract: Tract):
        """Mutating generation_configs on a returned CompiledContext
        should not affect subsequent compile() results."""
        config = {"temperature": 0.5}
        tract.commit(DialogueContent(role="user", text="Hi"), generation_config=config)
        result1 = tract.compile()
        # Mutate the returned dict
        result1.generation_configs[0]["temperature"] = 999
        # Compile again -- should return clean copy from cache
        result2 = tract.compile()
        assert result2.generation_configs[0] == config


# ===========================================================================
# LRU Compile Cache and Snapshot Patching (Phase 1.4)
# ===========================================================================


class TestLRUCompileCacheAndPatching:
    """Tests for LRU compile cache, EDIT patching, annotate patching, and oracle verification."""

    def test_lru_cache_hit_at_same_head(self):
        """Two compile() calls at same HEAD: second is cache hit."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="hello"))
            r1 = t.compile()
            r2 = t.compile()
            assert r1.messages == r2.messages
            assert r1.token_count == r2.token_count

    def test_lru_multiple_heads_cached(self):
        """After compiling at different HEADs, both snapshots are in cache."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="first"))
            head_1 = t.head
            t.compile()
            t.commit(DialogueContent(role="user", text="second"))
            head_2 = t.head
            t.compile()
            assert t._cache_get(head_1) is not None
            assert t._cache_get(head_2) is not None

    def test_lru_eviction_at_maxsize(self):
        """Cache evicts LRU entry when maxsize is exceeded."""
        config = TractConfig(compile_cache_maxsize=2)
        with Tract.open(config=config) as t:
            t.commit(InstructionContent(text="a"))
            h1 = t.head
            t.compile()
            t.commit(DialogueContent(role="user", text="b"))
            h2 = t.head
            t.compile()
            t.commit(DialogueContent(role="assistant", text="c"))
            h3 = t.head
            t.compile()
            assert t._cache_get(h1) is None
            assert t._cache_get(h2) is not None
            assert t._cache_get(h3) is not None

    def test_append_incremental_extends_snapshot(self):
        """APPEND commits extend the cached snapshot incrementally."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="system"))
            t.compile()
            t.commit(DialogueContent(role="user", text="hello"))
            r = t.compile()
            assert r.commit_count == 2
            assert len(r.messages) == 2

    def test_edit_patching_matches_full_recompile(self):
        """EDIT patching produces identical result to full recompile (oracle test)."""
        with Tract.open(verify_cache=True) as t:
            c1 = t.commit(InstructionContent(text="System prompt"))
            c2 = t.commit(DialogueContent(role="user", text="Original question"))
            c3 = t.commit(DialogueContent(role="assistant", text="Original answer"))
            t.compile()  # Populate cache with commit_hashes

            t.commit(
                DialogueContent(role="user", text="Edited question"),
                operation=CommitOperation.EDIT,
                response_to=c2.commit_hash,
            )
            # verify_cache=True asserts patched == fresh
            result = t.compile()
            assert any("Edited question" in m.content for m in result.messages)
            assert not any("Original question" in m.content for m in result.messages)

    def test_edit_patching_preserves_generation_config(self):
        """EDIT without generation_config preserves original's config."""
        with Tract.open(verify_cache=True) as t:
            c1 = t.commit(
                InstructionContent(text="System"),
                generation_config={"temperature": 0.7},
            )
            t.compile()

            t.commit(
                InstructionContent(text="Edited system"),
                operation=CommitOperation.EDIT,
                response_to=c1.commit_hash,
            )
            result = t.compile()
            assert result.generation_configs[0] == {"temperature": 0.7}

    def test_edit_patching_with_new_config(self):
        """EDIT with its own generation_config replaces the original's config."""
        with Tract.open(verify_cache=True) as t:
            c1 = t.commit(
                InstructionContent(text="System"),
                generation_config={"temperature": 0.7},
            )
            t.compile()

            t.commit(
                InstructionContent(text="Edited system"),
                operation=CommitOperation.EDIT,
                response_to=c1.commit_hash,
                generation_config={"temperature": 0.9},
            )
            result = t.compile()
            assert result.generation_configs[0] == {"temperature": 0.9}

    def test_annotate_skip_removes_message(self):
        """Annotating with SKIP patches snapshot by removing message."""
        with Tract.open(verify_cache=True) as t:
            c1 = t.commit(InstructionContent(text="System"))
            c2 = t.commit(DialogueContent(role="user", text="Hello"))
            c3 = t.commit(DialogueContent(role="assistant", text="Hi"))
            t.compile()

            t.annotate(c2.commit_hash, Priority.SKIP)
            result = t.compile()
            assert result.commit_count == 2  # c1 and c3 only
            assert not any("Hello" in m.content for m in result.messages)

    def test_annotate_unskip_falls_back_to_recompile(self):
        """Un-skipping a commit triggers full recompile."""
        with Tract.open(verify_cache=True) as t:
            c1 = t.commit(InstructionContent(text="System"))
            c2 = t.commit(DialogueContent(role="user", text="Hello"))
            t.annotate(c2.commit_hash, Priority.SKIP)
            t.compile()

            t.annotate(c2.commit_hash, Priority.NORMAL)
            result = t.compile()
            assert result.commit_count == 2
            assert any("Hello" in m.content for m in result.messages)

    def test_annotate_clears_stale_cache_entries(self):
        """Annotation clears other cached snapshots (they may contain the annotated commit)."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="System"))
            h1 = t.head
            t.compile()  # Cache entry for h1
            t.commit(DialogueContent(role="user", text="Hello"))
            h2 = t.head
            t.compile()  # Cache entry for h2
            assert t._cache_get(h1) is not None
            assert t._cache_get(h2) is not None

            # Annotate a commit -- should clear h1 (stale), keep patched h2
            t.annotate(h1, Priority.SKIP)
            assert t._cache_get(h1) is None  # Cleared
            assert t._cache_get(h2) is not None  # Patched and re-added

    def test_batch_clears_entire_cache(self):
        """batch() clears the entire LRU cache."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="first"))
            t.compile()
            assert len(t._snapshot_cache) == 1
            with t.batch():
                t.commit(DialogueContent(role="user", text="batched"))
            assert len(t._snapshot_cache) == 0

    def test_record_usage_updates_correct_cache_entry(self):
        """record_usage() updates token count in the LRU cache entry."""
        with Tract.open() as t:
            t.commit(InstructionContent(text="hello"))
            t.compile()

            updated = t.record_usage({"prompt_tokens": 42, "completion_tokens": 10, "total_tokens": 52})
            assert updated.token_count == 42
            assert "api:" in updated.token_source

            cached = t._cache_get(t.head)
            assert cached is not None
            assert cached.token_count == 42

    def test_commit_hashes_parallel_to_messages(self):
        """commit_hashes and messages have same length after all operations."""
        with Tract.open() as t:
            c1 = t.commit(InstructionContent(text="a"))
            c2 = t.commit(DialogueContent(role="user", text="b"))
            result = t.compile()
            assert len(result.commit_hashes) == len(result.messages) == 2
            assert result.commit_hashes[0] == c1.commit_hash
            assert result.commit_hashes[1] == c2.commit_hash

    def test_commit_hashes_extend_on_append(self):
        """Incremental APPEND extends commit_hashes."""
        with Tract.open() as t:
            c1 = t.commit(InstructionContent(text="a"))
            t.compile()
            c2 = t.commit(DialogueContent(role="user", text="b"))
            snapshot = t._cache_get(t.head)
            assert snapshot is not None
            assert len(snapshot.commit_hashes) == 2
            assert snapshot.commit_hashes[1] == c2.commit_hash

    def test_consecutive_same_role_with_edit_patching(self):
        """EDIT patching works correctly with consecutive same-role messages (no aggregation)."""
        with Tract.open(verify_cache=True) as t:
            c1 = t.commit(DialogueContent(role="user", text="Q1"))
            c2 = t.commit(DialogueContent(role="user", text="Q2"))
            c3 = t.commit(DialogueContent(role="assistant", text="A1"))
            t.compile()

            # Edit c1's content
            t.commit(
                DialogueContent(role="user", text="Edited Q1"),
                operation=CommitOperation.EDIT,
                response_to=c1.commit_hash,
            )
            result = t.compile()
            # 3 messages: "Edited Q1", "Q2", "A1" (no aggregation)
            assert len(result.messages) == 3
            assert result.messages[0].content == "Edited Q1"
            assert result.messages[1].content == "Q2"
            assert result.messages[2].content == "A1"
