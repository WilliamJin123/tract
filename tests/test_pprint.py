"""Tests for pretty-print support on tract output objects.

Tests __str__ and pprint() for ChatResponse, CompiledContext, CommitInfo,
and StatusInfo.
"""
from __future__ import annotations

from io import StringIO
from datetime import datetime, timezone

import pytest

from tract import Tract
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.config import LLMConfig
from tract.operations.history import StatusInfo
from tract.protocols import (
    ChatResponse,
    CompiledContext,
    Message,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_chat_response() -> ChatResponse:
    """A ChatResponse with all fields populated."""
    return ChatResponse(
        text="Hello, how can I help you today?",
        usage=TokenUsage(prompt_tokens=42, completion_tokens=18, total_tokens=60),
        commit_info=CommitInfo(
            commit_hash="abc12345deadbeef",
            tract_id="test-tract",
            parent_hash=None,
            content_hash="hash123",
            content_type="dialogue",
            operation=CommitOperation.APPEND,
            message="Hello, how can I help you today?",
            token_count=18,
            created_at=datetime(2026, 2, 21, tzinfo=timezone.utc),
        ),
        generation_config=LLMConfig(model="gpt-4o", temperature=0.7),
    )


@pytest.fixture
def sample_compiled_context() -> CompiledContext:
    """A CompiledContext with several messages."""
    return CompiledContext(
        messages=[
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="What is Python?"),
            Message(
                role="assistant",
                content="Python is a high-level programming language known for its readability.",
            ),
        ],
        token_count=150,
        commit_count=3,
        token_source="tiktoken:o200k_base",
        commit_hashes=["aaa", "bbb", "ccc"],
    )


@pytest.fixture
def sample_commit_info() -> CommitInfo:
    """A CommitInfo with typical fields."""
    return CommitInfo(
        commit_hash="deadbeef12345678",
        tract_id="test-tract",
        parent_hash="00000000aaaabbbb",
        content_hash="content123",
        content_type="dialogue",
        operation=CommitOperation.APPEND,
        message="Explain quantum computing in simple terms",
        token_count=45,
        generation_config=LLMConfig(model="gpt-4o", temperature=0.5),
        created_at=datetime(2026, 2, 21, 12, 30, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def sample_status_info() -> StatusInfo:
    """A StatusInfo with budget."""
    return StatusInfo(
        head_hash="deadbeef12345678",
        branch_name="main",
        is_detached=False,
        commit_count=5,
        token_count=500,
        token_budget_max=2000,
        token_source="tiktoken:o200k_base",
    )


# ---------------------------------------------------------------------------
# __str__ tests
# ---------------------------------------------------------------------------

class TestChatResponseStr:
    def test_str_returns_text(self, sample_chat_response: ChatResponse) -> None:
        assert str(sample_chat_response) == "Hello, how can I help you today?"

    def test_str_empty_text(self) -> None:
        resp = ChatResponse(
            text="",
            usage=None,
            commit_info=CommitInfo(
                commit_hash="abc12345",
                tract_id="t",
                content_hash="h",
                content_type="dialogue",
                operation=CommitOperation.APPEND,
                message="",
                token_count=0,
                created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            ),
            generation_config=LLMConfig(),
        )
        assert str(resp) == ""


class TestCompiledContextStr:
    def test_str_contains_messages_and_tokens(
        self, sample_compiled_context: CompiledContext
    ) -> None:
        s = str(sample_compiled_context)
        assert "messages=3" in s
        assert "tokens=150" in s
        assert "tiktoken:o200k_base" in s

    def test_str_empty_context(self) -> None:
        ctx = CompiledContext()
        s = str(ctx)
        assert "messages=0" in s
        assert "tokens=0" in s


class TestCommitInfoStr:
    def test_str_starts_with_short_hash(self, sample_commit_info: CommitInfo) -> None:
        s = str(sample_commit_info)
        assert s.startswith("deadbeef")

    def test_str_contains_message(self, sample_commit_info: CommitInfo) -> None:
        s = str(sample_commit_info)
        assert "Explain quantum computing" in s

    def test_str_truncates_long_message(self) -> None:
        info = CommitInfo(
            commit_hash="aabbccdd11223344",
            tract_id="t",
            content_hash="h",
            content_type="dialogue",
            operation=CommitOperation.APPEND,
            message="A" * 100,
            token_count=10,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        s = str(info)
        assert len(s) <= 8 + 1 + 60  # hash + space + truncated message
        assert s.endswith("...")

    def test_str_no_message(self) -> None:
        info = CommitInfo(
            commit_hash="aabbccdd11223344",
            tract_id="t",
            content_hash="h",
            content_type="dialogue",
            operation=CommitOperation.APPEND,
            message=None,
            token_count=10,
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        s = str(info)
        assert s == "aabbccdd "


class TestStatusInfoStr:
    def test_str_contains_branch_and_commits(
        self, sample_status_info: StatusInfo
    ) -> None:
        s = str(sample_status_info)
        assert "main" in s
        assert "5 commits" in s
        assert "500" in s

    def test_str_with_budget_percentage(self, sample_status_info: StatusInfo) -> None:
        s = str(sample_status_info)
        assert "/2000" in s
        assert "25%" in s

    def test_str_no_budget(self) -> None:
        status = StatusInfo(
            head_hash="deadbeef12345678",
            branch_name="feature",
            is_detached=False,
            commit_count=3,
            token_count=200,
            token_budget_max=None,
            token_source="tiktoken:o200k_base",
        )
        s = str(status)
        assert "feature" in s
        assert "200" in s
        assert "/" not in s

    def test_str_detached(self) -> None:
        status = StatusInfo(
            head_hash="deadbeef12345678",
            branch_name=None,
            is_detached=True,
            commit_count=1,
            token_count=50,
            token_budget_max=None,
            token_source="tiktoken:o200k_base",
        )
        s = str(status)
        assert "detached" in s

    def test_str_no_head(self) -> None:
        status = StatusInfo(
            head_hash=None,
            branch_name="main",
            is_detached=False,
            commit_count=0,
            token_count=0,
            token_budget_max=None,
            token_source="tiktoken:o200k_base",
        )
        s = str(status)
        assert "None" in s


# ---------------------------------------------------------------------------
# pprint() tests
# ---------------------------------------------------------------------------

class TestChatResponsePprint:
    def test_pprint_runs_without_error(
        self, sample_chat_response: ChatResponse
    ) -> None:
        """pprint() should not raise."""
        buf = StringIO()
        from tract.formatting import pprint_chat_response
        pprint_chat_response(sample_chat_response, file=buf)
        output = buf.getvalue()
        assert len(output) > 0

    def test_pprint_contains_assistant_title(
        self, sample_chat_response: ChatResponse
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_chat_response
        pprint_chat_response(sample_chat_response, file=buf)
        output = buf.getvalue()
        assert "Assistant" in output

    def test_pprint_contains_text(
        self, sample_chat_response: ChatResponse
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_chat_response
        pprint_chat_response(sample_chat_response, file=buf)
        output = buf.getvalue()
        assert "Hello, how can I help you today?" in output

    def test_pprint_contains_usage(
        self, sample_chat_response: ChatResponse
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_chat_response
        pprint_chat_response(sample_chat_response, file=buf)
        output = buf.getvalue()
        assert "42 prompt" in output
        assert "18 completion" in output
        assert "60 tokens" in output

    def test_pprint_contains_config(
        self, sample_chat_response: ChatResponse
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_chat_response
        pprint_chat_response(sample_chat_response, file=buf)
        output = buf.getvalue()
        assert "gpt-4o" in output
        assert "0.7" in output

    def test_pprint_method_delegates(
        self, sample_chat_response: ChatResponse
    ) -> None:
        """The .pprint() method on the object should work."""
        # Just verify it doesn't crash -- output goes to stdout
        sample_chat_response.pprint()


class TestCompiledContextPprint:
    def test_pprint_runs_without_error(
        self, sample_compiled_context: CompiledContext
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_compiled_context
        pprint_compiled_context(sample_compiled_context, file=buf)
        output = buf.getvalue()
        assert len(output) > 0

    def test_pprint_contains_roles(
        self, sample_compiled_context: CompiledContext
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_compiled_context
        pprint_compiled_context(sample_compiled_context, file=buf)
        output = buf.getvalue()
        assert "system" in output
        assert "user" in output
        assert "assistant" in output

    def test_pprint_contains_token_summary(
        self, sample_compiled_context: CompiledContext
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_compiled_context
        pprint_compiled_context(sample_compiled_context, file=buf)
        output = buf.getvalue()
        assert "150" in output
        assert "3" in output

    def test_pprint_method_delegates(
        self, sample_compiled_context: CompiledContext
    ) -> None:
        sample_compiled_context.pprint()


class TestCommitInfoPprint:
    def test_pprint_runs_without_error(
        self, sample_commit_info: CommitInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_commit_info
        pprint_commit_info(sample_commit_info, file=buf)
        output = buf.getvalue()
        assert len(output) > 0

    def test_pprint_contains_hash(
        self, sample_commit_info: CommitInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_commit_info
        pprint_commit_info(sample_commit_info, file=buf)
        output = buf.getvalue()
        assert "deadbeef" in output

    def test_pprint_contains_operation(
        self, sample_commit_info: CommitInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_commit_info
        pprint_commit_info(sample_commit_info, file=buf)
        output = buf.getvalue()
        assert "append" in output

    def test_pprint_contains_config(
        self, sample_commit_info: CommitInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_commit_info
        pprint_commit_info(sample_commit_info, file=buf)
        output = buf.getvalue()
        assert "gpt-4o" in output

    def test_pprint_method_delegates(
        self, sample_commit_info: CommitInfo
    ) -> None:
        sample_commit_info.pprint()


class TestStatusInfoPprint:
    def test_pprint_runs_without_error(
        self, sample_status_info: StatusInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_status_info
        pprint_status_info(sample_status_info, file=buf)
        output = buf.getvalue()
        assert len(output) > 0

    def test_pprint_contains_branch(
        self, sample_status_info: StatusInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_status_info
        pprint_status_info(sample_status_info, file=buf)
        output = buf.getvalue()
        assert "main" in output

    def test_pprint_contains_token_budget(
        self, sample_status_info: StatusInfo
    ) -> None:
        buf = StringIO()
        from tract.formatting import pprint_status_info
        pprint_status_info(sample_status_info, file=buf)
        output = buf.getvalue()
        assert "500" in output
        assert "2000" in output
        assert "25%" in output

    def test_pprint_detached_warning(self) -> None:
        status = StatusInfo(
            head_hash="deadbeef12345678",
            branch_name=None,
            is_detached=True,
            commit_count=1,
            token_count=50,
            token_budget_max=None,
            token_source="tiktoken:o200k_base",
        )
        buf = StringIO()
        from tract.formatting import pprint_status_info
        pprint_status_info(status, file=buf)
        output = buf.getvalue()
        assert "detached" in output.lower()

    def test_pprint_method_delegates(
        self, sample_status_info: StatusInfo
    ) -> None:
        sample_status_info.pprint()


# ---------------------------------------------------------------------------
# Integration: real Tract objects
# ---------------------------------------------------------------------------

class TestIntegrationWithTract:
    """Test pprint on objects produced by actual Tract operations."""

    def test_commit_info_from_tract(self, tmp_path: object) -> None:
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("You are a helpful assistant.")
        t.user("Hello!")
        log = t.log()
        assert len(log) >= 1

        # __str__
        s = str(log[0])
        assert len(s) > 8  # at least the hash prefix

        # pprint
        buf = StringIO()
        from tract.formatting import pprint_commit_info
        pprint_commit_info(log[0], file=buf)
        assert len(buf.getvalue()) > 0

    def test_status_info_from_tract(self, tmp_path: object) -> None:
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("You are a helpful assistant.")
        t.user("Hello!")
        status = t.status()

        # __str__
        s = str(status)
        assert "main" in s
        assert "2 commits" in s

        # pprint
        buf = StringIO()
        from tract.formatting import pprint_status_info
        pprint_status_info(status, file=buf)
        assert "main" in buf.getvalue()

    def test_compiled_context_from_tract(self, tmp_path: object) -> None:
        t = Tract.open(str(tmp_path / "test.db"))
        t.system("You are a helpful assistant.")
        t.user("What is Python?")
        ctx = t.compile()

        # __str__
        s = str(ctx)
        assert "messages=2" in s
        assert "tokens=" in s

        # pprint
        buf = StringIO()
        from tract.formatting import pprint_compiled_context
        pprint_compiled_context(ctx, file=buf)
        output = buf.getvalue()
        assert "system" in output
        assert "user" in output
