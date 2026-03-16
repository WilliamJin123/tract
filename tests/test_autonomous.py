"""Tests for autonomous tract operations.

Tests auto_split, auto_rebase, auto_branch, and
Tract API integration using mock LLM clients.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import (
    AutoBranchResult,
    AutoRebaseResult,
    AutoSplitResult,
    Priority,
    Tract,
)
from tract.autonomous import (
    _parse_branch_response,
    _parse_rebase_response,
    _parse_split_response,
    auto_branch,
    auto_rebase,
    auto_split,
)


# ---------------------------------------------------------------------------
# Mock LLM clients
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Fake LLM client that returns a canned response."""

    def __init__(self, response_text: str):
        self.response_text = response_text
        self.calls: list[tuple[list[dict], dict]] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        self.calls.append((messages, kwargs))
        return {"choices": [{"message": {"content": self.response_text}}]}

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"]

    def extract_usage(self, response: dict) -> dict:
        return {"total_tokens": 42}

    def close(self) -> None:
        pass


class ErrorLLMClient:
    """LLM client that raises on every chat() call."""

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        raise ConnectionError("LLM service unavailable")

    def extract_content(self, response: dict) -> str:
        return ""

    def extract_usage(self, response: dict) -> dict:
        return {"total_tokens": 0}

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_commits(t: Tract, n: int = 5) -> list[str]:
    """Add *n* commits and return their hashes."""
    hashes = []
    for i in range(n):
        info = t.commit(
            {"content_type": "dialogue", "role": "user", "text": f"Message {i + 1}"},
            message=f"commit {i + 1}",
        )
        hashes.append(info.commit_hash)
    return hashes


# ===========================================================================
# auto_split tests
# ===========================================================================

class TestAutoSplit:
    """Tests for auto_split function."""

    def test_split_with_mock_llm(self):
        """auto_split creates new commits from LLM split pieces."""
        split_response = json.dumps({
            "reasoning": "Content has two distinct topics",
            "pieces": [
                {"content": "Authentication setup with JWT", "message": "Auth setup"},
                {"content": "Database schema for users", "message": "DB schema"},
            ],
        })
        mock = MockLLMClient(split_response)
        t = Tract.open()
        t.configure_llm(mock)
        hashes = _seed_commits(t, 3)

        result = auto_split(t, hashes[1])

        assert isinstance(result, AutoSplitResult)
        assert result.original_hash == hashes[1]
        assert result.split_count == 2
        assert len(result.new_hashes) == 2
        assert result.tokens_used == 42
        assert "Split into 2 pieces" in result.reasoning
        t.close()

    def test_split_fail_open_on_llm_error(self):
        """auto_split returns original hash on LLM error (fail-open)."""
        error_client = ErrorLLMClient()
        t = Tract.open()
        t.configure_llm(error_client)
        hashes = _seed_commits(t, 3)

        result = auto_split(t, hashes[1])

        assert isinstance(result, AutoSplitResult)
        assert result.original_hash == hashes[1]
        assert result.new_hashes == (hashes[1],)
        assert result.split_count == 1
        assert "fail-open" in result.reasoning.lower()
        t.close()

    def test_split_fail_open_no_client(self):
        """auto_split returns original hash when no LLM client configured."""
        t = Tract.open()
        hashes = _seed_commits(t, 3)

        result = auto_split(t, hashes[0])

        assert result.split_count == 1
        assert result.new_hashes == (hashes[0],)
        assert "no llm client" in result.reasoning.lower()
        t.close()

    def test_split_empty_pieces(self):
        """auto_split keeps original when LLM returns no pieces."""
        response = json.dumps({
            "reasoning": "Content is already atomic",
            "pieces": [],
        })
        mock = MockLLMClient(response)
        t = Tract.open()
        t.configure_llm(mock)
        hashes = _seed_commits(t, 2)

        result = auto_split(t, hashes[0])

        assert result.split_count == 1
        assert result.new_hashes == (hashes[0],)
        assert "no split" in result.reasoning.lower() or "keeping original" in result.reasoning.lower()
        t.close()

    def test_split_skips_original(self):
        """auto_split annotates the original commit as SKIP after splitting."""
        split_response = json.dumps({
            "reasoning": "Splitting",
            "pieces": [
                {"content": "Part one", "message": "First"},
                {"content": "Part two", "message": "Second"},
            ],
        })
        mock = MockLLMClient(split_response)
        t = Tract.open()
        t.configure_llm(mock)
        hashes = _seed_commits(t, 2)

        result = auto_split(t, hashes[0])

        assert result.split_count == 2
        # Verify the original commit got SKIP annotation
        log_entries = t.log(limit=50)
        original_entry = None
        for e in log_entries:
            if e.commit_hash == hashes[0]:
                original_entry = e
                break
        assert original_entry is not None
        assert original_entry.effective_priority.upper() == "SKIP"
        t.close()


# ===========================================================================
# auto_rebase tests
# ===========================================================================

class TestAutoRebase:
    """Tests for auto_rebase function."""

    def test_rebase_recommended(self):
        """auto_rebase executes rebase when LLM recommends it."""
        rebase_response = json.dumps({
            "reasoning": "Feature branch is behind main",
            "should_rebase": True,
            "target_branch": "main",
        })
        mock = MockLLMClient(rebase_response)
        t = Tract.open()
        t.configure_llm(mock)
        _seed_commits(t, 3)

        # Create a feature branch and add a commit
        t.branch("feature")
        t.commit(
            {"content_type": "dialogue", "role": "user", "text": "Feature work"},
            message="feature commit",
        )

        # Go back to main and add a commit
        t.checkout("main")
        t.commit(
            {"content_type": "dialogue", "role": "user", "text": "Main update"},
            message="main update",
        )

        # Switch to feature and auto_rebase
        t.checkout("feature")
        result = auto_rebase(t)

        assert isinstance(result, AutoRebaseResult)
        assert result.rebased is True
        assert result.target_branch == "main"
        assert result.tokens_used == 42
        t.close()

    def test_rebase_not_recommended(self):
        """auto_rebase does nothing when LLM says no rebase needed."""
        response = json.dumps({
            "reasoning": "Branch is up to date",
            "should_rebase": False,
            "target_branch": None,
        })
        mock = MockLLMClient(response)
        t = Tract.open()
        t.configure_llm(mock)
        _seed_commits(t, 3)

        result = auto_rebase(t)

        assert isinstance(result, AutoRebaseResult)
        assert result.rebased is False
        assert result.target_branch is None
        assert "up to date" in result.reason.lower()
        t.close()

    def test_rebase_fail_open_on_error(self):
        """auto_rebase returns rebased=False on LLM error."""
        error_client = ErrorLLMClient()
        t = Tract.open()
        t.configure_llm(error_client)
        _seed_commits(t, 3)

        result = auto_rebase(t)

        assert result.rebased is False
        assert "fail-open" in result.reason.lower()
        t.close()

    def test_rebase_fail_open_no_client(self):
        """auto_rebase returns rebased=False when no client configured."""
        t = Tract.open()
        _seed_commits(t, 3)

        result = auto_rebase(t)

        assert result.rebased is False
        assert "no llm client" in result.reason.lower()
        t.close()


# ===========================================================================
# auto_branch tests
# ===========================================================================

class TestAutoBranch:
    """Tests for auto_branch function."""

    def test_branch_recommended(self):
        """auto_branch creates branch when LLM recommends it."""
        response = json.dumps({
            "reasoning": "New topic requires isolation",
            "should_branch": True,
            "branch_name": "feature/auth-module",
        })
        mock = MockLLMClient(response)
        t = Tract.open()
        t.configure_llm(mock)
        _seed_commits(t, 3)

        result = auto_branch(t, context="Starting auth implementation")

        assert isinstance(result, AutoBranchResult)
        assert result.branched is True
        assert result.branch_name == "feature/auth-module"
        assert result.tokens_used == 42

        # Verify we're on the new branch
        assert t.current_branch == "feature/auth-module"
        t.close()

    def test_branch_not_recommended(self):
        """auto_branch does nothing when LLM says no branch needed."""
        response = json.dumps({
            "reasoning": "Current branch is appropriate",
            "should_branch": False,
            "branch_name": None,
        })
        mock = MockLLMClient(response)
        t = Tract.open()
        t.configure_llm(mock)
        _seed_commits(t, 3)

        result = auto_branch(t, context="Continuing current work")

        assert result.branched is False
        assert result.branch_name is None
        assert t.current_branch == "main"
        t.close()

    def test_branch_fail_open_on_error(self):
        """auto_branch returns branched=False on LLM error."""
        error_client = ErrorLLMClient()
        t = Tract.open()
        t.configure_llm(error_client)
        _seed_commits(t, 3)

        result = auto_branch(t, context="Some context")

        assert result.branched is False
        assert "fail-open" in result.reason.lower()
        t.close()

    def test_branch_fail_open_no_client(self):
        """auto_branch returns branched=False when no client configured."""
        t = Tract.open()
        _seed_commits(t, 3)

        result = auto_branch(t)

        assert result.branched is False
        assert "no llm client" in result.reason.lower()
        t.close()


# ===========================================================================
# Parse helpers tests
# ===========================================================================

class TestParseHelpers:
    """Tests for response parsing helpers."""

    def test_parse_split_response_valid(self):
        text = json.dumps({
            "reasoning": "Split",
            "pieces": [
                {"content": "A", "message": "Part A"},
                {"content": "B", "message": "Part B"},
            ],
        })
        pieces = _parse_split_response(text)
        assert len(pieces) == 2
        assert pieces[0]["content"] == "A"
        assert pieces[1]["message"] == "Part B"

    def test_parse_split_response_empty_pieces(self):
        text = json.dumps({"reasoning": "Atomic", "pieces": []})
        pieces = _parse_split_response(text)
        assert pieces == []

    def test_parse_split_response_invalid_json(self):
        pieces = _parse_split_response("not json")
        assert pieces == []

    def test_parse_split_response_fenced(self):
        inner = json.dumps({
            "reasoning": "Split",
            "pieces": [{"content": "X", "message": "Y"}],
        })
        text = f"```json\n{inner}\n```"
        pieces = _parse_split_response(text)
        assert len(pieces) == 1

    def test_parse_rebase_response_yes(self):
        text = json.dumps({
            "reasoning": "Behind main",
            "should_rebase": True,
            "target_branch": "main",
        })
        result = _parse_rebase_response(text)
        assert result is not None
        should, target, reasoning = result
        assert should is True
        assert target == "main"

    def test_parse_rebase_response_no(self):
        text = json.dumps({
            "reasoning": "Up to date",
            "should_rebase": False,
            "target_branch": None,
        })
        result = _parse_rebase_response(text)
        assert result is not None
        should, target, reasoning = result
        assert should is False
        assert target is None

    def test_parse_rebase_response_invalid(self):
        result = _parse_rebase_response("garbage")
        assert result is None

    def test_parse_branch_response_yes(self):
        text = json.dumps({
            "reasoning": "New topic",
            "should_branch": True,
            "branch_name": "feature/new-thing",
        })
        result = _parse_branch_response(text)
        assert result is not None
        should, name, reasoning = result
        assert should is True
        assert name == "feature/new-thing"

    def test_parse_branch_response_no(self):
        text = json.dumps({
            "reasoning": "Stay on current",
            "should_branch": False,
            "branch_name": None,
        })
        result = _parse_branch_response(text)
        assert result is not None
        should, name, reasoning = result
        assert should is False
        assert name is None

    def test_parse_branch_response_invalid(self):
        result = _parse_branch_response("not valid json")
        assert result is None


# ===========================================================================
# Tract API integration tests
# ===========================================================================

class TestTractIntegration:
    """Tests for Tract API methods that delegate to autonomous module."""

    def test_t_auto_split(self):
        """t.auto_split() delegates to auto_split function."""
        split_response = json.dumps({
            "reasoning": "Split content",
            "pieces": [
                {"content": "Part 1", "message": "First half"},
                {"content": "Part 2", "message": "Second half"},
            ],
        })
        mock = MockLLMClient(split_response)
        t = Tract.open()
        t.configure_llm(mock)
        hashes = _seed_commits(t, 3)

        result = t.auto_split(hashes[1])

        assert isinstance(result, AutoSplitResult)
        assert result.split_count == 2
        assert len(result.new_hashes) == 2
        t.close()

    def test_t_auto_rebase(self):
        """t.auto_rebase() delegates to auto_rebase function."""
        response = json.dumps({
            "reasoning": "No rebase needed",
            "should_rebase": False,
            "target_branch": None,
        })
        mock = MockLLMClient(response)
        t = Tract.open()
        t.configure_llm(mock)
        _seed_commits(t, 3)

        result = t.auto_rebase()

        assert isinstance(result, AutoRebaseResult)
        assert result.rebased is False
        t.close()

    def test_t_auto_branch(self):
        """t.auto_branch() delegates to auto_branch function."""
        response = json.dumps({
            "reasoning": "New feature needs isolation",
            "should_branch": True,
            "branch_name": "feature/test-branch",
        })
        mock = MockLLMClient(response)
        t = Tract.open()
        t.configure_llm(mock)
        _seed_commits(t, 3)

        result = t.auto_branch(context="Starting new feature")

        assert isinstance(result, AutoBranchResult)
        assert result.branched is True
        assert result.branch_name == "feature/test-branch"
        assert t.current_branch == "feature/test-branch"
        t.close()



# ===========================================================================
# Dataclass tests
# ===========================================================================

class TestDataclasses:
    """Tests for result dataclass properties."""

    def test_auto_split_result_frozen(self):
        r = AutoSplitResult(
            original_hash="abc",
            new_hashes=("def", "ghi"),
            split_count=2,
            tokens_used=10,
            reasoning="test",
        )
        with pytest.raises(AttributeError):
            r.split_count = 99  # type: ignore[misc]

    def test_auto_rebase_result_frozen(self):
        r = AutoRebaseResult(
            rebased=True,
            reason="test",
            target_branch="main",
            tokens_used=10,
        )
        with pytest.raises(AttributeError):
            r.rebased = False  # type: ignore[misc]

    def test_auto_branch_result_frozen(self):
        r = AutoBranchResult(
            branched=True,
            branch_name="feature/x",
            reason="test",
            tokens_used=10,
        )
        with pytest.raises(AttributeError):
            r.branched = False  # type: ignore[misc]
