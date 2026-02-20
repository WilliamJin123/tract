"""Tests for compile record persistence via chat()/generate().

Compile records track what was compiled (sent to the LLM): head hash,
token count, commit count, token source, and the ordered effective commits.
Only chat()/generate() create records; manual compile() does not.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from tract import Tract


# ---------------------------------------------------------------------------
# MockLLMClient -- predictable LLM responses for testing
# ---------------------------------------------------------------------------

class MockLLMClient:
    """Minimal mock conforming to the LLMClient protocol."""

    def __init__(self, responses=None, model="mock-model"):
        self.responses = responses or ["Mock response"]
        self._call_count = 0
        self.last_messages = None
        self.last_kwargs: dict = {}
        self._model = model

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            "model": kwargs.get("model", self._model),
        }

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Compile record tests
# ---------------------------------------------------------------------------

class TestCompileRecords:
    """Tests for compile record creation and querying."""

    def test_generate_creates_compile_record(self):
        """generate() should auto-create a compile record."""
        t = Tract.open()
        mock = MockLLMClient(responses=["I am helpful!"])
        t.configure_llm(mock)
        t.system("You are helpful.")
        t.user("Hello")

        t.generate()

        records = t.compile_records()
        assert len(records) == 1
        rec = records[0]
        assert rec.head_hash != ""
        assert rec.token_count > 0
        assert rec.commit_count > 0
        assert rec.token_source.startswith("tiktoken:")
        t.close()

    def test_chat_creates_compile_record(self):
        """chat() delegates to generate(), so it should also create a record."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Hi there!"])
        t.configure_llm(mock)
        t.system("You are a chatbot.")

        t.chat("hello")

        records = t.compile_records()
        assert len(records) == 1
        rec = records[0]
        assert rec.token_count > 0
        assert rec.commit_count > 0
        t.close()

    def test_compile_record_has_correct_effective_commits(self):
        """Effective commits should match the commit hashes from compilation."""
        t = Tract.open()
        mock = MockLLMClient(responses=["response"])
        t.configure_llm(mock)

        # Commit 3 messages
        t.system("System prompt")
        t.user("First question")
        t.user("Second question")

        # Compile to capture what we expect
        compiled = t.compile()
        expected_hashes = list(compiled.commit_hashes)

        t.generate()

        records = t.compile_records()
        assert len(records) == 1
        rec = records[0]

        effectives = t.compile_record_commits(rec.record_id)
        # The effective commits should match the compile output
        # (generate compiles again, but after the same 3 messages, hashes match)
        assert len(effectives) == len(expected_hashes)
        assert effectives == expected_hashes
        t.close()

    def test_manual_compile_does_not_create_record(self):
        """Manual compile() should NOT create a compile record."""
        t = Tract.open()
        t.system("System prompt")
        t.user("Question")

        t.compile()

        records = t.compile_records()
        assert len(records) == 0
        t.close()

    def test_multiple_generates_create_multiple_records(self):
        """Each generate() call should create its own record."""
        t = Tract.open()
        mock = MockLLMClient(responses=["r1", "r2", "r3"])
        t.configure_llm(mock)

        t.system("System")
        t.user("Q1")
        t.generate()

        t.user("Q2")
        t.generate()

        t.user("Q3")
        t.generate()

        records = t.compile_records()
        assert len(records) == 3
        # Newest first
        assert records[0].commit_count > records[2].commit_count
        t.close()

    def test_compile_record_params_json_is_none(self):
        """Standard compile should have params_json as None."""
        t = Tract.open()
        mock = MockLLMClient(responses=["response"])
        t.configure_llm(mock)
        t.system("System")
        t.user("Q")

        t.generate()

        records = t.compile_records()
        assert len(records) == 1
        assert records[0].params_json is None
        t.close()

    def test_compile_record_survives_session(self):
        """Compile records should persist across open/close cycles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test.db")

            # Session 1: create records
            t1 = Tract.open(db_path, tract_id="t1")
            mock = MockLLMClient(responses=["r1"])
            t1.configure_llm(mock)
            t1.system("System")
            t1.user("Q")
            t1.generate()

            records1 = t1.compile_records()
            assert len(records1) == 1
            record_id = records1[0].record_id
            t1.close()

            # Session 2: verify records persist
            t2 = Tract.open(db_path, tract_id="t1")
            records2 = t2.compile_records()
            assert len(records2) == 1
            assert records2[0].record_id == record_id
            assert records2[0].token_count > 0

            # Also verify effective commits persist
            effectives = t2.compile_record_commits(record_id)
            assert len(effectives) > 0
            t2.close()

    def test_compile_record_head_hash_matches_head(self):
        """Record head_hash should match the tract head at compile time."""
        t = Tract.open()
        mock = MockLLMClient(responses=["resp"])
        t.configure_llm(mock)
        t.system("System")
        t.user("Q")

        head_before = t.head
        t.generate()

        records = t.compile_records()
        # head_hash is captured BEFORE generate commits the assistant response,
        # so it should match the head at compile time (= after user commit)
        assert records[0].head_hash == head_before
        t.close()

    def test_compile_records_limit(self):
        """compile_records() should respect the limit parameter."""
        t = Tract.open()
        mock = MockLLMClient(responses=["r"] * 5)
        t.configure_llm(mock)

        t.system("System")
        for i in range(5):
            t.user(f"Q{i}")
            t.generate()

        all_records = t.compile_records(limit=100)
        assert len(all_records) == 5

        limited = t.compile_records(limit=2)
        assert len(limited) == 2
        # Should be newest first, so limited should be the 2 newest
        assert limited[0].record_id == all_records[0].record_id
        assert limited[1].record_id == all_records[1].record_id
        t.close()

    def test_compile_record_commits_empty_for_unknown_id(self):
        """compile_record_commits() returns [] for unknown record_id."""
        t = Tract.open()
        result = t.compile_record_commits("nonexistent-id")
        assert result == []
        t.close()

    def test_compile_records_no_repo(self):
        """compile_records() returns [] when no compile_record_repo is set."""
        t = Tract.open()
        # Simulate no repo (e.g., from_components without it)
        t._compile_record_repo = None
        assert t.compile_records() == []
        assert t.compile_record_commits("any-id") == []
        t.close()
