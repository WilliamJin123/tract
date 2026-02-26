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
        """generate() should auto-create compile records (tiktoken + API)."""
        t = Tract.open()
        mock = MockLLMClient(responses=["I am helpful!"])
        t.configure_llm(mock)
        t.system("You are helpful.")
        t.user("Hello")

        t.generate()

        records = t.compile_records()
        # generate() creates 2 records: tiktoken (at compile) + API (from record_usage)
        assert len(records) == 2
        tiktoken_recs = [r for r in records if r.token_source.startswith("tiktoken:")]
        api_recs = [r for r in records if r.token_source.startswith("api:")]
        assert len(tiktoken_recs) == 1
        assert len(api_recs) == 1
        rec = tiktoken_recs[0]
        assert rec.head_hash != ""
        assert rec.token_count > 0
        assert rec.commit_count > 0
        t.close()

    def test_chat_creates_compile_record(self):
        """chat() delegates to generate(), so it should also create records."""
        t = Tract.open()
        mock = MockLLMClient(responses=["Hi there!"])
        t.configure_llm(mock)
        t.system("You are a chatbot.")

        t.chat("hello")

        records = t.compile_records()
        # chat -> generate creates tiktoken + API records
        assert len(records) == 2
        tiktoken_recs = [r for r in records if r.token_source.startswith("tiktoken:")]
        assert len(tiktoken_recs) == 1
        rec = tiktoken_recs[0]
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
        assert len(records) == 2  # tiktoken + API
        # Use the tiktoken record (created at compile time, before assistant commit)
        tiktoken_rec = [r for r in records if r.token_source.startswith("tiktoken:")][0]

        effectives = t.compile_record_commits(tiktoken_rec.record_id)
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
        """Each generate() call should create its own records."""
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
        # Each generate() creates 2 records: 1 compile + 1 record_usage.
        # First generate: tiktoken compile + API record_usage = 2.
        # Subsequent: cache propagates API token_source, so compile record
        # is also API-sourced, plus record_usage = 2 each.
        assert len(records) == 6
        # Commit counts should increase (newest first)
        assert records[0].commit_count > records[-1].commit_count
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
        assert len(records) == 2  # tiktoken + API
        assert all(r.params_json is None for r in records)
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
            assert len(records1) == 2  # tiktoken + API
            # Pick the tiktoken record for persistence check
            tiktoken_rec = [r for r in records1 if r.token_source.startswith("tiktoken:")][0]
            record_id = tiktoken_rec.record_id
            t1.close()

            # Session 2: verify records persist
            t2 = Tract.open(db_path, tract_id="t1")
            records2 = t2.compile_records()
            assert len(records2) == 2
            record_ids = {r.record_id for r in records2}
            assert record_id in record_ids
            assert tiktoken_rec.token_count > 0

            # Also verify effective commits persist
            effectives = t2.compile_record_commits(record_id)
            assert len(effectives) > 0
            t2.close()

    def test_compile_record_head_hash_matches_head(self):
        """Tiktoken record head_hash should match the tract head at compile time."""
        t = Tract.open()
        mock = MockLLMClient(responses=["resp"])
        t.configure_llm(mock)
        t.system("System")
        t.user("Q")

        head_before = t.head
        t.generate()

        records = t.compile_records()
        # The tiktoken record is captured BEFORE generate commits the assistant
        # response, so its head_hash matches the head at compile time.
        tiktoken_rec = [r for r in records if r.token_source.startswith("tiktoken:")][0]
        assert tiktoken_rec.head_hash == head_before
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
        assert len(all_records) == 10  # 2 per generate (tiktoken + API)

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


# ---------------------------------------------------------------------------
# API token persistence tests
# ---------------------------------------------------------------------------

class TestAPITokenPersistence:
    """Tests for record_usage() persisting API tokens as compile records."""

    def test_record_usage_creates_api_compile_record(self):
        """record_usage() should persist an API compile record."""
        t = Tract.open()
        t.system("System")
        t.user("Hello")
        t.assistant("Hi")

        t.record_usage({"prompt_tokens": 50, "completion_tokens": 20})

        records = t.compile_records()
        api_records = [r for r in records if r.token_source.startswith("api:")]
        assert len(api_records) == 1
        rec = api_records[0]
        assert rec.token_count == 70
        assert rec.token_source == "api:50+20"
        assert rec.head_hash == t.head
        t.close()

    def test_token_checkpoints_returns_only_api_records(self):
        """After manual commits + record_usage(), only API records returned."""
        t = Tract.open()
        t.system("System")
        t.user("Q")
        t.assistant("A")

        # compile() creates a tiktoken record? No — manual compile() does not.
        # record_usage() creates an API record
        t.record_usage({"prompt_tokens": 100, "completion_tokens": 30})

        checkpoints = t.token_checkpoints()
        assert len(checkpoints) == 1
        assert checkpoints[0].token_source.startswith("api:")
        assert checkpoints[0].token_count == 130

        # All records should be just the API one (no tiktoken from manual commits)
        all_records = t.compile_records()
        assert len(all_records) == 1
        t.close()

    def test_token_checkpoints_empty_when_no_api_usage(self):
        """Only tiktoken records exist — token_checkpoints() returns []."""
        t = Tract.open()
        t.system("System")
        t.user("Q")

        # Manual compile does not create any compile records
        t.compile()

        checkpoints = t.token_checkpoints()
        assert checkpoints == []
        t.close()

    def test_record_usage_api_record_has_effective_commits(self):
        """API compile record should have effective commits linked."""
        t = Tract.open()
        t.system("System")
        t.user("Hello")
        t.assistant("Hi")

        compiled = t.compile()
        expected_hashes = list(compiled.commit_hashes)

        t.record_usage({"prompt_tokens": 40, "completion_tokens": 10})

        checkpoints = t.token_checkpoints()
        assert len(checkpoints) == 1
        effectives = t.compile_record_commits(checkpoints[0].record_id)
        assert len(effectives) == len(expected_hashes)
        assert effectives == expected_hashes
        t.close()

    def test_token_checkpoints_respects_limit(self):
        """limit=0 means all, limit=2 caps at 2."""
        t = Tract.open()
        t.system("System")

        # Create 3 API records
        for i in range(3):
            t.user(f"Q{i}")
            t.assistant(f"A{i}")
            t.record_usage({"prompt_tokens": 10 * (i + 1), "completion_tokens": 5})

        all_checkpoints = t.token_checkpoints(limit=0)
        assert len(all_checkpoints) == 3

        limited = t.token_checkpoints(limit=2)
        assert len(limited) == 2
        # Newest first
        assert limited[0].record_id == all_checkpoints[0].record_id
        t.close()

    def test_record_usage_no_compile_record_repo_still_works(self):
        """record_usage() works gracefully when repo is None."""
        t = Tract.open()
        t.system("System")
        t.user("Hello")
        t.assistant("Hi")
        t._compile_record_repo = None

        result = t.record_usage({"prompt_tokens": 50, "completion_tokens": 20})
        assert result.token_count == 70
        assert result.token_source == "api:50+20"

        # token_checkpoints also graceful
        assert t.token_checkpoints() == []
        t.close()
