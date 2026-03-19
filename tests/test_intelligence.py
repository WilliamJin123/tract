"""Tests for LLM-driven context intelligence: cherry-pick and deduplication.

Tests cherry_pick, deduplicate, manifest building, fail-open behavior,
and Tract API integration using mock LLM clients.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import CherryPickResult, DedupResult, Priority, Tract
from tract.intelligence import (
    _build_intelligence_manifest,
    _parse_cherry_pick_response,
    _parse_dedup_response,
    acherry_pick,
    adeduplicate,
    cherry_pick,
    deduplicate,
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


def _seed_duplicate_commits(t: Tract) -> list[str]:
    """Add commits with some duplicate content, return hashes."""
    hashes = []
    # Two commits with identical content
    info1 = t.commit(
        {"content_type": "dialogue", "role": "user", "text": "Implement the auth module with JWT tokens"},
        message="auth implementation plan",
    )
    hashes.append(info1.commit_hash)

    info2 = t.commit(
        {"content_type": "dialogue", "role": "assistant", "text": "Some unique response about testing"},
        message="testing discussion",
    )
    hashes.append(info2.commit_hash)

    info3 = t.commit(
        {"content_type": "dialogue", "role": "user", "text": "Implement the auth module with JWT tokens"},
        message="auth implementation plan (duplicate)",
    )
    hashes.append(info3.commit_hash)

    info4 = t.commit(
        {"content_type": "dialogue", "role": "assistant", "text": "Database schema design for users table"},
        message="database design",
    )
    hashes.append(info4.commit_hash)

    return hashes


# ---------------------------------------------------------------------------
# 1. Manifest building tests
# ---------------------------------------------------------------------------

class TestManifestBuilding:
    def test_build_manifest_empty_tract(self):
        """Manifest for empty tract returns 'no commits'."""
        with Tract.open() as t:
            manifest, entries = _build_intelligence_manifest(t)
            assert "(no commits)" in manifest
            assert entries == []

    def test_build_manifest_with_commits(self):
        """Manifest includes commit hashes and content types."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 3)
            manifest, entries = _build_intelligence_manifest(t)

            assert len(entries) == 3
            assert "CONTEXT MANIFEST" in manifest
            assert "dialogue" in manifest
            for h in hashes:
                assert h[:8] in manifest

    def test_build_manifest_with_content_preview(self):
        """Manifest includes content previews when enabled."""
        with Tract.open() as t:
            _seed_commits(t, 2)
            manifest, entries = _build_intelligence_manifest(
                t, include_content_preview=True, preview_length=50,
            )
            assert "Preview:" in manifest

    def test_build_manifest_without_content_preview(self):
        """Manifest omits content previews when disabled."""
        with Tract.open() as t:
            _seed_commits(t, 2)
            manifest, entries = _build_intelligence_manifest(
                t, include_content_preview=False,
            )
            assert "Preview:" not in manifest

    def test_build_manifest_respects_max_log_entries(self):
        """Manifest only includes up to max_log_entries commits."""
        with Tract.open() as t:
            _seed_commits(t, 10)
            manifest, entries = _build_intelligence_manifest(
                t, max_log_entries=3,
            )
            assert len(entries) == 3

    def test_build_manifest_entries_have_full_hashes(self):
        """Commit entries contain full hashes for resolution."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 2)
            _, entries = _build_intelligence_manifest(t)

            for entry in entries:
                assert len(entry["hash"]) > 8
                assert entry["short_hash"] == entry["hash"][:8]
                assert entry["content_type"] == "dialogue"
                assert "token_count" in entry


# ---------------------------------------------------------------------------
# 2. Response parsing tests
# ---------------------------------------------------------------------------

class TestCherryPickParsing:
    def test_parse_valid_json_response(self):
        """Valid JSON cherry-pick response is parsed correctly."""
        entries = [
            {"hash": "abcdef1234567890", "short_hash": "abcdef12"},
            {"hash": "fedcba0987654321", "short_hash": "fedcba09"},
            {"hash": "1111111122222222", "short_hash": "11111111"},
        ]
        response = json.dumps({
            "reasoning": "Selected auth-related commits",
            "selected": ["abcdef12", "fedcba09"],
        })
        reasoning, selected = _parse_cherry_pick_response(response, entries)
        assert reasoning == "Selected auth-related commits"
        assert selected == ["abcdef1234567890", "fedcba0987654321"]

    def test_parse_response_with_code_fences(self):
        """Response wrapped in code fences is parsed correctly."""
        entries = [{"hash": "abcdef1234567890", "short_hash": "abcdef12"}]
        response = '```json\n{"reasoning": "ok", "selected": ["abcdef12"]}\n```'
        reasoning, selected = _parse_cherry_pick_response(response, entries)
        assert selected == ["abcdef1234567890"]

    def test_parse_invalid_json_returns_empty(self):
        """Invalid JSON returns empty selection."""
        entries = [{"hash": "abcdef1234567890", "short_hash": "abcdef12"}]
        reasoning, selected = _parse_cherry_pick_response("not json at all", entries)
        assert selected == []
        assert "Could not parse" in reasoning


class TestDedupParsing:
    def test_parse_valid_dedup_response(self):
        """Valid JSON dedup response is parsed correctly."""
        entries = [
            {"hash": "aaaa111122223333", "short_hash": "aaaa1111"},
            {"hash": "bbbb111122223333", "short_hash": "bbbb1111"},
            {"hash": "cccc111122223333", "short_hash": "cccc1111"},
        ]
        response = json.dumps({
            "reasoning": "Found one duplicate group",
            "groups": [["aaaa1111", "bbbb1111"]],
        })
        reasoning, groups = _parse_dedup_response(response, entries)
        assert reasoning == "Found one duplicate group"
        assert len(groups) == 1
        assert groups[0] == ["aaaa111122223333", "bbbb111122223333"]

    def test_parse_dedup_filters_single_element_groups(self):
        """Groups with fewer than 2 members are filtered out."""
        entries = [{"hash": "aaaa111122223333", "short_hash": "aaaa1111"}]
        response = json.dumps({
            "reasoning": "Only one",
            "groups": [["aaaa1111"]],
        })
        _, groups = _parse_dedup_response(response, entries)
        assert groups == []

    def test_parse_dedup_invalid_json(self):
        """Invalid JSON returns empty groups."""
        entries = [{"hash": "aaaa111122223333", "short_hash": "aaaa1111"}]
        reasoning, groups = _parse_dedup_response("garbage", entries)
        assert groups == []
        assert "Could not parse" in reasoning


# ---------------------------------------------------------------------------
# 3. Cherry-pick function tests
# ---------------------------------------------------------------------------

class TestCherryPick:
    def test_cherry_pick_selects_commits(self):
        """cherry_pick returns commits selected by the LLM."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 5)

            # LLM will select the first two
            h0_short = hashes[0][:8]
            h1_short = hashes[1][:8]
            mock = MockLLMClient(json.dumps({
                "reasoning": "These are most relevant",
                "selected": [h0_short, h1_short],
            }))
            t.config.configure_llm(mock)

            result = cherry_pick(t, "Find auth-related commits", limit=3)

            assert isinstance(result, CherryPickResult)
            assert len(result.selected_hashes) == 2
            assert hashes[0] in result.selected_hashes
            assert hashes[1] in result.selected_hashes
            assert result.total_candidates == 5
            assert result.tokens_used == 42
            assert result.reasoning == "These are most relevant"

    def test_cherry_pick_respects_limit(self):
        """cherry_pick caps selection at the limit."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 5)

            # LLM tries to return 4 commits but limit is 2
            selected = [h[:8] for h in hashes[:4]]
            mock = MockLLMClient(json.dumps({
                "reasoning": "Selected many",
                "selected": selected,
            }))
            t.config.configure_llm(mock)

            result = cherry_pick(t, "query", limit=2)
            assert len(result.selected_hashes) <= 2

    def test_cherry_pick_empty_tract(self):
        """cherry_pick on empty tract returns no candidates."""
        with Tract.open() as t:
            mock = MockLLMClient(json.dumps({
                "reasoning": "nothing",
                "selected": [],
            }))
            t.config.configure_llm(mock)

            result = cherry_pick(t, "query")
            assert result.total_candidates == 0
            assert result.selected_hashes == ()
            # Should not have called the LLM
            assert len(mock.calls) == 0

    def test_cherry_pick_fail_open_on_llm_error(self):
        """cherry_pick returns all commits when LLM fails (fail-open)."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 3)
            t.config.configure_llm(ErrorLLMClient())

            result = cherry_pick(t, "query")

            assert isinstance(result, CherryPickResult)
            # All commits returned in fail-open mode
            assert len(result.selected_hashes) == 3
            assert result.tokens_used == 0
            assert "fail-open" in result.reasoning.lower()

    def test_cherry_pick_fail_open_no_client(self):
        """cherry_pick returns all commits when no LLM client configured."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 3)
            # No LLM client configured

            result = cherry_pick(t, "query")

            assert len(result.selected_hashes) == 3
            assert "fail-open" in result.reasoning.lower()


# ---------------------------------------------------------------------------
# 4. Deduplicate function tests
# ---------------------------------------------------------------------------

class TestDeduplicate:
    def test_deduplicate_finds_groups(self):
        """deduplicate identifies duplicate groups via LLM."""
        with Tract.open() as t:
            hashes = _seed_duplicate_commits(t)
            # hashes: [0]=auth, [1]=testing, [2]=auth(dup), [3]=db

            h0_short = hashes[0][:8]
            h2_short = hashes[2][:8]
            mock = MockLLMClient(json.dumps({
                "reasoning": "Commits 0 and 2 have identical auth content",
                "groups": [[h0_short, h2_short]],
            }))
            t.config.configure_llm(mock)

            result = deduplicate(t, auto_skip=False)

            assert isinstance(result, DedupResult)
            assert len(result.duplicate_groups) == 1
            assert hashes[0] in result.duplicate_groups[0]
            assert hashes[2] in result.duplicate_groups[0]
            assert result.actions_taken == 0  # auto_skip=False
            assert result.tokens_used == 42

    def test_deduplicate_auto_skip_applies_annotations(self):
        """deduplicate with auto_skip=True marks older duplicates as SKIP."""
        with Tract.open() as t:
            hashes = _seed_duplicate_commits(t)
            # In log order (newest first): [3]=db, [2]=auth(dup), [1]=testing, [0]=auth
            # So in group [0, 2], commit 2 is newer, commit 0 should be skipped

            h0_short = hashes[0][:8]
            h2_short = hashes[2][:8]
            mock = MockLLMClient(json.dumps({
                "reasoning": "Duplicate auth commits",
                "groups": [[h0_short, h2_short]],
            }))
            t.config.configure_llm(mock)

            result = deduplicate(t, auto_skip=True)

            assert result.actions_taken == 1

            # Verify the older commit (hash[0]) got SKIP
            log_entries = t.log(limit=50)
            for entry in log_entries:
                if entry.commit_hash == hashes[0]:
                    assert entry.effective_priority == "skip"
                elif entry.commit_hash == hashes[2]:
                    # Newer commit should NOT be skipped
                    assert entry.effective_priority != "skip"

    def test_deduplicate_empty_tract(self):
        """deduplicate on empty tract returns no groups."""
        with Tract.open() as t:
            mock = MockLLMClient(json.dumps({
                "reasoning": "nothing",
                "groups": [],
            }))
            t.config.configure_llm(mock)

            result = deduplicate(t)
            assert result.duplicate_groups == ()
            assert len(mock.calls) == 0

    def test_deduplicate_fail_open_on_llm_error(self):
        """deduplicate returns empty groups when LLM fails (fail-open)."""
        with Tract.open() as t:
            _seed_commits(t, 3)
            t.config.configure_llm(ErrorLLMClient())

            result = deduplicate(t)

            assert isinstance(result, DedupResult)
            assert result.duplicate_groups == ()
            assert result.actions_taken == 0
            assert "fail-open" in result.reasoning.lower()

    def test_deduplicate_fail_open_no_client(self):
        """deduplicate returns empty groups when no LLM client configured."""
        with Tract.open() as t:
            _seed_commits(t, 3)

            result = deduplicate(t)

            assert result.duplicate_groups == ()
            assert "fail-open" in result.reasoning.lower()

    def test_deduplicate_threshold_description(self):
        """deduplicate passes threshold info to the LLM prompt."""
        with Tract.open() as t:
            _seed_commits(t, 2)
            mock = MockLLMClient(json.dumps({
                "reasoning": "No duplicates",
                "groups": [],
            }))
            t.config.configure_llm(mock)

            deduplicate(t, threshold=0.95)

            # Check that the prompt includes threshold info
            assert len(mock.calls) == 1
            user_msg = mock.calls[0][0][-1]["content"]
            assert "0.95" in user_msg
            assert "very strict" in user_msg


# ---------------------------------------------------------------------------
# 5. Tract API integration tests
# ---------------------------------------------------------------------------

class TestTractAPIIntegration:
    def test_t_cherry_pick(self):
        """cherry_pick(t, ) delegates to intelligence.cherry_pick."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 3)
            h0_short = hashes[0][:8]
            mock = MockLLMClient(json.dumps({
                "reasoning": "Selected one",
                "selected": [h0_short],
            }))
            t.config.configure_llm(mock)

            result = cherry_pick(t, "Find something", limit=5)

            assert isinstance(result, CherryPickResult)
            assert len(result.selected_hashes) == 1
            assert hashes[0] in result.selected_hashes

    def test_t_deduplicate(self):
        """deduplicate(t) delegates to intelligence.deduplicate."""
        with Tract.open() as t:
            _seed_commits(t, 3)
            mock = MockLLMClient(json.dumps({
                "reasoning": "No duplicates",
                "groups": [],
            }))
            t.config.configure_llm(mock)

            result = deduplicate(t)

            assert isinstance(result, DedupResult)
            assert result.duplicate_groups == ()

    def test_t_cherry_pick_passes_llm_kwargs(self):
        """cherry_pick(t, ) passes model/temperature/max_tokens to LLM."""
        with Tract.open() as t:
            _seed_commits(t, 2)
            mock = MockLLMClient(json.dumps({
                "reasoning": "ok",
                "selected": [],
            }))
            t.config.configure_llm(mock)

            cherry_pick(t, "query", model="gpt-4o", temperature=0.5, max_tokens=500)

            assert len(mock.calls) == 1
            _, kwargs = mock.calls[0]
            assert kwargs.get("model") == "gpt-4o"
            assert kwargs.get("temperature") == 0.5
            assert kwargs.get("max_tokens") == 500

    def test_t_deduplicate_passes_llm_kwargs(self):
        """deduplicate(t) passes model/temperature/max_tokens to LLM."""
        with Tract.open() as t:
            _seed_commits(t, 2)
            mock = MockLLMClient(json.dumps({
                "reasoning": "ok",
                "groups": [],
            }))
            t.config.configure_llm(mock)

            deduplicate(t, model="gpt-4o-mini", temperature=0.2)

            assert len(mock.calls) == 1
            _, kwargs = mock.calls[0]
            assert kwargs.get("model") == "gpt-4o-mini"
            assert kwargs.get("temperature") == 0.2

    def test_t_cherry_pick_closed_tract_raises(self):
        """cherry_pick on a closed tract raises ClosedError."""
        t = Tract.open()
        t.close()
        with pytest.raises(Exception):  # ClosedError
            cherry_pick(t, "query")

    def test_t_deduplicate_closed_tract_raises(self):
        """deduplicate on a closed tract raises ClosedError."""
        t = Tract.open()
        t.close()
        with pytest.raises(Exception):  # ClosedError
            deduplicate(t)


# ---------------------------------------------------------------------------
# 6. Async tests
# ---------------------------------------------------------------------------

class TestAsyncIntelligence:
    @pytest.mark.asyncio
    async def test_acherry_pick(self):
        """acherry_pick() works via the async path."""
        with Tract.open() as t:
            hashes = _seed_commits(t, 3)
            h0_short = hashes[0][:8]
            mock = MockLLMClient(json.dumps({
                "reasoning": "Async selected",
                "selected": [h0_short],
            }))
            t.config.configure_llm(mock)

            result = await acherry_pick(t, "query", limit=2)

            assert isinstance(result, CherryPickResult)
            assert hashes[0] in result.selected_hashes

    @pytest.mark.asyncio
    async def test_adeduplicate(self):
        """adeduplicate() works via the async path."""
        with Tract.open() as t:
            _seed_commits(t, 3)
            mock = MockLLMClient(json.dumps({
                "reasoning": "No dupes",
                "groups": [],
            }))
            t.config.configure_llm(mock)

            result = await adeduplicate(t)

            assert isinstance(result, DedupResult)
            assert result.duplicate_groups == ()


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_cherry_pick_with_tags(self):
        """cherry_pick works correctly with tagged commits."""
        with Tract.open() as t:
            t.register_tag("important", "Important tag")
            info = t.commit(
                {"content_type": "dialogue", "role": "user", "text": "Tagged content"},
                tags=["important"],
                message="tagged commit",
            )
            h_short = info.commit_hash[:8]
            mock = MockLLMClient(json.dumps({
                "reasoning": "Tagged commit is relevant",
                "selected": [h_short],
            }))
            t.config.configure_llm(mock)

            result = cherry_pick(t, "find tagged stuff")
            assert len(result.selected_hashes) == 1

            # Verify tags appear in manifest
            manifest, _ = _build_intelligence_manifest(t)
            assert "important" in manifest

    def test_deduplicate_multiple_groups(self):
        """deduplicate handles multiple duplicate groups."""
        with Tract.open() as t:
            hashes = []
            for i in range(6):
                info = t.commit(
                    {"content_type": "dialogue", "role": "user", "text": f"Content {i % 3}"},
                    message=f"commit {i}",
                )
                hashes.append(info.commit_hash)

            # Two groups of duplicates
            mock = MockLLMClient(json.dumps({
                "reasoning": "Two groups of similar content",
                "groups": [
                    [hashes[0][:8], hashes[3][:8]],
                    [hashes[1][:8], hashes[4][:8]],
                ],
            }))
            t.config.configure_llm(mock)

            result = deduplicate(t, auto_skip=True)

            assert len(result.duplicate_groups) == 2
            assert result.actions_taken == 2  # One skip per group

    def test_cherry_pick_returns_frozen_result(self):
        """CherryPickResult is immutable (frozen dataclass)."""
        result = CherryPickResult(
            selected_hashes=("abc",),
            total_candidates=1,
            tokens_used=10,
            reasoning="test",
        )
        with pytest.raises(AttributeError):
            result.reasoning = "changed"  # type: ignore[misc]

    def test_dedup_result_frozen(self):
        """DedupResult is immutable (frozen dataclass)."""
        result = DedupResult(
            duplicate_groups=(("a", "b"),),
            actions_taken=1,
            tokens_used=10,
            reasoning="test",
        )
        with pytest.raises(AttributeError):
            result.reasoning = "changed"  # type: ignore[misc]
