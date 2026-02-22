"""Tests for the IMPORTANT priority tier and retention criteria.

Covers:
- Priority ordering (SKIP < NORMAL < IMPORTANT < PINNED)
- RetentionCriteria model (instructions, match_patterns, match_mode)
- annotate() with retain / retain_match params
- Shorthand commit methods (user(), system(), assistant()) with priority
- _validate_retention deterministic checks
- _classify_by_priority returning 4 groups
- Compression with IMPORTANT using enriched prompt
- Compression with deterministic validation and retry
- Retention criteria round-trip through DB storage
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from tract import (
    Priority,
    PriorityAnnotation,
    RetentionCriteria,
    Tract,
)
from tract.operations.compression import _classify_by_priority, _validate_retention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tract(**kwargs) -> Tract:
    """Open an in-memory Tract."""
    return Tract.open(":memory:", **kwargs)


def _make_llm_response(content: str) -> dict:
    """Build a mock LLM response dict."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": content,
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# Test: Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_priority_ordering(self):
        """SKIP < NORMAL < IMPORTANT < PINNED."""
        values = [p.value for p in Priority]
        assert values.index("skip") < values.index("normal")
        assert values.index("normal") < values.index("important")
        assert values.index("important") < values.index("pinned")

    def test_priority_enum_members(self):
        """All four priorities exist."""
        assert Priority.SKIP.value == "skip"
        assert Priority.NORMAL.value == "normal"
        assert Priority.IMPORTANT.value == "important"
        assert Priority.PINNED.value == "pinned"


# ---------------------------------------------------------------------------
# Test: RetentionCriteria model
# ---------------------------------------------------------------------------


class TestRetentionCriteriaModel:
    def test_create_with_instructions(self):
        rc = RetentionCriteria(instructions="Keep the API key format")
        assert rc.instructions == "Keep the API key format"
        assert rc.match_patterns is None
        assert rc.match_mode == "substring"

    def test_create_with_match_patterns(self):
        rc = RetentionCriteria(
            match_patterns=["api_key", "secret"],
            match_mode="substring",
        )
        assert rc.match_patterns == ["api_key", "secret"]
        assert rc.match_mode == "substring"

    def test_create_with_regex_mode(self):
        rc = RetentionCriteria(
            match_patterns=[r"\d{4}-\d{2}-\d{2}"],
            match_mode="regex",
        )
        assert rc.match_mode == "regex"

    def test_create_with_both(self):
        rc = RetentionCriteria(
            instructions="Preserve the date format",
            match_patterns=[r"\d{4}-\d{2}-\d{2}"],
            match_mode="regex",
        )
        assert rc.instructions == "Preserve the date format"
        assert rc.match_patterns == [r"\d{4}-\d{2}-\d{2}"]
        assert rc.match_mode == "regex"

    def test_serialization_round_trip(self):
        rc = RetentionCriteria(
            instructions="Keep the UUID",
            match_patterns=["abc-123"],
        )
        data = rc.model_dump()
        rc2 = RetentionCriteria(**data)
        assert rc2.instructions == rc.instructions
        assert rc2.match_patterns == rc.match_patterns
        assert rc2.match_mode == rc.match_mode


# ---------------------------------------------------------------------------
# Test: annotate() with retention
# ---------------------------------------------------------------------------


class TestAnnotateWithRetention:
    def test_annotate_with_retain(self):
        """annotate() with fuzzy retention stores RetentionCriteria."""
        t = _make_tract()
        info = t.user("Test message")
        ann = t.annotate(
            info.commit_hash, Priority.IMPORTANT,
            retain="Preserve the user's name",
        )
        assert ann.priority == Priority.IMPORTANT
        assert ann.retention is not None
        assert ann.retention.instructions == "Preserve the user's name"
        assert ann.retention.match_patterns is None

    def test_annotate_with_retain_match(self):
        """annotate() with deterministic retention stores match_patterns."""
        t = _make_tract()
        info = t.user("My API key is sk-12345")
        ann = t.annotate(
            info.commit_hash, Priority.IMPORTANT,
            retain_match=["sk-12345"],
        )
        assert ann.retention is not None
        assert ann.retention.match_patterns == ["sk-12345"]
        assert ann.retention.match_mode == "substring"

    def test_annotate_with_both(self):
        """annotate() with both fuzzy and deterministic retention."""
        t = _make_tract()
        info = t.user("Important data: 2024-01-01")
        ann = t.annotate(
            info.commit_hash, Priority.IMPORTANT,
            retain="Keep the date",
            retain_match=["2024-01-01"],
        )
        assert ann.retention is not None
        assert ann.retention.instructions == "Keep the date"
        assert ann.retention.match_patterns == ["2024-01-01"]

    def test_annotate_without_retention(self):
        """annotate() without retention params produces None retention."""
        t = _make_tract()
        info = t.user("Normal message")
        ann = t.annotate(info.commit_hash, Priority.NORMAL)
        assert ann.retention is None


# ---------------------------------------------------------------------------
# Test: shorthand with retention
# ---------------------------------------------------------------------------


class TestShorthandWithRetention:
    def test_user_with_priority_important(self):
        """user() with priority=IMPORTANT and retain sets annotation."""
        t = _make_tract()
        info = t.user(
            "Keep this data: endpoint=/api/v1",
            priority=Priority.IMPORTANT,
            retain="Preserve the endpoint URL",
        )
        annotations = t.get_annotations(info.commit_hash)
        assert len(annotations) == 1
        assert annotations[0].priority == Priority.IMPORTANT
        assert annotations[0].retention is not None
        assert annotations[0].retention.instructions == "Preserve the endpoint URL"

    def test_assistant_with_priority_important(self):
        """assistant() with priority=IMPORTANT and retain_match sets annotation."""
        t = _make_tract()
        info = t.assistant(
            "The answer is 42",
            priority=Priority.IMPORTANT,
            retain_match=["42"],
        )
        annotations = t.get_annotations(info.commit_hash)
        assert len(annotations) == 1
        assert annotations[0].retention.match_patterns == ["42"]

    def test_system_with_priority(self):
        """system() with priority sets annotation."""
        t = _make_tract()
        info = t.system(
            "You are a helpful assistant",
            priority=Priority.IMPORTANT,
            retain="Preserve the system instruction",
        )
        annotations = t.get_annotations(info.commit_hash)
        # instruction content type auto-pins (default priority), so 2 annotations:
        # one from default and one from our explicit priority=IMPORTANT
        assert len(annotations) == 2
        # Latest annotation (our explicit one) should be IMPORTANT
        latest = annotations[-1]
        assert latest.priority == Priority.IMPORTANT
        assert latest.retention is not None
        assert latest.retention.instructions == "Preserve the system instruction"

    def test_shorthand_without_priority(self):
        """Shorthand without priority does not create annotation."""
        t = _make_tract()
        info = t.user("Hello")
        annotations = t.get_annotations(info.commit_hash)
        assert len(annotations) == 0


# ---------------------------------------------------------------------------
# Test: _validate_retention
# ---------------------------------------------------------------------------


class TestValidateRetention:
    def test_passes_all_substrings(self):
        summary = "The API key is sk-12345 and the date is 2024-01-01"
        criteria = [
            RetentionCriteria(match_patterns=["sk-12345", "2024-01-01"]),
        ]
        ok, diag = _validate_retention(summary, criteria)
        assert ok is True
        assert diag is None

    def test_fails_missing_substring(self):
        summary = "The API key is redacted"
        criteria = [
            RetentionCriteria(match_patterns=["sk-12345"]),
        ]
        ok, diag = _validate_retention(summary, criteria)
        assert ok is False
        assert "substring not found: sk-12345" in diag

    def test_fails_missing_regex(self):
        summary = "No dates here"
        criteria = [
            RetentionCriteria(
                match_patterns=[r"\d{4}-\d{2}-\d{2}"],
                match_mode="regex",
            ),
        ]
        ok, diag = _validate_retention(summary, criteria)
        assert ok is False
        assert "regex not found" in diag

    def test_passes_regex(self):
        summary = "The date was 2024-01-15 and it was important"
        criteria = [
            RetentionCriteria(
                match_patterns=[r"\d{4}-\d{2}-\d{2}"],
                match_mode="regex",
            ),
        ]
        ok, diag = _validate_retention(summary, criteria)
        assert ok is True

    def test_no_match_patterns_passes(self):
        """Criteria with only instructions (no match_patterns) always pass."""
        summary = "Anything goes"
        criteria = [
            RetentionCriteria(instructions="Keep the name"),
        ]
        ok, diag = _validate_retention(summary, criteria)
        assert ok is True

    def test_empty_criteria_passes(self):
        ok, diag = _validate_retention("any summary", [])
        assert ok is True

    def test_multiple_criteria_all_must_pass(self):
        summary = "Contains alpha and beta"
        criteria = [
            RetentionCriteria(match_patterns=["alpha"]),
            RetentionCriteria(match_patterns=["gamma"]),
        ]
        ok, diag = _validate_retention(summary, criteria)
        assert ok is False
        assert "gamma" in diag


# ---------------------------------------------------------------------------
# Test: _classify_by_priority returns four groups
# ---------------------------------------------------------------------------


class TestClassifyReturnsFourGroups:
    def test_classify_returns_four_groups(self):
        """_classify_by_priority returns (pinned, important, normal, skip)."""
        t = _make_tract()

        # Create commits with different priorities
        c1 = t.user("pinned msg")
        t.annotate(c1.commit_hash, Priority.PINNED)

        c2 = t.user("important msg")
        t.annotate(c2.commit_hash, Priority.IMPORTANT)

        c3 = t.user("normal msg")
        # No annotation = NORMAL by default

        c4 = t.user("skip msg")
        t.annotate(c4.commit_hash, Priority.SKIP)

        # Get commit rows from the repo
        commit_repo = t._commit_repo
        annotation_repo = t._annotation_repo
        all_rows = list(commit_repo.get_ancestors(t.head))
        all_rows.reverse()  # oldest first

        pinned, important, normal, skip = _classify_by_priority(
            all_rows, annotation_repo
        )

        assert len(pinned) == 1
        assert pinned[0].commit_hash == c1.commit_hash
        assert len(important) == 1
        assert important[0].commit_hash == c2.commit_hash
        assert len(normal) == 1
        assert normal[0].commit_hash == c3.commit_hash
        assert len(skip) == 1
        assert skip[0].commit_hash == c4.commit_hash


# ---------------------------------------------------------------------------
# Test: Compression with IMPORTANT content
# ---------------------------------------------------------------------------


class TestCompressPreservesImportantContent:
    def test_compress_enriched_prompt(self):
        """Compression with IMPORTANT commits injects retention instructions into prompt."""
        t = _make_tract()

        # Create commits
        t.user("Hello, my name is Alice")
        c2 = t.assistant("Nice to meet you, Alice!")
        t.annotate(
            c2.commit_hash, Priority.IMPORTANT,
            retain="Preserve the user's name Alice",
        )
        t.user("What is the weather?")
        t.assistant("It's sunny today")

        # Mock LLM client
        mock_llm = MagicMock()
        mock_llm.chat.return_value = _make_llm_response(
            "Previously in this conversation: Alice was greeted. Weather discussed."
        )
        t.configure_llm(mock_llm)

        result = t.compress()

        # Verify the prompt included retention instructions
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        user_prompt = messages[1]["content"]
        assert "IMPORTANT" in user_prompt
        assert "Preserve the user's name Alice" in user_prompt

    def test_compress_important_with_validation_passes(self):
        """Compression with IMPORTANT + match_patterns passes on first try."""
        t = _make_tract()

        t.user("The API endpoint is /api/v1/users")
        c2 = t.assistant("Using endpoint /api/v1/users for the query")
        t.annotate(
            c2.commit_hash, Priority.IMPORTANT,
            retain="Preserve the endpoint",
            retain_match=["/api/v1/users"],
        )
        t.user("Thanks!")

        mock_llm = MagicMock()
        mock_llm.chat.return_value = _make_llm_response(
            "Previously in this conversation: The endpoint /api/v1/users was used."
        )
        t.configure_llm(mock_llm)

        result = t.compress()
        assert result.compressed_tokens > 0

    def test_compress_important_with_validation_retries(self):
        """Compression with IMPORTANT + match_patterns triggers retry on failure."""
        t = _make_tract()

        t.user("The secret code is XYZ-789")
        c2 = t.assistant("Noted the code XYZ-789")
        t.annotate(
            c2.commit_hash, Priority.IMPORTANT,
            retain_match=["XYZ-789"],
        )
        t.user("Continue")

        mock_llm = MagicMock()
        # First call: fails validation (missing pattern)
        # Second call: passes validation (includes pattern)
        mock_llm.chat.side_effect = [
            _make_llm_response("Previously: A secret code was discussed."),
            _make_llm_response("Previously: The code XYZ-789 was noted."),
        ]
        t.configure_llm(mock_llm)

        result = t.compress()
        # Should have called LLM twice (retry)
        assert mock_llm.chat.call_count == 2
        assert result.compressed_tokens > 0


# ---------------------------------------------------------------------------
# Test: Retention criteria round-trip through DB storage
# ---------------------------------------------------------------------------


class TestRetentionRoundTripStorage:
    def test_retention_round_trip_storage(self):
        """Store and retrieve retention criteria from DB."""
        t = _make_tract()
        info = t.user("Important data point: threshold=0.95")
        t.annotate(
            info.commit_hash, Priority.IMPORTANT,
            retain="Preserve the threshold value",
            retain_match=["threshold=0.95"],
            retain_match_mode="substring",
        )

        # Read back from DB via get_annotations
        annotations = t.get_annotations(info.commit_hash)
        assert len(annotations) == 1
        ann = annotations[0]
        assert ann.priority == Priority.IMPORTANT
        assert ann.retention is not None
        assert ann.retention.instructions == "Preserve the threshold value"
        assert ann.retention.match_patterns == ["threshold=0.95"]
        assert ann.retention.match_mode == "substring"

    def test_retention_none_round_trip(self):
        """Annotation without retention round-trips as None."""
        t = _make_tract()
        info = t.user("Plain message")
        t.annotate(info.commit_hash, Priority.NORMAL)
        annotations = t.get_annotations(info.commit_hash)
        assert len(annotations) == 1
        assert annotations[0].retention is None

    def test_retention_regex_round_trip(self):
        """Regex retention criteria round-trips correctly."""
        t = _make_tract()
        info = t.user("Date: 2024-01-15")
        t.annotate(
            info.commit_hash, Priority.IMPORTANT,
            retain_match=[r"\d{4}-\d{2}-\d{2}"],
            retain_match_mode="regex",
        )
        annotations = t.get_annotations(info.commit_hash)
        assert annotations[0].retention.match_mode == "regex"
        assert annotations[0].retention.match_patterns == [r"\d{4}-\d{2}-\d{2}"]
