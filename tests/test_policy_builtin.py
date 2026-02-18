"""Unit tests for built-in policies: CompressPolicy, PinPolicy, BranchPolicy, ArchivePolicy.

Covers:
- CompressPolicy: no budget, below/above threshold, custom threshold, properties, config roundtrip
- PinPolicy: pins instruction/session, skips dialogue, respects manual override, skips already-pinned,
  custom types, retroactive scan, properties, config roundtrip
- BranchPolicy: no tangent, detects tangent, too few commits, detached HEAD, custom threshold, properties
- ArchivePolicy: main branch skipped, active branch skipped, stale branch detected, archive prefix,
  already-archived skipped, properties
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from tract import (
    DialogueContent,
    InstructionContent,
    Priority,
    Tract,
    TokenBudgetConfig,
    TractConfig,
)
from tract.models.content import ArtifactContent, ReasoningContent
from tract.models.session import SessionContent
from tract.policy.builtin.branch import BranchPolicy
from tract.policy.builtin.compress import CompressPolicy
from tract.policy.builtin.pin import PinPolicy
from tract.policy.builtin.archive import ArchivePolicy


# ---------------------------------------------------------------------------
# CompressPolicy Tests
# ---------------------------------------------------------------------------


class TestCompressPolicy:
    """Unit tests for CompressPolicy."""

    def test_compress_policy_no_budget(self):
        """No budget configured -- returns None."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = CompressPolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_compress_policy_below_threshold(self):
        """Token count below 90% of budget -- returns None."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10000))
        t = Tract.open(":memory:", config=config)
        try:
            t.commit(DialogueContent(role="user", text="short"))
            p = CompressPolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_compress_policy_above_threshold(self):
        """Token count above 90% of budget -- returns PolicyAction."""
        # Use a very low budget to easily exceed threshold
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            # Commit enough to exceed 90% of 10 tokens
            t.commit(InstructionContent(text="This is a long enough instruction to exceed token budget"))
            p = CompressPolicy(summary_content="Summary")
            action = p.evaluate(t)
            assert action is not None
            assert action.action_type == "compress"
            assert action.autonomy == "collaborative"
            assert "exceeds 90%" in action.reason
            assert action.params["auto_commit"] is True
            assert action.params["content"] == "Summary"
        finally:
            t.close()

    def test_compress_policy_custom_threshold(self):
        """Custom threshold (50%) triggers earlier."""
        config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=10))
        t = Tract.open(":memory:", config=config)
        try:
            t.commit(InstructionContent(text="This should exceed 50% of 10 tokens"))
            p = CompressPolicy(threshold=0.5)
            action = p.evaluate(t)
            assert action is not None
            assert "exceeds 50%" in action.reason
        finally:
            t.close()

    def test_compress_policy_properties(self):
        """Correct name, priority, trigger."""
        p = CompressPolicy()
        assert p.name == "auto-compress"
        assert p.priority == 200
        assert p.trigger == "compile"

    def test_compress_policy_config_roundtrip(self):
        """to_config/from_config roundtrip."""
        p = CompressPolicy(threshold=0.8, summary_content="test")
        cfg = p.to_config()
        p2 = CompressPolicy.from_config(cfg)
        assert p2._threshold == 0.8
        assert p2._summary_content == "test"
        assert p2.to_config() == cfg


# ---------------------------------------------------------------------------
# PinPolicy Tests
# ---------------------------------------------------------------------------


class TestPinPolicy:
    """Unit tests for PinPolicy."""

    def test_pin_policy_pins_instruction_content(self):
        """InstructionContent commits are auto-pinned by the engine's default
        priority annotation, so PinPolicy correctly defers (no duplicate pin).
        When tested directly without pre-existing annotation, PinPolicy would pin."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="System prompt"))
            # The commit engine already created a PINNED annotation for instruction
            annotations = t.get_annotations(info.commit_hash)
            assert any(a.priority == Priority.PINNED for a in annotations)
            # PinPolicy correctly returns None (respects existing annotation)
            p = PinPolicy()
            action = p.evaluate(t)
            assert action is None
        finally:
            t.close()

    def test_pin_policy_pins_session_content(self):
        """SessionContent commits are auto-pinned by PinPolicy."""
        t = Tract.open(":memory:")
        try:
            t.commit(SessionContent(
                session_type="start",
                summary="Starting new session",
            ))
            p = PinPolicy()
            action = p.evaluate(t)
            assert action is not None
            assert action.action_type == "annotate"
            assert action.params["priority"] == "pinned"
        finally:
            t.close()

    def test_pin_policy_skips_dialogue(self):
        """DialogueContent commits are not pinned."""
        t = Tract.open(":memory:")
        try:
            t.commit(DialogueContent(role="user", text="Hello"))
            p = PinPolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_pin_policy_respects_manual_override(self):
        """Does not re-pin commits that have existing annotations."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="System prompt"))
            # User manually sets to NORMAL
            t.annotate(info.commit_hash, Priority.NORMAL, reason="User override")

            p = PinPolicy()
            action = p.evaluate(t)
            assert action is None  # Respects manual override
        finally:
            t.close()

    def test_pin_policy_skips_already_pinned(self):
        """Does not re-pin commits that are already pinned."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="System prompt"))
            t.annotate(info.commit_hash, Priority.PINNED, reason="Already pinned")

            p = PinPolicy()
            action = p.evaluate(t)
            assert action is None
        finally:
            t.close()

    def test_pin_policy_custom_types(self):
        """Custom pin_types parameter works."""
        t = Tract.open(":memory:")
        try:
            t.commit(DialogueContent(role="user", text="Hello"))
            # Customize to pin dialogue
            p = PinPolicy(pin_types={"dialogue"})
            action = p.evaluate(t)
            assert action is not None
            assert action.params["priority"] == "pinned"
        finally:
            t.close()

    def test_pin_policy_retroactive_scan(self):
        """retroactive_scan() pins matching commits that lack annotations.

        Uses SessionContent because InstructionContent gets auto-annotated
        by the commit engine (default priority for instruction type).
        """
        t = Tract.open(":memory:")
        try:
            c1 = t.commit(SessionContent(session_type="start", summary="Session 1"))
            c2 = t.commit(DialogueContent(role="user", text="Hello"))
            c3 = t.commit(SessionContent(session_type="end", summary="Session 2"))

            p = PinPolicy()
            pinned = p.retroactive_scan(t)

            # Both session commits should be pinned
            assert c1.commit_hash in pinned
            assert c3.commit_hash in pinned
            assert c2.commit_hash not in pinned
            assert len(pinned) == 2

            # Verify actual annotations
            for h in pinned:
                annotations = t.get_annotations(h)
                assert any(a.priority == Priority.PINNED for a in annotations)
        finally:
            t.close()

    def test_pin_policy_retroactive_scan_respects_manual(self):
        """retroactive_scan() skips commits with existing annotations."""
        t = Tract.open(":memory:")
        try:
            c1 = t.commit(SessionContent(session_type="start", summary="Session 1"))
            # Manually set to NORMAL
            t.annotate(c1.commit_hash, Priority.NORMAL, reason="User set")

            c2 = t.commit(SessionContent(session_type="end", summary="Session 2"))

            p = PinPolicy()
            pinned = p.retroactive_scan(t)

            # c1 should be skipped (manual annotation), c2 should be pinned
            assert c1.commit_hash not in pinned
            assert c2.commit_hash in pinned
        finally:
            t.close()

    def test_pin_policy_properties(self):
        """Correct name, priority, trigger."""
        p = PinPolicy()
        assert p.name == "auto-pin"
        assert p.priority == 100
        assert p.trigger == "commit"

    def test_pin_policy_config_roundtrip(self):
        """to_config/from_config roundtrip."""
        p = PinPolicy(pin_types={"dialogue", "reasoning"})
        cfg = p.to_config()
        p2 = PinPolicy.from_config(cfg)
        assert p2._pin_types == {"dialogue", "reasoning"}
        assert p2.to_config() == cfg


# ---------------------------------------------------------------------------
# BranchPolicy Tests
# ---------------------------------------------------------------------------


class TestBranchPolicy:
    """Unit tests for BranchPolicy."""

    def test_branch_policy_no_tangent(self):
        """Normal conversation -- no tangent detected."""
        t = Tract.open(":memory:")
        try:
            # 5 consecutive dialogue commits -- no switching
            for i in range(5):
                t.commit(DialogueContent(role="user", text=f"msg {i}"))

            p = BranchPolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_branch_policy_detects_tangent(self):
        """Rapid type switching triggers branch proposal."""
        t = Tract.open(":memory:")
        try:
            # Create rapid switching: instruction -> reasoning -> artifact -> instruction -> reasoning
            t.commit(InstructionContent(text="instruction 1"))
            t.commit(ReasoningContent(text="reasoning 1"))
            t.commit(ArtifactContent(artifact_type="code", content="print('hello')"))
            t.commit(InstructionContent(text="instruction 2"))
            t.commit(ReasoningContent(text="reasoning 2"))

            p = BranchPolicy(switch_threshold=3)
            action = p.evaluate(t)
            assert action is not None
            assert action.action_type == "branch"
            assert action.autonomy == "collaborative"
            assert "tangent" in action.params["name"]
        finally:
            t.close()

    def test_branch_policy_too_few_commits(self):
        """Fewer than 3 commits -- returns None."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="only one"))
            t.commit(DialogueContent(role="user", text="two"))

            p = BranchPolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_branch_policy_detached_head(self):
        """Detached HEAD -- returns None."""
        t = Tract.open(":memory:")
        try:
            info = t.commit(InstructionContent(text="hello"))
            t.commit(DialogueContent(role="user", text="world"))
            # Detach HEAD
            t.checkout(info.commit_hash)

            p = BranchPolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_branch_policy_custom_threshold(self):
        """Custom switch_threshold works."""
        t = Tract.open(":memory:")
        try:
            # 4 commits with 3 transitions
            t.commit(InstructionContent(text="a"))
            t.commit(ReasoningContent(text="b"))
            t.commit(ArtifactContent(artifact_type="code", content="c"))
            t.commit(InstructionContent(text="d"))

            # Default threshold (4) should not trigger
            p_default = BranchPolicy()
            assert p_default.evaluate(t) is None

            # Lower threshold (2) should trigger
            p_low = BranchPolicy(switch_threshold=2)
            action = p_low.evaluate(t)
            assert action is not None
        finally:
            t.close()

    def test_branch_policy_properties(self):
        """Correct name, priority, trigger."""
        p = BranchPolicy()
        assert p.name == "auto-branch"
        assert p.priority == 300
        assert p.trigger == "commit"

    def test_branch_policy_config_roundtrip(self):
        """to_config/from_config roundtrip."""
        p = BranchPolicy(content_type_window=10, switch_threshold=5)
        cfg = p.to_config()
        p2 = BranchPolicy.from_config(cfg)
        assert p2._content_type_window == 10
        assert p2._switch_threshold == 5
        assert p2.to_config() == cfg


# ---------------------------------------------------------------------------
# ArchivePolicy Tests
# ---------------------------------------------------------------------------


class TestArchivePolicy:
    """Unit tests for ArchivePolicy."""

    def test_archive_policy_main_branch_skipped(self):
        """Main branch is never archived."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            p = ArchivePolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_archive_policy_active_branch_skipped(self):
        """Active branch (recent commits) is not archived."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            t.branch("feature-x")
            t.commit(DialogueContent(role="user", text="recent"))

            p = ArchivePolicy(stale_days=7)
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_archive_policy_stale_branch_detected(self):
        """Stale branch with few commits triggers archive proposal."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            t.branch("stale-branch")
            t.commit(DialogueContent(role="user", text="old content"))

            # Mock datetime.now() to make commits appear old
            old_date = datetime.now() + timedelta(days=10)
            with patch("tract.policy.builtin.archive.datetime") as mock_dt:
                mock_dt.now.return_value = old_date
                p = ArchivePolicy(stale_days=7, min_commits=3)
                action = p.evaluate(t)

            assert action is not None
            assert action.action_type == "archive"
            assert "archive/stale-branch" in action.params["archive_name"]
            assert action.autonomy == "collaborative"
        finally:
            t.close()

    def test_archive_policy_archive_prefix(self):
        """Custom archive prefix is used."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            t.branch("old-branch")
            t.commit(DialogueContent(role="user", text="old"))

            old_date = datetime.now() + timedelta(days=30)
            with patch("tract.policy.builtin.archive.datetime") as mock_dt:
                mock_dt.now.return_value = old_date
                p = ArchivePolicy(stale_days=7, min_commits=3, archive_prefix="archived/")
                action = p.evaluate(t)

            assert action is not None
            assert action.params["archive_name"].startswith("archived/")
        finally:
            t.close()

    def test_archive_policy_already_archived_skipped(self):
        """Already-archived branches are skipped."""
        t = Tract.open(":memory:")
        try:
            t.commit(InstructionContent(text="hello"))
            t.branch("archive/old-branch")
            t.commit(DialogueContent(role="user", text="archived"))

            p = ArchivePolicy()
            assert p.evaluate(t) is None
        finally:
            t.close()

    def test_archive_policy_properties(self):
        """Correct name, priority, trigger."""
        p = ArchivePolicy()
        assert p.name == "auto-archive"
        assert p.priority == 500
        assert p.trigger == "compile"

    def test_archive_policy_config_roundtrip(self):
        """to_config/from_config roundtrip."""
        p = ArchivePolicy(stale_days=14, min_commits=5, archive_prefix="old/")
        cfg = p.to_config()
        p2 = ArchivePolicy.from_config(cfg)
        assert p2._stale_days == 14
        assert p2._min_commits == 5
        assert p2._archive_prefix == "old/"
        assert p2.to_config() == cfg


# ---------------------------------------------------------------------------
# Priority Ordering Test
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    """Verify built-in policies have correct priority ordering."""

    def test_priority_ordering(self):
        """PinPolicy(100) < CompressPolicy(200) < BranchPolicy(300) < ArchivePolicy(500)."""
        assert PinPolicy().priority < CompressPolicy().priority
        assert CompressPolicy().priority < BranchPolicy().priority
        assert BranchPolicy().priority < ArchivePolicy().priority

    def test_exact_priorities(self):
        """Exact priority values match specification."""
        assert PinPolicy().priority == 100
        assert CompressPolicy().priority == 200
        assert BranchPolicy().priority == 300
        assert ArchivePolicy().priority == 500
