"""Tests for the recent_ratio parameter on adaptive compile strategy.

Covers:
- Backward compatibility (recent_ratio=None changes nothing)
- Correct k computation from ratio
- Edge ratios (0.0, 1.0)
- Validation (out-of-range raises ValueError)
- Interaction with strategy_k (ratio overrides fixed k)
- Ignored for non-adaptive strategies
"""

import pytest

from tract import Tract


def _make_tract_with_n_effective(n: int) -> Tract:
    """Create a tract with exactly n effective (compilable) commits.

    Uses user/assistant pairs so there are n/2 pairs when n is even,
    or (n-1)/2 pairs + 1 extra user commit when n is odd.
    Each commit has distinct long content and a short commit message
    so the adaptive strategy produces measurably different token counts.
    """
    t = Tract.open()
    created = 0
    pair = 0
    while created < n:
        if created + 1 < n:
            t.user(
                f"Detailed user message number {pair} with enough content to "
                f"make the full strategy meaningfully different from messages. "
                f"Extra filler text to increase token count: {'word ' * 30}",
                message=f"User msg {pair}",
            )
            t.assistant(
                f"Detailed assistant response number {pair} with thorough "
                f"analysis and recommendations for the query above. "
                f"Additional detail: {'detail ' * 30}",
                message=f"Asst resp {pair}",
            )
            created += 2
            pair += 1
        else:
            t.user(
                f"Final user message {pair} with extra content to pad out "
                f"the token count for a meaningful difference. "
                f"Padding: {'pad ' * 30}",
                message=f"User msg {pair}",
            )
            created += 1
            pair += 1
    return t


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    def test_none_ratio_matches_default(self):
        """recent_ratio=None produces the same result as omitting it."""
        t = _make_tract_with_n_effective(10)
        default = t.compile(strategy="adaptive", strategy_k=5)
        explicit_none = t.compile(strategy="adaptive", strategy_k=5, recent_ratio=None)
        assert default.token_count == explicit_none.token_count
        assert default.commit_count == explicit_none.commit_count

    def test_none_ratio_full_strategy(self):
        """recent_ratio=None with full strategy works normally."""
        t = _make_tract_with_n_effective(6)
        result = t.compile(strategy="full", recent_ratio=None)
        assert result.commit_count == 6


# ---------------------------------------------------------------------------
# Ratio computation
# ---------------------------------------------------------------------------


class TestRatioComputation:
    def test_ratio_0_5_with_10_commits(self):
        """recent_ratio=0.5 with 10 commits -> last 5 full, first 5 summaries."""
        t = _make_tract_with_n_effective(10)

        # ratio=0.5 -> k = max(1, int(10 * 0.5)) = 5
        ratio_result = t.compile(strategy="adaptive", recent_ratio=0.5)
        fixed_result = t.compile(strategy="adaptive", strategy_k=5)

        assert ratio_result.commit_count == fixed_result.commit_count == 10
        assert ratio_result.token_count == fixed_result.token_count

    def test_ratio_produces_expected_k(self):
        """Verify the ratio computes the expected k for various sizes."""
        # 20 commits, ratio=0.3 -> k = max(1, int(20 * 0.3)) = 6
        t = _make_tract_with_n_effective(20)
        ratio_result = t.compile(strategy="adaptive", recent_ratio=0.3)
        fixed_result = t.compile(strategy="adaptive", strategy_k=6)
        assert ratio_result.token_count == fixed_result.token_count

    def test_ratio_with_odd_result(self):
        """Ratio that doesn't divide evenly truncates correctly."""
        # 7 commits, ratio=0.5 -> k = max(1, int(7 * 0.5)) = 3
        t = _make_tract_with_n_effective(7)
        ratio_result = t.compile(strategy="adaptive", recent_ratio=0.5)
        fixed_result = t.compile(strategy="adaptive", strategy_k=3)
        assert ratio_result.token_count == fixed_result.token_count


# ---------------------------------------------------------------------------
# Edge ratios
# ---------------------------------------------------------------------------


class TestEdgeRatios:
    def test_ratio_1_0_all_full(self):
        """recent_ratio=1.0 keeps all commits at full detail."""
        t = _make_tract_with_n_effective(10)
        ratio_result = t.compile(strategy="adaptive", recent_ratio=1.0)
        full_result = t.compile(strategy="full")
        assert ratio_result.token_count == full_result.token_count

    def test_ratio_0_0_minimal_full(self):
        """recent_ratio=0.0 -> k = max(1, 0) = 1, only last commit full."""
        t = _make_tract_with_n_effective(10)
        ratio_result = t.compile(strategy="adaptive", recent_ratio=0.0)
        k1_result = t.compile(strategy="adaptive", strategy_k=1)
        assert ratio_result.token_count == k1_result.token_count

    def test_ratio_0_0_at_least_one_full(self):
        """Even with ratio=0.0, at least 1 commit stays at full detail."""
        t = _make_tract_with_n_effective(10)
        all_messages = t.compile(strategy="messages")
        ratio_0 = t.compile(strategy="adaptive", recent_ratio=0.0)
        # ratio=0.0 still keeps 1 at full, so it should have more tokens
        # than pure messages strategy (assuming the last commit's full content
        # is longer than its commit message)
        assert ratio_0.token_count >= all_messages.token_count


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_ratio_raises(self):
        """recent_ratio < 0 raises ValueError."""
        t = _make_tract_with_n_effective(4)
        with pytest.raises(ValueError, match="recent_ratio must be between 0.0 and 1.0"):
            t.compile(strategy="adaptive", recent_ratio=-0.1)

    def test_ratio_above_1_raises(self):
        """recent_ratio > 1 raises ValueError."""
        t = _make_tract_with_n_effective(4)
        with pytest.raises(ValueError, match="recent_ratio must be between 0.0 and 1.0"):
            t.compile(strategy="adaptive", recent_ratio=1.5)

    def test_ratio_2_raises(self):
        """recent_ratio=2.0 raises ValueError."""
        t = _make_tract_with_n_effective(2)
        with pytest.raises(ValueError, match="recent_ratio must be between 0.0 and 1.0"):
            t.compile(strategy="adaptive", recent_ratio=2.0)

    def test_ratio_negative_large_raises(self):
        """Large negative recent_ratio raises ValueError."""
        t = _make_tract_with_n_effective(2)
        with pytest.raises(ValueError, match="recent_ratio must be between 0.0 and 1.0"):
            t.compile(strategy="adaptive", recent_ratio=-100.0)


# ---------------------------------------------------------------------------
# Interaction with strategy and strategy_k
# ---------------------------------------------------------------------------


class TestStrategyInteraction:
    def test_ratio_ignored_for_full_strategy(self):
        """recent_ratio is ignored when strategy is 'full'."""
        t = _make_tract_with_n_effective(10)
        full_default = t.compile(strategy="full")
        full_with_ratio = t.compile(strategy="full", recent_ratio=0.3)
        assert full_default.token_count == full_with_ratio.token_count

    def test_ratio_ignored_for_messages_strategy(self):
        """recent_ratio is ignored when strategy is 'messages'."""
        t = _make_tract_with_n_effective(10)
        msg_default = t.compile(strategy="messages")
        msg_with_ratio = t.compile(strategy="messages", recent_ratio=0.3)
        assert msg_default.token_count == msg_with_ratio.token_count

    def test_ratio_overrides_strategy_k(self):
        """When both recent_ratio and strategy_k are set, ratio wins."""
        t = _make_tract_with_n_effective(10)

        # ratio=0.7 -> k = max(1, int(10 * 0.7)) = 7
        # strategy_k=2 would give very different results
        ratio_result = t.compile(strategy="adaptive", strategy_k=2, recent_ratio=0.7)
        ratio_only = t.compile(strategy="adaptive", recent_ratio=0.7)
        fixed_k2 = t.compile(strategy="adaptive", strategy_k=2)

        # ratio result should match ratio_only (ratio overrides k)
        assert ratio_result.token_count == ratio_only.token_count
        # and differ from fixed k=2
        assert ratio_result.token_count != fixed_k2.token_count

    def test_ratio_with_single_commit(self):
        """Ratio with a single commit: max(1, int(1*ratio)) = 1 always."""
        t = Tract.open()
        t.user("Only message")
        result = t.compile(strategy="adaptive", recent_ratio=0.5)
        assert result.commit_count == 1

    def test_ratio_with_empty_tract(self):
        """Ratio with empty tract returns empty context."""
        t = Tract.open()
        result = t.compile(strategy="adaptive", recent_ratio=0.5)
        assert result.commit_count == 0
        assert result.token_count == 0


# ---------------------------------------------------------------------------
# Configure integration
# ---------------------------------------------------------------------------


class TestConfigureIntegration:
    def test_compile_recent_ratio_accepted_by_configure(self):
        """configure() accepts compile_recent_ratio as a well-known key."""
        t = Tract.open()
        t.configure(compile_recent_ratio=0.7)
        val = t.get_config("compile_recent_ratio")
        assert val == 0.7

    def test_configure_rejects_wrong_type(self):
        """configure() rejects non-numeric compile_recent_ratio."""
        t = Tract.open()
        with pytest.raises(ValueError, match="compile_recent_ratio"):
            t.configure(compile_recent_ratio="high")

    def test_configure_int_ratio_accepted(self):
        """configure() accepts int for compile_recent_ratio (e.g. 1)."""
        t = Tract.open()
        t.configure(compile_recent_ratio=1)
        val = t.get_config("compile_recent_ratio")
        assert val == 1
