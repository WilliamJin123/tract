"""Tests for the two-stage guidance pattern (Phase 4).

Tests GuidanceMixin.edit_guidance(), guidance_source tracking,
regenerate_guidance() stub, and guidance fields on PendingCompress
and PendingMerge.
"""

from __future__ import annotations

import pytest

from tract import Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.guidance import GuidanceMixin
from tract.hooks.merge import PendingMerge
from tract.hooks.pending import Pending


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compressible_tract() -> Tract:
    """Create an in-memory Tract with 3 commits (system + user + assistant)."""
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello, how are you?")
    t.assistant("I'm doing well, thank you!")
    return t


def _make_pending_compress(
    *,
    guidance: str | None = None,
    guidance_source: str | None = None,
) -> PendingCompress:
    """Create a PendingCompress via the real Tract pipeline."""
    t = _make_compressible_tract()
    pending = t.compress(content="summary", review=True)
    if guidance is not None:
        pending.guidance = guidance
    if guidance_source is not None:
        pending.guidance_source = guidance_source
    return pending


def _make_pending_merge(
    *,
    guidance: str | None = None,
    guidance_source: str | None = None,
) -> PendingMerge:
    """Create a PendingMerge with given guidance fields."""
    t = Tract.open(":memory:")
    pending = PendingMerge(
        operation="merge",
        tract=t,
        source_branch="feature",
        target_branch="main",
        guidance=guidance,
        guidance_source=guidance_source,
    )
    return pending


# ===========================================================================
# 1. GuidanceMixin inheritance
# ===========================================================================


class TestGuidanceMixinInheritance:
    """Verify GuidanceMixin is mixed into the correct classes."""

    def test_pending_compress_has_guidance_mixin(self):
        """PendingCompress inherits from GuidanceMixin."""
        assert issubclass(PendingCompress, GuidanceMixin)

    def test_pending_merge_has_guidance_mixin(self):
        """PendingMerge inherits from GuidanceMixin."""
        assert issubclass(PendingMerge, GuidanceMixin)

    def test_pending_compress_mro_guidance_before_pending(self):
        """GuidanceMixin appears before Pending in PendingCompress MRO."""
        mro = PendingCompress.__mro__
        guidance_idx = mro.index(GuidanceMixin)
        pending_idx = mro.index(Pending)
        assert guidance_idx < pending_idx, (
            f"GuidanceMixin (at {guidance_idx}) should be before "
            f"Pending (at {pending_idx}) in MRO"
        )

    def test_pending_merge_mro_guidance_before_pending(self):
        """GuidanceMixin appears before Pending in PendingMerge MRO."""
        mro = PendingMerge.__mro__
        guidance_idx = mro.index(GuidanceMixin)
        pending_idx = mro.index(Pending)
        assert guidance_idx < pending_idx


# ===========================================================================
# 2. edit_guidance() on PendingCompress
# ===========================================================================


class TestEditGuidanceCompress:
    """Tests for edit_guidance() on PendingCompress."""

    def test_edit_guidance_updates_text(self):
        """edit_guidance() replaces the guidance text."""
        pending = _make_pending_compress(guidance="original guidance")
        pending.edit_guidance("new guidance")
        assert pending.guidance == "new guidance"

    def test_edit_guidance_sets_user_source_from_none(self):
        """edit_guidance() sets guidance_source to 'user' when source was None."""
        pending = _make_pending_compress(guidance=None, guidance_source=None)
        pending.edit_guidance("user guidance")
        assert pending.guidance == "user guidance"
        assert pending.guidance_source == "user"

    def test_edit_guidance_sets_user_source_from_llm(self):
        """edit_guidance() sets guidance_source to 'user' when source was 'llm'."""
        pending = _make_pending_compress(
            guidance="llm guidance", guidance_source="llm"
        )
        pending.edit_guidance("user override")
        assert pending.guidance == "user override"
        assert pending.guidance_source == "user"

    def test_edit_guidance_keeps_user_source(self):
        """edit_guidance() keeps guidance_source as 'user' when already 'user'."""
        pending = _make_pending_compress(
            guidance="first edit", guidance_source="user"
        )
        pending.edit_guidance("second edit")
        assert pending.guidance == "second edit"
        assert pending.guidance_source == "user"

    def test_edit_guidance_user_plus_llm_stays(self):
        """edit_guidance() on 'user+llm' keeps it as 'user+llm'."""
        pending = _make_pending_compress(
            guidance="combined", guidance_source="user+llm"
        )
        pending.edit_guidance("re-edited")
        assert pending.guidance == "re-edited"
        assert pending.guidance_source == "user+llm"

    def test_edit_guidance_requires_pending_status(self):
        """edit_guidance() raises RuntimeError if status is not 'pending'."""
        pending = _make_pending_compress(guidance="some guidance")
        pending.approve()  # status -> approved
        with pytest.raises(RuntimeError):
            pending.edit_guidance("too late")


# ===========================================================================
# 3. edit_guidance() on PendingMerge
# ===========================================================================


class TestEditGuidanceMerge:
    """Tests for edit_guidance() on PendingMerge."""

    def test_edit_guidance_updates_text(self):
        """edit_guidance() replaces the guidance text on PendingMerge."""
        pending = _make_pending_merge(guidance="original")
        pending.edit_guidance("updated")
        assert pending.guidance == "updated"

    def test_edit_guidance_sets_user_source(self):
        """edit_guidance() sets source to 'user' from None."""
        pending = _make_pending_merge()
        pending.edit_guidance("user guidance")
        assert pending.guidance_source == "user"

    def test_edit_guidance_requires_pending_status(self):
        """edit_guidance() raises RuntimeError if status is not 'pending'."""
        pending = _make_pending_merge()
        pending.reject("bad merge")
        with pytest.raises(RuntimeError):
            pending.edit_guidance("too late")


# ===========================================================================
# 4. regenerate_guidance() raises RuntimeError without LLM
# ===========================================================================


class TestRegenerateGuidance:
    """Tests for regenerate_guidance() without LLM client."""

    def test_regenerate_guidance_raises_on_compress_without_llm(self):
        """regenerate_guidance() raises RuntimeError on PendingCompress without LLM."""
        pending = _make_pending_compress()
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            pending.regenerate_guidance()

    def test_regenerate_guidance_raises_on_merge_without_llm(self):
        """regenerate_guidance() raises RuntimeError on PendingMerge without LLM."""
        pending = _make_pending_merge()
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            pending.regenerate_guidance()

    def test_regenerate_guidance_raises_with_overrides_without_llm(self):
        """regenerate_guidance() raises RuntimeError even with llm_overrides kwargs."""
        pending = _make_pending_compress()
        with pytest.raises(RuntimeError, match="requires an LLM client"):
            pending.regenerate_guidance(model="gpt-4", temperature=0.5)


# ===========================================================================
# 5. Guidance fields on PendingCompress
# ===========================================================================


class TestGuidanceFieldsCompress:
    """Tests for guidance/guidance_source fields on PendingCompress."""

    def test_guidance_defaults_to_none(self):
        """PendingCompress.guidance defaults to None."""
        pending = _make_pending_compress()
        assert pending.guidance is None

    def test_guidance_source_defaults_to_none(self):
        """PendingCompress.guidance_source defaults to None."""
        pending = _make_pending_compress()
        assert pending.guidance_source is None

    def test_guidance_can_be_set_directly(self):
        """guidance can be set as a string field."""
        pending = _make_pending_compress()
        pending.guidance = "custom guidance"
        assert pending.guidance == "custom guidance"


# ===========================================================================
# 6. Guidance fields on PendingMerge
# ===========================================================================


class TestGuidanceFieldsMerge:
    """Tests for guidance/guidance_source fields on PendingMerge."""

    def test_guidance_defaults_to_none(self):
        """PendingMerge.guidance defaults to None."""
        pending = _make_pending_merge()
        assert pending.guidance is None

    def test_guidance_source_defaults_to_none(self):
        """PendingMerge.guidance_source defaults to None."""
        pending = _make_pending_merge()
        assert pending.guidance_source is None


# ===========================================================================
# 7. _public_actions includes guidance methods
# ===========================================================================


class TestPublicActionsIncludeGuidance:
    """Verify guidance methods appear in _public_actions whitelist."""

    def test_edit_guidance_in_compress_actions(self):
        """'edit_guidance' is in PendingCompress._public_actions."""
        pending = _make_pending_compress()
        assert "edit_guidance" in pending._public_actions

    def test_regenerate_guidance_not_in_compress_actions(self):
        """'regenerate_guidance' is NOT in PendingCompress._public_actions (stub)."""
        pending = _make_pending_compress()
        assert "regenerate_guidance" not in pending._public_actions

    def test_edit_guidance_in_merge_actions(self):
        """'edit_guidance' is in PendingMerge._public_actions."""
        pending = _make_pending_merge()
        assert "edit_guidance" in pending._public_actions

    def test_regenerate_guidance_not_in_merge_actions(self):
        """'regenerate_guidance' is NOT in PendingMerge._public_actions (stub)."""
        pending = _make_pending_merge()
        assert "regenerate_guidance" not in pending._public_actions

    def test_execute_tool_edit_guidance(self):
        """execute_tool('edit_guidance', ...) works through dispatch."""
        pending = _make_pending_compress(guidance="original")
        pending.execute_tool("edit_guidance", {"new_guidance": "tool-edited"})
        assert pending.guidance == "tool-edited"

    def test_apply_decision_edit_guidance(self):
        """apply_decision({'action': 'edit_guidance', ...}) works."""
        pending = _make_pending_compress(guidance="original")
        pending.apply_decision({
            "action": "edit_guidance",
            "args": {"new_guidance": "decision-edited"},
        })
        assert pending.guidance == "decision-edited"


# ===========================================================================
# 8. two_stage parameter on compress()
# ===========================================================================


class TestTwoStageParameter:
    """Tests for the two_stage parameter on Tract.compress()."""

    def test_two_stage_param_accepted(self):
        """compress(two_stage=True) does not raise."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True, two_stage=True)
        assert isinstance(pending, PendingCompress)
        t.close()

    def test_two_stage_false_accepted(self):
        """compress(two_stage=False) does not raise."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True, two_stage=False)
        assert isinstance(pending, PendingCompress)
        t.close()

    def test_two_stage_none_is_default(self):
        """compress() with no two_stage arg defaults to False."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True)
        # _two_stage is a proper field, defaulting to False (None is coerced)
        assert pending._two_stage is False  # type: ignore[attr-defined]
        t.close()

    def test_two_stage_stored_on_pending(self):
        """two_stage value is stored on the PendingCompress."""
        t = _make_compressible_tract()
        pending = t.compress(content="summary", review=True, two_stage=True)
        assert pending._two_stage is True  # type: ignore[attr-defined]
        t.close()
