"""Tests for content improvement helpers and prompts (Phase 4).

Tests prompt templates from prompts/guidance.py and prompts/improve.py,
and the _improve_instructions helper from hooks/improve.py.
"""

from __future__ import annotations

import pytest


# ===========================================================================
# 1. Prompt template tests — prompts/guidance.py
# ===========================================================================


class TestCompressGuidancePrompts:
    """Tests for compress guidance prompt templates."""

    def test_compress_guidance_system_is_nonempty(self):
        """COMPRESS_GUIDANCE_SYSTEM is a non-empty string."""
        from tract.prompts.guidance import COMPRESS_GUIDANCE_SYSTEM

        assert isinstance(COMPRESS_GUIDANCE_SYSTEM, str)
        assert len(COMPRESS_GUIDANCE_SYSTEM) > 0

    def test_compress_guidance_system_does_not_write_summary(self):
        """COMPRESS_GUIDANCE_SYSTEM instructs NOT to write the summary."""
        from tract.prompts.guidance import COMPRESS_GUIDANCE_SYSTEM

        assert "NOT" in COMPRESS_GUIDANCE_SYSTEM or "not" in COMPRESS_GUIDANCE_SYSTEM.lower()

    def test_build_compress_guidance_prompt_basic(self):
        """build_compress_guidance_prompt returns non-empty string with messages."""
        from tract.prompts.guidance import build_compress_guidance_prompt

        result = build_compress_guidance_prompt("User: Hello\nAssistant: Hi")
        assert isinstance(result, str)
        assert "Hello" in result
        assert len(result) > 20

    def test_build_compress_guidance_prompt_with_instructions(self):
        """build_compress_guidance_prompt includes instructions when provided."""
        from tract.prompts.guidance import build_compress_guidance_prompt

        result = build_compress_guidance_prompt(
            "User: Hello", instructions="Focus on technical details"
        )
        assert "Focus on technical details" in result

    def test_build_compress_guidance_prompt_without_instructions(self):
        """build_compress_guidance_prompt omits instruction section when None."""
        from tract.prompts.guidance import build_compress_guidance_prompt

        result = build_compress_guidance_prompt("User: Hello", instructions=None)
        assert "User instructions:" not in result


class TestMergeGuidancePrompts:
    """Tests for merge guidance prompt templates."""

    def test_merge_guidance_system_is_nonempty(self):
        """MERGE_GUIDANCE_SYSTEM is a non-empty string."""
        from tract.prompts.guidance import MERGE_GUIDANCE_SYSTEM

        assert isinstance(MERGE_GUIDANCE_SYSTEM, str)
        assert len(MERGE_GUIDANCE_SYSTEM) > 0

    def test_merge_guidance_system_does_not_resolve(self):
        """MERGE_GUIDANCE_SYSTEM instructs NOT to resolve conflicts."""
        from tract.prompts.guidance import MERGE_GUIDANCE_SYSTEM

        assert "NOT" in MERGE_GUIDANCE_SYSTEM or "not" in MERGE_GUIDANCE_SYSTEM.lower()

    def test_build_merge_guidance_prompt_basic(self):
        """build_merge_guidance_prompt returns non-empty string with conflicts."""
        from tract.prompts.guidance import build_merge_guidance_prompt

        result = build_merge_guidance_prompt("Conflict 1: ours vs theirs")
        assert isinstance(result, str)
        assert "Conflict 1" in result
        assert len(result) > 20

    def test_build_merge_guidance_prompt_with_instructions(self):
        """build_merge_guidance_prompt includes instructions when provided."""
        from tract.prompts.guidance import build_merge_guidance_prompt

        result = build_merge_guidance_prompt(
            "conflict data", instructions="Prefer ours"
        )
        assert "Prefer ours" in result

    def test_build_merge_guidance_prompt_without_instructions(self):
        """build_merge_guidance_prompt omits instruction section when None."""
        from tract.prompts.guidance import build_merge_guidance_prompt

        result = build_merge_guidance_prompt("conflict data", instructions=None)
        assert "User instructions:" not in result


# ===========================================================================
# 2. Prompt template tests — prompts/improve.py
# ===========================================================================


class TestImprovePrompts:
    """Tests for content improvement prompt templates."""

    def test_improve_content_system_is_nonempty(self):
        """IMPROVE_CONTENT_SYSTEM is a non-empty string."""
        from tract.prompts.improve import IMPROVE_CONTENT_SYSTEM

        assert isinstance(IMPROVE_CONTENT_SYSTEM, str)
        assert len(IMPROVE_CONTENT_SYSTEM) > 0

    def test_improve_content_system_preserves_meaning(self):
        """IMPROVE_CONTENT_SYSTEM instructs to preserve meaning."""
        from tract.prompts.improve import IMPROVE_CONTENT_SYSTEM

        assert "meaning" in IMPROVE_CONTENT_SYSTEM.lower()

    def test_build_improve_prompt_default_context(self):
        """build_improve_prompt with default context returns message prompt."""
        from tract.prompts.improve import build_improve_prompt

        result = build_improve_prompt("hello world")
        assert isinstance(result, str)
        assert "hello world" in result
        assert "message" in result

    def test_build_improve_prompt_custom_context(self):
        """build_improve_prompt with custom context uses that context."""
        from tract.prompts.improve import build_improve_prompt

        result = build_improve_prompt("rough summary", context="summary")
        assert "summary" in result
        assert "rough summary" in result

    def test_build_improve_prompt_returns_only(self):
        """build_improve_prompt instructs to return only the improved version."""
        from tract.prompts.improve import build_improve_prompt

        result = build_improve_prompt("test")
        assert "only" in result.lower() or "Return" in result


# ===========================================================================
# 3. _improve_instructions helper
# ===========================================================================


class TestImproveInstructions:
    """Tests for _improve_instructions helper."""

    def test_returns_dict_with_both_keys(self):
        """_improve_instructions returns dict with original and effective."""
        from tract.hooks.improve import _improve_instructions

        result = _improve_instructions("rough text", "polished text")
        assert isinstance(result, dict)
        assert "original_instructions" in result
        assert "effective_instructions" in result

    def test_original_preserved(self):
        """_improve_instructions preserves the original text."""
        from tract.hooks.improve import _improve_instructions

        result = _improve_instructions("original", "improved")
        assert result["original_instructions"] == "original"

    def test_effective_is_improved(self):
        """_improve_instructions stores the improved text as effective."""
        from tract.hooks.improve import _improve_instructions

        result = _improve_instructions("original", "improved")
        assert result["effective_instructions"] == "improved"

    def test_empty_strings(self):
        """_improve_instructions handles empty strings."""
        from tract.hooks.improve import _improve_instructions

        result = _improve_instructions("", "")
        assert result["original_instructions"] == ""
        assert result["effective_instructions"] == ""


# ===========================================================================
# 4. Import smoke tests
# ===========================================================================


class TestImportSmoke:
    """Verify all new modules are importable."""

    def test_import_guidance_mixin(self):
        """GuidanceMixin is importable from tract.hooks.guidance."""
        from tract.hooks.guidance import GuidanceMixin

        assert GuidanceMixin is not None

    def test_import_guidance_from_hooks_init(self):
        """GuidanceMixin is importable from tract.hooks."""
        from tract.hooks import GuidanceMixin

        assert GuidanceMixin is not None

    def test_import_prompts_guidance(self):
        """prompts.guidance module is importable."""
        from tract.prompts.guidance import (
            COMPRESS_GUIDANCE_SYSTEM,
            MERGE_GUIDANCE_SYSTEM,
            build_compress_guidance_prompt,
            build_merge_guidance_prompt,
        )

        assert COMPRESS_GUIDANCE_SYSTEM
        assert MERGE_GUIDANCE_SYSTEM
        assert callable(build_compress_guidance_prompt)
        assert callable(build_merge_guidance_prompt)

    def test_import_prompts_improve(self):
        """prompts.improve module is importable."""
        from tract.prompts.improve import (
            IMPROVE_CONTENT_SYSTEM,
            build_improve_prompt,
        )

        assert IMPROVE_CONTENT_SYSTEM
        assert callable(build_improve_prompt)

    def test_import_hooks_improve(self):
        """hooks.improve module is importable."""
        from tract.hooks.improve import _improve_content, _improve_instructions

        assert callable(_improve_content)
        assert callable(_improve_instructions)
