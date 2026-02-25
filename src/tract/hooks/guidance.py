"""Two-stage guidance pattern: judgment (what to cover) + execution (produce output).

LLM operations decompose into guidance (what should the output cover?) and
execution (produce the output). Guidance is the harder cognitive task --
judgment about what matters. This mixin makes guidance a first-class editable
field on Pending subclasses.

Mixed into PendingCompress and PendingMerge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class GuidanceMixin:
    """Mixin for Pending subclasses that have LLM-generated output with editable guidance.

    Mixed into PendingCompress and PendingMerge.

    Fields (set on the dataclass that mixes this in):
    - guidance: str | None -- the guidance text
    - guidance_source: str | None -- None (one-shot) | "user" | "llm" | "user+llm"
    """

    def edit_guidance(self, new_guidance: str) -> None:
        """Replace the current guidance text.

        Args:
            new_guidance: New guidance text to use for retry.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()  # type: ignore[attr-defined]  # from Pending base
        self.guidance = new_guidance  # type: ignore[attr-defined]
        # Track that guidance was user-edited
        if self.guidance_source in (None, "llm"):  # type: ignore[attr-defined]
            self.guidance_source = "user"  # type: ignore[attr-defined]
        elif self.guidance_source == "user":  # type: ignore[attr-defined]
            # Already user, keep it
            pass
        else:
            # "user+llm" or anything else -- user is editing
            self.guidance_source = "user+llm"  # type: ignore[attr-defined]

    def regenerate_guidance(self, **llm_overrides: Any) -> str:
        """Re-generate guidance using LLM.

        This is a stub -- actual LLM call will be wired when two_stage
        compress is implemented. For now, raise NotImplementedError.

        Args:
            **llm_overrides: Override LLM parameters for guidance generation.

        Returns:
            The newly generated guidance text.

        Raises:
            NotImplementedError: Until two_stage=True is fully wired.
        """
        raise NotImplementedError(
            "regenerate_guidance() requires LLM wiring (two_stage=True on compress/merge)"
        )
