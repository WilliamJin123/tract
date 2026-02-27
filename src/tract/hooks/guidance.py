"""Two-stage guidance pattern: judgment (what to cover) + execution (produce output).

LLM operations decompose into guidance (what should the output cover?) and
execution (produce the output). Guidance is the harder cognitive task --
judgment about what matters. This mixin makes guidance a first-class editable
field on Pending subclasses.

Mixed into PendingCompress and PendingMerge.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable


@runtime_checkable
class GuidanceHost(Protocol):
    """Structural contract for classes that mix in GuidanceMixin.

    Pending subclasses using GuidanceMixin must declare these fields
    and methods. Enforced at type-check time (mypy/pyright).
    """

    guidance: str | None
    guidance_source: str | None

    def _require_pending(self) -> None: ...


class GuidanceMixin:
    """Mixin for Pending subclasses that have LLM-generated output with editable guidance.

    Mixed into PendingCompress and PendingMerge.

    Host classes must satisfy :class:`GuidanceHost` (i.e. declare
    ``guidance``, ``guidance_source``, and ``_require_pending()``).
    """

    if TYPE_CHECKING:
        # Let type checkers see the fields from GuidanceHost
        guidance: str | None
        guidance_source: str | None
        def _require_pending(self) -> None: ...

    def edit_guidance(self, new_guidance: str) -> None:
        """Replace the current guidance text.

        Args:
            new_guidance: New guidance text to use for retry.

        Raises:
            RuntimeError: If status is not "pending".
        """
        self._require_pending()
        self.guidance = new_guidance
        # Track that guidance was user-edited
        if self.guidance_source in (None, "llm"):
            self.guidance_source = "user"
        elif self.guidance_source == "user":
            # Already user, keep it
            pass
        else:
            # "user+llm" or anything else -- user is editing
            self.guidance_source = "user+llm"

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
            "regenerate_guidance() is not yet implemented. Use edit_guidance() to set guidance manually."
        )
