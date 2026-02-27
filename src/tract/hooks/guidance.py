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

        Determines the operation type (compress or merge) and calls the
        appropriate LLM with guidance prompts. Updates self.guidance and
        self.guidance_source.

        Args:
            **llm_overrides: Override LLM parameters for guidance generation.

        Returns:
            The newly generated guidance text.

        Raises:
            RuntimeError: If status is not "pending" or no LLM client available.
        """
        self._require_pending()

        from tract.prompts.guidance import (
            COMPRESS_GUIDANCE_SYSTEM,
            MERGE_GUIDANCE_SYSTEM,
            build_compress_guidance_prompt,
            build_merge_guidance_prompt,
        )

        operation = self.operation  # type: ignore[attr-defined]
        tract = self.tract  # type: ignore[attr-defined]

        if operation == "compress":
            from tract.operations.compression import _build_messages_text

            if not tract._has_llm_client("compress"):
                raise RuntimeError(
                    "regenerate_guidance() requires an LLM client. "
                    "Call configure_llm() or pass api_key to Tract.open()."
                )
            llm_client = tract._resolve_llm_client("compress")

            # Build combined messages text for all groups
            groups = self._groups  # type: ignore[attr-defined]
            if groups is None:
                raise RuntimeError(
                    "Cannot regenerate guidance: no compression groups available. "
                    "This PendingCompress was created with content= (manual mode)."
                )
            blob_repo = tract._blob_repo
            all_text = "\n\n".join(
                _build_messages_text(group, blob_repo) for group in groups
            )

            response = llm_client.chat(
                [
                    {"role": "system", "content": COMPRESS_GUIDANCE_SYSTEM},
                    {"role": "user", "content": build_compress_guidance_prompt(
                        all_text, instructions=self._instructions  # type: ignore[attr-defined]
                    )},
                ],
                **llm_overrides,
            )

        elif operation == "merge":
            if not tract._has_llm_client("merge"):
                raise RuntimeError(
                    "regenerate_guidance() requires an LLM client. "
                    "Call configure_llm() or pass api_key to Tract.open()."
                )
            llm_client = tract._resolve_llm_client("merge")

            # Concatenate conflict descriptions
            conflicts = self.conflicts  # type: ignore[attr-defined]
            conflicts_text = "\n\n".join(str(c) for c in conflicts)

            response = llm_client.chat(
                [
                    {"role": "system", "content": MERGE_GUIDANCE_SYSTEM},
                    {"role": "user", "content": build_merge_guidance_prompt(
                        conflicts_text
                    )},
                ],
                **llm_overrides,
            )
        else:
            raise RuntimeError(
                f"regenerate_guidance() is not supported for operation {operation!r}"
            )

        guidance_text = response["choices"][0]["message"]["content"]
        self.guidance = guidance_text
        # Track source: if user had previously edited, it's now user+llm
        if self.guidance_source == "user":
            self.guidance_source = "user+llm"
        else:
            self.guidance_source = "llm"

        return guidance_text
