"""Content type system for Trace.

Defines 7 built-in content types as Pydantic models with a discriminated union
(ContentPayload). Each content type has behavioral hints (ContentTypeHints) for
compilation and compression.

The content type registry is per-Repo instance (implemented in Plan 03).
Built-in types and the discriminated union remain module-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from trace_context.exceptions import ContentValidationError


# ---------------------------------------------------------------------------
# Built-in content type models
# ---------------------------------------------------------------------------


class InstructionContent(BaseModel):
    """System-level instructions for the LLM."""

    content_type: Literal["instruction"] = "instruction"
    text: str


class DialogueContent(BaseModel):
    """A dialogue message (user, assistant, or system)."""

    content_type: Literal["dialogue"] = "dialogue"
    role: Literal["user", "assistant", "system"]
    text: str
    name: str | None = None


class ToolIOContent(BaseModel):
    """Tool call or tool result."""

    content_type: Literal["tool_io"] = "tool_io"
    tool_name: str
    direction: Literal["call", "result"]
    payload: dict
    status: Literal["success", "error"] | None = None


class ReasoningContent(BaseModel):
    """Internal reasoning or chain-of-thought."""

    content_type: Literal["reasoning"] = "reasoning"
    text: str


class ArtifactContent(BaseModel):
    """A produced artifact (code, document, config, etc.)."""

    content_type: Literal["artifact"] = "artifact"
    artifact_type: str
    content: str
    language: str | None = None


class OutputContent(BaseModel):
    """Final output content."""

    content_type: Literal["output"] = "output"
    text: str
    format: Literal["text", "markdown", "json"] = "text"


class FreeformContent(BaseModel):
    """Freeform content with no schema enforcement."""

    content_type: Literal["freeform"] = "freeform"
    payload: dict


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------

ContentPayload = Annotated[
    Union[
        InstructionContent,
        DialogueContent,
        ToolIOContent,
        ReasoningContent,
        ArtifactContent,
        OutputContent,
        FreeformContent,
    ],
    Field(discriminator="content_type"),
]

# TypeAdapter for built-in validation
_builtin_adapter = TypeAdapter(ContentPayload)

# Map of built-in content type names for quick membership check
BUILTIN_CONTENT_TYPES: set[str] = {
    "instruction",
    "dialogue",
    "tool_io",
    "reasoning",
    "artifact",
    "output",
    "freeform",
}


# ---------------------------------------------------------------------------
# Validation function (supports per-repo custom registry)
# ---------------------------------------------------------------------------


def validate_content(
    data: dict,
    *,
    custom_registry: dict[str, type[BaseModel]] | None = None,
) -> BaseModel:
    """Validate content data against built-in or custom type schema.

    If custom_registry is provided and data["content_type"] matches a
    registered custom type, validates against that model. Otherwise,
    falls through to the built-in discriminated union.

    Args:
        data: Content dict with a "content_type" field.
        custom_registry: Optional per-repo dict mapping type names to
            Pydantic model classes.

    Returns:
        Validated Pydantic model instance.

    Raises:
        ContentValidationError: If validation fails.
    """
    content_type = data.get("content_type")

    # Check custom registry first (if provided)
    if custom_registry and content_type in custom_registry:
        try:
            model_class = custom_registry[content_type]
            adapter = TypeAdapter(model_class)
            return adapter.validate_python(data)
        except ValidationError as e:
            raise ContentValidationError(
                f"Custom content type '{content_type}' validation failed: {e}"
            ) from e

    # Fall through to built-in discriminated union
    try:
        return _builtin_adapter.validate_python(data)
    except ValidationError as e:
        raise ContentValidationError(
            f"Content validation failed: {e}"
        ) from e


# ---------------------------------------------------------------------------
# Content type behavioral hints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentTypeHints:
    """Default behavioral hints for a content type.

    Used by the compiler and compressor to determine default behavior.
    """

    default_priority: str = "normal"  # Priority enum value name
    default_role: str = "assistant"
    compression_priority: int = 50  # 0=compress first, 100=protect
    aggregation_rule: str = "concatenate"


BUILTIN_TYPE_HINTS: dict[str, ContentTypeHints] = {
    "instruction": ContentTypeHints(
        default_priority="pinned",
        default_role="system",
        compression_priority=90,
    ),
    "dialogue": ContentTypeHints(
        default_priority="normal",
        default_role="user",
        compression_priority=50,
    ),
    "tool_io": ContentTypeHints(
        default_priority="normal",
        default_role="tool",
        compression_priority=30,
    ),
    "reasoning": ContentTypeHints(
        default_priority="normal",
        default_role="assistant",
        compression_priority=40,
    ),
    "artifact": ContentTypeHints(
        default_priority="normal",
        default_role="assistant",
        compression_priority=60,
    ),
    "output": ContentTypeHints(
        default_priority="normal",
        default_role="assistant",
        compression_priority=70,
    ),
    "freeform": ContentTypeHints(
        default_priority="normal",
        default_role="assistant",
        compression_priority=50,
    ),
}
