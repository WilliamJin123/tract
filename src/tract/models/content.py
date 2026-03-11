"""Content type system for Trace.

Defines 8 built-in content types as Pydantic models with a discriminated union
(ContentPayload). Each content type has behavioral hints (ContentTypeHints) for
compilation and compression.

The content type registry is per-Repo instance (implemented in Plan 03).
Built-in types and the discriminated union remain module-level.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from tract.exceptions import ContentValidationError
from tract.models.session import SessionContent


# ---------------------------------------------------------------------------
# Built-in content type models
# ---------------------------------------------------------------------------


class InstructionContent(BaseModel):
    """System-level instructions for the LLM. Pinned by default.

    When ``name`` is set, the compiler deduplicates: same name -> closest
    to HEAD wins (directive override-by-name semantics).
    """

    content_type: Literal["instruction"] = "instruction"
    text: str
    name: str | None = None


class DialogueContent(BaseModel):
    """A dialogue message (user, assistant, or system)."""

    content_type: Literal["dialogue"] = "dialogue"
    role: Literal["user", "assistant", "system", "tool"]
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
    format: Literal["parsed", "raw", "think_tags", "anthropic"] = "parsed"


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


class ConfigContent(BaseModel):
    """Key-value config settings stored in the DAG.

    The system reads these; the LLM never sees them (compilable=False).
    """

    model_config = ConfigDict(frozen=True)

    content_type: Literal["config"] = "config"
    settings: dict[str, Any]


class MetadataContent(BaseModel):
    """Structured metadata attached to commits via the content system.

    Used for annotations, tags, and structured data that should be
    preserved in the DAG but not compiled to LLM messages.
    """

    model_config = ConfigDict(frozen=True)

    content_type: Literal["metadata"] = "metadata"
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)
    path: str | None = None


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
        SessionContent,
        ConfigContent,
        MetadataContent,
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
    "session",
    "config",
    "metadata",
}


# ---------------------------------------------------------------------------
# Validation function (supports per-repo custom registry)
# ---------------------------------------------------------------------------


def validate_content(
    data: dict[str, Any],
    *,
    custom_registry: dict[str, type[BaseModel]] | None = None,
) -> BaseModel:
    """Validate content data against built-in or custom type schema.

    If custom_registry is provided and data["content_type"] matches a
    registered custom type, validates against that model. Otherwise,
    falls through to the built-in discriminated union.

    Error messages are designed to be actionable for LLMs: they state
    which content_type was used, what fields are required vs provided,
    and what valid types exist.

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

    # --- No content_type at all ---
    if content_type is None:
        valid_types = _all_valid_types(custom_registry)
        raise ContentValidationError(
            f"Missing required field 'content_type'. "
            f"Provided fields: {sorted(data.keys())}. "
            f"Valid content_type values: {valid_types}"
        )

    # Check custom registry first (if provided)
    if custom_registry and content_type in custom_registry:
        try:
            model_class = custom_registry[content_type]
            adapter = TypeAdapter(model_class)
            return adapter.validate_python(data)
        except ValidationError as e:
            raise ContentValidationError(
                _format_field_error(content_type, data, model_class)
            ) from e

    # --- Unknown content_type ---
    if content_type not in BUILTIN_CONTENT_TYPES:
        valid_types = _all_valid_types(custom_registry)
        raise ContentValidationError(
            f"Unknown content_type '{content_type}'. "
            f"Valid types: {valid_types}. "
            f"Use one of these as the 'content_type' field value."
        )

    # Fall through to built-in discriminated union
    try:
        return _builtin_adapter.validate_python(data)
    except ValidationError as e:
        model_class = _BUILTIN_TYPE_MODELS.get(content_type)
        if model_class is not None:
            raise ContentValidationError(
                _format_field_error(content_type, data, model_class)
            ) from e
        # Fallback (should not happen for known types)
        raise ContentValidationError(
            f"Content validation failed for type '{content_type}': {e}"
        ) from e


# Map content_type name -> model class for error formatting
_BUILTIN_TYPE_MODELS: dict[str, type[BaseModel]] = {
    "instruction": InstructionContent,
    "dialogue": DialogueContent,
    "tool_io": ToolIOContent,
    "reasoning": ReasoningContent,
    "artifact": ArtifactContent,
    "output": OutputContent,
    "freeform": FreeformContent,
    "session": SessionContent,
    "config": ConfigContent,
    "metadata": MetadataContent,
}


def _all_valid_types(
    custom_registry: dict[str, type[BaseModel]] | None = None,
) -> list[str]:
    """Return sorted list of all valid content_type values."""
    types = sorted(BUILTIN_CONTENT_TYPES)
    if custom_registry:
        types = sorted(set(types) | set(custom_registry.keys()))
    return types


def _format_field_error(
    content_type: str,
    data: dict[str, Any],
    model_class: type[BaseModel],
) -> str:
    """Build an LLM-actionable error message for field validation failures."""
    fields = model_class.model_fields
    required = sorted(
        k for k, v in fields.items()
        if v.is_required() and k != "content_type"
    )
    optional = sorted(
        k for k, v in fields.items()
        if not v.is_required() and k != "content_type"
    )
    provided = sorted(k for k in data.keys() if k != "content_type")
    missing = [f for f in required if f not in data]
    extra = [f for f in provided if f not in fields]

    parts = [f"Invalid fields for content_type '{content_type}'."]
    parts.append(f"Required fields: {required}.")
    if optional:
        parts.append(f"Optional fields: {optional}.")
    parts.append(f"You provided: {provided}.")
    if missing:
        parts.append(f"Missing required: {missing}.")
    if extra:
        parts.append(f"Unknown fields (will be ignored): {extra}.")

    # Add a corrective example
    example_fields = {f: "..." for f in required}
    example_fields["content_type"] = content_type
    parts.append(f"Example: {example_fields}")

    return " ".join(parts)


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
    format_roles: frozenset[str] = frozenset()
    summary_instruction: str = ""
    compilable: bool = True


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
        default_priority="skip",
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
    "session": ContentTypeHints(
        default_priority="pinned",
        default_role="system",
        compression_priority=95,  # Protect session boundaries from compression
    ),
    "config": ContentTypeHints(
        default_priority="normal",
        default_role="system",
        compression_priority=85,
        compilable=False,
    ),
    "metadata": ContentTypeHints(
        format_roles=frozenset({"system"}),
        summary_instruction="Preserve metadata kind and key data points.",
        compilable=False,
    ),
}
