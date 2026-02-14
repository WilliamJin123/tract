"""Tests for the content type system.

Covers:
- Each built-in type validates with correct data
- Discriminated union selects correct type based on content_type field
- Invalid content_type raises ContentValidationError
- validate_content with custom_registry
- Round-trip model_dump() -> model_validate() for all content types
- BUILTIN_TYPE_HINTS has entries for all 7 types
"""

import pytest
from hypothesis import given
from pydantic import BaseModel, TypeAdapter

from tract.exceptions import ContentValidationError
from tract.models.content import (
    BUILTIN_CONTENT_TYPES,
    BUILTIN_TYPE_HINTS,
    ArtifactContent,
    ContentPayload,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
    validate_content,
)
from tests.strategies import any_content


# ---------------------------------------------------------------------------
# Individual type validation
# ---------------------------------------------------------------------------


class TestInstructionContent:
    def test_valid(self):
        c = InstructionContent(text="You are a helpful assistant.")
        assert c.content_type == "instruction"
        assert c.text == "You are a helpful assistant."

    def test_missing_text_fails(self):
        with pytest.raises(Exception):
            InstructionContent()  # type: ignore[call-arg]


class TestDialogueContent:
    def test_valid_user(self):
        c = DialogueContent(role="user", text="Hello")
        assert c.content_type == "dialogue"
        assert c.role == "user"

    def test_valid_with_name(self):
        c = DialogueContent(role="assistant", text="Hi", name="Claude")
        assert c.name == "Claude"

    def test_invalid_role(self):
        with pytest.raises(Exception):
            DialogueContent(role="invalid", text="Hello")  # type: ignore[arg-type]


class TestToolIOContent:
    def test_valid_call(self):
        c = ToolIOContent(
            tool_name="search",
            direction="call",
            payload={"query": "test"},
        )
        assert c.content_type == "tool_io"
        assert c.status is None

    def test_valid_result_with_status(self):
        c = ToolIOContent(
            tool_name="search",
            direction="result",
            payload={"results": [1, 2, 3]},
            status="success",
        )
        assert c.status == "success"


class TestReasoningContent:
    def test_valid(self):
        c = ReasoningContent(text="Let me think about this...")
        assert c.content_type == "reasoning"


class TestArtifactContent:
    def test_valid_code(self):
        c = ArtifactContent(
            artifact_type="code",
            content="def hello(): pass",
            language="python",
        )
        assert c.content_type == "artifact"
        assert c.language == "python"

    def test_valid_without_language(self):
        c = ArtifactContent(artifact_type="document", content="# Title")
        assert c.language is None


class TestOutputContent:
    def test_valid_default_format(self):
        c = OutputContent(text="The answer is 42.")
        assert c.format == "text"

    def test_valid_markdown(self):
        c = OutputContent(text="# Result", format="markdown")
        assert c.format == "markdown"


class TestFreeformContent:
    def test_valid(self):
        c = FreeformContent(payload={"any": "data", "nested": {"ok": True}})
        assert c.content_type == "freeform"


# ---------------------------------------------------------------------------
# Discriminated union
# ---------------------------------------------------------------------------


class TestDiscriminatedUnion:
    """Test that ContentPayload correctly dispatches based on content_type."""

    adapter = TypeAdapter(ContentPayload)

    @pytest.mark.parametrize("data,expected_type", [
        ({"content_type": "instruction", "text": "test"}, InstructionContent),
        ({"content_type": "dialogue", "role": "user", "text": "test"}, DialogueContent),
        ({"content_type": "tool_io", "tool_name": "search", "direction": "call", "payload": {}}, ToolIOContent),
        ({"content_type": "reasoning", "text": "thinking..."}, ReasoningContent),
        ({"content_type": "artifact", "artifact_type": "code", "content": "x=1"}, ArtifactContent),
        ({"content_type": "output", "text": "done"}, OutputContent),
        ({"content_type": "freeform", "payload": {"any": "thing"}}, FreeformContent),
    ])
    def test_dispatch_by_content_type(self, data, expected_type):
        result = self.adapter.validate_python(data)
        assert isinstance(result, expected_type)

    def test_invalid_content_type(self):
        with pytest.raises(Exception):
            self.adapter.validate_python(
                {"content_type": "nonexistent", "data": "test"}
            )


# ---------------------------------------------------------------------------
# validate_content function
# ---------------------------------------------------------------------------


class TestValidateContent:
    def test_builtin_type(self):
        result = validate_content({"content_type": "instruction", "text": "test"})
        assert isinstance(result, InstructionContent)

    def test_invalid_data_raises_content_validation_error(self):
        with pytest.raises(ContentValidationError):
            validate_content({"content_type": "nonexistent_type", "text": "test"})

    def test_invalid_fields_raises_content_validation_error(self):
        with pytest.raises(ContentValidationError):
            # dialogue requires 'role' field
            validate_content({"content_type": "dialogue", "text": "test"})

    def test_custom_registry_type(self):
        """Test validate_content with per-repo custom type registry."""

        class CustomAnalysis(BaseModel):
            content_type: str = "analysis"
            summary: str
            score: float

        registry = {"analysis": CustomAnalysis}
        result = validate_content(
            {"content_type": "analysis", "summary": "good", "score": 0.95},
            custom_registry=registry,
        )
        assert isinstance(result, CustomAnalysis)
        assert result.summary == "good"
        assert result.score == 0.95

    def test_custom_registry_invalid_data(self):
        """Custom type with invalid data raises ContentValidationError."""

        class CustomType(BaseModel):
            content_type: str = "custom"
            required_field: int

        registry = {"custom": CustomType}
        with pytest.raises(ContentValidationError):
            validate_content(
                {"content_type": "custom", "required_field": "not_an_int_but_coercible_nope"},
                custom_registry=registry,
            )

    def test_custom_registry_does_not_shadow_builtin(self):
        """When custom registry has no matching key, builtins are used."""
        registry = {}  # empty registry
        result = validate_content(
            {"content_type": "instruction", "text": "test"},
            custom_registry=registry,
        )
        assert isinstance(result, InstructionContent)

    def test_custom_registry_can_shadow_builtin(self):
        """A custom registry CAN override a built-in type name."""

        class CustomInstruction(BaseModel):
            content_type: str = "instruction"
            text: str
            priority_override: int = 0

        registry = {"instruction": CustomInstruction}
        result = validate_content(
            {"content_type": "instruction", "text": "test", "priority_override": 5},
            custom_registry=registry,
        )
        assert isinstance(result, CustomInstruction)
        assert result.priority_override == 5


# ---------------------------------------------------------------------------
# Round-trip property tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    @given(content=any_content)
    def test_model_dump_validate_round_trip(self, content):
        """All content types survive model_dump -> model_validate round-trip."""
        dumped = content.model_dump()
        restored = type(content).model_validate(dumped)
        assert restored == content

    @given(content=any_content)
    def test_model_dump_json_mode_round_trip(self, content):
        """All content types survive model_dump(mode='json') -> model_validate round-trip."""
        dumped = content.model_dump(mode="json")
        restored = type(content).model_validate(dumped)
        assert restored == content


# ---------------------------------------------------------------------------
# Type hints registry
# ---------------------------------------------------------------------------


class TestBuiltinTypeHints:
    def test_all_seven_types_present(self):
        expected = {"instruction", "dialogue", "tool_io", "reasoning", "artifact", "output", "freeform"}
        assert set(BUILTIN_TYPE_HINTS.keys()) == expected

    def test_instruction_is_pinned(self):
        assert BUILTIN_TYPE_HINTS["instruction"].default_priority == "pinned"

    def test_instruction_is_system_role(self):
        assert BUILTIN_TYPE_HINTS["instruction"].default_role == "system"

    def test_tool_io_compresses_aggressively(self):
        assert BUILTIN_TYPE_HINTS["tool_io"].compression_priority < 50

    def test_builtin_content_types_set(self):
        assert BUILTIN_CONTENT_TYPES == {
            "instruction", "dialogue", "tool_io", "reasoning",
            "artifact", "output", "freeform",
        }
