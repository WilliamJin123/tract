"""Hypothesis strategies for Trace content types.

Provides strategies for generating valid instances of all 7 built-in
content types, plus a combined `any_content` strategy.
"""

from hypothesis import strategies as st

from trace_context.models.content import (
    ArtifactContent,
    DialogueContent,
    FreeformContent,
    InstructionContent,
    OutputContent,
    ReasoningContent,
    ToolIOContent,
)

# Text that is valid for content (non-empty, reasonable size)
content_text = st.text(
    min_size=1,
    max_size=5000,
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Z", "S")),
)

instruction_content = st.builds(
    InstructionContent,
    text=content_text,
)

dialogue_content = st.builds(
    DialogueContent,
    role=st.sampled_from(["user", "assistant", "system"]),
    text=content_text,
    name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
)

tool_io_content = st.builds(
    ToolIOContent,
    tool_name=st.text(min_size=1, max_size=100),
    direction=st.sampled_from(["call", "result"]),
    payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(max_size=200),
        max_size=10,
    ),
    status=st.one_of(st.none(), st.sampled_from(["success", "error"])),
)

reasoning_content = st.builds(
    ReasoningContent,
    text=content_text,
)

artifact_content = st.builds(
    ArtifactContent,
    artifact_type=st.sampled_from(["code", "document", "config"]),
    content=content_text,
    language=st.one_of(st.none(), st.sampled_from(["python", "javascript", "rust"])),
)

output_content = st.builds(
    OutputContent,
    text=content_text,
    format=st.sampled_from(["text", "markdown", "json"]),
)

freeform_content = st.builds(
    FreeformContent,
    payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(max_size=200),
        max_size=10,
    ),
)

any_content = st.one_of(
    instruction_content,
    dialogue_content,
    tool_io_content,
    reasoning_content,
    artifact_content,
    output_content,
    freeform_content,
)
