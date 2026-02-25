"""Introspection utilities for auto-generating agent interfaces from Pending subclasses.

Reads type hints, docstrings, and parameter defaults from Pending subclass
methods to produce JSON Schema tool definitions, structured dicts, and
human/LLM-readable API descriptions.
"""

from __future__ import annotations

import inspect
from typing import Any, get_type_hints

# Mapping from Python types to JSON Schema types
_PYTHON_TO_JSON_SCHEMA: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


def _python_type_to_json_schema(annotation: Any) -> dict:
    """Convert a Python type annotation to a JSON Schema type descriptor.

    Handles basic types (str, int, float, bool, dict, list) and falls back
    to {"type": "string"} for unrecognized types (safe default for LLM tools).

    Args:
        annotation: A Python type annotation (from inspect or typing).

    Returns:
        A dict like {"type": "string"} or {"type": "integer"}.
    """
    # Handle None / missing annotation
    if annotation is inspect.Parameter.empty or annotation is None:
        return {"type": "string"}

    # Direct type match
    if annotation in _PYTHON_TO_JSON_SCHEMA:
        return {"type": _PYTHON_TO_JSON_SCHEMA[annotation]}

    # Handle typing generics (e.g. list[str], dict[str, Any])
    origin = getattr(annotation, "__origin__", None)
    if origin is not None:
        if origin is list:
            return {"type": "array"}
        if origin is dict:
            return {"type": "object"}

    # Fallback: treat as string (safe default for LLM consumption)
    return {"type": "string"}


def _format_type_name(annotation: Any) -> str:
    """Format a type annotation as a human-readable string.

    Args:
        annotation: A Python type annotation.

    Returns:
        A readable string like "str", "int", "list[str]", etc.
    """
    if annotation is inspect.Parameter.empty or annotation is None:
        return "Any"

    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # For typing generics like list[str], dict[str, Any]
    return str(annotation).replace("typing.", "")


def _extract_param_description(docstring: str | None, param_name: str) -> str:
    """Extract parameter description from a docstring's Args section.

    Looks for patterns like:
        param_name: Description text.

    Args:
        docstring: The method's docstring (may be None).
        param_name: The parameter name to find.

    Returns:
        The description text, or an empty string if not found.
    """
    if not docstring:
        return ""

    lines = docstring.split("\n")
    in_args = False
    for line in lines:
        stripped = line.strip()
        # Detect Args: section header
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        # Detect next section header (Returns:, Raises:, etc.)
        if in_args and stripped and not stripped[0].isspace() and stripped.endswith(":"):
            # Check if this looks like a section header (single word + colon)
            if " " not in stripped.rstrip(":"):
                in_args = False
                continue
        if in_args:
            # Look for "param_name: description" pattern
            if stripped.startswith(f"{param_name}:"):
                desc = stripped[len(param_name) + 1 :].strip()
                return desc
    return ""


def method_to_tool_schema(method: Any, name: str) -> dict:
    """Convert a method to a JSON Schema tool definition.

    Reads type hints, docstring, and parameter defaults to produce a
    tool definition compatible with OpenAI/Anthropic function calling format.

    Args:
        method: A bound or unbound method.
        name: The tool name to use in the schema.

    Returns:
        A dict with the structure::

            {
                "type": "function",
                "function": {
                    "name": "edit_summary",
                    "description": "Replace the summary text at the given index.",
                    "parameters": {
                        "type": "object",
                        "properties": { ... },
                        "required": [...]
                    }
                }
            }
    """
    sig = inspect.signature(method)
    docstring = inspect.getdoc(method) or ""

    # Extract first line of docstring as description
    description = docstring.split("\n")[0].strip() if docstring else f"Execute the {name} action."

    # Try to get type hints; fall back to parameter annotations
    try:
        hints = get_type_hints(method)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        # Skip 'self'
        if param_name == "self":
            continue

        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        # Get type from hints first, then parameter annotation
        annotation = hints.get(param_name, param.annotation)

        prop = _python_type_to_json_schema(annotation)

        # Add description from docstring
        param_desc = _extract_param_description(docstring, param_name)
        if param_desc:
            prop["description"] = param_desc

        properties[param_name] = prop

        # Parameters without defaults are required
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        parameters["required"] = required

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def pending_to_dict(pending: Any) -> dict:
    """Serialize a Pending to a structured dict for LLM consumption.

    Extracts all public fields (not starting with _), the operation name,
    pending_id, status, and available actions.

    Args:
        pending: A Pending instance (or subclass).

    Returns:
        A dict with the structure::

            {
                "operation": "compress",
                "pending_id": "abc123",
                "status": "pending",
                "fields": {
                    "summaries": ["..."],
                    "source_commits": ["..."],
                    ...
                },
                "available_actions": ["approve", "reject", "edit_summary", ...]
            }
    """
    import dataclasses

    fields: dict[str, Any] = {}

    # Get all dataclass fields, filter to public non-internal ones
    skip_fields = {"operation", "pending_id", "status", "tract", "triggered_by", "rejection_reason", "created_at"}
    for f in dataclasses.fields(pending):
        if f.name.startswith("_"):
            continue
        if f.name in skip_fields:
            continue
        value = getattr(pending, f.name)
        # Convert non-serializable types to strings
        fields[f.name] = _serialize_value(value)

    return {
        "operation": pending.operation,
        "pending_id": pending.pending_id,
        "status": pending.status,
        "fields": fields,
        "available_actions": sorted(pending._public_actions),
    }


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable form.

    Handles common types: lists, dicts, sets, tuples, primitives.
    Falls back to str() for unrecognized types.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, set):
        return sorted(_serialize_value(v) for v in value)
    # Fallback
    return str(value)


def pending_to_tools(pending: Any) -> list[dict]:
    """Generate tool definitions for all public actions on a Pending.

    For each method name in ``_public_actions``, retrieves the method and
    converts it to a JSON Schema tool definition.

    Args:
        pending: A Pending instance (or subclass).

    Returns:
        A list of JSON Schema tool definitions.
    """
    tools = []
    for action_name in sorted(pending._public_actions):
        method = getattr(pending, action_name, None)
        if method is None:
            continue
        tools.append(method_to_tool_schema(method, action_name))
    return tools


def pending_describe_api(pending: Any) -> str:
    """Generate a human/LLM-readable API description.

    Produces a markdown-formatted string describing the Pending's class,
    fields, and available actions with their signatures and docstrings.

    Args:
        pending: A Pending instance (or subclass).

    Returns:
        A formatted markdown string.
    """
    import dataclasses

    lines: list[str] = []
    class_name = type(pending).__name__

    lines.append(f"## {class_name} API")
    lines.append("")

    # Class docstring
    class_doc = inspect.getdoc(type(pending))
    if class_doc:
        first_line = class_doc.split("\n")[0].strip()
        lines.append(first_line)
        lines.append("")

    # Fields
    lines.append("### Fields")
    skip_fields = {"operation", "pending_id", "status", "tract", "triggered_by", "rejection_reason", "created_at"}
    for f in dataclasses.fields(pending):
        if f.name.startswith("_"):
            continue
        if f.name in skip_fields:
            continue
        value = getattr(pending, f.name)
        type_name = _format_type_name(f.type) if isinstance(f.type, type) else str(f.type)
        lines.append(f"- **{f.name}**: {type_name} = {_format_field_value(value)}")
    lines.append("")

    # Actions
    lines.append("### Actions")
    for action_name in sorted(pending._public_actions):
        method = getattr(pending, action_name, None)
        if method is None:
            continue

        sig = inspect.signature(method)
        # Build a simplified signature string (skip self)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            annotation = param.annotation
            type_str = _format_type_name(annotation) if annotation is not inspect.Parameter.empty else "Any"
            if param.default is not inspect.Parameter.empty:
                params.append(f"{pname}: {type_str} = {param.default!r}")
            else:
                params.append(f"{pname}: {type_str}")

        params_str = ", ".join(params)

        # Return type
        return_annotation = sig.return_annotation
        if return_annotation is not inspect.Signature.empty:
            return_str = f" -> {_format_type_name(return_annotation)}"
        else:
            return_str = ""

        # Docstring first line
        doc = inspect.getdoc(method) or ""
        doc_line = doc.split("\n")[0].strip() if doc else ""
        desc = f" -- {doc_line}" if doc_line else ""

        lines.append(f"- **{action_name}**({params_str}){return_str}{desc}")

    lines.append("")
    return "\n".join(lines)


def _format_field_value(value: Any) -> str:
    """Format a field value for display in the API description.

    Truncates long lists/strings to keep output readable.
    """
    if isinstance(value, str):
        if len(value) > 60:
            return repr(value[:57] + "...")
        return repr(value)
    if isinstance(value, list):
        if len(value) > 3:
            return f"[{len(value)} items]"
        return repr(value)
    if isinstance(value, dict):
        if len(value) > 3:
            return f"{{{len(value)} entries}}"
        return repr(value)
    return repr(value)
