"""Convert ToolDefinitions to typed Python callables for framework integration.

Every major agent framework (Agno, LangChain, CrewAI, LangGraph) accepts
plain Python functions and introspects their signatures to build tool schemas.
This module converts tract's ToolDefinition objects (JSON schema + lambda handler)
into properly typed callables that any framework can consume.

Usage::

    from tract.toolkit.callables import tools_to_callables

    callables = tools_to_callables(tool_definitions)
    # Pass to any framework: Agent(tools=[*callables])
"""

from __future__ import annotations

import inspect
from typing import Any

from tract.toolkit.models import ToolDefinition

# JSON Schema type string -> Python type
_SCHEMA_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
}


def tool_to_callable(tool_def: ToolDefinition) -> Any:
    """Convert a single ToolDefinition into a typed Python callable.

    The returned function has:
    - ``__name__`` set to the tool name
    - ``__doc__`` set to the tool description
    - ``__signature__`` with typed parameters derived from JSON schema
    - ``__annotations__`` matching the signature

    Args:
        tool_def: A ToolDefinition with name, description, parameters schema,
            and handler.

    Returns:
        A callable with proper type annotations for framework introspection.
    """
    schema_props = tool_def.parameters.get("properties", {})
    required = set(tool_def.parameters.get("required", []))

    # Build inspect.Parameter list from JSON schema
    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}

    for name, prop in schema_props.items():
        py_type = _schema_to_type(prop)
        annotations[name] = py_type

        if name in required:
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=py_type,
                )
            )
        else:
            default = _schema_default(prop)
            params.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=py_type,
                )
            )

    annotations["return"] = str
    sig = inspect.Signature(params, return_annotation=str)

    # Capture handler in closure
    handler = tool_def.handler

    def wrapper(*args: Any, **kwargs: Any) -> str:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        result = handler(**bound.arguments)
        return str(result) if result is not None else ""

    wrapper.__name__ = tool_def.name
    wrapper.__qualname__ = tool_def.name
    wrapper.__doc__ = tool_def.description
    wrapper.__signature__ = sig  # type: ignore[attr-defined]
    wrapper.__annotations__ = annotations

    return wrapper


def tools_to_callables(tool_defs: list[ToolDefinition]) -> list[Any]:
    """Convert a list of ToolDefinitions into typed Python callables.

    Convenience wrapper around :func:`tool_to_callable`.

    Args:
        tool_defs: List of ToolDefinition objects.

    Returns:
        List of typed callables, one per tool definition.
    """
    return [tool_to_callable(td) for td in tool_defs]


def _schema_to_type(prop: dict) -> type:
    """Map a JSON Schema property dict to a Python type."""
    schema_type = prop.get("type", "string")

    # Handle array types like ["string", "null"]
    if isinstance(schema_type, list):
        non_null = [t for t in schema_type if t != "null"]
        if non_null:
            return _SCHEMA_TYPE_MAP.get(non_null[0], str)
        return str

    return _SCHEMA_TYPE_MAP.get(schema_type, str)


def _schema_default(prop: dict) -> Any:
    """Get a default value for an optional JSON Schema property."""
    if "default" in prop:
        return prop["default"]
    return None
