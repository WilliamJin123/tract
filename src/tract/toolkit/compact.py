"""Compact tool definitions that group 29 tools into 6 domain tools + discover.

Reduces JSON schema overhead from ~4K tokens to ~300-500 tokens while
preserving full functionality via action dispatch.  LLMs call a domain
tool (e.g. ``tract_context``) with an ``action`` enum and a ``params``
object; the handler dispatches to the matching individual tool handler.

The ``tract_discover`` meta-tool lets the LLM introspect parameter
schemas on demand rather than paying the token cost for all 29 schemas
upfront.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tract.toolkit.models import ToolDefinition

if TYPE_CHECKING:
    from tract.tract import Tract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain -> action names
# ---------------------------------------------------------------------------
COMPACT_DOMAINS: dict[str, list[str]] = {
    "context": [
        "commit",
        "compile",
        "status",
        "log",
        "diff",
        "get_commit",
        "compress",
        "gc",
    ],
    "branch": [
        "branch",
        "switch",
        "merge",
        "reset",
        "checkout",
        "list_branches",
        "transition",
    ],
    "annotate": [
        "annotate",
        "directive",
    ],
    "tag": [
        "tag",
        "untag",
        "query_by_tags",
        "register_tag",
        "get_tags",
        "list_tags",
    ],
    "config": [
        "configure",
        "configure_model",
        "get_config",
        "create_metadata",
    ],
    "middleware": [
        "create_middleware",
        "remove_middleware",
    ],
}

# Reverse lookup: action_name -> domain
ACTION_TO_DOMAIN: dict[str, str] = {
    action: domain
    for domain, actions in COMPACT_DOMAINS.items()
    for action in actions
}

# ---------------------------------------------------------------------------
# Domain tool descriptions (concise for minimal token footprint)
# ---------------------------------------------------------------------------
_DOMAIN_DESCRIPTIONS: dict[str, str] = {
    "context": (
        "Manage context: commit content, compile messages, check status/log, "
        "diff commits, get commit details, compress history, or run gc. "
        "Actions: commit, compile, status, log, diff, get_commit, compress, gc."
    ),
    "branch": (
        "Manage branches: create, switch, merge, reset, checkout, list, "
        "or transition between branches. "
        "Actions: branch, switch, merge, reset, checkout, list_branches, transition."
    ),
    "annotate": (
        "Annotate commits or set directives. "
        "Actions: annotate, directive."
    ),
    "tag": (
        "Manage tags on commits: add, remove, query, register, and list tags. "
        "Actions: tag, untag, query_by_tags, register_tag, get_tags, list_tags."
    ),
    "config": (
        "Manage configuration: set key-value config, configure LLM model, "
        "read config, or create metadata. "
        "Actions: configure, configure_model, get_config, create_metadata."
    ),
    "middleware": (
        "Manage middleware: create or remove event middleware handlers. "
        "Actions: create_middleware, remove_middleware."
    ),
}


def get_compact_tools(tract: Tract) -> list[ToolDefinition]:
    """Build compact domain tools + discover tool bound to a Tract instance.

    Each call returns fresh closures bound to ``tract``.  The individual
    tool lookup is built once per call and captured by the handlers.

    Args:
        tract: The Tract instance to bind tool handlers to.

    Returns:
        List of 7 ToolDefinition objects (6 domains + discover).
    """
    from tract.toolkit.definitions import get_all_tools

    # Build lookup of all individual tools (once)
    all_tools = get_all_tools(tract)
    tool_lookup: dict[str, ToolDefinition] = {t.name: t for t in all_tools}

    tools: list[ToolDefinition] = []

    # 6 domain tools
    for domain, actions in COMPACT_DOMAINS.items():
        tools.append(_build_domain_tool(domain, actions, tool_lookup))

    # Discover meta-tool
    tools.append(_build_discover_tool(tool_lookup))

    return tools


# ---------------------------------------------------------------------------
# Domain tool builder
# ---------------------------------------------------------------------------


def _build_domain_tool(
    domain: str,
    actions: list[str],
    tool_lookup: dict[str, ToolDefinition],
) -> ToolDefinition:
    """Create a single domain tool that dispatches action+params to individual tools."""

    def handler(action: str, params: dict | None = None) -> str:
        if params is None:
            params = {}
        if action not in actions:
            return f"Unknown action '{action}'. Available: {actions}"
        tool = tool_lookup.get(action)
        if tool is None:
            return f"Tool '{action}' not found."
        try:
            result = tool.handler(**params)
            return str(result)
        except Exception as exc:
            logger.debug(
                "compact %s.%s failed: %s", domain, action, exc, exc_info=True,
            )
            return f"Error: {type(exc).__name__}: {exc}"

    return ToolDefinition(
        name=f"tract_{domain}",
        description=_DOMAIN_DESCRIPTIONS[domain],
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": actions,
                    "description": "The action to perform.",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Action-specific parameters. "
                        "Use tract_discover to see parameters for each action."
                    ),
                },
            },
            "required": ["action"],
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Discover meta-tool builder
# ---------------------------------------------------------------------------

_DISCOVER_DESCRIPTION = (
    "Get detailed parameter info for any tract tool. "
    "Call with a domain name to list actions, or with "
    "domain + action to get the full parameter schema."
)


def _build_discover_tool(
    tool_lookup: dict[str, ToolDefinition],
) -> ToolDefinition:
    """Create the discover meta-tool that returns schemas on demand."""

    domain_names = list(COMPACT_DOMAINS.keys())

    def handler(domain: str, action: str | None = None) -> str:
        if domain not in COMPACT_DOMAINS:
            return (
                f"Unknown domain: '{domain}'. "
                f"Available: {domain_names}"
            )

        actions = COMPACT_DOMAINS[domain]

        if action is not None:
            # Return full parameter schema for a specific action
            if action not in actions:
                return (
                    f"Unknown action '{action}' in domain '{domain}'. "
                    f"Available: {actions}"
                )
            tool = tool_lookup.get(action)
            if tool is None:
                return f"Tool '{action}' not available in current profile."
            return json.dumps(
                {
                    "action": action,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
                indent=2,
            )

        # Return list of actions with brief descriptions
        result = []
        for act in actions:
            tool = tool_lookup.get(act)
            if tool is not None:
                result.append({"action": act, "description": tool.description})
            else:
                result.append(
                    {"action": act, "description": "(not available in current profile)"}
                )
        return json.dumps(result, indent=2)

    return ToolDefinition(
        name="tract_discover",
        description=_DISCOVER_DESCRIPTION,
        parameters={
            "type": "object",
            "properties": {
                "domain": {
                    "type": "string",
                    "enum": domain_names,
                    "description": "Domain to inspect.",
                },
                "action": {
                    "type": "string",
                    "description": (
                        "Optional: specific action within the domain "
                        "to get full parameter schema for."
                    ),
                },
            },
            "required": ["domain"],
        },
        handler=handler,
    )
