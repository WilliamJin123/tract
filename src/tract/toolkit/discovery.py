"""Discovery profile: 3 meta-tools for progressive capability discovery.

Reduces 29 individual tools to 3 that let an LLM agent discover and use
all tract capabilities on-demand:

- tract_help(topic?) — progressive disclosure of available operations
- tract_do(action, params?) — single execution surface for all operations
- tract_inspect(what?) — unified state inspection dashboard

This implements the --help progressive discovery pattern: the agent starts
with a high-level overview and drills down only when needed, keeping context
usage minimal.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from tract.toolkit.compact import COMPACT_DOMAINS
from tract.toolkit.models import ToolDefinition

if TYPE_CHECKING:
    from tract.tract import Tract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Domain groupings with descriptions — derived from COMPACT_DOMAINS
# ---------------------------------------------------------------------------
DISCOVERY_DOMAINS: dict[str, dict] = {
    "context": {
        "description": "Read/write context: commit, compile, status, log, diff, inspect commits, compress, garbage collect",
        "actions": COMPACT_DOMAINS["context"],
    },
    "branch": {
        "description": "Parallel lines of work: create, switch, merge, reset, checkout, list, transition stages",
        "actions": COMPACT_DOMAINS["branch"],
    },
    "annotate": {
        "description": "Annotate commits with priority or set standing directives",
        "actions": COMPACT_DOMAINS["annotate"],
    },
    "tag": {
        "description": "Tag commits for search and organization",
        "actions": COMPACT_DOMAINS["tag"],
    },
    "config": {
        "description": "Configure behavior: key-value settings, LLM model, metadata",
        "actions": COMPACT_DOMAINS["config"],
    },
    "middleware": {
        "description": "Event middleware: create or remove pre/post hooks",
        "actions": COMPACT_DOMAINS["middleware"],
    },
}

# Reverse lookup: action -> domain
_ACTION_TO_DOMAIN: dict[str, str] = {
    action: domain
    for domain, info in DISCOVERY_DOMAINS.items()
    for action in info["actions"]
}

# All known action names
_ALL_ACTIONS: list[str] = list(_ACTION_TO_DOMAIN.keys())


def get_discovery_tools(tract: Tract) -> list[ToolDefinition]:
    """Build the 3 discovery meta-tools bound to a Tract instance.

    Args:
        tract: The Tract instance to bind tool handlers to.

    Returns:
        List of 3 ToolDefinition objects (help, do, inspect).
    """
    from tract.toolkit.definitions import get_all_tools

    all_tools = get_all_tools(tract)
    tool_lookup: dict[str, ToolDefinition] = {t.name: t for t in all_tools}

    return [
        _build_help_tool(tool_lookup),
        _build_do_tool(tool_lookup),
        _build_inspect_tool(tract, tool_lookup),
    ]


# ---------------------------------------------------------------------------
# tract_help — progressive disclosure
# ---------------------------------------------------------------------------

def _build_help_tool(tool_lookup: dict[str, ToolDefinition]) -> ToolDefinition:
    """Create the help meta-tool for progressive capability discovery."""

    def handler(topic: str | None = None) -> str:
        if topic is None:
            # Level 1: high-level overview of all domains
            lines = ["Tract capabilities:"]
            for domain, info in DISCOVERY_DOMAINS.items():
                lines.append(f"  {domain}: {info['description']}")
            lines.append("")
            lines.append(
                "Use tract_help(topic=<domain>) for actions in a domain, "
                "or tract_help(topic=<action>) for full parameter details."
            )
            return "\n".join(lines)

        # Check if topic is a domain name
        if topic in DISCOVERY_DOMAINS:
            # Level 2: list actions in domain with descriptions
            info = DISCOVERY_DOMAINS[topic]
            lines = [f"Domain '{topic}' — {info['description']}:", ""]
            for action in info["actions"]:
                tool = tool_lookup.get(action)
                desc = tool.description if tool else "(not available)"
                lines.append(f"  {action}: {desc}")
            lines.append("")
            lines.append(
                "Use tract_help(topic=<action>) for full parameter schema."
            )
            return "\n".join(lines)

        # Check if topic is an action name
        if topic in _ACTION_TO_DOMAIN:
            tool = tool_lookup.get(topic)
            if tool is None:
                return f"Action '{topic}' exists but is not available."
            # Level 3: full parameter schema
            return json.dumps(
                {
                    "action": topic,
                    "domain": _ACTION_TO_DOMAIN[topic],
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
                indent=2,
            )

        # Unknown topic — error as navigation
        return (
            f"Unknown topic '{topic}'. "
            f"Available domains: {list(DISCOVERY_DOMAINS.keys())}. "
            f"Available actions: {_ALL_ACTIONS}"
        )

    return ToolDefinition(
        name="tract_help",
        description=(
            "Discover tract capabilities. No args: overview of all domains. "
            "topic=<domain>: list actions. topic=<action>: full parameter schema. "
            "Start here to learn what you can do."
        ),
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": (
                        "A domain name (context, branch, annotate, tag, config, "
                        "middleware) to list its actions, or an action name "
                        "(commit, compile, etc.) to get its parameter schema."
                    ),
                },
            },
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# tract_do — single execution surface
# ---------------------------------------------------------------------------

def _build_do_tool(tool_lookup: dict[str, ToolDefinition]) -> ToolDefinition:
    """Create the do meta-tool for executing any tract operation."""

    def handler(action: str, params: dict | None = None) -> str:
        if params is None:
            params = {}

        tool = tool_lookup.get(action)
        if tool is None:
            # Error as navigation: list available actions
            return (
                f"Unknown action '{action}'. Available actions: {_ALL_ACTIONS}. "
                f"Use tract_help(topic=<action>) for parameter details."
            )

        try:
            result = tool.handler(**params)
            return str(result)
        except TypeError as exc:
            # Missing/wrong params — show the schema as navigation
            logger.debug(
                "discovery do '%s' param error: %s", action, exc, exc_info=True,
            )
            schema = json.dumps(tool.parameters, indent=2)
            return (
                f"Parameter error: {exc}\n\n"
                f"Expected parameters for '{action}':\n{schema}"
            )
        except Exception as exc:
            logger.debug(
                "discovery do '%s' failed: %s", action, exc, exc_info=True,
            )
            hint = getattr(exc, "hint", "")
            error_msg = f"Error: {type(exc).__name__}: {exc}"
            if hint:
                error_msg += f"\n[hint] {hint}"
            return error_msg

    return ToolDefinition(
        name="tract_do",
        description=(
            "Execute any tract operation. action: the operation name "
            "(e.g. 'commit', 'branch', 'compile'). params: keyword arguments "
            "for that operation. Use tract_help to discover available actions "
            "and their parameters."
        ),
        parameters={
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The operation to perform (e.g. 'commit', 'branch', 'compile').",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Keyword arguments for the action. "
                        "Use tract_help(topic=<action>) to see required parameters."
                    ),
                },
            },
            "required": ["action"],
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# tract_inspect — state inspection dashboard
# ---------------------------------------------------------------------------

def _build_inspect_tool(
    tract: Tract,
    tool_lookup: dict[str, ToolDefinition],
) -> ToolDefinition:
    """Create the inspect meta-tool for unified state inspection."""

    def handler(what: str | None = None) -> str:
        try:
            if what is None:
                return _inspect_dashboard(tract)
            if what == "branches":
                return _inspect_branches(tract)
            if what in ("history", "log"):
                return _inspect_log(tract)
            if what == "config":
                return _inspect_config(tract)
            if what == "tags":
                return _inspect_tags(tract)
            if what == "directives":
                return _inspect_directives(tract)
            return (
                f"Unknown inspection target '{what}'. "
                f"Available: branches, history, log, config, tags, directives. "
                f"Or omit for a dashboard overview."
            )
        except Exception as exc:
            logger.debug("inspect '%s' failed: %s", what, exc, exc_info=True)
            return f"Error: {type(exc).__name__}: {exc}"

    return ToolDefinition(
        name="tract_inspect",
        description=(
            "Inspect current state. No args: dashboard overview with branch, HEAD, "
            "tokens, recent commits, and more. what='branches': all branches. "
            "what='history': commit log. what='config': branch config. "
            "what='tags': tag registry. what='directives': active directives."
        ),
        parameters={
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "enum": ["branches", "history", "log", "config", "tags", "directives"],
                    "description": "What to inspect. Omit for a combined dashboard.",
                },
            },
        },
        handler=handler,
    )


# ---------------------------------------------------------------------------
# Inspect sub-handlers
# ---------------------------------------------------------------------------

def _inspect_dashboard(tract: Tract) -> str:
    """Combined dashboard: the most useful state at a glance."""
    status = tract.status()
    branches = tract.list_branches()
    all_recent = tract.log(limit=50)
    recent = all_recent[:5]

    lines = ["=== Tract Dashboard ==="]

    # Branch and HEAD
    branch = status.branch_name or "(detached)"
    head = status.head_hash[:8] if status.head_hash else "None"
    lines.append(f"Branch: {branch}")
    lines.append(f"HEAD: {head}")

    # Tokens / budget
    if status.token_budget_max and status.token_budget_max > 0:
        pct = (status.token_count / status.token_budget_max) * 100
        lines.append(
            f"Tokens: {status.token_count}/{status.token_budget_max} ({pct:.0f}%)"
        )
    else:
        lines.append(f"Tokens: {status.token_count}")

    lines.append(f"Commits: {status.commit_count}")

    # Branch count
    lines.append(f"Branches: {len(branches)}")

    # Directive count (from the 50 we already fetched)
    directive_count = sum(
        1 for c in all_recent if c.content_type == "instruction"
    )
    lines.append(f"Active directives: {directive_count}")

    # Recent commits
    if recent:
        lines.append("")
        lines.append("Recent commits:")
        for entry in recent:
            msg = entry.message or ""
            short = entry.commit_hash[:8]
            pri = entry.effective_priority or "normal"
            pri_tag = f" [{pri}]" if pri != "normal" else ""
            lines.append(
                f"  {short} {entry.operation.value:6s} "
                f"{entry.content_type:12s} {entry.token_count:5d}t{pri_tag}  {msg}"
            )
    else:
        lines.append("")
        lines.append("No commits yet.")

    return "\n".join(lines)


def _inspect_branches(tract: Tract) -> str:
    """List all branches with HEAD info."""
    branches = tract.list_branches()
    if not branches:
        return "No branches."
    lines = ["Branches:"]
    for b in branches:
        marker = "* " if b.is_current else "  "
        lines.append(f"{marker}{b.name} ({b.commit_hash[:8]})")
    return "\n".join(lines)


def _inspect_log(tract: Tract) -> str:
    """Recent commit log (last 10)."""
    entries = tract.log(limit=10)
    if not entries:
        return "No commits found."
    lines = ["Commit log (last 10):"]
    for entry in entries:
        msg = entry.message or ""
        short = entry.commit_hash[:8]
        pri = entry.effective_priority or "normal"
        pri_tag = f" [{pri}]" if pri != "normal" else ""
        lines.append(
            f"  {short} {entry.operation.value:6s} "
            f"{entry.content_type:12s} {entry.token_count:5d}t{pri_tag}  {msg}"
        )
    return "\n".join(lines)


def _inspect_config(tract: Tract) -> str:
    """Current branch configuration."""
    configs = tract.get_all_configs()
    if not configs:
        return "No configuration set on current branch."
    lines = ["Branch config:"]
    for key, value in sorted(configs.items()):
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _inspect_tags(tract: Tract) -> str:
    """Registered tags with counts."""
    tags = tract.list_tags()
    if not tags:
        return "No tags registered."
    lines = ["Registered tags:"]
    for t in tags:
        auto = " (auto)" if t.get("auto_created") else ""
        lines.append(f"  {t['name']}: {t.get('count', 0)} uses{auto}")
    return "\n".join(lines)


def _inspect_directives(tract: Tract) -> str:
    """Active directives (instruction-type commits)."""
    entries = tract.log(limit=50)
    directives = [e for e in entries if e.content_type == "instruction"]
    if not directives:
        return "No active directives."
    lines = ["Active directives:"]
    for d in directives:
        name = d.message or d.commit_hash[:8]
        pri = d.effective_priority or "normal"
        lines.append(f"  {name} [{pri}] ({d.token_count}t)")
    return "\n".join(lines)
