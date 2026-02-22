"""Pretty-print support for tract output objects.

Uses rich library for formatted terminal output.
All functions accept their target object and print to a rich Console.

To avoid circular imports, this module does NOT import domain models
at module level. Functions access object attributes dynamically.
"""
from __future__ import annotations

from io import StringIO
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Brighter markdown theme for dark terminals — replaces Rich's default
# magenta/cyan with white/bright_white tones.
_MARKDOWN_THEME = Theme({
    "markdown.h1": "bold bright_white underline",
    "markdown.h2": "bold bright_white",
    "markdown.h3": "bold bright_white",
    "markdown.h4": "bright_white italic",
    "markdown.h5": "bright_white",
    "markdown.code": "bold white on grey11",
    "markdown.code_block": "white on grey11",
    "markdown.block_quote": "bright_white italic",
    "markdown.list": "bright_white",
    "markdown.item.bullet": "bold bright_white",
    "markdown.item.number": "bold bright_white",
    "markdown.link": "bright_cyan underline",
    "markdown.link_url": "dim bright_cyan",
    "markdown.strong": "bold bright_white",
    "markdown.em": "italic bright_white",
    "markdown.table.border": "dim white",
    "markdown.table.header": "bold bright_white",
})


def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 on Windows to avoid cp1252 encoding errors."""
    import sys
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _make_console(file: Any = None) -> Console:
    """Create a Console, optionally writing to a file-like object."""
    if file is not None:
        return Console(file=file, force_terminal=True, width=100, theme=_MARKDOWN_THEME)
    _ensure_utf8_stdout()
    return Console(theme=_MARKDOWN_THEME)


def pprint_chat_response(response: Any, *, abbreviate: bool = False, file: Any = None) -> None:
    """Pretty-print a ChatResponse.

    Args:
        response: A ChatResponse instance.
        abbreviate: If True, truncate long text. Default False (show full).
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    # Main text — rendered as Markdown (LLM responses are almost always markdown)
    text = response.text
    if abbreviate and len(text) > 200:
        text = text[:197] + "..."
    body = Markdown(text) if text else Text("(empty response)")

    # Footer metadata (usage + config) as plain Rich markup
    footer_parts: list[str] = []
    if response.usage is not None:
        u = response.usage
        footer_parts.append(
            f"[dim]{u.prompt_tokens} prompt + {u.completion_tokens} completion"
            f" = {u.total_tokens} tokens[/dim]"
        )
    if response.generation_config is not None:
        fields = response.generation_config.non_none_fields()
        if fields:
            parts = [f"{k}={v}" for k, v in fields.items()]
            footer_parts.append(f"[dim]config: {', '.join(parts)}[/dim]")

    # Combine markdown body + plain-text footer into a single panel
    if footer_parts:
        content = Group(body, Text(""), Text.from_markup("\n".join(footer_parts)))
    else:
        content = body

    panel = Panel(
        content,
        title="[bold]Assistant[/bold]",
        border_style="green",
    )
    console.print(panel)


def pprint_compiled_context(ctx: Any, *, abbreviate: bool = False, file: Any = None) -> None:
    """Pretty-print a CompiledContext.

    Args:
        ctx: A CompiledContext instance.
        abbreviate: If True, truncate long content. Default False (show full).
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    table = Table(title="Compiled Context", show_lines=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Role", style="bold", width=10)
    table.add_column("Content", no_wrap=False)
    table.add_column("Tokens", justify="right", width=8)

    for i, msg in enumerate(ctx.messages):
        content = msg.content
        if abbreviate and len(content) > 80:
            content = content[:77] + "..."
        tokens_str = str(msg.token_count) if msg.token_count > 0 else ""
        table.add_row(str(i + 1), msg.role, Markdown(content), tokens_str)

    console.print(table)

    # Footer summary
    summary = Text()
    summary.append(f"  {ctx.token_count}", style="bold")
    summary.append(f" tokens | ", style="dim")
    summary.append(f"{ctx.commit_count}", style="bold")
    summary.append(f" commits | ", style="dim")
    summary.append(f"source: {ctx.token_source}", style="dim")
    console.print(summary)


def pprint_commit_info(
    info: Any,
    *,
    abbreviate: bool = False,
    content: str | None = None,
    file: Any = None,
) -> None:
    """Pretty-print a CommitInfo.

    Args:
        info: A CommitInfo instance.
        abbreviate: If True, truncate long messages/content. Default False.
        content: Full content text to display (loaded via ``Tract.get_content()``
            or passed by ``Tract.show()``).
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    body_parts: list[str] = []

    # Hash and operation
    body_parts.append(f"[bold]Hash:[/bold]      {info.commit_hash}")
    body_parts.append(f"[bold]Operation:[/bold] {info.operation.value}")
    if info.parent_hash:
        body_parts.append(f"[bold]Parent:[/bold]    {info.parent_hash}")

    # Content type
    body_parts.append(f"[bold]Type:[/bold]      {info.content_type}")

    # Message
    if info.message:
        msg = info.message
        if abbreviate and len(msg) > 120:
            msg = msg[:117] + "..."
        body_parts.append(f"[bold]Message:[/bold]   {msg}")

    # Token count and timestamp
    body_parts.append(f"[bold]Tokens:[/bold]    {info.token_count}")
    body_parts.append(f"[bold]Created:[/bold]   {info.created_at}")

    # Generation config
    if info.generation_config is not None:
        fields = info.generation_config.non_none_fields()
        if fields:
            parts = [f"{k}={v}" for k, v in fields.items()]
            body_parts.append(f"[bold]Config:[/bold]    {', '.join(parts)}")

    # Content (when provided via Tract.show())
    if content is not None:
        display_content = content
        if abbreviate and len(display_content) > 200:
            display_content = display_content[:197] + "..."
        body_parts.append("")
        body_parts.append(f"[bold]Content:[/bold]")
        body_parts.append(display_content)

    short_hash = info.commit_hash[:8]
    panel = Panel(
        "\n".join(body_parts),
        title=f"[bold]Commit {short_hash}[/bold]",
        border_style="blue",
    )
    console.print(panel)


def pprint_status_info(status: Any, *, abbreviate: bool = False, file: Any = None) -> None:
    """Pretty-print a StatusInfo.

    Args:
        status: A StatusInfo instance.
        abbreviate: If True, truncate long content. Default False (show full).
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    body_parts: list[str] = []

    # Branch and HEAD
    branch = status.branch_name or "[italic]detached[/italic]"
    head = status.head_hash[:8] if status.head_hash else "None"
    body_parts.append(f"[bold]Branch:[/bold]  {branch}")
    body_parts.append(f"[bold]HEAD:[/bold]    {head}")

    # Commit count
    body_parts.append(f"[bold]Commits:[/bold] {status.commit_count}")

    # Token count / budget
    if status.token_budget_max:
        pct = status.token_count / status.token_budget_max * 100
        body_parts.append(
            f"[bold]Tokens:[/bold]  {status.token_count}/{status.token_budget_max} ({pct:.0f}%)"
        )
    else:
        body_parts.append(f"[bold]Tokens:[/bold]  {status.token_count}")

    # Token source
    body_parts.append(f"[bold]Source:[/bold]  {status.token_source}")

    # Detached warning
    if status.is_detached:
        body_parts.append("\n[bold yellow]WARNING: HEAD is detached[/bold yellow]")

    panel = Panel(
        "\n".join(body_parts),
        title="[bold]Status[/bold]",
        border_style="cyan",
    )
    console.print(panel)
