"""Pretty-print support for tract output objects.

Uses rich library for formatted terminal output.
All functions accept their target object and print to a rich Console.

To avoid circular imports, this module does NOT import domain models
at module level. Functions access object attributes dynamically.
"""
from __future__ import annotations

from io import StringIO
from typing import Any, Literal

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


def _is_estimate(token_source: str) -> bool:
    """Whether token_source indicates a tiktoken estimate (not API-reported)."""
    return not token_source or token_source.startswith("tiktoken:")


def pprint_chat_response(response: Any, *, abbreviate: bool = False, file: Any = None) -> None:
    """Pretty-print a ChatResponse.

    Args:
        response: A ChatResponse instance.
        abbreviate: If True, truncate long text. Default False (show full).
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    # User prompt (when available — set by chat(), not generate())
    prompt = getattr(response, "prompt", None)
    if prompt:
        prompt_text = prompt
        if abbreviate and len(prompt_text) > 200:
            prompt_text = prompt_text[:197] + "..."
        console.print(Panel(
            prompt_text,
            title="[bold]User[/bold]",
            border_style="blue",
        ))

    # Main text — rendered as Markdown (LLM responses are almost always markdown)
    text = response.text
    if abbreviate and len(text) > 200:
        text = text[:197] + "..."

    # Tool-calling responses: show function calls in magenta
    tool_calls = getattr(response, "tool_calls", None)
    has_tool_calls = bool(tool_calls)

    if has_tool_calls:
        body_parts: list[Any] = []
        if text:
            body_parts.append(Markdown(text))
            body_parts.append(Text(""))
        for tc in tool_calls:
            call_text = Text()
            call_text.append(f"{tc.name}", style="bold cyan")
            call_text.append("(", style="dim")
            arg_parts = [f"{k}={v!r}" for k, v in tc.arguments.items()]
            call_text.append(", ".join(arg_parts), style="white")
            call_text.append(")", style="dim")
            body_parts.append(call_text)
        body: Any = Group(*body_parts) if len(body_parts) > 1 else body_parts[0]
        panel_title = "[bold]Tool Call[/bold]"
        panel_border = "magenta"
    else:
        body = Markdown(text) if text else Text("(empty response)")
        panel_title = "[bold]Assistant[/bold]"
        panel_border = "green"

    # Footer metadata (usage + config) as plain Rich markup
    footer_parts_str: list[str] = []
    if response.usage is not None:
        u = response.usage
        footer_parts_str.append(
            f"[dim]{u.prompt_tokens} prompt + {u.completion_tokens} completion"
            f" = {u.total_tokens} tokens[/dim]"
        )
    if response.generation_config is not None:
        fields = response.generation_config.non_none_fields()
        if fields:
            parts = [f"{k}={v}" for k, v in fields.items()]
            footer_parts_str.append(f"[dim]config: {', '.join(parts)}[/dim]")

    # Combine markdown body + plain-text footer into a single panel
    if footer_parts_str:
        content = Group(body, Text(""), Text.from_markup("\n".join(footer_parts_str)))
    else:
        content = body

    panel = Panel(
        content,
        title=panel_title,
        border_style=panel_border,
    )
    console.print(panel)


def pprint_compiled_context(
    ctx: Any,
    *,
    abbreviate: bool = False,
    style: Literal["table", "chat", "compact"] = "table",
    file: Any = None,
) -> None:
    """Pretty-print a CompiledContext.

    Args:
        ctx: A CompiledContext instance.
        abbreviate: If True, truncate long content. Default False (show full).
        style: Display style — ``"table"`` (default) for a data table,
            ``"chat"`` for a chat transcript with panels per message,
            ``"compact"`` for a one-line-per-message summary.
        file: Optional file-like object for output (used in tests).
    """
    if style == "chat":
        _pprint_compiled_chat(ctx, abbreviate=abbreviate, file=file)
        return
    if style == "compact":
        _pprint_compiled_compact(ctx, abbreviate=abbreviate, file=file)
        return

    console = _make_console(file)

    table = Table(title="Compiled Context", show_lines=False)
    table.add_column("#", style="dim", width=4, justify="right")
    table.add_column("Role", width=10)
    table.add_column("Content", no_wrap=False)
    table.add_column("Tokens", justify="right", width=8)

    # Per-message token counts are always tiktoken estimates
    estimate = _is_estimate(ctx.token_source)

    for i, msg in enumerate(ctx.messages):
        # Tool-calling assistant messages: show function calls
        if msg.role == "assistant" and getattr(msg, "tool_calls", None):
            call_parts = []
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
                call_parts.append(f"{tc.name}({args})")
            display_text = "; ".join(call_parts)
            if abbreviate and len(display_text) > 80:
                display_text = display_text[:77] + "..."
            role_label = Text("tool call", style="bold magenta")
            cell: Any = Text(display_text, style="bold magenta")
        else:
            content = msg.content
            if abbreviate and len(content) > 80:
                content = content[:77] + "..."
            color = _ROLE_COLORS.get(msg.role, "white")
            role_label = Text(msg.role, style=f"bold {color}")
            # Role-appropriate content rendering:
            # - system: dim (configuration, not conversation)
            # - assistant: Markdown (preserves code blocks, headers, lists)
            # - user/tool/other: plain white text
            if msg.role == "system":
                cell = Text(content, style="italic white")
            elif msg.role == "assistant":
                cell = Markdown(content)
            else:
                cell = Text(content)

        if msg.token_count > 0:
            tokens_str = f"\u2248{msg.token_count}" if estimate else str(msg.token_count)
        else:
            tokens_str = ""
        table.add_row(str(i + 1), role_label, cell, tokens_str)

    console.print(table)

    # Footer summary
    prefix = "\u2248" if estimate else ""
    summary = Text()
    summary.append(f"  {prefix}{ctx.token_count}", style="bold")
    summary.append(f" tokens | ", style="dim")
    summary.append(f"{ctx.commit_count}", style="bold")
    summary.append(f" commits | ", style="dim")
    summary.append(f"source: {ctx.token_source}", style="dim")
    console.print(summary)


_ROLE_STYLES: dict[str, tuple[str, str]] = {
    "system": ("System", "yellow"),
    "user": ("User", "blue"),
    "assistant": ("Assistant", "green"),
    "tool": ("Tool Result", "magenta"),
}

_ROLE_COLORS: dict[str, str] = {
    "system": "yellow",
    "user": "blue",
    "assistant": "green",
    "tool": "magenta",
}


def _pprint_compiled_compact(ctx: Any, *, abbreviate: bool = False, file: Any = None) -> None:
    """Render a CompiledContext as a compact one-line-per-message summary."""
    console = _make_console(file)

    max_width = 60 if abbreviate else 80

    for msg in ctx.messages:
        content = (msg.content or "").replace("\n", " ").strip()
        # Filter internal merge commit metadata
        if content.startswith("{") and "message" in content:
            continue

        # Tool-calling assistant messages: show function calls
        if msg.role == "assistant" and getattr(msg, "tool_calls", None):
            color = "magenta"
            role_label = "tool call"
            call_parts = []
            for tc in msg.tool_calls:
                args = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
                call_parts.append(f"{tc.name}({args})")
            preview = "; ".join(call_parts)
            if len(preview) > max_width:
                preview = preview[:max_width - 3] + "..."
        else:
            color = _ROLE_COLORS.get(msg.role, "white")
            role_label = msg.role
            preview = content[:max_width - 3] + "..." if len(content) > max_width else content

        line = Text()
        line.append(f"  {role_label:10s}", style=f"bold {color}")
        line.append(" | ", style="dim")
        line.append(preview)
        console.print(line)

    # Footer
    estimate = _is_estimate(ctx.token_source)
    prefix = "\u2248" if estimate else ""
    footer = Text()
    footer.append(f"  {len(ctx.messages)} messages", style="bold")
    footer.append(f" | {prefix}{ctx.token_count} tokens", style="dim")
    console.print(footer)


def _pprint_compiled_chat(ctx: Any, *, abbreviate: bool = False, file: Any = None) -> None:
    """Render a CompiledContext as a chat transcript with panels per message."""
    console = _make_console(file)

    for msg in ctx.messages:
        content = msg.content
        if abbreviate and len(content) > 200:
            content = content[:197] + "..."

        title, border = _ROLE_STYLES.get(msg.role, (msg.role.title(), "white"))

        # Tool-calling assistant messages: show calls instead of "(empty)"
        if msg.role == "assistant" and msg.tool_calls:
            parts: list[Any] = []
            if content:
                parts.append(Markdown(content))
                parts.append(Text(""))
            for tc in msg.tool_calls:
                call_text = Text()
                call_text.append(f"{tc.name}", style="bold cyan")
                call_text.append("(", style="dim")
                # Show arguments concisely: key=value pairs
                arg_parts = [f"{k}={v!r}" for k, v in tc.arguments.items()]
                call_text.append(", ".join(arg_parts), style="white")
                call_text.append(")", style="dim")
                parts.append(call_text)
            body: Any = Group(*parts) if len(parts) > 1 else parts[0]
            title = "Tool Call"
            border = "magenta"
        elif msg.role == "assistant":
            body = Markdown(content) if content else Text("(empty)")
        else:
            body = content

        console.print(Panel(body, title=f"[bold]{title}[/bold]", border_style=border))

    # Footer
    estimate = _is_estimate(ctx.token_source)
    prefix = "\u2248" if estimate else ""
    summary = Text()
    summary.append(f"  {len(ctx.messages)} messages", style="bold")
    summary.append(f" | ", style="dim")
    summary.append(f"{prefix}{ctx.token_count} tokens", style="bold")
    summary.append(f" | ", style="dim")
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


def pprint_diff_result(result: Any, *, stat_only: bool = False, file: Any = None) -> None:
    """Pretty-print a DiffResult with VS Code-style inline word highlights.

    Modified lines show the specific changed words highlighted against a
    dimmer background — like VS Code / Claude Code inline diffs.

    Args:
        result: A DiffResult instance.
        stat_only: If True, show only summary statistics.
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    stat = result.stat

    if stat_only:
        _pprint_diff_stat(stat, result.generation_config_changes, console)
        return

    # Header
    console.print(
        f"[bold]diff[/bold] [yellow]{result.commit_a[:8]}[/yellow] "
        f"[yellow]{result.commit_b[:8]}[/yellow]"
    )
    console.print()

    for md in result.message_diffs:
        if md.status == "unchanged":
            continue

        if md.status == "added":
            console.print(f"[green]+++ message [{md.index}] role={md.role_b}[/green]")
        elif md.status == "removed":
            console.print(f"[red]--- message [{md.index}] role={md.role_a}[/red]")
        elif md.status == "modified":
            if md.role_a == md.role_b:
                role_str = f"role={md.role_a}"
            else:
                role_str = f"role={md.role_a} → {md.role_b}"
            console.print(f"[yellow]~~~ message [{md.index}] {role_str}[/yellow]")
            _render_inline_diff(md.content_diff_lines, console)

    console.print()
    _pprint_diff_stat(stat, result.generation_config_changes, console)


def _render_inline_diff(diff_lines: list[str], console: Console) -> None:
    """Render unified diff lines with word-level inline highlights.

    Pairs consecutive -/+ lines and highlights the specific changed words.
    Unpaired lines are shown as full red/green.
    """
    import difflib

    # Collect lines, stripping trailing newlines
    lines = [l.rstrip("\n") for l in diff_lines]

    # Group into removed/added pairs for word-level diffing
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("@@"):
            console.print(Text(line, style="cyan"))
            i += 1
        elif line.startswith("---") or line.startswith("+++"):
            # File headers from unified diff — skip (we show our own header)
            i += 1
        elif line.startswith("-"):
            # Collect consecutive - lines, then consecutive + lines
            removed: list[str] = []
            while i < len(lines) and lines[i].startswith("-"):
                removed.append(lines[i][1:])  # strip leading -
                i += 1
            added: list[str] = []
            while i < len(lines) and lines[i].startswith("+"):
                added.append(lines[i][1:])  # strip leading +
                i += 1

            # Pair them up for word-level highlighting
            paired = min(len(removed), len(added))
            for k in range(paired):
                old_text, new_text = removed[k], added[k]
                old_rich, new_rich = _word_level_highlight(old_text, new_text)
                console.print(old_rich)
                console.print(new_rich)

            # Unpaired remainder
            for k in range(paired, len(removed)):
                console.print(Text(f"- {removed[k]}", style="red"))
            for k in range(paired, len(added)):
                console.print(Text(f"+ {added[k]}", style="green"))

        elif line.startswith("+"):
            console.print(Text(f"+ {line[1:]}", style="green"))
            i += 1
        elif line.startswith(" "):
            console.print(Text(f"  {line[1:]}", style="white"))
            i += 1
        else:
            console.print(line)
            i += 1


def _word_level_highlight(old_text: str, new_text: str) -> tuple[Text, Text]:
    """Produce two Rich Text lines with word-level change highlights.

    Unchanged words are dim red/green; changed words are bright bold.
    """
    import difflib

    old_words = old_text.split()
    new_words = new_text.split()

    matcher = difflib.SequenceMatcher(None, old_words, new_words)

    old_rich = Text("- ", style="red")
    new_rich = Text("+ ", style="green")

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            old_rich.append(" ".join(old_words[i1:i2]) + " ", style="red")
            new_rich.append(" ".join(new_words[j1:j2]) + " ", style="green")
        elif tag == "replace":
            old_rich.append(" ".join(old_words[i1:i2]) + " ", style="bold red on #3d0000")
            new_rich.append(" ".join(new_words[j1:j2]) + " ", style="bold green on #003d00")
        elif tag == "delete":
            old_rich.append(" ".join(old_words[i1:i2]) + " ", style="bold red on #3d0000")
        elif tag == "insert":
            new_rich.append(" ".join(new_words[j1:j2]) + " ", style="bold green on #003d00")

    return old_rich, new_rich


def _pprint_diff_stat(stat: Any, config_changes: dict, console: Console) -> None:
    """Print diff summary statistics."""
    parts: list[str] = []
    if stat.messages_added:
        parts.append(f"[green]+{stat.messages_added} added[/green]")
    if stat.messages_removed:
        parts.append(f"[red]-{stat.messages_removed} removed[/red]")
    if stat.messages_modified:
        parts.append(f"[yellow]~{stat.messages_modified} modified[/yellow]")
    if stat.messages_unchanged:
        parts.append(f"[dim]{stat.messages_unchanged} unchanged[/dim]")

    if stat.total_token_delta:
        sign = "+" if stat.total_token_delta > 0 else ""
        parts.append(f"[bold]{sign}{stat.total_token_delta} tokens[/bold]")

    if config_changes:
        for field_name, (old, new) in config_changes.items():
            parts.append(f"[cyan]{field_name}: {old} → {new}[/cyan]")

    console.print("  ".join(parts) if parts else "[dim]No changes[/dim]")


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
    estimate = _is_estimate(status.token_source)
    prefix = "\u2248" if estimate else ""
    if status.token_budget_max:
        pct = status.token_count / status.token_budget_max * 100
        body_parts.append(
            f"[bold]Tokens:[/bold]  {prefix}{status.token_count}/{status.token_budget_max} ({pct:.0f}%)"
        )
    else:
        body_parts.append(f"[bold]Tokens:[/bold]  {prefix}{status.token_count}")

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


# ---------------------------------------------------------------------------
# Merge / conflict display
# ---------------------------------------------------------------------------


def pprint_conflict_info(conflict: Any, *, file: Any = None) -> None:
    """Pretty-print a ConflictInfo as a git-style conflict diff.

    Shows both sides with ``<<<<<<<`` / ``=======`` / ``>>>>>>>`` markers,
    colored red (side A / target branch) and green (side B / source branch).

    Args:
        conflict: A ConflictInfo instance.
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    # Header
    target_short = conflict.target_hash[:8] if conflict.target_hash else "???"
    ctype = conflict.conflict_type
    if hasattr(ctype, "value"):
        ctype = ctype.value
    header = Text()
    header.append("CONFLICT ", style="bold red")
    header.append(f"({ctype}) ", style="dim")
    header.append(f"on {target_short}", style="bold")
    console.print(header)

    # Ancestor (if available)
    if conflict.ancestor_content_text:
        console.print(Text(f"  ||||||| ancestor", style="dim"))
        console.print(Text(f"  {conflict.ancestor_content_text}", style="dim"))

    # Side A (target / current branch)
    label_a = "current"
    if conflict.commit_a and hasattr(conflict.commit_a, "commit_hash"):
        label_a = conflict.commit_a.commit_hash[:8]
    console.print(Text(f"  <<<<<<< {label_a}", style="bold red"))
    for line in (conflict.content_a_text or "").splitlines():
        rich_line = Text(f"  {line}", style="red")
        console.print(rich_line)

    console.print(Text("  =======", style="bold"))

    # Side B (source branch)
    label_b = "incoming"
    if conflict.commit_b and hasattr(conflict.commit_b, "commit_hash"):
        label_b = conflict.commit_b.commit_hash[:8]
    for line in (conflict.content_b_text or "").splitlines():
        rich_line = Text(f"  {line}", style="green")
        console.print(rich_line)
    console.print(Text(f"  >>>>>>> {label_b}", style="bold green"))


def pprint_merge_result(result: Any, *, file: Any = None) -> None:
    """Pretty-print a MergeResult summary.

    Shows merge type, status, conflict count, and resolutions. For conflict
    merges, renders each conflict with :func:`pprint_conflict_info`.

    Args:
        result: A MergeResult instance.
        file: Optional file-like object for output (used in tests).
    """
    console = _make_console(file)

    # Type + status line
    mtype = result.merge_type
    status_style = "green" if result.committed else "yellow"
    status_text = "committed" if result.committed else "pending"

    header = Text()
    header.append("Merge ", style="bold")
    header.append(f"{result.source_branch}", style="cyan")
    header.append(" -> ", style="dim")
    header.append(f"{result.target_branch}", style="cyan")
    console.print(Panel(header, border_style=status_style, expand=False))

    # Summary fields
    info = Text()
    info.append(f"  type:      ", style="dim")
    info.append(f"{mtype}\n", style="bold")
    info.append(f"  status:    ", style="dim")
    info.append(f"{status_text}\n", style=f"bold {status_style}")
    info.append(f"  conflicts: ", style="dim")
    conflict_count = len(result.conflicts)
    info.append(
        f"{conflict_count}\n",
        style="bold red" if conflict_count > 0 else "bold green",
    )
    if result.merge_commit_hash and result.merge_type != "fast_forward":
        info.append(f"  commit:    ", style="dim")
        info.append(f"{result.merge_commit_hash[:8]}\n", style="bold")
    console.print(info)

    # Show each conflict
    for conflict in result.conflicts:
        pprint_conflict_info(conflict, file=file)
        console.print()

    # Show resolutions
    if result.resolutions:
        console.print(Text("  Resolutions:", style="bold"))
        for target_hash, text in result.resolutions.items():
            line = Text()
            line.append(f"    {target_hash[:8]}", style="bold cyan")
            line.append(" -> ", style="dim")
            line.append(text)
            console.print(line)
