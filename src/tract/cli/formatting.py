"""Rich formatting helpers for the Tract CLI.

Provides functions that format SDK data structures for terminal display.
Rich auto-detects TTY and degrades gracefully when piped (no ANSI codes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from tract.models.commit import CommitInfo
    from tract.operations.diff import DiffResult
    from tract.operations.history import StatusInfo


def get_console() -> Console:
    """Create a Rich Console that auto-detects TTY for graceful pipe degradation."""
    return Console(stderr=False)


def format_log_compact(entries: list[CommitInfo], console: Console) -> None:
    """Display commit log in compact table format."""
    if not entries:
        console.print("[dim]No commits.[/dim]")
        return

    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("Hash", style="yellow", width=8)
    table.add_column("Time", style="dim")
    table.add_column("Op", style="cyan", width=6)
    table.add_column("Tokens", justify="right", style="green")
    table.add_column("Message")

    for entry in entries:
        time_str = entry.created_at.strftime("%Y-%m-%d %H:%M")
        msg = escape(entry.message) if entry.message else ""
        table.add_row(
            entry.commit_hash[:8],
            time_str,
            entry.operation.value,
            str(entry.token_count),
            msg,
        )

    console.print(table)


def format_log_verbose(entries: list[CommitInfo], console: Console) -> None:
    """Display commit log in verbose format with full details."""
    if not entries:
        console.print("[dim]No commits.[/dim]")
        return

    for i, entry in enumerate(entries):
        if i > 0:
            console.print()

        console.print(f"[yellow]commit {entry.commit_hash}[/yellow]")
        console.print(f"  Operation: [cyan]{entry.operation.value}[/cyan]")
        console.print(f"  Type:      {escape(entry.content_type)}")
        console.print(f"  Tokens:    [green]{entry.token_count}[/green]")
        console.print(f"  Date:      {entry.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

        if entry.parent_hash:
            console.print(f"  Parent:    {entry.parent_hash[:8]}")
        if entry.response_to:
            console.print(f"  Edits:     {entry.response_to[:8]}")
        if entry.message:
            console.print(f"  Message:   {escape(entry.message)}")
        if entry.generation_config:
            config_parts = [f"{k}={v}" for k, v in entry.generation_config.items()]
            console.print(f"  Config:    {escape(', '.join(config_parts))}")


def format_status(info: StatusInfo, console: Console) -> None:
    """Display tract status information."""
    # HEAD position
    if info.head_hash is None:
        console.print("[dim]No commits yet.[/dim]")
        return

    if info.is_detached:
        console.print(f"HEAD detached at [yellow]{info.head_hash[:8]}[/yellow]")
    else:
        console.print(
            f"On branch [green]{escape(info.branch_name)}[/green]  "
            f"([yellow]{info.head_hash[:8]}[/yellow])"
        )

    # Commit count
    console.print(f"  Commits: {info.commit_count}")

    # Token budget bar
    if info.token_budget_max is not None:
        pct = info.token_count / info.token_budget_max if info.token_budget_max > 0 else 0
        bar_width = 30
        filled = int(pct * bar_width)
        filled = min(filled, bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)

        if pct > 0.9:
            color = "red"
        elif pct > 0.7:
            color = "yellow"
        else:
            color = "green"

        console.print(
            f"  Tokens:  [{color}]{info.token_count}[/{color}] / {info.token_budget_max} "
            f"[{color}][{bar}][/{color}] {pct:.0%}"
        )
    else:
        console.print(f"  Tokens:  {info.token_count} (no budget set)")

    # Token source
    if info.token_source:
        console.print(f"  Source:  [dim]{info.token_source}[/dim]")

    # Recent commits
    if info.recent_commits:
        console.print()
        console.print("[bold]Recent commits:[/bold]")
        for entry in info.recent_commits:
            time_str = entry.created_at.strftime("%H:%M:%S")
            msg = escape(entry.message) if entry.message else ""
            op = entry.operation.value
            console.print(
                f"  [yellow]{entry.commit_hash[:8]}[/yellow] "
                f"[dim]{time_str}[/dim] "
                f"[cyan]{op}[/cyan] "
                f"{msg}"
            )


def _format_stat_summary(
    stat: "DiffStat",
    generation_config_changes: dict,
    console: Console,
) -> None:
    """Print the stat summary block (shared between full and stat-only diff)."""
    from tract.operations.diff import DiffStat  # noqa: F811

    console.print(
        f"[green]+{stat.messages_added}[/green] added  "
        f"[red]-{stat.messages_removed}[/red] removed  "
        f"[yellow]~{stat.messages_modified}[/yellow] modified  "
        f"[dim]={stat.messages_unchanged} unchanged[/dim]"
    )
    if stat.total_token_delta != 0:
        delta_sign = "+" if stat.total_token_delta > 0 else ""
        console.print(f"Token delta: {delta_sign}{stat.total_token_delta}")
    if generation_config_changes:
        console.print()
        console.print("[bold]Config changes:[/bold]")
        for field_name, (old_val, new_val) in generation_config_changes.items():
            console.print(f"  {field_name}: {old_val} -> {new_val}")


def format_diff(result: DiffResult, console: Console, stat_only: bool = False) -> None:
    """Display diff results with colors.

    Args:
        result: DiffResult from Tract.diff()
        console: Rich console
        stat_only: If True, show only stat summary (like git diff --stat)
    """
    if stat_only:
        _format_stat_summary(result.stat, result.generation_config_changes, console)
        return

    # Full diff display
    console.print(
        f"diff [yellow]{result.commit_a[:8]}[/yellow] "
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
            console.print(
                f"[yellow]~~~ message [{md.index}] "
                f"role={md.role_a} -> {md.role_b}[/yellow]"
            )
            for line in md.content_diff_lines:
                line_stripped = line.rstrip("\n")
                if line_stripped.startswith("+"):
                    console.print(Text(line_stripped, style="green"))
                elif line_stripped.startswith("-"):
                    console.print(Text(line_stripped, style="red"))
                elif line_stripped.startswith("@@"):
                    console.print(Text(line_stripped, style="cyan"))
                else:
                    console.print(line_stripped)

    # Summary
    console.print()
    _format_stat_summary(result.stat, result.generation_config_changes, console)


def format_error(message: str, console: Console) -> None:
    """Display an error message."""
    console.print(f"[red]Error:[/red] {message}", highlight=False)
