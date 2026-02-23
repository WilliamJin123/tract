"""Rich formatting helpers for the Tract CLI.

Provides functions that format SDK data structures for terminal display.
Rich auto-detects TTY and degrades gracefully when piped (no ANSI codes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.markup import escape
from rich.table import Table

if TYPE_CHECKING:
    from tract.models.branch import BranchInfo
    from tract.models.commit import CommitInfo
    from tract.models.merge import MergeResult
    from tract.operations.diff import DiffResult


def format_short_hash(commit_hash: str) -> str:
    """Format a commit hash as a short yellow-highlighted string."""
    return f"[yellow]{commit_hash[:8]}[/yellow]"
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
        if entry.edit_target:
            console.print(f"  Edits:     {entry.edit_target[:8]}")
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


def format_diff(result: DiffResult, console: Console, stat_only: bool = False) -> None:
    """Display diff results with colors.

    Delegates to the SDK's canonical pprint_diff_result for consistent
    rendering between CLI and Python API.

    Args:
        result: DiffResult from Tract.diff()
        console: Rich console (unused — SDK creates its own)
        stat_only: If True, show only stat summary (like git diff --stat)
    """
    from tract.formatting import pprint_diff_result
    pprint_diff_result(result, stat_only=stat_only)


def format_branches(branches: list[BranchInfo], console: Console) -> None:
    """Display branch list with current branch highlighted.

    Mimics ``git branch`` output: ``*`` marker on the current branch (green/bold),
    other branches indented.
    """
    if not branches:
        console.print("[dim]No branches.[/dim]")
        return

    for b in branches:
        if b.is_current:
            console.print(
                f"[green bold]* {escape(b.name)}[/green bold]  "
                f"[yellow]{b.commit_hash[:8]}[/yellow]"
            )
        else:
            console.print(
                f"  {escape(b.name)}  "
                f"[yellow]{b.commit_hash[:8]}[/yellow]"
            )


def format_merge_result(result: MergeResult, console: Console) -> None:
    """Display merge result summary.

    Shows different output based on merge_type:
    - fast_forward: "Fast-forward: branch -> hash"
    - clean / semantic: "Merged source into target (merge commit: hash)"
    - conflict: List conflicts with types and targets
    """
    if result.merge_type == "fast_forward":
        head_hash = result.merge_commit_hash or ""
        console.print(
            f"[green]Fast-forward:[/green] {escape(result.source_branch)} "
            f"-> [yellow]{head_hash[:8] if head_hash else '(unknown)'}[/yellow]"
        )
    elif result.merge_type in ("clean", "semantic"):
        merge_hash = result.merge_commit_hash or ""
        console.print(
            f"[green]Merged[/green] {escape(result.source_branch)} into "
            f"{escape(result.target_branch)} "
            f"(merge commit: [yellow]{merge_hash[:8] if merge_hash else '(unknown)'}[/yellow])"
        )
    elif result.merge_type == "conflict":
        n = len(result.conflicts)
        console.print(
            f"[red]CONFLICT:[/red] {n} conflict(s) detected. "
            f"Use the SDK to resolve and commit."
        )
        for conflict in result.conflicts:
            target = conflict.target_hash or "(unknown)"
            console.print(
                f"  [red]-[/red] {conflict.conflict_type}: target "
                f"[yellow]{target[:8]}[/yellow]"
            )


def format_error(message: str, console: Console) -> None:
    """Display an error message."""
    console.print(f"[red]Error:[/red] {message}", highlight=False)
