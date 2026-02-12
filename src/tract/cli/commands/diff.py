"""tract diff -- compare two commits."""

from __future__ import annotations

import click

from tract.cli.formatting import format_diff, format_error, get_console


@click.command()
@click.argument("commit_a", required=False, default=None)
@click.argument("commit_b", required=False, default=None)
@click.option("--stat", "stat_only", is_flag=True, help="Show only summary statistics.")
@click.pass_context
def diff(ctx: click.Context, commit_a: str | None, commit_b: str | None, stat_only: bool) -> None:
    """Show differences between two commits.

    With no arguments, diffs HEAD against its parent.
    With one argument, diffs that commit against its parent.
    With two arguments, diffs COMMIT_A against COMMIT_B.
    """
    from tract.cli import _get_tract

    console = get_console()
    try:
        t = _get_tract(ctx)
        try:
            result = t.diff(commit_a=commit_a, commit_b=commit_b)
            format_diff(result, console, stat_only=stat_only)
        finally:
            t.close()
    except SystemExit:
        raise
    except Exception as e:
        format_error(str(e), console)
        raise SystemExit(1) from None
