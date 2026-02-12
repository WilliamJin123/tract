"""tract log -- show commit history."""

from __future__ import annotations

import click

from tract.cli.formatting import format_error, format_log_compact, format_log_verbose, get_console


@click.command()
@click.option("-n", "--limit", default=20, type=int, help="Maximum number of commits to show.")
@click.option("-v", "--verbose", is_flag=True, help="Show verbose commit details.")
@click.option("--op", "op_filter", default=None, type=click.Choice(["append", "edit"], case_sensitive=False), help="Filter by operation type.")
@click.pass_context
def log(ctx: click.Context, limit: int, verbose: bool, op_filter: str | None) -> None:
    """Show commit history from HEAD backward."""
    from tract.cli import _get_tract
    from tract.models.commit import CommitOperation

    console = get_console()
    try:
        t = _get_tract(ctx)
        try:
            op = CommitOperation(op_filter) if op_filter else None
            entries = t.log(limit=limit, op_filter=op)
            if verbose:
                format_log_verbose(entries, console)
            else:
                format_log_compact(entries, console)
        finally:
            t.close()
    except SystemExit:
        raise
    except Exception as e:
        format_error(str(e), console)
        raise SystemExit(1) from None
