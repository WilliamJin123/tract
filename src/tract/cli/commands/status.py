"""tract status -- show current tract status."""

from __future__ import annotations

import click

from tract.cli.formatting import format_status, format_error, get_console


@click.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current HEAD position, branch, token usage, and recent commits."""
    from tract.cli import _get_tract

    console = get_console()
    try:
        t = _get_tract(ctx)
        try:
            info = t.status()
            format_status(info, console)
        finally:
            t.close()
    except SystemExit:
        raise
    except Exception as e:
        format_error(str(e), console)
        raise SystemExit(1) from None
