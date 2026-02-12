"""tract checkout -- switch to a commit or branch."""

from __future__ import annotations

import click

from tract.cli.formatting import format_error, get_console


@click.command()
@click.argument("target")
@click.pass_context
def checkout(ctx: click.Context, target: str) -> None:
    """Checkout a commit or branch.

    TARGET can be a branch name, commit hash, hash prefix, or "-" to
    return to the previous position.

    Checking out a branch attaches HEAD (commits go to that branch).
    Checking out a commit detaches HEAD (read-only inspection).
    """
    from tract.cli import _get_tract

    console = get_console()
    try:
        t = _get_tract(ctx)
        try:
            resolved = t.checkout(target)
            is_detached = t.is_detached
            if is_detached:
                console.print(
                    f"HEAD detached at [yellow]{resolved[:8]}[/yellow]"
                )
            else:
                branch = t.current_branch
                console.print(
                    f"Switched to branch [green]{branch}[/green] "
                    f"([yellow]{resolved[:8]}[/yellow])"
                )
        finally:
            t.close()
    except SystemExit:
        raise
    except Exception as e:
        format_error(str(e), console)
        raise SystemExit(1) from None
