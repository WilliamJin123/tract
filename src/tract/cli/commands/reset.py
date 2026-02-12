"""tract reset -- move HEAD to a target commit."""

from __future__ import annotations

import click

from tract.cli.formatting import format_error, get_console


@click.command()
@click.argument("target")
@click.option("--soft", "mode", flag_value="soft", default=True, help="Soft reset (default).")
@click.option("--hard", "mode", flag_value="hard", help="Hard reset (same as soft in Trace; requires --force).")
@click.option("--force", is_flag=True, help="Required for --hard reset.")
@click.pass_context
def reset(ctx: click.Context, target: str, mode: str, force: bool) -> None:
    """Reset HEAD to TARGET commit.

    TARGET can be a commit hash, hash prefix (min 4 chars), or branch name.
    In Trace, soft and hard resets behave identically (no working tree).
    Hard reset requires --force as a safety guard.
    """
    from tract.cli import _get_tract

    console = get_console()

    # Force guard for hard reset
    if mode == "hard" and not force:
        format_error("Hard reset requires --force flag.", console)
        raise SystemExit(1)

    try:
        t = _get_tract(ctx)
        try:
            resolved = t.reset(target, mode=mode)
            console.print(f"HEAD is now at [yellow]{resolved[:8]}[/yellow] ({mode} reset)")
        finally:
            t.close()
    except SystemExit:
        raise
    except Exception as e:
        format_error(str(e), console)
        raise SystemExit(1) from None
