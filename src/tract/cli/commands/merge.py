"""tract merge -- merge a branch into the current branch."""

from __future__ import annotations

import click

from tract.cli.formatting import format_merge_result


@click.command()
@click.argument("source")
@click.option("--no-ff", is_flag=True, help="Always create a merge commit (no fast-forward).")
@click.option(
    "--strategy",
    type=click.Choice(["auto", "semantic"], case_sensitive=False),
    default="auto",
    help="Merge strategy (default: auto).",
)
@click.option(
    "--delete-branch/--no-delete-branch",
    default=None,
    help="Delete source branch after merge (default: from config).",
)
@click.pass_context
def merge(ctx: click.Context, source: str, no_ff: bool, strategy: str, delete_branch: bool | None) -> None:
    """Merge SOURCE branch into the current branch.

    SOURCE is the name of the branch to merge in.
    """
    from tract.cli import _tract_session
    from tract.exceptions import NothingToMergeError

    with _tract_session(ctx) as (t, console):
        try:
            result = t.merge(source, no_ff=no_ff, strategy=strategy, delete_branch=delete_branch)
            format_merge_result(result, console)
        except NothingToMergeError:
            console.print("Already up to date.")
