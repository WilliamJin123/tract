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
@click.option(
    "--review",
    "do_review",
    is_flag=True,
    help="Interactively resolve conflicts before committing.",
)
@click.pass_context
def merge(
    ctx: click.Context,
    source: str,
    no_ff: bool,
    strategy: str,
    delete_branch: bool | None,
    do_review: bool,
) -> None:
    """Merge SOURCE branch into the current branch.

    SOURCE is the name of the branch to merge in.
    Use --review to interactively resolve conflicts with a quick-pick menu.
    """
    from tract.cli import _tract_session
    from tract.exceptions import NothingToMergeError
    from tract.hooks.merge import PendingMerge

    with _tract_session(ctx) as (t, console):
        try:
            if do_review:
                result = t.merge(
                    source,
                    no_ff=no_ff,
                    strategy=strategy,
                    delete_branch=delete_branch,
                    review=True,
                )

                if not isinstance(result, PendingMerge):
                    # No conflicts (fast-forward or clean) — display normally
                    format_merge_result(result, console)
                    return

                console.print(
                    f"[bold]Pending merge:[/bold] "
                    f"{len(result.conflicts)} conflict(s) to resolve"
                )

                result.edit_interactive()

                # Check all conflicts resolved
                unresolved = [
                    c for c in result.conflicts
                    if c.target_hash and c.target_hash not in result.resolutions
                ]
                if unresolved:
                    console.print(
                        f"\n[yellow]{len(unresolved)} conflict(s) still unresolved. "
                        f"Cannot commit.[/yellow]"
                    )
                    return

                if not click.confirm("\nApprove and commit merge?", default=True):
                    console.print("[yellow]Cancelled.[/yellow] Nothing was committed.")
                    return

                committed = result.approve()
                format_merge_result(committed, console)
            else:
                result = t.merge(
                    source, no_ff=no_ff, strategy=strategy, delete_branch=delete_branch
                )
                format_merge_result(result, console)
        except NothingToMergeError:
            console.print("Already up to date.")
