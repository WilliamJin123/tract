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
    help="Review and edit conflict resolutions in $EDITOR before committing.",
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
    Use --review to interactively resolve conflicts in $EDITOR.
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

                for i, conflict in enumerate(result.conflicts):
                    key = conflict.target_hash
                    if key is None:
                        continue

                    # Show existing resolution or marker text
                    if key in result.resolutions:
                        initial = result.resolutions[key]
                    else:
                        initial = conflict.to_marker_text()

                    console.print(
                        f"\n[bold]Opening conflict [{i}] in editor...[/bold]"
                    )
                    edited = click.edit(initial)

                    if edited is None:
                        console.print(f"[dim]Conflict [{i}] skipped (editor closed).[/dim]")
                        continue

                    from tract.models.merge import ConflictInfo

                    parsed = ConflictInfo.parse_conflict_markers(edited)
                    if parsed is None:
                        console.print(
                            f"[yellow]Conflict [{i}] still has markers — skipped.[/yellow]"
                        )
                        continue

                    result.set_resolution(key, parsed)
                    console.print(f"[green]Conflict [{i}] resolved.[/green]")

                # Check all conflicts resolved
                unresolved = [
                    c for c in result.conflicts
                    if c.target_hash and c.target_hash not in result.resolutions
                ]
                if unresolved:
                    console.print(
                        f"[yellow]{len(unresolved)} conflict(s) still unresolved. "
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
