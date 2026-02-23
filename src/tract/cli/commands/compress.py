"""tract compress -- compress commit chains into summaries."""

from __future__ import annotations

import click


@click.command()
@click.option("--target-tokens", type=int, default=None, help="Target token count for the summary.")
@click.option("--from", "from_commit", default=None, help="Start of range (inclusive).")
@click.option("--to", "to_commit", default=None, help="End of range (inclusive).")
@click.option("--content", "manual_content", default=None, help="Manual summary text (bypasses LLM).")
@click.option("--instructions", default=None, help="Extra guidance appended to the summarization prompt.")
@click.option("--edit", "do_edit", is_flag=True, help="Review and edit summaries in $EDITOR before committing.")
@click.option("--preserve", multiple=True, help="Commit hashes to treat as PINNED (repeatable).")
@click.pass_context
def compress(
    ctx: click.Context,
    target_tokens: int | None,
    from_commit: str | None,
    to_commit: str | None,
    manual_content: str | None,
    instructions: str | None,
    do_edit: bool,
    preserve: tuple[str, ...],
) -> None:
    """Compress commit chains into summaries.

    Uses LLM summarization by default. Pass --content for manual mode.
    Use --edit to review and edit each summary in $EDITOR before committing.
    """
    from tract.cli import _tract_session
    from tract.cli.formatting import format_compress_result

    with _tract_session(ctx) as (t, console):
        kwargs: dict = {}
        if target_tokens is not None:
            kwargs["target_tokens"] = target_tokens
        if from_commit is not None:
            kwargs["from_commit"] = from_commit
        if to_commit is not None:
            kwargs["to_commit"] = to_commit
        if manual_content is not None:
            kwargs["content"] = manual_content
        if instructions is not None:
            kwargs["instructions"] = instructions
        if preserve:
            kwargs["preserve"] = list(preserve)

        if do_edit:
            # Collaborative mode: LLM drafts, user edits in $EDITOR
            kwargs["auto_commit"] = False
            pending = t.compress(**kwargs)

            console.print(
                f"[bold]Pending compression:[/bold] "
                f"{len(pending.summaries)} draft(s), "
                f"{pending.original_tokens} -> ~{pending.estimated_tokens} tokens"
            )

            for i, summary in enumerate(pending.summaries):
                console.print(f"\n[bold]Opening summary [{i}] in editor...[/bold]")
                edited = click.edit(summary)
                if edited is not None and edited.strip() != summary.strip():
                    pending.edit_summary(i, edited.strip())
                    console.print(f"[green]Summary [{i}] updated.[/green]")
                else:
                    console.print(f"[dim]Summary [{i}] kept as-is.[/dim]")

            if not click.confirm("\nApprove and commit?", default=True):
                console.print("[yellow]Cancelled.[/yellow] Nothing was committed.")
                return

            result = pending.approve()
        else:
            # Auto-commit mode
            result = t.compress(**kwargs)

        format_compress_result(result, console)
