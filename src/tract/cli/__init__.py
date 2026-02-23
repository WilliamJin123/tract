"""Tract CLI -- terminal interface for Trace context version control.

This module is NEVER imported from tract/__init__.py.
It is only loaded via the ``tract`` entry point defined in pyproject.toml.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

try:
    import click
except ImportError:
    raise ImportError(
        "CLI dependencies not installed. Install with: pip install tract[cli]"
    ) from None

from tract.cli.formatting import format_error, get_console

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.console import Console

    from tract.tract import Tract


@click.group()
@click.option(
    "--db",
    default=".tract.db",
    envvar="TRACT_DB",
    help="Path to tract database.",
)
@click.option(
    "--tract-id",
    default=None,
    envvar="TRACT_ID",
    help="Tract ID (auto-discovered if omitted).",
)
@click.pass_context
def cli(ctx: click.Context, db: str, tract_id: str | None) -> None:
    """Tract: Git-like version control for LLM context windows."""
    ctx.ensure_object(dict)
    ctx.obj["db_path"] = db
    ctx.obj["tract_id"] = tract_id


def _get_tract(ctx: click.Context) -> "Tract":  # noqa: F821 (forward ref)
    """Open a Tract instance from Click context.

    Auto-discovers the tract_id if not provided via --tract-id.
    """
    from tract.tract import Tract

    db_path = ctx.obj["db_path"]
    tract_id = ctx.obj["tract_id"]

    if tract_id is None:
        tract_id = _discover_tract(db_path)

    return Tract.open(path=db_path, tract_id=tract_id)


@contextmanager
def _tract_session(ctx: click.Context) -> Iterator[tuple[Tract, Console]]:
    """Context manager that opens a Tract, yields (tract, console), and handles cleanup.

    Ensures the tract is closed on exit and formats exceptions as CLI errors.
    Commands with special exception handling can catch specific errors inside
    the ``with`` block before this context manager's generic handler runs.
    """
    console = get_console()
    try:
        t = _get_tract(ctx)
        try:
            yield t, console
        finally:
            t.close()
    except SystemExit:
        raise
    except Exception as e:
        format_error(str(e), console)
        raise SystemExit(1) from None


def _discover_tract(db_path: str) -> str:
    """Auto-discover the tract_id from a database.

    Looks for a single tract in the database.  If multiple exist,
    the user must specify --tract-id.
    """
    import os

    from sqlalchemy import text

    from tract.storage.engine import create_session_factory, create_trace_engine

    if not os.path.exists(db_path):
        console = get_console()
        format_error(f"Database not found: {db_path}", console)
        raise SystemExit(1)

    engine = create_trace_engine(db_path)
    try:
        session_factory = create_session_factory(engine)
        session = session_factory()

        try:
            # Query distinct tract_ids from refs table
            result = session.execute(text("SELECT DISTINCT tract_id FROM refs"))
            tract_ids = [row[0] for row in result]

            if len(tract_ids) == 0:
                console = get_console()
                format_error("No tracts found in database.", console)
                raise SystemExit(1)
            elif len(tract_ids) == 1:
                return tract_ids[0]
            else:
                console = get_console()
                format_error(
                    f"Multiple tracts found ({len(tract_ids)}). "
                    f"Use --tract-id to specify one.",
                    console,
                )
                raise SystemExit(1)
        finally:
            session.close()
    finally:
        engine.dispose()


# Register subcommands after cli group is defined
from tract.cli.commands.log import log  # noqa: E402
from tract.cli.commands.status import status  # noqa: E402
from tract.cli.commands.diff import diff  # noqa: E402
from tract.cli.commands.reset import reset  # noqa: E402
from tract.cli.commands.checkout import checkout  # noqa: E402
from tract.cli.commands.branch import branch  # noqa: E402
from tract.cli.commands.switch import switch  # noqa: E402
from tract.cli.commands.merge import merge  # noqa: E402
from tract.cli.commands.compress import compress  # noqa: E402

cli.add_command(log)
cli.add_command(status)
cli.add_command(diff)
cli.add_command(reset)
cli.add_command(checkout)
cli.add_command(branch)
cli.add_command(switch)
cli.add_command(merge)
cli.add_command(compress)
