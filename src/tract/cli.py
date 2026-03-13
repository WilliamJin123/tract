"""Thin argparse CLI wrapping the tract library.

All tract imports are lazy (inside command functions) to keep startup fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DB_PATH = ".tract/tract.db"
CURRENT_FILE = ".tract/current"
SPAWNED_DIR = ".tract/spawned"
PROMPTS_DIR = ".tract/prompts"


def _err(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def _read_current_id() -> str:
    p = Path(CURRENT_FILE)
    if not p.exists():
        _err("No tract initialized. Run 'tract init' first.")
    return p.read_text().strip()


def _open_tract() -> object:
    """Open a Tract instance from .tract/current. Returns a Tract."""
    from tract.tract import Tract

    tract_id = _read_current_id()
    if not Path(DB_PATH).exists():
        _err(f"Database not found at {DB_PATH}. Run 'tract init' first.")
    return Tract.open(DB_PATH, tract_id=tract_id)


def _flatten_compiled(compiled: object) -> str:
    """Flatten a CompiledContext to readable plain text."""
    if hasattr(compiled, "to_text"):
        return compiled.to_text()  # type: ignore[union-attr]
    lines = []
    for msg in compiled.messages:  # type: ignore[attr-defined]
        role = msg.role.upper()
        name_suffix = f" ({msg.name})" if msg.name else ""
        lines.append(f"[{role}{name_suffix}]:\n{msg.content}")
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace) -> None:
    """Create .tract/ directory structure and initialize a new tract."""
    from tract.session import Session

    db_path = Path(DB_PATH)
    current_path = Path(CURRENT_FILE)
    spawned_dir = Path(SPAWNED_DIR)

    # Derive the .tract/ root from the db path
    tract_dir = db_path.parent
    prompts_dir = tract_dir / "prompts"

    # Idempotent: create directories (parents=True for monkeypatch compatibility)
    tract_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    spawned_dir.mkdir(parents=True, exist_ok=True)

    if current_path.exists():
        tract_id = current_path.read_text().strip()
        print(f"Already initialized. Tract: {tract_id}", file=sys.stderr)
        return

    # Create session + main tract
    session = Session.open(str(db_path))
    tract = session.create_tract(display_name="main")
    tract_id = tract.tract_id

    # Seed with an initial commit
    tract.commit(
        {"content_type": "instruction", "text": "Tract initialized."},
        message="init",
    )

    # Write tract_id to current file
    current_path.write_text(tract_id)

    print(f"Initialized tract in .tract/", file=sys.stderr)
    print(f"  Tract ID: {tract_id}", file=sys.stderr)
    print(f"  Database: {db_path}", file=sys.stderr)


def cmd_log(args: argparse.Namespace) -> None:
    """Show commit history."""
    t = _open_tract()
    commits = t.log(limit=args.limit)  # type: ignore[attr-defined]

    if not commits:
        print("No commits.")
        return

    for c in commits:
        priority_tag = ""
        if c.effective_priority and c.effective_priority not in ("normal", None):
            priority_tag = f"  [{c.effective_priority.upper()}]"
        msg = c.message or ""
        print(
            f"{c.commit_hash[:8]}  {c.created_at}  {c.content_type:<12}  "
            f"{c.token_count:>5}tok  {msg}{priority_tag}"
        )


def cmd_status(args: argparse.Namespace) -> None:
    """Show current tract status."""
    t = _open_tract()
    tract_id = t.tract_id  # type: ignore[attr-defined]

    compiled = t.compile(strategy="full")  # type: ignore[attr-defined]
    branch = t.current_branch or "(default)"  # type: ignore[attr-defined]

    print(f"Tract:   {tract_id[:8]}")
    print(f"Branch:  {branch}")
    print(f"Tokens:  {compiled.token_count}")
    print(f"Commits: {compiled.commit_count}")

    # List spawned children if any
    spawned_path = Path(SPAWNED_DIR)
    if spawned_path.exists():
        children = list(spawned_path.iterdir())
        if children:
            print(f"\nSpawned children ({len(children)}):")
            for child in children:
                child_id = child.read_text().strip() if child.is_file() else child.name
                print(f"  {child.name}: {child_id[:8]}")


def cmd_compile(args: argparse.Namespace) -> None:
    """Compile context and output in the chosen format."""
    t = _open_tract()
    compiled = t.compile(strategy=args.strategy)  # type: ignore[attr-defined]

    fmt = args.format
    if fmt == "text":
        print(_flatten_compiled(compiled))
    elif fmt == "json":
        output = [{"role": m.role, "content": m.content} for m in compiled.messages]
        print(json.dumps(output, indent=2))
    elif fmt == "openai":
        print(json.dumps(compiled.to_openai(), indent=2))
    elif fmt == "anthropic":
        print(json.dumps(compiled.to_anthropic(), indent=2))


def cmd_show(args: argparse.Namespace) -> None:
    """Show details of a single commit."""
    t = _open_tract()

    try:
        full_hash = t.resolve_commit(args.hash)  # type: ignore[attr-defined]
    except Exception:
        _err(f"Commit not found: {args.hash}")

    commit = t.get_commit(full_hash)  # type: ignore[attr-defined]
    if commit is None:
        _err(f"Commit not found: {args.hash}")

    content = t.get_content(full_hash)  # type: ignore[attr-defined]
    content_text = content if isinstance(content, str) else json.dumps(content, indent=2)

    priority = commit.effective_priority or "normal"
    branch = "(default)"
    # Try to determine branch from current context
    if hasattr(t, "current_branch") and t.current_branch:  # type: ignore[attr-defined]
        branch = t.current_branch  # type: ignore[attr-defined]

    print(f"Commit:   {commit.commit_hash}")
    print(f"Type:     {commit.content_type}")
    print(f"Branch:   {branch}")
    print(f"Date:     {commit.created_at}")
    print(f"Message:  {commit.message or ''}")
    print(f"Tokens:   {commit.token_count}")
    print(f"Priority: {priority}")
    print()
    print(content_text)


def cmd_diff(args: argparse.Namespace) -> None:
    """Show diff between commits or branches."""
    t = _open_tract()
    ref = args.ref

    if ref and ".." in ref:
        parts = ref.split("..", 1)
        branch_a, branch_b = parts[0], parts[1]
        result = t.compare(branch_a=branch_a, branch_b=branch_b)  # type: ignore[attr-defined]
    else:
        result = t.diff()  # type: ignore[attr-defined]

    stat = result.stat
    tokens_added = max(0, stat.total_token_delta)
    tokens_removed = abs(min(0, stat.total_token_delta))
    print(
        f"+{tokens_added}/-{tokens_removed} tokens, "
        f"+{stat.messages_added}/-{stat.messages_removed} messages"
    )

    for md in result.message_diffs:
        if md.status == "unchanged":
            continue
        role = md.role_b or md.role_a or "?"
        content_preview = ""
        if md.content_diff_lines:
            # Show first non-header diff line as preview
            for line in md.content_diff_lines:
                if line.startswith(("+++", "---", "@@")):
                    continue
                content_preview = f"  {line[:80]}"
                break
        print(f"  [{md.status}] {role}{content_preview}")


def cmd_branches(args: argparse.Namespace) -> None:
    """List branches."""
    t = _open_tract()
    branches = t.list_branches()  # type: ignore[attr-defined]

    if not branches:
        print("No branches.")
        return

    for b in branches:
        marker = "* " if b.is_current else "  "
        print(f"{marker}{b.name}  {b.commit_hash[:8]}")


def cmd_config(args: argparse.Namespace) -> None:
    """Show all config key-value pairs."""
    t = _open_tract()
    configs = t.get_all_configs()  # type: ignore[attr-defined]

    if not configs:
        print("No configs set.")
        return

    for key, value in configs.items():
        print(f"{key}: {value}")


def cmd_search(args: argparse.Namespace) -> None:
    """Search commits by term."""
    from tract.session import Session

    if not Path(DB_PATH).exists():
        _err(f"Database not found at {DB_PATH}. Run 'tract init' first.")

    tract_id: str | None = None
    if not args.all_tracts:
        tract_id = _read_current_id()

    session = Session.open(DB_PATH)
    results = session.search(args.term, tract_id=tract_id)

    if not results:
        print("No matches.")
        return

    for c in results:
        priority_tag = ""
        if c.effective_priority and c.effective_priority not in ("normal", None):
            priority_tag = f"  [{c.effective_priority.upper()}]"
        msg = c.message or ""
        print(
            f"{c.tract_id[:8]}  {c.commit_hash[:8]}  {c.created_at}  "
            f"{c.content_type:<12}  {c.token_count:>5}tok  {msg}{priority_tag}"
        )


def cmd_compress(args: argparse.Namespace) -> None:
    """Compress commit chain with a manual summary."""
    t = _open_tract()

    kwargs: dict = {"content": args.content}
    if args.target_tokens is not None:
        kwargs["target_tokens"] = args.target_tokens

    result = t.compress(**kwargs)  # type: ignore[attr-defined]

    ratio = result.compression_ratio
    print(
        f"Compressed {result.original_tokens} -> {result.compressed_tokens} tokens "
        f"(ratio: {ratio:.0%})"
    )


# ---------------------------------------------------------------------------
# Parser setup
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tract", description="CLI for the tract library")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    sub = parser.add_subparsers(dest="command")

    # init
    sub.add_parser("init", help="Initialize a new .tract/ directory")

    # log
    p_log = sub.add_parser("log", help="Show commit history")
    p_log.add_argument("--limit", "-n", type=int, default=20, help="Max commits to show")

    # status
    sub.add_parser("status", help="Show current tract status")

    # compile
    p_compile = sub.add_parser("compile", help="Compile context")
    p_compile.add_argument(
        "--format", "-f",
        choices=["text", "json", "openai", "anthropic"],
        default="text",
        help="Output format (default: text)",
    )
    p_compile.add_argument(
        "--strategy", "-s",
        choices=["adaptive", "full", "messages"],
        default="full",
        help="Compile strategy (default: full)",
    )

    # show
    p_show = sub.add_parser("show", help="Show a single commit")
    p_show.add_argument("hash", help="Commit hash (or prefix)")

    # diff
    p_diff = sub.add_parser("diff", help="Show diff between commits or branches")
    p_diff.add_argument("ref", nargs="?", default=None, help="branch_a..branch_b or omit for HEAD diff")

    # branches
    sub.add_parser("branches", help="List branches")

    # config
    sub.add_parser("config", help="Show all config values")

    # search
    p_search = sub.add_parser("search", help="Search commits")
    p_search.add_argument("term", help="Search term")
    p_search.add_argument("--all-tracts", action="store_true", help="Search across all tracts")

    # compress
    p_compress = sub.add_parser("compress", help="Compress commit chain")
    p_compress.add_argument("--content", required=True, help="Summary content for compression")
    p_compress.add_argument("--target-tokens", type=int, default=None, help="Target token count")

    return parser


def _get_version() -> str:
    try:
        from tract._version import __version__
        return __version__
    except ImportError:
        return "unknown"


_COMMANDS = {
    "init": cmd_init,
    "log": cmd_log,
    "status": cmd_status,
    "compile": cmd_compile,
    "show": cmd_show,
    "diff": cmd_diff,
    "branches": cmd_branches,
    "config": cmd_config,
    "search": cmd_search,
    "compress": cmd_compress,
}


def main(argv: list[str] | None = None) -> None:
    """Entry point for the tract CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args)


if __name__ == "__main__":
    main()
