"""Shared editor-launch utility for opening diffs in external editors."""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import tempfile


def open_in_editor(
    text_a: str,
    text_b: str,
    label_a: str,
    label_b: str,
    *,
    editor: str | None = None,
    prefix: str = "tract-diff-",
) -> None:
    """Write two text sides to temp files and open in an editor diff view.

    Supports VS Code (``code --diff``), any editor accepting two file
    arguments (``$EDITOR``), and raises if no editor is found.

    Detection order:
      1. *editor* argument
      2. ``$TRACT_DIFF_EDITOR``
      3. ``$EDITOR``
      4. Auto-detect ``code`` on ``$PATH``

    Temp files are cleaned up at process exit via :func:`atexit.register`.

    Args:
        text_a: Content for the left side.
        text_b: Content for the right side.
        label_a: Filename label for the left temp file (without extension).
        label_b: Filename label for the right temp file (without extension).
        editor: Explicit editor command.  *None* triggers auto-detection.
        prefix: Prefix for the temp directory name.
    """
    editor = editor or os.environ.get("TRACT_DIFF_EDITOR") or os.environ.get("EDITOR")

    # Auto-detect VS Code if no editor configured
    auto_detected = False
    if not editor:
        code_path = shutil.which("code")
        if code_path:
            editor = code_path
            auto_detected = True

    if not editor:
        raise EnvironmentError(
            "No editor found.  Set $EDITOR or $TRACT_DIFF_EDITOR, "
            "or install VS Code ('code' on PATH)."
        )

    # Write temp files (cleaned up at process exit)
    tmpdir = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, tmpdir, True)
    path_a = os.path.join(tmpdir, f"{label_a}.md")
    path_b = os.path.join(tmpdir, f"{label_b}.md")

    with open(path_a, "w", encoding="utf-8") as f:
        f.write(text_a)
    with open(path_b, "w", encoding="utf-8") as f:
        f.write(text_b)

    # Build command list.  Auto-detected paths may contain spaces
    # (e.g. "C:\Program Files\Microsoft VS Code\bin\code.CMD")
    # so keep them as a single element; env-var editors may contain
    # extra arguments (e.g. "code --wait") so split those.
    cmd_base = [editor] if auto_detected else editor.split()

    # Resolve bare command names to full paths (needed on Windows
    # where e.g. "code" is actually a .CMD wrapper).
    resolved = shutil.which(cmd_base[0])
    if resolved:
        cmd_base[0] = resolved

    prog = os.path.basename(cmd_base[0]).lower()

    if prog in ("code", "code.cmd", "code.exe"):
        cmd = [*cmd_base, "--diff", path_a, path_b]
    else:
        cmd = [*cmd_base, path_a, path_b]

    # On Windows, .cmd/.bat scripts require shell=True to execute
    use_shell = (
        os.name == "nt"
        and cmd[0].lower().endswith((".cmd", ".bat"))
    )
    subprocess.Popen(cmd, shell=use_shell)  # noqa: S603
