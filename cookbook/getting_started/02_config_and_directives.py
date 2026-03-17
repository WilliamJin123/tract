"""Config and Directives: Behavior as Data

Three clean primitives for controlling LLM behavior:

  1. t.config.set()  -- commit key-value settings to the DAG
  2. t.directive()  -- commit named instructions (compiled, deduplicated)
  3. t.middleware.add()        -- register Python middleware for event hooks

Config and directives travel with the conversation (they are commits),
follow DAG precedence (closest to HEAD wins), and appear in the log.

Directives can load text from markdown files via path=, with auto-
discovery from a .tract/prompts/ directory.

Demonstrates: t.config.set(), t.directive(), t.directive(path=),
              t.system(path=), t.config.get(), t.config.get_all(),
              t.middleware.add() basics, .tract/prompts/ auto-discovery

No LLM required.
"""

import shutil
import tempfile
import os
from pathlib import Path

from tract import Tract, Priority, BlockedError, MiddlewareContext
from tract.formatting import pprint_log


def main() -> None:
    with Tract.open() as t:

        # --- 1. Config: key-value settings as commits ---

        print("=== Config (t.configure) ===\n")

        t.config.set(model="gpt-4o", temperature=0.7)
        t.config.set(max_tokens=4096)

        print(f"  model:       {t.config.get('model')}")
        print(f"  temperature: {t.config.get('temperature')}")
        print(f"  max_tokens:  {t.config.get('max_tokens')}")
        print(f"  missing key: {t.config.get('nonexistent', 'default-val')}")

        # Resolve all active configs at once
        all_cfg = t.config.get_all()
        print(f"  all configs: {all_cfg}")

        # --- 2. DAG precedence: closer to HEAD wins ---

        print("\n=== DAG Precedence ===\n")

        t.user("Hello, world!")
        t.assistant("Hi there!")

        # Override model -- closer to HEAD wins
        t.config.set(model="claude-sonnet")

        print(f"  model (after override): {t.config.get('model')}")
        print(f"  temperature (unchanged): {t.config.get('temperature')}")

        # --- 3. Directives: named standing instructions ---

        print("\n=== Directives (t.directive) ===\n")

        t.directive(
            "tone",
            "Always respond in a professional, concise tone. "
            "Avoid filler words and unnecessary qualifiers.",
        )
        t.directive(
            "format",
            "Use markdown formatting for all responses. "
            "Include code blocks with language tags.",
        )

        print("  Directive 'tone' committed (PINNED by default)")
        print("  Directive 'format' committed (PINNED by default)")

        # Override by name: same name -> closest to HEAD wins
        t.directive(
            "tone",
            "Respond in a friendly, casual tone. Use analogies and humor.",
        )
        print("  Directive 'tone' overridden (casual instead of formal)")

        # --- 4. Middleware basics ---

        print("\n=== Middleware (t.use) ===\n")

        commit_count = {"n": 0}

        def count_commits(ctx: MiddlewareContext):
            commit_count["n"] += 1
            print(f"    [middleware] post_commit #{commit_count['n']}")

        handler_id = t.middleware.add("post_commit", count_commits)
        print(f"  Registered post_commit handler: {handler_id}")

        t.user("This commit triggers middleware.")
        t.assistant("So does this one.")

        print(f"  Commits counted: {commit_count['n']}")

        # Remove middleware when done
        t.middleware.remove(handler_id)
        print(f"  Middleware removed: {handler_id}")

        # This commit will NOT trigger the handler
        t.user("No middleware fires for this one.")
        print(f"  Commits counted (unchanged): {commit_count['n']}")

        # --- 5. Log: configs and directives are visible commits ---

        print("\n=== Log (configs and directives are commits) ===\n")
        pprint_log(t.search.log())

    # --- 6. File-backed directives and .tract/prompts/ auto-discovery ---

    print("\n=== File-Backed Directives (path=) ===\n")

    # Create a temp directory to simulate a project with .tract/prompts/
    project_dir = tempfile.mkdtemp()
    prompts_dir = Path(project_dir) / ".tract" / "prompts"
    prompts_dir.mkdir(parents=True)

    # Write prompt files
    (prompts_dir / "system.md").write_text(
        "You are a senior code reviewer.\n"
        "Focus on correctness, security, and maintainability.\n"
        "Flag any OWASP top-10 vulnerabilities.",
        encoding="utf-8",
    )
    (prompts_dir / "tone.md").write_text(
        "Be direct and constructive. Lead with what works,\n"
        "then address what needs improvement. No sugarcoating.",
        encoding="utf-8",
    )

    print(f"  Created .tract/prompts/ with 2 files")
    print(f"    system.md  -- code reviewer persona")
    print(f"    tone.md    -- review tone guidelines")

    # Auto-discovery: Tract.open() finds .tract/prompts/ automatically
    old_cwd = os.getcwd()
    os.chdir(project_dir)
    try:
        with Tract.open() as t:
            # Just use the filename -- auto-resolved from .tract/prompts/
            t.system(path="system.md")
            t.directive("tone", path="tone.md")

            compiled = t.compile()
            compiled.pprint(style="compact")

            print(f"\n  prompt_dir auto-discovered: {t._prompt_dir}")

        # Explicit override: point to a different directory
        custom_dir = Path(project_dir) / "my-prompts"
        custom_dir.mkdir()
        (custom_dir / "safety.md").write_text(
            "Never execute user-provided code without sandboxing.",
            encoding="utf-8",
        )

        print("\n  --- Explicit prompt_dir override ---\n")

        with Tract.open(prompt_dir=str(custom_dir)) as t:
            t.directive("safety", path="safety.md")

            compiled = t.compile()
            compiled.pprint(style="compact")

            print(f"  prompt_dir (explicit): {t._prompt_dir}")
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(project_dir)


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:       getting_started/01_quick_start.py
# Custom tools:      getting_started/03_custom_tools.py
# Config patterns:   config_and_middleware/01_config_and_precedence.py
