"""Config and Directives: Behavior as Data

Three clean primitives for controlling LLM behavior:

  1. t.configure()  -- commit key-value settings to the DAG
  2. t.directive()  -- commit named instructions (compiled, deduplicated)
  3. t.use()        -- register Python middleware for event hooks

Config and directives travel with the conversation (they are commits),
follow DAG precedence (closest to HEAD wins), and appear in the log.

Demonstrates: t.configure(), t.directive(), t.get_config(),
              t.get_all_configs(), t.use() basics

No LLM required.
"""

from tract import Tract, Priority, BlockedError


def main():
    with Tract.open() as t:

        # --- 1. Config: key-value settings as commits ---

        print("=== Config (t.configure) ===\n")

        t.configure(model="gpt-4o", temperature=0.7)
        t.configure(max_tokens=4096)

        print(f"  model:       {t.get_config('model')}")
        print(f"  temperature: {t.get_config('temperature')}")
        print(f"  max_tokens:  {t.get_config('max_tokens')}")
        print(f"  missing key: {t.get_config('nonexistent', 'default-val')}")

        # Resolve all active configs at once
        all_cfg = t.get_all_configs()
        print(f"  all configs: {all_cfg}")

        # --- 2. DAG precedence: closer to HEAD wins ---

        print("\n=== DAG Precedence ===\n")

        t.user("Hello, world!")
        t.assistant("Hi there!")

        # Override model -- closer to HEAD wins
        t.configure(model="claude-sonnet")

        print(f"  model (after override): {t.get_config('model')}")
        print(f"  temperature (unchanged): {t.get_config('temperature')}")

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

        def count_commits(ctx):
            commit_count["n"] += 1
            print(f"    [middleware] post_commit #{commit_count['n']}")

        handler_id = t.use("post_commit", count_commits)
        print(f"  Registered post_commit handler: {handler_id}")

        t.user("This commit triggers middleware.")
        t.assistant("So does this one.")

        print(f"  Commits counted: {commit_count['n']}")

        # Remove middleware when done
        t.remove_middleware(handler_id)
        print(f"  Middleware removed: {handler_id}")

        # This commit will NOT trigger the handler
        t.user("No middleware fires for this one.")
        print(f"  Commits counted (unchanged): {commit_count['n']}")

        # --- 5. Log: configs and directives are visible commits ---

        print("\n=== Log (configs and directives are commits) ===\n")
        for ci in t.log():
            print(f"  {ci.commit_hash[:8]}  {ci.content_type:14s}  {ci.message}")


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:       getting_started/01_quick_start.py
# Custom tools:      getting_started/03_custom_tools.py
# Config patterns:   config_and_middleware/01_config_and_strategy.py
