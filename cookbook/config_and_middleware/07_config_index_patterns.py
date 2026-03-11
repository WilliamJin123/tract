"""Config Index Patterns -- DAG-based configuration resolution.

Shows how ConfigIndex resolves per-key settings from the commit DAG,
with branch isolation, invalidation, middleware queries, and
stage-driven overrides matching consumer workflow patterns.

Patterns shown:
  1. Config precedence      -- closer-to-HEAD config values win
  2. Branch-isolated configs -- branches resolve independently
  3. Config invalidation     -- new commits mark the index stale
  4. Middleware config query  -- read configs dynamically in handlers
  5. Per-stage configuration  -- consumer workflow pattern with stages
  6. Config unset semantics   -- None values clear a key

Demonstrates: t.configure(), t.get_config(), t.get_all_configs(),
              t.config_index, ConfigIndex.invalidate(), t.use(),
              branch/switch, config + stage workflow

No LLM required.
"""

from tract import Tract


def config_precedence():
    """Closer-to-HEAD config values win over older ones."""
    with Tract.open() as t:
        t.configure(model="gpt-4", temperature=0.7)
        t.user("Hello")
        t.assistant("Hi there")
        t.configure(temperature=0.2)  # Override just temperature

        assert t.get_config("model") == "gpt-4"         # inherited
        assert t.get_config("temperature") == 0.2        # overridden
        assert t.get_config("missing_key") is None       # unset returns None
        assert t.get_config("missing_key", "fallback") == "fallback"

        # get_all_configs() returns only the resolved key-value pairs
        all_cfg = t.get_all_configs()
        assert all_cfg["model"] == "gpt-4"
        assert all_cfg["temperature"] == 0.2
        assert "missing_key" not in all_cfg

        print("1. Config precedence: PASSED")


def branch_isolated_configs():
    """Configs on different branches resolve independently."""
    with Tract.open() as t:
        # Set base config on main
        t.configure(model="gpt-4", temperature=0.7)

        # Create and switch to experiment branch
        t.branch("experiment")
        t.switch("experiment")
        t.configure(model="claude-3", temperature=0.0)

        # Experiment branch sees its own overrides
        assert t.get_config("model") == "claude-3"
        assert t.get_config("temperature") == 0.0

        # Switch back to main -- main still has original values
        t.switch("main")
        assert t.get_config("model") == "gpt-4"
        assert t.get_config("temperature") == 0.7

        # Create another branch from main (inherits main's config)
        t.branch("feature")
        t.switch("feature")
        assert t.get_config("model") == "gpt-4"       # inherited from main
        assert t.get_config("temperature") == 0.7      # inherited from main

        # Override just one key on the feature branch
        t.configure(max_tokens=2048)
        assert t.get_config("model") == "gpt-4"        # still inherited
        assert t.get_config("max_tokens") == 2048       # branch-specific

        print("2. Branch-isolated configs: PASSED")


def config_invalidation():
    """New commits mark the config index as stale and trigger rebuild."""
    with Tract.open() as t:
        t.configure(model="gpt-4")

        # Force the index to build by querying
        assert t.get_config("model") == "gpt-4"
        idx = t.config_index
        assert not idx.is_stale

        # A new commit (any kind) invalidates the cached index.
        # configure() explicitly calls invalidate().
        t.configure(temperature=0.5)

        # The property rebuilds on next access
        assert t.get_config("model") == "gpt-4"
        assert t.get_config("temperature") == 0.5

        # Non-config commits do not invalidate the index,
        # but configure() always does
        fresh_idx = t.config_index
        assert not fresh_idx.is_stale

        t.configure(model="gpt-4o")   # Triggers invalidation
        assert t.config_index.is_stale or t.get_config("model") == "gpt-4o"
        # After query, it rebuilds
        assert t.get_config("model") == "gpt-4o"

        print("3. Config invalidation: PASSED")


def middleware_config_query():
    """Read config values dynamically inside middleware handlers.

    Note: configure() invalidates the config index *after* commit() returns,
    so a post_commit handler firing inside configure() may still see the
    previous config index state. Subsequent commits will see the update.
    """
    with Tract.open() as t:
        # Pre-configure model limits
        t.configure(model="gpt-4", max_tokens=4096)

        config_snapshots = []

        def capture_config_on_commit(ctx):
            """Post-commit handler that reads current config."""
            model = ctx.tract.get_config("model")
            max_tokens = ctx.tract.get_config("max_tokens")
            config_snapshots.append({
                "model": model,
                "max_tokens": max_tokens,
                "branch": ctx.branch,
            })

        mid_id = t.use("post_commit", capture_config_on_commit)

        # These commits fire the handler, which reads config
        t.user("Hello, world!")
        t.assistant("Hi there!")

        # Override config mid-conversation
        t.configure(model="gpt-4o")

        # After configure() returns, the index is invalidated.
        # The next commit will trigger a rebuild and see the new value.
        t.user("Updated model!")

        # First two commits saw model="gpt-4"
        assert config_snapshots[0]["model"] == "gpt-4"
        assert config_snapshots[1]["model"] == "gpt-4"
        # The user commit after configure() sees the updated model
        assert config_snapshots[-1]["model"] == "gpt-4o"

        t.remove_middleware(mid_id)
        print("4. Middleware config query: PASSED")


def per_stage_config():
    """Consumer workflow pattern: different configs for each stage.

    This mirrors the coding/research/ecommerce consumer workflows where
    each stage (e.g., design -> implementation -> review) needs different
    LLM settings (temperature, model, token budget).
    """
    with Tract.open() as t:
        # --- Base project config ---
        t.configure(model="gpt-4", max_tokens=4096)
        t.system("You are a coding assistant.")

        # --- Design stage: high temperature for brainstorming ---
        t.transition("design")
        t.configure(temperature=0.8, compile_strategy="full")

        assert t.get_config("temperature") == 0.8
        assert t.get_config("model") == "gpt-4"           # inherited
        assert t.get_config("compile_strategy") == "full"

        t.user("Design the API structure.")
        t.assistant("Here is the proposed API design...")

        # --- Implementation stage: low temperature for precision ---
        t.transition("implementation")
        t.configure(temperature=0.1, max_tokens=8192)

        assert t.get_config("temperature") == 0.1
        assert t.get_config("max_tokens") == 8192
        assert t.get_config("model") == "gpt-4"           # still inherited

        t.user("Implement the authentication module.")
        t.assistant("Here is the auth module code...")

        # --- Review stage: moderate temperature ---
        t.transition("review")
        t.configure(temperature=0.3, compile_strategy="messages", compile_strategy_k=10)

        assert t.get_config("temperature") == 0.3
        assert t.get_config("compile_strategy") == "messages"
        assert t.get_config("compile_strategy_k") == 10

        # All configs remain accessible
        all_cfg = t.get_all_configs()
        assert "model" in all_cfg
        assert "temperature" in all_cfg
        assert "compile_strategy" in all_cfg

        print("5. Per-stage configuration: PASSED")


def config_unset_semantics():
    """None values effectively unset a config key.

    The ConfigIndex treats None values as 'not set', so setting a key
    to None hides it from get_config() and get_all_configs().
    """
    with Tract.open() as t:
        # Set initial config
        t.configure(model="gpt-4", temperature=0.5)
        assert t.get_config("model") == "gpt-4"
        assert t.get_config("temperature") == 0.5

        # "Unset" temperature by configuring it to None
        t.configure(temperature=None)
        assert t.get_config("temperature") is None          # returns None (unset)
        assert t.get_config("temperature", 0.7) == 0.7      # default kicks in
        assert t.get_config("model") == "gpt-4"             # unaffected

        # get_all_configs() excludes None values
        all_cfg = t.get_all_configs()
        assert "temperature" not in all_cfg
        assert all_cfg["model"] == "gpt-4"

        # Custom (non-well-known) keys work the same way
        t.configure(custom_key="hello")
        assert t.get_config("custom_key") == "hello"
        t.configure(custom_key=None)
        assert t.get_config("custom_key") is None

        print("6. Config unset semantics: PASSED")


def main():
    config_precedence()
    branch_isolated_configs()
    config_invalidation()
    middleware_config_query()
    per_stage_config()
    config_unset_semantics()
    print("\nAll config index patterns: PASSED")


# Alias for pytest discovery
test_config_index_patterns = main


if __name__ == "__main__":
    main()
