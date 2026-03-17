"""Config and Precedence

t.config.set() commits key-value settings into the DAG.  Well-known keys are
type-checked; unknown keys pass through for custom use.

Key concepts:
  - Each set() call commits one or more key-value pairs
  - DAG precedence: closest to HEAD wins when the same key appears multiple times
  - Branch isolation: each branch resolves config independently
  - Invalidation: the cached config index rebuilds lazily after mutations
  - Unset semantics: setting a key to None removes it from resolution
  - Compile strategies: config drives how the context window is built

Well-known config keys:
  model, temperature, max_tokens, max_commit_tokens,
  auto_compress_threshold, compact_tools, compile_strategy,
  compile_strategy_k, handoff_summary_k

Demonstrates: t.config.set(), t.config.get(), t.config.get_all(),
              t.config_index, branch isolation, compile strategies

No LLM required.
"""

from tract import Tract, MiddlewareContext


# -- 1. Basics: set, get, defaults -----------------------------------------
def config_basics():
    """Set and retrieve config values, including defaults for missing keys."""
    with Tract.open() as t:
        t.config.set(model="gpt-4", temperature=0.7, max_tokens=4096)

        assert t.config.get("model") == "gpt-4"
        assert t.config.get("temperature") == 0.7
        assert t.config.get("max_tokens") == 4096

        # Missing keys return None, or an explicit default
        assert t.config.get("nonexistent") is None
        assert t.config.get("nonexistent", "fallback") == "fallback"

        # Complex values work too
        t.config.set(stop=["END", "DONE", "---"])
        assert t.config.get("stop") == ["END", "DONE", "---"]

        print("1. Config basics: PASSED")


# -- 2. DAG precedence: closer to HEAD wins ---------------------------------
def config_precedence():
    """When the same key is set more than once, the commit nearest HEAD wins."""
    with Tract.open() as t:
        t.config.set(model="gpt-4", temperature=0.7)
        t.user("Hello")
        t.assistant("Hi there")
        t.config.set(temperature=0.2)  # Override just temperature

        assert t.config.get("model") == "gpt-4"      # inherited (unchanged)
        assert t.config.get("temperature") == 0.2     # overridden

        all_cfg = t.config.get_all()
        assert all_cfg["model"] == "gpt-4"
        assert all_cfg["temperature"] == 0.2
        assert "missing_key" not in all_cfg

        print("2. DAG precedence: PASSED")


# -- 3. Branch isolation ----------------------------------------------------
def branch_isolated_configs():
    """Configs on different branches resolve independently."""
    with Tract.open() as t:
        t.config.set(model="gpt-4", temperature=0.7)

        # Experiment branch gets its own overrides
        t.branches.create("experiment")
        t.branches.switch("experiment")
        t.config.set(model="claude-3", temperature=0.0)

        assert t.config.get("model") == "claude-3"
        assert t.config.get("temperature") == 0.0

        # Main still has original values
        t.branches.switch("main")
        assert t.config.get("model") == "gpt-4"
        assert t.config.get("temperature") == 0.7

        # Feature branch inherits from main at fork point
        t.branches.create("feature")
        t.branches.switch("feature")
        assert t.config.get("model") == "gpt-4"       # inherited
        t.config.set(max_tokens=2048)                  # branch-specific addition
        assert t.config.get("max_tokens") == 2048

        print("3. Branch isolation: PASSED")


# -- 4. Invalidation -------------------------------------------------------
def config_invalidation():
    """New config commits invalidate the cached index; it rebuilds lazily."""
    with Tract.open() as t:
        t.config.set(model="gpt-4")
        assert t.config.get("model") == "gpt-4"

        idx = t.config_index
        assert not idx.is_stale

        # A new config.set() invalidates the cached index
        t.config.set(temperature=0.5)
        assert t.config.get("temperature") == 0.5      # triggers rebuild
        assert t.config.get("model") == "gpt-4"        # still present

        # Verify rebuild happened
        fresh_idx = t.config_index
        assert not fresh_idx.is_stale

        t.config.set(model="gpt-4o")
        assert t.config.get("model") == "gpt-4o"

        print("4. Invalidation: PASSED")


# -- 5. Unset semantics (None clears a key) ---------------------------------
def config_unset_semantics():
    """Setting a key to None hides it from get() and get_all()."""
    with Tract.open() as t:
        t.config.set(model="gpt-4", temperature=0.5)

        # "Unset" temperature
        t.config.set(temperature=None)
        assert t.config.get("temperature") is None
        assert t.config.get("temperature", 0.7) == 0.7  # default kicks in
        assert t.config.get("model") == "gpt-4"         # unaffected

        all_cfg = t.config.get_all()
        assert "temperature" not in all_cfg
        assert all_cfg["model"] == "gpt-4"

        # Custom (non-well-known) keys follow the same pattern
        t.config.set(custom_key="hello")
        assert t.config.get("custom_key") == "hello"
        t.config.set(custom_key=None)
        assert t.config.get("custom_key") is None

        print("5. Unset semantics: PASSED")


# -- 6. Middleware reading config -------------------------------------------
def middleware_config_query():
    """Read live config inside a middleware handler."""
    with Tract.open() as t:
        t.config.set(model="gpt-4", max_tokens=4096)

        snapshots = []

        def on_commit(ctx: MiddlewareContext):
            snapshots.append({
                "model": ctx.tract.config.get("model"),
                "max_tokens": ctx.tract.config.get("max_tokens"),
            })

        mid_id = t.middleware.add("post_commit", on_commit)

        t.user("Hello")
        t.assistant("Hi")

        # Override mid-conversation
        t.config.set(model="gpt-4o")
        t.user("Updated model")

        assert snapshots[0]["model"] == "gpt-4"
        assert snapshots[-1]["model"] == "gpt-4o"

        t.middleware.remove(mid_id)
        print("6. Middleware config query: PASSED")


# -- 7. Compile strategies -------------------------------------------------
def compile_strategies():
    """Config drives which compile strategy builds the context window.

    Strategies differ in *content detail*, not message count:
      full     -- every commit's full content
      messages -- lightweight commit-message summaries only
      adaptive -- last k full, rest lightweight
    """
    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        for i in range(8):
            t.user(f"Question {i + 1}: Tell me about topic {i + 1}.")
            t.assistant(f"Answer {i + 1}: Information about topic {i + 1}.")

        # Full strategy
        t.config.set(compile_strategy="full")
        ctx_full = t.compile(strategy=t.config.get("compile_strategy"))

        # Messages strategy (lightweight)
        t.config.set(compile_strategy="messages", compile_strategy_k=5)
        ctx_msg = t.compile(strategy=t.config.get("compile_strategy"))

        # Adaptive strategy (hybrid)
        t.config.set(compile_strategy="adaptive")
        k = t.config.get("compile_strategy_k")
        ctx_adapt = t.compile(
            strategy=t.config.get("compile_strategy"), strategy_k=k,
        )

        print(f"  full:     {len(ctx_full.messages)} msgs (full content)")
        print(f"  messages: {len(ctx_msg.messages)} msgs (commit summaries)")
        print(f"  adaptive: {len(ctx_adapt.messages)} msgs (last {k} full, rest light)")

        print("7. Compile strategies: PASSED")


# -- 8. Per-stage config (consumer workflow pattern) -------------------------
def per_stage_config():
    """Different stages get different LLM settings via config overrides."""
    with Tract.open() as t:
        t.config.set(model="gpt-4", max_tokens=4096)
        t.system("You are a coding assistant.")

        # Design: high temperature for brainstorming
        t.transition("design")
        t.config.set(temperature=0.8, compile_strategy="full")
        assert t.config.get("temperature") == 0.8
        assert t.config.get("model") == "gpt-4"           # inherited

        t.user("Design the API structure.")
        t.assistant("Here is the proposed API design...")

        # Implementation: low temperature for precision
        t.transition("implementation")
        t.config.set(temperature=0.1, max_tokens=8192)
        assert t.config.get("temperature") == 0.1
        assert t.config.get("max_tokens") == 8192

        # Review: moderate temperature, lightweight compile
        t.transition("review")
        t.config.set(temperature=0.3, compile_strategy="messages")
        assert t.config.get("temperature") == 0.3
        assert t.config.get("compile_strategy") == "messages"

        all_cfg = t.config.get_all()
        assert "model" in all_cfg
        assert "temperature" in all_cfg

        print("8. Per-stage config: PASSED")


# -- Runner -----------------------------------------------------------------
def main() -> None:
    config_basics()
    config_precedence()
    branch_isolated_configs()
    config_invalidation()
    config_unset_semantics()
    middleware_config_query()
    compile_strategies()
    per_stage_config()
    print("\nAll config and precedence patterns: PASSED")


# Alias for pytest discovery
test_config_and_precedence = main


if __name__ == "__main__":
    main()
