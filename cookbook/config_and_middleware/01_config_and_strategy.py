"""Config and Compile Strategy

t.configure() commits key-value settings to the DAG. Well-known keys are
type-checked; unknown keys pass through for custom use.

  - Each configure() call commits one or more key-value pairs
  - When multiple calls set the same key, closest to HEAD wins (DAG precedence)
  - Query with t.get_config() or t.get_all_configs()

The most common use: selecting the compile strategy that controls how
tract builds the LLM context window.

Well-known config keys:
  model, temperature, max_tokens, max_commit_tokens,
  auto_compress_threshold, compact_tools, compile_strategy,
  compile_strategy_k, handoff_summary_k

Demonstrates: t.configure(), t.get_config(), t.get_all_configs(),
              DAG precedence, compile_strategy, strategy comparison

No LLM required.
"""

from tract import Tract


def main():
    with Tract.open() as t:

        # --- Config as key-value store ---

        print("=== Config ===\n")

        t.configure(model="gpt-4o")
        t.configure(temperature=0.7)
        t.configure(max_tokens=4096)

        print(f"  model:       {t.get_config('model')}")
        print(f"  temperature: {t.get_config('temperature')}")
        print(f"  max_tokens:  {t.get_config('max_tokens')}")

        # --- DAG precedence: closer to HEAD wins ---

        print("\n=== DAG Precedence ===\n")

        t.user("Hello, world!")
        t.assistant("Hi there!")

        # Override model -- new configure() call is closer to HEAD
        t.configure(model="claude-sonnet")

        print(f"  model (overridden): {t.get_config('model')}")
        print(f"  temperature (unchanged): {t.get_config('temperature')}")
        print(f"  missing key:  {t.get_config('nonexistent')}")
        print(f"  with default: {t.get_config('nonexistent', 'fallback')}")

        # Complex values work too
        t.configure(stop=["END", "DONE", "---"])
        print(f"  stop (list): {t.get_config('stop')}")

        # --- Build conversation for strategy demos ---

        print("\n=== Building History ===\n")

        t.system("You are a helpful assistant.")
        for i in range(8):
            t.user(f"Question {i + 1}: Tell me about topic {i + 1}.")
            t.assistant(f"Answer {i + 1}: Here is information about topic {i + 1}.")

        print(f"  Total commits: {len(t.log())}")

        # --- Compile strategy: full ---

        print("\n=== Strategy: full ===\n")

        t.configure(compile_strategy="full")

        strategy = t.get_config("compile_strategy")
        ctx_full = t.compile(strategy=strategy)
        print(f"  {strategy}: {len(ctx_full.messages)} messages")

        # --- Compile strategy: messages (lightweight summaries) ---

        print("\n=== Strategy: messages (lightweight) ===\n")

        t.configure(compile_strategy="messages", compile_strategy_k=5)

        strategy = t.get_config("compile_strategy")
        ctx_messages = t.compile(strategy=strategy)
        print(f"  {strategy}: {len(ctx_messages.messages)} messages (commit-message text only)")

        # --- Compile strategy: adaptive ---

        print("\n=== Strategy: adaptive ===\n")

        t.configure(compile_strategy="adaptive")

        strategy = t.get_config("compile_strategy")
        k = t.get_config("compile_strategy_k")
        ctx_adaptive = t.compile(strategy=strategy, strategy_k=k)
        print(f"  {strategy} (k={k}): {len(ctx_adaptive.messages)} messages")

        # --- Comparison ---

        print("\n=== Strategy Comparison ===\n")

        # All strategies produce the same number of messages -- they differ
        # in *content detail*, not in message count.
        print(f"  full:     {len(ctx_full.messages)} messages (full content)")
        print(f"  messages: {len(ctx_messages.messages)} messages (lightweight commit messages)")
        print(f"  adaptive: {len(ctx_adaptive.messages)} messages (last {k} full, rest lightweight)")

        # --- All active config ---

        print("\n=== All Active Configs ===\n")
        for key, val in sorted(t.get_all_configs().items()):
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
