"""Config and Compile Strategy Rules

Rules are commits with a name, trigger, condition (optional), and action.
Active rules (trigger="active") act as a key-value config layer for the agent:

  - Each set_config rule contributes one key-value pair
  - When multiple rules set the same key, closest to HEAD wins (DAG precedence)
  - Override a config by committing a new rule with the same name
  - Query with get_config() or resolve_all_configs()

The most common use: selecting the compile strategy that controls how
tract builds the LLM context window.

Demonstrates: active trigger, set_config action, DAG precedence,
              compile_strategy, compile_strategy_k, strategy comparison
"""

from tract import Tract, resolve_all_configs


def main():
    with Tract.open() as t:

        # --- Config rules as key-value store ---

        print("=== Config Rules ===\n")

        t.rule("model", trigger="active",
               action={"type": "set_config", "key": "model", "value": "gpt-4o"})
        t.rule("temperature", trigger="active",
               action={"type": "set_config", "key": "temperature", "value": 0.7})
        t.rule("max-tokens", trigger="active",
               action={"type": "set_config", "key": "max_tokens", "value": 4096})

        print(f"  model:       {t.get_config('model')}")
        print(f"  temperature: {t.get_config('temperature')}")
        print(f"  max_tokens:  {t.get_config('max_tokens')}")

        # --- DAG precedence: closer to HEAD wins ---

        print("\n=== DAG Precedence ===\n")

        t.user("Hello, world!")
        t.assistant("Hi there!")

        # Override model -- same name, new value closer to HEAD
        t.rule("model", trigger="active",
               action={"type": "set_config", "key": "model", "value": "claude-sonnet"})

        print(f"  model (overridden): {t.get_config('model')}")
        print(f"  temperature (unchanged): {t.get_config('temperature')}")
        print(f"  missing key:  {t.get_config('nonexistent')}")
        print(f"  with default: {t.get_config('nonexistent', 'fallback')}")

        # Complex values work too
        t.rule("stop-sequences", trigger="active",
               action={"type": "set_config", "key": "stop",
                        "value": ["END", "DONE", "---"]})

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

        t.rule("strategy", trigger="active",
               action={"type": "set_config", "key": "compile_strategy", "value": "full"})

        strategy = t.get_config("compile_strategy")
        ctx_full = t.compile(strategy=strategy)
        print(f"  {strategy}: {len(ctx_full.messages)} messages")

        # --- Compile strategy: messages (last K) ---

        print("\n=== Strategy: messages (last 5) ===\n")

        t.rule("strategy", trigger="active",
               action={"type": "set_config", "key": "compile_strategy", "value": "messages"})
        t.rule("strategy-k", trigger="active",
               action={"type": "set_config", "key": "compile_strategy_k", "value": 5})

        strategy = t.get_config("compile_strategy")
        k = t.get_config("compile_strategy_k")
        ctx_messages = t.compile(strategy=strategy, strategy_k=k)
        print(f"  {strategy} (k={k}): {len(ctx_messages.messages)} messages")

        # --- Compile strategy: adaptive ---

        print("\n=== Strategy: adaptive ===\n")

        t.rule("strategy", trigger="active",
               action={"type": "set_config", "key": "compile_strategy", "value": "adaptive"})

        strategy = t.get_config("compile_strategy")
        k = t.get_config("compile_strategy_k")
        ctx_adaptive = t.compile(strategy=strategy, strategy_k=k)
        print(f"  {strategy} (k={k}): {len(ctx_adaptive.messages)} messages")

        # --- Comparison ---

        print("\n=== Strategy Comparison ===\n")

        print(f"  full:     {len(ctx_full.messages)} messages")
        print(f"  messages: {len(ctx_messages.messages)} messages (last {k})")
        print(f"  adaptive: {len(ctx_adaptive.messages)} messages")

        # --- All active config ---

        print("\n=== All Active Configs ===\n")
        for key, val in sorted(resolve_all_configs(t.rule_index).items()):
            print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
