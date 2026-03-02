"""Config Provenance

Chat with different settings across calls, then query: "what temperature
was used for this output?" Every assistant commit auto-captures the
fully-resolved generation_config. query_by_config() searches by single
field, multi-field AND, or comparison operators.

Demonstrates: generation_config, query_by_config() patterns (single-field,
              multi-field AND, comparison operators, IN operator, between),
              pprint(style="chat"), response.pprint(), log()
"""

import os

import click
from dotenv import load_dotenv

from tract import Priority, Tract, ToolCall

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def part1_config_provenance():
    print("=" * 60)
    print("PART 1 -- Manual: CONFIG PROVENANCE")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        t.system("You are a creative writing assistant. Keep responses under 2 sentences.")

        # --- Turn 1: default settings ---

        print("=== Turn 1: default config ===\n")
        r1 = t.chat("Write a one-sentence opening for a mystery novel.")
        r1.pprint()

        # --- Turn 2: high temperature for more creativity ---

        print("\n=== Turn 2: temperature=1.5 ===\n")
        r2 = t.chat(
            "Now write a wilder, more surreal version.",
            temperature=1.5,
        )
        r2.pprint()

        # --- Turn 3: low temperature + limited tokens ---

        print("\n=== Turn 3: temperature=0.0, max_tokens=200 ===\n")
        r3 = t.chat(
            "Write a final version — precise, clinical, no embellishment.",
            temperature=0.0,
            max_tokens=200,
        )
        r3.pprint()

        # --- Full session view ---

        print("\n=== Full session ===\n")
        t.compile().pprint(style="chat")

        # --- Query: single field with comparison operator ---
        # "Which calls used a high temperature?"

        print("\n=== Query: temperature > 1.0 ===\n")
        hot = t.query_by_config("temperature", ">", 1.0)
        print(f"  {len(hot)} commit(s) with temperature > 1.0:")
        for c in hot:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

        # --- Query: equality match ---

        print("\n=== Query: max_tokens = 200 ===\n")
        limited = t.query_by_config("max_tokens", "=", 200)
        print(f"  {len(limited)} commit(s) with max_tokens=200:")
        for c in limited:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  max_tokens={cfg.max_tokens}")

        # --- Query: inclusive range with "between" ---
        # "Which calls used a temperature between 0.0 and 1.0 (inclusive)?"

        print("\n=== Query: temperature between [0.0, 1.0] ===\n")
        moderate = t.query_by_config("temperature", "between", [0.0, 1.0])
        print(f"  {len(moderate)} commit(s) with temperature between [0.0, 1.0]:")
        for c in moderate:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

        # --- Query: multi-field AND ---
        # "Which calls used THIS model AND temperature=0.0?"

        print(f"\n=== Query: model={MODEL_ID} AND temperature=0.0 ===\n")
        specific = t.query_by_config(conditions=[
            ("model", "=", MODEL_ID),
            ("temperature", "=", 0.0),
        ])
        print(f"  {len(specific)} commit(s) match:")
        for c in specific:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  model={cfg.model}, temp={cfg.temperature}")

        # --- Query: IN operator (set membership) ---
        # "Which calls used temperature 0.0 or 1.5?"

        print("\n=== Query: temperature in list [0.0, 1.5] ===\n")
        extremes = t.query_by_config("temperature", "in", [0.0, 1.5])
        print(f"  {len(extremes)} commit(s) at extreme temperatures:")
        for c in extremes:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

        # --- All commits: config provenance summary ---

        print("\n=== All assistant commits — config provenance ===\n")
        for entry in reversed(t.log()):
            if entry.generation_config:
                fields = entry.generation_config.non_none_fields()
                parts = [f"{k}={v}" for k, v in fields.items()]
                print(f"  {entry.commit_hash[:8]}: {', '.join(parts)}")


def part2_interactive():
    """Part 2: Interactive -- human picks temperature and queries config."""
    print(f"\n{'=' * 60}")
    print("PART 2 -- Interactive: CONFIG PROVENANCE")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a creative writing assistant. Keep responses under 2 sentences.")

        # Let the user pick temperature for each call
        for i in range(2):
            temp = click.prompt(
                f"  Temperature for call {i+1}?", type=float, default=0.7,
            )
            prompt = click.prompt(f"  Prompt for call {i+1}?",
                                  default="Write a one-sentence story.")
            r = t.chat(prompt, temperature=temp)
            r.pprint()
            print()

        # Interactive query builder
        print("  --- Interactive query builder ---\n")
        field = click.prompt("  Query field",
                             type=click.Choice(["temperature", "max_tokens", "model"]))
        operator = click.prompt("  Operator",
                                type=click.Choice(["=", ">", "<"]))
        value_str = click.prompt("  Value")

        # Coerce value to the right type
        try:
            value = float(value_str)
        except ValueError:
            value = value_str

        results = t.query_by_config(field, operator, value)
        print(f"\n  {len(results)} commit(s) match {field} {operator} {value}:")
        for c in results:
            cfg = c.generation_config
            fields = cfg.non_none_fields()
            parts = [f"{k}={v}" for k, v in fields.items()]
            print(f"    {c.commit_hash[:8]}: {', '.join(parts)}")


# =============================================================================
# Part 3 -- Agent: Queries Own Config Provenance
# =============================================================================
# Agents can audit which model produced each response, useful for
# A/B testing provenance and debugging unexpected outputs.

def part3_agent():
    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: QUERIES OWN CONFIG PROVENANCE")
    print("=" * 60)
    print()

    from tract.toolkit import ToolExecutor

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        executor = ToolExecutor(t)

        # Make calls with different configs
        t.chat("What is Python?", temperature=0.2)
        t.chat("Write a poem about code.", temperature=0.9)

        # Agent queries its own generation_config provenance
        results = t.query_by_config("model", "=", MODEL_ID)
        print(f"  {len(results)} commit(s) using model={MODEL_ID}:")
        for c in results:
            cfg = c.generation_config
            print(f"    {c.commit_hash[:8]}  temp={cfg.temperature}")

    # Note: Agents can audit which model produced each response.
    # This is useful for A/B testing provenance and debugging
    # unexpected outputs across different model configurations.


def main():
    part1_config_provenance()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()
