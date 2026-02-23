"""Config Provenance

Chat with different settings across calls, then query: "what temperature
was used for this output?" Every assistant commit auto-captures the
fully-resolved generation_config. query_by_config() searches by single
field, multi-field AND, or comparison operators.

Demonstrates: generation_config on commits, query_by_config() patterns,
              chat(temperature=), chat(max_tokens=), CommitInfo.generation_config
"""

import os

from dotenv import load_dotenv

from tract import Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def main():
    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:

        t.system("You are a creative writing assistant. Keep responses under 2 sentences.")

        # --- Turn 1: default settings ---

        print("=== Turn 1: default config ===\n")
        r1 = t.chat("Write a one-sentence opening for a mystery novel.")
        r1.pprint()
        cfg1 = r1.commit_info.generation_config
        print(f"  Config: model={cfg1.model}, temperature={cfg1.temperature}\n")

        # --- Turn 2: high temperature for more creativity ---

        print("=== Turn 2: temperature=1.5 ===\n")
        r2 = t.chat(
            "Now write a wilder, more surreal version.",
            temperature=1.5,
        )
        r2.pprint()
        cfg2 = r2.commit_info.generation_config
        print(f"  Config: model={cfg2.model}, temperature={cfg2.temperature}\n")

        # --- Turn 3: low temperature + limited tokens ---

        print("=== Turn 3: temperature=0.0, max_tokens=200 ===\n")
        r3 = t.chat(
            "Write a final version — precise, clinical, no embellishment.",
            temperature=0.0,
            max_tokens=200,
        )
        r3.pprint()
        cfg3 = r3.commit_info.generation_config
        print(f"  Config: model={cfg3.model}, temperature={cfg3.temperature}, "
              f"max_tokens={cfg3.max_tokens}\n")

        # --- Query by single field: find the high-creativity call ---

        print("=== Query: temperature > 1.0 ===\n")
        hot = t.query_by_config("temperature", ">", 1.0)
        print(f"  {len(hot)} commits with temperature > 1.0:")
        for c in hot:
            print(f"    {c}  (temp={c.generation_config.temperature})")

        # --- Query: find the constrained call ---

        print("\n=== Query: max_tokens = 50 ===\n")
        limited = t.query_by_config("max_tokens", "=", 50)
        print(f"  {len(limited)} commits with max_tokens=50:")
        for c in limited:
            print(f"    {c}")

        # --- Multi-field AND query ---

        print(f"\n=== Query: model={CEREBRAS_MODEL} AND temperature=0.0 ===\n")
        specific = t.query_by_config(conditions=[
            ("model", "=", CEREBRAS_MODEL),
            ("temperature", "=", 0.0),
        ])
        print(f"  {len(specific)} commits match:")
        for c in specific:
            print(f"    {c}")

        # --- IN operator: find multiple temperatures ---

        print("\n=== Query: temperature IN [0.0, 1.5] ===\n")
        extremes = t.query_by_config("temperature", "IN", [0.0, 1.5])
        print(f"  {len(extremes)} commits at extreme temperatures:")
        for c in extremes:
            print(f"    {c}  (temp={c.generation_config.temperature})")

        # --- All commits: show config variation ---

        print("\n=== All assistant commits with configs ===\n")
        for entry in reversed(t.log()):
            if entry.generation_config:
                fields = entry.generation_config.non_none_fields()
                parts = [f"{k}={v}" for k, v in fields.items()]
                print(f"  {entry.commit_hash[:8]}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
