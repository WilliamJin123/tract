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

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
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


if __name__ == "__main__":
    main()
