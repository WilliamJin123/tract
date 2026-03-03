"""Self-correcting agent: validation + retry + edit + metadata.

  PART 1 -- Manual:      generate() -> validate -> if fail: commit correction, retry loop
  PART 2 -- LLM / Agent:  generate(validator=fn, max_retries=3, hide_retries=True)
"""

import json
import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: generate + validate + retry loop
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Validation Retry Loop")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a data assistant. Always respond with valid JSON only. "
                 "No markdown, no explanation, just raw JSON.")
        t.user("Return a JSON array of 3 planets with name and distance_au fields.")

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            response = t.generate()
            print(f"\n  Attempt {attempt}: {response.text[:80]}...")

            try:
                data = json.loads(response.text)
                print(f"  Valid JSON: {len(data)} items")

                print("\n  Conversation after successful validation:\n")
                t.compile().pprint(style="compact")
                break
            except json.JSONDecodeError as e:
                print(f"  Invalid JSON: {e}")
                if attempt < max_retries:
                    t.user(f"That was not valid JSON. Error: {e}. "
                           "Please return ONLY a raw JSON array, no markdown.")
                else:
                    print("  Max retries reached.")


# =====================================================================
# PART 2 -- LLM / Agent: built-in validation + retry
# =====================================================================

def part2_agent():
    print("\n" + "=" * 60)
    print("PART 2 -- LLM / Agent: Built-In Validator")
    print("=" * 60)

    def json_validator(text: str) -> tuple[bool, str | None]:
        """Validate that the response is valid JSON."""
        try:
            json.loads(text)
            return (True, None)
        except json.JSONDecodeError as e:
            return (False, f"Invalid JSON: {e}")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a data assistant. Always respond with valid JSON only. "
                 "No markdown, no explanation.")

        # generate() handles the full retry loop internally
        t.user("Return a JSON array of 3 galaxies with name and "
               "distance_mly (million light years) fields.")
        response = t.generate(
            validator=json_validator,
            max_retries=3,
            hide_retries=True,
        )

        print(f"\n  Final response: {response.text[:120]}")
        print(f"  Retries used: {response.metadata.get('retries', 0) if response.metadata else 0}")

        # Verify the result
        try:
            data = json.loads(response.text)
            print(f"  Parsed: {len(data)} galaxies")
            for item in data[:3]:
                print(f"    {item}")
        except json.JSONDecodeError:
            print("  Still invalid after retries.")

        # Check retry metadata on the commit
        print(f"\n  Retry metadata: {response.commit_info.metadata}")

        print("\n  Final context (retries hidden):\n")
        t.compile().pprint(style="compact")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()
