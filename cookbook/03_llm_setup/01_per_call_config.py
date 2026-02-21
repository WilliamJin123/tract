"""Per-Call LLM Config

Override model, temperature, or any LLM setting for a single call —
without changing your defaults. Two styles: sugar params for quick tweaks,
LLMConfig for full control.

Demonstrates: LLMConfig, chat(temperature=), generate(llm_config=),
              sugar params vs LLMConfig, generate() two-step control,
              response.pprint() for full response details
"""

import os

from dotenv import load_dotenv

from tract import LLMConfig, Tract

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
        t.system("You are a helpful assistant. Be concise.")

        # --- Style 1: Sugar params ---
        # Pass temperature=, model=, max_tokens= directly on chat().
        # Quick and readable for one-off tweaks.

        print("=== Sugar params ===\n")

        response = t.chat("What is Python?", temperature=0.2)
        gc = response.generation_config
        # Keep the field-level prints — they teach what's resolved where
        print(f"  temperature={gc.temperature}, model={gc.model}")
        # pprint() shows the full response panel (text + usage + config)
        response.pprint()
        print()

        # --- Style 2: LLMConfig object ---
        # For more settings or when you want to pass config around.

        print("=== LLMConfig object ===\n")

        creative = LLMConfig(temperature=0.9, top_p=0.95)
        response = t.chat("Give a creative analogy for Python.", llm_config=creative)
        gc = response.generation_config
        print(f"  temperature={gc.temperature}, top_p={gc.top_p}")
        print(f"  Response: {response.text[:100]}...\n")


        # --- Style 3: Both (sugar wins) ---
        # If you pass both llm_config= and a sugar param, sugar wins
        # for that specific field.

        print("=== Sugar overrides LLMConfig ===\n")

        response = t.chat(
            "Explain decorators.",
            llm_config=LLMConfig(temperature=0.3, max_tokens=200),
            temperature=0.8,  # overrides the 0.3 from llm_config
        )
        gc = response.generation_config
        print(f"  temperature={gc.temperature} (sugar won over 0.3)")
        print(f"  max_tokens={gc.max_tokens} (from llm_config)")
        print(f"  Response: {response.text[:100]}...\n")

        # --- generate(): two-step control ---
        # Commit the user message yourself, then call generate() separately.
        # Same config params, but you choose when to call.

        print("=== generate() two-step ===\n")

        t.user("What is the GIL?")
        response = t.generate(temperature=0.1, max_tokens=100)
        print(f"  temperature={response.generation_config.temperature}")
        print(f"  Response: {response.text[:100]}...")
        # Note: response.pprint() would show the same info as a rich panel


if __name__ == "__main__":
    main()
