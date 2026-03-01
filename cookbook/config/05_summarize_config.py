"""
05 — Auto-Summarize Commit Messages
====================================

When you commit without a ``message=`` parameter, Tract can use an LLM to
generate a concise one-sentence commit message instead of truncating content.

Pass ``auto_summarize=`` to ``Tract.open()`` to enable it:
- ``True``: use the tract-level default model
- ``"model-name"``: use a specific (cheap) model
- ``LLMConfig(...)``: full control over the summarization config

This example shows all four modes.
"""

import os
from dotenv import load_dotenv
from tract import LLMConfig, Tract


load_dotenv()
TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"
MESSAGE_MODEL_ID = "llama3.1-8b"  # cheap model for summarization


# --- 1. Cheapest: point auto_summarize at a small model ---
# One parameter does it all.  Uses llama3.1-8b for commit messages
# while the main tract model stays gpt-oss-120b.

with Tract.open(
    api_key=TRACT_OPENAI_API_KEY,
    base_url=TRACT_OPENAI_BASE_URL,
    model=MODEL_ID,
    auto_summarize=MESSAGE_MODEL_ID,
) as t:
    t.system("You are a helpful coding assistant specializing in Python.")
    t.user("How do I read a CSV file with pandas?")

    # Log shows clean messages like:
    #   "Set up Python coding assistant"
    #   "Ask how to read CSV with pandas"
    for entry in t.log():
        print(f"  {entry.commit_hash[:8]} {entry.message}")


# --- 2. Reuse the tract model ---
# If your tract model is already cheap, just pass True.

with Tract.open(
    api_key=TRACT_OPENAI_API_KEY,
    base_url=TRACT_OPENAI_BASE_URL,
    model=MESSAGE_MODEL_ID,
    auto_summarize=True,
) as t:
    t.system("You are a helpful assistant.")
    t.chat("Tell me a joke.")

    print(t.log())

# --- 3. Full control with LLMConfig ---

with Tract.open(
    api_key=TRACT_OPENAI_API_KEY,
    base_url=TRACT_OPENAI_BASE_URL,
    model=MODEL_ID,
    auto_summarize=LLMConfig(model=MESSAGE_MODEL_ID, temperature=0.0, max_tokens=60),
) as t:
    t.system("You are a helpful assistant.")
    t.chat("Tell me a joke.")
    print(t.log())

# --- 4. Off by default ---
# Without auto_summarize=, commit messages are truncated content previews.

with Tract.open(
    api_key=TRACT_OPENAI_API_KEY,
    base_url=TRACT_OPENAI_BASE_URL,
    model=MODEL_ID,
) as t:
    t.system("You are a helpful assistant.")
    t.chat("Write a haiku.")
    print(t.log())
    # Message will be: "You are a helpful assistant."  (raw text, up to 500 chars)


# --- 5. Per-operation client (advanced) ---
# Use a completely different LLM provider for summarization.

# from openai import OpenAI
# summarize_client = OpenAI(api_key="sk-other-key")
# with Tract.open(api_key="sk-...", auto_summarize=True) as t:
#     t.configure_clients(summarize=summarize_client)
#     t.system("You are helpful.")
