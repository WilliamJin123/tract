"""Auto-Message Commit Messages
================================

When you commit without a ``message=`` parameter, Tract can use an LLM to
generate a concise one-sentence commit message instead of truncating content.

Pass ``auto_message=`` to ``Tract.open()`` to enable it:
- ``"model-name"``: use a specific (cheap) model
- ``True``: reuse the tract-level default model
- ``LLMConfig(...)``: full control over the message generation config

This example runs all three modes plus "off" for comparison.

Demonstrates: auto_message modes, compile().pprint(style="compact"),
              log table via pprint_log, LLMConfig for message generation
"""

import sys
from pathlib import Path

from tract import LLMConfig, Tract
from tract.formatting import pprint_log

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm

MODEL_ID = llm.large
MESSAGE_MODEL_ID = llm.small


# --- 1. Cheapest: point auto_message at a small model ---
# One parameter does it all.  Uses the small model for commit messages
# while the main tract model stays large.

print("=" * 60)
print("1. auto_message = small model name")
print("=" * 60)

with Tract.open(
    api_key=llm.api_key,
    base_url=llm.base_url,
    model=MODEL_ID,
    auto_message=MESSAGE_MODEL_ID,
) as t:
    t.system("You are a helpful coding assistant specializing in Python.")
    t.user("How do I read a CSV file with pandas?")

    # Log table shows auto-generated commit messages
    pprint_log(t.log())
    print()
    # Compact view shows the actual conversation content
    t.compile().pprint(style="compact")


# --- 2. Reuse the tract model ---
# If your tract model is already cheap, just pass True.

print(f"\n{'=' * 60}")
print("2. auto_message = True (reuses tract model)")
print("=" * 60)

with Tract.open(
    api_key=llm.api_key,
    base_url=llm.base_url,
    model=MESSAGE_MODEL_ID,
    auto_message=True,
) as t:
    t.system("You are a helpful assistant.")
    t.chat("Tell me a joke.")

    pprint_log(t.log())
    print()
    t.compile().pprint(style="compact")


# --- 3. Full control with LLMConfig ---

print(f"\n{'=' * 60}")
print("3. auto_message = LLMConfig (full control)")
print("=" * 60)

with Tract.open(
    api_key=llm.api_key,
    base_url=llm.base_url,
    model=MODEL_ID,
    auto_message=LLMConfig(model=MESSAGE_MODEL_ID, temperature=0.0, max_tokens=60),
) as t:
    t.system("You are a helpful assistant.")
    t.chat("Tell me a joke.")

    pprint_log(t.log())
    print()
    t.compile().pprint(style="compact")


# --- 4. Off by default ---
# Without auto_message=, commit messages are truncated content previews.

print(f"\n{'=' * 60}")
print("4. auto_message OFF (raw text previews)")
print("=" * 60)

with Tract.open(
    api_key=llm.api_key,
    base_url=llm.base_url,
    model=MODEL_ID,
) as t:
    t.system("You are a helpful assistant.")
    t.chat("Write a haiku.")

    # Compare: messages are raw content truncated, not LLM-generated
    pprint_log(t.log())
    print()
    t.compile().pprint(style="compact")
