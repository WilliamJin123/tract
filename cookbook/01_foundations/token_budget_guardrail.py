"""Tier 1.2 — Token Budget Guardrail

A chatbot with a tight token budget (4096 tokens) that checks remaining
capacity before every LLM call. After each call, it records the API's actual
token usage so tracking reflects reality, not just tiktoken estimates.

Demonstrates: status(), TokenBudgetConfig, record_usage()
"""

import os

import httpx
from dotenv import load_dotenv

from tract import (
    DialogueContent,
    InstructionContent,
    TokenBudgetConfig,
    Tract,
    TractConfig,
)

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


def call_llm(messages: list[dict]) -> dict:
    """Call Cerebras and return the full response (including usage)."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{CEREBRAS_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            },
            json={"model": CEREBRAS_MODEL, "messages": messages},
        )
        response.raise_for_status()
        return response.json()


def main():
    # Configure a budget of 4096 tokens — tight enough to see the guardrail fire
    config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )

    with Tract.open(config=config) as t:
        # System prompt
        t.commit(
            InstructionContent(text="You are a concise assistant. Keep answers under 3 sentences."),
            message="system prompt",
        )

        # Simulate a multi-turn conversation
        questions = [
            "What is a Python decorator?",
            "How does asyncio work?",
            "Explain the GIL in Python.",
            "What are metaclasses?",
            "How does garbage collection work in CPython?",
        ]

        for i, question in enumerate(questions, 1):
            # --- Pre-call check: do we have budget left? ---
            status = t.status()
            budget_max = status.token_budget_max or float("inf")
            usage_pct = (status.token_count / budget_max * 100) if budget_max else 0

            print(f"\n--- Turn {i} ---")
            print(f"  Tokens: {status.token_count}/{budget_max} ({usage_pct:.0f}%)")

            if usage_pct > 90:
                print(f"  WARNING: Budget nearly exhausted ({usage_pct:.0f}%). Stopping.")
                print(f"  (In production, you'd compress or branch here.)")
                break

            # Commit the user question
            t.commit(
                DialogueContent(role="user", text=question),
                message=f"user question {i}",
            )

            # Compile and call LLM
            compiled = t.compile()
            messages = [{"role": m.role, "content": m.content} for m in compiled.messages]

            print(f"  Tiktoken estimate: {compiled.token_count} tokens")
            print(f"  Sending {len(messages)} messages to LLM...")

            response = call_llm(messages)
            assistant_text = response["choices"][0]["message"]["content"]

            # Commit the response
            t.commit(
                DialogueContent(role="assistant", text=assistant_text),
                message=f"assistant answer {i}",
            )

            # --- Post-call: record actual API usage ---
            api_usage = response.get("usage")
            if api_usage:
                updated = t.record_usage(api_usage)
                print(f"  API reported: {api_usage.get('prompt_tokens', '?')} prompt + "
                      f"{api_usage.get('completion_tokens', '?')} completion tokens")
                print(f"  Token source updated: {updated.token_source}")
            else:
                print(f"  (No usage data in API response)")

            print(f"  Assistant: {assistant_text[:100]}...")

        # Final summary
        print("\n=== Final Status ===")
        status = t.status()
        budget_max = status.token_budget_max or 0
        print(f"Commits: {status.commit_count}")
        print(f"Tokens: {status.token_count}/{budget_max}")
        print(f"Token source: {status.token_source}")
        print(f"Branch: {status.branch_name}")


if __name__ == "__main__":
    main()
