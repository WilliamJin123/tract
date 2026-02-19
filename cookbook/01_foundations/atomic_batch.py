"""Tier 1.3 — Atomic Multi-Turn Exchange

A RAG pipeline that retrieves context, asks the LLM a question, and gets a
response. All three commits (retrieval, user query, assistant response) land
atomically via batch() — either all succeed or none do.

Also demonstrates generation_config tracking: the exact model and temperature
that produced each response are stored with the commit.

Demonstrates: batch(), generation_config, compile(), log()
"""

import os

import httpx
from dotenv import load_dotenv

from tract import (
    DialogueContent,
    FreeformContent,
    InstructionContent,
    Tract,
)

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"
TEMPERATURE = 0.3


def call_llm(messages: list[dict]) -> dict:
    """Call Cerebras and return the full response."""
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{CEREBRAS_BASE_URL}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {CEREBRAS_API_KEY}",
            },
            json={
                "model": CEREBRAS_MODEL,
                "messages": messages,
                "temperature": TEMPERATURE,
            },
        )
        response.raise_for_status()
        return response.json()


def fake_retrieval(query: str) -> str:
    """Simulate a RAG retrieval step — in production this would hit a vector DB."""
    return (
        f"[Retrieved context for: '{query}']\n"
        "Python 3.12 introduced several performance improvements:\n"
        "- Comprehension inlining (PEP 709) makes list/dict/set comprehensions faster\n"
        "- The new type statement (PEP 695) simplifies generic type aliases\n"
        "- Immortal objects reduce reference counting overhead for singletons\n"
        "- The Linux perf profiler support was improved\n"
    )


def main():
    with Tract.open() as t:
        # System prompt (outside the batch — it's a one-time setup)
        t.commit(
            InstructionContent(
                text="You are a Python expert. Answer based on the provided context. "
                     "Cite specific features when possible."
            ),
            message="system prompt",
        )

        # --- Batch 1: RAG retrieval + question + answer ---
        print("=== Batch 1: RAG-augmented Q&A ===\n")

        query = "What performance improvements came in Python 3.12?"
        retrieved_context = fake_retrieval(query)

        # Compile context BEFORE the batch to get the messages for the LLM call
        # (the batch hasn't started yet, so we include the system prompt)
        pre_batch_compiled = t.compile()

        with t.batch():
            # Commit the retrieved context as freeform content
            t.commit(
                FreeformContent(payload={"source": "vector_db", "text": retrieved_context}),
                message="RAG retrieval: python 3.12 perf",
            )

            # Commit the user question
            user_commit = t.commit(
                DialogueContent(role="user", text=query),
                message="user asks about 3.12 perf",
            )

            # Build messages for the LLM call (system prompt + retrieval + question)
            messages = [{"role": m.role, "content": m.content} for m in pre_batch_compiled.messages]
            messages.append({"role": "assistant", "content": retrieved_context})
            messages.append({"role": "user", "content": query})

            # Call the LLM
            response = call_llm(messages)
            assistant_text = response["choices"][0]["message"]["content"]

            # Commit the response WITH generation_config so we know exactly
            # what model/temperature produced it
            t.commit(
                DialogueContent(role="assistant", text=assistant_text),
                message="assistant answers re: 3.12 perf",
                generation_config={
                    "model": CEREBRAS_MODEL,
                    "temperature": TEMPERATURE,
                    "provider": "cerebras",
                },
            )

        # The batch committed atomically — all 3 or nothing
        print(f"Assistant: {assistant_text[:200]}...\n")

        # --- Batch 2: A follow-up exchange ---
        print("=== Batch 2: Follow-up ===\n")

        follow_up = "Which of those changes has the biggest real-world impact?"

        # Compile now includes everything from batch 1
        compiled = t.compile()
        messages = [{"role": m.role, "content": m.content} for m in compiled.messages]
        messages.append({"role": "user", "content": follow_up})

        with t.batch():
            t.commit(
                DialogueContent(role="user", text=follow_up),
                message="user asks about biggest impact",
            )

            response = call_llm(messages)
            assistant_text = response["choices"][0]["message"]["content"]

            t.commit(
                DialogueContent(role="assistant", text=assistant_text),
                message="assistant explains biggest impact",
                generation_config={
                    "model": CEREBRAS_MODEL,
                    "temperature": TEMPERATURE,
                    "provider": "cerebras",
                },
            )

        print(f"Assistant: {assistant_text[:200]}...\n")

        # --- Inspect the full history ---
        print("=== Full History ===\n")
        history = t.log(limit=20)
        for entry in reversed(history):
            config_str = ""
            if entry.generation_config:
                config_str = f" | config: {entry.generation_config}"
            print(
                f"  {entry.commit_hash[:8]} "
                f"[{entry.content_type:12s}] "
                f"{entry.message}"
                f"{config_str}"
            )

        # --- Query by generation config ---
        print("\n=== Commits produced by Cerebras ===\n")
        cerebras_commits = t.query_by_config("provider", "=", "cerebras")
        for entry in cerebras_commits:
            print(f"  {entry.commit_hash[:8]} | {entry.message} | model={entry.generation_config.get('model')}")

        # Final status
        status = t.status()
        print(f"\nFinal: {status.commit_count} commits, {status.token_count} tokens")


if __name__ == "__main__":
    main()
