"""Atomic Multi-Turn Exchange

A RAG pipeline that retrieves context, asks the LLM a question, and gets a
response. The retrieval context and user question are committed atomically
via batch() — either both land or neither does. Then generate() handles the
LLM call and assistant commit separately.

Also demonstrates auto-captured generation_config: the exact model and
temperature that produced each response are stored with the commit
automatically — no manual tracking required.

Demonstrates: batch(), generate(), auto generation_config, query_by_config()
"""

import os

from dotenv import load_dotenv

from tract import FreeformContent, Tract

load_dotenv()

CEREBRAS_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
CEREBRAS_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
CEREBRAS_MODEL = "gpt-oss-120b"


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
    with Tract.open(
        api_key=CEREBRAS_API_KEY,
        base_url=CEREBRAS_BASE_URL,
        model=CEREBRAS_MODEL,
    ) as t:
        # System prompt
        t.system(
            "You are a Python expert. Answer based on the provided context. "
            "Cite specific features when possible."
        )

        # --- Batch 1: RAG retrieval + question (atomic), then LLM call ---
        print("=== Batch 1: RAG-augmented Q&A ===\n")

        query = "What performance improvements came in Python 3.12?"
        retrieved_context = fake_retrieval(query)

        # Batch the context setup: retrieval + user question land atomically
        with t.batch():
            t.commit(
                FreeformContent(payload={"source": "vector_db", "text": retrieved_context}),
                message="RAG retrieval: python 3.12 perf",
            )
            t.user(query)

        # generate() compiles all context, calls the LLM, commits the response,
        # and auto-captures generation_config (model, temperature, etc.)
        response = t.generate(temperature=0.3)

        print(f"Assistant: {response.text[:200]}...")
        print(f"Config captured: {response.generation_config}\n")

        # --- Batch 2: Follow-up exchange ---
        print("=== Batch 2: Follow-up ===\n")

        follow_up = "Which of those changes has the biggest real-world impact?"

        # For simple exchanges, user() + generate() works without batch
        t.user(follow_up)
        response = t.generate(temperature=0.3)

        print(f"Assistant: {response.text[:200]}...\n")

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
        print(f"\n=== Commits using model {CEREBRAS_MODEL} ===\n")
        model_commits = t.query_by_config("model", "=", CEREBRAS_MODEL)
        for entry in model_commits:
            print(f"  {entry.commit_hash[:8]} | {entry.message} | model={entry.generation_config.get('model')}")

        # Final status
        status = t.status()
        print(f"\nFinal: {status.commit_count} commits, {status.token_count} tokens")


if __name__ == "__main__":
    main()
