"""Reasoning Traces

Capture LLM reasoning (chain-of-thought, thinking tokens) as first-class
commits. Reasoning commits are SKIP by default — excluded from compile() to
avoid bloating the next LLM call — but always inspectable via log() and
retrievable via compile(include_reasoning=True).

Part 1 — Manual reasoning commits: use t.reasoning() to commit thinking
text, inspect it in log(), and verify it's excluded from compile().

Part 2 — Compile control: compile(include_reasoning=True) promotes
reasoning from SKIP to NORMAL. Explicit annotations always win.

Part 3 — Formatting: pprint() renders reasoning in dim cyan, visually
distinct from dialogue. All three styles (table/chat/compact) handle it.

Part 4 — LLM integration: generate() auto-extracts reasoning from
provider responses (Cerebras, OpenAI o1/o3, Anthropic thinking, <think>
tags). Auto-committed before the assistant response. Per-call and global
opt-out available.

No LLM required for Parts 1-3. Part 4 uses an LLM.

Demonstrates: t.reasoning(), compile(include_reasoning=True),
              Tract.open(commit_reasoning=False), generate(reasoning=False),
              ChatResponse.reasoning, ChatResponse.reasoning_commit,
              annotate() overrides, pprint() reasoning style, log()
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract, ReasoningContent

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Manual Reasoning Commits
# =============================================================================
# t.reasoning() commits a ReasoningContent with SKIP priority by default.
# The reasoning is in the commit chain (visible in log()) but excluded
# from compile() — the LLM never sees it unless you ask for it.

def part1_manual_reasoning():
    print("=" * 60)
    print("Part 1: MANUAL REASONING COMMITS")
    print("=" * 60)
    print()
    print("  t.reasoning() commits chain-of-thought text.")
    print("  Default priority is SKIP — excluded from compile().")
    print()

    t = Tract.open()

    # Build a conversation with reasoning between user and assistant
    t.system("You are a math tutor. Show your work.")
    t.user("What is 17 * 23?")

    # Reasoning: the model's internal thinking (committed manually here)
    r_info = t.reasoning(
        "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391",
        format="parsed",
    )

    t.assistant("17 × 23 = 391")

    # --- Reasoning is in log() ---

    print("  log() shows reasoning commits:\n")
    for entry in reversed(t.log()):
        print(f"    {entry}")

    print(f"\n  Reasoning commit: {r_info.commit_hash[:8]}")
    print(f"  Content type:     {r_info.content_type}")

    # --- But excluded from compile() ---

    ctx = t.compile()
    print(f"\n  compile() -> {ctx.commit_count} messages (reasoning excluded):")
    for msg in ctx.messages:
        print(f"    [{msg.role}] {msg.content[:60]}")

    # --- Format and metadata ---

    print(f"\n  t.reasoning() also accepts format= and metadata=:")
    t2 = Tract.open()
    info = t2.reasoning(
        "Let me think step by step...",
        format="think_tags",
        metadata={"source": "deepseek-r1"},
    )
    content = t2.get_content(info.commit_hash)
    print(f"    format:   {content['format']}")
    meta = t2.get_metadata(info.commit_hash)
    print(f"    metadata: {meta}")
    t2.close()

    t.close()


# =============================================================================
# Part 2: Compile Control
# =============================================================================
# compile(include_reasoning=True) promotes reasoning SKIP -> NORMAL.
# Explicit annotations from t.annotate() always take precedence.

def part2_compile_control():
    print(f"\n{'=' * 60}")
    print("Part 2: COMPILE CONTROL (include_reasoning=)")
    print("=" * 60)
    print()

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Explain photosynthesis.")
    t.reasoning(
        "Photosynthesis converts CO2 + H2O into glucose + O2 using "
        "sunlight. The light reactions happen in thylakoids, the Calvin "
        "cycle in the stroma. I should keep this concise."
    )
    t.assistant(
        "Photosynthesis converts CO2 and water into glucose and oxygen "
        "using sunlight energy, primarily in chloroplasts."
    )

    # --- Default: reasoning excluded ---

    ctx_default = t.compile()
    print(f"  compile() default:")
    print(f"    {ctx_default.commit_count} messages, {ctx_default.token_count} tokens")
    roles = [m.role for m in ctx_default.messages]
    print(f"    roles: {roles}")

    # --- include_reasoning=True: reasoning included ---

    ctx_with = t.compile(include_reasoning=True)
    print(f"\n  compile(include_reasoning=True):")
    print(f"    {ctx_with.commit_count} messages, {ctx_with.token_count} tokens")
    roles = [m.role for m in ctx_with.messages]
    print(f"    roles: {roles}")
    extra = ctx_with.token_count - ctx_default.token_count
    print(f"    +{extra} tokens from reasoning content")

    # --- Explicit annotation overrides ---

    print(f"\n  Explicit annotations always win:")

    t2 = Tract.open()
    t2.user("Hello")
    info = t2.reasoning("Important chain of thought")
    t2.annotate(info.commit_hash, Priority.PINNED, reason="keep this")
    t2.assistant("Hi!")

    # PINNED reasoning appears even without include_reasoning
    ctx = t2.compile()
    texts = [m.content for m in ctx.messages]
    has_reasoning = "Important chain of thought" in texts
    print(f"    PINNED reasoning in compile(): {has_reasoning}")

    # Explicit SKIP is respected even with include_reasoning=True
    t3 = Tract.open()
    t3.user("Hello")
    info2 = t3.reasoning("Thinking...")
    t3.annotate(info2.commit_hash, Priority.SKIP, reason="exclude this")
    t3.assistant("Hi!")

    ctx2 = t3.compile(include_reasoning=True)
    texts2 = [m.content for m in ctx2.messages]
    has_reasoning2 = "Thinking..." in texts2
    print(f"    Explicit SKIP with include_reasoning=True: hidden={not has_reasoning2}")

    t.close()
    t2.close()
    t3.close()


# =============================================================================
# Part 3: Formatting
# =============================================================================
# pprint() renders reasoning commits in dim cyan — visually distinct from
# dialogue. All three styles handle reasoning content.

def part3_formatting():
    print(f"\n{'=' * 60}")
    print("Part 3: FORMATTING (pprint with reasoning)")
    print("=" * 60)
    print()
    print("  Reasoning commits render in dim cyan, visually")
    print("  distinct from regular dialogue.\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What is the capital of France?")
    t.reasoning(
        "The user is asking about France's capital. This is a "
        "straightforward geography question. The answer is Paris."
    )
    t.assistant("The capital of France is Paris.")

    # Include reasoning so pprint() can show it
    ctx = t.compile(include_reasoning=True)

    print("  --- table style ---\n")
    ctx.pprint(style="table")

    print("\n  --- chat style ---\n")
    ctx.pprint(style="chat")

    print("\n  --- compact style ---\n")
    ctx.pprint(style="compact")

    t.close()


# =============================================================================
# Part 4: LLM Integration (requires API key)
# =============================================================================
# generate() auto-extracts reasoning from LLM responses. The reasoning
# is committed before the assistant response (matching execution order).
# Supports 4 provider formats: Cerebras parsed, OpenAI o1/o3, Anthropic
# thinking, and <think> tags.

def part4_llm_integration():
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("Part 4: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("Part 4: LLM INTEGRATION (auto-extract reasoning)")
    print("=" * 60)
    print()

    # --- 4a: generate() with reasoning ---

    print("  4a: generate() auto-commits reasoning traces\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step before answering.")
        t.user("What is 15% of 240?")

        resp = t.generate()

        print(f"  ChatResponse.text:     {resp.text[:80]}...")
        print(f"  ChatResponse.reasoning: {repr(resp.reasoning)[:80] if resp.reasoning else 'None'}")

        if resp.reasoning_commit:
            print(f"  reasoning_commit hash: {resp.reasoning_commit.commit_hash[:8]}")
            print(f"  reasoning_commit type: {resp.reasoning_commit.content_type}")
        else:
            print("  (Model did not produce reasoning tokens)")

        # Log shows the full chain
        print(f"\n  log() (newest first):")
        for entry in t.log(limit=5):
            print(f"    {entry}")

    # --- 4b: Per-call opt-out ---

    print(f"\n  4b: reasoning=False skips the commit\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("Think step by step.")
        t.user("What is 7 * 8?")

        resp = t.generate(reasoning=False)

        # Reasoning text is still extracted (if available)...
        print(f"  reasoning extracted: {resp.reasoning is not None}")
        # ...but NOT committed
        print(f"  reasoning committed: {resp.reasoning_commit is not None}")

        log_types = [e.content_type for e in t.log()]
        print(f"  content types in log: {log_types}")
        print(f"  'reasoning' in log: {'reasoning' in log_types}")

    # --- 4c: Global opt-out ---

    print(f"\n  4c: Tract.open(commit_reasoning=False) disables globally\n")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
        commit_reasoning=False,
    ) as t:
        t.system("Think carefully.")
        t.user("What is 12 + 13?")

        resp = t.generate()

        print(f"  reasoning extracted: {resp.reasoning is not None}")
        print(f"  reasoning committed: {resp.reasoning_commit is not None}")
        print(f"  (t.reasoning() shorthand still works even with global opt-out)")

        # Manual reasoning is always allowed
        manual = t.reasoning("This was added manually.")
        print(f"  manual commit type:  {manual.content_type}")


def main():
    part1_manual_reasoning()
    part2_compile_control()
    part3_formatting()
    part4_llm_integration()


if __name__ == "__main__":
    main()
