"""Compile Control

Three tiers of compile control for reasoning: manual flag toggling,
interactive priority management, and agent-driven annotation.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: compile(include_reasoning=True), annotate() overrides,
              Priority.PINNED, Priority.SKIP, click.prompt(), ToolExecutor
"""

import os

import click
from dotenv import load_dotenv

from tract import Priority, Tract
from tract.toolkit import ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


def part1_compile_control():
    print(f"\n{'=' * 60}")
    print("Part 1: COMPILE CONTROL  [Manual Tier]")
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
# Part 2: Interactive Priority for Reasoning  (PART 2 — Interactive)
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("Part 2: INTERACTIVE REASONING PRIORITY  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  Walk reasoning commits from log(). For each, let the user")
    print("  change its visibility (NORMAL/SKIP/PINNED).")
    print()

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Explain quantum entanglement.")
    r1 = t.reasoning(
        "Quantum entanglement is when two particles become correlated. "
        "Measuring one instantly affects the other regardless of distance."
    )
    t.assistant("Quantum entanglement links two particles so measuring one "
                "instantly determines the state of the other.")
    t.user("Can it be used for communication?")
    r2 = t.reasoning(
        "No — entanglement doesn't allow faster-than-light communication. "
        "The no-communication theorem proves this."
    )
    t.assistant("No, entanglement cannot transmit information faster than light.")

    # Walk reasoning commits
    reasoning_entries = [e for e in t.log() if e.content_type == "reasoning"]
    for entry in reasoning_entries:
        content = (entry.content_text or "")[:60].replace("\n", " ")
        print(f"  {entry.commit_hash[:8]} [{entry.content_type}] {content}...")
        choice = click.prompt(
            "    Change visibility? (NORMAL/SKIP/PINNED, Enter to skip)",
            type=click.Choice(["NORMAL", "SKIP", "PINNED", ""], case_sensitive=False),
            default="",
            show_default=False,
        )
        if choice:
            t.annotate(entry.commit_hash, Priority[choice])
            print(f"    -> set to {choice}")
        else:
            print(f"    -> kept default")

    # Show final compiled context
    print()
    ctx = t.compile(include_reasoning=True)
    print(f"  Final compiled: {ctx.commit_count} messages, {ctx.token_count} tokens")
    ctx.pprint(style="compact")

    print()
    t.close()


# =============================================================================
# Part 3: Agent-Driven Reasoning Annotation  (PART 3 — LLM / Agent)
# =============================================================================

def part3_agent():
    print("=" * 60)
    print("Part 3: AGENT-DRIVEN REASONING ANNOTATION  [Agent Tier]")
    print("=" * 60)
    print()
    print("  An agent manages reasoning visibility via ToolExecutor.")
    print()

    t = Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    )

    t.system("You are a helpful assistant.")
    t.user("What causes tides?")
    r_info = t.reasoning("Tides are caused by gravitational pull of the moon and sun.")
    t.assistant("Tides are primarily caused by the moon's gravitational pull.")

    # Agent uses ToolExecutor to mark reasoning as SKIP
    executor = ToolExecutor(t)
    result = executor.execute("annotate", {
        "commit_hash": r_info.commit_hash,
        "priority": "SKIP",
    })
    print(f"  executor.execute('annotate', SKIP) -> success={result.success}")
    print(f"  Reasoning commit {r_info.commit_hash[:8]} is now hidden from compile().")
    print()

    t.close()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_compile_control()
    part2_interactive()
    part3_agent()
    print("=" * 60)
    print("Done -- all 3 tiers of compile control demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
