"""Manual Reasoning Commits

Three tiers of reasoning management: manual commits, interactive
confirmation, and agent-driven reasoning via generate().

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: t.reasoning(), log(), compile(), get_content(), get_metadata(),
              click.confirm(), compile(include_reasoning=True)
"""

import click

from tract import Tract


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
# Part 2: Interactive Reasoning Toggle  (PART 2 — Interactive)
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("Part 2: INTERACTIVE REASONING TOGGLE  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  Compile with reasoning included, then let the user decide")
    print("  whether to include reasoning in the next compile.")
    print()

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Explain gravity.")
    t.reasoning(
        "Gravity is the fundamental force of attraction between masses. "
        "I should mention Newton's law and Einstein's general relativity."
    )
    t.assistant("Gravity is the force of attraction between objects with mass.")

    # Show with reasoning
    ctx_with = t.compile(include_reasoning=True)
    print(f"  With reasoning: {ctx_with.commit_count} messages, "
          f"{ctx_with.token_count} tokens")
    ctx_with.pprint(style="chat")

    if click.confirm("\n  Include reasoning in next compile?", default=False):
        ctx = t.compile(include_reasoning=True)
    else:
        ctx = t.compile()

    print(f"\n  Your choice: {ctx.commit_count} messages, {ctx.token_count} tokens")
    print()

    t.close()


# =============================================================================
# Part 3: Agent Reasoning  (PART 3 — LLM / Agent)
# =============================================================================

def part3_agent_note():
    print("=" * 60)
    print("Part 3: AGENT REASONING  [Agent Tier — Note]")
    print("=" * 60)
    print()
    print("  See 04_llm_integration.py for generate() auto-extracting")
    print("  reasoning from LLM responses. The agent generates reasoning")
    print("  automatically via generate(reasoning_effort='high').")
    print()
    print("  Key patterns:")
    print("    resp = t.generate(reasoning_effort='high')")
    print("    resp.reasoning         # extracted text")
    print("    resp.reasoning_commit  # auto-committed CommitInfo")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_manual_reasoning()
    part2_interactive()
    part3_agent_note()
    print("=" * 60)
    print("Done -- all 3 tiers of reasoning management demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
