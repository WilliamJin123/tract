"""Formatting

Three tiers: manual pprint styles, interactive toggle, and format output
for API consumption.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: pprint() reasoning style (table, chat, compact),
              compile(include_reasoning=True), click.confirm(),
              to_dicts(), to_openai()
"""

import click

from tract import Tract


def part1_formatting():
    print(f"\n{'=' * 60}")
    print("Part 1: FORMATTING  [Manual Tier]")
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
# Part 2: Interactive Reasoning Toggle  (PART 2 — Interactive)
# =============================================================================

def part2_interactive():
    print("=" * 60)
    print("Part 2: INTERACTIVE REASONING TOGGLE  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  Choose whether to show reasoning traces in the output.\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What is the speed of light?")
    t.reasoning("The speed of light in vacuum is approximately 3 x 10^8 m/s.")
    t.assistant("The speed of light is approximately 299,792,458 meters per second.")

    if click.confirm("  Show reasoning traces?", default=True):
        ctx = t.compile(include_reasoning=True)
    else:
        ctx = t.compile()

    print()
    ctx.pprint()

    t.close()


# =============================================================================
# Part 3: Format Output for APIs  (PART 3 — LLM / Agent)
# =============================================================================

def part3_format_output():
    print("=" * 60)
    print("Part 3: FORMAT OUTPUT FOR APIS  [Agent Tier]")
    print("=" * 60)
    print()
    print("  to_dicts() and to_openai() include reasoning when compiled")
    print("  with include_reasoning=True. This is what agents consume.\n")

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("What is 2 + 2?")
    t.reasoning("Simple arithmetic: 2 + 2 = 4.")
    t.assistant("2 + 2 = 4.")

    # Without reasoning
    ctx_no = t.compile()
    dicts_no = ctx_no.to_dicts()
    print(f"  to_dicts() without reasoning: {len(dicts_no)} messages")
    for d in dicts_no:
        print(f"    [{d['role']}] {d['content'][:50]}")

    # With reasoning
    ctx_yes = t.compile(include_reasoning=True)
    dicts_yes = ctx_yes.to_dicts()
    print(f"\n  to_dicts() with reasoning: {len(dicts_yes)} messages")
    for d in dicts_yes:
        print(f"    [{d['role']}] {d['content'][:50]}")

    print()
    t.close()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_formatting()
    part2_interactive()
    part3_format_output()
    print("=" * 60)
    print("Done -- all 3 tiers of formatting demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
