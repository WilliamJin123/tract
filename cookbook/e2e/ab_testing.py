"""A/B testing: branch, compare configs, pick winner.

  PART 1 -- Manual:      Branch, chat with different configs, diff() + query_by_config()
  PART 3 -- LLM / Agent:  Orchestrator compares branches, merges winner
"""

import sys
from pathlib import Path

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: branch + config comparison + diff
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Branch and Compare Configs")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Seed shared context
        t.system("You are a science communicator. Explain concepts clearly.")
        t.user("Explain how black holes form.")
        baseline_hash = t.head

        # Variant A: low temperature (precise, focused)
        t.branch("variant-a")
        t.switch("variant-a")
        resp_a = t.chat("Explain the event horizon of a black hole.",
                        temperature=0.2)
        hash_a = t.head
        print(f"\n  Variant A (temp=0.2): {resp_a.text[:80]}...")

        # Variant B: high temperature (creative, exploratory)
        t.switch("main")
        t.branch("variant-b")
        t.switch("variant-b")
        resp_b = t.chat("Explain the event horizon of a black hole.",
                        temperature=0.9)
        hash_b = t.head
        print(f"  Variant B (temp=0.9): {resp_b.text[:80]}...")

        # Compare with diff
        t.switch("main")
        diff_result = t.diff(hash_a, hash_b)
        print(f"\n  Diff stats: {diff_result.stat}")

        # Query by config to find commits with specific temperature
        low_temp = t.query_by_config("temperature", "=", 0.2)
        high_temp = t.query_by_config("temperature", "=", 0.9)
        print(f"  Commits with temp=0.2: {len(low_temp)}")
        print(f"  Commits with temp=0.9: {len(high_temp)}")


# =====================================================================
# PART 3 -- LLM / Agent: automated comparison
# =====================================================================

def part3_agent():
    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Automated A/B Comparison")
    print("=" * 60)

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are an expert technical writer.")
        t.user("Explain quantum entanglement.")

        # Run both variants
        t.branch("formal")
        t.switch("formal")
        resp_formal = t.chat("Elaborate on the EPR paradox.",
                             temperature=0.2)

        t.switch("main")
        t.branch("casual")
        t.switch("casual")
        resp_casual = t.chat("Elaborate on the EPR paradox.",
                             temperature=0.8)

        # Use LLM to judge which is better
        t.switch("main")
        t.user(f"Compare these two explanations of the EPR paradox:\n\n"
               f"A (formal): {resp_formal.text[:300]}\n\n"
               f"B (casual): {resp_casual.text[:300]}\n\n"
               "Which is more clear and accurate? Reply with just 'A' or 'B'.")
        judge = t.generate()
        winner_text = judge.text.strip().upper()

        if "A" in winner_text:
            winner_branch = "formal"
        else:
            winner_branch = "casual"

        result = t.merge(winner_branch)
        print(f"\n  LLM judged: '{winner_branch}' is better")
        print(f"  Merged: {result.merge_type}")
        print(f"  Final commits: {len(t.log())}")


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()
