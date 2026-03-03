"""Research delegation: parallel sub-agents with merge.

  PART 1 -- Manual:      Deploy 3 children, each compresses, parent merges
  PART 3 -- LLM / Agent:  Full session deploy + compress + collapse
"""

import sys
from pathlib import Path

from tract import Session
from tract.models.config import LLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


def _configure_llm(tract):
    """Configure LLM client on a session-created tract."""
    from tract.llm.client import OpenAIClient

    client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url or None,
        default_model=MODEL_ID,
    )
    tract.configure_llm(client)
    tract._owns_llm_client = True
    tract._default_config = LLMConfig(model=MODEL_ID)


# =====================================================================
# PART 1 -- Manual: 3 children research, compress, merge
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Parallel Research Delegation")
    print("=" * 60)

    session = Session.open()
    parent = session.create_tract(display_name="research-lead")

    parent.system("You are a research coordinator studying stellar evolution.")
    parent.user("We need reports on three stellar lifecycle stages.")
    parent.assistant("I will delegate each stage to a specialist.")

    # Deploy 3 children, each on a separate branch
    topics = {
        "child-protostar": [
            ("How do protostars form?", "Protostars form from collapsing molecular clouds. Gravity pulls gas and dust inward, forming a dense core that heats up."),
            ("What triggers the collapse?", "Shockwaves from supernovae or cloud-cloud collisions can trigger gravitational collapse in dense regions."),
            ("How long is the protostar phase?", "The protostar phase lasts roughly 100,000 to 1 million years before nuclear fusion ignites."),
        ],
        "child-main-seq": [
            ("What defines main-sequence stars?", "Main-sequence stars fuse hydrogen into helium in their cores. They follow the mass-luminosity relation on the HR diagram."),
            ("How long do they last?", "Lifespan depends on mass: our Sun has ~10 billion years; massive stars burn out in millions of years."),
            ("What is the pp chain?", "The proton-proton chain fuses hydrogen nuclei into helium-4, releasing energy as gamma rays and neutrinos."),
        ],
        "child-remnant": [
            ("What are stellar remnants?", "After exhausting fuel, stars become white dwarfs, neutron stars, or black holes depending on initial mass."),
            ("What is the Chandrasekhar limit?", "The Chandrasekhar limit (~1.4 solar masses) is the maximum mass for a stable white dwarf."),
            ("How do pulsars work?", "Pulsars are rapidly rotating neutron stars that emit beams of radiation from their magnetic poles."),
        ],
    }

    summaries = {
        "child-protostar": "Protostars: form from molecular cloud collapse triggered by shockwaves. Phase lasts 100K-1M years before fusion ignites.",
        "child-main-seq": "Main-sequence: hydrogen fusion via pp chain. Lifespan inversely proportional to mass. Sun ~10 Gyr, massive stars ~few Myr.",
        "child-remnant": "Remnants: white dwarfs (<1.4 Msun), neutron stars/pulsars, or black holes. Chandrasekhar limit governs WD stability.",
    }

    children = {}
    for branch_name, qa_pairs in topics.items():
        child = session.deploy(parent, purpose=branch_name, branch_name=branch_name)
        child._seed_base_tags()
        for q, a in qa_pairs:
            child.user(q)
            child.assistant(a)
        child.compress(content=summaries[branch_name])
        children[branch_name] = child
        print(f"\n  {branch_name}: {len(child.log())} commits -> compressed")

    # Parent merges each child
    for branch_name in children:
        result = parent.merge(branch_name)
        print(f"  Merged '{branch_name}': {result.merge_type}")

    print(f"\n  Parent final: {len(parent.log())} commits")
    ctx = parent.compile()
    print(f"  Compiled: {len(ctx.messages)} messages, {ctx.token_count} tokens")

    session.close()


# =====================================================================
# PART 3 -- LLM / Agent: deploy + compress + collapse
# =====================================================================

def part3_agent():
    if not llm.api_key:
        print("\n" + "=" * 60)
        print("PART 3: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Deploy-Compress-Collapse")
    print("=" * 60)

    session = Session.open()
    parent = session.create_tract(display_name="orchestrator")
    _configure_llm(parent)

    parent.system("You are a particle physics research coordinator.")
    parent.user("We need summaries of three fundamental forces.")

    research = [
        ("force-em", "Describe the electromagnetic force in 2-3 sentences: carrier particle, range, and role in nature."),
        ("force-strong", "Describe the strong nuclear force in 2-3 sentences: carrier particle, range, and role in nature."),
        ("force-weak", "Describe the weak nuclear force in 2-3 sentences: carrier particle, range, and role in nature."),
    ]

    for branch_name, question in research:
        child = session.deploy(parent, purpose=branch_name, branch_name=branch_name)
        child._seed_base_tags()
        _configure_llm(child)

        # Real LLM call: child researches the topic
        response = child.chat(question)
        summary = response.text[:200]
        print(f"\n  {branch_name}: LLM responded ({len(response.text)} chars)")

        # Collapse child into parent with LLM-generated summary
        session.collapse(
            child, into=parent,
            content=summary,
            auto_commit=True,
        )
        print(f"  Collapsed '{branch_name}': {summary[:60]}...")

    ctx = parent.compile()
    print(f"\n  Parent context: {len(ctx.messages)} messages, {ctx.token_count} tokens")
    print(f"  All three research branches collapsed into parent.")

    session.close()


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()
