"""Collaborative Multi-Agent Research with Consensus

Three specialist agents evaluate a technology decision in parallel, each
bringing a distinct perspective. A coordinator synthesizes their findings
into a final recommendation with consensus analysis.

Agents:
  Performance Analyst   -- benchmarks, scalability, resource usage
  DevEx Analyst         -- API design, documentation, ecosystem
  Business Analyst      -- licensing, vendor lock-in, total cost of ownership

Flow:
  1. Session creates a coordinator tract with the research question
  2. Spawn 3 child tracts (one per specialist)
  3. Each child does independent research with role-specific config/directives
  4. Collapse each child back to the coordinator with a summary
  5. Coordinator synthesizes: consensus, disagreements, final recommendation
  6. Tag the final recommendation for easy retrieval

Demonstrates: Session.spawn(), Session.collapse(), per-child configuration,
              per-child directives, tag-based finding classification,
              metadata for confidence scores, DAG-native multi-agent research
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Session, TractConfig, TokenBudgetConfig, OpenAIClient, LLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# =====================================================================
# Analyst definitions
# =====================================================================

ANALYSTS = [
    {
        "name": "performance",
        "title": "Performance Analyst",
        "temperature": 0.3,
        "directive": (
            "You are a performance engineer. Focus on benchmarks, scalability,\n"
            "resource usage, and latency. Ignore devex and business concerns."
        ),
        "prompt": (
            "Evaluate PostgreSQL vs MongoDB for a high-traffic e-commerce "
            "platform (50K req/sec, 500M records). Cover read/write throughput, "
            "scalability approach, resource usage, and latency. "
            "Tag findings as 'pro', 'con', or 'risk'."
        ),
    },
    {
        "name": "devex",
        "title": "Developer Experience Analyst",
        "temperature": 0.5,
        "directive": (
            "You are a developer experience specialist. Focus on API design,\n"
            "documentation, ecosystem maturity, and productivity. Ignore raw\n"
            "performance and business costs."
        ),
        "prompt": (
            "Evaluate PostgreSQL vs MongoDB for developer experience on a "
            "40+ person team. Cover query expressiveness, schema management, "
            "ecosystem tooling, and onboarding curve. "
            "Tag findings as 'pro', 'con', or 'risk'."
        ),
    },
    {
        "name": "business",
        "title": "Business Analyst",
        "temperature": 0.4,
        "directive": (
            "You are a business analyst. Focus on licensing, vendor lock-in,\n"
            "talent availability, and long-term support. Ignore technical\n"
            "performance and API design."
        ),
        "prompt": (
            "Evaluate PostgreSQL vs MongoDB for a Series B startup planning "
            "5-year growth. Cover licensing costs, vendor lock-in, hiring "
            "market, and enterprise support. "
            "Tag findings as 'pro', 'con', or 'opportunity'."
        ),
    },
]


# =====================================================================
# Helper: configure a tract with LLM
# =====================================================================

def _configure_llm(tract, *, temperature: float = 0.5) -> None:
    """Wire up LLM client on a tract (coordinator or spawned child).

    Session.create_tract() and Session.spawn() don't accept LLM params,
    so we must configure the client after creation. The _llm_state access
    is an internal API -- no public equivalent exists yet for setting
    ownership, default config, or auto-message on a session-created tract.
    """
    client = llm.client(model=MODEL_ID)
    tract.config.configure_llm(client)
    # HACK: internal API -- no public setter for these on session-created tracts
    tract._llm_state.owns_llm_client = True
    tract._llm_state.default_config = LLMConfig(model=MODEL_ID, temperature=temperature)
    tract._llm_state.auto_message_enabled = True
    tract.config.configure_operations(message=LLMConfig(model=llm.small, temperature=0.0))


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    log = StepLogger()

    # 1. Setup: Session + Coordinator
    session = Session.open()
    coordinator = session.create_tract(
        display_name="coordinator",
        config=TractConfig(token_budget=TokenBudgetConfig(max_tokens=4000)),
    )
    _configure_llm(coordinator, temperature=0.3)

    coordinator.system(
        "You are a technical research coordinator. You synthesize findings "
        "from multiple specialist analysts into actionable recommendations."
    )
    coordinator.user(
        "We need to choose between PostgreSQL and MongoDB for our new "
        "e-commerce platform. Three analysts will evaluate from different "
        "angles: performance, developer experience, and business."
    )
    coordinator.assistant(
        "I'll coordinate the evaluation. Each analyst will investigate "
        "independently, then I'll synthesize their findings."
    )

    for tag_name in ["pro", "con", "risk", "opportunity",
                     "performance", "devex", "business",
                     "consensus", "disagreement", "recommendation"]:
        coordinator.register_tag(tag_name)

    print("Coordinator created, 10 research tags registered")

    # 2. Spawn specialist agents
    children = {}
    for analyst in ANALYSTS:
        child = session.spawn(
            coordinator,
            purpose=f"{analyst['title']}: evaluate PostgreSQL vs MongoDB",
            display_name=analyst["name"],
            directives={f"role-{analyst['name']}": analyst["directive"]},
            configure={
                "stage": "research",
                "analyst_role": analyst["name"],
                "temperature": analyst["temperature"],
            },
        )
        _configure_llm(child, temperature=analyst["temperature"])
        children[analyst["name"]] = {"tract": child, "analyst": analyst}
        print(f"  Spawned: {analyst['title']} (temp={analyst['temperature']})")

    # 3. Research: each child does independent analysis
    print("\n--- Independent Research Phase ---")

    for name, entry in children.items():
        child = entry["tract"]
        analyst = entry["analyst"]

        response = child.llm.chat(analyst["prompt"], max_tokens=600)
        child.tag(response.commit_info.commit_hash, analyst["name"])

        preview = (response.text or "(no response)")[:200].replace("\n", "\n    ")
        status = child.status()
        print(f"\n  {analyst['title']}: {status.token_count} tokens")
        print(f"    {preview}...")

    # 4. Collapse: bring each child's findings back to coordinator
    print("\n--- Collapse Phase ---")

    collapse_results = {}
    for name, entry in children.items():
        child = entry["tract"]
        analyst = entry["analyst"]

        summary_response = child.llm.chat(
            f"Summarize your findings as the {analyst['title']}. "
            f"Bullet points, label PRO/CON/RISK, under 200 words.",
            max_tokens=400,
        )

        result = session.collapse(
            child,
            into=coordinator,
            content=f"[{analyst['title']} Summary]\n\n{summary_response.text}",
            auto_commit=True,
        )
        collapse_results[name] = result

        if result.parent_commit_hash:
            coordinator.tag(result.parent_commit_hash, analyst["name"])

        print(f"  {analyst['title']}: "
              f"{result.source_tokens} -> {result.summary_tokens} tokens")

    # 5. Synthesis: coordinator identifies consensus + disagreements
    print("\n--- Synthesis ---")

    response = coordinator.llm.chat(
        "All three analyst reports are now in your context. Produce a "
        "FINAL RECOMMENDATION with these sections:\n\n"
        "1. CONSENSUS -- Points where all analysts agree\n"
        "2. DISAGREEMENTS -- Points where analysts contradict each other\n"
        "3. RECOMMENDATION -- Final choice with confidence (high/medium/low)\n"
        "4. KEY RISKS -- Top 3 risks regardless of choice\n\n"
        "Be concise. Reference which analyst supports each point.",
        max_tokens=800,
    )

    print("\n  Final Recommendation:\n")
    for line in response.text.split("\n"):
        print(f"    {line}")

    # 6. Tag the recommendation
    coordinator.tag(response.commit_info.commit_hash, "recommendation")
    coordinator.tag(response.commit_info.commit_hash, "consensus")

    # Final state
    print("\n--- Final State ---")
    status = coordinator.status()
    print(f"  Coordinator: {status.token_count} tokens, "
          f"{status.commit_count} commits")

    tracts = session.list_tracts()
    print(f"  Session tracts: {len(tracts)}")

    print(f"\n  Coordinator log (last 8):")
    for ci in coordinator.log()[-8:]:
        tags_str = f" [{', '.join(ci.tags)}]" if ci.tags else ""
        msg = (ci.message or "")[:45]
        print(f"    {ci.commit_hash[:8]}  {ci.content_type:12s}{tags_str}  {msg}")

    session.close()


if __name__ == "__main__":
    main()
