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

Why this is SUPERIOR to naive "call LLM 3 times and concatenate":
  - Each analyst has isolated context (no cross-contamination of perspectives)
  - Each analyst can use different temperature/directives for their role
  - The coordinator sees structured summaries, not raw chat dumps
  - The full research history is preserved (expand any collapse for audit)
  - Tags enable querying specific finding types across all analysts
  - Token budget is managed per-agent (no single massive prompt)
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
        "temperature": 0.3,  # Low: precise, factual analysis
        "directive": (
            "You are a performance engineer. Focus ONLY on:\n"
            "- Benchmark data and throughput characteristics\n"
            "- Scalability limits and bottlenecks\n"
            "- Memory and CPU resource usage patterns\n"
            "- Latency profiles under load\n"
            "Do NOT discuss developer experience, cost, or business concerns."
        ),
        "prompt": (
            "Evaluate PostgreSQL vs MongoDB for a high-traffic e-commerce "
            "platform (50K req/sec, 500M product records). Analyze:\n"
            "1. Read/write throughput for product catalog queries\n"
            "2. Scalability approach (vertical vs horizontal)\n"
            "3. Resource usage at scale (memory, CPU, storage)\n"
            "4. Latency characteristics for mixed workloads\n\n"
            "Tag your key findings as 'pro', 'con', or 'risk'. "
            "Be specific with numbers where possible."
        ),
    },
    {
        "name": "devex",
        "title": "Developer Experience Analyst",
        "temperature": 0.5,  # Moderate: balanced evaluation
        "directive": (
            "You are a developer experience specialist. Focus ONLY on:\n"
            "- API design and query expressiveness\n"
            "- Documentation quality and learning curve\n"
            "- Ecosystem maturity (ORMs, drivers, tooling)\n"
            "- Developer productivity and debugging experience\n"
            "Do NOT discuss raw performance numbers or business costs."
        ),
        "prompt": (
            "Evaluate PostgreSQL vs MongoDB for developer experience on a "
            "large engineering team (40+ developers, mixed experience). "
            "Analyze:\n"
            "1. Query language expressiveness and flexibility\n"
            "2. Schema management and migration workflows\n"
            "3. Ecosystem (ORMs, admin tools, monitoring)\n"
            "4. Onboarding time and learning curve\n\n"
            "Tag your key findings as 'pro', 'con', or 'risk'. "
            "Consider both junior and senior developer perspectives."
        ),
    },
    {
        "name": "business",
        "title": "Business Analyst",
        "temperature": 0.4,  # Lower: cost analysis should be precise
        "directive": (
            "You are a business and procurement analyst. Focus ONLY on:\n"
            "- Licensing models and total cost of ownership\n"
            "- Vendor lock-in risk and exit strategies\n"
            "- Talent availability and hiring market\n"
            "- Long-term support and enterprise readiness\n"
            "Do NOT discuss technical performance or API design."
        ),
        "prompt": (
            "Evaluate PostgreSQL vs MongoDB from a business perspective for "
            "a Series B startup planning for 5-year growth. Analyze:\n"
            "1. Licensing costs (self-hosted vs managed service)\n"
            "2. Vendor lock-in and data portability\n"
            "3. Hiring: talent pool and salary expectations\n"
            "4. Enterprise support options and SLAs\n\n"
            "Tag your key findings as 'pro', 'con', or 'opportunity'. "
            "Include cost estimates where possible."
        ),
    },
]


# =====================================================================
# Helper: configure a child tract with LLM
# =====================================================================

def _configure_child(child: "Tract", *, temperature: float = 0.5) -> None:
    """Wire up LLM client on a spawned child tract."""
    client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url,
        default_model=MODEL_ID,
    )
    child.config.configure_llm(client)
    child._llm_state.owns_llm_client = True
    child._llm_state.default_config = LLMConfig(model=MODEL_ID, temperature=temperature)
    child._llm_state.auto_message_enabled = True
    child.config.configure_operations(message=LLMConfig(model=llm.small, temperature=0.0))


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    print("=" * 70)
    print("Collaborative Multi-Agent Research with Consensus")
    print("=" * 70)
    print()
    print("  3 specialist agents evaluate PostgreSQL vs MongoDB.")
    print("  Each has isolated context, role-specific config, and directives.")
    print("  Coordinator synthesizes consensus from collapsed summaries.")
    print()

    log = StepLogger()

    # =================================================================
    # 1. Setup: Session + Coordinator tract
    # =================================================================

    print("=== Setup: Session + Coordinator ===\n")

    session = Session.open()
    coordinator = session.create_tract(
        display_name="coordinator",
        config=TractConfig(token_budget=TokenBudgetConfig(max_tokens=4000)),
    )

    # Configure coordinator LLM
    coord_client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url,
        default_model=MODEL_ID,
    )
    coordinator.config.configure_llm(coord_client)
    coordinator._llm_state.owns_llm_client = True
    coordinator._llm_state.default_config = LLMConfig(model=MODEL_ID, temperature=0.3)
    coordinator._llm_state.auto_message_enabled = True
    coordinator.config.configure_operations(message=LLMConfig(model=llm.small, temperature=0.0))

    # Seed the research question
    coordinator.system(
        "You are a technical research coordinator. You synthesize findings "
        "from multiple specialist analysts into actionable recommendations."
    )
    coordinator.user(
        "We need to choose between PostgreSQL and MongoDB for our new "
        "e-commerce platform. Three analysts are evaluating this from "
        "different angles: performance, developer experience, and business. "
        "You will receive their findings and produce a final recommendation."
    )
    coordinator.assistant(
        "I'll coordinate the evaluation. Each analyst will investigate "
        "independently, then I'll synthesize their findings into a "
        "consensus recommendation."
    )

    # Register tags for finding classification
    for tag_name in ["pro", "con", "risk", "opportunity",
                     "performance", "devex", "business",
                     "consensus", "disagreement", "recommendation"]:
        coordinator.tags.register(tag_name)

    print(f"  Coordinator tract created")
    print(f"  Registered 10 research tags")
    coordinator.compile().pprint(style="compact")

    # =================================================================
    # 2. Spawn specialist agents
    # =================================================================

    print("\n=== Spawning 3 Specialist Agents ===\n")

    children = {}
    for analyst in ANALYSTS:
        # spawn-with-persona: directives + config applied in one call
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
        _configure_child(child, temperature=analyst["temperature"])

        children[analyst["name"]] = {
            "tract": child,
            "analyst": analyst,
        }
        print(f"  Spawned: {analyst['title']} "
              f"(temp={analyst['temperature']})")

    # =================================================================
    # 3. Research: each child does independent analysis via t.llm.chat()
    # =================================================================

    print("\n=== Independent Research Phase ===\n")

    for name, entry in children.items():
        child = entry["tract"]
        analyst = entry["analyst"]

        print(f"\n--- {analyst['title']} ---\n")

        # Each child gets its own research prompt
        response = child.llm.chat(
            analyst["prompt"],
            max_tokens=600,
        )

        # Show a preview of findings
        preview = (response.text or "(no response)")[:300].replace("\n", "\n    ")
        print(f"  Findings preview:\n    {preview}...")

        # Tag the response commit with the analyst's domain
        child.tags.add(response.commit_info.commit_hash, analyst["name"])

        # Show child status
        status = child.search.status()
        print(f"\n  Tokens: {status.token_count}, "
              f"Commits: {status.commit_count}")

    # =================================================================
    # 4. Collapse: bring each child's findings back to coordinator
    # =================================================================

    print("\n\n=== Collapse Phase: Summarize Findings ===\n")

    # Each child summarizes its own findings, then we collapse the summary
    # into the coordinator. This keeps each analyst's voice in the summary
    # and avoids a single LLM trying to re-interpret all three perspectives.

    collapse_results = {}
    for name, entry in children.items():
        child = entry["tract"]
        analyst = entry["analyst"]

        # Child generates its own executive summary
        summary_response = child.llm.chat(
            f"Summarize your findings as the {analyst['title']}. "
            f"Use bullet points. Label each finding as PRO, CON, or RISK. "
            f"Keep it under 200 words. Start with your top-line verdict.",
            max_tokens=400,
        )

        # Collapse with manual content (child's self-summary)
        result = session.collapse(
            child,
            into=coordinator,
            content=f"[{analyst['title']} Summary]\n\n{summary_response.text}",
            auto_commit=True,
        )

        collapse_results[name] = result

        # Tag the collapse commit in the coordinator
        if result.parent_commit_hash:
            coordinator.tags.add(result.parent_commit_hash, analyst["name"])

        print(f"  {analyst['title']}: "
              f"{result.source_tokens} tokens -> "
              f"{result.summary_tokens} token summary")

        # Show collapse summary preview
        preview = result.summary_text[:200].replace("\n", "\n    ")
        print(f"    Summary: {preview}...")
        print()

    # =================================================================
    # 5. Synthesis: coordinator identifies consensus + disagreements
    # =================================================================

    print("\n=== Synthesis: Consensus Analysis ===\n")

    response = coordinator.llm.chat(
        "All three analyst reports are now in your context. Produce a "
        "FINAL RECOMMENDATION with these sections:\n\n"
        "1. CONSENSUS -- Points where all analysts agree\n"
        "2. DISAGREEMENTS -- Points where analysts contradict each other\n"
        "3. RECOMMENDATION -- Your final choice with confidence level "
        "(high/medium/low) and a one-paragraph justification\n"
        "4. KEY RISKS -- Top 3 risks regardless of which option is chosen\n\n"
        "Be concise. Reference which analyst supports each point.",
        max_tokens=800,
    )

    print("  Final Recommendation:")
    print()
    # Print the full synthesis (this is the key output)
    for line in response.text.split("\n"):
        print(f"    {line}")

    # =================================================================
    # 6. Tag & persist the recommendation
    # =================================================================

    print("\n\n=== Tag & Persist ===\n")

    # Tag the synthesis commit
    coordinator.tags.add(response.commit_info.commit_hash, "recommendation")
    coordinator.tags.add(response.commit_info.commit_hash, "consensus")

    # Store confidence metadata on the synthesis commit
    # (metadata is set at commit time, so we record it as a follow-up note)
    coordinator.user(
        "Recommendation committed and tagged.",
        metadata={
            "type": "recommendation_receipt",
            "analysts_collapsed": list(children.keys()),
            "collapse_summaries": {
                name: res.summary_tokens
                for name, res in collapse_results.items()
            },
        },
        tags=["recommendation"],
    )

    # =================================================================
    # Report: final state
    # =================================================================

    print("=== Final State ===\n")

    # Coordinator status
    status = coordinator.search.status()
    print(f"  Coordinator: {status.token_count} tokens, "
          f"{status.commit_count} commits")
    print(f"  Branch: {status.branch_name}")

    # List all tracts in session
    tracts = session.list_tracts()
    print(f"\n  Session tracts: {len(tracts)}")
    for t_info in tracts:
        print(f"    - {t_info.get('display_name', t_info.get('tract_id', '?'))}")

    # Show tag distribution (immutable auto-tags + registered tags)
    print(f"\n  Tag registry:")
    for entry in coordinator.tags.list():
        print(f"    {entry['name']:20s} count={entry['count']}")

    # Show the commit log
    print(f"\n  Coordinator log (last 10):")
    for ci in coordinator.search.log()[-10:]:
        tags_str = f" [{', '.join(ci.tags)}]" if ci.tags else ""
        msg = (ci.message or "")[:45]
        print(f"    {ci.commit_hash[:8]}  {ci.content_type:12s}{tags_str}  {msg}")

    # Show compiled context
    print(f"\n  Final compiled context:")
    coordinator.compile().pprint(style="compact")

    # =================================================================
    # Why this pattern is superior
    # =================================================================

    print("\n" + "=" * 70)
    print("WHY TRACT'S APPROACH IS SUPERIOR")
    print("=" * 70)
    print("""
  Naive approach (3 LLM calls + concatenate):
    - All analysts see the same prompt (cross-contamination)
    - No way to give each analyst different temperature/behavior
    - Raw outputs concatenated = noisy, unstructured input to synthesis
    - No audit trail -- cannot trace which analyst said what
    - Token budget is one big blob -- no per-analyst management

  Tract's Session + Spawn model:
    - Isolated context per analyst (no cross-contamination)
    - Per-child configure() and directive() for role-specific behavior
    - Collapse produces structured summaries (not raw dumps)
    - Full DAG history preserved -- expand any collapse for audit
    - Tag system enables cross-analyst queries (e.g., all "risk" findings)
    - Token budget managed per-agent with compression if needed
    - Session.list_tracts() shows the full research topology
""")

    session.close()


if __name__ == "__main__":
    main()
