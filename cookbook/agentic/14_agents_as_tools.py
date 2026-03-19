"""Agents as Tools: Stateless subagent invocations

Treat spawned child tracts like tool calls -- each child is a single-use
specialist that receives a prompt, produces a result, and gets collapsed
back into the parent's context as if it were a tool output. No persistent
child state; the parent continues with the specialist's analysis inlined.

Mental model: spawn + single chat + collapse = LLM-powered tool call
where the "tool" has its own persona, temperature, and directive set.

Sections:
  1. Subagent-as-Tool Pattern
     Parent works on a system design task. At key decision points, it
     invokes specialist subagents (storage analyst, security reviewer,
     cost estimator) and folds their output back into its own context.

  2. Parallel Subagent Fan-Out
     Parent needs three independent analyses done simultaneously.
     Spawn three subagents, each with a focused directive. Collapse all
     results back, then synthesize.

Demonstrates: Session.spawn(), Session.collapse(), per-child directives,
              per-child temperature, stateless single-shot child usage,
              agent-as-tool abstraction over DAG primitives

Requires: LLM API key (uses Claude Code provider)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Session, TractConfig, TokenBudgetConfig, LLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# =====================================================================
# Helper: configure LLM on a session-created tract
# =====================================================================

def _configure_llm(tract, *, temperature: float = 0.5) -> None:
    """Wire up LLM client on a session-created tract.

    Session.create_tract() and Session.spawn() don't accept LLM params,
    so we configure after creation. The _llm_state access is an internal
    API -- no public equivalent exists yet.
    """
    client = llm.client(model=MODEL_ID)
    tract.config.configure_llm(client)
    # HACK: internal API -- no public setter for these on session-created tracts
    tract._llm_state.owns_llm_client = True
    tract._llm_state.default_config = LLMConfig(model=MODEL_ID, temperature=temperature)
    tract._llm_state.auto_message_enabled = True
    tract.config.configure_operations(message=LLMConfig(model=llm.small, temperature=0.0))


# =====================================================================
# Core abstraction: agent_tool()
# =====================================================================

def agent_tool(session, parent, *, purpose, directive, prompt, temperature=0.3):
    """Invoke a stateless subagent and collapse its output back to parent.

    This is the tract equivalent of a tool call where the "tool" is an LLM
    with a different persona/temperature/directive. The child tract is
    spawned, executes a single chat, and its response is collapsed into the
    parent as a committed message -- indistinguishable from a tool result.

    Args:
        session: The active Session.
        parent: The parent Tract receiving the result.
        purpose: Short label for this subagent (appears in collapsed output).
        directive: System-level instruction for the subagent's persona.
        prompt: The single question/task for the subagent.
        temperature: LLM temperature for the subagent (default 0.3).

    Returns:
        CollapseResult with summary details.
    """
    child = session.spawn(
        parent,
        purpose=purpose,
        display_name=purpose.lower().replace(" ", "_"),
        directives={"role": directive},
        configure={"temperature": temperature},
    )
    _configure_llm(child, temperature=temperature)

    response = child.llm.chat(prompt, max_tokens=500)

    result = session.collapse(
        child,
        into=parent,
        content=f"[{purpose}]\n\n{response.text}",
        auto_commit=True,
    )
    return result


# =====================================================================
# Section 1: Subagent-as-Tool Pattern
# =====================================================================

def section_1_subagent_as_tool() -> None:
    print("=" * 60)
    print("  Section 1: Subagent-as-Tool Pattern")
    print("=" * 60)
    print()
    print("  Parent is designing a system. At decision points, it invokes")
    print("  specialist subagents like tool calls: spawn, chat, collapse.")
    print()

    session = Session.open()
    coordinator = session.create_tract(
        display_name="architect",
        config=TractConfig(token_budget=TokenBudgetConfig(max_tokens=6000)),
    )
    _configure_llm(coordinator, temperature=0.5)

    coordinator.system(
        "You are a senior systems architect designing a high-traffic "
        "event processing platform. You have access to specialist analysts "
        "whose findings appear inline in the conversation."
    )
    coordinator.user(
        "Design an event processing platform with these requirements:\n"
        "- Ingest 100K events/second from IoT sensors\n"
        "- Store raw events for 1 year\n"
        "- Real-time aggregation dashboards (p50/p99 latency)\n"
        "- Alert on anomaly detection thresholds\n"
        "- Multi-tenant, 50 customers, shared infrastructure",
        message="requirements: event processing platform",
    )

    # Architect produces initial design via LLM
    print("  Architect drafting initial design...")
    design_response = coordinator.llm.chat(
        "Propose a high-level architecture for this platform. Cover: "
        "ingestion layer, storage strategy, processing pipeline, and "
        "dashboard serving. Keep it to 3-4 paragraphs.",
        max_tokens=600,
    )
    preview = (design_response.text or "")[:150].replace("\n", "\n    ")
    print(f"    Initial design: {preview}...\n")

    # --- Tool invocation 1: Storage Analyst ---
    print("  Invoking subagent: Storage Analyst...")
    storage_result = agent_tool(
        session, coordinator,
        purpose="Storage Analyst",
        directive=(
            "You are a database storage specialist. Focus on data volume "
            "calculations, storage engine selection, partitioning strategies, "
            "and cost projections. Be quantitative."
        ),
        prompt=(
            "Estimate storage requirements for an IoT event platform:\n"
            "- 100K events/sec, average 500 bytes per event\n"
            "- 1 year retention\n"
            "- 50 tenants, roughly equal load\n"
            "Include: raw storage, indexes, replication overhead, and "
            "estimated monthly cost at cloud rates."
        ),
        temperature=0.2,
    )
    print(f"    Collapsed: {storage_result.source_tokens} -> "
          f"{storage_result.summary_tokens} tokens\n")

    # --- Tool invocation 2: Security Reviewer ---
    print("  Invoking subagent: Security Reviewer...")
    security_result = agent_tool(
        session, coordinator,
        purpose="Security Reviewer",
        directive=(
            "You are an application security engineer. Focus on threat "
            "modeling, tenant isolation, data-at-rest encryption, and "
            "access control boundaries. Identify concrete risks."
        ),
        prompt=(
            "Review a multi-tenant IoT event platform for security concerns:\n"
            "- 50 tenants on shared infrastructure\n"
            "- Events ingested over HTTPS, stored in time-series DB\n"
            "- Real-time dashboards served via API gateway\n"
            "- Alerting engine reads aggregated metrics\n"
            "Flag the top 5 security risks with severity ratings."
        ),
        temperature=0.3,
    )
    print(f"    Collapsed: {security_result.source_tokens} -> "
          f"{security_result.summary_tokens} tokens\n")

    # --- Tool invocation 3: Cost Estimator ---
    print("  Invoking subagent: Cost Estimator...")
    cost_result = agent_tool(
        session, coordinator,
        purpose="Cost Estimator",
        directive=(
            "You are a cloud infrastructure cost analyst. Focus on compute, "
            "storage, networking, and managed service costs. Provide monthly "
            "estimates with clear assumptions."
        ),
        prompt=(
            "Estimate monthly cloud costs for an IoT event platform:\n"
            "- 100K events/sec ingestion (Kafka or equivalent)\n"
            "- ~1.5 PB/year raw storage with 3x replication\n"
            "- Real-time stream processing (Flink or equivalent)\n"
            "- Dashboard API serving ~1K concurrent users\n"
            "Break down by: compute, storage, networking, managed services."
        ),
        temperature=0.2,
    )
    print(f"    Collapsed: {cost_result.source_tokens} -> "
          f"{cost_result.summary_tokens} tokens\n")

    # Architect synthesizes all specialist findings
    print("  Architect synthesizing specialist analyses...")
    synthesis = coordinator.llm.chat(
        "Three specialist analyses are now in your context: storage estimates, "
        "security review, and cost projections. Revise your architecture to "
        "address the key findings. Produce a final design summary with:\n"
        "1. Updated architecture decisions\n"
        "2. How you addressed the top security risks\n"
        "3. Cost optimization opportunities\n"
        "Keep it concise -- bullet points preferred.",
        max_tokens=800,
    )

    print(f"\n  Final Synthesis:\n")
    for line in (synthesis.text or "").split("\n"):
        print(f"    {line}")

    # Summary
    status = coordinator.status()
    tracts = session.list_tracts()
    print(f"\n  --- Section 1 Summary ---")
    print(f"  Coordinator: {status.token_count} tokens, {status.commit_count} commits")
    print(f"  Session tracts: {len(tracts)} (1 coordinator + 3 collapsed children)")
    print(f"  Subagent invocations: 3 (storage, security, cost)")

    session.close()


# =====================================================================
# Section 2: Parallel Subagent Fan-Out
# =====================================================================

REVIEW_TASKS = [
    {
        "purpose": "Performance Reviewer",
        "directive": (
            "You are a performance engineer. Evaluate architectures for "
            "throughput bottlenecks, latency issues, and scaling limits. "
            "Be specific about numbers and failure thresholds."
        ),
        "prompt": (
            "Review this API gateway design for performance:\n"
            "- NGINX reverse proxy -> 8 Node.js workers -> PostgreSQL\n"
            "- Connection pooling: pgBouncer, 20 connections per worker\n"
            "- Rate limiting: 1000 req/sec per tenant, in-memory counter\n"
            "- Response caching: Redis, 60-second TTL\n"
            "Identify the top 3 performance bottlenecks and scaling limits."
        ),
        "temperature": 0.3,
    },
    {
        "purpose": "Reliability Reviewer",
        "directive": (
            "You are a site reliability engineer. Focus on failure modes, "
            "blast radius, recovery time, and operational complexity. "
            "Think about what breaks at 3 AM."
        ),
        "prompt": (
            "Review this API gateway design for reliability:\n"
            "- NGINX reverse proxy -> 8 Node.js workers -> PostgreSQL\n"
            "- Single PostgreSQL primary, async replica for reads\n"
            "- Redis for caching and rate limiting (single instance)\n"
            "- Health checks: HTTP /healthz every 10 seconds\n"
            "Identify single points of failure and recovery gaps."
        ),
        "temperature": 0.3,
    },
    {
        "purpose": "Operability Reviewer",
        "directive": (
            "You are a platform engineer. Focus on observability, "
            "deployment complexity, debugging experience, and on-call "
            "burden. Consider a team of 4 maintaining this."
        ),
        "prompt": (
            "Review this API gateway design for operability:\n"
            "- NGINX reverse proxy -> 8 Node.js workers -> PostgreSQL\n"
            "- Logging: stdout JSON, shipped to ELK via Filebeat\n"
            "- Metrics: StatsD counters, Grafana dashboards\n"
            "- Deploys: Docker Compose, manual rollback\n"
            "What would make on-call miserable? What's missing?"
        ),
        "temperature": 0.4,
    },
]


def section_2_parallel_fan_out() -> None:
    print()
    print("=" * 60)
    print("  Section 2: Parallel Subagent Fan-Out")
    print("=" * 60)
    print()
    print("  Three independent reviewer subagents evaluate the same design")
    print("  from different angles. Results collapse back for synthesis.")
    print()

    session = Session.open()
    coordinator = session.create_tract(
        display_name="lead_architect",
        config=TractConfig(token_budget=TokenBudgetConfig(max_tokens=6000)),
    )
    _configure_llm(coordinator, temperature=0.4)

    coordinator.system(
        "You are a lead architect collecting independent reviews of an "
        "API gateway design. Specialist reviews will appear inline. "
        "Your job is to synthesize them into prioritized action items."
    )
    coordinator.user(
        "We need a review of our API gateway before going to production.\n"
        "Design: NGINX -> Node.js workers -> PostgreSQL + Redis\n"
        "Three specialists will review: performance, reliability, operability.",
        message="review request: API gateway design",
    )

    # Fan-out: spawn all three reviewers (sequential execution, conceptually parallel)
    print("  Fan-out: invoking 3 reviewer subagents...\n")

    results = {}
    for task in REVIEW_TASKS:
        name = task["purpose"]
        print(f"    [{name}]...")

        result = agent_tool(
            session, coordinator,
            purpose=task["purpose"],
            directive=task["directive"],
            prompt=task["prompt"],
            temperature=task["temperature"],
        )
        results[name] = result

        print(f"      Collapsed: {result.source_tokens} -> "
              f"{result.summary_tokens} tokens")

    # Synthesize
    print("\n  All reviews collapsed. Synthesizing...")

    synthesis = coordinator.llm.chat(
        "Three specialist reviews are now in your context: performance, "
        "reliability, and operability. Produce a prioritized action plan:\n\n"
        "1. CRITICAL (must fix before production) -- issues all reviewers flag\n"
        "2. HIGH (fix within first sprint) -- issues from 2+ reviewers\n"
        "3. MEDIUM (backlog) -- single-reviewer concerns\n\n"
        "For each item, note which reviewer(s) raised it and a concrete fix. "
        "Keep it to bullet points, max 15 items total.",
        max_tokens=800,
    )

    print(f"\n  Prioritized Action Plan:\n")
    for line in (synthesis.text or "").split("\n"):
        print(f"    {line}")

    # Summary
    status = coordinator.status()
    tracts = session.list_tracts()
    print(f"\n  --- Section 2 Summary ---")
    print(f"  Coordinator: {status.token_count} tokens, {status.commit_count} commits")
    print(f"  Session tracts: {len(tracts)} (1 coordinator + 3 collapsed children)")
    print(f"  Fan-out invocations: {len(REVIEW_TASKS)}")

    print(f"\n  Coordinator log:")
    for ci in coordinator.log():
        msg = (ci.message or "")[:50]
        print(f"    {ci.commit_hash[:8]}  {ci.content_type:12s}  {msg}")

    session.close()


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    print()
    print("  Agents as Tools")
    print("  Stateless subagent invocations via spawn + chat + collapse")
    print()

    section_1_subagent_as_tool()
    section_2_parallel_fan_out()

    print("\n\n  Done. Both sections complete.")
    print("  Key insight: spawn + single chat + collapse is the tract equivalent")
    print("  of a tool call where the 'tool' is an LLM with a different persona.")
    print("  The parent sees collapsed results like any other context -- no")
    print("  persistent child state, no ongoing coordination.")


if __name__ == "__main__":
    main()


# --- See also ---
# Multi-agent collaboration:  agentic/02_multi_agent.py
# Tool compaction:            agentic/03_tool_compaction.py
# Adversarial review:         agentic/05_adversarial_review.py
