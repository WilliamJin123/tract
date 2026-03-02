"""Budget-Aware Self-Management

An agent that reads its own token budget via status() and adapts: compressing
when running hot, continuing normally when under budget. The agent makes this
decision inline without a separate sidecar process.

Demonstrates: status() via tools, budget-driven compress decisions,
              Orchestrator for autonomous budget-aware compression
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


# =====================================================================
# PART 1 -- Manual: Check status, compute fill %, decide to compress
# =====================================================================

def part1_manual():
    """Manual budget monitoring: status -> compute fill -> compress if needed."""
    print("=" * 60)
    print("PART 1 -- Manual: Budget Check and Compress Decision")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=500))

    with Tract.open(config=config) as t:
        t.system("You are a project planning assistant.")

        # Fill up the context with a conversation
        for i in range(6):
            t.user(f"Sprint {i} update: completed 3 stories, 2 carry-overs.")
            t.assistant(f"Noted sprint {i}. Velocity trending at 5 pts/sprint.")

        executor = ToolExecutor(t)

        # Step 1: Check status via tool
        result = executor.execute("status", {})
        print(f"\n  Status: {result.output}")

        # Step 2: Parse budget from status() directly
        status = t.status()
        budget_max = status.token_budget_max or 0
        fill_pct = (status.token_count / budget_max * 100) if budget_max else 0
        print(f"\n  Token count: {status.token_count}")
        print(f"  Budget max:  {budget_max}")
        print(f"  Fill:        {fill_pct:.1f}%")

        # Step 3: Decision tree
        COMPRESS_THRESHOLD = 70  # percent
        if fill_pct > COMPRESS_THRESHOLD:
            print(f"\n  Over {COMPRESS_THRESHOLD}% -- compressing...")
            target = int(budget_max * 0.4)
            t.compress(content=(
                "Project planning session: tracked 6 sprints with "
                "average velocity of 5 points/sprint. 2 carry-overs "
                "per sprint is the recurring pattern."
            ))

            # Verify compression worked
            status_after = t.status()
            new_pct = (status_after.token_count / budget_max * 100) if budget_max else 0
            print(f"  After compress: {status_after.token_count} tokens ({new_pct:.1f}%)")
        else:
            print(f"\n  Under {COMPRESS_THRESHOLD}% -- no action needed.")


# =====================================================================
# PART 2 -- Interactive: Agent reports budget, human decides
# =====================================================================

def part2_interactive():
    """Agent reports budget status; human decides whether to compress."""
    print(f"\n{'=' * 60}")
    print("PART 2 -- Interactive: Human-Gated Budget Management")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=400))

    with Tract.open(config=config) as t:
        t.system("You are a data analysis assistant.")

        for i in range(8):
            t.user(f"Dataset {i}: loaded {(i + 1) * 1000} rows from warehouse.")
            t.assistant(f"Processed dataset {i}. Found {i * 3} anomalies.")

        executor = ToolExecutor(t)

        # Check status
        result = executor.execute("status", {})
        print(f"\n  Status: {result.output}")

        status = t.status()
        budget_max = status.token_budget_max or 1
        fill_pct = status.token_count / budget_max * 100

        print(f"\n  Budget fill: {fill_pct:.1f}%")

        if fill_pct > 60:
            print(f"  Budget is above 60%.")
            if click.confirm("  Compress context to free up budget?", default=True):
                t.compress(content=(
                    "Data analysis session: processed 8 datasets from "
                    "warehouse, found increasing anomaly counts. Total "
                    "rows: 36,000."
                ))
                status_after = t.status()
                new_pct = status_after.token_count / budget_max * 100
                print(f"  Compressed: {status_after.token_count} tokens ({new_pct:.1f}%)")
            else:
                print(f"  Skipped compression (human override).")
        else:
            print(f"  Budget healthy -- no compression needed.")


# =====================================================================
# PART 3 -- Agent: Orchestrator self-compresses when budget is high
# =====================================================================

def part3_agent():
    """Orchestrator with budget-aware compression profile."""
    if not TRACT_OPENAI_API_KEY:
        print(f"\n{'=' * 60}")
        print("PART 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 3 -- Agent: Orchestrator Budget-Aware Compression")
    print("=" * 60)
    print()
    print("  The Orchestrator checks status, pins critical content,")
    print("  and compresses when budget is high. It decides what to")
    print("  do based on the context assessment -- no manual loop.")
    print()

    # Profile with budget-aware hints
    budget_profile = ToolProfile(
        name="budget-aware",
        tool_configs={
            "status": ToolConfig(
                enabled=True,
                description=(
                    "Check the token budget. If usage exceeds 70%, "
                    "compression is needed."
                ),
            ),
            "compress": ToolConfig(
                enabled=True,
                description=(
                    "Compress context to free up token budget. Provide a "
                    "concise summary of the conversation so far as content."
                ),
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commits to understand what to summarize.",
            ),
            "annotate": ToolConfig(
                enabled=True,
                description=(
                    "Pin critical context before compressing. Pinned commits "
                    "survive compression. Call with priority='pinned' on any "
                    "commit that must be preserved verbatim."
                ),
            ),
        },
    )

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=800))

    with Tract.open(
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a research assistant helping with ML questions."
        )

        # Fill context with enough content to push budget limits
        exchanges = [
            ("Explain the three main types of machine learning.",
             "The three types are supervised, unsupervised, and reinforcement learning."),
            ("What are the key differences between CNNs and transformers?",
             "CNNs use local convolutions; transformers use global self-attention."),
            ("How does attention work in transformers?",
             "Attention computes query-key-value dot products to weight token relevance."),
            ("Compare BERT and GPT architectures.",
             "BERT is bidirectional encoder; GPT is autoregressive decoder."),
        ]
        for q, a in exchanges:
            t.user(q)
            t.assistant(a)

        status_before = t.status()
        budget_max = status_before.token_budget_max or 1
        fill_pct = status_before.token_count / budget_max * 100
        print(f"  Before: {status_before.token_count}/{budget_max} tokens ({fill_pct:.0f}%)")

        # Orchestrator assesses context and takes action
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=15,
            profile=budget_profile,
            task_context=(
                "Check the context budget. If usage exceeds 70%, pin any "
                "critical findings and then compress to free up space. "
                "Provide a concise summary when compressing."
            ),
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"\n  Orchestrator completed: {result.total_tool_calls} tool calls, "
              f"state={result.state.value}")
        for step in result.steps:
            status_label = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:60]
            print(f"    [{status_label}] {step.tool_call.name}({args_short})")

        # Final report
        final_status = t.status()
        budget_max = final_status.token_budget_max or 1
        fill_pct = final_status.token_count / budget_max * 100

        print(f"\n  {'=' * 50}")
        print(f"  After orchestrator:")
        print(f"    Final tokens: {final_status.token_count}/{budget_max} ({fill_pct:.0f}%)")
        print(f"    Commits:      {final_status.commit_count}")


def main():
    part1_manual()
    part2_interactive()
    part3_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/conversations/03_status_and_budget.py  -- Status and budget basics
# cookbook/agentic/sidecar/02_assessment_loop.py            -- Orchestrator-driven budget management
# cookbook/agentic/sidecar/01_triggers.py                   -- CompressTrigger for automatic compression
# cookbook/developer/operations/01_compress.py              -- Manual compression patterns
