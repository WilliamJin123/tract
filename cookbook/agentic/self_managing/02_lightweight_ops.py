"""Lightweight Inline Operations

An agent that tags, pins, and checks status as part of its normal workflow.
These are simple enough meta-decisions that the agent handles inline --
no sidecar needed.

Demonstrates: ToolExecutor for tag/annotate/status, inline agent decisions,
              Orchestrator for autonomous tagging and pinning
"""

import sys
from pathlib import Path

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: Direct ToolExecutor calls for tag, annotate, status
# =====================================================================

def part1_manual():
    """Execute tag, annotate, and status operations via ToolExecutor."""
    print("=" * 60)
    print("PART 1 -- Manual: Tag, Annotate, Status via ToolExecutor")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a research assistant tracking experiment results.")

        ci_hyp = t.user("Hypothesis: the new optimizer converges 2x faster.")
        ci_res = t.assistant(
            "Experiment results: the optimizer converged in 45 epochs "
            "vs 90 for the baseline. Hypothesis confirmed."
        )
        ci_note = t.user("This is a key finding -- save it for the paper.")

        executor = ToolExecutor(t)

        # --- Tag: mark commits with semantic labels ---
        print(f"\n  Tagging commits:\n")

        result = executor.execute("register_tag", {
            "name": "hypothesis",
            "description": "A stated hypothesis to test",
        })
        print(f"    register_tag: {result.output}")

        result = executor.execute("register_tag", {
            "name": "key-finding",
            "description": "A significant result worth preserving",
        })
        print(f"    register_tag: {result.output}")

        result = executor.execute("tag", {
            "commit_hash": ci_hyp.commit_hash,
            "tag": "hypothesis",
        })
        print(f"    tag hypothesis: {result.output}")

        result = executor.execute("tag", {
            "commit_hash": ci_res.commit_hash,
            "tag": "key-finding",
        })
        print(f"    tag key-finding: {result.output}")

        # --- Annotate: pin critical context ---
        print(f"\n  Annotating commits:\n")

        result = executor.execute("annotate", {
            "target_hash": ci_res.commit_hash,
            "priority": "pinned",
            "reason": "Key experiment result -- protect from compression",
        })
        print(f"    annotate pin: {result.output}")

        print("\n  Conversation:")
        t.compile().pprint(style="chat")

        # --- Status: check current state ---
        print(f"\n  Checking status:\n")

        result = executor.execute("status", {})
        print(f"    status: {result.output}")

        # --- Query: find tagged commits ---
        print(f"\n  Querying by tags:\n")

        result = executor.execute("query_by_tags", {"tags": ["key-finding"]})
        print(f"    key-finding commits: {result.output}")

        result = executor.execute("get_tags", {
            "commit_hash": ci_res.commit_hash,
        })
        print(f"    tags on result commit: {result.output}")


# =====================================================================
# PART 2 -- Agent: Orchestrator autonomously tags and pins
# =====================================================================

def part2_agent():
    """Orchestrator autonomously tags and pins conversation content."""
    if not llm.api_key:
        print(f"\n{'=' * 60}")
        print("PART 2: SKIPPED (no llm.api_key)")
        print("=" * 60)
        return

    print(f"\n{'=' * 60}")
    print("PART 2 -- Agent: Orchestrator Tags and Pins Inline")
    print("=" * 60)
    print()
    print("  The Orchestrator gets tag, annotate, and status tools.")
    print("  It autonomously decides what to tag and pin based on")
    print("  the conversation content.")
    print()

    # Profile with hint-driven descriptions for inline ops
    inline_ops_profile = ToolProfile(
        name="inline-ops",
        tool_configs={
            "status": ToolConfig(
                enabled=True,
                description=(
                    "Check your context status. Call first to understand "
                    "the current state of the conversation."
                ),
            ),
            "tag": ToolConfig(
                enabled=True,
                description=(
                    "Tag a commit for retrieval. Call when a message contains "
                    "a key finding, decision, or action item. Use semantic "
                    "labels like 'decision', 'action_item', 'key_finding'."
                ),
            ),
            "annotate": ToolConfig(
                enabled=True,
                description=(
                    "Pin or skip a commit. Pin (priority='pinned') when "
                    "content is critical and must survive compression. "
                    "Skip (priority='skip') when content is superseded."
                ),
            ),
            "register_tag": ToolConfig(
                enabled=True,
                description=(
                    "Register a custom tag before first use. Call this once "
                    "per tag name before using it with the tag tool."
                ),
            ),
            "log": ToolConfig(
                enabled=True,
                description="View recent commit history to find hashes for tagging.",
            ),
        },
    )

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system(
            "You are a research assistant tracking experiment results."
        )

        # Seed a conversation with content worth tagging
        t.user("What are the main approaches to neural architecture search?")
        t.assistant(
            "Three main approaches: (1) reinforcement learning-based NAS, "
            "(2) differentiable NAS (DARTS), and (3) evolutionary NAS. "
            "DARTS is fastest but prone to collapse; RL-NAS is robust but slow."
        )
        t.user("Which approach is most compute-efficient for our 8-GPU budget?")
        t.assistant(
            "For an 8-GPU budget, DARTS variants are the best fit. "
            "Specifically, PC-DARTS or FairDARTS avoid the collapse issue "
            "while completing search in under 1 GPU-day."
        )
        t.user("Good analysis. That recommendation is critical for planning.")

        # Orchestrator autonomously tags and pins
        orch_config = OrchestratorConfig(
            autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
            max_steps=20,
            profile=inline_ops_profile,
            task_context=(
                "Review the conversation and organize it with tags and pins. "
                "Register semantic tags (e.g. 'key_finding', 'decision', "
                "'action_item'), then tag relevant commits. Pin any critical "
                "recommendations that must survive compression."
            ),
        )
        orch = Orchestrator(t, config=orch_config)
        result = orch.run()

        print(f"  Orchestrator completed: {result.total_tool_calls} tool calls, "
              f"state={result.state.value}")
        for step in result.steps:
            status = "OK" if step.success else "FAIL"
            args_short = str(step.tool_call.arguments)[:60]
            print(f"    [{status}] {step.tool_call.name}({args_short})")

        # Show what the agent organized
        print(f"\n  {'=' * 50}")
        print("  Tags applied by the orchestrator:")
        for entry in t.log():
            tags = t.get_tags(entry.commit_hash)
            if len(tags) > 1:  # More than just auto-classified
                msg = (entry.message or entry.content_text or "")[:50]
                print(f"    {entry.commit_hash[:8]}  tags={tags}  {msg}")

        print("\n  Context after orchestrator tagging:")
        t.compile().pprint(style="compact")

        # Quick summary of what the agent pinned
        pinned = t.pinned()
        print(f"\n  Pinned by orchestrator ({len(pinned)} commits):")
        for entry in pinned:
            print(f"    {entry.commit_hash[:8]}  {entry.message or ''}")
        if not pinned:
            print("    (none)")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/developer/metadata/01_tags.py          -- Full tag system (register, query, strict mode)
# cookbook/developer/metadata/02_priority.py      -- Priority annotations (pin, skip, normal)
# cookbook/agentic/sidecar/04_auto_tagger.py      -- LLM-driven auto-tagging via orchestrator
# cookbook/agentic/self_managing/01_tool_hints.py  -- Description-driven tool selection
