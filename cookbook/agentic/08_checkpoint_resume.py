"""Checkpoint and Resume: branches as save-states for long-running agents

When an agent hits a milestone, it snapshots progress as a branch or tagged
commit. If a subsequent step fails or produces garbage, the agent resets to
the last known-good state and retries -- without losing the checkpoint.

Sections:
  1. Checkpoint Before Risky Operation  -- branch snapshot, reset on failure
  2. Resume From Last-Known-Good State  -- tag-based checkpoints, find + reset

Demonstrates: t.branch(), t.reset(), t.switch(),
              t.find(), t.log(), t.status(),
              tag-based checkpoint discovery, DAG isolation of failed work
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError, MiddlewareContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# =====================================================================
# Section 1: Checkpoint Before Risky Operation
# =====================================================================
# Snapshot a branch before a risky merge. If it goes wrong, reset to
# the checkpoint and retry. Failed work stays isolated on its branch.

def section_checkpoint_before_risky_op() -> None:
    print("=" * 70)
    print("  Section 1: Checkpoint Before Risky Operation")
    print("=" * 70)
    print()

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:
        t.system("You are a research agent analyzing database architectures.")
        log = StepLogger()

        # Step 1: Build research on main
        print("  --- Step 1: Research (main) ---\n")
        t.llm.run(
            "Research SQL vs NoSQL trade-offs for a high-traffic e-commerce "
            "platform. Cover consistency, scaling, query flexibility. "
            "Commit 2-3 findings.",
            max_steps=8, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        print(f"\n  Main: {t.status().commit_count} commits")

        # Step 2: Experimental work on a side branch
        print("\n  --- Step 2: Experiment branch ---\n")
        t.branch("experiment", switch=True)
        t.llm.run(
            "Take a contrarian position: argue that a single-node SQLite "
            "database is sufficient. Be bold, commit your argument.",
            max_steps=6, max_tokens=512,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )

        # Step 3: Checkpoint before risky merge
        print("\n  --- Step 3: Checkpoint + risky merge ---\n")
        t.switch("main")

        # Save checkpoint BEFORE the merge
        cp_hash = t.branch("checkpoint/pre-merge", switch=False)
        print(f"  Checkpoint: checkpoint/pre-merge -> {cp_hash[:8]}")

        merge_result = t.merge("experiment", strategy="theirs")
        print(f"  Merged experiment (type: {merge_result.merge_type})")
        print(f"  Post-merge: {t.status().token_count} tokens")

        # Step 4: Bad merge -- reset to checkpoint
        print("\n  --- Step 4: Reset to checkpoint ---\n")
        print("  Contrarian SQLite argument contaminated main research.")
        reset_hash = t.reset(target="checkpoint/pre-merge")
        print(f"  HEAD reset to: {reset_hash[:8]}")
        print(f"  Restored: {t.status().commit_count} commits")

        # Step 5: Retry with better approach
        print("\n  --- Step 5: Retry ---\n")
        t.llm.run(
            "Write a balanced conclusion: SQLite is good for prototyping "
            "but PostgreSQL is better for production e-commerce. Commit it.",
            max_steps=6, max_tokens=512,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )

        # Final DAG state
        print("\n  --- Final DAG ---")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")
        print(f"  Main: {t.status().commit_count} commits")
        print("  checkpoint/pre-merge: preserved | experiment: isolated")


# =====================================================================
# Section 2: Resume From Last-Known-Good State
# =====================================================================
# Tag checkpoints at stage boundaries. On failure, find() the last
# checkpoint and reset. Commits after the checkpoint become unreachable.

def section_resume_from_checkpoint() -> None:
    print(f"\n{'=' * 70}")
    print("  Section 2: Resume From Last-Known-Good State")
    print("=" * 70)
    print()

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:
        t.system(
            "You are a research synthesis agent. Work through stages: "
            "research, analysis, synthesis. Commit your work at each step."
        )
        for tag in ["checkpoint", "research", "analysis", "synthesis"]:
            t.register_tag(tag)
        log = StepLogger()

        # Stage 1: Research
        print("  --- Stage 1: Research ---\n")
        t.llm.run(
            "Research three caching approaches for distributed systems: "
            "write-through, write-back, write-around. Commit a summary.",
            max_steps=8, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        t.commit(
            content={"content_type": "freeform", "text": "Research complete."},
            message="checkpoint: research complete",
            tags=["checkpoint", "research"],
            metadata={"stage": "research_complete"},
        )
        print(f"\n  Checkpoint tagged (research_complete)")

        # Stage 2: Analysis
        print("\n  --- Stage 2: Analysis ---\n")
        t.llm.run(
            "Analyze which caching strategy is best for a read-heavy "
            "social media feed (90% reads, 10% writes, 50ms SLA). "
            "Commit your recommendation.",
            max_steps=8, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        t.commit(
            content={"content_type": "freeform", "text": "Analysis complete."},
            message="checkpoint: analysis complete",
            tags=["checkpoint", "analysis"],
            metadata={"stage": "analysis_complete"},
        )
        print(f"\n  Checkpoint tagged (analysis_complete)")

        # Stage 3: Synthesis -- simulated failure
        print("\n  --- Stage 3: Synthesis (simulated failure) ---\n")
        t.commit(
            content={"content_type": "freeform",
                     "text": "SYNTHESIS: [garbled -- LLM hallucinated about "
                             "quantum computing instead of caching. Unusable.]"},
            message="synthesis: bad output (hallucination)",
            tags=["synthesis"],
            metadata={"stage": "synthesis", "quality": "failed"},
        )
        t.commit(
            content={"content_type": "freeform",
                     "text": "MORE GARBAGE: continued irrelevant output."},
            message="synthesis: continued bad output",
            metadata={"stage": "synthesis", "quality": "failed"},
        )
        print(f"  After failure: {t.status().commit_count} commits "
              f"(2 garbage commits polluting context)")

        # Recovery: find last checkpoint and reset
        print("\n  --- Recovery ---\n")
        checkpoints = t.find(tag="checkpoint", limit=10)
        print(f"  Found {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            stage = (cp.metadata or {}).get("stage", "?")
            print(f"    {cp.commit_hash[:8]}  {stage}")

        latest = checkpoints[0]
        t.reset(target=latest.commit_hash)
        print(f"\n  Reset to {latest.commit_hash[:8]} "
              f"({(latest.metadata or {}).get('stage')})")
        print(f"  Restored: {t.status().commit_count} commits")

        # Retry synthesis from clean state
        print("\n  --- Retry synthesis ---\n")
        t.llm.run(
            "Synthesize research and analysis into a final caching "
            "recommendation for the read-heavy social media feed. "
            "3-4 sentences. Commit it.",
            max_steps=6, max_tokens=512,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )

        # Verify garbage is gone
        print("\n  --- Final State ---\n")
        final_log = t.log(limit=10)
        for entry in final_log:
            tags_str = f" [{', '.join(entry.tags)}]" if entry.tags else ""
            print(f"    {entry.commit_hash[:8]}  {(entry.message or '')[:50]}{tags_str}")

        garbage = t.find(
            metadata_key="quality", metadata_value="failed", limit=10
        )
        print(f"\n  Failed commits reachable from HEAD: {len(garbage)}")
        print("  Clean." if len(garbage) == 0 else "  WARNING: garbage still reachable.")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    section_checkpoint_before_risky_op()
    section_resume_from_checkpoint()

    print(f"\n{'=' * 70}")
    print("  Done. Both checkpoint/resume patterns demonstrated.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()


# --- See also ---
# Error recovery:          agentic/10_error_recovery.py
# Implicit discovery:      agentic/01_implicit_discovery.py
# Adversarial review:      agentic/05_adversarial_review.py
