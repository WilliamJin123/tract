"""Streaming Pipeline -- Live output with stage transitions

A multi-stage workflow where each stage streams its output live using
StreamPrinter for Rich-formatted markdown panels.  Combines streaming
with config-based stage management and middleware gates.

Stages:
  research    -- gather information (streamed)
  synthesize  -- combine findings (streamed)

Demonstrates: StreamPrinter per-stage, live Rich markdown panels,
              streaming + middleware gates, stage transitions

Requires: LLM API key (uses Cerebras provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError, MiddlewareContext
from tract.formatting import StreamPrinter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm

MODEL_ID = llm.large


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # Stage configuration
        t.config.set(stage="research", temperature=0.8)

        # Gate: require commits before synthesis
        def synthesis_gate(ctx: MiddlewareContext):
            if ctx.target != "synthesize":
                return
            commits = len(ctx.tract.search.log())
            if commits < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for synthesis (have {commits})"],
                )

        t.middleware.add("pre_transition", synthesis_gate)

        t.system(
            "You are a research assistant. Work through stages:\n"
            "1. RESEARCH: Explore the topic, list key points\n"
            "2. SYNTHESIZE: Combine findings into a summary\n"
            "Use 'transition' to move between stages."
        )

        # ---------------------------------------------------------
        # Stage 1: Research (streamed with Rich panel)
        # ---------------------------------------------------------
        print("=== Stage 1: Research ===")

        with StreamPrinter(title="Research", border_style="cyan") as printer:
            result1 = t.llm.run(
                "Research the concept of 'eventual consistency' in distributed systems. "
                "List 3-4 key points about how it works and why it matters.",
                max_steps=5,
                tools=[],
                on_token=printer,
            )
        print(f"  ({printer.chunk_count} chunks streamed)")

        # ---------------------------------------------------------
        # Stage 2: Transition and synthesize (streamed)
        # ---------------------------------------------------------
        print("\n=== Stage 2: Synthesize ===")

        # Manually transition since the agent didn't have the tool
        t.transition("synthesize", handoff="Summarize the research findings")
        t.config.set(stage="synthesize", temperature=0.3)

        with StreamPrinter(title="Synthesis", border_style="green") as printer:
            result2 = t.llm.run(
                "Now write a concise 2-paragraph summary of eventual consistency "
                "based on the research above.",
                max_steps=3,
                tools=[],
                on_token=printer,
            )
        print(f"  ({printer.chunk_count} chunks streamed)")

        # ---------------------------------------------------------
        # Final state
        # ---------------------------------------------------------
        print(f"\n=== Final State ===")
        print(f"  Research: {result1.status} ({result1.steps} steps)")
        print(f"  Synthesis: {result2.status} ({result2.steps} steps)")
        configs = t.config.get_all()
        print(f"  Active configs: {configs}")
        print(f"  Total commits: {len(t.search.log())}")


if __name__ == "__main__":
    main()


# --- See also ---
# Basic streaming:    getting_started/04_streaming.py
# Coding assistant:   workflows/01_coding_assistant.py
# Self-routing:       workflows/07_self_routing.py
