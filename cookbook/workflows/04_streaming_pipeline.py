"""Streaming Pipeline -- Live output with stage transitions

A multi-stage workflow where each stage streams its output live.
Combines streaming with config-based stage management and middleware gates.

Stages:
  research    -- gather information (streamed)
  synthesize  -- combine findings (streamed)

Demonstrates: on_token in multi-stage workflow, live progress display,
              streaming + tool use, rich terminal formatting

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.small


class StreamPrinter:
    """Collects streamed tokens and displays with a prefix."""

    def __init__(self, prefix: str = ""):
        self.prefix = prefix
        self.chunks = 0
        self._started = False

    def __call__(self, text: str) -> None:
        if not self._started:
            print(f"\n{self.prefix}", end="", flush=True)
            self._started = True
        print(text, end="", flush=True)
        self.chunks += 1


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # Stage configuration
        t.configure(stage="research", temperature=0.8)

        # Gate: require commits before synthesis
        def synthesis_gate(ctx):
            if ctx.target != "synthesize":
                return
            commits = len(ctx.tract.log())
            if commits < 4:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 4 commits for synthesis (have {commits})"],
                )

        t.use("pre_transition", synthesis_gate)

        t.system(
            "You are a research assistant. Work through stages:\n"
            "1. RESEARCH: Explore the topic, list key points\n"
            "2. SYNTHESIZE: Combine findings into a summary\n"
            "Use 'transition' to move between stages."
        )

        # ---------------------------------------------------------
        # Stage 1: Research (streamed)
        # ---------------------------------------------------------
        print("=== Stage 1: Research (streaming) ===")
        research_printer = StreamPrinter(prefix="  [research] ")

        result1 = t.run(
            "Research the concept of 'eventual consistency' in distributed systems. "
            "List 3-4 key points about how it works and why it matters.",
            max_steps=5,
            tools=[],
            on_token=research_printer,
        )
        print(f"\n  ({research_printer.chunks} chunks streamed)")

        # ---------------------------------------------------------
        # Stage 2: Transition and synthesize (streamed)
        # ---------------------------------------------------------
        print("\n=== Stage 2: Synthesize (streaming) ===")

        # Manually transition since the agent didn't have the tool
        t.transition("synthesize", handoff="Summarize the research findings")
        t.configure(temperature=0.3)

        synth_printer = StreamPrinter(prefix="  [synthesize] ")

        result2 = t.run(
            "Now write a concise 2-paragraph summary of eventual consistency "
            "based on the research above.",
            max_steps=3,
            tools=[],
            on_token=synth_printer,
        )
        print(f"\n  ({synth_printer.chunks} chunks streamed)")

        # ---------------------------------------------------------
        # Final state
        # ---------------------------------------------------------
        print(f"\n=== Final State ===")
        print(f"  Research: {result1.status} ({result1.steps} steps)")
        print(f"  Synthesis: {result2.status} ({result2.steps} steps)")
        configs = t.get_all_configs()
        print(f"  Active configs: {configs}")
        print(f"  Total commits: {len(t.log())}")


if __name__ == "__main__":
    main()


# --- See also ---
# Basic streaming:    getting_started/04_streaming.py
# Coding assistant:   workflows/01_coding_assistant.py
# Research pipeline:  workflows/02_research_pipeline.py
