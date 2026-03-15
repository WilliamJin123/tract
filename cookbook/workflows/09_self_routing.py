"""Self-Routing Workflow: Middleware-Driven Stage Transitions

Instead of the agent calling transition() explicitly, a post_commit
middleware handler detects content patterns and routes to the
appropriate stage automatically. The agent just works; stage
transitions happen in the background.

Stages:
  research       -- gathering information, reading, note-taking
  implementation -- writing code, building artifacts
  validation     -- testing, verifying, quality checks

The middleware scans each assistant commit for stage-indicative
keywords. When enough signal accumulates, it transitions and
reconfigures the tract (temperature, compile strategy, directives).

Demonstrates: keyword-based routing middleware, automatic stage
              transitions, per-stage config/directives, agent-driven
              workflow without explicit transition tool calls

Requires: LLM API key (uses Cerebras provider)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, MiddlewareContext
from tract.formatting import pprint_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


# =====================================================================
# Routing table: stage -> (keywords, config overrides, directive text)
# =====================================================================

STAGES = {
    "research": {
        "keywords": [],  # default stage, no inbound keywords
        "config": {"temperature": 0.7, "compile_strategy": "full"},
        "directive": (
            "You are in RESEARCH mode. Focus on gathering information, "
            "reading sources, and taking structured notes. Do not write "
            "code yet."
        ),
    },
    "implementation": {
        "keywords": [
            "implement", "code", "write the", "class ", "def ",
            "function", "build", "create the",
        ],
        "config": {"temperature": 0.3, "compile_strategy": "messages"},
        "directive": (
            "You are in IMPLEMENTATION mode. Write precise, working "
            "code based on the research. Be exact and concise."
        ),
    },
    "validation": {
        "keywords": [
            "test", "verify", "assert", "check", "validate",
            "qa", "review", "confirm",
        ],
        "config": {"temperature": 0.1, "compile_strategy": "full"},
        "directive": (
            "You are in VALIDATION mode. Write tests, verify "
            "correctness, and check edge cases. Be thorough."
        ),
    },
}


def build_router(stages: dict, *, min_signals: int = 1) -> tuple:
    """Build a keyword-routing middleware handler.

    Args:
        stages: Mapping of stage_name -> {keywords, config, directive}.
        min_signals: Number of keyword hits required to trigger transition.

    Returns:
        A (handler_func, state_dict) tuple. The state dict tracks
        transitions for inspection.
    """
    state = {"current": "research", "transitions": [], "signal_counts": {}}

    def router(ctx: MiddlewareContext):
        # Only route on assistant commits (agent-generated content)
        if not ctx.commit or ctx.commit.content_type != "dialogue":
            return
        content = ctx.tract.get_content(ctx.commit)
        if not content:
            return

        # get_content() returns str for dialogue; use it directly
        text = (str(content) if not isinstance(content, dict) else content.get("text", "")).lower()
        if not text:
            return

        # Score each candidate stage by keyword hits
        best_stage = None
        best_hits = 0
        for stage_name, stage_def in stages.items():
            if stage_name == state["current"]:
                continue
            hits = sum(1 for kw in stage_def["keywords"] if kw in text)
            if hits >= min_signals and hits > best_hits:
                best_stage = stage_name
                best_hits = hits

        if best_stage:
            prev = state["current"]
            state["current"] = best_stage

            # Apply stage config
            stage_def = stages[best_stage]
            ctx.tract.configure(stage=best_stage, **stage_def["config"])

            # Update stage directive
            ctx.tract.directive("current-stage", stage_def["directive"])

            state["transitions"].append(f"{prev} -> {best_stage}")
            state["signal_counts"][best_stage] = best_hits

    return router, state


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    print("=" * 60)
    print("Self-Routing Workflow (Middleware-Driven)")
    print("=" * 60)
    print()
    print("  The agent works through a task. Middleware detects when")
    print("  the content shifts from research to implementation to")
    print("  validation and reconfigures automatically.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
    ) as t:

        # Install the routing middleware
        router, route_state = build_router(STAGES)
        t.use("post_commit", router)

        # Initial stage setup
        t.configure(stage="research", **STAGES["research"]["config"])
        t.directive("current-stage", STAGES["research"]["directive"])

        t.system(
            "You are a software engineer working on a task. Work through "
            "it naturally:\n"
            "1. First, research and outline the approach\n"
            "2. Then implement the solution in code\n"
            "3. Finally, write tests to validate it\n\n"
            "Just work -- you do NOT need to call transition or change "
            "any configuration. Focus entirely on the task."
        )

        # The agent works through all stages in a single run.
        # The routing middleware handles transitions automatically.
        print("=== Running Agent ===\n")

        log = StepLogger()
        result = t.run(
            "Design and implement a simple LRU cache in Python. "
            "Start by researching the approach (what data structures, "
            "what operations, edge cases). Then implement it as a class "
            "with get() and put() methods. Finally, write test cases.\n\n"
            "Commit your work at each stage: research notes, then "
            "the implementation code, then the test cases.",
            max_steps=12, max_tokens=2048,
            profile="full",
            tool_names=["commit", "status", "get_config"],
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Report: what happened
        # =============================================================

        print(f"\n=== Routing Report ===\n")

        final_stage = t.get_config("stage") or "research"
        print(f"  Final stage:   {final_stage}")
        print(f"  Transitions:   {len(route_state['transitions'])}")
        for tr in route_state["transitions"]:
            print(f"    {tr}")

        if route_state["transitions"]:
            print(f"\n  The middleware auto-routed {len(route_state['transitions'])} "
                  f"time(s) based on content keywords.")
            print(f"  The agent never called transition() -- it just worked.")
        else:
            print(f"\n  No automatic transitions detected.")
            print(f"  (The agent may not have produced keyword-rich content.)")

        print(f"\n  Commit log:")
        pprint_log(t.log()[-10:])

        print(f"\n  Compiled context:")
        t.compile().pprint(style="compact")


if __name__ == "__main__":
    main()


# --- See also ---
# Explicit transitions:       workflows/01_coding_assistant.py
# Middleware basics:           config_and_middleware/02_event_automation.py
# Autonomous behaviors:       config_and_middleware/08_autonomous_behaviors.py
# Profile-staged workflows:   agent/11_profile_staged_agent.py
