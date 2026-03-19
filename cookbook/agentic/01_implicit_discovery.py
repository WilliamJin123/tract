"""Adaptive Agent Behavior: Middleware Constraints That Shape Agent Actions

Two scenarios where middleware constraints force an LLM agent to adapt at
runtime -- without knowing about the constraints in advance.

Scenarios:
  1. Quality Gate    -- pre_transition blocks advancement until enough research
                        artifacts exist. Agent recovers from BlockedError.
  2. Phase Detection -- post_commit detects shift from exploratory to precise
                        work, lowers temperature + injects precision directive.

Demonstrates: middleware.add(), BlockedError recovery, dynamic config.set(),
              dynamic directive(), MiddlewareContext, t.llm.run()
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError, MiddlewareContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


def _section(num: int, title: str, desc: str) -> None:
    print(f"\n{'=' * 70}\n  {num}. {title}\n{'=' * 70}\n  {desc}\n")


# ---------------------------------------------------------------------------
# Scenario 1 -- Quality Gate: BlockedError Recovery
# ---------------------------------------------------------------------------

def scenario_quality_gate() -> None:
    _section(1, "Quality Gate: BlockedError Recovery",
             "pre_transition blocks until 3 artifacts are committed.")

    log = StepLogger()

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:
        t.system(
            "You are a software engineer working on an API project. "
            "Research topics thoroughly before moving to implementation."
        )
        t.config.set(stage="research")
        t.branch("implementation", switch=False)

        # Gate: require >= 3 artifact commits before transition
        def research_gate(ctx: MiddlewareContext):
            if ctx.target != "implementation":
                return
            entries = ctx.tract.log(limit=50)
            artifacts = [e for e in entries if e.content_type == "artifact"]
            if len(artifacts) < 3:
                raise BlockedError(
                    "pre_transition",
                    f"Research incomplete: {len(artifacts)} artifact(s) "
                    f"committed, need at least 3.",
                )

        t.middleware.add("pre_transition", research_gate)
        print(f"  Branch: {t.current_branch}")
        print(f"  Gate: pre_transition requires >= 3 artifact commits\n")

        result = t.llm.run(
            "Research authentication patterns, database schema design, and "
            "error handling for a REST API. Commit each finding as an "
            "artifact. When your research is thorough, transition to "
            "implementation.",
            max_steps=15, max_tokens=1024,
            profile="full",
            tool_names=["commit", "transition", "status", "log"],
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # --- Results ---
        entries = t.log(limit=50)
        artifacts = [e for e in entries if e.content_type == "artifact"]
        reached = t.current_branch == "implementation"

        print(f"\n  Artifacts committed: {len(artifacts)}")
        for entry in artifacts[:5]:
            print(f"    [{entry.commit_hash[:8]}] {(entry.message or '')[:60]}")
        print(f"  Reached implementation: {reached}")

        if not reached:
            print("  Forcing transition attempt to demo the gate...")
            try:
                t.transition("implementation")
                print("  Transition succeeded.")
            except BlockedError as e:
                print(f"  Gate blocked: {e}")

        s = t.status()
        print(f"  Final: {s.commit_count} commits, {s.token_count} tokens")
    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Scenario 2 -- Phase Detection: Dynamic Config Adjustment
# ---------------------------------------------------------------------------

def scenario_phase_detection() -> None:
    _section(2, "Phase Detection: Dynamic Config Adjustment",
             "post_commit detects exploratory->precise shift, lowers temp.")

    log = StepLogger()
    phase_state = {"shifted": False, "shift_at": None}

    IMPL_SIGNALS = [
        "def ", "class ", "CREATE TABLE", "INSERT INTO", "SELECT ",
        "import ", "function ", "```python", "```sql", "```json",
    ]

    def detect_phase_shift(ctx: MiddlewareContext):
        """Lower temperature and inject directive when code appears."""
        if phase_state["shifted"] or not ctx.commit:
            return
        content = ctx.tract.get_content(ctx.commit)
        if not content:
            return
        text = str(content) if not isinstance(content, dict) else content.get("text", "")
        hits = sum(1 for sig in IMPL_SIGNALS if sig in text)
        if hits >= 2:
            phase_state["shifted"] = True
            phase_state["shift_at"] = ctx.commit.commit_hash[:8]
            ctx.tract.config.set(temperature=0.2, stage="implementation")
            ctx.tract.directive(
                "precision-mode",
                "Phase shift detected: you are now producing implementation "
                "artifacts. Be precise. Provide exact code, schemas, and "
                "configurations. No hedging.",
            )

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:
        t.system(
            "You are a backend engineer designing and building a user "
            "authentication service. Start with broad research, then "
            "produce concrete implementation artifacts."
        )
        t.config.set(stage="research", temperature=0.9)
        t.middleware.add("post_commit", detect_phase_shift)

        print(f"  Config: stage=research, temperature=0.9")
        print(f"  Middleware: post_commit watches for implementation signals\n")

        result = t.llm.run(
            "Design an authentication service:\n"
            "1. Research trade-offs between JWT and session-based auth. "
            "Commit your analysis.\n"
            "2. Write the implementation: a Python auth module with "
            "login/logout/verify and the database schema. Commit the code.\n\n"
            "Use the commit tool for each deliverable.",
            max_steps=12, max_tokens=2048,
            profile="full",
            tool_names=["commit", "status", "get_config"],
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # --- Results ---
        if phase_state["shifted"]:
            print(f"\n  Phase shift at commit: {phase_state['shift_at']}")
            print(f"  Config after: temperature={t.config.get('temperature')}, "
                  f"stage={t.config.get('stage')}")
        else:
            print(f"\n  No phase shift detected.")
            print(f"  Config: temperature={t.config.get('temperature')}, "
                  f"stage={t.config.get('stage')}")

        directives = [e for e in t.log(limit=30)
                      if e.content_type == "directive"]
        if directives:
            print(f"  Directives injected: {len(directives)}")
            for d in directives:
                print(f"    [{d.commit_hash[:8]}] {(d.message or '')[:60]}")

        s = t.status()
        print(f"  Final: {s.commit_count} commits, {s.token_count} tokens\n")
        t.compile().pprint(style="chat")
    print("\n  PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    print("=" * 70)
    print("  Adaptive Agent Behavior")
    print("  Middleware constraints that shape agent actions at runtime.")
    print("=" * 70)

    scenario_quality_gate()
    scenario_phase_detection()

    print(f"\n{'=' * 70}")
    print("  Both patterns work because the agent never needs to know about")
    print("  the constraints. Middleware enforces invariants (Scenario 1) or")
    print("  adapts the environment (Scenario 2); the agent responds naturally.")
    print(f"{'=' * 70}")
    print("\nDone.")


test_adaptive_behavior = main

if __name__ == "__main__":
    main()


# --- See also ---
# Semantic gates (LLM-judged):    agentic/04_semantic_automation.py
# Error recovery:                  agentic/10_error_recovery.py
# Middleware reference:             reference/02_middleware.py
