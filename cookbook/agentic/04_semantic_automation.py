"""Semantic Automation: LLM-Powered Gates and Maintenance

Two middleware patterns that use LLM judgment instead of deterministic rules:

  Semantic Gates -- block operations until the LLM judges that a quality
  criterion is met. The gate evaluates the *meaning* of the context, not
  just metadata. Cheap model, condition callbacks for efficiency, fail-open
  on errors.

  Context Health Monitor -- take housekeeping actions (annotate, configure,
  tag, compress, directive) based on LLM judgment. The monitor tracks context
  health and acts when it detects redundancy or phase shifts. Two-pass
  peeking for content-aware decisions.

Both patterns share the same manifest-based architecture: register once,
the middleware system handles invocation timing, condition pre-checks,
and error recovery.

Sections:
  1. Semantic Gate: Quality-Gated Transitions
  2. Context Health Monitor

Demonstrates: t.middleware.gate(), t.middleware.maintain(),
              t.middleware.list_gates(), t.middleware.list_maintainers(),
              condition callbacks, BlockedError recovery, MaintainResult,
              fail-open error handling

Requires: LLM API key (uses claude_code provider)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# =====================================================================
# Helpers
# =====================================================================

def _section(num: int, title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {num}. {title}")
    print("=" * 70)
    print()


def _get_maintainer_handler(t: Tract, name: str):
    """Retrieve the SemanticMaintainer object via public API."""
    return t.middleware.get_maintainer(name)


def _print_maintainer_result(handler) -> None:
    """Print a maintainer's last_result if available."""
    if not handler or not handler.last_result:
        print("    Maintainer has not fired yet (condition not met).")
        return
    r = handler.last_result
    print(f"    reasoning:         {r.reasoning[:120]}")
    print(f"    actions requested: {r.actions_requested}")
    print(f"    actions executed:  {r.actions_executed}")
    print(f"    actions failed:    {r.actions_failed}")
    print(f"    tokens used:       {r.tokens_used}")
    for err in r.errors:
        print(f"    error: {err}")


# =====================================================================
# Section 1: Semantic Gate -- Quality-Gated Transitions
# =====================================================================

def semantic_gate_transitions():
    """Register a semantic gate that evaluates research quality via LLM.

    The gate fires on pre_transition, but ONLY when transitioning to
    "synthesis". It uses a cheap model to judge whether the research
    commits show substantive analysis from multiple angles.
    """

    _section(1, "Semantic Gate: Quality-Gated Transitions")

    print("  A semantic gate evaluates research QUALITY via LLM judgment,")
    print("  not just commit count. The agent must produce genuinely diverse")
    print("  analysis before the gate allows transition to synthesis.")
    print()

    log = StepLogger()

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
        auto_message=llm.small,
    ) as t:
        # --- Setup: system prompt and stages ---
        t.system(
            "You are a technology research analyst. Your job is to produce "
            "thorough, multi-perspective analysis. When researching a topic, "
            "commit each distinct finding or perspective separately so your "
            "research log shows breadth and depth."
        )
        t.config.set(stage="research")
        t.branch("synthesis", switch=False)

        # --- Register the semantic gate ---
        # Condition callback: only fire when transitioning to "synthesis".
        # This avoids an LLM call for transitions to other branches.
        t.middleware.gate(
            "research-depth",
            event="pre_transition",
            check=(
                "Does the research contain substantive analysis from at least "
                "2 genuinely different angles or perspectives? Look at the "
                "commit messages and content types. The commits should cover "
                "distinct viewpoints (e.g., technical trade-offs AND organizational "
                "impact, or performance AND developer experience). Repetition of "
                "the same angle with different wording does NOT count. "
                "Be strict: superficial one-liners do not count as substantive."
            ),
            model=llm.small,
            condition=lambda ctx: ctx.target == "synthesis",
        )

        print(f"  Branch: {t.current_branch}")
        print(f"  Gates registered: {t.middleware.list_gates()}")
        print(f"  Gate fires on: pre_transition (to synthesis only)")

        # --- Phase 1: Initial research ---
        print()
        print("  " + "-" * 60)
        print("  Phase 1: Initial Research")
        print("  " + "-" * 60)

        result = t.llm.run(
            "Research the trade-offs of microservices vs monolith architecture. "
            "Focus on ONE specific angle (e.g., deployment complexity OR data "
            "consistency). Commit your findings as you go. Keep it focused.",
            max_steps=6,
            max_tokens=1024,
            profile="full",
            tool_names=["commit", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        result.pprint()

        entries = t.log(limit=20)
        print(f"\n  Commits after Phase 1: {len(entries)}")
        for entry in entries[:5]:
            msg = (entry.message or "(no message)")[:60]
            print(f"    [{entry.commit_hash[:8]}] {entry.content_type:12s} \"{msg}\"")

        # --- Attempt transition: gate evaluates quality ---
        print()
        print("  " + "-" * 60)
        print("  Transition Attempt 1: Does the gate pass?")
        print("  " + "-" * 60)

        gate_blocked = False
        try:
            t.transition("synthesis")
            print("  Gate PASSED on first attempt -- research deemed sufficient.")
            print("  (The LLM judge found enough depth/diversity in Phase 1.)")
        except BlockedError as e:
            gate_blocked = True
            print(f"  Gate BLOCKED: {e.reasons[0]}")
            print()
            print("  The semantic gate judged that the research lacks diversity.")
            print("  The agent must research from a DIFFERENT angle before retrying.")

        # --- Phase 2: Deeper research (if blocked) ---
        if gate_blocked:
            print()
            print("  " + "-" * 60)
            print("  Phase 2: Expanding Research (Different Angle)")
            print("  " + "-" * 60)

            result = t.llm.run(
                "The quality gate blocked our transition to synthesis because "
                "the research lacks diverse perspectives. Research from a "
                "COMPLETELY DIFFERENT angle than before. If you covered "
                "technical trade-offs, now cover organizational/team impact. "
                "If you covered deployment, now cover data management. "
                "Commit your findings.",
                max_steps=6,
                max_tokens=1024,
                profile="full",
                tool_names=["commit", "status"],
                on_step=log.on_step,
                on_tool_result=log.on_tool_result,
            )
            result.pprint()

            entries = t.log(limit=20)
            print(f"\n  Total commits after Phase 2: {len(entries)}")
            for entry in entries[:8]:
                msg = (entry.message or "(no message)")[:60]
                print(f"    [{entry.commit_hash[:8]}] {entry.content_type:12s} \"{msg}\"")

            # --- Retry transition ---
            print()
            print("  " + "-" * 60)
            print("  Transition Attempt 2: Retry after deeper research")
            print("  " + "-" * 60)

            try:
                t.transition("synthesis")
                print("  Gate PASSED on retry -- research now has sufficient diversity.")
            except BlockedError as e:
                print(f"  Gate BLOCKED again: {e.reasons[0]}")
                print("  (In production, you would loop: research more, retry.)")

        # --- Final state ---
        print()
        print("  " + "-" * 60)
        print("  Final State")
        print("  " + "-" * 60)

        print(f"  Current branch: {t.current_branch}")
        entries = t.log(limit=50)
        print(f"  Total commits: {len(entries)}")
        print(f"  Gates: {t.middleware.list_gates()}")
        status = t.status()
        print(f"  Tokens: {status.token_count}")

        print(f"\n  Compiled context:")
        t.compile().pprint(style="chat")

        reached_synthesis = t.current_branch == "synthesis"
        print(f"\n  Reached synthesis: {reached_synthesis}")
        if gate_blocked:
            print("  Gate blocked initial transition: YES")
            print("  Agent adapted with deeper research: YES")
        else:
            print("  Gate blocked initial transition: NO (research was sufficient)")

    print()
    print("  PASSED")


# =====================================================================
# Section 2: Context Health Monitor
# =====================================================================

def semantic_maintainer():
    """Register a context health monitor that maintains context quality.

    The maintainer fires on post_commit and uses LLM judgment to:
      - Annotate redundant commits as SKIP
      - Compress clusters of related commits into summaries
      - Reconfigure stage when it detects a phase shift
      - Add directives when a clear research focus emerges
      - Tag commits that represent key findings
    A condition callback skips the LLM call when the log is still small.
    """

    _section(2, "Context Health Monitor")

    log = StepLogger()

    with Tract.open(
        **llm.tract_kwargs(MODEL_ID),
        auto_message=llm.small,
    ) as t:
        # --- Setup ---
        t.system(
            "You are a supply chain research analyst. Investigate strategies "
            "for building resilient supply chains. Commit each distinct "
            "finding separately so the research log shows clear progression."
        )
        t.config.set(stage="research")

        # --- Register context health monitor ---
        # Condition: only fire when log has more than 5 entries.
        # Actions: annotate, compress, configure, directive, tag.
        t.middleware.maintain(
            name="context-health",
            event="post_commit",
            instructions=(
                "Review the commit log for context health.\n"
                "1. Annotate redundant commits (restating earlier content) as SKIP.\n"
                "2. If research has shifted to recommendations, configure stage='synthesis'.\n"
                "3. Use compress_range to summarize contiguous runs of related commits.\n"
                "4. Edit verbose commits to shorten them (keep key facts, drop filler).\n"
                "5. Add a directive when you detect a clear research focus emerging.\n"
                "6. Tag commits that represent key findings with 'key-finding'.\n"
                "Be conservative: only act on clear signals."
            ),
            actions=["annotate", "compress_range", "configure", "directive", "edit", "tag"],
            model=llm.small,
            condition=lambda ctx: len(ctx.tract.log()) > 5,
        )

        print(f"  Branch: {t.current_branch}")
        print(f"  Maintainers: {t.middleware.list_maintainers()}")
        print(f"  Fires on: post_commit (when log > 5 entries)")

        # --- Phase 1: Broad research ---
        print()
        print("  " + "-" * 60)
        print("  Phase 1: Broad Research")
        print("  " + "-" * 60)

        result = t.llm.run(
            "Research strategies for building resilient supply chains. "
            "Cover geographic diversification, inventory management, and "
            "supplier relationships. Commit each finding as you go.",
            max_steps=8,
            max_tokens=1024,
            profile="full",
            tool_names=["commit", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        result.pprint()

        entries = t.log(limit=20)
        print(f"\n  Commits after Phase 1: {len(entries)}")
        for entry in entries[:6]:
            msg = (entry.message or "(no message)")[:60]
            prio = entry.priority.name if hasattr(entry, "priority") else "?"
            print(f"    [{entry.commit_hash[:8]}] {prio:8s} \"{msg}\"")

        print(f"\n  Maintainer last_result after Phase 1:")
        _print_maintainer_result(_get_maintainer_handler(t, "context-health"))

        # --- Phase 2: Specific recommendations ---
        print()
        print("  " + "-" * 60)
        print("  Phase 2: Specific Recommendations")
        print("  " + "-" * 60)

        result = t.llm.run(
            "Shift from broad research to specific, actionable recommendations "
            "for a mid-size manufacturer. Commit each recommendation.",
            max_steps=8,
            max_tokens=1024,
            profile="full",
            tool_names=["commit", "status"],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # --- Final state ---
        print()
        print("  " + "-" * 60)
        print("  Final State")
        print("  " + "-" * 60)

        entries = t.log(limit=30)
        print(f"  Branch: {t.current_branch}")
        print(f"  Total commits: {len(entries)}")
        print(f"  Maintainers: {t.middleware.list_maintainers()}")
        print(f"  Tokens: {t.status().token_count}")

        print(f"\n  Maintainer 'context-health' last_result:")
        _print_maintainer_result(_get_maintainer_handler(t, "context-health"))

        print(f"\n  Compiled context:")
        t.compile().pprint(style="chat")

        skip_count = sum(
            1 for e in entries
            if hasattr(e, "priority") and e.priority.name == "SKIP"
        )
        print(f"\n  Commits marked SKIP by maintainer: {skip_count}")
        print(f"  Current stage: {t.config.get('stage')}")

    print()
    print("  PASSED")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return

    semantic_gate_transitions()
    semantic_maintainer()

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print()
    print("  Section  Pattern                          Tract API Used")
    print("  -------  ------------------------------   ----------------------------------")
    print("  1        Semantic gate transitions         middleware.gate(), BlockedError")
    print("  2        Context health monitor             middleware.maintain(), MaintainResult")
    print()
    print("Done.")


# Alias for pytest discovery
test_semantic_automation = main


if __name__ == "__main__":
    main()


# --- See also ---
# Implicit discovery (LLM):       agentic/01_implicit_discovery.py
# Adversarial review:              agentic/05_adversarial_review.py
