"""Semantic Maintenance: LLM-Powered Context Housekeeping

A semantic maintainer takes actions on the context (annotate, configure,
tag, etc.) unlike gates which block. This example runs a research workflow
where a post_commit maintainer monitors context health -- marking stale
commits as SKIP and reconfiguring the stage when it detects a phase shift.

The maintainer is NOT scripted. It makes a real LLM call to decide what
actions to take. If the context is clean, it takes zero actions.

Demonstrates: t.middleware.maintain(), condition callbacks, last_result observability,
list_maintainers().

Requires: LLM API key (uses Cerebras provider)
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


def _get_maintainer_handler(t: Tract, name: str):
    """Retrieve the SemanticMaintainer object from the middleware system."""
    handler_id = t._maintainers.get(name)
    if not handler_id:
        return None
    for _event, handlers in t._middleware.items():
        for hid, handler in handlers:
            if hid == handler_id:
                return handler
    return None


def _print_result(handler) -> None:
    """Print a maintainer's last_result if available."""
    if not handler or not handler.last_result:
        print("  Maintainer has not fired yet (condition not met).")
        return
    r = handler.last_result
    print(f"    reasoning:        {r.reasoning[:120]}")
    print(f"    actions requested: {r.actions_requested}")
    print(f"    actions executed:  {r.actions_executed}")
    print(f"    actions failed:    {r.actions_failed}")
    print(f"    tokens used:       {r.tokens_used}")
    for err in r.errors:
        print(f"    error: {err}")


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    print("=" * 70)
    print("Semantic Maintenance: LLM-Powered Context Housekeeping")
    print("=" * 70)
    print()

    log = StepLogger()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
    ) as t:
        # --- Setup --------------------------------------------------------
        t.system(
            "You are a supply chain research analyst. Investigate strategies "
            "for building resilient supply chains. Commit each distinct "
            "finding separately so the research log shows clear progression."
        )
        t.config.set(stage="research")

        # --- Register semantic maintainer ---------------------------------
        t.middleware.maintain(
            name="context-health",
            event="post_commit",
            instructions=(
                "Review the commit log for context health.\n"
                "1. Annotate redundant commits (restating earlier content) as SKIP.\n"
                "2. If research has shifted from exploration to recommendations, "
                "configure stage='synthesis'.\n"
                "Be conservative: only act on clear redundancy or phase shifts."
            ),
            actions=["annotate", "configure"],
            model=llm.small,
            condition=lambda ctx: len(ctx.tract.search.log()) > 5,
        )

        print(f"  Branch: {t.current_branch}")
        print(f"  Maintainers: {t.middleware.list_maintainers()}")
        print(f"  Fires on: post_commit (when log > 5 entries)")

        # --- Phase 1: Broad research --------------------------------------
        print("\n" + "=" * 70)
        print("Phase 1: Broad Research")
        print("=" * 70 + "\n")

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

        entries = t.search.log(limit=20)
        print(f"\n  Commits after Phase 1: {len(entries)}")
        for entry in entries[:6]:
            msg = (entry.message or "(no message)")[:60]
            prio = entry.priority.name if hasattr(entry, "priority") else "?"
            print(f"    [{entry.commit_hash[:8]}] {prio:8s} \"{msg}\"")

        print(f"\n  Maintainer last_result after Phase 1:")
        _print_result(_get_maintainer_handler(t, "context-health"))

        # --- Phase 2: Specific recommendations ----------------------------
        print("\n" + "=" * 70)
        print("Phase 2: Specific Recommendations")
        print("=" * 70 + "\n")

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

        # --- Final report -------------------------------------------------
        print("\n" + "=" * 70)
        print("Final State")
        print("=" * 70 + "\n")

        entries = t.search.log(limit=30)
        print(f"  Branch: {t.current_branch}")
        print(f"  Total commits: {len(entries)}")
        print(f"  Maintainers: {t.middleware.list_maintainers()}")
        print(f"  Tokens: {t.search.status().token_count}")

        print(f"\n  Maintainer 'context-health' last_result:")
        _print_result(_get_maintainer_handler(t, "context-health"))

        print(f"\n  Compiled context:")
        t.compile().pprint(style="compact")

        skip_count = sum(
            1 for e in entries
            if hasattr(e, "priority") and e.priority.name == "SKIP"
        )
        print(f"\n  Commits marked SKIP by maintainer: {skip_count}")
        print(f"  Current stage: {t.config.get('stage')}")

    # --- Why semantic maintenance matters ---------------------------------
    print("\n" + "=" * 70)
    print("WHY SEMANTIC MAINTENANCE")
    print("=" * 70)
    print("""
  Deterministic maintenance (count-based TTL, fixed thresholds):
    - Rigid: marks the Nth-oldest commit as stale regardless of content
    - Blind to meaning: a critical early finding gets evicted by age
    - No phase awareness: cannot detect research-to-synthesis shifts

  Semantic maintenance (LLM evaluates context health):
    - Evaluates CONTENT to decide what is truly redundant
    - Preserves important early findings even when old
    - Detects phase shifts from the meaning of recent commits
    - Condition callbacks skip the LLM call when context is small

  Cost: one cheap LLM call per invocation (~500-2000 tokens).
  Fail-open: if the LLM call fails, no actions are taken.
""")

if __name__ == "__main__":
    main()

# --- See also ---
# Semantic quality gates:    agent/08_semantic_gates.py
# Event-driven middleware:   config_and_middleware/02_event_automation.py
