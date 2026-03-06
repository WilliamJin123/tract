"""Gates and Transitions

Transitions move work between branches. Gates guard them:
  - require: blocks if the inner condition is NOT met
  - block:   blocks if the rule's condition IS met

Two transition trigger forms:
  - "transition"          -- fires on ANY transition from this branch
  - "transition:{target}" -- fires only for a specific target branch

The pipeline: gates (require/block) -> work -> handoff (compile_filter)

Demonstrates: transition triggers, require/block gates, compile_filter,
              pattern-based blocks, combined gate conditions
"""

from tract import Tract


def main():
    with Tract.open() as t:

        # --- Transition gates ---

        print("=== Transition Gates ===\n")

        t.system("You are a coding assistant.")

        # Require at least 6 commits before transitioning to review
        t.rule(
            "review-gate",
            trigger="transition:review",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">=",
                    "value": 6,
                },
            },
        )

        # Generic transition rule: log all transitions
        t.rule(
            "transition-log",
            trigger="transition",
            action={"type": "set_config", "key": "last_transition", "value": "logged"},
        )

        # Compile filter controls what context crosses the boundary
        t.rule(
            "review-filter",
            trigger="transition:review",
            action={"type": "compile_filter", "mode": "full"},
        )

        print("  Rules created:")
        for trig in ["transition", "transition:review"]:
            for r in t.rule_index.get_by_trigger(trig):
                print(f"    [{trig}] {r.name} -> {r.action['type']}")

        # --- Attempt transition too early ---

        print("\n=== Transition Blocked (too few commits) ===\n")

        result = t.transition("review")
        print(f"  Transition result: {result}")
        print(f"  Current branch:    {t.current_branch}")

        # --- Add enough content ---

        t.user("Implement a fibonacci function")
        t.assistant("Here is the implementation...")
        t.user("Add error handling")
        t.assistant("Updated with input validation...")

        print(f"\n  Commit count: {len(t.log())}")

        # --- Transition succeeds ---

        print("\n=== Transition Succeeds ===\n")

        result = t.transition("review")
        if result:
            print(f"  Handoff commit: {result.commit_hash[:8]}")
            print(f"  Current branch: {t.current_branch}")
        else:
            print("  Blocked by rules")

        # --- Pattern-based block gate ---

        print("\n=== Pattern-Based Block ===\n")

        t.switch("main")

        t.rule(
            "no-secrets",
            trigger="commit",
            condition={"type": "pattern", "regex": r"(?i)(api[_-]?key|secret|password)\s*[:=]"},
            action={"type": "block", "reason": "Potential secret detected in commit"},
        )

        print("  Block rule: reject commits containing potential secrets")
        print("  Pattern: api_key, secret, password followed by : or =")

        # --- Combined gates for production ---

        print("\n=== Combined Gate (production) ===\n")

        t.rule(
            "production-gate",
            trigger="transition:production",
            action={
                "type": "require",
                "condition": {
                    "type": "all",
                    "conditions": [
                        {"type": "threshold", "metric": "commit_count", "op": ">=", "value": 3},
                        {"type": "tag", "tag": "approved", "present": True},
                    ],
                },
            },
        )

        print("  Production gate requires: commit_count >= 3 AND tag 'approved'")

        # --- Compress block gate ---

        print("\n=== Compress Block Gate ===\n")

        t.rule(
            "no-compress-early",
            trigger="compress",
            condition={
                "type": "threshold",
                "metric": "commit_count",
                "op": "<",
                "value": 20,
            },
            action={"type": "block", "reason": "Too few commits to compress"},
        )

        print("  Block rule: no compression under 20 commits")

        # --- Gate summary ---

        print("\n=== Gate Summary ===\n")
        for trigger in ["commit", "compress", "transition:review", "transition:production"]:
            rules = t.rule_index.get_by_trigger(trigger)
            gates = [r for r in rules if r.action.get("type") in ("require", "block")]
            if gates:
                print(f"  {trigger}:")
                for g in gates:
                    print(f"    {g.name} -> {g.action['type']}")


if __name__ == "__main__":
    main()
