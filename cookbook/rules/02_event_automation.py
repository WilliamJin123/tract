"""Event Automation Rules

Automation rules fire when the agent does something: commit, compile,
compress, merge, or gc. They let you configure automatic reactions --
flagging errors, triggering compression, enforcing limits -- without
manual intervention.

Conditions can be combined with all/any/not for complex logic.

Demonstrates: commit/compile/compress triggers, threshold conditions,
              pattern conditions, operation actions, combined conditions
"""

from tract import Tract


def main():
    with Tract.open() as t:

        # --- Commit-triggered automation ---

        print("=== Commit Automation ===\n")

        # Alert when conversation gets long
        t.rule(
            "high-volume-alert",
            trigger="commit",
            condition={
                "type": "threshold",
                "metric": "commit_count",
                "op": ">",
                "value": 20,
            },
            action={"type": "set_config", "key": "alert", "value": "high-volume"},
        )

        # Detect error patterns in content
        t.rule(
            "detect-error",
            trigger="commit",
            condition={"type": "pattern", "regex": r"(?i)error|exception|traceback"},
            action={"type": "set_config", "key": "has_errors", "value": True},
        )

        # Auto-compress when history is long
        t.rule(
            "auto-compress",
            trigger="commit",
            condition={
                "type": "threshold",
                "metric": "commit_count",
                "op": ">",
                "value": 100,
            },
            action={"type": "operation", "op": "compress", "params": {}},
        )

        print("  Commit rules:")
        for r in t.rule_index.get_by_trigger("commit"):
            print(f"    {r.name} -> {r.action['type']}")

        # --- Compile-triggered automation ---

        print("\n=== Compile Automation ===\n")

        t.rule(
            "compile-mode",
            trigger="compile",
            action={"type": "set_config", "key": "compile_mode", "value": "strict"},
        )

        print("  Compile rules:")
        for r in t.rule_index.get_by_trigger("compile"):
            print(f"    {r.name} -> {r.action['type']}")

        # --- Compress-triggered automation ---

        print("\n=== Compress Automation ===\n")

        t.rule(
            "protect-critical",
            trigger="compress",
            condition={"type": "tag", "tag": "critical", "present": True},
            action={"type": "block", "reason": "Cannot compress critical data"},
        )

        print("  Compress rules:")
        for r in t.rule_index.get_by_trigger("compress"):
            print(f"    {r.name} -> {r.action['type']}")

        # --- Combined conditions ---

        print("\n=== Combined Conditions (all / any / not) ===\n")

        t.rule(
            "strict-guard",
            trigger="commit",
            condition={
                "type": "all",
                "conditions": [
                    {"type": "threshold", "metric": "commit_count", "op": ">", "value": 10},
                    {"type": "not", "condition": {
                        "type": "tag", "tag": "approved", "present": True,
                    }},
                ],
            },
            action={"type": "block", "reason": "Unapproved commits after 10"},
        )

        for r in t.rule_index.get_by_trigger("commit"):
            if r.name == "strict-guard":
                cond = r.condition
                subs = cond.get("conditions", [])
                print(f"  {r.name}: {cond['type']} with {len(subs)} sub-conditions")

        # --- Summary ---

        print("\n=== Summary ===\n")
        for trigger in ["commit", "compile", "compress"]:
            rules = t.rule_index.get_by_trigger(trigger)
            print(f"  {trigger:10s} -> {len(rules)} rule(s)")


if __name__ == "__main__":
    main()
