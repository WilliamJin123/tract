"""Rules: Behavior as Data

Rules are first-class commits that configure behavior. Instead of hardcoding
settings in Python, you commit rules to the tract -- they travel with the
conversation, follow DAG precedence, and are visible in the log.

Three patterns:
  1. Config rules    (trigger="active")  -- key-value settings, always in scope
  2. Event rules     (trigger="commit")  -- fire when something happens
  3. Guard rules     (trigger="compress") -- protect data from operations

Demonstrates: t.rule(), t.get_config(), resolve_all_configs(), conditions,
              actions, DAG precedence

No LLM required.
"""

from tract import Tract, Priority, resolve_all_configs


def main():
    with Tract.open() as t:

        # --- 1. Config rules: key-value settings as commits ---

        print("=== Config Rules ===\n")

        t.rule(
            "model-selection",
            trigger="active",
            action={"type": "set_config", "key": "model", "value": "gpt-4o"},
        )
        t.rule(
            "temperature",
            trigger="active",
            action={"type": "set_config", "key": "temperature", "value": 0.7},
        )

        print(f"  model:       {t.get_config('model')}")
        print(f"  temperature: {t.get_config('temperature')}")
        print(f"  missing key: {t.get_config('nonexistent', 'default-val')}")

        # Resolve all active configs at once
        all_cfg = resolve_all_configs(t.rule_index)
        print(f"  all configs: {all_cfg}")

        # --- 2. Event rules: react when something happens ---

        print("\n=== Event Rules ===\n")

        # Block commits over 500 tokens
        t.rule(
            "size-guard",
            trigger="commit",
            condition={"type": "threshold", "metric": "token_count", "op": ">", "value": 500},
            action={"type": "block", "reason": "Commit too large (>500 tokens)"},
        )

        # Log how many rules are active
        entries = t.rule_index.get_by_trigger("commit")
        print(f"  Commit-trigger rules: {len(entries)}")
        for e in entries:
            print(f"    - {e.name} (distance={e.dag_distance})")

        # --- 3. Data preservation: protect tagged commits ---

        print("\n=== Data Preservation Rules ===\n")

        t.rule(
            "protect-important",
            trigger="compress",
            condition={"type": "tag", "tag": "important", "present": True},
            action={"type": "block", "reason": "Cannot compress important data"},
        )

        # Tag a commit as important
        t.user("This is critical context that must survive compression.")
        important = t.log()[-1]
        t.annotate(important.commit_hash, Priority.PINNED, reason="critical")

        print(f"  Protected commit: {important.commit_hash[:8]}")
        print(f"  Compress-trigger rules: {len(t.rule_index.get_by_trigger('compress'))}")

        # --- 4. DAG precedence: closer to HEAD wins ---

        print("\n=== DAG Precedence ===\n")

        t.rule(
            "model-selection",
            trigger="active",
            action={"type": "set_config", "key": "model", "value": "claude-sonnet"},
        )

        # Same name "model-selection" -- closer rule overrides the earlier one
        print(f"  model (after override): {t.get_config('model')}")
        print(f"  temperature (unchanged): {t.get_config('temperature')}")

        # Show commit log -- rules are visible
        print("\n=== Log (rules are commits) ===\n")
        for ci in t.log():
            print(f"  {ci.commit_hash[:8]}  {ci.content_type:10s}  {ci.message}")


if __name__ == "__main__":
    main()


# --- See also ---
# Quick start:       getting_started/01_quick_start.py
# Custom tools:      getting_started/03_custom_tools.py
# Workflow rules:    workflows/01_coding_assistant.py
