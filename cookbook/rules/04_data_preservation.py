"""Data Preservation Rules

Compression is powerful but destructive. Rules let you protect specific
content from being compressed away:

  - Block compression on tagged commits (tag condition)
  - Require conditions before compression proceeds
  - Use Priority.PINNED annotation for commit-level protection
  - Combine rules for layered defense

Demonstrates: compress trigger, tag condition, block action, require action,
              Priority.PINNED, t.annotate()
"""

from tract import Tract, Priority, DialogueContent


def main():
    with Tract.open() as t:

        # --- Create some conversation history ---

        print("=== Building History ===\n")

        t.register_tag("important", "High-value content")
        t.register_tag("credentials", "Credential-related data")

        t.system("You are a helpful assistant.")
        t.user("What is Python?")
        t.assistant("Python is a high-level programming language.")
        t.user("What about decorators?")
        t.assistant("Decorators are functions that modify other functions.")
        important = t.commit(
            DialogueContent(role="user", text="CRITICAL: Remember this API key format: sk-xxx"),
            tags=["important", "credentials"],
        )
        t.assistant("I will remember that format.")
        t.user("What about list comprehensions?")
        t.assistant("List comprehensions are concise ways to create lists.")

        print(f"  Total commits: {len(t.log())}")
        print(f"  Important commit: {important.commit_hash[:8]} (tagged: {important.tags})")

        # --- Rule: block compression of tagged commits ---

        print("\n=== Rule: Block Compression of Tagged Commits ===\n")

        t.rule(
            "protect-credentials",
            trigger="compress",
            condition={"type": "tag", "tag": "credentials", "present": True},
            action={"type": "block", "reason": "Cannot compress credential data"},
        )

        print("  Rule created: protect-credentials")
        print("  When compression targets a commit tagged 'credentials',")
        print("  the rule engine blocks the operation.")

        # --- Rule: require minimum commit count before compress ---

        print("\n=== Rule: Require Minimum History ===\n")

        t.rule(
            "min-history",
            trigger="compress",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "commit_count",
                    "op": ">=",
                    "value": 5,
                },
            },
        )

        print("  Rule created: min-history")
        print("  Compression requires at least 5 commits in the log.")

        # --- Annotate with PINNED ---

        print("\n=== Priority Annotation ===\n")

        t.annotate(important.commit_hash, Priority.PINNED, reason="credential data")

        print(f"  Pinned commit: {important.commit_hash[:8]}")
        print("  PINNED commits are preserved by the compression engine")
        print("  regardless of rules -- it's a hard guarantee.")

        # --- Rule: protect all important-tagged data ---

        print("\n=== Rule: Protect All Important Data ===\n")

        t.rule(
            "protect-important",
            trigger="compress",
            condition={"type": "tag", "tag": "important", "present": True},
            action={"type": "block", "reason": "Important data must be preserved"},
        )

        # --- Show all compress rules ---

        print("\n=== All Compress Rules ===\n")

        compress_rules = t.rule_index.get_by_trigger("compress")
        for r in compress_rules:
            cond_desc = r.condition["type"] if r.condition else "always"
            print(f"  {r.name:25s} condition={cond_desc:10s} action={r.action['type']}")

        # --- Layered defense summary ---

        print("\n=== Layered Defense Summary ===\n")
        print("  Layer 1: tag-based rules -> block compression of tagged commits")
        print("  Layer 2: threshold rules -> require minimum history size")
        print("  Layer 3: PINNED priority -> hard protection at the engine level")


if __name__ == "__main__":
    main()
