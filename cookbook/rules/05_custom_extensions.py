"""Custom Extensions

Register custom conditions, actions, and metrics. Per-Tract instance,
not global. Custom types work seamlessly with built-in types in any
rule condition or action.

Demonstrates: register_condition(), register_action(), register_metric()
"""

from tract import Tract


class LanguageCondition:
    """Check if commit message contains language-specific markers."""
    MARKERS = {
        "python": ["def ", "import ", "class ", "print("],
        "javascript": ["function ", "const ", "let ", "=>"],
    }
    def evaluate(self, params: dict, ctx) -> bool:
        if ctx.commit is None:
            return False
        text = (ctx.commit.message or "").lower()
        return any(m in text for m in self.MARKERS.get(params.get("language", ""), []))


class IncrementCounterAction:
    """Custom action: increment a named counter."""
    def execute(self, params: dict, ctx):
        from tract.rules.models import ActionResult
        name = params.get("counter", "event_count")
        value = (ctx.tract.get_config(name) or 0) + 1
        return ActionResult("increment_counter", True, {"key": name, "value": value})


class UserMessageCountMetric:
    """Custom metric: count user dialogue commits by inspecting stored content."""
    def compute(self, ctx) -> float:
        import json
        count = 0
        for ci in ctx.tract.log():
            if ci.content_type != "dialogue":
                continue
            blob = ctx.tract._blob_repo.get(ci.content_hash)
            if blob is None:
                continue
            try:
                payload = json.loads(blob.payload_json)
                if payload.get("role") == "user":
                    count += 1
            except (json.JSONDecodeError, TypeError):
                pass
        return float(count)


def main():
    with Tract.open() as t:

        print("=== Registering Custom Extensions ===\n")

        t.register_condition("language", LanguageCondition())
        t.register_action("increment_counter", IncrementCounterAction())
        t.register_metric("user_message_count", UserMessageCountMetric())

        print("  Registered: condition 'language'")
        print("  Registered: action 'increment_counter'")
        print("  Registered: metric 'user_message_count'")

        print("\n=== Custom Condition in a Rule ===\n")

        t.rule(
            "python-content-flag",
            trigger="commit",
            condition={"type": "language", "language": "python"},
            action={"type": "set_config", "key": "has_python", "value": True},
        )

        print("  Rule: flag commits containing Python code")

        print("\n=== Custom Metric in Threshold ===\n")

        t.rule(
            "user-engagement-gate",
            trigger="transition:review",
            action={
                "type": "require",
                "condition": {
                    "type": "threshold",
                    "metric": "user_message_count",
                    "op": ">=",
                    "value": 3,
                },
            },
        )

        print("  Rule: require 3+ user messages before review transition")

        print("\n=== Building Conversation ===\n")

        t.system("You are a Python tutor.")
        t.user("What are generators?")
        t.assistant("Generators yield values lazily.")
        t.user("Show me an example with def and yield.")
        t.assistant("def countdown(n): yield from range(n, 0, -1)")

        print(f"  Total commits: {len(t.log())}")

        print("\n=== Combining Custom + Built-in Conditions ===\n")

        t.rule(
            "complex-gate",
            trigger="compile",
            condition={
                "type": "all",
                "conditions": [
                    {"type": "threshold", "metric": "commit_count", "op": ">", "value": 3},
                    {"type": "threshold", "metric": "user_message_count", "op": ">=", "value": 2},
                ],
            },
            action={"type": "set_config", "key": "rich_context", "value": True},
        )

        print("  Rule: set 'rich_context' when commit_count>3 AND user_messages>=2")

        print("\n=== All Rules ===\n")

        for ci in t.log():
            if ci.content_type == "rule":
                print(f"  {ci.commit_hash[:8]}  {ci.message}")


if __name__ == "__main__":
    main()
