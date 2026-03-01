"""Dynamic operation introspection: to_dict, to_tools, describe_api.

Dynamic operations get full introspection for free. Compiled actions
have proper type annotations, so to_tools() produces correct JSON Schema.
The fields dict appears in to_dict() output alongside status and actions.

Part 1 -- to_dict() includes dynamic fields
Part 2 -- to_tools() generates tool schemas for custom actions
Part 3 -- describe_api() produces human-readable docs
Part 4 -- apply_decision() dispatches by name (agent loop pattern)

Demonstrates: to_dict(), to_tools(), describe_api(), apply_decision(),
              execute_tool(), type annotations on compiled actions
"""

from tract import Tract
from tract.hooks.dynamic import ActionDef, OperationSpec


def _make_quality_spec() -> OperationSpec:
    """Build a quality-check operation with typed params."""
    return OperationSpec(
        name="quality_check",
        description="Validate context quality before compression",
        fields={
            "min_tokens": {"type": "int", "default": 100},
            "max_ratio": {"type": "float", "default": 0.5},
            "passed": {"type": "bool", "default": False},
        },
        actions={
            "check": ActionDef(
                name="check",
                description="Verify context meets quality threshold",
                params={"threshold": "int", "strict": "bool"},
                required=["threshold"],
                code='''
compiled = pending.tract.compile()
total = compiled.token_count
pending.fields["passed"] = total >= threshold
if pending.fields["passed"]:
    pending.approve()
elif strict:
    pending.reject(f"Context too short: {total} tokens < {threshold}")
else:
    pending.approve()
''',
            ),
            "override": ActionDef(
                name="override",
                description="Force-pass the quality check",
                params={"reason": "str"},
                required=["reason"],
                code='''
pending.fields["passed"] = True
pending.approve()
''',
            ),
        },
    )


def introspection_demo() -> None:
    with Tract.open() as t:
        t.system("You are a helpful assistant.")
        t.user("Hello, how are you?")
        t.assistant("I'm doing well!")

        spec = _make_quality_spec()
        t.register_operation(spec)

        # Get a Pending for inspection
        pending = t.fire("quality_check", review=True)

        # -----------------------------------------------------------------
        # Part 1 -- to_dict()
        # -----------------------------------------------------------------
        print("=" * 60)
        print("PART 1 -- to_dict()")
        print("=" * 60)
        print()
        print("  Dynamic fields appear alongside operation metadata.")

        d = pending.to_dict()
        print(f"\n  operation: {d['operation']}")
        print(f"  status: {d['status']}")
        print(f"  fields: {d['fields']}")
        print(f"  available_actions: {d['available_actions']}")

        # -----------------------------------------------------------------
        # Part 2 -- to_tools()
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 2 -- to_tools()")
        print("=" * 60)
        print()
        print("  Each action becomes a JSON Schema tool definition.")
        print("  Type annotations from ActionDef.params map to JSON Schema types.")

        tools = pending.to_tools()
        for tool in tools:
            fn = tool["function"]
            params = fn.get("parameters", {})
            props = params.get("properties", {})
            req = params.get("required", [])
            print(f"\n  {fn['name']}:")
            print(f"    description: {fn['description']}")
            for pname, pdef in props.items():
                marker = " (required)" if pname in req else ""
                print(f"    {pname}: {pdef.get('type', '?')}{marker}")

        # -----------------------------------------------------------------
        # Part 3 -- describe_api()
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 3 -- describe_api()")
        print("=" * 60)
        print()
        print("  Markdown-formatted API docs, ready for an LLM system prompt.")

        api_desc = pending.describe_api()
        for line in api_desc.split("\n"):
            print(f"  {line}")

        # -----------------------------------------------------------------
        # Part 4 -- apply_decision() dispatch
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 4 -- apply_decision() dispatch")
        print("=" * 60)
        print()
        print("  LLM returns a decision dict; apply_decision() routes it.")

        # Simulate LLM choosing the "check" action
        pending2 = t.fire("quality_check", review=True)
        decision = {"action": "check", "args": {"threshold": 10, "strict": False}}
        print(f"\n  Decision: {decision}")
        pending2.apply_decision(decision)
        print(f"  Status after: {pending2.status}")
        print(f"  Passed: {pending2.fields['passed']}")

        # Simulate LLM choosing "override"
        pending3 = t.fire("quality_check", review=True)
        decision2 = {"action": "override", "args": {"reason": "Admin bypass"}}
        print(f"\n  Decision: {decision2}")
        pending3.apply_decision(decision2)
        print(f"  Status after: {pending3.status}")
        print(f"  Passed: {pending3.fields['passed']}")


if __name__ == "__main__":
    introspection_demo()
