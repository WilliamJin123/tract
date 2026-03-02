"""Review gate and execute_fn: vet LLM-generated operations before activation.

When LLMs generate OperationSpecs at runtime, the host application must
review the code before activation. This demo shows:

Part 1 -- review=True on register_operation() (inspect class without activating)
Part 2 -- execute_fn on fire() (side effects with provenance recording)
Part 3 -- Unregistering an operation (cleanup)

Demonstrates: register_operation(review=True), fire(execute_fn=...),
              unregister_operation(), spec_to_dict(), spec_from_dict()
"""

from tract import Tract
from tract.hooks.dynamic import ActionDef, OperationSpec, spec_from_dict, spec_to_dict


def review_and_execute() -> None:
    # -----------------------------------------------------------------
    # Part 1 -- Review gate
    # -----------------------------------------------------------------
    print("=" * 60)
    print("PART 1 -- review=True on register_operation()")
    print("=" * 60)
    print()
    print("  review=True compiles the spec and returns the Pending class")
    print("  without activating the operation. Inspect it, then register for real.")

    with Tract.open() as t:
        # Imagine an LLM generated this spec
        spec = OperationSpec(
            name="test_gate",
            description="Run tests before committing code changes",
            fields={
                "test_suite": {"type": "str", "default": "unit"},
                "passed": {"type": "bool", "default": False},
            },
            actions={
                "run": ActionDef(
                    name="run",
                    description="Execute the test suite",
                    params={"verbose": "bool"},
                    required=[],
                    code='''
pending.fields["passed"] = True
pending.approve()
''',
                ),
            },
        )

        # Review first -- get the class without activating
        cls = t.register_operation(spec, review=True)
        print(f"\n  Class name: {cls.__name__}")
        print(f"  Docstring: {cls.__doc__}")
        print(f"  Inherits from Pending: {issubclass(cls, __import__('tract.hooks.pending', fromlist=['Pending']).Pending)}")

        # Verify it's NOT hookable yet
        try:
            t.on("test_gate", lambda p: p.approve())
            print("  ERROR: Should not be hookable yet!")
        except ValueError:
            print("  Correctly NOT hookable yet (review only)")

        # Satisfied with inspection -- register for real
        t.register_operation(spec)
        print("  Now registered and hookable")
        t.on("test_gate", lambda p: p.run(), name="auto-run")

        result = t.fire("test_gate")
        print(f"  Fire result: status={result.status}, passed={result.fields['passed']}")

        # -----------------------------------------------------------------
        # Part 2 -- execute_fn with provenance
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 2 -- execute_fn (side effects with provenance)")
        print("=" * 60)
        print()
        print("  execute_fn is called when the operation is approved.")
        print("  If an event_repo exists, an OperationEventRow is written.")

        side_effects = []

        def deploy_action(pending):
            """Simulate deployment side effect."""
            side_effects.append(f"deployed:{pending.fields.get('test_suite')}")
            return {"deployed": True, "suite": pending.fields.get("test_suite")}

        # fire() with execute_fn returns the execute_fn's result
        result = t.fire(
            "test_gate",
            fields={"test_suite": "integration"},
            execute_fn=deploy_action,
        )
        print(f"\n  fire() returned: {result}")
        print(f"  Side effects: {side_effects}")

        # -----------------------------------------------------------------
        # Part 3 -- Serialization roundtrip
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 3 -- Serialize / Deserialize OperationSpec")
        print("=" * 60)
        print()
        print("  spec_to_dict / spec_from_dict for persistence.")

        d = spec_to_dict(spec)
        print(f"\n  Serialized keys: {list(d.keys())}")
        print(f"  Actions: {list(d['actions'].keys())}")

        restored = spec_from_dict(d)
        print(f"  Restored: name={restored.name}, fields={list(restored.fields.keys())}")

        # -----------------------------------------------------------------
        # Part 4 -- Unregister
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 4 -- Unregister an Operation")
        print("=" * 60)
        print()
        print("  unregister_operation() removes the op, its handlers, and hookability.")

        t.unregister_operation("test_gate")
        print("  Unregistered 'test_gate'")

        try:
            t.fire("test_gate")
            print("  ERROR: Should have raised!")
        except ValueError as e:
            print(f"  Correctly raises: {e}")

        try:
            t.on("test_gate", lambda p: p.approve())
            print("  ERROR: Should not be hookable!")
        except ValueError:
            print("  Correctly not hookable after unregister")

        t.print_hooks()


if __name__ == "__main__":
    review_and_execute()
