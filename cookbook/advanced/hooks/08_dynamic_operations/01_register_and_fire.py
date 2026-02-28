"""Define and fire a custom dynamic operation at runtime.

Dynamic operations let you define new hookable operations without writing
Python classes. You provide an OperationSpec with field definitions and
action code as strings; Tract compiles them into a real Pending subclass.

Part 1 -- Define and register a "citation_check" operation
Part 2 -- Fire it with and without a handler
Part 3 -- Use review=True for manual control

Demonstrates: OperationSpec, ActionDef, register_operation(), fire(),
              t.on() with dynamic ops, review=True, pprint()
"""

from tract import Tract
from tract.hooks.dynamic import ActionDef, OperationSpec


def register_and_fire() -> None:
    # -----------------------------------------------------------------
    # Part 1 -- Define and register an operation
    # -----------------------------------------------------------------
    print("=" * 60)
    print("PART 1 -- Define and Register a Dynamic Operation")
    print("=" * 60)
    print()
    print("  OperationSpec describes the operation: fields + actions.")
    print("  ActionDef.code is a Python string compiled once at registration.")

    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.user("Summarize findings from https://example.com and https://test.org")
        t.assistant("Based on the sources, here are the key findings...")

        # Define a "citation_check" operation
        spec = OperationSpec(
            name="citation_check",
            description="Verify source URLs survive compression",
            fields={
                "urls": {"type": "list", "description": "URLs to verify"},
                "verified": {"type": "bool", "default": False},
            },
            actions={
                "verify": ActionDef(
                    name="verify",
                    description="Check all URLs are present in context",
                    params={"strict": "bool"},
                    required=[],
                    code='''
compiled = pending.tract.compile()
text = " ".join(str(m.get("content", "")) for m in compiled.to_dicts())
missing = [u for u in pending.fields["urls"] if u not in text]
if missing and strict:
    pending.reject(f"Missing {len(missing)} URLs: {missing}")
else:
    pending.fields["verified"] = len(missing) == 0
    pending.approve()
''',
                ),
            },
        )

        t.register_operation(spec)
        print(f"\n  Registered 'citation_check' operation")

        # -----------------------------------------------------------------
        # Part 2 -- Fire with auto-approve (no handler)
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 2 -- Fire with Auto-Approve (no handler)")
        print("=" * 60)
        print()
        print("  With no handler registered, fire() auto-approves.")

        result = t.fire("citation_check", fields={
            "urls": ["https://example.com", "https://test.org"],
        })
        print(f"\n  Status: {result.status}")
        print(f"  Fields: {result.fields}")

        # -----------------------------------------------------------------
        # Part 3 -- Fire with a handler
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 3 -- Fire with a Registered Handler")
        print("=" * 60)
        print()
        print("  Register a handler that calls the custom 'verify' action.")

        def verify_citations(pending):
            """Handler that runs the verify action with strict=True."""
            print(f"  [handler] Running verify(strict=True)...")
            pending.verify(strict=True)

        t.on("citation_check", verify_citations, name="strict-verify")

        result = t.fire("citation_check", fields={
            "urls": ["https://example.com", "https://test.org"],
        })
        print(f"\n  Status: {result.status}")
        print(f"  Verified: {result.fields.get('verified')}")

        # -----------------------------------------------------------------
        # Part 4 -- review=True for manual control
        # -----------------------------------------------------------------
        print()
        print("=" * 60)
        print("PART 4 -- review=True (manual control)")
        print("=" * 60)
        print()
        print("  review=True returns the Pending without firing hooks.")

        pending = t.fire("citation_check", fields={
            "urls": ["https://example.com"],
        }, review=True)

        print(f"\n  Returned: {type(pending).__name__}")
        pending.pprint()

        # Manually call an action
        pending.verify(strict=False)
        print(f"\n  After verify(strict=False):")
        print(f"  Status: {pending.status}")
        print(f"  Verified: {pending.fields.get('verified')}")

        t.print_hooks()


if __name__ == "__main__":
    register_and_fire()
