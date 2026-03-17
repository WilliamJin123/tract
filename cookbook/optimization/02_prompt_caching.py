"""Anthropic Prompt Caching -- automatic cache_control breakpoints.

Anthropic's prompt caching gives a 90% cost reduction on cached input token
prefixes.  Tract's compiler knows which content is stable (system prompts,
old compressed context, pinned instructions) vs volatile (recent user and
assistant messages).  By calling ``to_anthropic(cache_control=True)`` or
``to_anthropic_params(cache_control=True)``, cache_control breakpoints are
placed automatically at the stable/volatile boundary.

Patterns shown:
  1. Basic Usage               -- enable caching with a single flag
  2. Priority-Aware Caching    -- PINNED content anchors the cache boundary
  3. Full Workflow              -- build context, compress, compile, export

Demonstrates: CompiledContext.to_anthropic(cache_control=True),
              CompiledContext.to_anthropic_params(cache_control=True),
              t.pin(), priorities, system prompt block conversion

No external dependencies required.
"""

from tract import Tract


# ===================================================================
# Pattern 1: Basic Usage
# ===================================================================

def basic_caching() -> None:
    """Enable Anthropic prompt caching with a single flag."""

    print("=" * 60)
    print("1. Basic Usage")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a helpful research assistant.")
        t.user("What is quantum computing?")
        t.assistant(
            "Quantum computing uses quantum bits (qubits) that leverage "
            "superposition and entanglement to process information."
        )
        t.user("How does it compare to classical computing?")
        t.assistant(
            "Classical bits are 0 or 1; qubits can be both simultaneously, "
            "enabling exponential parallelism for certain problem classes."
        )

        ctx = t.compile()

        # --- Without caching (default) ---
        normal = ctx.to_anthropic()
        print("  Without cache_control:")
        print(f"    system type: {type(normal['system']).__name__}")
        print(f"    messages: {len(normal['messages'])}")
        assert isinstance(normal["system"], str), "Default system is a string"

        # --- With caching ---
        cached = ctx.to_anthropic(cache_control=True)
        print()
        print("  With cache_control=True:")
        print(f"    system type: {type(cached['system']).__name__}")

        # System prompt is now a block list with cache_control
        system = cached["system"]
        assert isinstance(system, list), "Cached system is a block list"
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        print(f"    system block: {system[0]}")

        # One message in the conversation gets a cache_control marker
        marker_count = 0
        for msg in cached["messages"]:
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if "cache_control" in block:
                        marker_count += 1
            # String content never has cache_control

        assert marker_count == 1, f"Expected 1 message marker, got {marker_count}"
        print(f"    message cache markers: {marker_count}")
        print()
        print("  Total breakpoints: 2 (system + 1 message)")
        print("  Anthropic limit: 4 breakpoints (well within budget)")

    print()
    print("PASSED")


# ===================================================================
# Pattern 2: Priority-Aware Caching
# ===================================================================

def priority_caching() -> None:
    """PINNED/IMPORTANT content anchors the cache boundary."""

    print()
    print("=" * 60)
    print("2. Priority-Aware Caching")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a coding assistant.")

        # Pin an important instruction so it stays in context
        t.user(
            "IMPORTANT: Always use type hints in Python code.",
            priority="pinned",
        )
        t.assistant("Understood. I will always use type hints.")

        # Regular conversation
        t.user("Write a function to add two numbers.")
        t.assistant("def add(a: int, b: int) -> int:\n    return a + b")
        t.user("Now write one for multiplication.")
        t.assistant("def multiply(a: int, b: int) -> int:\n    return a * b")

        ctx = t.compile()
        cached = ctx.to_anthropic(cache_control=True)

        # The pinned message drives the boundary placement:
        # everything up to and including the pinned message is "stable"
        # and benefits from caching across requests.
        msgs = cached["messages"]

        print(f"  Messages: {len(msgs)}")
        print(f"  Priorities: {ctx.priorities}")
        print()

        for i, msg in enumerate(msgs):
            role = msg["role"]
            content = msg["content"]
            has_cache = False
            if isinstance(content, list):
                for block in content:
                    if "cache_control" in block:
                        has_cache = True
            marker = " << CACHE BOUNDARY" if has_cache else ""
            preview = (
                content[:50] if isinstance(content, str)
                else str(content[0])[:50]
            )
            print(f"  [{i}] {role:>10}: {preview}...{marker}")

        print()
        print("  The cache boundary is placed at the last PINNED/IMPORTANT")
        print("  message. All content up to that point gets cached on the")
        print("  first request and reused at 90% discount on subsequent calls.")

    print()
    print("PASSED")


# ===================================================================
# Pattern 3: Full Workflow
# ===================================================================

def full_workflow() -> None:
    """Build context, compress, compile, export with caching."""

    print()
    print("=" * 60)
    print("3. Full Workflow with to_anthropic_params")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a market analyst reviewing quarterly reports.")

        # Research phase
        t.user("Q1 revenue was $10M, up 15% YoY.")
        t.assistant("Noted. Strong growth trajectory.")
        t.user("Q2 revenue was $11.5M, up 20% YoY.")
        t.assistant("Accelerating growth. Good sign.")
        t.user("Q3 revenue was $13M, up 22% YoY.")
        t.assistant("Three consecutive quarters of acceleration.")

        # Compress old research into a stable summary
        t.compression.compress(
            content="Revenue trend: Q1 $10M (+15%), Q2 $11.5M (+20%), "
            "Q3 $13M (+22%). Three quarters of accelerating growth."
        )

        # Analysis phase (volatile - changes each session)
        t.user("Project Q4 revenue based on the trend.")
        t.assistant(
            "Based on the acceleration pattern, Q4 revenue is projected "
            "at $14.5-15.5M, representing 24-28% YoY growth."
        )

        ctx = t.compile()

        # --- Export with cache_control for Anthropic API ---
        params = ctx.to_anthropic_params(cache_control=True)

        print(f"  System: {type(params['system']).__name__}")
        if isinstance(params["system"], list):
            print(f"    cached: {params['system'][0].get('cache_control')}")

        print(f"  Messages: {len(params['messages'])}")
        print()

        # Show what the API call would look like
        print("  API call structure:")
        print("  {")
        print(f'    "model": "claude-sonnet-4-20250514",')
        print(f'    "system": [')
        if isinstance(params["system"], list):
            for block in params["system"]:
                cc = block.get("cache_control", "")
                print(f'      {{"type": "text", "text": "...", "cache_control": {cc}}}')
        print(f'    ],')
        print(f'    "messages": [')
        for msg in params["messages"]:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                has_cc = any("cache_control" in b for b in content)
                cc_note = " + cache_control" if has_cc else ""
                print(f'      {{"role": "{role}", "content": [{len(content)} blocks{cc_note}]}}')
            else:
                print(f'      {{"role": "{role}", "content": "..."}}')
        print(f'    ],')
        print(f'    "max_tokens": 1024')
        print("  }")

        print()
        print("  Cost breakdown (approximate):")
        print("    First request:   full price on all input tokens")
        print("    Repeat requests: 90% discount on tokens before cache boundary")
        print("    Savings:         system prompt + compressed history = cached")
        print("    Volatile:        only recent Q&A charged at full price")

        # Verify structure
        assert isinstance(params["system"], list)
        assert params["system"][0]["cache_control"] == {"type": "ephemeral"}
        assert len(params["messages"]) > 0

    print()
    print("PASSED")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    basic_caching()
    priority_caching()
    full_workflow()

    print()
    print("=" * 60)
    print("Summary: Anthropic Prompt Caching")
    print("=" * 60)
    print()
    print("  Method                                  Effect")
    print("  --------------------------------------  --------------------------------")
    print("  to_anthropic()                          No caching (default)")
    print("  to_anthropic(cache_control=True)        System + boundary markers added")
    print("  to_anthropic_params(cache_control=True) Same, plus tools dict")
    print()
    print("  Breakpoint placement:")
    print("    1. System prompt  -> always cached (it never changes)")
    print("    2. Last PINNED/IMPORTANT message -> stable content boundary")
    print("       (falls back to midpoint if no priorities set)")
    print()
    print("  At most 2 of Anthropic's 4 allowed breakpoints are used.")
    print()
    print("Done.")


# Alias for pytest discovery
test_prompt_caching = main


if __name__ == "__main__":
    main()


# --- See also ---
# Budget management:     optimization/01_budget_management.py
# Production monitoring: config_and_middleware/05_observability.py
