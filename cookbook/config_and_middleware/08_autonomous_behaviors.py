"""Autonomous Behaviors via Middleware

Middleware can do more than validate or log -- it can take autonomous
action. These patterns show how to make a tract self-manage its context
without burning agent loop steps.

Each pattern is a standalone middleware handler that hooks into an
existing event and calls existing operations. No new APIs required.

Patterns:
  1. Auto-skip low-value commits by type (post_commit)
  2. Commit-count triggered compression (post_commit)
  3. Keyword-based stage routing (post_commit)
  4. Adaptive config based on error rate (post_commit)
  5. Tool result budget guard (post_tool_execute)

No LLM required for the middleware trigger logic itself.
Some triggered operations (compress, compress_tool_calls) need
a configured LLM in production -- noted inline.

Demonstrates: autonomous middleware patterns, stateful closures,
              composing middleware with existing operations
"""

from tract import Tract, Priority


def main():

    # =================================================================
    # Pattern 1: Auto-skip low-value content types
    # =================================================================
    # Config commits, reasoning, and metadata are useful bookkeeping
    # but often not worth context budget. Auto-SKIP them so they
    # drop out of compilation.

    print("=== Pattern 1: Auto-Skip by Content Type ===\n")

    with Tract.open() as t:

        skip_types = {"config", "reasoning"}

        def auto_skip(ctx):
            if ctx.commit and ctx.commit.content_type in skip_types:
                ctx.tract.annotate(ctx.commit.commit_hash, Priority.SKIP)

        t.use("post_commit", auto_skip)

        t.system("You are a helpful assistant.")
        t.configure(temperature=0.5)        # config commit -- auto-skipped
        t.reasoning("Let me think...")       # reasoning -- auto-skipped
        t.user("What is Python?")
        t.assistant("A programming language.")

        compiled = t.compile()
        total = len(t.log())
        skipped = len(t.skipped())
        print(f"  Commits made:")
        for ci in t.log():
            skip = " [SKIPPED]" if ci.effective_priority == "skip" else ""
            print(f"    {ci.content_type:12s} {(ci.message or '')[:45]}{skip}")
        print()
        print(f"\n  Compiled context ({compiled.commit_count} of {total} commits):")
        compiled.pprint(style="compact")
        print(f"\n  Auto-skipped: {skipped} (config + reasoning)")
        assert skipped == 2, f"Expected 2 skipped, got {skipped}"

    # =================================================================
    # Pattern 2: Commit-count triggered compression
    # =================================================================
    # After N commits accumulate, compress older content to keep
    # context lean. The closure tracks state across invocations.
    # In production, replace the print with ctx.tract.compress().

    print("\n=== Pattern 2: Commit-Count Compression Trigger ===\n")

    with Tract.open() as t:

        compress_state = {"triggered": False, "commit_count": 0}

        def auto_compress_trigger(ctx):
            compress_state["commit_count"] += 1
            threshold = 6
            if compress_state["commit_count"] >= threshold:
                # In production with a configured LLM:
                # ctx.tract.compress(strategy="sliding_window", window_size=3)
                compress_state["triggered"] = True
                compress_state["commit_count"] = 0  # reset counter
                print(f"    >> Compression triggered at {threshold} commits")

        t.use("post_commit", auto_compress_trigger)

        t.system("You are a research analyst.")
        for i in range(7):
            t.user(f"Finding #{i}: data point about topic {i}")
            t.assistant(f"Noted finding #{i}.")

        print(f"  Built up {len(t.log())} commits (7 user + 7 assistant + 1 system)")
        print(f"  Trigger fired: {compress_state['triggered']}")
        print(f"  (In production, compress() would summarize older commits)")
        assert compress_state["triggered"], "Should have triggered"

    # =================================================================
    # Pattern 3: Keyword-based stage routing
    # =================================================================
    # Detect content keywords and auto-transition between workflow
    # stages. No LLM needed -- pure string matching.

    print("\n=== Pattern 3: Keyword Routing ===\n")

    with Tract.open() as t:

        route_state = {"transitions": []}

        routes = {
            "implementation": ["implement", "code", "function", "class", "def "],
            "validation": ["test", "verify", "assert", "check"],
        }

        def keyword_router(ctx):
            if not ctx.commit:
                return
            text = (ctx.commit.message or "").lower()
            content = str(ctx.tract.get_content(ctx.commit) or "").lower()
            combined = text + " " + content

            current_stage = ctx.tract.get_config("stage") or "research"
            for target, keywords in routes.items():
                if target == current_stage:
                    continue
                if any(kw in combined for kw in keywords):
                    ctx.tract.configure(stage=target)
                    route_state["transitions"].append(
                        f"{current_stage} -> {target}"
                    )
                    break

        t.use("post_commit", keyword_router)

        t.configure(stage="research")
        t.system("You are a software engineer.")

        print("  [research stage]")
        t.user("Research stack data structures.")
        print('    user:      "Research stack data structures."')
        t.assistant("A stack is a LIFO structure...")
        print('    assistant: "A stack is a LIFO structure..."')

        # This commit contains "implement" -- triggers routing
        t.user("Now implement a Stack class with push and pop.")
        print('    user:      "Now implement a Stack class with push and pop."')
        print(f"    >> routed: {route_state['transitions'][-1]}")

        stage = t.get_config("stage")
        print(f"\n  [implementation stage]")
        t.assistant("Here's the implementation.")
        print('    assistant: "Here\'s the implementation."')

        # This commit contains "test" -- triggers routing again
        t.user("Now write tests to verify the Stack works.")
        print('    user:      "Now write tests to verify the Stack works."')
        print(f"    >> routed: {route_state['transitions'][-1]}")

        stage = t.get_config("stage")
        print(f"\n  [validation stage]")
        print(f"  All transitions: {route_state['transitions']}")
        assert stage == "validation"

    # =================================================================
    # Pattern 4: Adaptive config based on error count
    # =================================================================
    # Track tool errors and lower temperature after repeated failures.
    # Reset when errors clear.

    print("\n=== Pattern 4: Adaptive Config ===\n")

    with Tract.open() as t:

        error_state = {"consecutive_errors": 0, "adjustments": []}

        def adaptive_config(ctx):
            if not ctx.commit:
                return
            # Only evaluate tool results for error tracking
            if "tool_result" not in (ctx.commit.tags or []):
                return
            meta = ctx.commit.metadata or {}
            is_error = meta.get("is_error", False)

            if is_error:
                error_state["consecutive_errors"] += 1
                if error_state["consecutive_errors"] >= 2:
                    ctx.tract.configure(temperature=0.1)
                    error_state["adjustments"].append("lowered temp to 0.1")
            else:
                if error_state["consecutive_errors"] >= 2:
                    ctx.tract.configure(temperature=0.7)
                    error_state["adjustments"].append("restored temp to 0.7")
                error_state["consecutive_errors"] = 0

        t.use("post_commit", adaptive_config)
        t.configure(temperature=0.7)

        t.system("You are a deployment agent.")

        t.assistant("Deploying...", metadata={"tool_calls": [
            {"id": "c1", "name": "deploy", "arguments": {}},
        ]})
        t.tool_result("c1", "deploy", "Connection refused", is_error=True)
        print('  deploy() -> "Connection refused" [ERROR]')
        print(f"    consecutive errors: {error_state['consecutive_errors']}")

        t.assistant("Retrying...", metadata={"tool_calls": [
            {"id": "c2", "name": "deploy", "arguments": {}},
        ]})
        t.tool_result("c2", "deploy", "Timeout", is_error=True)
        print('  deploy() -> "Timeout" [ERROR]')
        print(f"    consecutive errors: {error_state['consecutive_errors']}")

        # After 2 errors, temperature should have dropped
        temp = t.get_config("temperature")
        print(f"    >> temperature adapted: 0.7 -> {temp}")
        assert temp == 0.1, f"Expected 0.1, got {temp}"

        # Successful deploy restores temperature
        t.assistant("Third attempt...", metadata={"tool_calls": [
            {"id": "c3", "name": "deploy", "arguments": {}},
        ]})
        t.tool_result("c3", "deploy", "Deployed successfully.")
        print('  deploy() -> "Deployed successfully." [OK]')

        temp = t.get_config("temperature")
        print(f"    >> temperature restored: 0.1 -> {temp}")
        assert temp == 0.7, f"Expected 0.7, got {temp}"

    # =================================================================
    # Pattern 5: Tool result budget guard
    # =================================================================
    # Track tool result token accumulation. When tool results exceed
    # a threshold, flag for compaction. In production, call
    # compress_tool_calls() instead of printing.

    print("\n=== Pattern 5: Tool Result Budget Guard ===\n")

    with Tract.open() as t:

        tool_state = {"result_tokens": 0, "compaction_needed": False}

        def tool_budget_guard(ctx):
            if not ctx.commit:
                return
            if "tool_result" not in (ctx.commit.tags or []):
                return
            tool_state["result_tokens"] += ctx.commit.token_count
            if tool_state["result_tokens"] > 20:
                # In production with a configured LLM:
                # ctx.tract.compress_tool_calls()
                tool_state["compaction_needed"] = True
                tool_state["result_tokens"] = 0  # reset after compaction
                print(f"    >> Tool compaction triggered")

        t.use("post_commit", tool_budget_guard)

        t.system("You are a file search agent.")
        # Simulate a sequence of tool calls with verbose results
        for i in range(5):
            t.assistant(f"Searching batch {i}...", metadata={"tool_calls": [
                {"id": f"c{i}", "name": "search", "arguments": {"q": f"topic_{i}"}},
            ]})
            result_text = f"Result {i}: " + "x" * 100  # verbose result
            t.tool_result(f"c{i}", "search", result_text)
            print(f"  search(topic_{i}) -> {result_text[:50]}...")

        print(f"\n  Tool tokens accumulated: {tool_state['result_tokens']}")
        print(f"  Compaction flagged:      {tool_state['compaction_needed']}")

    # =================================================================
    # Composing patterns
    # =================================================================
    # All patterns are independent middleware handlers. Compose them
    # by registering multiple handlers on the same event:
    #
    #   t.use("post_commit", auto_skip)
    #   t.use("post_commit", keyword_router)
    #   t.use("post_commit", adaptive_config)
    #
    # Handlers fire in registration order. Each sees the same
    # MiddlewareContext. Use t.remove_middleware(id) to disable.

    print("\n=== Summary ===\n")
    print("  Pattern 1: auto_skip        post_commit    -> annotate(SKIP)")
    print("  Pattern 2: auto_compress    post_commit    -> compress()")
    print("  Pattern 3: keyword_router   post_commit    -> configure(stage=)")
    print("  Pattern 4: adaptive_config  post_commit    -> configure(temperature=)")
    print("  Pattern 5: tool_budget      post_commit    -> compress_tool_calls()")
    print()
    print("  All patterns: stateful closures, zero new APIs, composable.")


if __name__ == "__main__":
    main()


# --- See also ---
# Middleware basics:        config_and_middleware/02_event_automation.py
# Observability middleware: config_and_middleware/06_observability.py
# Agent context management: agent/01_context_management.py
# Tool compaction (LLM):   agent/15_tool_compaction.py
# Self-routing workflow:    workflows/09_self_routing.py
