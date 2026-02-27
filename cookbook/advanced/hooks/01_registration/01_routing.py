"""Three-tier hook routing across different operations.

The routing logic is universal -- it works identically for tool_result,
compress, gc, and any future hookable operation.  This demo uses three
different operations (one per tier) to prove the point:

  Tier 3 (auto-approve):  tool_result with no hook registered
  Tier 2 (registered hook): compress with a handler
  Tier 1 (review=True):    tool_result with manual control
"""

import os

from dotenv import load_dotenv

from tract import Priority, Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.event import HookEvent
from tract.hooks.tool_result import PendingToolResult
from tract.models.commit import CommitInfo
from tract.models.compression import CompressResult
from tract.protocols import CompiledContext

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def three_tier_routing() -> None:
    """Each tier uses a different operation to show the routing is universal."""
    # -----------------------------------------------------------------
    # Tier 3 (auto-approve): tool_result with NO hook registered
    # -----------------------------------------------------------------
    print("=" * 60)
    print("TIER 3 -- Auto-Approve (no hook): tool_result")
    print("=" * 60)
    print()
    print("  No hook registered for tool_result.")
    print("  The operation commits directly and hook_log records 'auto-approved'.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        sys_ci = t.system("You are a code assistant with access to development tools.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.assistant("I'll search the codebase for that pattern.", metadata={
            "tool_calls": [{"id": "call_001", "name": "grep", "arguments": {"pattern": "TODO"}}],
        })

        # No t.on("tool_result", ...) -- tier 3 fires
        result: CommitInfo = t.tool_result(
            "call_001", "grep",
            "src/main.py:42: # TODO refactor this\nsrc/utils.py:15: # TODO add tests",
        )
        print(f"\n  tool_result returned: {type(result).__name__}")

        # hook_log captures the auto-approve even with no handler
        last_event: HookEvent = t.hook_log[-1]
        print(f"  hook_log: handler={last_event.handler_name}, result={last_event.result}")
        print(f"  (No hook was called -- the system auto-approved silently.)")

        t.print_hooks()

    # -----------------------------------------------------------------
    # Tier 2 (registered hook): compress with a handler
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("TIER 2 -- Registered Hook: compress")
    print("=" * 60)
    print()
    print("  A compress hook inspects the PendingCompress and approves.")
    print("  This is the bread-and-butter: register once, it fires every time.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Seed a conversation worth compressing
        sys_ci = t.system("You are a DevOps assistant helping engineers with CI/CD pipelines and infrastructure.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.chat("How do I set up GitHub Actions for a Python project with pytest and linting?")
        t.chat("What about adding Docker build and push steps to the pipeline?")
        t.chat("How should I handle secrets and environment variables in CI?")
        t.chat("What's the best strategy for running tests in parallel?")

        ctx_before: CompiledContext = t.compile()
        print(f"\n  Before compress: {len(ctx_before.messages)} messages, {ctx_before.token_count} tokens")

        # Register a compress hook that inspects and approves
        def review_compress(pending: PendingCompress) -> None:
            """Inspect summaries, show token savings, approve."""
            print(f"\n  [hook] PendingCompress received:")
            pending.pprint(verbose=True)

            ratio = pending.estimated_tokens / max(pending.original_tokens, 1)
            print(f"  [hook] Token ratio: {ratio:.2f} ({int((1 - ratio) * 100)}% reduction)")
            print(f"  [hook] Approving compression.")
            pending.approve()

        t.on("compress", review_compress, name="review_compress")
        t.print_hooks()

        # Trigger compress -- the hook fires (tier 2)
        result: CompressResult = t.compress(target_tokens=200)
        assert isinstance(result, CompressResult), f"Expected CompressResult, got {type(result).__name__}"
        print(f"\n  compress returned: {type(result).__name__}")
        print(f"  compression_ratio: {result.compression_ratio:.1%}")

        # hook_log shows the handler that fired
        last_event: HookEvent = t.hook_log[-1]
        print(f"  hook_log: handler={last_event.handler_name}, result={last_event.result}")

        ctx_after: CompiledContext = t.compile()
        print(f"  After compress: {len(ctx_after.messages)} messages, {ctx_after.token_count} tokens")

    # -----------------------------------------------------------------
    # Tier 1 (review=True): tool_result with manual control
    # -----------------------------------------------------------------
    print()
    print("=" * 60)
    print("TIER 1 -- review=True: tool_result (manual control)")
    print("=" * 60)
    print()
    print("  review=True returns PendingToolResult to the caller.")
    print("  Even if a hook is registered, review=True bypasses it.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        sys_ci = t.system("You are a code assistant with access to development tools.")
        t.annotate(sys_ci.commit_hash, Priority.PINNED)

        t.assistant("I'll search the codebase for that pattern.", metadata={
            "tool_calls": [{"id": "call_002", "name": "grep", "arguments": {"pattern": "TODO"}}],
        })

        # Even register a hook -- review=True should bypass it
        hook_fired: list[bool] = []

        def should_not_fire(pending: PendingToolResult) -> None:
            hook_fired.append(True)
            pending.approve()

        t.on("tool_result", should_not_fire, name="should_not_fire")

        # review=True -- tier 1, returns PendingToolResult
        pending: PendingToolResult = t.tool_result(
            "call_002", "grep",
            "src/main.py:42: # TODO refactor this\nsrc/utils.py:15: # TODO add tests",
            review=True,
        )
        print(f"\n  tool_result(review=True) returned: {type(pending).__name__}")
        pending.pprint()

        # The registered hook was NOT called
        print(f"  Hook fired? {bool(hook_fired)} (should be False)")
        assert not hook_fired, "review=True should bypass registered hooks"

        # Approve manually
        result: CommitInfo = pending.approve()
        print(f"\n  After approve(): status={pending.status}")
        print(f"  Committed: {type(result).__name__}")

        t.print_hooks()


if __name__ == "__main__":
    three_tier_routing()
