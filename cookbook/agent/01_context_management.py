"""Agent-Driven Context Management

An LLM agent monitors and maintains its own context health autonomously.
It checks status, pins important information, compresses older turns,
adjusts model settings, and runs garbage collection -- all through
genuine tool calls decided by the agent, not hardcoded logic.

Tools exercised: status, compile, compress, annotate, gc, configure_model, log

Demonstrates: LLM-driven context maintenance, budget-aware compression,
              priority annotations, model switching, garbage collection
"""

import io
import json
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm

MODEL_ID = llm.large


# Tool profile: context management tools
CONTEXT_PROFILE = ToolProfile(
    name="context-manager",
    tool_configs={
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check context health: branch, HEAD, token count, budget "
                "percentage. Call this first to understand the current state "
                "before taking any maintenance action."
            ),
        ),
        "compile": ToolConfig(
            enabled=True,
            description=(
                "Compile current context into messages. Shows message count "
                "and total tokens. Use to verify state after operations."
            ),
        ),
        "compress": ToolConfig(
            enabled=True,
            description=(
                "Compress a range of commits into a summary to reduce token "
                "usage. Pinned commits are preserved verbatim. Optionally "
                "set target_tokens and provide instructions for the summary."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Set priority on a commit. Use 'pinned' to protect important "
                "context from compression. Use 'skip' to hide irrelevant "
                "content. Use 'normal' to reset. Requires a commit hash "
                "(find hashes via log)."
            ),
        ),
        "gc": ToolConfig(
            enabled=True,
            description=(
                "Run garbage collection to remove orphaned commits. Frees "
                "storage space. Set min_age_days to control retention."
            ),
        ),
        "configure_model": ToolConfig(
            enabled=True,
            description=(
                "Change model or temperature. Use a smaller model for "
                "compression tasks, a larger model for complex reasoning. "
                "Set operation='compress' to configure only compression."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description="View recent commits to find hashes for annotate/compress.",
        ),
    },
)


def _log_step(step_num, response):
    """on_step callback -- print step number."""
    print(f"    [step {step_num}]")


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 60)
    print("Agent-Driven Context Management")
    print("=" * 60)
    print()
    print("  The agent will: check status, pin important content,")
    print("  compress old turns, and run GC -- all autonomously.")
    print()

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        # Register tools from the profile
        tools = t.as_tools(profile=CONTEXT_PROFILE)
        t.set_tools(tools)

        t.system(
            "You are a context management assistant. You have tools to "
            "inspect, compress, annotate, and clean up conversation history. "
            "When asked to maintain the context, use these tools to keep it "
            "healthy and within budget."
        )

        # Fill up the context with research notes
        for i in range(8):
            t.user(f"Research note {i}: Quantum computing uses qubits which "
                   f"can be in superposition states. Error rate: {0.1 / (i+1):.3f}")
            t.assistant(f"Recorded note {i}. Key finding: error rate {0.1/(i+1):.3f}.")

        # Pin the most important note manually (so agent can find it)
        important = t.log(limit=2)[0]
        t.annotate(important.commit_hash, "pinned", reason="Key error rate data")

        print(f"  Context filled: {t.status().token_count} tokens")
        print(f"  Budget: {t.status().token_budget_max} max")

        print("\n  BEFORE maintenance:")
        t.compile().pprint(style="compact")

        # Ask the agent to maintain the context
        print("\n  --- Task: Assess and maintain context ---")
        result = t.run(
            "The context is getting large. Please:\n"
            "1. Check status to see budget usage\n"
            "2. Use log to find commits, then pin any important ones\n"
            "3. Compress older turns to free up space\n"
            "4. Run GC to clean up orphaned commits\n"
            "5. Check status again to confirm improvement",
            max_steps=12, on_step=_log_step,
        )
        print(f"\n  Loop result: {result.status} ({result.steps} steps, "
              f"{result.tool_calls} tool calls)")
        if result.final_response:
            print(f"  Agent: {result.final_response[:200]}")

        # Show final state
        print("\n  AFTER maintenance:")
        t.compile().pprint(style="compact")

        status = t.status()
        print(f"\n  Final: {status.token_count} tokens, "
              f"{status.commit_count} commits")


if __name__ == "__main__":
    main()
