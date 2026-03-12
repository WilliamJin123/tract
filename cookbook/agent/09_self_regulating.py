"""Self-Regulating Agent (Implicit)

The agent is given two tasks with dramatically conflicting behavioral
requirements: first write enthusiastic marketing copy, then write a
brutally honest security audit of the same product. The shift from
"advocate" to "adversary" creates natural pressure for self-regulation.

Tools available: configure, directive, create_middleware, remove_middleware,
                 get_config, commit, transition

Demonstrates: Does the model proactively set directives, configure behavior,
              or create middleware to enforce phase-specific constraints?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.xlarge


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
        return

    print("=" * 70)
    print("Self-Regulating Agent (Implicit)")
    print("=" * 70)
    print()
    print("  Phase 1: Write marketing copy (enthusiastic advocate)")
    print("  Phase 2: Write security audit (ruthless critic)")
    print("  Will the agent self-regulate for the behavioral shift?")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
    ) as t:

        t.system(
            "You are a technical writer who adapts style to the task.\n"
            "IMPORTANT: Before writing, you MUST set a behavioral mode using "
            "configure(mode='advocate') or configure(mode='critic') to declare "
            "your current stance. Use directive() to set writing guidelines for "
            "each phase. A middleware gate requires the mode config to be set "
            "before any content can be committed."
        )

        # Gate: require mode config before agent content commits
        def mode_gate(ctx):
            from tract.models.content import ConfigContent, InstructionContent
            pending = ctx.pending
            if pending is not None:
                # Allow system/user messages, config changes, and directives through
                role = getattr(pending, "role", None)
                if role in ("system", "user"):
                    return
                if isinstance(pending, (ConfigContent, InstructionContent)):
                    return
            mode = ctx.tract.get_config("mode")
            if not mode:
                raise BlockedError(
                    "pre_commit",
                    ["Set a mode first: configure(mode='advocate') or "
                     "configure(mode='critic')"],
                )

        t.use("pre_commit", mode_gate)

        log = StepLogger()
        _tool_names = [
            "configure", "directive", "create_middleware",
            "remove_middleware", "get_config", "commit",
            "transition",
        ]

        # Phase 1: Marketing copy (enthusiastic, positive)
        print("=== Phase 1: Marketing copy ===\n")
        result = t.run(
            "Write short marketing copy for 'Nexus' API framework. "
            "Selling points: 1M req/sec, built-in auth, auto OpenAPI docs. "
            "Make it compelling for the landing page.",
            profile="full",
            tool_names=_tool_names,
            max_steps=6, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Phase 2: Security audit (critical, adversarial)
        # The behavioral shift is dramatic — the agent just wrote glowing
        # marketing copy and now must tear the same product apart.
        print("\n\n=== Phase 2: Security audit ===\n")
        result = t.run(
            "Now write a security audit of Nexus as a penetration tester. "
            "Be ruthlessly critical — find gaps in the auth claims, "
            "question the performance numbers, flag what was left out.",
            profile="full",
            tool_names=_tool_names,
            max_steps=8, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # Report
        print("\n\n=== Self-Regulation Report ===\n")
        print(f"  Branch: {t.current_branch}")
        print(f"  Commits: {len(t.log())}")

        configs = t.get_all_configs()
        if configs:
            print(f"  Configs set: {configs}")
        else:
            print("  No configs set by agent.")

        ctx = t.compile()
        directives = [m for m in ctx.messages if "directive" in m.content.lower()
                      or m.role == "system" and m.content != ctx.messages[0].content]
        if len(directives) > 0:
            print(f"  Directives created: {len(directives)}")
        else:
            print("  No directives created by agent.")

        middleware_count = sum(len(v) for v in t._middleware.values())
        if middleware_count > 0:
            print(f"  Middleware handlers: {middleware_count}")
        else:
            print("  No middleware created by agent.")


if __name__ == "__main__":
    main()


# --- See also ---
# Config & directives (no LLM):  getting_started/02_config_and_directives.py
# Middleware patterns (no LLM):   config_and_middleware/02_event_automation.py
# Quality gates (LLM):            agent/07_quality_gates.py
