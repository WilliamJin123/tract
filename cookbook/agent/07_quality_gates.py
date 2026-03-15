"""Agent Quality Gates (Implicit)

The developer sets up middleware gates that block premature transitions.
The agent is given a research problem and must produce enough work to
satisfy quality requirements it doesn't know about in advance. When
the gate blocks a transition, the agent reads the error and adapts.

Tools available: transition, commit, get_config, status, log

Demonstrates: Can the model adapt when middleware blocks its actions,
              without being told about the gate rules in advance?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError, MiddlewareContext
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.xlarge


PROFILE = ToolProfile(
    name="engineer",
    tool_configs={
        "transition": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
        "get_config": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
    },
)


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("Agent Quality Gates (Implicit)")
    print("=" * 70)
    print()
    print("  Developer-configured gates block premature transitions.")
    print("  Agent must discover and satisfy requirements on its own.")
    print()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        # System: role only, no gate protocol
        t.system(
            "You are a software engineer working on an API project. "
            "Research topics thoroughly before moving to implementation."
        )

        # Developer sets up gated workflow infrastructure
        t.branch("research", switch=True)
        t.configure(stage="research")

        # Gate: require at least 3 artifact commits before transition
        def research_gate(ctx: MiddlewareContext):
            if ctx.target == "implementation":
                entries = ctx.tract.log(limit=50)
                artifacts = [e for e in entries if e.content_type == "artifact"]
                if len(artifacts) < 3:
                    raise BlockedError(
                        "pre_transition",
                        f"Research incomplete: {len(artifacts)} artifact(s) committed, "
                        f"need at least 3 before moving to implementation.",
                    )

        t.use("pre_transition", research_gate)

        t.switch("main")
        t.branch("implementation", switch=True)
        t.configure(stage="implementation")
        t.switch("research")

        log = StepLogger()

        print(f"  Starting on: {t.current_branch}")
        print(f"  Branches: {[b.name for b in t.list_branches()]}")

        # Task: research then attempt transition
        print("\n  --- Task: Research ---")
        result = t.run(
            "Research authentication, database schema, and error handling "
            "patterns for a REST API. When you feel your research is "
            "thorough, move to implementation.",
            max_steps=15, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # If the agent didn't try transitioning, force an attempt to show the gate
        if t.current_branch != "implementation":
            print("\n  --- Agent didn't transition, forcing attempt ---")
            entries = t.log(limit=50)
            artifacts = [e for e in entries if e.content_type == "artifact"]
            print(f"  Artifacts committed: {len(artifacts)}")
            try:
                t.transition("implementation")
                print("  Transition succeeded!")
            except BlockedError as e:
                print(f"  Gate blocked: {e}")
                print("  (This demonstrates the quality gate working)")

        # Report
        print("\n\n=== Final State ===\n")
        print(f"  Current branch: {t.current_branch}")
        status = t.status()
        print(f"  Commits: {status.commit_count}, Tokens: {status.token_count}")

        print("\n  Context:")
        t.compile().pprint(style="compact")

        reached_impl = t.current_branch == "implementation"
        print(f"\n  Reached implementation: {reached_impl}")


if __name__ == "__main__":
    main()
