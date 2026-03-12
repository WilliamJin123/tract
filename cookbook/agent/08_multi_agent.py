"""Multi-Agent Delegation (Implicit)

A parent agent delegates research to a child agent on a branch. The child
has a tight budget and a substantial research task — it must figure out
how to organize its findings and stay within limits. The parent merges
and synthesizes.

Tools available:
  Parent: branch, switch, merge, list_branches, commit
  Child:  commit, compress, status

Demonstrates: Does the child agent autonomously commit structured findings
              and compress to stay within budget for clean handoff?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Session, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolProfile

# Budget applied to parent — child inherits it via deploy(), creating
# natural pressure for the child to organize and compress its research.
SHARED_CONFIG = TractConfig(token_budget=TokenBudgetConfig(max_tokens=2000))

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.xlarge


PARENT_PROFILE = ToolProfile(
    name="coordinator",
    tool_configs={
        "branch": ToolConfig(enabled=True),
        "switch": ToolConfig(enabled=True),
        "merge": ToolConfig(enabled=True),
        "list_branches": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
    },
)

CHILD_PROFILE = ToolProfile(
    name="researcher",
    tool_configs={
        "commit": ToolConfig(enabled=True),
        "compress": ToolConfig(enabled=True),
        "status": ToolConfig(enabled=True),
    },
)


def main():
    if not llm.api_key:
        print("SKIPPED (no API key)")
        return

    print("=" * 70)
    print("Multi-Agent Delegation (Implicit)")
    print("=" * 70)
    print()
    print("  Parent delegates to child on a branch.")
    print("  Child has tight budget — must organize and compress for handoff.")
    print()

    session = Session.open()
    parent_tract = session.create_tract(
        display_name="coordinator", config=SHARED_CONFIG,
    )

    # Configure LLM
    from dataclasses import replace
    from tract import OpenAIClient, LLMConfig

    parent_client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url,
        default_model=MODEL_ID,
    )
    parent_tract.configure_llm(parent_client)
    parent_tract._owns_llm_client = True
    parent_tract._default_config = LLMConfig(model=MODEL_ID)

    # Enable LLM-generated commit messages (using the small model)
    parent_tract._auto_message_enabled = True
    parent_tract._operation_configs = replace(
        parent_tract._operation_configs,
        message=LLMConfig(model=llm.small, temperature=0.0),
    )

    parent_tract.system(
        "You are a coordinator evaluating caching for microservices."
    )
    parent_tract.user(
        "Evaluate caching strategies for our microservices. "
        "Research options and recommend one."
    )
    parent_tract.assistant("I'll delegate the research and synthesize.")

    log = StepLogger()

    print("  Parent context:")
    parent_tract.compile().pprint(style="compact")

    # Create research branch and deploy child
    print("\n=== Child agent researches ===\n")
    parent_tract.branch("research-caching", switch=False)

    child = session.deploy(
        parent_tract,
        purpose="research caching patterns",
        branch_name="research-caching",
    )

    child_client = OpenAIClient(
        api_key=llm.api_key,
        base_url=llm.base_url,
        default_model=MODEL_ID,
    )
    child.configure_llm(child_client)
    child._owns_llm_client = True
    child._default_config = LLMConfig(model=MODEL_ID)
    child._auto_message_enabled = True
    child._operation_configs = replace(
        child._operation_configs,
        message=LLMConfig(model=llm.small, temperature=0.0),
    )

    # Research task — more content than budget comfortably holds
    result = child.run(
        "Research 3 caching patterns: write-through, cache-aside, "
        "and write-back. For each: how it works, pros/cons, best use case.",
        profile=CHILD_PROFILE,
        max_steps=10, max_tokens=1024,
        on_step=log.on_step, on_tool_result=log.on_tool_result,
    )
    result.pprint()

    child_status = child.status()
    child_pct = child_status.token_count / child_status.token_budget_max * 100
    print(f"\n  Child: {child_status.token_count} tokens "
          f"({child_pct:.0f}% of {child_status.token_budget_max}), "
          f"{child_status.commit_count} commits")
    print("\n  Child context:")
    child.compile().pprint(style="compact")

    # Check if child used compress
    child_compressed = any(
        "compress" in (e.message or "") for e in child.log(limit=50)
    )
    if child_compressed:
        print("  Child compressed its context for handoff.")
    else:
        print("  Child did not compress.")

    # Parent merges and synthesizes
    print("\n=== Parent merges and synthesizes ===\n")
    merge_result = parent_tract.merge("research-caching")
    merge_result.pprint()

    result = parent_tract.run(
        "Research complete. Which caching strategy and why? "
        "Give a concrete recommendation.",
        profile=PARENT_PROFILE,
        max_steps=6, max_tokens=1024,
        on_step=log.on_step, on_tool_result=log.on_tool_result,
    )
    result.pprint()

    # Final state
    print("\n\n=== Final State ===\n")
    branches = [b.name for b in parent_tract.list_branches()]
    print(f"  Branches: {branches}")
    print(f"  Parent commits: {len(parent_tract.log())}")

    print("\n  Final context:")
    parent_tract.compile().pprint(style="compact")

    session.close()


if __name__ == "__main__":
    main()
