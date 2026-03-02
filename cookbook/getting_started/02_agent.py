"""Hello Agent -- A ReAct Loop with Tract Tools (Sidecar Pattern)

An agent that manages its own context window using a sidecar branch.
The LLM receives tract operations as callable tools (via as_tools()),
decides when to use them, and ToolExecutor dispatches the calls.

Key insight: the tool interaction runs on a scratch branch so the
agent's own reasoning/tool turns don't pollute the main conversation.
Annotations (like pinning) are branch-independent, so they persist
when we switch back to main.

This is the minimal ReAct pattern: compile context + tools -> LLM ->
parse tool_calls -> execute -> feed results back -> repeat.

Demonstrates: as_tools(), ToolExecutor, t.generate(), t.tool_result(),
              sidecar branch pattern, agent-driven annotate/status/log
"""

import os

from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolExecutor, ToolProfile

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"

# --- Tool profile: only expose the context-management tools ---
# The agent gets status, annotate, and log -- enough to monitor
# and maintain its own context window. We use description overrides
# to hint WHEN to use each tool, not just what it does.

CONTEXT_MGMT_PROFILE = ToolProfile(
    name="context-mgmt",
    tool_configs={
        "status": ToolConfig(
            enabled=True,
            description=(
                "Check context health: branch, HEAD, token count, and budget "
                "usage. Call this BEFORE deciding whether to pin or skip."
            ),
        ),
        "log": ToolConfig(
            enabled=True,
            description=(
                "View recent commit history with hashes, types, and token "
                "counts. Call this to find commit hashes for annotate."
            ),
        ),
        "annotate": ToolConfig(
            enabled=True,
            description=(
                "Pin or skip a commit. Use priority='pinned' to protect "
                "important context from compression. Use priority='skip' to "
                "exclude irrelevant content. Requires a commit hash from log."
            ),
        ),
    },
)

MAX_TOOL_TURNS = 6


def react_loop(t: Tract, executor: ToolExecutor) -> str:
    """Run a ReAct loop: generate -> tool calls -> execute -> repeat.

    Returns the final text response from the agent.
    """
    for _ in range(MAX_TOOL_TURNS):
        try:
            response = t.generate()
        except Exception as exc:
            print(f"  [generate error: {exc}]\n")
            return "(agent stopped due to LLM error)"

        if not response.tool_calls:
            # Text response -- the agent is done
            return response.text

        # Execute each tool call the LLM requested
        for tc in response.tool_calls:
            result = executor.execute(tc.name, tc.arguments)
            t.tool_result(tc.id, tc.name, str(result))

            args_short = ", ".join(f"{k}={v!r}" for k, v in tc.arguments.items())
            status = "ok" if result.success else "ERR"
            print(f"  [{tc.name}] {args_short}")
            print(f"    -> [{status}] {result}\n")

    return "(max tool turns reached)"


def main():
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "getting_started_agent.db")
    if os.path.exists(db_path):
        os.unlink(db_path)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=4000))

    with Tract.open(
        db_path,
        tract_id="agent-demo",
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # --- Give the agent its tools ---

        tools = t.as_tools(format="openai", profile=CONTEXT_MGMT_PROFILE)
        t.set_tools(tools)
        tool_names = [tool["function"]["name"] for tool in tools]
        print(f"Agent tools: {', '.join(tool_names)}\n")

        executor = ToolExecutor(t)
        executor.set_profile(CONTEXT_MGMT_PROFILE)

        # --- Seed the conversation with enough history to make
        #     context management interesting ---

        t.system(
            "You are an AI research assistant with context management tools. "
            "Use your tools to monitor and maintain your context window: "
            "check status, pin important findings, and compress when needed."
        )

        t.user("What is chain-of-thought prompting?")
        t.assistant(
            "Chain-of-thought prompting encourages LLMs to break down complex "
            "reasoning into intermediate steps, improving accuracy on math and "
            "logic tasks. Wei et al. (2022) showed this emerges at scale."
        )

        t.user("How does it relate to tree-of-thought?")
        t.assistant(
            "Tree-of-thought extends chain-of-thought by exploring multiple "
            "reasoning paths in parallel, using search (BFS/DFS) to find the "
            "best solution. It trades compute for reasoning quality."
        )

        t.user("What about ReAct?")
        t.assistant(
            "ReAct interleaves reasoning and acting -- the model generates "
            "thought traces AND takes actions (tool calls), grounding its "
            "reasoning in external observations."
        )

        # --- Show the conversation before agent ops ---

        print("=== Conversation (before agent ops) ===\n")
        t.compile().pprint(style="chat")

        # --- Sidecar branch: run context management on a scratch branch ---
        # Annotations (pinning) are commit-level, not branch-level, so they
        # persist when we switch back to main. The tool interaction turns
        # (reasoning traces, tool results) stay on the scratch branch.

        print("=== Agent managing context (sidecar branch) ===\n")
        t.branch("_context-mgmt", switch=True)

        t.user(
            "Review your context window. Check the status, pin any important "
            "definitions worth preserving, and skip anything that is not "
            "essential. Do not compress -- just annotate."
        )

        reply = react_loop(t, executor)
        print(f"Agent: {reply}\n")

        # --- Switch back to main -- tool turns stay on scratch branch ---

        t.switch("main")

        # --- Show the conversation after agent ops ---
        # Main branch is unchanged, but annotations (pinning/skipping) now
        # affect the compiled output. Compare priorities before vs after.

        print("=== Conversation (after agent ops) ===\n")
        t.compile().pprint(style="chat")
        print()
        t.status().pprint()


if __name__ == "__main__":
    main()


# --- See also ---
# Self-managing patterns: agentic/self_managing/01_tool_hints.py
# Sidecar patterns: agentic/sidecar/01_triggers.py
# Full toolkit reference: agentic/sidecar/03_toolkit.py
# Developer on-ramp: getting_started/01_chat.py
