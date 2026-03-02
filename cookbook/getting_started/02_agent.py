"""Hello Agent -- The Agent On-Ramp

An agent that manages its own context window using tract tools. Instead of
you calling compress() or annotate() directly, the agent gets these as
callable tools and decides when to use them.

This file shows the building blocks: as_tools() exposes tract operations
as LLM-callable tool schemas, and ToolExecutor dispatches the calls. For
the full self-managing pattern (description-driven tool hints, no system
prompt crutches), see agentic/self_managing/. For the sidecar pattern
(companion agent handles context), see agentic/sidecar/.

Demonstrates: as_tools(), ToolExecutor, agent-driven status/annotate/compress
"""

import os

from dotenv import load_dotenv

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolExecutor

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    db_path = os.path.join(os.path.curdir, "getting_started_agent.db")
    if os.path.exists(db_path):
        os.unlink(db_path)

    # --- Open a tract with LLM config and a token budget ---

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=4000))

    with Tract.open(
        db_path,
        tract_id="agent-demo",
        config=config,
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # --- Build up some conversation history ---

        t.system("You are an AI research assistant with context management tools.")

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

        # --- Get tract tools in OpenAI format (self profile) ---

        tools = t.as_tools(format="openai", profile="self")
        tool_names = [tool["function"]["name"] for tool in tools]
        print(f"Agent has {len(tools)} tools: {', '.join(tool_names)}\n")

        # --- ToolExecutor: the dispatch layer agents use ---

        executor = ToolExecutor(t)
        executor.set_profile("self")

        # Agent checks its context health
        result = executor.execute("status", {})
        print(f"[status] {result}\n")

        # Agent pins an important message it wants to protect from compression.
        # Grab the chain-of-thought commit hash from the log.
        log_entries = t.log(limit=10)
        cot_hash = None
        for entry in log_entries:
            if entry.message and "chain-of-thought" in entry.message.lower():
                cot_hash = entry.commit_hash
                break

        # Fall back to the first assistant response if auto-message didn't match
        if cot_hash is None:
            assistant_entries = [e for e in log_entries if "dialogue" in e.content_type]
            if len(assistant_entries) >= 2:
                cot_hash = assistant_entries[-2].commit_hash  # second-oldest dialogue

        if cot_hash:
            result = executor.execute("annotate", {
                "target_hash": cot_hash,
                "priority": "pinned",
                "reason": "Key definition -- protect from compression",
            })
            print(f"[annotate] {result}\n")

        # Agent compresses when context budget is getting high
        result = executor.execute("compress", {"target_tokens": 200})
        print(f"[compress] {result}\n")

        # --- After agent operations, check the result ---

        result = executor.execute("status", {})
        print(f"[status after ops] {result}\n")

        result = executor.execute("log", {"limit": 10})
        print(f"[log]\n{result}")


if __name__ == "__main__":
    main()


# --- See also ---
# Self-managing patterns: agentic/self_managing/01_tool_hints.py
# Sidecar patterns: agentic/sidecar/01_triggers.py
# Full toolkit reference: agentic/sidecar/03_toolkit.py
# Developer on-ramp: getting_started/01_chat.py
