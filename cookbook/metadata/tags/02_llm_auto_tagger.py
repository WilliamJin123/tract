"""LLM-Driven Auto-Tagging via Orchestrator

An orchestrator agent reviews a completed conversation and retrospectively
tags each message — no hardcoded rules, no manual LLM calls.  The agent
uses tract's built-in toolkit (log, get_tags, register_tag, tag) to read
history and apply semantic labels like "question", "decision",
"action_item", etc.

This is the key difference from 01_classify_and_query.py: the LLM decides
what tags to apply by calling tools autonomously, rather than the developer
hardcoding tags=["..."] at commit time.

Demonstrates: Orchestrator, OrchestratorConfig, AutonomyLevel,
              register_tag / get_tags / list_tags / tag tools,
              TAGGER_SYSTEM_PROMPT + build_tagger_task_prompt from prompts
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.formatting import pprint_log, pprint_tag_registry
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel
from tract.prompts.tagger import TAGGER_SYSTEM_PROMPT, build_tagger_task_prompt

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def main():
    # =================================================================
    # Step 1: Build a conversation (no custom tags)
    # =================================================================

    print("=" * 60)
    print("Step 1: BUILD CONVERSATION (no custom tags)")
    print("=" * 60)
    print()

    t = Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    )

    t.system("You are a project manager AI tracking a sprint.")

    messages = [
        ("user",      "What's the status of the auth migration?"),
        ("assistant", "The OAuth2 migration is 70% complete. Token refresh "
                      "is done, but the session store hasn't been switched yet."),
        ("user",      "The legacy session store is blocking QA — they can't "
                      "test the new flow until it's gone."),
        ("assistant", "Understood. I recommend we prioritize the session store "
                      "cutover this sprint. That unblocks QA and lets us run "
                      "integration tests in parallel."),
        ("user",      "Agreed. Let's move the session store task to the top "
                      "of the backlog. Who should own it?"),
        ("assistant", "Alice has the most context on the session layer. I'll "
                      "assign the task to her with a Wednesday deadline. "
                      "Summary: session store cutover is sprint priority #1, "
                      "owned by Alice, due Wednesday."),
    ]

    commit_hashes = []
    for role, text in messages:
        if role == "user":
            ci = t.user(text)
        else:
            ci = t.assistant(text)
        commit_hashes.append(ci.commit_hash)
        print(f"  [{role:9s}] {text[:60]}...")

    print(f"\n  {len(commit_hashes)} messages committed, 0 custom tags.\n")

    # Show auto-classified tags before orchestrator runs
    print("  Auto-classified tags (built-in heuristics):")
    for h in commit_hashes:
        tags = t.get_tags(h)
        ci = t.get_commit(h)
        label = (ci.message or "")[:40]
        print(f"    {h[:8]}  {tags}  {label}")
    print()

    # =================================================================
    # Step 2: Run the orchestrator as a tagger agent
    # =================================================================

    print("=" * 60)
    print("Step 2: ORCHESTRATOR TAGS MESSAGES VIA TOOLS")
    print("=" * 60)
    print()

    config = OrchestratorConfig(
        autonomy_ceiling=AutonomyLevel.AUTONOMOUS,
        max_steps=50,       # enough for register + log + tag calls
        profile="self",     # includes tag, get_tags, register_tag, list_tags, log
        system_prompt=TAGGER_SYSTEM_PROMPT,
        task_context=build_tagger_task_prompt(),
    )

    orch = Orchestrator(t, config=config)
    result = orch.run()

    print(f"  Orchestrator completed: {result.total_tool_calls} tool calls, "
          f"state={result.state.value}")
    print()

    # Print what the agent did
    for step in result.steps:
        status = "OK" if step.success else "FAIL"
        args_short = str(step.tool_call.arguments)[:60]
        print(f"  [{status:4s}] {step.tool_call.name}({args_short})")
    print()

    # =================================================================
    # Step 3: Show the results
    # =================================================================

    print("=" * 60)
    print("Step 3: TAGGED CONVERSATION")
    print("=" * 60)
    print()

    tagged_entries = [t.get_commit(h) for h in commit_hashes]
    pprint_log(tagged_entries)

    # =================================================================
    # Step 4: Query by tags
    # =================================================================

    print("=" * 60)
    print("Step 4: QUERY BY TAGS")
    print("=" * 60)
    print()

    for tag_name in ["question", "decision", "action_item", "blocker", "context", "summary"]:
        results = t.query_by_tags([tag_name])
        if results:
            print(f"  [{tag_name}] — {len(results)} commit(s):")
            for r in results:
                msg = (r.message or r.content_text or "")[:60]
                print(f"    {r.commit_hash[:8]}  {msg}")
            print()

    # AND query
    multi = t.query_by_tags(["decision", "action_item"], match="all")
    if multi:
        print(f"  [decision AND action_item] — {len(multi)} commit(s):")
        for r in multi:
            msg = (r.message or r.content_text or "")[:60]
            print(f"    {r.commit_hash[:8]}  {msg}")
        print()

    # =================================================================
    # Step 5: Tag registry with counts
    # =================================================================

    print("=" * 60)
    print("Step 5: TAG REGISTRY")
    print("=" * 60)
    print()

    pprint_tag_registry([e for e in t.list_tags() if e["count"] > 0])

    print()
    print("=" * 60)
    print("Done — the LLM tagged every message using built-in tools,")
    print("no hardcoded rules or manual API calls.")
    print("=" * 60)

    t.close()


if __name__ == "__main__":
    main()
