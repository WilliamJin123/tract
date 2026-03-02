"""LLM-Driven Auto-Tagging via Orchestrator

Three tiers of auto-tagging: manual heuristics, interactive confirmation,
and fully autonomous LLM-driven tagging via the orchestrator.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

The key difference from 01_classify_and_query.py: here the focus is on
*automated* tagging strategies rather than explicit tag management.

Demonstrates: register_tag, role-based heuristics, click.confirm(),
              Orchestrator, OrchestratorConfig, AutonomyLevel,
              TAGGER_SYSTEM_PROMPT + build_tagger_task_prompt
"""

import os

import click
from dotenv import load_dotenv

from tract import Tract
from tract.formatting import pprint_log, pprint_tag_registry
from tract.orchestrator import Orchestrator, OrchestratorConfig, AutonomyLevel
from tract.prompts.tagger import TAGGER_SYSTEM_PROMPT, build_tagger_task_prompt

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Manual Heuristic Tagging  (PART 1 — Manual)
# =============================================================================

def part1_manual_heuristics():
    """Tag commits based on role heuristics — no LLM needed."""
    print("=" * 60)
    print("Part 1: MANUAL HEURISTIC TAGGING  [Manual Tier]")
    print("=" * 60)
    print()
    print("  Register tags, then loop through log() and apply tags based")
    print("  on simple role heuristics: user -> 'question', assistant -> 'answer'.")
    print()

    t = Tract.open()

    # Register custom tags
    t.register_tag("question", "A question from the user")
    t.register_tag("answer", "An answer from the assistant")

    # Build a short conversation
    t.system("You are a project manager AI tracking a sprint.")
    t.user("What's the status of the auth migration?")
    t.assistant("The OAuth2 migration is 70% complete.")
    t.user("The legacy session store is blocking QA.")
    t.assistant("I recommend we prioritize the session store cutover.")

    # Apply tags based on role
    tagged = 0
    for entry in t.log():
        if entry.role == "user":
            t.tag(entry.commit_hash, "question")
            tagged += 1
        elif entry.role == "assistant":
            t.tag(entry.commit_hash, "answer")
            tagged += 1

    print(f"  Tagged {tagged} commits by role heuristic.\n")

    # Show results
    for entry in reversed(t.log()):
        tags = t.get_tags(entry.commit_hash)
        label = (entry.message or "")[:50]
        print(f"    {entry.commit_hash[:8]}  {entry.role:9s}  tags={tags}  {label}")

    print()
    t.close()


# =============================================================================
# Part 2: Interactive Semi-Auto Tagging  (PART 2 — Interactive)
# =============================================================================

def part2_interactive():
    """Suggest tags based on content, confirm with the user."""
    print("=" * 60)
    print("Part 2: INTERACTIVE SEMI-AUTO TAGGING  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  For each commit, suggest a tag based on role/content length.")
    print("  The user confirms or skips each suggestion.")
    print()

    t = Tract.open()

    t.register_tag("question", "A question from the user")
    t.register_tag("answer", "An answer from the assistant")
    t.register_tag("context", "Background context or status update")

    t.system("You are a project manager AI tracking a sprint.")
    t.user("What's the status of the auth migration?")
    t.assistant(
        "The OAuth2 migration is 70% complete. Token refresh "
        "is done, but the session store hasn't been switched yet."
    )
    t.user("The legacy session store is blocking QA — they can't "
           "test the new flow until it's gone.")
    t.assistant("I recommend we prioritize the session store cutover.")

    for entry in t.log():
        if entry.role == "system":
            continue

        # Suggest based on role and content length
        content = entry.content_text or ""
        if entry.role == "user" and "?" in content:
            suggested = "question"
        elif entry.role == "user":
            suggested = "context"
        elif entry.role == "assistant" and len(content) > 80:
            suggested = "context"
        else:
            suggested = "answer"

        preview = content[:60].replace("\n", " ")
        print(f"  {entry.commit_hash[:8]} [{entry.role}] {preview}...")
        if click.confirm(f"    Tag as '{suggested}'?", default=True):
            t.tag(entry.commit_hash, suggested)
            print(f"    -> tagged '{suggested}'")
        else:
            print(f"    -> skipped")

    print()
    pprint_log(list(reversed(t.log())))
    print()
    t.close()


# =============================================================================
# Part 3: LLM-Driven Auto-Tagging  (PART 3 — LLM / Agent)
# =============================================================================

def part3_agent():
    if not TRACT_OPENAI_API_KEY:
        print("=" * 60)
        print("Part 3: SKIPPED (no TRACT_OPENAI_API_KEY)")
        print("=" * 60)
        return

    # -----------------------------------------------------------------
    # Step 1: Build a conversation (no custom tags)
    # -----------------------------------------------------------------

    print("=" * 60)
    print("Part 3: LLM-DRIVEN AUTO-TAGGING  [Agent Tier]")
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


# =============================================================================
# Main
# =============================================================================

def main():
    part1_manual_heuristics()
    part2_interactive()
    part3_agent()
    print("=" * 60)
    print("Done -- all 3 tiers of auto-tagging demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
