"""Edit in Place

Three tiers of editing: manual edit with chat verification, interactive
editing via $EDITOR, and a note on agent-driven edits.

PART 1 -- Manual           Direct API calls, no LLM, deterministic
PART 2 -- Interactive       review=True, click.edit/confirm, human decides
PART 3 -- LLM / Agent      Orchestrator, triggers, hooks auto-manage

Demonstrates: system(edit=hash), annotate(SKIP) for stale responses,
              chat() before/after edit, compile() serves corrected
              content, log() preserves both versions, click.edit()
"""

import os

import click
from dotenv import load_dotenv

from tract import Priority, Tract

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ.get("TRACT_OPENAI_API_KEY", "")
TRACT_OPENAI_BASE_URL = os.environ.get("TRACT_OPENAI_BASE_URL", "")
MODEL_ID = "gpt-oss-120b"


# =============================================================================
# Part 1: Edit in Place  (PART 1 — Manual, with LLM verification)
# =============================================================================

def part1_edit_in_place():
    print("=" * 60)
    print("Part 1: EDIT IN PLACE  [Manual Tier]")
    print("=" * 60)
    print()
    print("  A support agent's system prompt says '60-day return policy' —")
    print("  but it's actually 30 days. Chat with the LLM, see it parrot the")
    print("  wrong info, then edit the system prompt in place and ask again.")
    print("  Both versions stay in history for audit.")
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:

        # --- System prompt with a mistake baked in ---

        bad_prompt = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 60 days.",
            message="Initial system prompt with incorrect return policy",
        )
        print(f"System prompt committed: {bad_prompt.commit_hash[:8]}\n")

        # --- Ask about returns — the LLM will cite the wrong policy ---

        print("=== Before edit ===\n")
        stale_q = t.user("What's your return policy?")
        stale_a = t.generate()
        stale_a.pprint()

        # --- Fix the system prompt: 60 days -> 30 days ---
        # edit= replaces the target commit in compiled context.
        # Same shorthand, one extra param.

        fix = t.system(
            "You are a customer support agent for Acme Corp.\n"
            "Return policy: customers may return any item within 30 days.",
            edit=bad_prompt.commit_hash,
            message="Fix return policy from 60 days to 30 days",
        )
        fix.pprint()

        # --- Skip the stale Q&A — it was based on the wrong prompt ---

        t.annotate(stale_q.commit_hash, Priority.SKIP, reason="based on wrong prompt")
        t.annotate(stale_a.commit_info.commit_hash, Priority.SKIP, reason="based on wrong prompt")
        print("Skipped stale Q&A (based on the old 60-day prompt)\n")

        # --- Ask again — the LLM now sees the corrected prompt ---

        print("=== After edit ===\n")
        response = t.chat("What's your return policy?")
        response.pprint()

        # --- Compiled context: only the corrected version appears ---

        print("\n=== Compiled context (what the LLM sees now) ===\n")
        t.compile().pprint()

        # --- Full history: both system prompts preserved for audit ---

        print("=== Full history (both versions preserved) ===\n")
        for entry in reversed(t.log()):
            print(f"  {entry}")


# =============================================================================
# Part 2: Interactive Edit via $EDITOR  (PART 2 — Interactive)
# =============================================================================

def part2_interactive_edit():
    print("=" * 60)
    print("Part 2: INTERACTIVE EDIT VIA $EDITOR  [Interactive Tier]")
    print("=" * 60)
    print()
    print("  Open the current system prompt in $EDITOR, make changes,")
    print("  then confirm and apply the edit.")
    print()

    t = Tract.open()

    sys_ci = t.system(
        "You are a customer support agent for Acme Corp.\n"
        "Return policy: customers may return any item within 60 days.\n"
        "Tone: friendly and professional.",
    )
    t.user("Tell me about returns.")
    t.assistant("Our return policy allows returns within 60 days of purchase.")

    # Show current system prompt
    old_text = t.get_content(sys_ci.commit_hash)["text"]
    print(f"  Current system prompt:\n    {old_text[:80]}...\n")

    # Open in $EDITOR
    edited = click.edit(old_text)
    if edited and edited.strip() != old_text.strip():
        if click.confirm("  Apply this edit to the system prompt?"):
            new_ci = t.system(edited.strip(), edit=sys_ci.commit_hash)
            print(f"  Edit applied: {new_ci.commit_hash[:8]}\n")

            print("  --- Compiled context (after edit) ---\n")
            t.compile().pprint()
        else:
            print("  Edit cancelled.")
    else:
        print("  No changes made (or editor closed without saving).")

    print()
    t.close()


# =============================================================================
# Part 3: Agent-Driven Edits  (PART 3 — LLM / Agent)
# =============================================================================

def part3_agent_note():
    print("=" * 60)
    print("Part 3: AGENT-DRIVEN EDITS  [Agent Tier — Note]")
    print("=" * 60)
    print()
    print("  Editing is inherently a manual or interactive operation —")
    print("  an agent needs human judgment to know *what* to change.")
    print()
    print("  For agent-driven edits via the toolkit, see:")
    print("    orchestrator/01_toolkit.py")
    print("  where agents can execute edit operations using ToolExecutor:")
    print("    executor.execute('commit', {'content': new_text, 'edit': old_hash})")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    part1_edit_in_place()
    part2_interactive_edit()
    part3_agent_note()
    print("=" * 60)
    print("Done -- all 3 tiers of edit-in-place demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
