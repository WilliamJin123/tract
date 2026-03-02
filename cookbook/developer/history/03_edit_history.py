"""Edit History

Chat with an LLM, then iteratively refine a response via edits.
Use edit_history() to see every version of a commit, and restore()
to roll back when the edits go too far.

Demonstrates: t.assistant(edit=), edit_history(), restore(),
              get_content(), Priority.SKIP for cleaning up
              intermediate commits, pprint(style="chat"),
              response.pprint()
"""

import os

import click
from dotenv import load_dotenv

from tract import Priority, Tract, ToolCall

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"
MODEL_ID_SMALL = "llama3.1-8b"


def part3_edit_history():
    print(f"\n{'=' * 60}")
    print("PART 3 -- Manual: EDIT HISTORY")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID_SMALL,
    ) as t:

        t.system("You are a concise writing assistant. Keep answers under 2 sentences.")

        # --- Initial conversation ---

        print("=== Initial question ===\n")
        r1 = t.chat("Explain what a black hole is.")
        r1.pprint()
        original_hash = r1.commit_info.commit_hash

        # --- Ask a follow-up so we have surrounding context ---

        print("=== Follow-up ===\n")
        r2 = t.chat("How are they detected?")
        r2.pprint()

        # --- Edit the first response to add more detail ---
        # t.assistant(edit=...) replaces the content of a previous response.
        # All edits point to the ORIGINAL commit (flat design, not chained).

        print("=== Edit 1: ask LLM to improve the first answer ===\n")
        improve = t.chat(
            "Please rewrite your first answer about black holes to also "
            "mention the event horizon. Keep it to 2 sentences."
        )
        # Use the LLM's improved text as an edit of the original
        e1 = t.assistant(
            improve.text,
            edit=original_hash,
            message="Add event horizon detail",
        )
        print(f"  Edit commit: {e1.commit_hash[:8]}")
        print(f"  Content: {t.get_content(e1)}\n")

        # The t.chat() call above created intermediate commits (user prompt +
        # LLM response) that would clutter the compiled context. SKIP them
        # so only the edit itself survives in the conversation view.
        t.annotate(improve.commit_info.parent_hash, Priority.SKIP)
        t.annotate(improve.commit_info.commit_hash, Priority.SKIP)

        # --- Edit again: further refinement ---

        print("=== Edit 2: manual refinement ===\n")
        e2 = t.assistant(
            "A black hole is a region of spacetime where gravity is so "
            "extreme that nothing, not even light, can escape past its "
            "event horizon. They form when massive stars collapse at the "
            "end of their life cycle.",
            edit=original_hash,
            message="Manual rewrite for clarity",
        )
        print(f"  Edit commit: {e2.commit_hash[:8]}")

        # --- View the full edit history ---
        # edit_history() returns [original, edit1, edit2, ...] in order.
        # This is a lightweight query -- no full context compilation needed.

        print("\n=== Edit history for the first answer ===\n")
        history = t.edit_history(original_hash)
        for i, version in enumerate(history):
            label = "ORIGINAL" if i == 0 else f"EDIT {i}"
            content = t.get_content(version)
            print(f"  v{i} ({label}) [{version.commit_hash[:8]}]")
            print(f"     {content}")
            print()

        print(f"  Total versions: {len(history)}")

        # --- The compiled context uses the latest edit automatically ---

        print("\n=== Compiled context (latest edit wins) ===\n")
        t.compile().pprint(style="chat")

        # --- Restore: the manual edit was too verbose, go back to v1 ---
        # restore() creates a NEW edit pointing to the original, with the
        # content from the specified version. The full history is preserved.

        print("\n=== Restore to v1 (LLM-improved version) ===\n")
        restored = t.restore(original_hash, version=1)
        print(f"  Restore commit: {restored.commit_hash[:8]}")
        print(f"  edit_target: {restored.edit_target[:8]} (points to original)")
        print(f"  Content: {t.get_content(restored)}\n")

        # --- Verify the restore is tracked in history ---

        print("=== Updated edit history (restore is itself an edit) ===\n")
        updated_history = t.edit_history(original_hash)
        for i, version in enumerate(updated_history):
            msg = version.message or "(no message)"
            if len(msg) > 60:
                msg = msg[:57] + "..."
            print(f"  v{i} [{version.commit_hash[:8]}] {msg}")
        print(f"\n  Total versions: {len(updated_history)} "
              f"(was {len(history)}, +1 from restore)")

        # --- Surrounding context is unaffected ---

        print("\n=== Full compiled context after restore ===\n")
        ctx = t.compile()
        ctx.pprint(style="chat")
        print(f"\n  The follow-up answer about detection is still intact.")
        print(f"  Only the black hole definition was rolled back to v1.")


def part2_interactive():
    """Part 2: Interactive -- human picks a response to edit and can restore."""
    print(f"\n{'=' * 60}")
    print("PART 2 -- Interactive: EDIT HISTORY")
    print("=" * 60)
    print()

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID_SMALL,
    ) as t:
        t.system("You are a concise writing assistant. Keep answers under 2 sentences.")

        # Build a few assistant responses
        r1 = t.chat("Explain what a black hole is.")
        r2 = t.chat("Explain what dark matter is.")
        r3 = t.chat("Explain what a neutron star is.")

        responses = [r1, r2, r3]
        prompts = ["black hole", "dark matter", "neutron star"]

        # Show numbered list of assistant responses
        print("  Assistant responses:\n")
        for i, r in enumerate(responses):
            print(f"    {i+1}. [{prompts[i]}] {r.text[:80]}...")
        print()

        # Let user pick which to edit
        choice = click.prompt(
            "  Edit which response? (number)", type=int, default=1,
        )
        idx = max(1, min(choice, len(responses))) - 1
        target = responses[idx]
        original_hash = target.commit_info.commit_hash

        # Open in editor
        edited = click.edit(target.text)
        if edited and edited.strip() != target.text:
            ci = t.assistant(
                edited.strip(),
                edit=original_hash,
                message=f"Interactive edit of '{prompts[idx]}' response",
            )
            print(f"  Edit committed: {ci.commit_hash[:8]}")
        else:
            print("  (no changes)")

        # Offer to view edit history
        if click.confirm("\n  View edit history?", default=True):
            history = t.edit_history(original_hash)
            print(f"\n  {len(history)} version(s) for {original_hash[:8]}:\n")
            for i, version in enumerate(history):
                label = "ORIGINAL" if i == 0 else f"EDIT {i}"
                content = t.get_content(version)
                print(f"    v{i} ({label}) [{version.commit_hash[:8]}]")
                print(f"       {str(content)[:80]}")
                print()

            # Offer restore if there are edits
            if len(history) > 1:
                if click.confirm(f"  Restore to version 0 (original)?", default=False):
                    restored = t.restore(original_hash, version=0)
                    print(f"  Restored: {restored.commit_hash[:8]}")
                    print(f"  Content: {t.get_content(restored)[:80]}...")


# =============================================================================
# Part 3b -- Agent: Traces Own Edit History
# =============================================================================
# Agents can detect and understand their own corrections by walking
# the edit chain. Useful for self-reflection and learning from mistakes.

def part3b_agent():
    print(f"\n{'=' * 60}")
    print("PART 3b -- Agent: TRACES OWN EDIT HISTORY")
    print("=" * 60)
    print()

    from tract.toolkit import ToolExecutor

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a concise assistant.")
        executor = ToolExecutor(t)

        # Initial response
        r1 = t.chat("Define entropy.")
        original_hash = r1.commit_info.commit_hash

        # Edit the response
        t.assistant("Entropy is a measure of disorder or randomness "
                    "in a thermodynamic system.", edit=original_hash)

        # Agent traces edit history
        history = t.edit_history(original_hash)
        print(f"  Edit chain for {original_hash[:8]}:")
        for i, version in enumerate(history):
            label = "ORIGINAL" if i == 0 else f"EDIT {i}"
            content = t.get_content(version)
            print(f"    v{i} ({label}): {str(content)[:60]}...")

    # Note: Agents can detect and understand their own corrections
    # by walking the edit chain. This enables self-reflection patterns
    # where the agent reasons about why it needed to revise an answer.


def main():
    part3_edit_history()
    part2_interactive()
    part3b_agent()


if __name__ == "__main__":
    main()
