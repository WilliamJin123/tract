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

from dotenv import load_dotenv

from tract import Priority, Tract, ToolCall

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"
MODEL_ID_SMALL = "llama3.1-8b"


def part3_edit_history():
    print(f"\n{'=' * 60}")
    print("Part 3: EDIT HISTORY")
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


if __name__ == "__main__":
    part3_edit_history()
