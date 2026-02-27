"""PendingMerge conflict resolution hooks: intercept merge conflicts, inspect
resolutions, edit individual conflict resolutions with edit_resolution(), and
steer the resolver with edit_guidance().
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.merge import PendingMerge

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def merge_conflict_hooks():
    print("=" * 60)
    print("PendingMerge: Conflict Resolution Hooks")
    print("=" * 60)
    print()
    print("  Only merges WITH conflicts fire hooks.")
    print("  Fast-forward and clean merges proceed without interception.")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Build a conflict: both branches EDIT the same message
        sys_ci = t.system("You are a helpful assistant.")
        user_ci = t.user("What is Python?")
        t.assistant("Python is a programming language.")

        # Feature branch edits the assistant message
        t.branch("feature")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Python is a high-level, interpreted language created by Guido van Rossum.",
        )

        # Main also edits the same message -> conflict
        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Python is a versatile language popular in data science and web development.",
        )

        # --- review=True: get PendingMerge ---
        pending: PendingMerge = t.merge("feature", review=True)

        # pprint shows branches, conflicts with resolution status, guidance, actions
        pending.pprint()

        # --- edit_resolution: replace one ---
        first_key = list(pending.resolutions.keys())[0]
        pending.edit_resolution(
            first_key,
            "Python is a high-level language popular in data science, web dev, and automation.",
        )
        print(f"\n  After edit_resolution({first_key[:8]}, ...):")
        print(f"    Resolution: {pending.resolutions[first_key][:80]}")

        # --- Guidance ---
        pending.edit_guidance("Prefer the version that mentions more use cases.")
        print(f"\n  guidance: {pending.guidance}")
        print(f"  guidance_source: {pending.guidance_source}")

        # --- Approve ---
        result = pending.approve()
        print(f"\n  Approved! Merge complete")
        pending.pprint()

    # --- Hook handler pattern: auto-resolve ---
    print(f"\n  Hook pattern: auto-pick incoming version")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("What is Rust?")
        t.assistant("Rust is a language.")

        t.branch("feature2")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Rust is a systems programming language focused on safety.",
        )

        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Rust is a modern compiled language.",
        )

        def prefer_incoming(pending: PendingMerge):
            """Always pick the incoming (source branch) version."""
            for conflict in pending.conflicts:
                key = getattr(conflict, "target_hash", None)
                if key and hasattr(conflict, "content_b_text"):
                    pending.set_resolution(key, conflict.content_b_text)
            pending.approve()

        t.on("merge", prefer_incoming, name="prefer-incoming")
        result = t.merge("feature2")

        t.print_hooks()


if __name__ == "__main__":
    merge_conflict_hooks()
