"""PendingRebase hooks: intercept rebase to review the replay plan,
exclude commits you want to drop, and check warnings before approving.
"""

import os

from dotenv import load_dotenv

from typing import Any

from tract import Tract
from tract.hooks.event import HookEvent
from tract.hooks.rebase import PendingRebase
from tract.models.commit import CommitInfo

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def rebase_hooks() -> None:
    print("=" * 60)
    print("PendingRebase: Review Before Replay")
    print("=" * 60)

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a fitness coach helping design personalized workout plans.")
        t.chat("I want to start strength training three days a week. What's a good beginner split?")

        # Create a feature branch with several commits
        t.branch("feature")
        feature_commits: list[CommitInfo] = []
        feature_questions = [
            "What about adding cardio on off days?",
            "How many sets and reps should I aim for?",
            "Should I do compound or isolation exercises first?",
            "When should I increase the weight?",
        ]
        feature_answers = [
            "Light cardio like walking or cycling on rest days can improve recovery.",
            "For beginners, 3 sets of 8-12 reps per exercise is a solid starting point.",
            "Always start with compound movements when you're freshest, then isolations.",
            "When you can complete all sets with good form, add 5 lbs for upper body, 10 for lower.",
        ]
        for q, a in zip(feature_questions, feature_answers):
            ci: CommitInfo = t.user(q)
            feature_commits.append(ci)
            ci2: CommitInfo = t.assistant(a)
            feature_commits.append(ci2)

        # Add a commit on main so rebase has something to do
        t.switch("main")
        t.user("Actually, I also want to improve my flexibility. Any quick stretching routines?")
        t.assistant("A 10-minute dynamic warmup before lifting and static stretching after works well.")
        t.switch("feature")

        # --- review=True: get PendingRebase ---
        pending: PendingRebase = t.rebase("main", review=True)

        # pprint shows status, replay_plan, target_base, warnings, actions
        pending.pprint()

        # --- Exclude a commit: skip it during replay ---
        drop_hash: str = pending.replay_plan[0]
        pending.exclude(drop_hash)
        print(f"\n  Excluded {drop_hash[:12]} from replay")
        print(f"    replay_plan: {len(pending.replay_plan)} commits (was {len(pending.replay_plan) + 1})")

        # --- Approve ---
        result: Any = pending.approve()
        print(f"\n  Approved! Rebase complete")
        pending.pprint()

    # --- Hook handler pattern ---
    print(f"\n  Hook pattern: warn-and-approve")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a fitness coach helping design personalized workout plans.")
        t.chat("I want to start strength training three days a week. What's a good beginner split?")

        t.branch("experiment")
        t.user("What about training for a 5K while strength training?")
        t.assistant("You can do both — run on off days and keep lifting sessions under an hour.")
        t.user("Will running hurt my muscle gains?")
        t.assistant("Not significantly if you eat enough protein and keep runs moderate.")
        t.user("How should I structure the weekly schedule?")
        t.assistant("Try Mon/Wed/Fri lifting, Tue/Thu running, weekends rest or light activity.")

        t.switch("main")
        t.user("I just realized I should also track my nutrition. Any simple approach?")
        t.assistant("Start by tracking protein intake — aim for 0.7-1g per pound of bodyweight daily.")
        t.switch("experiment")

        def warn_and_approve(pending: PendingRebase) -> None:
            """Log the replay plan, then approve."""
            pending.pprint()
            if pending.warnings:
                for w in pending.warnings:
                    print(f"    [hook] WARNING: {w}")
            pending.approve()

        t.on("rebase", warn_and_approve, name="warn-and-approve")
        t.rebase("main")  # Handler fires

        # hook_log shows the handler invocation and result
        for evt in t.hook_log:
            ts = evt.timestamp.strftime("%H:%M:%S")
            print(f"  [{ts}] {evt.operation} -> {evt.handler_name}: {evt.result}")


if __name__ == "__main__":
    rebase_hooks()
