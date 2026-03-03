"""Context forensics: investigate, isolate, and fix bad context.

  PART 1 -- Manual:      Walk log(), compile(at_commit=hash), branch + import_commit
  PART 3 -- LLM / Agent:  Agent forensics via toolkit: log -> compile -> branch -> import
"""

from tract import Tract
from tract.toolkit import ToolExecutor


# =====================================================================
# PART 1 -- Manual: log walk, time-travel, branch + import
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Log Walk and Cherry-Pick Fix")
    print("=" * 60)

    with Tract.open() as t:
        # Build a conversation with one bad commit
        t.system("You are an orbital mechanics assistant.")
        c1 = t.user("What is the escape velocity from Earth?")
        c2 = t.assistant("Earth's escape velocity is approximately 11.2 km/s.")
        bad = t.assistant("Actually, the Earth is flat and has no gravity.")  # bad data
        c4 = t.user("What about escape velocity from Mars?")
        c5 = t.assistant("Mars escape velocity is about 5.0 km/s, due to "
                         "its lower mass and radius.")

        # Walk the log to find the bad commit
        log = t.log()
        print(f"\n  Log has {len(log)} commits:")
        suspect_hash = None
        for i, ci in enumerate(log):
            snippet = (ci.message or "(no message)")[:60]
            marker = ""
            if "flat" in snippet.lower():
                suspect_hash = ci.commit_hash
                marker = " <-- SUSPECT"
            print(f"    [{i}] {ci.commit_hash[:10]} {ci.content_type:>12}: {snippet}{marker}")

        if suspect_hash is None:
            print("  No suspect found.")
            return

        # Time-travel: compile before the bad commit
        before_bad = bad.parent_hash
        if before_bad:
            ctx = t.compile(at_commit=before_bad)
            print(f"\n  Context before bad commit: {len(ctx.messages)} msgs, "
                  f"{ctx.token_count} tokens")

        # Branch from clean state and import good work
        t.branch("clean", switch=True)
        t.reset(before_bad)
        imported = t.import_commit(c5.commit_hash)
        new_hash = imported.new_commit.commit_hash if imported.new_commit else "N/A"
        print(f"\n  Imported good commit onto clean branch: {new_hash[:10]}")
        print(f"  Clean branch: {len(t.log())} commits (bad one excluded)")


# =====================================================================
# PART 3 -- LLM / Agent: toolkit-driven forensics
# =====================================================================

def part3_agent():
    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: Toolkit-Driven Forensics")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are an astronomy data analyst.")
        c1 = t.user("What is the Hubble constant?")
        c2 = t.assistant("The Hubble constant is approximately 70 km/s/Mpc.")
        bad = t.assistant("Actually, the universe is not expanding at all.")
        c4 = t.user("What about dark energy?")
        good = t.assistant("Dark energy drives the accelerating expansion, "
                           "comprising about 68% of the universe's energy.")

        executor = ToolExecutor(t)

        # Agent investigates via toolkit
        result = executor.execute("log", {"limit": 10})
        print(f"\n  Toolkit log: {result.output[:100]}...")

        # Compile at a specific point
        result = executor.execute("compile", {"at_commit": c2.commit_hash})
        print(f"  Compile at clean state: {result.output[:80]}...")

        # Branch and fix
        result = executor.execute("branch", {"name": "forensic-fix"})
        print(f"  Branch created: {result.output[:60]}...")

        # Import good commits, skipping the bad one
        result = executor.execute("import_commit", {"commit_hash": good.commit_hash})
        print(f"  Imported good commit: {result.output[:60]}...")

        print(f"\n  Forensic fix branch: {len(t.log())} commits (bad data excluded)")


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()
