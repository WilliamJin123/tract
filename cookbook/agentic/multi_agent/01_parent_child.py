"""Parent-child tract relationships.

session.spawn(), parent.children(), child.parent()

Session management (spawn, deploy, collapse) is a developer-side concern --
the Orchestrator does not handle it.
"""

from tract import Session


# =====================================================================
# Spawn, navigate parent/child, import
# =====================================================================

def manual():
    print("=" * 60)
    print("Parent-Child Spawn")
    print("=" * 60)

    session = Session.open()
    parent = session.create_tract(display_name="coordinator")

    # Parent seeds the conversation
    parent.system("You are a research coordinator for exoplanet studies.")
    parent.user("We need to catalogue habitable zone exoplanets.")
    parent.assistant("I will organize research across multiple sub-topics.")

    print("\n  Parent context:")
    parent.compile().pprint(style="chat")

    print(f"\n  Parent tract: {parent.tract_id[:12]}")
    print(f"  Parent commits: {len(parent.log())}")

    # Spawn a child for focused research
    child = session.spawn(parent, purpose="research Kepler mission results")
    print(f"\n  Child tract: {child.tract_id[:12]}")
    print(f"  Child inherited commits: {len(child.log())}")

    # Child works independently
    child.user("What were the key findings from the Kepler space telescope?")
    child.assistant("Kepler discovered over 2,600 confirmed exoplanets, "
                    "showing that planets are common in our galaxy.")
    child.user("Which Kepler planets are in the habitable zone?")
    child.assistant("Notable habitable-zone planets include Kepler-442b, "
                    "Kepler-452b, and Kepler-186f.")

    print(f"  Child commits after work: {len(child.log())}")
    print(f"  Parent commits (unchanged): {len(parent.log())}")

    print("\n  Child context (after research):")
    child.compile().pprint(style="compact")

    # Collapse child findings back into parent
    result = session.collapse(
        child, into=parent,
        content="Kepler mission: 2,600+ confirmed exoplanets. Key habitable-zone "
                "candidates: Kepler-442b, 452b, 186f.",
        auto_commit=True,
    )
    print(f"\n  Collapsed: {result.summary_tokens} tokens into parent")
    print(f"  Parent commits now: {len(parent.log())}")

    print("\n  Parent context after collapse:")
    parent.compile().pprint(style="compact")

    session.close()


if __name__ == "__main__":
    manual()
