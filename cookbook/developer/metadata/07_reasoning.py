"""Manual reasoning commits and compile control.

Two aspects of reasoning without LLM involvement:

  Part 1 -- Manual Reasoning Commits:  t.reasoning(), log(), compile(), get_content()
  Part 2 -- Compile Control:            compile(include_reasoning=True), annotate() overrides

Demonstrates: t.reasoning(), ReasoningContent, format=, metadata=,
              compile(include_reasoning=True), annotate(), Priority.PINNED,
              Priority.SKIP, get_content(), get_metadata()
"""

from tract import Priority, Tract


# =====================================================================
# Part 1 -- Manual Reasoning Commits
# =====================================================================

def manual_reasoning():
    """Commit reasoning manually -- no LLM needed."""
    print("=" * 60)
    print("Part 1: MANUAL REASONING COMMITS")
    print("=" * 60)
    print()
    print("  t.reasoning() commits chain-of-thought text.")
    print("  Default priority is SKIP -- excluded from compile().")
    print()

    t = Tract.open()

    # Build a conversation with reasoning between user and assistant
    t.system("You are a math tutor. Show your work.")
    t.user("What is 17 * 23?")

    # Reasoning: the model's internal thinking (committed manually here)
    r_info = t.reasoning(
        "17 * 23 = 17 * 20 + 17 * 3 = 340 + 51 = 391",
        format="parsed",
    )

    t.assistant("17 x 23 = 391")

    # --- Reasoning is in log() ---

    print("  log() shows reasoning commits:\n")
    for entry in reversed(t.log()):
        print(f"    {entry}")

    print(f"\n  Reasoning commit: {r_info.commit_hash[:8]}")
    print(f"  Content type:     {r_info.content_type}")

    # --- But excluded from compile() ---

    ctx = t.compile()
    print(f"\n  compile() -> {ctx.commit_count} messages (reasoning excluded):")
    for msg in ctx.messages:
        print(f"    [{msg.role}] {msg.content[:60]}")

    # --- Format and metadata ---

    print(f"\n  t.reasoning() also accepts format= and metadata=:")
    t2 = Tract.open()
    info = t2.reasoning(
        "Let me think step by step...",
        format="think_tags",
        metadata={"source": "deepseek-r1"},
    )
    content = t2.get_content(info.commit_hash)
    print(f"    format:   {content['format']}")
    meta = t2.get_metadata(info.commit_hash)
    print(f"    metadata: {meta}")
    t2.close()

    t.close()


# =====================================================================
# Part 2 -- Compile Control
# =====================================================================

def compile_control():
    """Manual flag toggling for reasoning visibility."""
    print(f"\n{'=' * 60}")
    print("Part 2: COMPILE CONTROL")
    print("=" * 60)
    print()

    t = Tract.open()
    t.system("You are a helpful assistant.")
    t.user("Explain photosynthesis.")
    t.reasoning(
        "Photosynthesis converts CO2 + H2O into glucose + O2 using "
        "sunlight. The light reactions happen in thylakoids, the Calvin "
        "cycle in the stroma. I should keep this concise."
    )
    t.assistant(
        "Photosynthesis converts CO2 and water into glucose and oxygen "
        "using sunlight energy, primarily in chloroplasts."
    )

    # --- Default: reasoning excluded ---

    ctx_default = t.compile()
    print(f"  compile() default:")
    print(f"    {ctx_default.commit_count} messages, {ctx_default.token_count} tokens")
    roles = [m.role for m in ctx_default.messages]
    print(f"    roles: {roles}")

    # --- include_reasoning=True: reasoning included ---

    ctx_with = t.compile(include_reasoning=True)
    print(f"\n  compile(include_reasoning=True):")
    print(f"    {ctx_with.commit_count} messages, {ctx_with.token_count} tokens")
    roles = [m.role for m in ctx_with.messages]
    print(f"    roles: {roles}")
    extra = ctx_with.token_count - ctx_default.token_count
    print(f"    +{extra} tokens from reasoning content")

    # --- Explicit annotation overrides ---

    print(f"\n  Explicit annotations always win:")

    t2 = Tract.open()
    t2.user("Hello")
    info = t2.reasoning("Important chain of thought")
    t2.annotate(info.commit_hash, Priority.PINNED, reason="keep this")
    t2.assistant("Hi!")

    # PINNED reasoning appears even without include_reasoning
    ctx = t2.compile()
    texts = [m.content for m in ctx.messages]
    has_reasoning = "Important chain of thought" in texts
    print(f"    PINNED reasoning in compile(): {has_reasoning}")

    # Explicit SKIP is respected even with include_reasoning=True
    t3 = Tract.open()
    t3.user("Hello")
    info2 = t3.reasoning("Thinking...")
    t3.annotate(info2.commit_hash, Priority.SKIP, reason="exclude this")
    t3.assistant("Hi!")

    ctx2 = t3.compile(include_reasoning=True)
    texts2 = [m.content for m in ctx2.messages]
    has_reasoning2 = "Thinking..." in texts2
    print(f"    Explicit SKIP with include_reasoning=True: hidden={not has_reasoning2}")

    t.close()
    t2.close()
    t3.close()


# =====================================================================
# Main
# =====================================================================

def main():
    manual_reasoning()
    compile_control()
    print("=" * 60)
    print("Done -- manual reasoning and compile control demonstrated.")
    print("=" * 60)


if __name__ == "__main__":
    main()
