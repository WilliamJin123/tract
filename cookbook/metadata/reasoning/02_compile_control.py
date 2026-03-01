"""Compile Control

Part 2 of Reasoning Traces: compile(include_reasoning=True) promotes
reasoning from SKIP to NORMAL. Explicit annotations always win. No LLM needed.

Demonstrates: compile(include_reasoning=True), annotate() overrides,
              Priority.PINNED, Priority.SKIP
"""

from tract import Priority, Tract


def part2_compile_control():
    print(f"\n{'=' * 60}")
    print("Part 2: COMPILE CONTROL (include_reasoning=)")
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


if __name__ == "__main__":
    part2_compile_control()
