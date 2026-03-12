"""Profile-Driven Staged Agent -- workflow profiles guide agent stages.

Demonstrates how WorkflowProfile bundles (config + directives + stage gates)
automate the transition through multi-phase workflows.

Patterns shown:
  1. Profile discovery      -- list_workflow_profiles(), get_workflow_profile() introspection
  2. Profile loading        -- load_profile() applies config + directives
  3. Stage transitions      -- apply_stage() changes config per phase
  4. Directive templates    -- profile templates inject parameterized guidance
  5. Stage gating           -- middleware blocks premature transitions

Demonstrates: t.load_profile(), t.apply_stage(), t.get_config(),
              t.directive(), t.use(), WorkflowProfile, DirectiveTemplate

No LLM required.
"""

from tract import Tract, BlockedError, get_workflow_profile, list_workflow_profiles


def profile_discovery():
    """List available profiles and their stages."""

    print("=" * 60)
    print("1. Profile Discovery")
    print("=" * 60)
    print()

    profiles = list_workflow_profiles()
    print(f"  Built-in profiles: {len(profiles)}")
    print()

    for p in profiles:
        stage_names = list(p.stages.keys())
        print(f"  [{p.name}] {p.description}")
        print(f"    stages:     {stage_names}")
        print(f"    directives: {list(p.directives.keys())}")
        print(f"    config:     temperature={p.config.get('temperature')}, "
              f"strategy={p.config.get('compile_strategy')}")
        print()

    # Verify all three built-in profiles exist
    assert len(profiles) >= 3
    names = {p.name for p in profiles}
    assert "coding" in names
    assert "research" in names
    assert "ecommerce" in names

    # Inspect one profile in detail
    coding = get_workflow_profile("coding")
    assert "design" in coding.stages
    assert "implement" in coding.stages
    assert coding.config["temperature"] == 0.3

    print("  All assertions passed.")
    print()
    print("PASSED")


def profile_loading_and_stages():
    """Load a profile and walk through its stages."""

    print()
    print("=" * 60)
    print("2. Profile Loading and Stage Transitions")
    print("=" * 60)
    print()

    with Tract.open() as t:
        # Load the coding profile -- applies base config + directives
        t.load_profile("coding")
        print(f"  Active profile: {t.active_profile.name}")

        # Base config from the profile
        base_temp = t.get_config("temperature")
        base_strategy = t.get_config("compile_strategy")
        print(f"  Base config: temperature={base_temp}, strategy={base_strategy}")

        assert base_temp == 0.3
        assert base_strategy == "messages"

        # Walk through each stage and show config changes
        coding = get_workflow_profile("coding")
        stages_visited = []

        for stage_name, expected_config in coding.stages.items():
            t.apply_stage(stage_name)
            stages_visited.append(stage_name)

            # Create a branch for this stage's work
            t.branch(stage_name, switch=True)

            temp = t.get_config("temperature")
            strategy = t.get_config("compile_strategy")
            print(f"  Stage '{stage_name}': temperature={temp}, strategy={strategy}")

            assert temp == expected_config["temperature"], (
                f"Stage {stage_name}: expected temp={expected_config['temperature']}, got {temp}"
            )
            assert strategy == expected_config["compile_strategy"], (
                f"Stage {stage_name}: expected strategy="
                f"{expected_config['compile_strategy']}, got {strategy}"
            )

            # Simulate stage work
            t.user(f"Working on {stage_name} phase...")
            t.assistant(f"Completed {stage_name} work.")

            t.switch("main")

        print()
        print(f"  Stages visited: {stages_visited}")
        assert stages_visited == ["design", "implement", "test", "review"]

    print("  All assertions passed.")
    print()
    print("PASSED")


def directives_from_profile():
    """Show how profile directives appear in compiled context."""

    print()
    print("=" * 60)
    print("3. Directives from Profile")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a software engineer.")

        # Before loading profile -- no directives
        ctx_before = t.compile()
        text_before = " ".join((m.content or "") for m in ctx_before.messages)

        t.load_profile("coding")

        # After loading -- directives are in the DAG
        t.user("Write a sorting function.")
        ctx_after = t.compile()
        text_after = " ".join((m.content or "") for m in ctx_after.messages)

        # The coding profile injects "methodology" and "code_quality" directives
        print(f"  Messages before profile: {len(ctx_before.messages)}")
        print(f"  Messages after profile:  {len(ctx_after.messages)}")
        print()

        # Check that directive content appears in compiled context
        has_tdd = "test-driven" in text_after.lower() or "test" in text_after.lower()
        has_quality = "clean" in text_after.lower() or "readable" in text_after.lower()
        print(f"  Contains TDD guidance:     {has_tdd}")
        print(f"  Contains quality guidance:  {has_quality}")

        assert has_tdd, "Coding profile should inject TDD methodology"
        assert has_quality, "Coding profile should inject code quality guidance"

    print("  All assertions passed.")
    print()
    print("PASSED")


def stage_gating_with_middleware():
    """Use middleware to enforce stage ordering."""

    print()
    print("=" * 60)
    print("4. Stage Gating with Middleware")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.load_profile("coding")
        t.system("You are a coding assistant.")

        # Define stage ordering
        stage_order = ["design", "implement", "test", "review"]
        gate_state = {"current_index": 0}

        # Map each stage to its unique temperature for identification
        coding = get_workflow_profile("coding")
        temp_to_stage = {}
        for sname, sconfig in coding.stages.items():
            temp_to_stage[sconfig["temperature"]] = sname

        def stage_gate(ctx):
            """Block config commits that skip stages.

            apply_stage() calls configure(), which creates a ConfigContent commit.
            We inspect the pending ConfigContent.settings to detect stage skips.
            """
            pending = ctx.pending
            # ConfigContent is a Pydantic model with .settings dict
            settings = getattr(pending, "settings", None)
            if settings is None or "temperature" not in settings:
                return  # not a stage config commit

            temp = settings["temperature"]
            requested_stage = temp_to_stage.get(temp)
            if requested_stage and requested_stage in stage_order:
                requested_idx = stage_order.index(requested_stage)
                current_idx = gate_state["current_index"]
                if requested_idx > current_idx + 1:
                    raise BlockedError(
                        ctx.event,
                        f"Cannot skip from '{stage_order[current_idx]}' "
                        f"to '{requested_stage}'. "
                        f"Next allowed: '{stage_order[current_idx + 1]}'",
                    )
                gate_state["current_index"] = requested_idx

        gate_id = t.use("pre_commit", stage_gate)

        # Walk stages in order -- should succeed
        for stage in stage_order:
            t.apply_stage(stage)
            t.user(f"Starting {stage} phase.")
            t.assistant(f"Completed {stage}.")
            print(f"  Stage '{stage}': OK")

        # Reset for skip test
        gate_state["current_index"] = 0
        t.apply_stage("design")

        # Attempt to skip from design -> test (skipping implement)
        blocked = False
        try:
            t.apply_stage("test")
        except BlockedError as e:
            blocked = True
            print(f"  Skip attempt blocked: {e.reasons[0]}")

        assert blocked, "Skipping stages should be blocked"

        t.remove_middleware(gate_id)

    print("  All assertions passed.")
    print()
    print("PASSED")


def research_profile_walkthrough():
    """End-to-end research workflow using the research profile."""

    print()
    print("=" * 60)
    print("5. Research Profile Walkthrough")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.load_profile("research")
        print(f"  Profile: {t.active_profile.name}")
        print(f"  Base temp: {t.get_config('temperature')}")

        research = get_workflow_profile("research")
        stage_data = {}

        for stage_name in research.stages:
            t.apply_stage(stage_name)
            t.branch(stage_name, switch=True)

            temp = t.get_config("temperature")
            strategy = t.get_config("compile_strategy")

            # Simulate stage-specific work
            t.user(f"Execute {stage_name} phase for AI market research.")
            t.assistant(f"[{stage_name}] Analysis complete. Key findings recorded.")

            ctx = t.compile()
            stage_data[stage_name] = {
                "temperature": temp,
                "strategy": strategy,
                "messages": len(ctx.messages),
                "tokens": ctx.token_count,
            }

            print(f"  [{stage_name}] temp={temp}, strategy={strategy}, "
                  f"msgs={len(ctx.messages)}")

            t.switch("main")

        print()

        # Verify stage configs match profile definition
        for stage_name, expected in research.stages.items():
            actual_temp = stage_data[stage_name]["temperature"]
            assert actual_temp == expected["temperature"], (
                f"Stage {stage_name}: temp mismatch"
            )

        # Merge all stage branches into main
        for stage_name in research.stages:
            t.merge(stage_name)

        ctx_final = t.compile()
        print(f"  Final merged context: {len(ctx_final.messages)} messages, "
              f"~{ctx_final.token_count} tokens")

    print("  All assertions passed.")
    print()
    print("PASSED")


def main():
    profile_discovery()
    profile_loading_and_stages()
    directives_from_profile()
    stage_gating_with_middleware()
    research_profile_walkthrough()

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print()
    print("  Pattern                      Tract API Used")
    print("  ---------------------------  -----------------------------------")
    print("  Profile discovery            list_workflow_profiles(), get_workflow_profile()")
    print("  Profile loading              t.load_profile(), t.active_profile")
    print("  Stage transitions            t.apply_stage(), t.get_config()")
    print("  Directive injection          t.directive() via profile.directives")
    print("  Stage gating                 t.use('pre_commit', gate_fn)")
    print()
    print("Done.")


# Alias for pytest discovery
test_profile_staged_agent = main


if __name__ == "__main__":
    main()


# --- See also ---
# Profiles reference:      profiles.py source
# Directive templates:      templates.py source
# Staged workflow (LLM):    agent/05_staged_workflow.py
# Config basics:            config_and_middleware/01_config_basics.py
