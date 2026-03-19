"""Agent Infrastructure: Profiles, Discovery Tools, and Presentation

Non-LLM infrastructure patterns that support agentic workflows. These are
the building blocks an agent (or its harness) uses to configure stages,
discover capabilities, and format tool output -- all without making a
single LLM call.

Sections:
  1. Profile Discovery & Loading  -- list/get/load profiles, inspect stages
  2. Stage Transitions             -- apply_stage() changes config per phase
  3. Directives from Profiles      -- profile directives appear in compiled context
  4. Stage Gating with Middleware   -- deterministic middleware enforces stage ordering
  5. Research Profile Walkthrough   -- end-to-end research workflow with profile stages
  6. Discovery Meta-Tools           -- 3-tool surface (tract_help/tract_do/tract_inspect)
  7. Error-as-Navigation            -- bad inputs return guidance, not dead ends
  8. Presentation Layer             -- ToolPresenter formatting, truncation, hints
  9. Custom Presentation Config     -- tuning output limits and metadata

Demonstrates: list_workflow_profiles(), get_workflow_profile(), t.templates.load_profile(),
              t.templates.apply_stage(), t.config.get(), t.templates.active_profile,
              t.middleware.add(), WorkflowProfile, DirectiveTemplate,
              ToolExecutor, ToolPresenter, PresentationConfig,
              get_discovery_tools(), LoopConfig.presentation

No LLM required.
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, BlockedError, MiddlewareContext, get_workflow_profile, list_workflow_profiles
from tract.toolkit.executor import ToolExecutor
from tract.toolkit.presentation import ToolPresenter, PresentationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# =====================================================================
# Helpers
# =====================================================================

def _section(num: int, title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {num}. {title}")
    print("=" * 70)
    print()


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent each line of text for display."""
    return "\n".join(prefix + line for line in text.split("\n"))


# =====================================================================
# Section 1: Profile Discovery & Loading
# =====================================================================

def profile_discovery_and_loading():
    """List built-in profiles, inspect stages, load into a tract."""

    _section(1, "Profile Discovery & Loading")

    # --- Discovery: what profiles exist? ---
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

    # --- Loading: apply a profile to a tract ---
    with Tract.open() as t:
        t.templates.load_profile("coding")
        print(f"  Active profile: {t.templates.active_profile.name}")

        base_temp = t.config.get("temperature")
        base_strategy = t.config.get("compile_strategy")
        print(f"  Base config: temperature={base_temp}, strategy={base_strategy}")

        assert base_temp == 0.3
        assert base_strategy == "messages"

    print()
    print("  PASSED")


# =====================================================================
# Section 2: Stage Transitions
# =====================================================================

def stage_transitions():
    """Walk through each stage of the coding profile, verifying config changes."""

    _section(2, "Stage Transitions")

    with Tract.open() as t:
        t.templates.load_profile("coding")

        coding = get_workflow_profile("coding")
        stages_visited = []

        for stage_name, expected_config in coding.stages.items():
            t.templates.apply_stage(stage_name)
            stages_visited.append(stage_name)

            # Create a branch for this stage's work
            t.branch(stage_name, switch=True)

            temp = t.config.get("temperature")
            strategy = t.config.get("compile_strategy")
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

    print()
    print("  PASSED")


# =====================================================================
# Section 3: Directives from Profiles
# =====================================================================

def directives_from_profile():
    """Show how profile directives appear in compiled context."""

    _section(3, "Directives from Profiles")

    with Tract.open() as t:
        t.system("You are a software engineer.")

        # Before loading profile -- no directives
        ctx_before = t.compile()
        text_before = " ".join((m.content or "") for m in ctx_before.messages)

        t.templates.load_profile("coding")

        # After loading -- directives are in the DAG
        t.user("Write a sorting function.")
        ctx_after = t.compile()
        text_after = " ".join((m.content or "") for m in ctx_after.messages)

        # The coding profile injects "methodology" and "code_quality" directives
        print("  Before profile:")
        ctx_before.pprint(style="chat")
        print("  After profile:")
        ctx_after.pprint(style="chat")
        print()

        has_tdd = "test-driven" in text_after.lower() or "test" in text_after.lower()
        has_quality = "clean" in text_after.lower() or "readable" in text_after.lower()
        print(f"  Contains TDD guidance:     {has_tdd}")
        print(f"  Contains quality guidance:  {has_quality}")

        assert has_tdd, "Coding profile should inject TDD methodology"
        assert has_quality, "Coding profile should inject code quality guidance"

    print()
    print("  PASSED")


# =====================================================================
# Section 4: Stage Gating with Middleware
# =====================================================================

def stage_gating_with_middleware():
    """Use deterministic middleware to enforce stage ordering."""

    _section(4, "Stage Gating with Middleware")

    with Tract.open() as t:
        t.templates.load_profile("coding")
        t.system("You are a coding assistant.")

        # Define stage ordering
        stage_order = ["design", "implement", "test", "review"]
        gate_state = {"current_index": 0}

        # Map each stage to its unique temperature for identification
        coding = get_workflow_profile("coding")
        temp_to_stage = {}
        for sname, sconfig in coding.stages.items():
            temp_to_stage[sconfig["temperature"]] = sname

        def stage_gate(ctx: MiddlewareContext):
            """Block config commits that skip stages.

            apply_stage() calls configure(), which creates a ConfigContent commit.
            We inspect the pending ConfigContent.settings to detect stage skips.
            """
            pending = ctx.pending
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

        gate_id = t.middleware.add("pre_commit", stage_gate)

        # Walk stages in order -- should succeed
        for stage in stage_order:
            t.templates.apply_stage(stage)
            t.user(f"Starting {stage} phase.")
            t.assistant(f"Completed {stage}.")
            print(f"  Stage '{stage}': OK")

        # Reset for skip test
        gate_state["current_index"] = 0
        t.templates.apply_stage("design")

        # Attempt to skip from design -> test (skipping implement)
        blocked = False
        try:
            t.templates.apply_stage("test")
        except BlockedError as e:
            blocked = True
            print(f"  Skip attempt blocked: {e.reasons[0]}")

        assert blocked, "Skipping stages should be blocked"

        t.middleware.remove(gate_id)

    print()
    print("  PASSED")


# =====================================================================
# Section 5: Research Profile Walkthrough
# =====================================================================

def research_profile_walkthrough():
    """End-to-end research workflow using the research profile."""

    _section(5, "Research Profile Walkthrough")

    with Tract.open() as t:
        t.templates.load_profile("research")
        print(f"  Profile: {t.templates.active_profile.name}")
        print(f"  Base temp: {t.config.get('temperature')}")

        research = get_workflow_profile("research")
        stage_data = {}

        for stage_name in research.stages:
            t.templates.apply_stage(stage_name)
            t.branch(stage_name, switch=True)

            temp = t.config.get("temperature")
            strategy = t.config.get("compile_strategy")

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
        ctx_final.pprint(style="chat")

    print()
    print("  PASSED")


# =====================================================================
# Section 6: Discovery Meta-Tools
# =====================================================================

def discovery_meta_tools():
    """The discovery profile collapses 29 tools into 3 meta-tools.

    Instead of exposing every tract operation as its own tool (burning
    context tokens on schemas the agent may never use), the discovery
    profile provides:
      - tract_help(topic?)   -- 3-level progressive drill-down
      - tract_do(action, params?)  -- single execution surface
      - tract_inspect(what?) -- unified state dashboard
    """

    _section(6, "Discovery Meta-Tools")

    with Tract.open() as t:
        executor = ToolExecutor(t)
        executor.set_profile("discovery")

        print(f"  Available tools: {executor.available_tools()}")
        print("  (Only 3 tools instead of 29!)")
        print()

        # --- tract_help: 3-level progressive drill-down ---
        print("  LEVEL 1: tract_help() -- high-level overview")
        print("  " + "-" * 50)
        result = executor.execute("tract_help", {})
        print(_indent(result.output))

        print()
        print("  LEVEL 2: tract_help(topic='context') -- actions in a domain")
        print("  " + "-" * 50)
        result = executor.execute("tract_help", {"topic": "context"})
        print(_indent(result.output))

        print()
        print("  LEVEL 3: tract_help(topic='commit') -- full parameter schema")
        print("  " + "-" * 50)
        result = executor.execute("tract_help", {"topic": "commit"})
        print(_indent(result.output))

        # --- tract_do: execute operations ---
        print()
        print("  EXECUTION: tract_do(action='commit', params={...})")
        print("  " + "-" * 50)

        t.system("You are a helpful research assistant.")

        result = executor.execute("tract_do", {
            "action": "commit",
            "params": {
                "content": {
                    "content_type": "freeform",
                    "payload": {
                        "text": "The discovery profile reduces tool count from 29 to 3.",
                    },
                },
                "message": "research note",
            },
        })
        print(f"  Result: {result.output[:120]}...")

        result = executor.execute("tract_do", {"action": "status"})
        print(f"  Status: {result.output}")

        result = executor.execute("tract_do", {
            "action": "branch",
            "params": {"name": "experiment"},
        })
        print(f"  Branch result: {result.output}")

        # --- tract_inspect: unified state views ---
        print()
        print("  INSPECT: tract_inspect() -- dashboard overview")
        print("  " + "-" * 50)
        result = executor.execute("tract_inspect", {})
        print(_indent(result.output))

        print()
        print("  INSPECT: tract_inspect(what='branches')")
        print("  " + "-" * 50)
        result = executor.execute("tract_inspect", {"what": "branches"})
        print(_indent(result.output))

        print()
        print("  INSPECT: tract_inspect(what='history')")
        print("  " + "-" * 50)
        result = executor.execute("tract_inspect", {"what": "history"})
        print(_indent(result.output))

    print()
    print("  PASSED")


# =====================================================================
# Section 7: Error-as-Navigation
# =====================================================================

def error_as_navigation():
    """Bad inputs return lists of valid options, not dead ends.

    This is a critical UX pattern for LLM agents: instead of opaque
    error messages, the system provides guidance that the agent can
    use to self-correct on the next turn.
    """

    _section(7, "Error-as-Navigation")

    with Tract.open() as t:
        executor = ToolExecutor(t)
        executor.set_profile("discovery")

        # --- tract_help: bad topic ---
        print("  tract_help(topic='bogus'):")
        result = executor.execute("tract_help", {"topic": "bogus"})
        print(_indent(result.output))
        print()
        print("  ^ The error itself lists all valid domains and actions.")

        # --- tract_do: bad action ---
        print()
        print("  tract_do(action='nonexistent'):")
        result = executor.execute("tract_do", {"action": "nonexistent"})
        print(f"  {result.output[:200]}")
        print()
        print("  ^ Lists all valid actions, plus a hint to use tract_help.")

        # --- tract_inspect: bad target ---
        print()
        print("  tract_inspect(what='bogus'):")
        result = executor.execute("tract_inspect", {"what": "bogus"})
        print(_indent(result.output))
        print()
        print("  ^ Lists valid inspection targets.")

    print()
    print("  PASSED")


# =====================================================================
# Section 8: Presentation Layer
# =====================================================================

def presentation_layer():
    """Layer 2 formatting transforms raw tool results for LLM consumption.

    Layer 1 (executor) returns raw, lossless results. Layer 2 (presenter)
    adds metadata footers, truncates overflow, and includes error hints --
    all configurable via PresentationConfig.
    """

    _section(8, "Presentation Layer")

    with Tract.open() as t:
        executor = ToolExecutor(t)
        presenter = ToolPresenter(t)

        print("  Executor returns raw Layer 1 output.")
        print("  Presenter applies Layer 2 formatting.")
        print(f"  Available tools: {len(executor.available_tools())}")

        # --- Raw vs presented ---
        print()
        print("  RAW vs PRESENTED:")
        print("  " + "-" * 50)

        t.system("You are a helpful coding assistant.")
        t.user("Help me refactor this module.")

        result = executor.execute("status", {})
        print(f"\n  Raw output:\n{_indent(result.output)}")

        presented = presenter.present_result(result)
        print(f"\n  Presented output:\n{_indent(presented)}")
        print(f"\n  ^ Notice the [ok | Xms | main] footer: status, timing, branch.")

        # --- LoopConfig controls presentation ---
        print()
        print("  LOOP CONFIG:")
        print("  " + "-" * 50)

        from tract.loop import LoopConfig

        cfg_default = LoopConfig()
        print(f"\n  LoopConfig().presentation = {cfg_default.presentation}")
        print("    -> Enabled with default PresentationConfig")

        cfg_off = LoopConfig(presentation=False)
        print(f"\n  LoopConfig(presentation=False).presentation = {cfg_off.presentation}")
        print("    -> Raw executor output sent to LLM")

        custom_cfg = PresentationConfig(max_output_lines=10, max_output_bytes=500)
        cfg_custom = LoopConfig(presentation=custom_cfg)
        print(f"\n  LoopConfig(presentation=PresentationConfig(...))")
        print(f"    -> Custom: max_output_lines={custom_cfg.max_output_lines}, "
              f"max_output_bytes={custom_cfg.max_output_bytes}")

        # --- Overflow truncation ---
        print()
        print("  OVERFLOW TRUNCATION:")
        print("  " + "-" * 50)

        for i in range(30):
            t.commit(
                content={
                    "content_type": "freeform",
                    "payload": {"text": f"Research finding #{i}: " + ("x" * 80)},
                },
                message=f"finding-{i:03d}",
            )

        tight_presenter = ToolPresenter(t, PresentationConfig(max_output_lines=10))
        result = executor.execute("log", {"limit": 50})
        presented = tight_presenter.present_result(result)
        print(f"\n  Output (10-line limit):\n{_indent(presented)}")
        print()
        print("  ^ Output clipped at 10 lines with a truncation notice")
        print("    and a tip to use filters or limit params.")

        # --- Error hints ---
        print()
        print("  ERROR HINTS:")
        print("  " + "-" * 50)

        result = executor.execute("switch", {"target": "does-not-exist"})
        print(f"\n  Raw error: {result.error}")
        print(f"  Raw hint:  {result.hint}")

        presented = presenter.present_result(result)
        print(f"\n  Presented error:\n{_indent(presented)}")
        print()
        print("  ^ Includes [hint] from the exception plus [error | Xms | main] footer.")

    print()
    print("  PASSED")


# =====================================================================
# Section 9: Custom Presentation Config
# =====================================================================

def custom_presentation_config():
    """Tune presentation behavior: line limits, byte limits, metadata toggle."""

    _section(9, "Custom Presentation Config")

    with Tract.open() as t:
        executor = ToolExecutor(t)

        # Seed data
        for i in range(20):
            t.commit(
                content={
                    "content_type": "freeform",
                    "payload": {"text": f"Item #{i}: " + ("data " * 20)},
                },
                message=f"item-{i:03d}",
            )

        # --- Tight limits ---
        tight_cfg = PresentationConfig(
            max_output_lines=5,
            max_output_bytes=500,
            include_metadata=True,
            include_hints=True,
        )
        tight_presenter = ToolPresenter(t, tight_cfg)

        result = executor.execute("log", {"limit": 20})
        presented = tight_presenter.present_result(result)
        print(f"  With tight limits (5 lines, 500 bytes):")
        print(f"  Output:\n{_indent(presented)}")

        # --- Metadata disabled ---
        no_meta_cfg = PresentationConfig(max_output_lines=5, include_metadata=False)
        no_meta_presenter = ToolPresenter(t, no_meta_cfg)

        result = executor.execute("log", {"limit": 20})
        presented = no_meta_presenter.present_result(result)
        print(f"\n  With metadata disabled:")
        print(f"  Output:\n{_indent(presented)}")
        print()
        print("  ^ Truncation notice but no [ok | ...] footer.")

        # --- Default values ---
        print()
        print("  DEFAULT CONFIG VALUES:")
        print("  " + "-" * 50)

        defaults = PresentationConfig()
        print(f"  max_output_lines: {defaults.max_output_lines}")
        print(f"  max_output_bytes: {defaults.max_output_bytes}")
        print(f"  include_metadata: {defaults.include_metadata}")
        print(f"  include_hints:    {defaults.include_hints}")

        # --- Direct access via get_discovery_tools ---
        print()
        print("  ALTERNATIVE: get_discovery_tools() for direct handler access")
        print("  " + "-" * 50)

        from tract.toolkit.discovery import get_discovery_tools

        tools = get_discovery_tools(t)
        for tool in tools:
            print(f"  {tool.name:16s}  {tool.description[:60]}...")

        help_tool = tools[0]  # tract_help
        overview = help_tool.handler()
        print(f"\n  Direct handler call returned {len(overview)} chars")

    print()
    print("  PASSED")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    profile_discovery_and_loading()
    stage_transitions()
    directives_from_profile()
    stage_gating_with_middleware()
    research_profile_walkthrough()
    discovery_meta_tools()
    error_as_navigation()
    presentation_layer()
    custom_presentation_config()

    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print()
    print("  Section  Pattern                        Tract API Used")
    print("  -------  ----------------------------   -----------------------------------")
    print("  1        Profile discovery/loading       list_workflow_profiles(), load_profile()")
    print("  2        Stage transitions               apply_stage(), config.get()")
    print("  3        Directives from profiles        directive() via profile.directives")
    print("  4        Stage gating                    middleware.add('pre_commit', fn)")
    print("  5        Research profile walkthrough    load_profile('research'), merge()")
    print("  6        Discovery meta-tools            ToolExecutor, set_profile('discovery')")
    print("  7        Error-as-navigation             tract_help/do/inspect error returns")
    print("  8        Presentation layer              ToolPresenter, PresentationConfig")
    print("  9        Custom presentation config      PresentationConfig options")
    print()
    print("  All sections passed. No LLM calls were made.")
    print()
    print("Done.")


# Alias for pytest discovery
test_agent_infrastructure = main


if __name__ == "__main__":
    main()


# --- See also ---
# Implicit discovery (LLM):     agentic/01_implicit_discovery.py
# Topology patterns:             reference/07_topology_patterns.py
# Semantic automation (LLM):    agentic/04_semantic_automation.py
