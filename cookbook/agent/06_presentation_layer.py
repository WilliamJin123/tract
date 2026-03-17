"""Presentation Layer: LLM-optimized tool output formatting

The Layer 2 presentation system transforms raw tool results into
LLM-friendly responses without changing execution semantics:

- Metadata footers: timing, branch, status on every result
- Overflow truncation: long outputs get clipped with exploration hints
- Error hints: failures include actionable suggestions from the exception

Layer 1 (executor) stays raw and lossless. Layer 2 (presentation)
optimizes the final string the LLM sees. In the loop, presentation
is enabled by default via LoopConfig(presentation=True).

Demonstrates: ToolPresenter.present_result(), LoopConfig.presentation,
              metadata footers, overflow truncation, error hints,
              PresentationConfig

No LLM required -- this example calls the tools directly.
"""

import sys
from pathlib import Path

from tract import Tract
from tract.toolkit.executor import ToolExecutor
from tract.toolkit.presentation import ToolPresenter, PresentationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    with Tract.open() as t:

        # =============================================================
        # Setup: Raw executor + separate presenter
        # =============================================================

        print("=" * 60)
        print("Layer 2: Presentation Layer")
        print("=" * 60)

        executor = ToolExecutor(t)
        presenter = ToolPresenter(t)

        print("\n  Executor returns raw Layer 1 output")
        print(f"  Presenter applies Layer 2 formatting")
        print(f"  Available tools: {len(executor.available_tools())}")

        # =============================================================
        # Raw vs presented output
        # =============================================================

        print()
        print("-" * 60)
        print("RAW vs PRESENTED: executor output + presenter formatting")
        print("-" * 60)

        # Commit some content first
        t.system("You are a helpful coding assistant.")
        t.user("Help me refactor this module.")

        result = executor.execute("status", {})
        print(f"\n  Raw output:\n{_indent(result.output)}")
        print(f"\n  ^ No footer -- just the raw Layer 1 result.")

        presented = presenter.present_result(result)
        print(f"\n  Presented output:\n{_indent(presented)}")
        print(f"\n  ^ Notice the [ok | Xms | main] footer: status, timing, branch.")

        # =============================================================
        # In the loop: LoopConfig controls presentation
        # =============================================================

        print()
        print("-" * 60)
        print("IN THE LOOP: LoopConfig.presentation controls Layer 2")
        print("-" * 60)

        from tract.loop import LoopConfig

        # Default: presentation=True (enabled with default config)
        cfg_default = LoopConfig()
        print(f"\n  LoopConfig().presentation = {cfg_default.presentation}")
        print("    -> Enabled with default PresentationConfig")

        # Disabled
        cfg_off = LoopConfig(presentation=False)
        print(f"\n  LoopConfig(presentation=False).presentation = {cfg_off.presentation}")
        print("    -> Raw executor output sent to LLM")

        # Custom config
        custom_cfg = PresentationConfig(max_output_lines=10, max_output_bytes=500)
        cfg_custom = LoopConfig(presentation=custom_cfg)
        print(f"\n  LoopConfig(presentation=PresentationConfig(...))")
        print(f"    -> Custom: max_output_lines={custom_cfg.max_output_lines}, "
              f"max_output_bytes={custom_cfg.max_output_bytes}")

        # =============================================================
        # Overflow truncation: large output gets clipped
        # =============================================================

        print()
        print("-" * 60)
        print("OVERFLOW TRUNCATION: large outputs get clipped with hints")
        print("-" * 60)

        # Load many commits to create a long log
        for i in range(30):
            t.commit(
                content={
                    "content_type": "freeform",
                    "payload": {"text": f"Research finding #{i}: " + ("x" * 80)},
                },
                message=f"finding-{i:03d}",
            )

        # Use a tight limit to trigger truncation
        tight_presenter = ToolPresenter(t, PresentationConfig(max_output_lines=10))
        result = executor.execute("log", {"limit": 50})
        presented = tight_presenter.present_result(result)
        print(f"\n  Output:\n{_indent(presented)}")
        print()
        print("  ^ Output clipped at 10 lines with a truncation notice")
        print("    and a tip to use filters or limit params.")

        # =============================================================
        # Error handling with hints
        # =============================================================

        print()
        print("-" * 60)
        print("ERROR WITH HINTS: failures include navigation guidance")
        print("-" * 60)

        # Try to switch to a non-existent branch
        result = executor.execute("switch", {"target": "does-not-exist"})
        print(f"\n  Raw error: {result.error}")
        print(f"  Raw hint:  {result.hint}")

        presented = presenter.present_result(result)
        print(f"\n  Presented error:\n{_indent(presented)}")
        print()
        print("  ^ The presented output includes [hint] from the exception")
        print("    plus the [error | Xms | main] metadata footer.")

        # Try an unknown tool
        result = executor.execute("nonexistent_tool", {})
        print(f"\n  Unknown tool error: {result.error}")
        print(f"  Hint: {result.hint[:80]}...")

        # =============================================================
        # Custom PresentationConfig
        # =============================================================

        print()
        print("-" * 60)
        print("CUSTOM CONFIG: tune presentation behavior")
        print("-" * 60)

        # Tight byte limit for aggressive truncation
        tight_cfg = PresentationConfig(
            max_output_lines=5,
            max_output_bytes=500,
            include_metadata=True,
            include_hints=True,
        )
        tight_presenter = ToolPresenter(t, tight_cfg)

        result = executor.execute("log", {"limit": 20})
        presented = tight_presenter.present_result(result)
        print(f"\n  With tight limits (5 lines, 500 bytes):")
        print(f"  Output:\n{_indent(presented)}")

        # Disable metadata but keep truncation
        no_meta_cfg = PresentationConfig(max_output_lines=5, include_metadata=False)
        no_meta_presenter = ToolPresenter(t, no_meta_cfg)

        result = executor.execute("log", {"limit": 20})
        presented = no_meta_presenter.present_result(result)
        print(f"\n  With metadata disabled:")
        print(f"  Output:\n{_indent(presented)}")
        print()
        print("  ^ Truncation notice but no [ok | ...] footer.")

        # =============================================================
        # PresentationConfig defaults
        # =============================================================

        print()
        print("-" * 60)
        print("DEFAULT CONFIG VALUES")
        print("-" * 60)

        defaults = PresentationConfig()
        print(f"  max_output_lines: {defaults.max_output_lines}")
        print(f"  max_output_bytes: {defaults.max_output_bytes}")
        print(f"  include_metadata: {defaults.include_metadata}")
        print(f"  include_hints:    {defaults.include_hints}")

        # =============================================================
        # Summary
        # =============================================================

        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)
        print("  Layer 1 (executor) returns raw, lossless tool results.")
        print("  Layer 2 (presenter) formats them for LLM consumption:")
        print("  - Metadata footers: [ok/error | timing | branch]")
        print("  - Overflow truncation: clips long output with hints")
        print("  - Error hints: actionable guidance from exceptions")
        print("  - Configurable: tune limits per use case")
        print()
        print("  In the loop:  LoopConfig(presentation=True)          # default")
        print("  Custom:       LoopConfig(presentation=PresentationConfig(...))")
        print("  Disabled:     LoopConfig(presentation=False)")
        print()
        print("  Manual usage: presenter = ToolPresenter(tract)")
        print("                presented = presenter.present_result(result)")


def _indent(text: str, prefix: str = "    ") -> str:
    """Indent each line of text for display."""
    return "\n".join(prefix + line for line in text.split("\n"))


if __name__ == "__main__":
    main()


# --- See also ---
# Discovery profile:     agent/05_discovery_profile.py
# Tool executor basics:  getting_started/03_custom_tools.py
# Implicit discovery:    agent/01_implicit_discovery.py
