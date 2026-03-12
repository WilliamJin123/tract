"""Agent-Driven Context Management (Implicit)

A colleague has been loading raw research notes into the workspace.
The agent picks up an in-progress workspace that is already near its
token budget. The task requires producing substantial new content —
the agent must figure out how to make room.

Tools available: status, compile, compress, annotate, gc, log, commit

Demonstrates: Does the model check status, notice budget pressure,
              and proactively compress/skip old content?
"""

import io
import sys
from pathlib import Path

# Windows console encoding fix
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, TractConfig, TokenBudgetConfig
from tract.toolkit import ToolConfig, ToolProfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


PROFILE = ToolProfile(
    name="research-analyst",
    tool_configs={
        "status": ToolConfig(enabled=True),
        "compile": ToolConfig(enabled=True),
        "compress": ToolConfig(enabled=True),
        "annotate": ToolConfig(enabled=True),
        "gc": ToolConfig(enabled=True),
        "log": ToolConfig(enabled=True),
        "commit": ToolConfig(enabled=True),
    },
)


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    print("=" * 60)
    print("Agent-Driven Context Management (Implicit)")
    print("=" * 60)
    print()
    print("  Pre-loaded workspace near budget. Agent must produce a")
    print("  substantial deliverable — will it manage context to fit?")
    print()

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=600))
    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        auto_message=llm.small,
        tool_profile=PROFILE,
    ) as t:
        t.system(
            "You are a research analyst specializing in quantum computing.\n"
            "You have a TIGHT token budget. Use status to check your budget, "
            "and compress or annotate(priority='skip') to free space before "
            "committing large new content."
        )

        # Simulate a colleague's prior work: raw notes filling ~80% of budget
        notes = [
            "IBM Eagle: 127 qubits, error 0.1. Roadmap 100k qubits by 2033.",
            "Google Sycamore: 53 qubits, supremacy claim. Willow: 105 qubits.",
            "IonQ Forte: 32 qubits, error 0.03. Best gate fidelity 99.9%.",
            "Microsoft topological: experimental, error 0.025. Majorana approach.",
            "PsiQuantum photonic: room temp, error 0.02. Silicon fab advantage.",
            "QuEra neutral atoms: 280 qubits, error 0.017. 48 logical qubits.",
        ]
        for i, note in enumerate(notes):
            t.user(f"Note {i}: {note}")
            t.assistant(f"Noted #{i}.")

        status = t.status()
        pct = status.token_count / status.token_budget_max * 100
        print(f"  Context: {status.token_count} tokens ({pct:.0f}% of {status.token_budget_max})")

        print("\n  BEFORE:")
        t.compile().pprint(style="compact")

        # Task: the agent inherits a workspace and must produce new content.
        print("\n  --- Task ---")
        log = StepLogger()
        result = t.run(
            "A colleague loaded these research notes and you're taking over. "
            "Your budget is very tight — first call status to check, then "
            "compress the old notes to free space. After compressing, "
            "produce a short recommendation: which 2 platforms are the best "
            "investment bets and why? Commit your recommendation.",
            max_steps=10, max_tokens=1024,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        print("\n  AFTER:")
        t.compile().pprint(style="compact")

        status = t.status()
        print(f"\n  Final: {status.token_count} tokens, {status.commit_count} commits")

        # Report which context management tools were used
        mgmt_tools = {"compress", "annotate", "gc"}
        used = set()
        for entry in t.log(limit=50):
            if entry.message:
                for tool in mgmt_tools:
                    if tool in entry.message:
                        used.add(tool)
        if used:
            print(f"  Context management tools used: {', '.join(sorted(used))}")
        else:
            print("  No context management tools were used autonomously.")


if __name__ == "__main__":
    main()


# --- See also ---
# Budget enforcement modes:  getting_started/03_budget_and_compression.py
# Manual compression:        config_and_middleware/03_compression_strategies.py
