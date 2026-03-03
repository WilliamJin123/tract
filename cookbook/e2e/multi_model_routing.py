"""Multi-model routing: different models for different operations.

  PART 1 -- Manual:      configure_operations() + configure_clients() for per-op routing
  PART 2 -- LLM / Agent: Trigger-driven automatic routing with cost optimization

Story: A developer builds a chat app where expensive operations (chat) use
a large capable model, while cheap operations (compression, auto-message)
use a small fast model for cost savings. Shows the full config stack from
per-operation defaults to trigger-driven automatic switching.
"""

import sys
from pathlib import Path

from tract import (
    CompressTrigger,
    Tract,
    TractConfig,
    TokenBudgetConfig,
)
from tract.hooks.compress import PendingCompress
from tract.models.config import LLMConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm

# Route: large model for chat, small model for compression/auto-message
LARGE_MODEL = llm.large
SMALL_MODEL = llm.small


# =====================================================================
# PART 1 -- Manual: Per-Operation Model Configuration
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Per-Operation Model Routing")
    print("=" * 60)
    print()
    print(f"  Large model (chat):        {LARGE_MODEL}")
    print(f"  Small model (compression): {SMALL_MODEL}")

    if not llm.api_key:
        print("\n  SKIPPED (no llm.api_key)")
        return

    # --- Approach 1: configure_operations() ---

    print(f"\n  --- Approach 1: configure_operations() ---\n")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=LARGE_MODEL,  # default model for chat
    ) as t:
        # Route compression to the small model
        t.configure_operations(
            compress=LLMConfig(model=SMALL_MODEL),
            message=LLMConfig(model=SMALL_MODEL),
        )

        t.system("You are a financial advisor helping with retirement planning.")
        t.user("I'm 30 years old with $50k saved. What should my strategy be?")

        # Chat uses the large model (default)
        resp = t.chat("What allocation would you recommend between stocks and bonds?")
        print(f"  Chat response (large model): {resp.text[:80]}...")

        print("\n  Context BEFORE compression:\n")
        t.compile().pprint(style="compact")

        # Check what config was recorded
        log = t.log()
        for ci in log:
            if ci.generation_config and ci.generation_config.get("model"):
                print(f"    Commit {ci.commit_hash[:8]}: model={ci.generation_config['model']}")

        # Manual compression uses the small model (from configure_operations)
        t.user("Add more context about tax-advantaged accounts.")
        t.assistant(
            "Consider maxing out your 401k ($23,000/year), then Roth IRA "
            "($7,000/year). HSA is also tax-advantaged if eligible ($4,150/year). "
            "After tax-advantaged accounts, use a taxable brokerage."
        )

        result = t.compress(
            content="Retirement planning for 30yo with $50k: discussed allocation "
            "strategy and tax-advantaged accounts (401k, Roth IRA, HSA).",
        )
        print(f"\n  Compression (small model): {result.compression_ratio:.1%} ratio")

        print("\n  Context AFTER compression:\n")
        t.compile().pprint(style="compact")

    # --- Approach 2: configure_clients() ---

    print(f"\n  --- Approach 2: configure_clients() ---\n")
    print("  configure_clients() lets you use entirely different LLM")
    print("  endpoints per operation (different providers, not just models).\n")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=LARGE_MODEL,
    ) as t:
        # For this demo, we use the same provider but different models.
        # In production, you might route compression to a cheaper provider.
        t.configure_clients(
            compress={
                "api_key": llm.api_key,
                "base_url": llm.base_url,
                "model": SMALL_MODEL,
            },
            message={
                "api_key": llm.api_key,
                "base_url": llm.base_url,
                "model": SMALL_MODEL,
            },
        )

        t.system("You are a travel planner.")
        resp = t.chat("Plan a 5-day trip to Tokyo on a budget.")
        print(f"  Chat (large model): {resp.text[:80]}...")

        # Show the operation configs
        op_configs = t.operation_configs
        if op_configs:
            print(f"\n  Operation configs:")
            for name, cfg in op_configs.items():
                if cfg and hasattr(cfg, 'model') and cfg.model:
                    print(f"    {name}: model={cfg.model}")

        print(f"\n  Both approaches route expensive ops to large models")
        print(f"  and cheap ops to small models for cost optimization.")


# =====================================================================
# PART 2 -- LLM / Agent: Trigger-Driven Automatic Routing
# =====================================================================

def part2_agent():
    print("\n" + "=" * 60)
    print("PART 2 -- Trigger-Driven Automatic Model Routing")
    print("=" * 60)
    print()
    print("  CompressTrigger fires automatically when budget is high.")
    print("  Compression always routes to the small model via configure_operations().")
    print("  Chat always uses the large model.")

    if not llm.api_key:
        print("\n  SKIPPED (no llm.api_key)")
        return

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=500))

    # Track which model handles each operation
    compression_count = 0

    def count_compress(pending: PendingCompress):
        nonlocal compression_count
        compression_count += 1
        pending.approve()

    with Tract.open(
        config=config,
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=LARGE_MODEL,
    ) as t:
        # Route compression to small model
        t.configure_operations(
            compress=LLMConfig(model=SMALL_MODEL),
            message=LLMConfig(model=SMALL_MODEL),
        )

        # Set up automatic compression trigger
        t.configure_triggers([
            CompressTrigger(
                threshold=0.75,
                summary_content="Auto-compressed: travel planning session.",
            ),
        ])
        t.on("compress", count_compress, name="counter")

        t.system("You are a world travel expert.")

        # Simulate a multi-turn session
        destinations = [
            ("What are the best hiking trails in Patagonia?",
             "Torres del Paine W Trek, Fitz Roy, Perito Moreno."),
            ("How about budget accommodation options?",
             "Refugios along the trail, hostels in El Chalten."),
            ("What gear do I need for Patagonia in March?",
             "Layered clothing, waterproof shell, 30L daypack, trekking poles."),
            ("Any visa requirements for US citizens?",
             "No visa needed for stays under 90 days in Argentina/Chile."),
            ("What about travel insurance recommendations?",
             "World Nomads or SafetyWing for adventure coverage."),
            ("Best time of year for the W Trek?",
             "October-March (southern summer), February-March for fewer crowds."),
            ("How do I get from Buenos Aires to El Calafate?",
             "Domestic flight (3h) or long-distance bus (36h)."),
        ]

        for question, answer in destinations:
            t.user(question)
            t.assistant(answer)

            # compile() evaluates triggers
            ctx = t.compile()
            status = t.status()
            budget_max = status.token_budget_max or 1
            usage = status.token_count / budget_max
            print(f"    {status.token_count}/{budget_max} ({usage:.0%}), "
                  f"auto-compressions: {compression_count}")

        # --- Show the model routing in action ---

        print(f"\n  --- Model Routing Summary ---\n")
        print(f"    Chat model:        {LARGE_MODEL}")
        print(f"    Compression model: {SMALL_MODEL}")
        print(f"    Auto-compressions: {compression_count}")

        # Show config provenance in the log
        print(f"\n  --- Config Provenance ---\n")
        models_by_type: dict[str, set] = {}
        for ci in t.log():
            ct = ci.content_type or "unknown"
            if ci.generation_config and ci.generation_config.get("model"):
                model = ci.generation_config["model"]
                models_by_type.setdefault(ct, set()).add(model)

        for ct, models in sorted(models_by_type.items()):
            print(f"    {ct}: {models}")

        if not models_by_type:
            print("    (No generation configs -- manual commits used small model via configure_operations)")

        print(f"\n  Final compiled context:")
        t.compile().pprint(style="compact")

        final = t.status()
        budget_max = final.token_budget_max or 1
        print(f"\n  Final: {final.token_count}/{budget_max} tokens "
              f"({final.token_count / budget_max:.0%}), "
              f"{final.commit_count} commits")
        print(f"  Cost optimization: chat on large model, everything else on small model.")


def main():
    part1_manual()
    part2_agent()


if __name__ == "__main__":
    main()
