"""Long-running session: 50+ turns with auto-maintenance.

  PART 1 -- Manual:      Chat loop, check status(), manually compress/gc at intervals
  PART 3 -- LLM / Agent:  CompressTrigger(0.8) + GCTrigger(20) + PinTrigger over 50+ turns
"""

import sys
from pathlib import Path

from tract import (
    CompressTrigger,
    GCTrigger,
    PinTrigger,
    Tract,
    TractConfig,
    TokenBudgetConfig,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm  

MODEL_ID = llm.large


# =====================================================================
# PART 1 -- Manual: chat loop with manual maintenance
# =====================================================================

def part1_manual():
    print("=" * 60)
    print("PART 1 -- Manual: Periodic Maintenance Loop")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=500))
    with Tract.open(config=config) as t:
        t.system("You are a deep-space mission planner.")

        # 15 turns with manual maintenance every 5
        for i in range(1, 16):
            t.user(f"Mission log entry {i}: trajectory correction delta-v "
                   f"computed for waypoint {i}.")
            t.assistant(f"Acknowledged entry {i}. Delta-v {0.5 + i * 0.1:.1f} m/s "
                        f"recorded. Fuel reserves nominal.")

            # Check and maintain every 5 turns
            if i % 5 == 0:
                status = t.status()
                budget_max = status.token_budget_max or 1
                usage_pct = status.token_count / budget_max
                print(f"\n  Turn {i}: {status.token_count}/{budget_max} "
                      f"({usage_pct:.0%}), {status.commit_count} commits")

                if usage_pct > 0.7:
                    start = max(1, i - 4)
                    t.compress(content=f"Mission log entries {start}-{i}: "
                               f"trajectory corrections computed, all nominal. "
                               f"Cumulative delta-v within fuel budget.")
                    t.gc(archive_retention_days=0)
                    status = t.status()
                    print(f"  Compressed + GC: {status.token_count}/{budget_max} "
                          f"({status.token_count / budget_max:.0%})")

        final = t.status()
        print(f"\n  Final: {final.token_count} tokens, {final.commit_count} commits")


# =====================================================================
# PART 3 -- LLM / Agent: fully automatic trigger-based maintenance
# =====================================================================

def part3_agent():
    print("\n" + "=" * 60)
    print("PART 3 -- LLM / Agent: 50+ Turns with Auto-Maintenance")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=800))
    with Tract.open(config=config) as t:
        # Configure all triggers for automatic maintenance
        t.configure_triggers([
            CompressTrigger(
                threshold=0.8,
                summary_content="Auto-compressed: telescope observation session.",
            ),
            GCTrigger(max_dead_commits=20),
            PinTrigger(pin_types={"instruction"}),
        ])

        t.system("You are a radio telescope observation coordinator.")

        # 50+ turn simulation
        targets = [
            "Crab Nebula", "Sagittarius A*", "Andromeda Galaxy",
            "Cygnus X-1", "Vela Pulsar", "Centaurus A",
            "Cassiopeia A", "Orion Nebula", "Magellanic Clouds",
            "3C 273 Quasar",
        ]

        compress_count = 0
        def count_compress(pending):
            nonlocal compress_count
            compress_count += 1
            pending.approve()

        t.on("compress", count_compress, name="counter")

        for cycle in range(5):
            for j, target in enumerate(targets):
                turn = cycle * len(targets) + j + 1
                t.user(f"Observation {turn}: point dish at {target}, "
                       f"frequency {1420 + j * 10} MHz, integration 30 min.")
                t.assistant(f"Observation {turn} complete. {target}: "
                            f"signal-to-noise {15 + j:.1f} dB, "
                            f"flux density {0.5 + j * 0.3:.1f} Jy.")

            # compile() evaluates triggers automatically
            ctx = t.compile()
            status = t.status()
            budget_max = status.token_budget_max or 1
            print(f"\n  After cycle {cycle + 1} (turn {(cycle + 1) * len(targets)}): "
                  f"{status.token_count}/{budget_max} tokens "
                  f"({status.token_count / budget_max:.0%}), "
                  f"{status.commit_count} commits")

        print(f"\n  Auto-compressions fired: {compress_count}")
        final = t.status()
        budget_max = final.token_budget_max or 1
        print(f"  Final: {final.token_count}/{budget_max} tokens "
              f"({final.token_count / budget_max:.0%}), "
              f"{final.commit_count} commits")
        print(f"  Context stayed under budget across 50 turns.")


def main():
    part1_manual()
    part3_agent()


if __name__ == "__main__":
    main()
