"""E-Commerce Optimization Pipeline: research -> landing pages -> ads -> metrics -> optimize

Developer-driven 5-stage pipeline. The developer controls all transitions;
the agent decides what content to produce at each stage based on the
product context and marketing goals.

Stages:
  product_research   -- high temperature (0.8), gather product intel
  landing_pages      -- branch per variant
  ad_copy            -- tagged ad segments
  metrics_analysis   -- evaluate variants
  optimization       -- select winner, iterate

Demonstrates: branching for A/B variants, per-stage config, directives,
              tags, metadata, transitions with handoff, middleware gates

Requires: LLM API key (uses Cerebras provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError, MiddlewareContext
from tract.formatting import pprint_log

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import cerebras as llm
from _logging import StepLogger

MODEL_ID = llm.large


def main() -> None:
    if not llm.api_key:
        print("SKIPPED (no API key -- set CEREBRAS_API_KEY)")
        return

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:

        # =============================================================
        # Stage configuration and transition gates
        # =============================================================

        print("=== Setting Up E-Commerce Pipeline ===\n")

        t.config.set(
            stage="product_research",
            temperature=0.8,
            compile_strategy="full",
            product="Ergonomic Standing Desk Converter",
            target_audience="remote workers and home office professionals",
            price_point="$299",
            min_variants=2,
        )

        t.directive(
            "ecomm-best-practices",
            "Follow direct-response copywriting principles:\n"
            "- Lead with the pain point, not the product\n"
            "- Use specific numbers and social proof\n"
            "- Every headline must pass the 'so what?' test\n"
            "- CTAs should be action-oriented and urgent",
        )

        # Transition gates
        def stage_gate(min_commits, stage_name):
            def gate(ctx: MiddlewareContext):
                if ctx.target != stage_name:
                    return
                count = len(ctx.tract.search.log())
                if count < min_commits:
                    raise BlockedError(
                        "pre_transition",
                        [f"Need >= {min_commits} commits for {stage_name} (have {count})"],
                    )
            return gate

        t.middleware.add("pre_transition", stage_gate(6, "landing_pages"))
        t.middleware.add("pre_transition", stage_gate(3, "ad_copy"))
        t.middleware.add("pre_transition", stage_gate(3, "metrics_analysis"))
        t.middleware.add("pre_transition", stage_gate(3, "optimization"))

        # Register tags
        for tag_name in [
            "research", "competitor", "feature", "pain-point",
            "variant-a", "variant-b", "headline", "cta",
            "ad-search", "ad-social", "metrics", "winner", "optimization",
        ]:
            t.tags.register(tag_name)

        print(f"  Product: {t.config.get('product')}")
        print(f"  Price: {t.config.get('price_point')}")

        t.system(
            "You are an e-commerce growth strategist and copywriter.\n"
            "Use commit() to save every piece of content. Include tags "
            "in your commit calls. Keep responses focused and actionable."
        )

        t.user(
            "Product: Ergonomic Standing Desk Converter -- $299\n"
            "- Converts any desk to sit-stand in 3 seconds\n"
            "- Supports dual monitors (up to 35 lbs)\n"
            "- 15 height positions with pneumatic lift\n"
            "- Built-in cable management and phone holder\n"
            "- 30-day money-back guarantee, 5-year warranty\n"
            "Competitor: VariDesk Pro Plus ($395)\n"
            "Differentiator: 25% cheaper, faster setup, better cable management"
        )

        log = StepLogger()
        _tool_names = ["commit", "tag", "get_config", "status"]

        # =============================================================
        # Stage 1: Product Research
        # =============================================================
        print("\n=== Stage 1: Product Research ===\n")

        result = t.llm.run(
            "Research the competitive landscape and target audience for this "
            "product. What insights should inform our marketing strategy? "
            "Consider the competition, customer pain points, and our key "
            "differentiators.",
            max_steps=6, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Stage 2: Landing Pages (developer drives transition)
        # =============================================================
        print("\n\n=== Stage 2: Landing Pages ===\n")

        t.transition("landing_pages", handoff="summary")
        t.config.set(stage="landing_pages", temperature=0.7)

        result = t.llm.run(
            "Create landing page variants with different value propositions. "
            "Each variant should target a different customer motivation. "
            "Include compelling headlines and calls to action.",
            max_steps=6, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Stage 3: Ad Copy
        # =============================================================
        print("\n\n=== Stage 3: Ad Copy ===\n")

        t.transition("ad_copy", handoff="summary")
        t.config.set(stage="ad_copy", temperature=0.6)

        result = t.llm.run(
            "Write ad copy for search and social channels that aligns with "
            "the landing page strategy. Adapt the messaging for each "
            "platform's format and audience expectations.",
            max_steps=6, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Stage 4: Metrics Analysis
        # =============================================================
        print("\n\n=== Stage 4: Metrics Analysis ===\n")

        t.transition("metrics_analysis", handoff="summary")
        t.config.set(stage="metrics_analysis", temperature=0.3)

        # Seed simulated metrics via metadata
        t.commit(
            {"content_type": "freeform", "payload": {
                "variant_a": {"conversion_rate": "3.2%", "cpc": "$1.45", "ctr": "4.8%"},
                "variant_b": {"conversion_rate": "4.1%", "cpc": "$1.20", "ctr": "5.6%"},
            }},
            message="Simulated A/B test metrics",
            metadata={"source": "simulation"},
            tags=["metrics"],
        )

        result = t.llm.run(
            "Analyze these A/B test results and determine which variant is "
            "performing better and why. What do the conversion rates and "
            "cost metrics tell us about customer preferences?",
            max_steps=4, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Stage 5: Optimization
        # =============================================================
        print("\n\n=== Stage 5: Optimization ===\n")

        t.transition("optimization", handoff="summary")
        t.config.set(stage="optimization", temperature=0.5)

        result = t.llm.run(
            "Based on the metrics analysis, declare a winner and explain "
            "the reasoning. What optimizations should we pursue in the "
            "next iteration to improve performance further?",
            max_steps=6, max_tokens=1024,
            profile="full", tool_names=_tool_names,
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        result.pprint()

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final Pipeline State ===\n")

        print(f"  Stage: {t.config.get('stage')}")
        print(f"  Branch: {t.current_branch}")

        print(f"\n  Branches:")
        for b in t.branches.list():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Tags with content:")
        for entry in t.tags.list():
            if entry["count"] > 0:
                print(f"    {entry['name']:20s} count={entry['count']}")

        print(f"\n  Log (last 12 commits):")
        pprint_log(t.search.log()[-12:])

        print(f"\n  Total commits: {len(t.search.log())}")
        print(f"  Stages completed: 5/5")


if __name__ == "__main__":
    main()


# --- See also ---
# Coding workflow:       workflows/01_coding_assistant.py
# Customer support:      workflows/02_customer_support.py
# Self-routing:          workflows/07_self_routing.py
