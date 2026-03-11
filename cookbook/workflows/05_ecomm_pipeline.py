"""E-Commerce Optimization Pipeline: research -> landing pages -> ads -> metrics -> optimize

A multi-stage pipeline for e-commerce product optimization. Uses branches for
A/B variant management, config for per-stage LLM tuning, tags for segment
tracking, metadata for performance metrics, and compression to keep context
lean between stages.

Stages:
  product_research   -- high temperature (0.8), gather product intel
  landing_pages      -- creative temperature (0.9), branch per variant
  ad_copy            -- moderate temperature (0.7), tagged ad segments
  metrics_analysis   -- low temperature (0.2), evaluate variants
  optimization       -- low temperature (0.3), select winner, iterate

Demonstrates: branching for A/B variants, t.configure() per stage,
              t.directive() for stage instructions, t.tag() for segments,
              t.commit() with metadata for metrics, t.transition() with
              handoff summaries, t.compress() between stages, middleware
              gates for stage progression

Requires: LLM API key (uses Groq provider)
"""

import sys
from pathlib import Path

from tract import Tract, BlockedError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import groq as llm
from _logging import StepLogger

MODEL_ID = llm.small


def main():
    if not llm.api_key:
        print("SKIPPED (no API key -- set GROQ_API_KEY)")
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

        # Start in product research stage
        t.configure(
            stage="product_research",
            temperature=0.8,
            compile_strategy="full",
            product="Ergonomic Standing Desk Converter",
            target_audience="remote workers and home office professionals",
            price_point="$299",
            min_variants=2,
        )

        # Directives: pipeline-wide instructions
        t.directive(
            "pipeline-overview",
            "This is a 5-stage e-commerce optimization pipeline:\n"
            "1. PRODUCT_RESEARCH -- Analyze product, competitors, USPs\n"
            "2. LANDING_PAGES -- Create A/B landing page variants (use branches)\n"
            "3. AD_COPY -- Write targeted ad copy for each variant\n"
            "4. METRICS_ANALYSIS -- Evaluate variant performance\n"
            "5. OPTIMIZATION -- Select winner, plan next iteration\n\n"
            "Use get_config to check the current stage and product details.\n"
            "Use transition to advance between stages.",
        )

        t.directive(
            "ecomm-best-practices",
            "Follow direct-response copywriting principles:\n"
            "- Lead with the pain point, not the product\n"
            "- Use specific numbers and social proof\n"
            "- Every headline must pass the 'so what?' test\n"
            "- CTAs should be action-oriented and urgent",
        )

        # Transition gates: enforce minimum work before advancing
        def landing_page_gate(ctx):
            """Require research before creating landing pages."""
            if ctx.target != "landing_pages":
                return
            count = len(ctx.tract.log())
            if count < 6:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 6 commits for landing pages (have {count})"],
                )

        def ad_copy_gate(ctx):
            """Require landing pages before writing ads."""
            if ctx.target != "ad_copy":
                return
            count = len(ctx.tract.log())
            if count < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for ad copy (have {count})"],
                )

        def metrics_gate(ctx):
            """Require ad copy before analyzing metrics."""
            if ctx.target != "metrics_analysis":
                return
            count = len(ctx.tract.log())
            if count < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for metrics analysis (have {count})"],
                )

        def optimization_gate(ctx):
            """Require metrics analysis before optimization."""
            if ctx.target != "optimization":
                return
            count = len(ctx.tract.log())
            if count < 3:
                raise BlockedError(
                    "pre_transition",
                    [f"Need >= 3 commits for optimization (have {count})"],
                )

        t.use("pre_transition", landing_page_gate)
        t.use("pre_transition", ad_copy_gate)
        t.use("pre_transition", metrics_gate)
        t.use("pre_transition", optimization_gate)

        print(f"  Product: {t.get_config('product')}")
        print(f"  Target audience: {t.get_config('target_audience')}")
        print(f"  Price: {t.get_config('price_point')}")
        print(f"  Initial configs: {t.get_all_configs()}")

        # =============================================================
        # Register tags for e-commerce segments
        # =============================================================

        for tag_name in [
            "research", "competitor", "feature", "pain-point",
            "variant-a", "variant-b", "headline", "cta",
            "social-proof", "urgency", "benefit",
            "ad-search", "ad-social", "ad-retargeting",
            "metrics", "winner", "optimization",
        ]:
            t.register_tag(tag_name)

        print(f"  Registered 17 e-commerce tags")

        # =============================================================
        # System prompt: e-commerce optimization context
        # =============================================================

        t.system(
            "You are an e-commerce growth strategist and copywriter.\n\n"
            "PIPELINE STAGES:\n"
            "1. PRODUCT_RESEARCH -- Analyze the product, identify competitors,\n"
            "   extract unique selling propositions. Tag with 'research',\n"
            "   'competitor', 'feature', and 'pain-point'.\n"
            "2. LANDING_PAGES -- Create 2 landing page variants. Use branch\n"
            "   for each variant ('variant/a' and 'variant/b'). Tag content\n"
            "   with 'variant-a' or 'variant-b' plus 'headline', 'cta', etc.\n"
            "3. AD_COPY -- Write ad copy for each variant. Tag with\n"
            "   'ad-search', 'ad-social', or 'ad-retargeting'.\n"
            "4. METRICS_ANALYSIS -- Evaluate variants with simulated metrics.\n"
            "   Use create_metadata to store conversion rates and costs.\n"
            "5. OPTIMIZATION -- Select the winner and propose improvements.\n\n"
            "Tools: commit, tag, register_tag, branch, switch, transition,\n"
            "create_metadata, get_config, status, compile, log.\n\n"
            "Use get_config to check the current stage and product details.\n"
            "Use transition to advance. Tag EVERY commit appropriately."
        )

        # =============================================================
        # Seed: product brief
        # =============================================================

        t.user(
            "Product: Ergonomic Standing Desk Converter -- $299\n\n"
            "Key details:\n"
            "- Converts any desk to a sit-stand workstation in 3 seconds\n"
            "- Supports dual monitors (up to 35 lbs)\n"
            "- 15 height positions with pneumatic lift\n"
            "- Built-in cable management and phone holder\n"
            "- Ships flat, assembles in under 10 minutes\n"
            "- 30-day money-back guarantee, 5-year warranty\n\n"
            "Target audience: Remote workers spending 8+ hours at desks\n"
            "Key competitor: VariDesk Pro Plus ($395)\n"
            "Main differentiator: 25% cheaper, faster setup, better cable management"
        )

        # =============================================================
        # Run: agent drives through all 5 stages
        # =============================================================

        print("\n=== Running Pipeline (5 stages) ===\n")

        log = StepLogger()

        result = t.run(
            "Execute the full e-commerce optimization pipeline for the "
            "Ergonomic Standing Desk Converter:\n\n"
            "STAGE 1 -- PRODUCT_RESEARCH:\n"
            "- Commit a competitor analysis (compare to VariDesk Pro Plus)\n"
            "- Commit a target audience pain points analysis\n"
            "- Commit a unique selling propositions summary\n"
            "- Tag each with 'research' and relevant sub-tags\n"
            "Then transition to 'landing_pages'.\n\n"
            "STAGE 2 -- LANDING_PAGES:\n"
            "- Create branch 'variant/a' for a pain-point-led variant\n"
            "  (headline focuses on back pain and posture)\n"
            "- Switch back to main, create branch 'variant/b' for a\n"
            "  value-proposition-led variant (headline focuses on price\n"
            "  and productivity gains)\n"
            "- On each branch, commit the headline and CTA\n"
            "- Tag with 'variant-a'/'variant-b' and 'headline'/'cta'\n"
            "Then transition to 'ad_copy'.\n\n"
            "STAGE 3 -- AD_COPY:\n"
            "- Commit a Google search ad for the product\n"
            "- Commit a social media ad (Facebook/Instagram style)\n"
            "- Tag with 'ad-search' or 'ad-social'\n"
            "Then transition to 'metrics_analysis'.\n\n"
            "STAGE 4 -- METRICS_ANALYSIS:\n"
            "- Create metadata entries with simulated metrics for each variant:\n"
            "  Variant A: conversion_rate=3.2%, cpc=$1.45, ctr=4.8%\n"
            "  Variant B: conversion_rate=4.1%, cpc=$1.20, ctr=5.6%\n"
            "- Commit an analysis comparing the two variants\n"
            "Then transition to 'optimization'.\n\n"
            "STAGE 5 -- OPTIMIZATION:\n"
            "- Declare the winner (should be Variant B based on metrics)\n"
            "- Tag the winning analysis with 'winner'\n"
            "- Commit 2-3 specific optimization recommendations for the next\n"
            "  iteration (headline tweaks, CTA improvements, audience targeting)\n"
            "- Tag recommendations with 'optimization'",
            max_steps=25,
            tool_names=[
                "commit", "tag", "register_tag", "branch", "switch",
                "transition", "create_metadata", "get_config", "status",
            ],
            on_step=log.on_step,
            on_tool_result=log.on_tool_result,
        )

        result.pprint()

        # =============================================================
        # Show final state
        # =============================================================

        print(f"\n=== Final Pipeline State ===\n")

        print(f"  Stage: {t.get_config('stage')}")
        print(f"  Branch: {t.current_branch}")

        print(f"\n  Branches:")
        for b in t.list_branches():
            marker = "*" if b.is_current else " "
            print(f"    {marker} {b.name}")

        print(f"\n  Registered tags:")
        for entry in t.list_tags():
            if entry["count"] > 0:
                print(f"    {entry['name']:20s} count={entry['count']}")

        print(f"\n  Log (last 12 commits):")
        for ci in t.log()[-12:]:
            tags_str = f" [{', '.join(ci.tags)}]" if ci.tags else ""
            meta_str = ""
            if ci.metadata:
                # Show key metrics if present
                for key in ("conversion_rate", "cpc", "ctr"):
                    if key in ci.metadata:
                        meta_str += f" {key}={ci.metadata[key]}"
            print(
                f"    {ci.commit_hash[:8]}  {ci.content_type:10s}{tags_str}"
                f"{meta_str}  {ci.message[:40]}"
            )

        print(f"\n  Total commits: {len(t.log())}")


if __name__ == "__main__":
    main()


# --- See also ---
# Coding workflow:       workflows/01_coding_assistant.py
# Research pipeline:     workflows/02_research_pipeline.py
# Customer support:      workflows/03_customer_support.py
# Branching patterns:    agent/06_tangent_isolation.py
