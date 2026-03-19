"""Dead Approach Graveyard: Track abandoned paths to avoid repeating mistakes

Agents hit dead ends. Without a record, future work repeats doomed approaches.
This cookbook uses tract's DAG, tags, and middleware to build a "graveyard" --
structured tombstone commits on dead branches that agents query before retrying.

Sections:
  1. Graveyard Pattern -- tombstones on dead branches, query before retrying
  2. Graveyard-Aware Agent Loop -- pre_commit middleware auto-warns on repeats

Demonstrates: t.register_tag(), t.find(tag=...), t.branches.create/switch(),
              t.commit() with metadata, t.middleware.add(), t.directive(),
              t.llm.run(), t.llm.chat(), pre_commit middleware

Requires: LLM API key (uses Claude Code provider)
"""

import io
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from tract import Tract, MiddlewareContext

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _providers import claude_code as llm
from _logging import StepLogger

MODEL_ID = llm.small


# =====================================================================
# Section 1: Graveyard Pattern -- Record Why Approaches Failed
# =====================================================================

def section_1_graveyard_pattern() -> None:
    print("=" * 60)
    print("  Section 1: Graveyard Pattern")
    print("=" * 60)
    print("  Approach A fails + gets tombstoned. Agent queries graveyard,")
    print("  then succeeds with Approach B.\n")

    log = StepLogger()

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:
        t.register_tag("dead")
        t.register_tag("solution")
        t.system("You are a systems architect designing a distributed rate limiter.")

        t.commit(
            content={"content_type": "freeform", "text":
                     "REQUIREMENT: Rate limiter for multi-region API gateway. "
                     "10k req/sec/region, global limits across 3 regions, <5ms latency."},
            message="requirement: distributed rate limiter",
        )

        # --- Approach A: token bucket with single Redis ---
        print("  --- Approach A: Token Bucket + Single Redis ---\n")
        t.branch("approach/token-bucket-redis", switch=True)
        t.llm.run(
            "Design a token bucket rate limiter backed by a single Redis node. "
            "Consider the multi-region requirement. Commit your design.",
            max_steps=5, max_tokens=1024, tool_names=["commit", "status"],
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )

        # Agent discovers the fatal flaw
        analysis = t.llm.chat(
            "Single Redis = cross-region latency 80-150ms, far above 5ms budget. "
            "Summarize why this is dead in 2 sentences.", max_tokens=200,
        )
        reason = analysis.text or "Cross-region RTT 80-150ms. No failover."

        # Commit tombstone before abandoning
        t.commit(
            content={"content_type": "freeform",
                     "text": f"TOMBSTONE: Token bucket + single Redis.\n\n{reason}"},
            message="DEAD: token_bucket_redis -- cross-region latency too high",
            tags=["dead"],
            metadata={"tombstone": True, "approach": "token_bucket_redis",
                      "reason": reason,
                      "keywords": "redis, token_bucket, single_node, cross_region"},
        )
        print(f"\n  Tombstone committed on: {t.current_branch}")
        t.switch("main")

        # --- Query graveyard before Approach B ---
        print("\n  --- Querying Graveyard ---")
        graveyard: list[str] = []
        for b in t.list_branches():
            if not b.name.startswith("approach/"):
                continue
            for ts in t.find(tag="dead", branch=b.name, limit=10):
                approach = (ts.metadata or {}).get("approach", "?")
                graveyard.append(f"- [{approach}] {ts.message}")
                print(f"    {graveyard[-1]}")

        # --- Approach B: avoids graveyard pitfalls ---
        print("\n  --- Approach B: Sliding Window + Local Counters ---\n")
        t.branch("approach/sliding-window-local", switch=True)
        t.directive("graveyard-context",
                    "GRAVEYARD -- these approaches FAILED:\n"
                    + "\n".join(graveyard)
                    + "\nDo NOT repeat any. Design something different.")

        t.llm.run(
            "Design a rate limiter avoiding graveyard problems. Consider local "
            "per-region counters with async sync. Commit your design.",
            max_steps=5, max_tokens=1024, tool_names=["commit", "status"],
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )

        top = t.log(limit=1)
        if top:
            t.tags.apply(top[0].commit_hash, ["solution"])
        t.switch("main")
        t.merge("approach/sliding-window-local", message="merge: sliding window")

        print(f"\n  Branches: {[b.name for b in t.list_branches()]}")
        print(f"  Context: {t.compile().token_count} tokens")
        t.compile().pprint(style="chat")


# =====================================================================
# Section 2: Graveyard-Aware Agent Loop
# =====================================================================

def section_2_graveyard_middleware() -> None:
    print("\n" + "=" * 60)
    print("  Section 2: Graveyard-Aware Middleware")
    print("=" * 60)
    print("  pre_commit middleware keyword-matches against tombstones.\n")

    log = StepLogger()
    warnings_fired: list[str] = []

    with Tract.open(**llm.tract_kwargs(MODEL_ID), auto_message=llm.small) as t:
        t.register_tag("dead")
        t.system("You are researching database connection pooling strategies.")

        # Seed tombstones on dead branches (simulating prior failed work)
        tombstones = [
            ("dead/pgbouncer-session",
             "TOMBSTONE: PgBouncer session mode pins connections 1:1. "
             "Under >500 clients, PostgreSQL exhausts connections.",
             "DEAD: pgbouncer_session -- no pooling under load",
             "pgbouncer_session_mode",
             "session mode pins connections, no pooling under load",
             "pgbouncer, session, pinned_connections"),
            ("dead/single-pool-global",
             "TOMBSTONE: Single global pool. Noisy neighbor problem -- "
             "one slow query starves other services.",
             "DEAD: single_global_pool -- noisy neighbor",
             "single_global_pool",
             "noisy neighbor, no per-service isolation",
             "global_pool, shared, noisy_neighbor"),
        ]
        for branch, text, msg, approach, reason, kw in tombstones:
            t.branch(branch, switch=True)
            t.commit(content={"content_type": "freeform", "text": text},
                     message=msg, tags=["dead"],
                     metadata={"tombstone": True, "approach": approach,
                               "reason": reason, "keywords": kw})
            t.switch("main")

        # Build keyword index from tombstones
        tomb_index: list[dict] = []
        for b in t.list_branches():
            if not b.name.startswith("dead/"):
                continue
            for ts in t.find(tag="dead", branch=b.name, limit=10):
                meta = ts.metadata or {}
                tomb_index.append({
                    "approach": meta.get("approach", "?"),
                    "reason": meta.get("reason", ""),
                    "keywords": set(meta.get("keywords", "").replace(",", " ").split()),
                })
        print(f"  Graveyard: {len(tomb_index)} tombstone(s)")

        # Middleware: keyword-match new commits against graveyard
        def graveyard_guard(ctx: MiddlewareContext):
            if not ctx.pending:
                return
            text = str(ctx.pending.get("text", "") if isinstance(ctx.pending, dict)
                       else ctx.pending).lower()
            for tomb in tomb_index:
                hits = {kw for kw in tomb["keywords"] if kw.lower() in text}
                if len(hits) >= 2:
                    w = f"Resembles dead '{tomb['approach']}' (matched: {', '.join(sorted(hits))})"
                    warnings_fired.append(w)
                    print(f"\n  >> GRAVEYARD WARNING: {w}")
                    ctx.tract.directive(
                        f"graveyard-warn-{tomb['approach']}",
                        f"WARNING: Known-dead pattern '{tomb['approach']}'.\n"
                        f"Why it failed: {tomb['reason']}\nCHANGE COURSE.")

        t.middleware.add("pre_commit", graveyard_guard)
        print("  Registered graveyard_guard on pre_commit.\n")

        # Phase 1: agent stumbles toward dead approach
        print("  --- Phase 1: Research (triggers warning) ---\n")
        r1 = t.llm.run(
            "Research PostgreSQL connection pooling for microservices. "
            "Explore PgBouncer in session mode. Commit findings.",
            max_steps=5, max_tokens=1024, tool_names=["commit", "status"],
            on_step=log.on_step, on_tool_result=log.on_tool_result,
        )
        print(f"\n  Phase 1: {r1.status}, warnings: {len(warnings_fired)}")

        # Phase 2: course correction after warning
        if warnings_fired:
            print("\n  --- Phase 2: Course correction ---\n")
            r2 = t.llm.run(
                "Graveyard warnings fired. Session mode and global pools are dead. "
                "Research per-service pools with PgBouncer TRANSACTION mode. Commit.",
                max_steps=5, max_tokens=1024, tool_names=["commit", "status"],
                on_step=log.on_step, on_tool_result=log.on_tool_result,
            )
            print(f"\n  Phase 2: {r2.status}")

        print(f"\n  Main: {len(t.log(limit=20))} commits, "
              f"{t.compile().token_count} tokens")
        for i, w in enumerate(warnings_fired, 1):
            print(f"  Warning {i}: {w}")
        t.compile().pprint(style="chat")


# =====================================================================
def main() -> None:
    if not llm.available:
        print("SKIPPED (no LLM provider)")
        return
    print("\n  Dead Approach Graveyard\n")
    section_1_graveyard_pattern()
    section_2_graveyard_middleware()
    print("\n  Key takeaway: structured tombstones + tag-based search give agents")
    print("  institutional memory. Middleware automates the check-before-you-leap")
    print("  pattern so dead approaches are never silently repeated.\n\nDone.")


if __name__ == "__main__":
    main()

# --- See also ---
# Error recovery:       agentic/10_error_recovery.py
# Checkpoint/resume:    agentic/08_checkpoint_resume.py
# Adversarial review:   agentic/05_adversarial_review.py
