"""Metadata & Persistence reference: tags, priorities, edits, tools, checkpoints, portability, snapshots.

Merged coverage:
  - Tags: auto-classification, explicit, mutable, registry, queries (any/all)
  - Priority annotations: PINNED, SKIP, NORMAL, IMPORTANT
  - Edit-in-place: system(edit=), tool_result(edit=)
  - Tool commits: tool_call/tool_result, is_error, drop_failed_turns
  - Reasoning commits: format, inclusion control
  - Named checkpoints: tag-based milestone markers
  - Cross-session persistence: close/reopen with tract_id
  - Context portability: export_state / load_state round-trip
  - Snapshots: create / list / restore (branch and direct reset)

No LLM required -- all scenarios work without API keys.
"""

import gc
import json
import os
import tempfile
from datetime import datetime, timezone

from tract import Priority, Tract
from tract.formatting import pprint_log


# -- Helpers ---------------------------------------------------------------

def _section(number: int, title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"{number}. {title}")
    print("=" * 60)
    print()


def _cleanup_db(db_path: str) -> None:
    """Force-close SQLite handles and delete the temp DB file."""
    gc.collect()
    try:
        os.unlink(db_path)
    except OSError:
        pass


# -- Scenarios -------------------------------------------------------------

def tags_auto_explicit_mutable() -> None:
    """Tags: auto-classification, explicit at commit time, mutable add/remove."""
    _section(1, "Tags: Auto-Classification, Explicit, and Mutable")

    with Tract.open() as t:
        # Auto-classification: every commit gets tags based on content type/role
        sys_ci = t.system("You are a research assistant.")
        print(f"  system()    auto-tags: {t.get_tags(sys_ci.commit_hash)}")

        usr_ci = t.user("Summarize the attention paper.")
        print(f"  user()      auto-tags: {t.get_tags(usr_ci.commit_hash)}")

        ast_ci = t.assistant("The Transformer replaces recurrence with self-attention.")
        print(f"  assistant() auto-tags: {t.get_tags(ast_ci.commit_hash)}")

        # Explicit tags at commit time (merged with auto-tags)
        obs_ci = t.user("Sparse attention scales linearly.", tags=["observation"])
        print(f"  user(tags=) all tags:  {t.get_tags(obs_ci.commit_hash)}")

        # Mutable annotation tags (add/remove after the fact)
        t.tag(ast_ci.commit_hash, "decision")
        print(f"  after add():    {t.get_tags(ast_ci.commit_hash)}")

        removed = t.untag(ast_ci.commit_hash, "decision")
        print(f"  after remove(): {t.get_tags(ast_ci.commit_hash)}  (removed={removed})")

        # Tag registry: register custom tags with descriptions
        t.register_tag("dead_end", "Agent determined this path was unproductive")
        ci = t.assistant("Approach A failed.")
        t.tag(ci.commit_hash, "dead_end")
        print(f"  registered 'dead_end', applied to {ci.commit_hash[:8]}")

    print("  PASSED")


def tag_queries() -> None:
    """Tag queries: any/all matching, log(tags=) filtering."""
    _section(2, "Tag Queries")

    with Tract.open() as t:
        t.system("You are a data analyst.")
        t.user("Load the CSV.", tags=["observation"])
        t.assistant("Loaded 10k rows.")
        t.assistant("Q4 revenue is strongest.", tags=["decision"])

        # match="any" (OR): commits with reasoning OR observation
        any_hits = t.tags.query(["reasoning", "observation"], match="any")
        print(f"  any-match ('reasoning' OR 'observation'): {len(any_hits)} commits")

        # match="all" (AND): commits with BOTH reasoning AND decision
        all_hits = t.tags.query(["reasoning", "decision"], match="all")
        print(f"  all-match ('reasoning' AND 'decision'):   {len(all_hits)} commits")

        # log(tags=) filters the commit log
        reasoning_log = t.log(tags=["reasoning"])
        print(f"  log(tags=['reasoning']): {len(reasoning_log)} entries")

    print("  PASSED")


def priority_annotations() -> None:
    """Priority annotations: PINNED, SKIP, NORMAL, IMPORTANT."""
    _section(3, "Priority Annotations")

    with Tract.open() as t:
        sys_ci = t.system("You are a research assistant.")
        print(f"  system default priority: PINNED (survives compression)")

        t.user("What reduces inference cost?")
        verbose = t.assistant("1. Quantization 2. Pruning 3. Distillation ...")
        t.user("Focus on quantization.")
        t.assistant("INT4 cuts memory 4x with <1% accuracy loss.")

        # Unpin the system prompt
        t.annotate(sys_ci.commit_hash, Priority.NORMAL, reason="temporary persona")

        # Skip verbose content (hidden from compile, still in history)
        t.annotate(verbose.commit_hash, Priority.SKIP, reason="user narrowed focus")
        ctx = t.compile()
        print(f"  after SKIP: {len(ctx.messages)} messages (verbose hidden)")

        # Reset back to normal
        t.annotate(verbose.commit_hash, Priority.NORMAL)
        ctx = t.compile()
        print(f"  after reset: {len(ctx.messages)} messages (restored)")

        # Convenience filters
        print(f"  skipped: {len(t.search.skipped())} commits")
        print(f"  pinned:  {len(t.search.pinned())} commits")

    print("  PASSED")


def edit_in_place() -> None:
    """Edit-in-place: system(edit=hash) and tool_result(edit=hash)."""
    _section(4, "Edit-in-Place")

    with Tract.open() as t:
        # -- System edit --
        bad = t.system("Return policy: 60 days.")  # mistake
        t.user("What's the return policy?")
        t.assistant("You can return within 60 days.")

        fix = t.system("Return policy: 30 days.", edit=bad.commit_hash)
        ctx = t.compile()
        assert "30 days" in ctx.messages[0].content
        print(f"  system edit: {bad.commit_hash[:8]} -> {fix.commit_hash[:8]}")
        print(f"  compiled system: {ctx.messages[0].content}")

        log = t.log()
        edit_count = sum(1 for e in log if e.operation.value == "edit")
        print(f"  log: {len(log)} entries, {edit_count} edits")

    with Tract.open() as t:
        # -- Surgical tool_result edit --
        t.system("You are a debugging agent.")
        t.user("Find the auth bug.")
        t.assistant("Searching...", metadata={"tool_calls": [
            {"id": "c1", "name": "grep", "arguments": {"pattern": "auth"}},
        ]})
        original = t.tool_result("c1", "grep",
            "login.py:15: def authenticate()\n"
            "login.py:22: if not auth_ldap()\n"
            "session.py:8: from login import authenticate\n"
            "middleware.py:12: from login import authenticate\n"
            "docs/api.md:88: The authenticate() function..."
        )
        before_tokens = t.compile().token_count

        # Trim to only relevant lines (original preserved for audit)
        edited = t.tool_result("c1", "grep",
            "login.py:15: def authenticate()\n"
            "login.py:22: if not auth_ldap()",
            edit=original.commit_hash,
        )
        after_tokens = t.compile().token_count
        print(f"  tool_result edit: {before_tokens} -> {after_tokens} tokens")

    print("  PASSED")


def tool_commits_and_errors() -> None:
    """Tool call/result commits, is_error flag, drop_failed_turns."""
    _section(5, "Tool Commits and Error Handling")

    with Tract.open() as t:
        t.system("You are a deployment agent.")
        t.user("Deploy to staging.")

        # Successful tool call
        t.assistant("Checking...", metadata={"tool_calls": [
            {"id": "c1", "name": "health_check", "arguments": {}},
        ]})
        t.tool_result("c1", "health_check", "Server healthy.")

        # Failed tool call
        t.assistant("Deploying...", metadata={"tool_calls": [
            {"id": "c2", "name": "deploy", "arguments": {"env": "staging"}},
        ]})
        t.tool_result("c2", "deploy", "Error: Connection refused", is_error=True)

        # Retry succeeds
        t.assistant("Retrying...", metadata={"tool_calls": [
            {"id": "c3", "name": "deploy", "arguments": {"env": "backup"}},
        ]})
        t.tool_result("c3", "deploy", "Deployed. Build #1847.")

        # Drop failed turns (SKIP-annotates the error turn)
        drop = t.tools.drop_failed_turns()
        drop.pprint()

        ctx = t.compile()
        print(f"  After dropping failed turns: {len(ctx.messages)} messages")

        # Query tool turns and results
        turns = t.tools.find_turns()
        print(f"  Total tool turns: {len(turns)}")
        deploy_turns = t.tools.find_turns(name="deploy")
        print(f"  'deploy' turns:   {len(deploy_turns)}")
        hc_results = t.tools.find_results(name="health_check")
        print(f"  'health_check' results: {len(hc_results)}")

    print("  PASSED")


def reasoning_commits() -> None:
    """Reasoning commits: format, inclusion control, pinning."""
    _section(6, "Reasoning Commits")

    with Tract.open() as t:
        t.system("You are a math tutor.")
        t.user("What is 17 * 23?")
        r = t.reasoning("17*20=340, 17*3=51, total=391", format="parsed")
        t.assistant("17 x 23 = 391")

        ctx = t.compile()  # reasoning excluded by default
        ctx_with = t.compile(include_reasoning=True)
        print(f"  without reasoning: {ctx.commit_count} messages")
        print(f"  with reasoning:    {ctx_with.commit_count} messages")
        print(f"  reasoning text:    '{r.message}'")

        # Force inclusion via PINNED annotation
        t.annotate(r.commit_hash, Priority.PINNED)
        pinned_ctx = t.compile()
        print(f"  after pinning:     {pinned_ctx.commit_count} messages")

    print("  PASSED")


def tool_provenance() -> None:
    """Tool provenance: set_tools + get_commit_tools."""
    _section(7, "Tool Provenance")

    with Tract.open() as t:
        tools = [{"type": "function", "function": {
            "name": "python_eval", "description": "Eval Python",
            "parameters": {"type": "object", "properties": {
                "expr": {"type": "string"}}, "required": ["expr"]},
        }}]
        t.tools.set(tools)
        ci = t.system("Calculator mode.")
        recorded = t.tools.get_for_commit(ci.commit_hash)
        print(f"  tool provenance: {len(recorded)} tools at {ci.commit_hash[:8]}")

        t.tools.set(None)  # clear tools mid-session
        ci2 = t.user("No tools now.")
        print(f"  after clear: {len(t.tools.get_for_commit(ci2.commit_hash) or [])} tools")

    print("  PASSED")


def named_checkpoints() -> None:
    """Named checkpoints via tag conventions."""
    _section(8, "Named Checkpoints")

    with Tract.open() as t:
        t.register_tag("checkpoint:start", "Beginning of workflow")
        t.register_tag("checkpoint:research-done", "All sources analyzed")

        sys_ci = t.system("You are a technical writer producing an API guide.")
        t.tag(sys_ci.commit_hash, "checkpoint:start")
        print(f"  Checkpoint [start] at {sys_ci.commit_hash[:8]}")

        t.user("We need documentation for the authentication module.")
        t.assistant(
            "Analysis complete. The module supports three auth methods:\n"
            "1. OAuth2 with PKCE flow\n"
            "2. API key authentication\n"
            "3. JWT bearer tokens with refresh"
        )

        research_head = t.head
        t.tag(research_head, "checkpoint:research-done")
        print(f"  Checkpoint [research-done] at {research_head[:8]}")

        t.user("Write the first draft.")
        t.assistant("# Authentication Guide\n\nOAuth2, API Keys, JWT sections...")

        # Query checkpoints from the log
        all_entries = t.log(limit=50)
        checkpoints = []
        for entry in all_entries:
            entry_tags = t.get_tags(entry.commit_hash)
            cp_tags = [tg for tg in entry_tags if tg.startswith("checkpoint:")]
            if cp_tags:
                checkpoints.append((entry, cp_tags))

        print(f"  Found {len(checkpoints)} checkpoints in history:")
        for entry, cp_tags in reversed(checkpoints):
            print(f"    {entry.commit_hash[:8]}  {', '.join(cp_tags)}")

        all_cp_tags = {tg for _, tags in checkpoints for tg in tags}
        assert "checkpoint:start" in all_cp_tags
        assert "checkpoint:research-done" in all_cp_tags
        print("  Verified: all checkpoints queryable from DAG")

    print("  PASSED")


def cross_session_persistence() -> None:
    """Cross-session persistence: close, reopen, continue."""
    _section(9, "Cross-Session Persistence")

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    TRACT_ID = "market-research-2025"

    try:
        # --- Session 1 ---
        print("  SESSION 1: Starting research project")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)
        t.register_tag("checkpoint:session-end", "End of session marker")

        t.system("You are a market research analyst.")
        t.user("Analyze the enterprise LLM market for 2025.")
        t.assistant(
            "Key findings:\n"
            "1. Market size: $47B (up 35% YoY)\n"
            "2. Top players: OpenAI, Anthropic, Google, Meta\n"
            "3. Enterprise adoption: 67% of Fortune 500"
        )
        t.tag(t.head, "checkpoint:session-end")

        session1_head = t.head
        session1_count = len(t.log(limit=50))
        print(f"    HEAD: {session1_head[:8]}, commits: {session1_count}")

        t.close()
        del t
        gc.collect()

        # --- Session 2 ---
        print("  SESSION 2: Reopening and continuing")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        assert t.head == session1_head, "HEAD should persist"
        assert len(t.log(limit=50)) == session1_count, "Commit count should match"
        print(f"    HEAD matches: {t.head[:8]}")

        last_content = str(t.get_content(t.head) or "")
        assert "Market size: $47B" in last_content
        print(f"    Content recovered: {last_content[:50]}...")

        head_tags = t.get_tags(t.head)
        assert "checkpoint:session-end" in head_tags
        print(f"    Tags persisted: {head_tags}")

        # Continue work
        t.user("What about the open-source LLM segment?")
        t.assistant("Llama 3 dominates at 45% of OSS deployments.")
        new_count = len(t.log(limit=50))
        assert new_count > session1_count
        print(f"    New commit count: {new_count} (was {session1_count})")

        t.close()
        del t
        gc.collect()

        # --- Session 3: verification ---
        print("  SESSION 3: Final verification")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)
        assert len(t.log(limit=50)) == new_count
        ctx = t.compile()
        text = " ".join((m.content or "") for m in ctx.messages)
        assert "enterprise LLM market" in text
        assert "Llama 3" in text
        print(f"    All {new_count} commits persisted across 3 sessions")
        t.close()
        del t
        gc.collect()
    finally:
        _cleanup_db(db_path)

    print("  PASSED")


def context_portability() -> None:
    """Context portability: export_state / load_state round-trip."""
    _section(10, "Context Portability (Export / Import)")

    # -- Basic round-trip --
    with Tract.open() as source:
        source.system("You are a data analyst.")
        source.user("Analyze Q4 revenue trends.")
        source.assistant("Revenue grew 12% QoQ driven by enterprise expansion.")
        state = source.persistence.export_state()

    with Tract.open() as target:
        loaded = target.persistence.load_state(state)
        ctx = target.compile()
        text = ctx.to_text()
        assert "data analyst" in text
        assert "revenue" in text.lower()
        print(f"  Basic round-trip: exported {len(state['commits'])} commits, loaded {loaded}")

    # -- JSON file persistence --
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(tmp_fd)
    try:
        with Tract.open() as t:
            t.system("You are an API design reviewer.")
            t.user("Review the /users endpoint schema.")
            t.assistant("Use cursor-based pagination with RFC 7807 error format.")
            state = t.persistence.export_state()

        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        file_size = os.path.getsize(tmp_path)
        print(f"  JSON file: {file_size:,} bytes on disk")

        with open(tmp_path, "r", encoding="utf-8") as f:
            loaded_state = json.load(f)

        with Tract.open() as t2:
            loaded_count = t2.persistence.load_state(loaded_state)
            ctx = t2.compile()
            assert "RFC 7807" in ctx.to_text()
            print(f"  Reloaded from file: {loaded_count} commits, content verified")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # -- Cross-agent transfer --
    with Tract.open() as agent_a:
        agent_a.system("You are a research assistant specializing in ML.")
        agent_a.user("Survey context window management advances.")
        agent_a.assistant("Key papers: Landmark Attention, Ring Attention, StreamingLLM.")
        handoff = agent_a.persistence.export_state()

    with Tract.open() as agent_b:
        agent_b.system("You are a technical writer.")
        imported = agent_b.persistence.load_state(handoff)
        agent_b.user("Write a summary of StreamingLLM.")
        agent_b.assistant("StreamingLLM uses attention sinks for infinite context with fixed memory.")
        ctx = agent_b.compile()
        text = ctx.to_text()
        assert "StreamingLLM" in text
        assert "technical writer" in text
        print(f"  Cross-agent transfer: Agent A -> Agent B ({imported} commits imported)")

    # -- Metadata-only export --
    with Tract.open() as t:
        t.system("You are a financial advisor.")
        t.user("Risks of concentrated stock positions?")
        t.assistant("Single-stock volatility, lack of diversification, liquidity risk.")
        full_state = t.persistence.export_state(include_blobs=True)
        meta_state = t.persistence.export_state(include_blobs=False)

    full_size = len(json.dumps(full_state))
    meta_size = len(json.dumps(meta_state))
    print(f"  Metadata-only: {meta_size:,} bytes vs full {full_size:,} bytes "
          f"({(1 - meta_size / full_size) * 100:.0f}% smaller)")

    # -- Branch export --
    with Tract.open() as t:
        t.system("You are a project planner.")
        t.user("Plan the Q1 roadmap.")
        t.assistant("Q1: Auth, API v2, Dashboard redesign.")

        t.branch("feature/auth")
        t.user("Detail the auth plan.")
        t.assistant("Auth: OAuth2 + PKCE, JWT refresh rotation, SAML 2.0 SSO.")

        feature_state = t.persistence.export_state()
        t.switch("main")
        main_state = t.persistence.export_state()

    assert len(feature_state["commits"]) > len(main_state["commits"])
    assert "SAML 2.0" in json.dumps(feature_state)
    assert "SAML 2.0" not in json.dumps(main_state)
    print(f"  Branch export: main={len(main_state['commits'])} commits, "
          f"feature={len(feature_state['commits'])} commits")

    print("  PASSED")


def snapshots_create_list_restore() -> None:
    """Snapshots: create, list, restore (branch and direct reset)."""
    _section(11, "Snapshots")

    # -- Create and list --
    with Tract.open() as t:
        t.system("You are a database administrator.")

        t.user("Plan the schema migration.")
        t.assistant("Step 1: Add columns. Step 2: Backfill. Step 3: Drop old columns.")
        t.persistence.snapshot("pre-migration")

        t.user("Index strategy?")
        t.assistant("Add composite index on (tenant_id, created_at).")
        t.persistence.snapshot("pre-index-changes")

        t.user("Partition the events table?")
        t.assistant("Yes, range-partition by month on event_date.")
        t.persistence.snapshot("pre-partitioning")

        snaps = t.persistence.list_snapshots()
        assert len(snaps) == 3
        print(f"  Created {len(snaps)} snapshots:")
        for snap in snaps:
            print(f"    [{snap['label']:20s}]  head={snap['head'][:8]}  branch={snap['branch']}")

        # Verify newest-first ordering
        assert snaps[0]["label"] == "pre-partitioning"
        assert snaps[2]["label"] == "pre-migration"
        expected_keys = {"tag", "label", "head", "branch", "timestamp", "hash"}
        for snap in snaps:
            assert expected_keys.issubset(snap.keys())
        print("  Ordering: newest first, all keys present")

    # -- Restore via branch (safe) --
    with Tract.open() as t:
        t.system("You are a security auditor.")
        t.user("Review the auth flow.")
        t.assistant("Auth flow looks solid: OAuth2 + PKCE, bcrypt cost=12.")
        head_at_snap = t.head
        t.persistence.snapshot("before-deep-audit")

        t.user("Audit session management.")
        t.assistant("Sessions use HTTP-only cookies with 30-min expiry.")
        assert t.head != head_at_snap

        restored = t.persistence.restore_snapshot("before-deep-audit")
        assert restored == head_at_snap
        assert t.current_branch == "restore/before-deep-audit"
        print(f"  Restore via branch: HEAD={restored[:8]}, branch={t.current_branch}")

    # -- Restore via direct reset --
    with Tract.open() as t:
        t.system("You are a deployment engineer.")
        t.user("Deploy v3.2 to production.")
        t.assistant("Canary at 5%, monitor 15 min, full rollout.")
        head_at_snap = t.head
        original_branch = t.current_branch
        t.persistence.snapshot("pre-deploy")

        t.user("Canary showing 5% error rate increase!")
        t.assistant("Rolling back to v3.1.")

        restored = t.persistence.restore_snapshot("pre-deploy", create_branch=False)
        assert restored == head_at_snap
        assert t.current_branch == original_branch
        print(f"  Direct reset: HEAD={restored[:8]}, stayed on {t.current_branch}")

    # -- Practical: snapshot before experimental work --
    with Tract.open() as t:
        t.system("You are a technical architect.")
        t.user("Design the caching layer.")
        t.assistant(
            "Caching: L1 in-process LRU, L2 Redis cluster, L3 CDN edge.\n"
            "Invalidation: event-driven via Kafka."
        )
        good_head = t.head
        t.persistence.snapshot("before-experiment")

        # Bad experiment
        t.user("Replace everything with a single Memcached instance.")
        t.assistant("Single Memcached: no failover, cold start 30+ minutes.")

        bad_text = t.compile().to_text()
        assert "no failover" in bad_text

        # Restore
        t.persistence.restore_snapshot("before-experiment")
        restored_text = t.compile().to_text()
        assert "no failover" not in restored_text
        assert "event-driven via Kafka" in restored_text
        print(f"  Experiment undo: bad advice removed, good design intact")

    print("  PASSED")


def session_metadata_recovery() -> None:
    """Session metadata for crash recovery: structured metadata + find()."""
    _section(12, "Session Metadata for Crash Recovery")

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    TRACT_ID = "research-pipeline"

    sources = [
        "Gartner Magic Quadrant 2025",
        "IDC MarketScape Report",
        "Forrester Wave: Enterprise AI",
        "McKinsey AI Index",
        "Stanford HAI Report",
    ]

    try:
        # -- Simulate crash after 3/5 sources --
        t = Tract.open(path=db_path, tract_id=TRACT_ID)
        t.register_tag("workflow:state", "Workflow state checkpoint")
        t.system("You are a research analyst compiling an industry report.")

        for i, source in enumerate(sources[:3]):
            t.user(f"Analyze source: {source}")
            t.assistant(f"Analysis of {source}: key finding for vertical {i + 1}.")
            progress = f"{i + 1}/{len(sources)} sources"
            t.assistant(
                f"Checkpoint: {progress} analyzed",
                metadata={
                    "workflow": "research_pipeline",
                    "stage": "source_analysis",
                    "progress": progress,
                    "completed_sources": sources[:i + 1],
                    "remaining_sources": sources[i + 1:],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                tags=["workflow:state"],
            )
            print(f"    Processed: {source} ({progress})")

        print(f"  CRASH at HEAD [{t.head[:8]}] -- 3/5 sources done")
        t.close()
        del t
        gc.collect()

        # -- Recovery --
        print("  RECOVERY: reopening...")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        state_commits = t.find(metadata_key="workflow", limit=10)
        assert len(state_commits) > 0
        latest = state_commits[0]
        meta = t.get_metadata(latest)
        print(f"    Found state at [{latest.commit_hash[:8]}]: {meta['progress']}")
        print(f"    Remaining: {meta['remaining_sources']}")

        remaining = meta["remaining_sources"]
        done = len(meta["completed_sources"])

        for i, source in enumerate(remaining):
            idx = done + i
            t.user(f"Analyze source: {source}")
            t.assistant(f"Analysis of {source}: key finding for vertical {idx + 1}.")
            progress = f"{idx + 1}/{len(sources)} sources"
            t.assistant(
                f"Checkpoint: {progress} analyzed",
                metadata={
                    "workflow": "research_pipeline",
                    "stage": "source_analysis",
                    "progress": progress,
                    "completed_sources": sources[:idx + 1],
                    "remaining_sources": sources[idx + 1:],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                tags=["workflow:state"],
            )
            print(f"    Resumed: {source} ({progress})")

        final = t.find(metadata_key="workflow", limit=1)[0]
        final_meta = t.get_metadata(final)
        assert final_meta["progress"] == "5/5 sources"
        assert len(final_meta["remaining_sources"]) == 0
        print(f"  Pipeline complete: {final_meta['progress']}")

        t.close()
        del t
        gc.collect()
    finally:
        _cleanup_db(db_path)

    print("  PASSED")


# -- Main ------------------------------------------------------------------

def main() -> None:
    tags_auto_explicit_mutable()       # 1
    tag_queries()                       # 2
    priority_annotations()              # 3
    edit_in_place()                     # 4
    tool_commits_and_errors()           # 5
    reasoning_commits()                 # 6
    tool_provenance()                   # 7
    named_checkpoints()                 # 8
    cross_session_persistence()         # 9
    context_portability()               # 10
    snapshots_create_list_restore()     # 11
    session_metadata_recovery()         # 12

    print(f"\n{'=' * 60}")
    print("All 12 scenarios passed.")
    print("=" * 60)


# Alias for pytest discovery
test_metadata_persistence = main


if __name__ == "__main__":
    main()
