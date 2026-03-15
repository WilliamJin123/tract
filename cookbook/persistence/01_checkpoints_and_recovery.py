"""Persistence, Checkpoints, and Session Recovery

Tract's SQLite-backed DAG provides durable, queryable persistence that
survives process restarts.  Unlike JSON/file checkpointing, every commit,
branch, tag, and annotation is ACID-transactional and query-ready the
moment it's written.

Patterns shown:
  1. Named Checkpoints      -- tag milestones, list/inspect them later
  2. Cross-Session Persistence -- close, reopen, continue where you left off
  3. Branch-Based Checkpoints  -- snapshot state before risky operations
  4. Session Metadata for Recovery -- store workflow state as metadata commits
  5. Selective History Recovery    -- find(), query_by_tags(), diff()

Demonstrates: t.tag(), t.log(), t.find(), t.get_tags(), t.query_by_tags(),
              t.branch(), t.switch(), t.reset(), t.diff(), t.compile(),
              t.get_content(), t.get_metadata(), Tract.open(path=, tract_id=),
              t.close()

No LLM required.
"""

import gc
import os
import tempfile
from datetime import datetime, timezone

from tract import Tract
from tract.formatting import pprint_log


def _cleanup_db(db_path: str) -> None:
    """Force-close SQLite connections and delete the temp DB file.

    On Windows, SQLAlchemy may keep file handles open after engine.dispose().
    A gc.collect() nudge releases them so os.unlink() succeeds.
    """
    gc.collect()
    try:
        os.unlink(db_path)
    except OSError:
        pass  # Best-effort cleanup in tests


def main() -> None:
    # =================================================================
    # 1. Named Checkpoints
    # =================================================================
    #
    # Tag commits at milestone moments to create queryable checkpoints.
    # Checkpoint tags follow a "checkpoint:<name>" convention so they
    # are easy to filter from the log.  Unlike file-based snapshots,
    # these are embedded in the DAG and survive indefinitely.

    print("=" * 60)
    print("1. Named Checkpoints")
    print("=" * 60)
    print()

    with Tract.open() as t:
        # Register a tag family for checkpoints
        t.register_tag("checkpoint:start", "Beginning of workflow")
        t.register_tag("checkpoint:research-complete", "All sources analyzed")
        t.register_tag("checkpoint:draft-complete", "First draft written")

        # --- Workflow: research phase ---
        sys_ci = t.system("You are a technical writer producing an API guide.")
        t.user("We need documentation for the authentication module.")

        # Checkpoint: workflow started
        t.tag(sys_ci.commit_hash, "checkpoint:start")
        print(f"  Checkpoint [start] at {sys_ci.commit_hash[:8]}")

        t.assistant("I will review the auth module. Key areas: OAuth2, API keys, JWT.")
        t.user("Here are the source files for the auth module.")
        t.assistant(
            "Analysis complete. The module supports three auth methods:\n"
            "1. OAuth2 with PKCE flow\n"
            "2. API key authentication\n"
            "3. JWT bearer tokens with refresh"
        )

        # Checkpoint: research done
        research_head = t.head
        t.tag(research_head, "checkpoint:research-complete")
        print(f"  Checkpoint [research-complete] at {research_head[:8]}")

        # --- Workflow: drafting phase ---
        t.user("Write the first draft of the authentication guide.")
        t.assistant(
            "# Authentication Guide\n\n"
            "## OAuth2 with PKCE\n"
            "Use the /authorize endpoint with code_challenge...\n\n"
            "## API Keys\n"
            "Generate keys in the dashboard under Settings > API...\n\n"
            "## JWT Bearer Tokens\n"
            "POST to /token with grant_type=client_credentials..."
        )

        # Checkpoint: draft done
        draft_head = t.head
        t.tag(draft_head, "checkpoint:draft-complete")
        print(f"  Checkpoint [draft-complete] at {draft_head[:8]}")

        # --- Query checkpoints from the log ---
        all_entries = t.log(limit=50)
        checkpoints = []
        for entry in all_entries:
            entry_tags = t.get_tags(entry.commit_hash)
            cp_tags = [tg for tg in entry_tags if tg.startswith("checkpoint:")]
            if cp_tags:
                checkpoints.append((entry, cp_tags))

        print(f"\n  Found {len(checkpoints)} checkpoints in history:")
        for entry, cp_tags in reversed(checkpoints):  # oldest first
            print(f"    {entry.commit_hash[:8]}  {', '.join(cp_tags)}")
            print(f"      message: {entry.message or '(none)'}")
            print(f"      tokens:  {entry.token_count}")

        # Verify all three checkpoints were found
        all_cp_tags = {tg for _, tags in checkpoints for tg in tags}
        assert "checkpoint:start" in all_cp_tags
        assert "checkpoint:research-complete" in all_cp_tags
        assert "checkpoint:draft-complete" in all_cp_tags
        print("\n  Verified: all 3 checkpoints queryable from DAG")

    # =================================================================
    # 2. Cross-Session Persistence
    # =================================================================
    #
    # Tract's SQLite backend means you can close a tract, shut down
    # your process, and reopen it later with the same tract_id.
    # Commits, branches, tags, and annotations are all durably stored.
    # This is the foundation for long-running workflows that span
    # multiple sessions.
    #
    # Key: pass the same tract_id= on each Tract.open() call so the
    # new session reconnects to the existing DAG in the database.

    print()
    print("=" * 60)
    print("2. Cross-Session Persistence")
    print("=" * 60)
    print()

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)

    # A stable tract_id is the key to cross-session persistence.
    # In production, derive this from a project name, user ID, or UUID
    # stored in your application's config.
    TRACT_ID = "market-research-2025"

    try:
        # --- Session 1: start a research project ---
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

        # Tag the session endpoint
        t.tag(t.head, "checkpoint:session-end")

        session1_head = t.head
        session1_branch = t.current_branch
        session1_log_count = len(t.log(limit=50))
        print(f"    HEAD:     {session1_head[:8]}")
        print(f"    Branch:   {session1_branch}")
        print(f"    Commits:  {session1_log_count}")

        t.close()
        del t
        gc.collect()
        print("    Session closed.\n")

        # --- Session 2: reopen and continue ---
        print("  SESSION 2: Reopening and continuing")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        # Everything is exactly where we left it
        session2_head = t.head
        session2_branch = t.current_branch
        session2_log_count = len(t.log(limit=50))
        print(f"    HEAD:     {session2_head[:8]}")
        print(f"    Branch:   {session2_branch}")
        print(f"    Commits:  {session2_log_count}")

        assert session2_head == session1_head, "HEAD should persist across sessions"
        assert session2_branch == session1_branch, "Branch should persist"
        assert session2_log_count == session1_log_count, "Commit count should match"
        print("    Verified: HEAD, branch, and commits all persisted")

        # Read back the content from the last commit
        last_content = str(t.get_content(t.head) or "")
        assert "Market size: $47B" in last_content
        print(f"    Last content recovered: {last_content[:50]}...")

        # Tags also persist across sessions
        head_tags = t.get_tags(t.head)
        assert "checkpoint:session-end" in head_tags
        print(f"    Tags on HEAD: {head_tags}")

        # Continue the workflow in session 2
        t.user("What about the open-source LLM segment?")
        t.assistant(
            "Open-source LLM segment:\n"
            "- Llama 3 family dominates at 45% of OSS deployments\n"
            "- Mistral growing fast in European enterprise\n"
            "- Fine-tuning services market: $3.2B"
        )

        new_log_count = len(t.log(limit=50))
        print(f"    New commit count: {new_log_count} (was {session2_log_count})")
        assert new_log_count > session2_log_count
        print("    Verified: continued work in session 2 persists")

        t.close()
        del t
        gc.collect()
        print("    Session closed.\n")

        # --- Session 3: verify everything is still there ---
        print("  SESSION 3: Final verification")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        final_count = len(t.log(limit=50))
        print(f"    Total commits: {final_count}")
        assert final_count == new_log_count
        print("    Verified: all work from sessions 1 and 2 persists")

        # Compile the full context to prove nothing was lost
        ctx = t.compile()
        compiled_text = " ".join((m.content or "") for m in ctx.messages)
        assert "enterprise LLM market" in compiled_text
        assert "Open-source" in compiled_text
        ctx.pprint(style="compact")

        t.close()
        del t
        gc.collect()
    finally:
        _cleanup_db(db_path)

    # =================================================================
    # 3. Branch-Based Checkpoints
    # =================================================================
    #
    # Before risky operations (experimental prompts, large edits,
    # aggressive compression), snapshot the current state onto a
    # checkpoint branch.  The checkpoint is immutable -- if the risky
    # work goes wrong, you can switch back or reset to it.

    print()
    print("=" * 60)
    print("3. Branch-Based Checkpoints")
    print("=" * 60)
    print()

    with Tract.open() as t:
        t.system("You are a data pipeline engineer.")
        t.user("Design an ETL pipeline for our clickstream data.")
        t.assistant(
            "Proposed pipeline:\n"
            "1. Ingest: Kafka -> S3 raw zone\n"
            "2. Transform: Spark job with dedup + sessionization\n"
            "3. Load: Write to Redshift star schema\n"
            "4. Validate: Great Expectations suite"
        )

        # Snapshot the good state before experimenting
        t.branch("checkpoint/pre-experiment", switch=False)
        main_head_before = t.head
        print(f"  Checkpoint branch created at [{main_head_before[:8]}]")
        print(f"  Current branch: {t.current_branch}")

        # --- Risky experiment: try a completely different architecture ---
        t.user("What if we used a real-time streaming architecture instead?")
        t.assistant(
            "Real-time approach:\n"
            "1. Kafka Streams for sessionization\n"
            "2. Flink for windowed aggregations\n"
            "3. Direct writes to ClickHouse\n"
            "WARNING: This requires 3x infrastructure budget."
        )
        t.user("That budget is too high. The streaming approach is a dead end.")

        risky_head = t.head
        experiment_count = len(t.log(limit=50))
        print(f"  After experiment: {experiment_count} commits, HEAD={risky_head[:8]}")
        print("  Experiment failed -- rolling back to checkpoint")

        # Roll back: reset to the checkpoint
        t.reset(main_head_before)
        assert t.head == main_head_before
        print(f"  Reset to [{main_head_before[:8]}] -- experiment undone")

        # The checkpoint branch is still intact
        branches = [b.name for b in t.list_branches()]
        assert "checkpoint/pre-experiment" in branches
        print(f"  Branches available: {branches}")

        # Continue with a better approach on main
        t.user("Let's optimize the existing batch pipeline instead.")
        t.assistant(
            "Optimized batch pipeline:\n"
            "1. Partition S3 by event_date for faster reads\n"
            "2. Add Spark caching for hot dimension tables\n"
            "3. Incremental loads instead of full refresh\n"
            "Result: 60% faster, same infrastructure cost."
        )

        # Verify the compiled context is clean
        ctx = t.compile()
        compiled_text = " ".join((m.content or "") for m in ctx.messages)
        assert "3x infrastructure" not in compiled_text, "Experiment should not be in context"
        assert "Incremental loads" in compiled_text, "Recovery work should be present"
        ctx.pprint(style="compact")
        print("  Verified: branch checkpoint preserved clean rollback point")

    # =================================================================
    # 4. Session Metadata for Recovery
    # =================================================================
    #
    # Store structured workflow state as metadata on commits.  When
    # recovering from a crash, read the metadata to understand exactly
    # where the workflow was: which stage, which step, what progress.
    # Use find(metadata_key=) to locate these state commits on reopen.

    print()
    print("=" * 60)
    print("4. Session Metadata for Recovery")
    print("=" * 60)
    print()

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    TRACT_ID = "research-pipeline"

    try:
        # --- Simulate a multi-stage pipeline that crashes mid-way ---
        print("  Simulating a 5-source research pipeline...")

        sources = [
            "Gartner Magic Quadrant 2025",
            "IDC MarketScape Report",
            "Forrester Wave: Enterprise AI",
            "McKinsey AI Index",
            "Stanford HAI Report",
        ]

        t = Tract.open(path=db_path, tract_id=TRACT_ID)
        t.register_tag("workflow:state", "Workflow state checkpoint")

        t.system("You are a research analyst compiling an industry report.")

        for i, source in enumerate(sources[:3]):
            # Process each source
            t.user(f"Analyze source: {source}")
            t.assistant(
                f"Analysis of {source}: Key finding -- "
                f"enterprise AI adoption accelerating in vertical {i + 1}."
            )

            # Record workflow state as metadata on a checkpoint commit.
            # The metadata dict stores structured recovery information;
            # the tag makes it easy to find later.
            progress = f"{i + 1}/{len(sources)} sources"
            t.assistant(
                f"Checkpoint: {progress} analyzed",
                metadata={
                    "workflow": "research_pipeline",
                    "stage": "source_analysis",
                    "progress": progress,
                    "completed_sources": sources[: i + 1],
                    "remaining_sources": sources[i + 1 :],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                tags=["workflow:state"],
            )
            print(f"    Processed: {source} ({progress})")

        # Simulate crash after 3 sources
        crash_head = t.head
        print(f"\n  CRASH at HEAD [{crash_head[:8]}] -- 3/5 sources done")
        t.close()
        del t
        gc.collect()

        # --- Recovery: reopen and figure out where we were ---
        print("\n  RECOVERY: Reopening database...")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        # Find the most recent workflow state commit using metadata_key
        state_commits = t.find(metadata_key="workflow", limit=10)
        assert len(state_commits) > 0, "Should find workflow state commits"

        latest_state = state_commits[0]  # newest first
        state_meta = t.get_metadata(latest_state)
        print(f"    Found workflow state at [{latest_state.commit_hash[:8]}]")
        print(f"    Stage:     {state_meta['stage']}")
        print(f"    Progress:  {state_meta['progress']}")
        print(f"    Completed: {state_meta['completed_sources']}")
        print(f"    Remaining: {state_meta['remaining_sources']}")

        # Resume from where we left off
        remaining = state_meta["remaining_sources"]
        completed_count = len(state_meta["completed_sources"])

        for i, source in enumerate(remaining):
            idx = completed_count + i
            t.user(f"Analyze source: {source}")
            t.assistant(
                f"Analysis of {source}: Key finding -- "
                f"enterprise AI adoption accelerating in vertical {idx + 1}."
            )

            progress = f"{idx + 1}/{len(sources)} sources"
            t.assistant(
                f"Checkpoint: {progress} analyzed",
                metadata={
                    "workflow": "research_pipeline",
                    "stage": "source_analysis",
                    "progress": progress,
                    "completed_sources": sources[: idx + 1],
                    "remaining_sources": sources[idx + 1 :],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                tags=["workflow:state"],
            )
            print(f"    Resumed: {source} ({progress})")

        # Final state check
        final_state = t.find(metadata_key="workflow", limit=1)[0]
        final_meta = t.get_metadata(final_state)
        assert final_meta["progress"] == "5/5 sources"
        assert len(final_meta["remaining_sources"]) == 0
        print(f"\n  Pipeline complete: {final_meta['progress']}")
        print("  Verified: crash recovery resumed exactly where it left off")

        t.close()
        del t
        gc.collect()
    finally:
        _cleanup_db(db_path)

    # =================================================================
    # 5. Selective History Recovery
    # =================================================================
    #
    # After reopening a tract, use find(), query_by_tags(), and diff()
    # to navigate the history and understand what happened.  This is
    # the "queryable" advantage over flat-file checkpoints: you can
    # search by content, filter by tags, and diff between any two
    # points in history.

    print()
    print("=" * 60)
    print("5. Selective History Recovery")
    print("=" * 60)
    print()

    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    TRACT_ID = "security-audit"

    try:
        # --- Build a realistic multi-branch history ---
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        t.register_tag("decision", "Key decision point")
        t.register_tag("finding", "Research finding")
        t.register_tag("risk", "Identified risk")

        t.system("You are a security auditor reviewing a web application.")

        # Phase 1: Initial findings
        t.user("Review the authentication system.")
        t.assistant(
            "Finding: Password hashing uses bcrypt with cost=12. Good.",
            tags=["finding"],
        )

        t.user("Review the session management.")
        t.assistant(
            "Finding: Sessions use JWT with 24h expiry. "
            "Risk: No token rotation on privilege escalation.",
            tags=["finding", "risk"],
        )

        # Decision checkpoint
        decision1 = t.assistant(
            "Decision: Recommend implementing token rotation for "
            "sensitive operations (payments, role changes).",
            tags=["decision"],
        )
        decision1_hash = decision1.commit_hash

        # Phase 2: deeper audit on a branch
        t.branch("deep-audit")
        t.user("Perform deeper analysis of the JWT implementation.")
        t.assistant(
            "Deep audit: JWT signing uses RS256 with 2048-bit keys. "
            "Key rotation happens quarterly. No issues found.",
            tags=["finding"],
        )
        t.assistant(
            "CRITICAL: Found that refresh tokens are stored in "
            "localStorage, vulnerable to XSS attacks.",
            tags=["finding", "risk"],
        )

        # Switch back to main and add more work
        t.switch("main")
        t.user("Review the API rate limiting.")
        t.assistant(
            "Finding: Rate limiting at 100 req/min per API key. "
            "No per-endpoint limits. Low risk.",
            tags=["finding"],
        )

        t.close()
        del t
        gc.collect()

        # --- Reopen and query the history ---
        print("  Reopened tract -- querying persisted history:\n")
        t = Tract.open(path=db_path, tract_id=TRACT_ID)

        # a) Find all commits tagged as risks (on main branch)
        risks = t.query_by_tags(["risk"], match="any")
        print(f"  a) Risks found on main: {len(risks)}")
        for r in risks:
            content = str(t.get_content(r) or "")
            preview = (content[:70] + "...") if len(content) > 70 else content
            print(f"     [{r.commit_hash[:8]}] {preview}")

        # b) Find all decision points
        decisions = t.query_by_tags(["decision"], match="any")
        print(f"\n  b) Decisions found: {len(decisions)}")
        for d in decisions:
            content = str(t.get_content(d) or "")
            preview = (content[:70] + "...") if len(content) > 70 else content
            print(f"     [{d.commit_hash[:8]}] {preview}")

        # c) Search by content substring across the current branch
        jwt_hits = t.find(content="JWT")
        print(f"\n  c) Commits mentioning 'JWT': {len(jwt_hits)}")
        for hit in jwt_hits:
            print(f"     [{hit.commit_hash[:8]}] {hit.content_type}: "
                  f"{hit.message or '(no message)'}")

        # d) Compare branches: check what the deep-audit branch has
        main_log = t.log(limit=50)
        t.switch("deep-audit")
        deep_log = t.log(limit=50)
        t.switch("main")
        print(f"\n  d) Branch comparison:")
        print(f"     main:       {len(main_log)} commits")
        print(f"     deep-audit: {len(deep_log)} commits")

        # e) Diff between the decision point and current HEAD
        diff_result = t.diff(decision1_hash, t.head)
        stat = diff_result.stat
        print(f"\n  e) Diff from decision [{decision1_hash[:8]}] to HEAD [{t.head[:8]}]:")
        print(f"     Added messages:   {stat.messages_added}")
        print(f"     Removed messages: {stat.messages_removed}")
        print(f"     Token delta:      {stat.total_token_delta:+d}")

        # Verify everything is queryable
        assert len(risks) >= 1, "Should find at least one risk"
        assert len(decisions) >= 1, "Should find at least one decision"
        assert len(jwt_hits) >= 1, "Should find JWT mentions"
        assert stat.messages_added > 0, "Should have added messages since decision"
        print("\n  Verified: full history queryable after reopen")

        t.close()
        del t
        gc.collect()
    finally:
        _cleanup_db(db_path)

    # =================================================================
    # Summary
    # =================================================================

    print()
    print("=" * 60)
    print("Summary: Why DAG Persistence Beats File Checkpoints")
    print("=" * 60)
    print()
    print("  Pattern                    Tract Primitives Used")
    print("  -------------------------  ----------------------------------")
    print("  Named Checkpoints          tag() + get_tags() + log()")
    print("  Cross-Session Persistence  Tract.open(path=, tract_id=)")
    print("  Branch-Based Checkpoints   branch(switch=False) + reset()")
    print("  Session Metadata Recovery  assistant(metadata=) + find()")
    print("  Selective History Recovery  find() + query_by_tags() + diff()")
    print()
    print("  Advantages over file/JSON checkpoints:")
    print("  - ACID transactions: no partial writes or corruption")
    print("  - Queryable: search by content, tags, metadata, content_type")
    print("  - Branchable: checkpoint branches are zero-cost snapshots")
    print("  - Auditable: full history preserved even after rollback")
    print("  - Portable: single .db file contains everything")
    print()
    print("Done.")


# Alias for pytest discovery
test_checkpoints_and_recovery = main


if __name__ == "__main__":
    main()
