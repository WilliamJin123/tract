"""Integration tests combining multiple tract features.

Each test exercises 2-3 features together to verify they work in combination.
Features tested: snapshot, health, sliding_window compress, templates, profiles,
batch, configindex, find, compare, middleware.
"""

from __future__ import annotations

import pytest

from tract import Tract


# ---------------------------------------------------------------------------
# Snapshot + Health
# ---------------------------------------------------------------------------


class TestSnapshotAndHealth:
    def test_snapshot_preserves_health(self):
        """After snapshot+restore, DAG health remains valid."""
        with Tract.open() as t:
            t.system("Test system prompt")
            t.user("Hello")
            t.assistant("Hi there")
            snap_tag = t.snapshot("checkpoint_1")
            # Add more commits
            t.user("More messages")
            t.assistant("More responses")
            # Restore via label match
            t.restore_snapshot("checkpoint_1")
            report = t.health()
            assert report.healthy

    def test_health_after_sliding_window(self):
        """Sliding window compression keeps DAG healthy."""
        with Tract.open() as t:
            t.system("You are helpful.")
            for i in range(15):
                t.user(f"Message {i}")
                t.assistant(f"Response {i}")
            t.compress(
                strategy="sliding_window",
                window_size=5,
                content="[Summary of early messages]",
            )
            report = t.health()
            assert report.healthy
            assert report.commit_count > 0

    def test_snapshot_list_and_restore(self):
        """Multiple snapshots can be listed and restored by label."""
        with Tract.open() as t:
            t.system("Base system")
            t.snapshot("alpha")
            t.user("After alpha")
            t.snapshot("beta")
            t.user("After beta")

            snapshots = t.list_snapshots()
            labels = [s["label"] for s in snapshots]
            assert "alpha" in labels
            assert "beta" in labels

            # Restore to alpha
            t.restore_snapshot("alpha")
            report = t.health()
            assert report.healthy


# ---------------------------------------------------------------------------
# Config + Branch
# ---------------------------------------------------------------------------


class TestConfigAndBranch:
    def test_branch_isolated_configs(self):
        """Configs on different branches stay isolated."""
        with Tract.open() as t:
            t.configure(model="gpt-4")
            t.branch("experiment")
            t.switch("experiment")
            t.configure(model="claude-3")
            assert t.get_config("model") == "claude-3"
            t.switch("main")
            assert t.get_config("model") == "gpt-4"

    def test_config_survives_batch(self):
        """Configs committed in batch are resolvable after."""
        with Tract.open() as t:
            with t.batch():
                t.configure(model="gpt-4", temperature=0.5)
                t.configure(max_tokens=1000)
            assert t.get_config("model") == "gpt-4"
            assert t.get_config("max_tokens") == 1000

    def test_config_precedence_across_branches(self):
        """Branch inherits parent config, override stays branch-local."""
        with Tract.open() as t:
            t.configure(model="gpt-4", temperature=0.7)
            t.branch("feature")
            t.switch("feature")
            # Inherits from main
            assert t.get_config("model") == "gpt-4"
            # Override locally
            t.configure(temperature=0.0)
            assert t.get_config("temperature") == 0.0
            # Main unchanged
            t.switch("main")
            assert t.get_config("temperature") == 0.7


# ---------------------------------------------------------------------------
# Batch + Middleware
# ---------------------------------------------------------------------------


class TestBatchAndMiddleware:
    def test_middleware_fires_for_batch_commits(self):
        """Post-commit middleware fires for commits inside batch."""
        events = []
        with Tract.open() as t:
            t.use("post_commit", lambda ctx: events.append(ctx.event))
            with t.batch():
                t.user("Hello")
                t.assistant("Hi")
            # Middleware should have fired for each commit
            assert len(events) >= 2

    def test_batch_rollback_on_error(self):
        """Batch rolls back all commits on error."""
        with Tract.open() as t:
            t.system("Base")
            initial_count = len(t.log())
            try:
                with t.batch():
                    t.user("Should rollback")
                    raise ValueError("Simulated error")
            except ValueError:
                pass
            assert len(t.log()) == initial_count

    def test_batch_with_config_and_middleware(self):
        """Batch containing configure() calls works with middleware."""
        config_events = []
        with Tract.open() as t:
            t.use("post_commit", lambda ctx: config_events.append(
                ctx.commit.content_type if ctx.commit else None
            ))
            with t.batch():
                t.configure(model="gpt-4")
                t.user("After config")
            # Both commits should have fired post_commit
            assert len(config_events) >= 2
            # Config is resolvable after batch
            assert t.get_config("model") == "gpt-4"


# ---------------------------------------------------------------------------
# Find + Compare
# ---------------------------------------------------------------------------


class TestFindAndCompare:
    def test_find_by_tag(self):
        """find() returns commits matching a specific tag."""
        with Tract.open() as t:
            t.register_tag("critical", description="Critical items")
            t.user("Regular message")
            t.user("Important message", tags=["critical"])
            t.user("Also important", tags=["critical"])
            results = t.find(tag="critical")
            assert len(results) == 2

    def test_find_by_content(self):
        """find() returns commits matching content substring."""
        with Tract.open() as t:
            t.user("The quick brown fox")
            t.user("The lazy dog")
            t.assistant("Response about animals")
            results = t.find(content="quick brown")
            assert len(results) == 1

    def test_compare_branches_with_configs(self):
        """compare() works on branches with different configs."""
        with Tract.open() as t:
            t.system("Base system prompt")
            t.branch("a")
            t.switch("a")
            t.configure(model="gpt-4")
            t.user("Branch A message")
            t.switch("main")
            t.branch("b")
            t.switch("b")
            t.configure(model="claude-3")
            t.user("Branch B message")
            diff = t.compare("a", "b")
            assert diff is not None
            assert diff.commit_a is not None
            assert diff.commit_b is not None

    def test_find_by_content_type(self):
        """find() filters by content_type."""
        with Tract.open() as t:
            t.system("System prompt")
            t.user("User message")
            t.configure(model="gpt-4")
            config_results = t.find(content_type="config")
            assert len(config_results) >= 1

    def test_find_by_metadata(self):
        """find() filters by metadata key/value."""
        with Tract.open() as t:
            t.user("First", metadata={"priority": "high"})
            t.user("Second", metadata={"priority": "low"})
            t.user("Third", metadata={"priority": "high"})
            results = t.find(metadata_key="priority", metadata_value="high")
            assert len(results) == 2


# ---------------------------------------------------------------------------
# Templates + Profiles
# ---------------------------------------------------------------------------


class TestTemplatesAndProfiles:
    def test_apply_template_creates_directive(self):
        """apply_template() creates a directive that compiles into context."""
        with Tract.open() as t:
            t.apply_template("research_protocol", topic="machine learning pipelines")
            compiled = t.compile()
            text = compiled.to_text()
            assert "machine learning pipelines" in text

    def test_profile_loads_config(self):
        """Loading a profile sets configuration values."""
        with Tract.open() as t:
            t.load_profile("coding")
            # The coding profile should set some config
            compiled = t.compile()
            assert compiled is not None
            assert compiled.commit_count > 0

    def test_profile_stage_overrides_config(self):
        """apply_stage() overrides config values for that stage."""
        with Tract.open() as t:
            t.load_profile("coding")
            # Get config after base profile
            base_all = t.get_all_configs()

            t.apply_stage("design")
            design_all = t.get_all_configs()

            # Stage should have set or overridden at least one config
            # (stage config is applied via configure())
            assert len(design_all) >= 1

    def test_template_and_find(self):
        """Templates create commits findable by content substring."""
        with Tract.open() as t:
            t.apply_template(
                "safety_guardrails",
                domain="healthcare",
                threshold="0.8",
            )
            # The template creates an instruction/directive commit
            results = t.find(content="healthcare")
            assert len(results) >= 1


# ---------------------------------------------------------------------------
# Snapshot + Find
# ---------------------------------------------------------------------------


class TestSnapshotAndFind:
    def test_find_across_snapshot_restore(self):
        """find() results reflect current branch state after restore."""
        with Tract.open() as t:
            t.register_tag("v1", description="Version 1")
            t.register_tag("v2", description="Version 2")
            t.user("First message", tags=["v1"])
            t.snapshot("s1")
            t.user("Second message", tags=["v2"])

            # Before restore: v2 is visible
            assert len(t.find(tag="v2")) == 1

            # After restore: we're on a different branch at the snapshot point
            t.restore_snapshot("s1")
            # The restored branch should not have the v2 commit in its ancestry
            v2_results = t.find(tag="v2")
            assert len(v2_results) == 0

    def test_snapshot_preserves_config(self):
        """Config values are restored when restoring a snapshot."""
        with Tract.open() as t:
            t.configure(model="gpt-4", temperature=0.5)
            t.snapshot("with_config")
            # Override config
            t.configure(model="gpt-4o", temperature=0.9)
            assert t.get_config("model") == "gpt-4o"

            # Restore to snapshot
            t.restore_snapshot("with_config")
            # Config should reflect the state at snapshot time
            assert t.get_config("model") == "gpt-4"
            assert t.get_config("temperature") == 0.5


# ---------------------------------------------------------------------------
# Health + Compress + Config
# ---------------------------------------------------------------------------


class TestHealthCompressConfig:
    def test_compress_then_config_then_health(self):
        """After compress + config, health check still passes."""
        with Tract.open() as t:
            t.system("System prompt")
            for i in range(10):
                t.user(f"Q{i}")
                t.assistant(f"A{i}")

            t.compress(
                content="[Summary of Q&A session]",
                strategy="sliding_window",
                window_size=3,
            )
            t.configure(model="gpt-4o", temperature=0.2)
            report = t.health()
            assert report.healthy

    def test_config_find_after_compress(self):
        """find() and get_config() work after compression."""
        with Tract.open() as t:
            t.configure(model="gpt-4")
            for i in range(8):
                t.user(f"Filler message {i}")
                t.assistant(f"Filler response {i}")

            t.compress(content="[Summary of filler messages]")

            # Re-configure after compression (common post-compress pattern)
            t.configure(model="gpt-4o", temperature=0.2)
            t.user("Post-compression message")

            # Config resolves to the post-compression values
            assert t.get_config("model") == "gpt-4o"
            assert t.get_config("temperature") == 0.2

            # find() locates the post-compression config commit
            results = t.find(content_type="config")
            assert len(results) >= 1

            # find() can locate the post-compression user commit
            results = t.find(content="Post-compression")
            assert len(results) == 1


# ---------------------------------------------------------------------------
# Middleware + Config + Find
# ---------------------------------------------------------------------------


class TestMiddlewareConfigFind:
    def test_middleware_tracks_config_changes(self):
        """Middleware sees config changes and find() locates config commits."""
        config_commits_seen = []
        with Tract.open() as t:
            def on_config_commit(ctx):
                if ctx.commit and ctx.commit.content_type == "config":
                    config_commits_seen.append(ctx.commit.commit_hash)

            mid_id = t.use("post_commit", on_config_commit)

            t.configure(model="gpt-4")
            t.configure(temperature=0.5)
            t.user("Message between configs")
            t.configure(max_tokens=2048)

            t.remove_middleware(mid_id)

            # Middleware saw 3 config commits
            assert len(config_commits_seen) == 3

            # find() also locates config commits
            found = t.find(content_type="config")
            assert len(found) == 3

    def test_middleware_with_branch_compare(self):
        """Middleware fires on both branches, compare() shows differences."""
        commit_counts = {"main": 0, "experiment": 0}
        with Tract.open() as t:
            def count_by_branch(ctx):
                commit_counts[ctx.branch] = commit_counts.get(ctx.branch, 0) + 1

            mid_id = t.use("post_commit", count_by_branch)

            t.system("Base")
            t.user("Main message")

            t.branch("experiment")
            t.switch("experiment")
            t.user("Experiment message 1")
            t.user("Experiment message 2")

            t.remove_middleware(mid_id)

            assert commit_counts["main"] >= 2
            assert commit_counts["experiment"] >= 2

            # Compare the two branches
            diff = t.compare("main", "experiment")
            assert diff is not None


# ---------------------------------------------------------------------------
# Batch + Find + Snapshot
# ---------------------------------------------------------------------------


class TestBatchFindSnapshot:
    def test_batch_commits_findable(self):
        """Commits inside batch are findable after batch completes."""
        with Tract.open() as t:
            t.register_tag("batch_test", description="Batch test tag")
            with t.batch():
                t.user("Batch item alpha", tags=["batch_test"])
                t.user("Batch item beta", tags=["batch_test"])
                t.assistant("Batch response")
            results = t.find(tag="batch_test")
            assert len(results) == 2

    def test_snapshot_before_batch_error(self):
        """Snapshot taken before failed batch allows recovery."""
        with Tract.open() as t:
            t.system("Initial state")
            t.user("Important data")
            snap_tag = t.snapshot("pre_batch")
            initial_log_len = len(t.log())

            try:
                with t.batch():
                    t.user("Will be rolled back")
                    t.assistant("Also rolled back")
                    raise RuntimeError("Batch failed")
            except RuntimeError:
                pass

            # After failed batch, log length should be unchanged
            assert len(t.log()) == initial_log_len

            # Snapshot is still valid and restorable
            snapshots = t.list_snapshots()
            assert any(s["label"] == "pre_batch" for s in snapshots)
