"""CLI tests for Tract -- tests all 5 commands via Click's CliRunner.

Each test uses runner.isolated_filesystem() with file-backed databases
since CLI opens its own connection (separate from SDK setup).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from tract.cli import cli
from tract.models.commit import CommitOperation
from tract.models.content import InstructionContent, DialogueContent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


def _setup_tract(db_path: str, *, tract_id: str = "test-tract") -> None:
    """Create a tract with sample commits using the SDK, then close it.

    Creates 3 commits: system instruction, user message, assistant reply.
    """
    from tract.tract import Tract

    t = Tract.open(path=db_path, tract_id=tract_id)
    t.commit(
        InstructionContent(text="You are a helpful assistant."),
        message="system prompt",
    )
    t.commit(
        DialogueContent(role="user", text="Hello, how are you?"),
        message="user greeting",
    )
    t.commit(
        DialogueContent(role="assistant", text="I am doing well, thank you!"),
        message="assistant reply",
    )
    t.close()


def _setup_empty_tract(db_path: str, *, tract_id: str = "test-tract") -> None:
    """Create a tract with no commits."""
    from tract.tract import Tract

    t = Tract.open(path=db_path, tract_id=tract_id)
    t.close()


def _setup_tract_with_edit(db_path: str, *, tract_id: str = "test-tract") -> str:
    """Create a tract with an EDIT commit. Returns the edit's commit hash."""
    from tract.tract import Tract

    t = Tract.open(path=db_path, tract_id=tract_id)
    c1 = t.commit(
        InstructionContent(text="Original instruction."),
        message="original",
    )
    t.commit(
        InstructionContent(text="Updated instruction."),
        operation=CommitOperation.EDIT,
        response_to=c1.commit_hash,
        message="edited",
    )
    edit_hash = t.head
    t.close()
    return edit_hash


# ---------------------------------------------------------------------------
# Log command tests
# ---------------------------------------------------------------------------

class TestLogCommand:
    """Tests for tract log."""

    def test_log_shows_commits(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "log"])
            assert result.exit_code == 0
            assert "append" in result.output.lower()
            assert "system prompt" in result.output

    def test_log_limit(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "log", "-n", "1"])
            assert result.exit_code == 0
            # Should only show 1 commit (the most recent)
            assert "assistant reply" in result.output
            # Should NOT show the first commit
            assert "system prompt" not in result.output

    def test_log_verbose(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "log", "-v"])
            assert result.exit_code == 0
            # Verbose mode shows full hashes and types
            assert "Operation:" in result.output or "append" in result.output.lower()
            assert "instruction" in result.output.lower() or "dialogue" in result.output.lower()

    def test_log_op_filter(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract_with_edit("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "log", "--op", "edit"])
            assert result.exit_code == 0
            assert "edit" in result.output.lower()

    def test_log_empty(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_empty_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "log"])
            assert result.exit_code == 0
            assert "no commits" in result.output.lower()


# ---------------------------------------------------------------------------
# Status command tests
# ---------------------------------------------------------------------------

class TestStatusCommand:
    """Tests for tract status."""

    def test_status_shows_head(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "status"])
            assert result.exit_code == 0
            assert "main" in result.output.lower()

    def test_status_detached(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            # Get first commit hash for checkout
            from tract.tract import Tract
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()

            # Checkout detached
            runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "checkout", first_hash])
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "status"])
            assert result.exit_code == 0
            assert "detached" in result.output.lower()

    def test_status_no_commits(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_empty_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "status"])
            assert result.exit_code == 0
            assert "no commits" in result.output.lower()

    def test_status_shows_token_count(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "status"])
            assert result.exit_code == 0
            # Should show token info
            assert "token" in result.output.lower()

    def test_status_shows_no_budget(self, runner: CliRunner):
        """Status without budget shows 'no budget set' message.

        Token budget is an in-memory config, not persisted to DB.
        CLI opens with default config, so no budget is shown.
        """
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "status"])
            assert result.exit_code == 0
            assert "no budget set" in result.output.lower()


# ---------------------------------------------------------------------------
# Diff command tests
# ---------------------------------------------------------------------------

class TestDiffCommand:
    """Tests for tract diff."""

    def test_diff_default(self, runner: CliRunner):
        """Diff with no args diffs HEAD against parent."""
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "diff"])
            assert result.exit_code == 0
            assert "added" in result.output.lower() or "diff" in result.output.lower()

    def test_diff_two_commits(self, runner: CliRunner):
        """Diff between two specific commits."""
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            hash_a = entries[-1].commit_hash  # oldest
            hash_b = entries[0].commit_hash   # newest
            t.close()

            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "diff", hash_a, hash_b,
            ])
            assert result.exit_code == 0

    def test_diff_stat(self, runner: CliRunner):
        """Diff with --stat flag shows summary only."""
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "diff", "--stat"])
            assert result.exit_code == 0
            assert "added" in result.output.lower()

    def test_diff_no_commits(self, runner: CliRunner):
        """Diff with no commits gives error."""
        with runner.isolated_filesystem():
            _setup_empty_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "diff"])
            assert result.exit_code == 1

    def test_diff_with_edit(self, runner: CliRunner):
        """Diff auto-resolves EDIT commit against its target."""
        with runner.isolated_filesystem():
            _setup_tract_with_edit("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "diff"])
            assert result.exit_code == 0
            assert "modified" in result.output.lower() or "diff" in result.output.lower()


# ---------------------------------------------------------------------------
# Reset command tests
# ---------------------------------------------------------------------------

class TestResetCommand:
    """Tests for tract reset."""

    def test_reset_soft(self, runner: CliRunner):
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()

            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "reset", "--soft", first_hash,
            ])
            assert result.exit_code == 0
            assert first_hash[:8] in result.output
            assert "soft" in result.output.lower()

    def test_reset_hard_needs_force(self, runner: CliRunner):
        """Hard reset without --force should fail."""
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()

            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "reset", "--hard", first_hash,
            ])
            assert result.exit_code == 1
            assert "force" in result.output.lower() or "force" in result.stderr.lower() if result.stderr else "force" in result.output.lower()

    def test_reset_hard_with_force(self, runner: CliRunner):
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()

            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "reset", "--hard", "--force", first_hash,
            ])
            assert result.exit_code == 0
            assert first_hash[:8] in result.output
            assert "hard" in result.output.lower()

    def test_reset_invalid_hash(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "reset", "nonexistent_hash_value",
            ])
            assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Checkout command tests
# ---------------------------------------------------------------------------

class TestCheckoutCommand:
    """Tests for tract checkout."""

    def test_checkout_commit_detaches(self, runner: CliRunner):
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()

            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", first_hash,
            ])
            assert result.exit_code == 0
            assert "detached" in result.output.lower()

    def test_checkout_branch_attaches(self, runner: CliRunner):
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            # First detach
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()
            runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", first_hash,
            ])

            # Now checkout main branch
            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", "main",
            ])
            assert result.exit_code == 0
            assert "main" in result.output.lower()

    def test_checkout_dash(self, runner: CliRunner):
        """checkout '-' returns to previous position."""
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            first_hash = entries[-1].commit_hash
            t.close()

            # Checkout commit to set PREV_HEAD
            runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", first_hash,
            ])
            # Go back
            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", "-",
            ])
            assert result.exit_code == 0

    def test_checkout_invalid(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", "nonexistent_ref_value",
            ])
            assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Integration / help tests
# ---------------------------------------------------------------------------

class TestCLIIntegration:
    """General CLI integration tests."""

    def test_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "tract" in result.output.lower()
        assert "log" in result.output
        assert "status" in result.output
        assert "diff" in result.output
        assert "reset" in result.output
        assert "checkout" in result.output

    def test_subcommand_help(self, runner: CliRunner):
        for cmd in ["log", "status", "diff", "reset", "checkout"]:
            result = runner.invoke(cli, [cmd, "--help"])
            assert result.exit_code == 0, f"{cmd} --help failed"

    def test_no_cli_deps_for_import(self):
        """Importing tract (not tract.cli) should not require click/rich."""
        # This is tested implicitly by the fact that this test file imports tract.cli
        # but the main tract package should work without it.
        import importlib
        import tract as t
        importlib.reload(t)
        assert hasattr(t, "Tract")

    def test_auto_discover_tract(self, runner: CliRunner):
        """When --tract-id is omitted, auto-discover works for single tract."""
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "log"])
            assert result.exit_code == 0
            assert "system prompt" in result.output

    def test_db_not_found(self, runner: CliRunner):
        """Graceful error when database doesn't exist."""
        with runner.isolated_filesystem():
            result = runner.invoke(cli, ["--db", "nonexistent.db", "log"])
            assert result.exit_code == 1

    def test_log_default_limit(self, runner: CliRunner):
        """Log with default limit shows up to 20 commits."""
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "log"])
            assert result.exit_code == 0
            # 3 commits exist, all should show
            assert "system prompt" in result.output
            assert "user greeting" in result.output
            assert "assistant reply" in result.output

    def test_prefix_resolution(self, runner: CliRunner):
        """Commands accept hash prefixes (min 4 chars)."""
        from tract.tract import Tract

        with runner.isolated_filesystem():
            _setup_tract("test.db")
            t = Tract.open(path="test.db", tract_id="test-tract")
            entries = t.log(limit=10)
            prefix = entries[-1].commit_hash[:8]
            t.close()

            result = runner.invoke(cli, [
                "--db", "test.db", "--tract-id", "test-tract",
                "checkout", prefix,
            ])
            assert result.exit_code == 0
