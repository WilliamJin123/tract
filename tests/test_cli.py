"""CLI tests for Tract -- tests all 5 commands via Click's CliRunner.

Each test uses runner.isolated_filesystem() with file-backed databases
since CLI opens its own connection (separate from SDK setup).
"""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from tract.cli import cli
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
            assert "Operation:" in result.output
            assert "append" in result.output.lower()



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

    def test_status_no_commits(self, runner: CliRunner):
        with runner.isolated_filesystem():
            _setup_empty_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "status"])
            assert result.exit_code == 0
            assert "no commits" in result.output.lower()



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
            assert "diff" in result.output.lower()
            assert "added" in result.output.lower()

    def test_diff_stat(self, runner: CliRunner):
        """Diff with --stat flag shows summary only."""
        with runner.isolated_filesystem():
            _setup_tract("test.db")
            result = runner.invoke(cli, ["--db", "test.db", "--tract-id", "test-tract", "diff", "--stat"])
            assert result.exit_code == 0
            assert "added" in result.output.lower()



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
            assert "force" in result.output.lower()

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
        import subprocess
        import sys

        # Run in a subprocess to test import isolation
        result = subprocess.run(
            [sys.executable, "-c", "import tract; assert hasattr(tract, 'Tract')"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Import failed: {result.stderr}"

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
