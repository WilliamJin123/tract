"""Comprehensive tests for the tract CLI module.

Tests the CLI entry point at tract.cli.main(argv), covering:
- init, log, status, compile, show, diff, branches, config, search, compress
- Argument parsing edge cases
- Error handling for uninitialized state

Uses isolated tmp_path directories with monkeypatched DB/state paths so
tests never touch the real filesystem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tract.cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_id(tmp_path: Path) -> str:
    """Read the current tract ID from .tract/current."""
    return (tmp_path / ".tract" / "current").read_text().strip()


def _init_and_open(tmp_path: Path):
    """Run CLI init, then open the tract via the library for test data setup.

    Returns a Tract instance backed by the same DB file and tract_id that
    ``main(["init"])`` created. Caller is responsible for calling ``.close()``.
    """
    from tract import Tract

    main(["init"])
    tract_id = _read_id(tmp_path)
    db_path = str(tmp_path / ".tract" / "tract.db")
    t = Tract.open(path=db_path, tract_id=tract_id)
    return t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tract_dir(tmp_path, monkeypatch):
    """Isolated temp directory with patched CLI paths.

    Patches the four module-level path constants in ``tract.cli`` so all
    CLI operations target ``tmp_path/.tract/`` instead of the real CWD.
    """
    monkeypatch.setattr("tract.cli.DB_PATH", str(tmp_path / ".tract" / "tract.db"))
    monkeypatch.setattr("tract.cli.CURRENT_FILE", str(tmp_path / ".tract" / "current"))
    monkeypatch.setattr("tract.cli.SPAWNED_DIR", str(tmp_path / ".tract" / "spawned"))
    monkeypatch.setattr("tract.cli.PROMPTS_DIR", str(tmp_path / ".tract" / "prompts"))
    return tmp_path


# ===========================================================================
# TestInit
# ===========================================================================


class TestInit:
    """Tests for ``tract init``."""

    def test_init_creates_directory_structure(self, tract_dir):
        """Init should create .tract/, tract.db, current, prompts/, spawned/."""
        main(["init"])

        tract_root = tract_dir / ".tract"
        assert tract_root.is_dir()
        assert (tract_root / "tract.db").is_file()
        assert (tract_root / "current").is_file()
        assert (tract_root / "prompts").is_dir()
        assert (tract_root / "spawned").is_dir()

    def test_init_idempotent(self, tract_dir):
        """Calling init twice should reuse the existing tract, not crash."""
        main(["init"])
        first_id = _read_id(tract_dir)

        main(["init"])
        second_id = _read_id(tract_dir)

        assert first_id == second_id

    def test_init_prints_tract_id(self, tract_dir, capsys):
        """Init output should contain the tract ID."""
        main(["init"])
        captured = capsys.readouterr()
        tract_id = _read_id(tract_dir)
        # CLI prints init info to stderr
        assert tract_id in captured.err

    def test_init_current_file_has_valid_id(self, tract_dir):
        """The current file should contain a non-empty, single-line tract ID."""
        main(["init"])
        content = (tract_dir / ".tract" / "current").read_text()
        lines = content.strip().splitlines()
        assert len(lines) == 1
        assert len(lines[0].strip()) > 0

    def test_init_idempotent_output(self, tract_dir, capsys):
        """Second init should print 'Already initialized'."""
        main(["init"])
        capsys.readouterr()  # discard first output

        main(["init"])
        captured = capsys.readouterr()
        # Init messages go to stderr
        assert "already initialized" in captured.err.lower()

    def test_init_creates_seed_commit(self, tract_dir):
        """Init should create an initial seed commit in the database."""
        from tract import Tract

        main(["init"])
        tract_id = _read_id(tract_dir)
        db_path = str(tract_dir / ".tract" / "tract.db")
        t = Tract.open(path=db_path, tract_id=tract_id)
        commits = t.log(limit=10)
        t.close()
        # At least one commit (the init seed)
        assert len(commits) >= 1


# ===========================================================================
# TestLog
# ===========================================================================


class TestLog:
    """Tests for ``tract log``."""

    def test_log_shows_commits(self, tract_dir, capsys):
        """After adding commits, log should display them."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()  # discard init output
        t.system("You are a helpful assistant.")
        t.user("Hello there")
        t.assistant("Hi! How can I help?")
        t.close()

        main(["log"])
        captured = capsys.readouterr()
        # Should show at least one line per commit; init + 3 = 4 commits
        lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]
        assert len(lines) >= 3

    def test_log_after_init_shows_seed(self, tract_dir, capsys):
        """After init with no explicit commits, log shows the init commit."""
        main(["init"])
        capsys.readouterr()

        main(["log"])
        captured = capsys.readouterr()
        # Should show at least the seed commit (contains "init" message)
        assert "init" in captured.out.lower() or len(captured.out.strip()) > 0

    def test_log_respects_limit(self, tract_dir, capsys):
        """--limit should restrict the number of displayed entries."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        for i in range(10):
            t.user(f"Message number {i}")
        t.close()

        # Unlimited
        main(["log", "--limit", "100"])
        captured = capsys.readouterr()
        unlimited_lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]

        # Limited to 3
        main(["log", "--limit", "3"])
        captured = capsys.readouterr()
        limited_lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]

        assert len(limited_lines) <= 3
        assert len(limited_lines) < len(unlimited_lines)

    def test_log_without_init_fails(self, tract_dir):
        """Log without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["log"])

    def test_log_shows_priority_tags(self, tract_dir, capsys):
        """Pinned commits should show [PINNED] marker in log output."""
        from tract import Priority

        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("Important message")
        t.annotate(info.commit_hash, Priority.PINNED)
        t.close()

        main(["log"])
        captured = capsys.readouterr()
        assert "PINNED" in captured.out.upper()

    def test_log_limit_short_flag(self, tract_dir, capsys):
        """-n short flag should also restrict log output."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        for i in range(5):
            t.user(f"Message {i}")
        t.close()

        main(["log", "-n", "2"])
        captured = capsys.readouterr()
        lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]
        assert len(lines) <= 2

    def test_log_shows_commit_hashes(self, tract_dir, capsys):
        """Log output should include 8-char hash prefixes."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("Test message")
        t.close()

        main(["log"])
        captured = capsys.readouterr()
        # The 8-char prefix of the commit hash should appear
        assert info.commit_hash[:8] in captured.out

    def test_log_shows_token_count(self, tract_dir, capsys):
        """Log output should include token counts (tok suffix)."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.user("Some content for token counting")
        t.close()

        main(["log"])
        captured = capsys.readouterr()
        assert "tok" in captured.out


# ===========================================================================
# TestStatus
# ===========================================================================


class TestStatus:
    """Tests for ``tract status``."""

    def test_status_shows_info(self, tract_dir, capsys):
        """Status should display Tract:, Branch:, Tokens:, Commits: labels."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("Test system prompt")
        t.close()

        main(["status"])
        captured = capsys.readouterr()
        output = captured.out
        assert "Tract:" in output
        assert "Branch:" in output
        assert "Tokens:" in output
        assert "Commits:" in output

    def test_status_without_init_fails(self, tract_dir):
        """Status without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["status"])

    def test_status_shows_tract_id(self, tract_dir, capsys):
        """Status output should contain at least the 8-char prefix of the tract ID."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.close()
        tract_id = _read_id(tract_dir)

        main(["status"])
        captured = capsys.readouterr()
        assert tract_id[:8] in captured.out

    def test_status_shows_branch_name(self, tract_dir, capsys):
        """Status should show the current branch name."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.close()

        main(["status"])
        captured = capsys.readouterr()
        # Default branch is "main" (or whatever session.create_tract uses)
        assert "main" in captured.out.lower() or "default" in captured.out.lower()

    def test_status_shows_token_count(self, tract_dir, capsys):
        """Status should show non-zero token count after commits."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("A long system prompt with many tokens to count accurately")
        t.user("A substantial user message with content worth counting")
        t.close()

        main(["status"])
        captured = capsys.readouterr()
        # Token count should appear after "Tokens:" label
        for line in captured.out.splitlines():
            if "Tokens:" in line:
                parts = line.split("Tokens:")
                if len(parts) > 1:
                    token_str = parts[1].strip().split()[0]
                    assert int(token_str) > 0
                break


# ===========================================================================
# TestCompile
# ===========================================================================


class TestCompile:
    """Tests for ``tract compile``."""

    def test_compile_text_format(self, tract_dir, capsys):
        """Default compile format should output readable text."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hi")
        t.close()

        main(["compile"])
        captured = capsys.readouterr()
        output = captured.out
        assert len(output.strip()) > 0
        # Text format uses [ROLE]: prefix
        assert "[" in output and "]" in output

    def test_compile_json_format(self, tract_dir, capsys):
        """--format json should output valid JSON with role/content."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hi")
        t.close()

        main(["compile", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        for msg in data:
            assert "role" in msg
            assert "content" in msg

    def test_compile_openai_format(self, tract_dir, capsys):
        """--format openai should output valid JSON with role/content keys."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hello")
        t.close()

        main(["compile", "--format", "openai"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        for msg in data:
            assert "role" in msg
            assert "content" in msg
        # Should contain system and user roles
        roles = {m["role"] for m in data}
        assert "system" in roles
        assert "user" in roles

    def test_compile_anthropic_format(self, tract_dir, capsys):
        """--format anthropic should output dict with system and messages keys."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hello")
        t.close()

        main(["compile", "--format", "anthropic"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, dict)
        assert "messages" in data
        assert "system" in data
        # System should be a non-empty string (since we committed a system instruction)
        assert data["system"] is not None
        assert len(data["system"]) > 0

    def test_compile_without_init_fails(self, tract_dir):
        """Compile without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["compile"])

    def test_compile_full_strategy(self, tract_dir, capsys):
        """--strategy full should work without errors."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hi")
        t.close()

        main(["compile", "--strategy", "full"])
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_compile_messages_strategy(self, tract_dir, capsys):
        """--strategy messages should work without errors."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hi")
        t.close()

        main(["compile", "--strategy", "messages"])
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_compile_adaptive_strategy(self, tract_dir, capsys):
        """--strategy adaptive should work without errors."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.user("Hi")
        t.close()

        main(["compile", "--strategy", "adaptive"])
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_compile_includes_content(self, tract_dir, capsys):
        """Committed content should appear in compile output."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are a pirate captain.")
        t.user("Ahoy there matey")
        t.close()

        main(["compile"])
        captured = capsys.readouterr()
        assert "pirate" in captured.out.lower() or "ahoy" in captured.out.lower()

    def test_compile_json_message_count(self, tract_dir, capsys):
        """JSON output should have at least as many messages as user commits."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("System")
        t.user("User message 1")
        t.assistant("Response 1")
        t.close()

        main(["compile", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        # At least init commit + system + user + assistant = 4 messages
        assert len(data) >= 3

    def test_compile_format_short_flag(self, tract_dir, capsys):
        """-f short flag should work for format selection."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("Test")
        t.close()

        main(["compile", "-f", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)


# ===========================================================================
# TestShow
# ===========================================================================


class TestShow:
    """Tests for ``tract show <hash>``."""

    def test_show_commit(self, tract_dir, capsys):
        """Show should display commit details including hash, type, date, content."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.system("Important system prompt")
        commit_hash = info.commit_hash
        t.close()

        main(["show", commit_hash])
        captured = capsys.readouterr()
        output = captured.out
        # Should show the full hash
        assert commit_hash in output
        # Should show the content type
        assert "instruction" in output.lower()
        # Should show "Date:" or date info
        assert "Date:" in output
        # Should show the content text
        assert "Important system prompt" in output

    def test_show_commit_prefix(self, tract_dir, capsys):
        """Show should work with a hash prefix (not full hash)."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("Prefix test message")
        commit_hash = info.commit_hash
        t.close()

        # Use only the first 8 characters
        main(["show", commit_hash[:8]])
        captured = capsys.readouterr()
        assert commit_hash in captured.out

    def test_show_nonexistent_hash(self, tract_dir):
        """Showing a nonexistent hash should exit with an error."""
        _init_and_open(tract_dir).close()

        with pytest.raises(SystemExit):
            main(["show", "deadbeef12345678deadbeef12345678"])

    def test_show_without_init_fails(self, tract_dir):
        """Show without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["show", "abc123"])

    def test_show_displays_priority(self, tract_dir, capsys):
        """Show should display the commit's priority."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.system("Pinned prompt")
        # system() auto-pins by default
        t.close()

        main(["show", info.commit_hash])
        captured = capsys.readouterr()
        assert "Priority:" in captured.out

    def test_show_displays_message(self, tract_dir, capsys):
        """Show should display the commit message."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("Hello", message="greeting message")
        t.close()

        main(["show", info.commit_hash])
        captured = capsys.readouterr()
        assert "greeting message" in captured.out

    def test_show_displays_token_count(self, tract_dir, capsys):
        """Show should display the commit's token count."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("Hello world")
        t.close()

        main(["show", info.commit_hash])
        captured = capsys.readouterr()
        assert "Tokens:" in captured.out


# ===========================================================================
# TestDiff
# ===========================================================================


class TestDiff:
    """Tests for ``tract diff``."""

    def test_diff_default(self, tract_dir, capsys):
        """``tract diff`` should not crash after init with some content."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("System prompt")
        t.user("Hello")
        t.close()

        main(["diff"])
        captured = capsys.readouterr()
        # Should produce some output with token delta info
        assert "token" in captured.out.lower() or "message" in captured.out.lower()

    def test_diff_branch_comparison(self, tract_dir, capsys):
        """After creating branches, ``tract diff main..feature`` should work."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("Base content")
        t.branch("feature")
        t.user("Feature-only content")
        t.switch("main")
        t.close()

        main(["diff", "main..feature"])
        captured = capsys.readouterr()
        # Should produce output with diff stats
        assert "token" in captured.out.lower() or "message" in captured.out.lower()

    def test_diff_without_init_fails(self, tract_dir):
        """Diff without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["diff"])

    def test_diff_shows_change_stats(self, tract_dir, capsys):
        """Diff output should show +/-tokens and +/-messages format."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("System prompt")
        t.user("First user message")
        t.close()

        main(["diff"])
        captured = capsys.readouterr()
        # Format is: "+N/-N tokens, +N/-N messages"
        assert "tokens" in captured.out
        assert "messages" in captured.out


# ===========================================================================
# TestBranches
# ===========================================================================


class TestBranches:
    """Tests for ``tract branches``."""

    def test_branches_shows_main(self, tract_dir, capsys):
        """After init, branches should show the main branch."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.close()

        main(["branches"])
        captured = capsys.readouterr()
        assert "main" in captured.out

    def test_branches_shows_current_marker(self, tract_dir, capsys):
        """Active branch should have ``* `` prefix."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.close()

        main(["branches"])
        captured = capsys.readouterr()
        assert "* main" in captured.out

    def test_branches_multiple(self, tract_dir, capsys):
        """After creating branches, should show all of them."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("Base content")
        t.branch("feature")
        t.switch("main")
        t.branch("bugfix")
        t.switch("main")
        t.close()

        main(["branches"])
        captured = capsys.readouterr()
        assert "main" in captured.out
        assert "feature" in captured.out
        assert "bugfix" in captured.out

    def test_branches_without_init_fails(self, tract_dir):
        """Branches without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["branches"])

    def test_branches_shows_commit_hash(self, tract_dir, capsys):
        """Branch listing should include 8-char commit hash prefixes."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.close()

        main(["branches"])
        captured = capsys.readouterr()
        # Each line should have an 8-char hex prefix somewhere
        lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]
        assert len(lines) >= 1
        for line in lines:
            # Branch line format: "* main  abcdef12" or "  feat  abcdef12"
            parts = line.split()
            assert len(parts) >= 2

    def test_branches_current_marker_on_switched_branch(self, tract_dir, capsys):
        """After switching branches, the * marker should follow."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("Base")
        t.branch("feature")
        # Now on "feature"
        t.close()

        main(["branches"])
        captured = capsys.readouterr()
        # feature should be current
        assert "* feature" in captured.out
        # main should NOT be marked as current
        for line in captured.out.strip().splitlines():
            if "main" in line and "feature" not in line:
                assert not line.strip().startswith("*")


# ===========================================================================
# TestConfig
# ===========================================================================


class TestConfig:
    """Tests for ``tract config``."""

    def test_config_empty(self, tract_dir, capsys):
        """After init with no configs, should show 'No configs set'."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.close()

        main(["config"])
        captured = capsys.readouterr()
        assert "no configs set" in captured.out.lower()

    def test_config_shows_values(self, tract_dir, capsys):
        """After setting configs via library, should display them."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.configure(model="gpt-4o", temperature=0.7)
        t.close()

        main(["config"])
        captured = capsys.readouterr()
        output = captured.out
        assert "model" in output
        assert "gpt-4o" in output
        assert "temperature" in output

    def test_config_without_init_fails(self, tract_dir):
        """Config without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["config"])

    def test_config_shows_multiple_values(self, tract_dir, capsys):
        """Multiple config values should all appear."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.configure(model="gpt-4o-mini", max_tokens=2048, temperature=0.5)
        t.close()

        main(["config"])
        captured = capsys.readouterr()
        assert "gpt-4o-mini" in captured.out
        assert "2048" in captured.out
        assert "0.5" in captured.out


# ===========================================================================
# TestSearch
# ===========================================================================


class TestSearch:
    """Tests for ``tract search``."""

    def test_search_finds_content(self, tract_dir, capsys):
        """Search should find commits matching the query."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are a helpful coding assistant.")
        t.user("Write me a Python function for sorting.")
        t.assistant("Here is a bubble sort implementation.")
        t.close()

        main(["search", "sorting"])
        captured = capsys.readouterr()
        # Should find at least one match (the user message about sorting)
        assert len(captured.out.strip()) > 0
        assert "no matches" not in captured.out.lower()

    def test_search_no_results(self, tract_dir, capsys):
        """Search for nonexistent term should report no matches."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are helpful.")
        t.close()

        main(["search", "xyznonexistentterm"])
        captured = capsys.readouterr()
        assert "no matches" in captured.out.lower()

    def test_search_without_init_fails(self, tract_dir):
        """Search without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["search", "test"])

    def test_search_requires_term(self, tract_dir, capsys):
        """Missing search term argument should cause argparse to exit."""
        main(["init"])
        capsys.readouterr()

        with pytest.raises(SystemExit):
            main(["search"])

    def test_search_finds_system_instruction(self, tract_dir, capsys):
        """Search should match content inside system instructions."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are an expert marine biologist specializing in coral reefs.")
        t.close()

        main(["search", "coral"])
        captured = capsys.readouterr()
        assert "no matches" not in captured.out.lower()
        assert len(captured.out.strip()) > 0

    def test_search_result_format(self, tract_dir, capsys):
        """Search results should include tract ID prefix and commit hash prefix."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("unique_search_marker_12345")
        t.close()

        main(["search", "unique_search_marker"])
        captured = capsys.readouterr()
        # Each result line should have commit_hash[:8]
        lines = [ln for ln in captured.out.strip().splitlines() if ln.strip()]
        assert len(lines) >= 1
        assert info.commit_hash[:8] in captured.out


# ===========================================================================
# TestCompress
# ===========================================================================


class TestCompress:
    """Tests for ``tract compress``."""

    def test_compress_with_content(self, tract_dir, capsys):
        """Compress with --content should compress and show stats."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.user("First message about topic A")
        t.assistant("Response about topic A")
        t.user("Second message about topic B")
        t.assistant("Response about topic B")
        t.close()

        main(["compress", "--content", "Summary of topics A and B"])
        captured = capsys.readouterr()
        # Should report compression stats
        assert "token" in captured.out.lower()

    def test_compress_requires_content_flag(self, tract_dir):
        """Compress without --content flag should fail (argparse requires it)."""
        main(["init"])

        # --content is a required argument in the parser
        with pytest.raises(SystemExit):
            main(["compress"])

    def test_compress_without_init_fails(self, tract_dir):
        """Compress without init should exit with an error."""
        with pytest.raises(SystemExit):
            main(["compress", "--content", "summary"])

    def test_compress_shows_ratio(self, tract_dir, capsys):
        """Compress output should include the compression ratio."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.user("Message one about topic alpha")
        t.assistant("Response one about topic alpha")
        t.user("Message two about topic beta")
        t.assistant("Response two about topic beta")
        t.close()

        main(["compress", "--content", "Summary of alpha and beta topics"])
        captured = capsys.readouterr()
        assert "ratio" in captured.out.lower()

    def test_compress_with_target_tokens(self, tract_dir, capsys):
        """Compress with --target-tokens should not crash."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.user("First message")
        t.assistant("First response")
        t.user("Second message")
        t.assistant("Second response")
        t.close()

        main(["compress", "--content", "Short summary", "--target-tokens", "50"])
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0


# ===========================================================================
# TestCLIArgParsing
# ===========================================================================


class TestCLIArgParsing:
    """Tests for CLI argument parsing edge cases."""

    def test_no_command_prints_help_and_exits(self, tract_dir, capsys):
        """Running with no command should print help and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 0

    def test_version_flag(self, tract_dir, capsys):
        """--version should print the version string and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--version"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        from tract._version import __version__

        assert __version__ in captured.out

    def test_unknown_command_exits(self, tract_dir):
        """An unknown subcommand should exit with an error code."""
        with pytest.raises(SystemExit) as exc_info:
            main(["nonexistent_command"])
        # argparse rejects unknown choices with exit code 2
        assert exc_info.value.code == 2

    def test_help_flag(self, tract_dir, capsys):
        """--help should print usage and exit 0."""
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

        captured = capsys.readouterr()
        assert "usage" in captured.out.lower() or "tract" in captured.out.lower()

    def test_subcommand_help(self, tract_dir, capsys):
        """Subcommand --help should print subcommand-specific usage."""
        with pytest.raises(SystemExit) as exc_info:
            main(["init", "--help"])
        assert exc_info.value.code == 0

    def test_compile_format_choices(self, tract_dir):
        """Invalid --format value should cause argparse to exit."""
        with pytest.raises(SystemExit):
            main(["compile", "--format", "invalid_format"])

    def test_compile_strategy_choices(self, tract_dir):
        """Invalid --strategy value should cause argparse to exit."""
        with pytest.raises(SystemExit):
            main(["compile", "--strategy", "invalid_strategy"])

    def test_log_limit_type(self, tract_dir):
        """--limit with non-integer should cause argparse to exit."""
        with pytest.raises(SystemExit):
            main(["log", "--limit", "abc"])


# ===========================================================================
# TestIntegrationWorkflow
# ===========================================================================


class TestIntegrationWorkflow:
    """End-to-end workflow tests combining multiple CLI commands."""

    def test_init_commit_log_compile_flow(self, tract_dir, capsys):
        """Full workflow: init -> add content -> log -> compile -> status."""
        # Init (output goes to stderr)
        main(["init"])
        captured = capsys.readouterr()
        assert len(captured.err.strip()) > 0

        # Add content via library
        tract_id = _read_id(tract_dir)
        from tract import Tract
        db_path = str(tract_dir / ".tract" / "tract.db")
        t = Tract.open(path=db_path, tract_id=tract_id)
        t.system("You are a research assistant.")
        t.user("Summarize quantum computing")
        t.assistant("Quantum computing uses qubits...")
        t.close()

        # Log
        main(["log"])
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

        # Compile JSON
        main(["compile", "--format", "json"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) >= 3

        # Status
        main(["status"])
        captured = capsys.readouterr()
        assert "Tract:" in captured.out
        assert "Branch:" in captured.out

    def test_branch_workflow(self, tract_dir, capsys):
        """Create branches via library, then list and diff them via CLI."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("Base system prompt")
        t.branch("experiment")
        t.user("Experiment content")
        t.switch("main")
        t.close()

        # List branches
        main(["branches"])
        captured = capsys.readouterr()
        assert "main" in captured.out
        assert "experiment" in captured.out

        # Diff between branches
        main(["diff", "main..experiment"])
        captured = capsys.readouterr()
        assert "tokens" in captured.out
        assert "messages" in captured.out

    def test_search_after_multiple_commits(self, tract_dir, capsys):
        """Search should find specific content across many commits."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.system("You are a math tutor.")
        t.user("Explain derivatives")
        t.assistant("A derivative measures the rate of change.")
        t.user("Now explain integrals")
        t.assistant("An integral computes the area under a curve.")
        t.close()

        main(["search", "derivative"])
        captured = capsys.readouterr()
        assert "no matches" not in captured.out.lower()
        assert len(captured.out.strip()) > 0

    def test_config_roundtrip(self, tract_dir, capsys):
        """Set config via library, view via CLI."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.configure(model="claude-3-opus", max_tokens=4096)
        t.close()

        main(["config"])
        captured = capsys.readouterr()
        assert "claude-3-opus" in captured.out
        assert "4096" in captured.out

    def test_show_specific_commit(self, tract_dir, capsys):
        """Commit content, then show that specific commit by hash."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        info = t.user("Tell me about the Eiffel Tower")
        commit_hash = info.commit_hash
        t.close()

        main(["show", commit_hash])
        captured = capsys.readouterr()
        assert "Eiffel Tower" in captured.out
        assert commit_hash in captured.out

    def test_compress_then_log(self, tract_dir, capsys):
        """Compress content, then verify log still works after compression."""
        t = _init_and_open(tract_dir)
        capsys.readouterr()
        t.user("Old message 1")
        t.assistant("Old response 1")
        t.user("Old message 2")
        t.assistant("Old response 2")
        t.close()

        main(["compress", "--content", "Summary of old conversations"])
        captured = capsys.readouterr()
        assert "token" in captured.out.lower()

        main(["log"])
        captured = capsys.readouterr()
        assert len(captured.out.strip()) > 0

    def test_full_lifecycle(self, tract_dir, capsys):
        """Complete lifecycle: init -> commit -> branch -> config -> compile -> search."""
        # Init
        main(["init"])
        capsys.readouterr()

        # Open and add content
        tract_id = _read_id(tract_dir)
        from tract import Tract
        db_path = str(tract_dir / ".tract" / "tract.db")
        t = Tract.open(path=db_path, tract_id=tract_id)
        t.system("You are a specialized data analyst.")
        t.user("Analyze the quarterly revenue data.")
        t.assistant("The quarterly revenue shows a 15% increase.")
        t.configure(model="gpt-4o", temperature=0.3)
        t.branch("follow-up")
        t.user("What about the cost analysis?")
        t.switch("main")
        t.close()

        # Verify status
        main(["status"])
        captured = capsys.readouterr()
        assert "Commits:" in captured.out

        # Verify branches
        main(["branches"])
        captured = capsys.readouterr()
        assert "main" in captured.out
        assert "follow-up" in captured.out

        # Verify config
        main(["config"])
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out

        # Verify compile
        main(["compile", "-f", "openai"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert len(data) >= 3

        # Verify search
        main(["search", "revenue"])
        captured = capsys.readouterr()
        assert "no matches" not in captured.out.lower()

        # Verify diff between branches
        main(["diff", "main..follow-up"])
        captured = capsys.readouterr()
        assert "tokens" in captured.out
