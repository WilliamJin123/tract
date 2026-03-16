"""Security tests for path traversal prevention."""

import os
import tempfile
from pathlib import Path

import pytest

from tract.tract import _resolve_text


# ---------------------------------------------------------------------------
# _resolve_text: path traversal prevention
# ---------------------------------------------------------------------------


class TestResolveTextPathTraversal:
    """Tests that _resolve_text rejects malicious paths."""

    def test_relative_traversal_rejected(self, tmp_path: Path):
        """../../../etc/passwd style traversal must raise ValueError."""
        # Create a file outside prompt_dir to prove it can't be read
        secret = tmp_path / "secret.txt"
        secret.write_text("secret data")
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            _resolve_text(path="../secret.txt", prompt_dir=str(prompt_dir))

    def test_absolute_path_bypasses_prompt_dir(self, tmp_path: Path):
        """Absolute paths bypass prompt_dir (explicit caller intent)."""
        target = tmp_path / "target.txt"
        target.write_text("absolute content")
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()

        result = _resolve_text(path=str(target), prompt_dir=str(prompt_dir))
        assert result == "absolute content"

    def test_absolute_path_allowed_without_prompt_dir(self, tmp_path: Path):
        """Without prompt_dir, absolute paths work directly."""
        target = tmp_path / "target.txt"
        target.write_text("allowed content")

        result = _resolve_text(path=str(target), prompt_dir=None)
        assert result == "allowed content"

    def test_relative_path_without_prompt_dir_not_found(self):
        """Relative path without prompt_dir uses CWD — missing file raises."""
        with pytest.raises(ValueError, match="File not found"):
            _resolve_text(path="nonexistent_file_abc123.txt", prompt_dir=None)

    def test_valid_relative_path_works(self, tmp_path: Path):
        """A legitimate relative path inside prompt_dir should work."""
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "hello.txt").write_text("hello world")

        result = _resolve_text(path="hello.txt", prompt_dir=str(prompt_dir))
        assert result == "hello world"

    def test_valid_nested_relative_path_works(self, tmp_path: Path):
        """Subdirectory paths within prompt_dir should work."""
        prompt_dir = tmp_path / "prompts"
        sub = prompt_dir / "sub"
        sub.mkdir(parents=True)
        (sub / "data.txt").write_text("nested content")

        result = _resolve_text(path="sub/data.txt", prompt_dir=str(prompt_dir))
        assert result == "nested content"

    def test_dot_dot_in_middle_still_inside(self, tmp_path: Path):
        """sub/../other.txt that resolves inside prompt_dir is OK."""
        prompt_dir = tmp_path / "prompts"
        prompt_dir.mkdir()
        (prompt_dir / "sub").mkdir()
        (prompt_dir / "other.txt").write_text("ok")

        result = _resolve_text(path="sub/../other.txt", prompt_dir=str(prompt_dir))
        assert result == "ok"

    def test_deep_traversal_rejected(self, tmp_path: Path):
        """Multiple ../ levels must be caught."""
        prompt_dir = tmp_path / "a" / "b" / "c"
        prompt_dir.mkdir(parents=True)

        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            _resolve_text(
                path="../../../../etc/passwd", prompt_dir=str(prompt_dir)
            )

    def test_inline_text_still_works(self):
        """Passing text= directly should remain unaffected."""
        result = _resolve_text(text="inline content")
        assert result == "inline content"


# ---------------------------------------------------------------------------
# save_workflow: name validation
# ---------------------------------------------------------------------------


class TestSaveWorkflowPathTraversal:
    """Tests that save_workflow rejects dangerous names."""

    @pytest.fixture()
    def tract_mem(self):
        from tract import Tract

        return Tract.open(":memory:")

    @pytest.fixture()
    def tract_file(self, tmp_path: Path):
        from tract import Tract

        return Tract.open(str(tmp_path / "test.db"))

    def test_rejects_forward_slash(self, tract_mem):
        with pytest.raises(ValueError, match="must not contain"):
            tract_mem.save_workflow("../../evil", code="x = 1")

    def test_rejects_backslash(self, tract_mem):
        with pytest.raises(ValueError, match="must not contain"):
            tract_mem.save_workflow("..\\evil", code="x = 1")

    def test_rejects_dot_dot(self, tract_mem):
        with pytest.raises(ValueError, match="must not contain"):
            tract_mem.save_workflow("..", code="x = 1")

    def test_rejects_slash_in_name(self, tract_mem):
        with pytest.raises(ValueError, match="must not contain"):
            tract_mem.save_workflow("sub/workflow", code="x = 1")

    def test_valid_name_works(self, tract_file):
        path = tract_file.save_workflow("my_workflow", code="x = 1")
        assert path.name == "my_workflow.py"
        assert path.exists()
        assert path.read_text() == "x = 1"
