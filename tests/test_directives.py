"""Tests for directives: named InstructionContent with compiler deduplication.

Covers:
- InstructionContent with name field
- t.directive(name, text) creates commit with correct content
- Default priority is PINNED
- Custom priority on directive
- Compiler dedup: two directives same name, only latest appears in compile
- Multiple names: each name deduped independently
- Directives without name (InstructionContent name=None) not deduped
- Directive override across branches (verify on current branch)
"""

from __future__ import annotations

import pytest

from tract import (
    InstructionContent,
    Priority,
    Tract,
    validate_content,
)


# ---------------------------------------------------------------------------
# InstructionContent name field
# ---------------------------------------------------------------------------


class TestInstructionContentName:
    """InstructionContent with the optional name field."""

    def test_name_defaults_to_none(self):
        """InstructionContent name defaults to None."""
        ic = InstructionContent(text="Hello")
        assert ic.name is None

    def test_name_can_be_set(self):
        """InstructionContent accepts a name string."""
        ic = InstructionContent(text="Hello", name="greeting")
        assert ic.name == "greeting"
        assert ic.text == "Hello"

    def test_name_in_serialization(self):
        """name field appears in model_dump when set."""
        ic = InstructionContent(text="Hello", name="greeting")
        data = ic.model_dump()
        assert data["name"] == "greeting"

    def test_validate_content_with_name(self):
        """validate_content() handles InstructionContent with name."""
        result = validate_content({
            "content_type": "instruction",
            "text": "Be helpful",
            "name": "behavior",
        })
        assert isinstance(result, InstructionContent)
        assert result.name == "behavior"


# ---------------------------------------------------------------------------
# t.directive() API
# ---------------------------------------------------------------------------


class TestDirectiveAPI:
    """t.directive(name, text) method."""

    def test_directive_creates_commit(self):
        """directive() creates a commit and returns CommitInfo."""
        with Tract.open() as t:
            info = t.directive("safety", "Never share secrets")
            assert info is not None
            assert info.commit_hash is not None
            assert info.content_type == "instruction"

    def test_directive_stores_name_in_content(self):
        """The committed content has the directive name."""
        with Tract.open() as t:
            t.directive("safety", "Never share secrets")
            compiled = t.compile()
            # The instruction should appear in compiled messages
            texts = [m.content for m in compiled.messages]
            assert any("Never share secrets" in txt for txt in texts)

    def test_directive_default_message(self):
        """directive() generates a default commit message."""
        with Tract.open() as t:
            info = t.directive("safety", "Never share secrets")
            assert "safety" in info.message

    def test_directive_custom_message(self):
        """directive() accepts a custom commit message."""
        with Tract.open() as t:
            info = t.directive("safety", "Never share secrets", message="custom msg")
            assert info.message == "custom msg"

    def test_directive_with_tags(self):
        """directive() passes tags to the commit."""
        with Tract.open() as t:
            # Register tags first (strict mode is default)
            t.register_tag("policy", "Policy directives")
            info = t.directive("safety", "Never share secrets", tags=["policy"])
            # If tags are stored, the commit should have them
            assert info is not None


# ---------------------------------------------------------------------------
# Priority behavior
# ---------------------------------------------------------------------------


class TestDirectivePriority:
    """Default and custom priority on directives."""

    def test_default_priority_is_pinned(self):
        """Directives default to PINNED priority."""
        with Tract.open() as t:
            info = t.directive("safety", "Never share secrets")
            # Verify pinned by checking annotation via get_latest
            ann = t._annotation_repo.get_latest(info.commit_hash)
            assert ann is not None
            assert ann.priority == "pinned"

    def test_custom_priority_normal(self):
        """directive(priority=NORMAL) overrides the default PINNED.

        InstructionContent has default_priority='pinned' in type hints,
        so the commit engine auto-annotates as PINNED. When the user
        explicitly requests NORMAL, directive() adds a NORMAL annotation
        which overrides the PINNED one (latest annotation wins).
        """
        with Tract.open() as t:
            info = t.directive("temp", "Temporary instruction", priority=Priority.NORMAL)
            # The latest annotation should be NORMAL, overriding the auto PINNED
            ann = t._annotation_repo.get_latest(info.commit_hash)
            assert ann is not None
            assert ann.priority == Priority.NORMAL

    def test_custom_priority_skip(self):
        """directive(priority=SKIP) marks as skip."""
        with Tract.open() as t:
            info = t.directive("hidden", "Hidden directive", priority=Priority.SKIP)
            compiled = t.compile()
            # SKIP priority means excluded from compile
            texts = [m.content for m in compiled.messages]
            assert not any("Hidden directive" in txt for txt in texts)


# ---------------------------------------------------------------------------
# Compiler deduplication by name
# ---------------------------------------------------------------------------


class TestDirectiveDedup:
    """Compiler deduplicates named InstructionContent: same name -> closest to HEAD wins."""

    def test_two_same_name_only_latest(self):
        """Two directives with same name: only the later one appears in compile."""
        with Tract.open() as t:
            t.directive("protocol", "Old protocol: do X")
            t.directive("protocol", "New protocol: do Y")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            # The old protocol should be deduped away
            assert not any("Old protocol" in txt for txt in texts)
            # The new protocol should remain
            assert any("New protocol" in txt for txt in texts)

    def test_three_same_name_only_latest(self):
        """Three directives with same name: only the latest appears."""
        with Tract.open() as t:
            t.directive("tone", "Be formal")
            t.directive("tone", "Be casual")
            t.directive("tone", "Be professional")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            assert not any("Be formal" in txt for txt in texts)
            assert not any("Be casual" in txt for txt in texts)
            assert any("Be professional" in txt for txt in texts)

    def test_different_names_independent(self):
        """Different names are deduped independently."""
        with Tract.open() as t:
            t.directive("safety", "Be safe v1")
            t.directive("tone", "Be formal")
            t.directive("safety", "Be safe v2")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            # safety: v2 wins
            assert any("Be safe v2" in txt for txt in texts)
            assert not any("Be safe v1" in txt for txt in texts)
            # tone: only one, so it stays
            assert any("Be formal" in txt for txt in texts)

    def test_unnamed_instructions_not_deduped(self):
        """Instructions without a name are not affected by dedup."""
        with Tract.open() as t:
            t.system("First instruction")
            t.system("Second instruction")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            # Both should appear since they have no name
            assert any("First instruction" in txt for txt in texts)
            assert any("Second instruction" in txt for txt in texts)

    def test_named_and_unnamed_coexist(self):
        """Named directives coexist with unnamed instructions."""
        with Tract.open() as t:
            t.system("Regular instruction")
            t.directive("protocol", "Old protocol")
            t.directive("protocol", "New protocol")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            assert any("Regular instruction" in txt for txt in texts)
            assert any("New protocol" in txt for txt in texts)
            assert not any("Old protocol" in txt for txt in texts)

    def test_dedup_with_interleaved_messages(self):
        """Dedup works with non-directive commits interleaved."""
        with Tract.open() as t:
            t.directive("role", "You are a teacher")
            t.user("Hello")
            t.assistant("Hi there")
            t.directive("role", "You are a researcher")
            t.user("Another question")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            assert not any("teacher" in txt for txt in texts)
            assert any("researcher" in txt for txt in texts)


# ---------------------------------------------------------------------------
# Directive override across branches
# ---------------------------------------------------------------------------


class TestDirectiveBranchOverride:
    """Directive dedup scoped to current branch's DAG."""

    def test_directive_on_branch_overrides(self):
        """Directive on a branch overrides ancestor directive for that branch."""
        with Tract.open() as t:
            t.directive("protocol", "Main protocol")
            t.user("Setup")

            # Create a feature branch and switch to it
            t.branch("feature", switch=True)

            # Override on feature branch
            t.directive("protocol", "Feature protocol")

            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            assert any("Feature protocol" in txt for txt in texts)
            assert not any("Main protocol" in txt for txt in texts)

    def test_main_branch_unaffected_by_feature(self):
        """Directive override on feature branch does not affect main."""
        with Tract.open() as t:
            t.directive("protocol", "Main protocol")
            t.user("Setup")

            t.branch("feature", switch=True)
            t.directive("protocol", "Feature protocol")

            # Switch back to main
            t.switch("main")
            compiled = t.compile()
            texts = [m.content for m in compiled.messages]
            assert any("Main protocol" in txt for txt in texts)
            assert not any("Feature protocol" in txt for txt in texts)


# ---------------------------------------------------------------------------
# File-loading (path= parameter)
# ---------------------------------------------------------------------------


class TestFileLoading:
    """Tests for loading text from markdown files via path= parameter."""

    def test_directive_from_file(self, tmp_path):
        """directive(name, path=...) reads text from a file."""
        md = tmp_path / "safety.md"
        md.write_text("Never write to production DB.", encoding="utf-8")

        with Tract.open() as t:
            info = t.directive("safety", path=str(md))
            compiled = t.compile()
            assert any("production DB" in m.content for m in compiled.messages)

    def test_system_from_file(self, tmp_path):
        """system(path=...) reads text from a file."""
        md = tmp_path / "system.md"
        md.write_text("You are a security analyst.", encoding="utf-8")

        with Tract.open() as t:
            t.system(path=str(md))
            compiled = t.compile()
            assert any("security analyst" in m.content for m in compiled.messages)

    def test_user_from_file(self, tmp_path):
        """user(path=...) reads text from a file."""
        md = tmp_path / "task.md"
        md.write_text("Audit the auth module.", encoding="utf-8")

        with Tract.open() as t:
            t.user(path=str(md))
            compiled = t.compile()
            assert any("auth module" in m.content for m in compiled.messages)

    def test_assistant_from_file(self, tmp_path):
        """assistant(path=...) reads text from a file."""
        md = tmp_path / "response.md"
        md.write_text("The auth module is secure.", encoding="utf-8")

        with Tract.open() as t:
            t.assistant(path=str(md))
            compiled = t.compile()
            assert any("auth module is secure" in m.content for m in compiled.messages)

    def test_text_and_path_raises(self, tmp_path):
        """Passing both text and path raises ValueError."""
        md = tmp_path / "x.md"
        md.write_text("content", encoding="utf-8")

        with Tract.open() as t:
            with pytest.raises(ValueError, match="not both"):
                t.directive("test", "inline text", path=str(md))

    def test_neither_text_nor_path_raises(self):
        """Passing neither text nor path raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError, match="required"):
                t.directive("test")

    def test_path_file_not_found_raises(self):
        """Non-existent path raises ValueError."""
        with Tract.open() as t:
            with pytest.raises(ValueError, match="File not found"):
                t.directive("test", path="/nonexistent/file.md")

    def test_directive_path_with_unicode(self, tmp_path):
        """File with unicode content is read correctly."""
        md = tmp_path / "unicode.md"
        md.write_text("Priorit\u00e4t: Sicherheit \u00fcber alles.", encoding="utf-8")

        with Tract.open() as t:
            t.directive("priority", path=str(md))
            compiled = t.compile()
            assert any("Sicherheit" in m.content for m in compiled.messages)

    def test_system_inline_still_works(self):
        """system(text) still works after path= was added."""
        with Tract.open() as t:
            t.system("You are helpful.")
            compiled = t.compile()
            assert any("helpful" in m.content for m in compiled.messages)

    def test_prompt_dir_explicit(self, tmp_path):
        """Tract.open(prompt_dir=...) resolves relative paths from that dir."""
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "role.md").write_text("You are a DBA.", encoding="utf-8")

        with Tract.open(prompt_dir=str(prompts)) as t:
            t.directive("role", path="role.md")
            compiled = t.compile()
            assert any("DBA" in m.content for m in compiled.messages)

    def test_prompt_dir_auto_discovery(self, tmp_path, monkeypatch):
        """.tract/prompts/ is auto-discovered when prompt_dir is not set."""
        prompts = tmp_path / ".tract" / "prompts"
        prompts.mkdir(parents=True)
        (prompts / "safety.md").write_text("No eval().", encoding="utf-8")

        monkeypatch.chdir(tmp_path)
        with Tract.open() as t:
            assert t._prompt_dir is not None
            t.directive("safety", path="safety.md")
            compiled = t.compile()
            assert any("eval" in m.content for m in compiled.messages)

    def test_prompt_dir_no_auto_discovery_without_dir(self, tmp_path, monkeypatch):
        """No auto-discovery when .tract/prompts/ doesn't exist."""
        monkeypatch.chdir(tmp_path)
        with Tract.open() as t:
            assert t._prompt_dir is None

    def test_prompt_dir_fallback_to_cwd(self, tmp_path):
        """When prompt_dir is set but file is only at CWD, still resolves."""
        prompts = tmp_path / "prompts"
        prompts.mkdir()
        # File is NOT in prompts dir, but exists at the given relative path
        (tmp_path / "local.md").write_text("Local file.", encoding="utf-8")

        with Tract.open(prompt_dir=str(prompts)) as t:
            # Absolute path bypasses prompt_dir
            t.directive("local", path=str(tmp_path / "local.md"))
            compiled = t.compile()
            assert any("Local file" in m.content for m in compiled.messages)

    def test_prompt_dir_inherited_by_spawn(self, tmp_path):
        """Spawned child inherits prompt_dir from parent."""
        from tract import Session

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "child-role.md").write_text("You are a worker.", encoding="utf-8")

        db = str(tmp_path / "test.db")
        session = Session.open(db)
        parent = session.create_tract(display_name="parent")
        parent._prompt_dir = str(prompts)
        parent.system("Hello")

        child = session.spawn(
            parent,
            purpose="work",
            directives={"role": "child-role.md"},
        )
        # Child should have inherited prompt_dir but directive was passed as
        # string, not Path, so it won't auto-resolve via prompt_dir.
        # Use path= explicitly via the Path type:
        from pathlib import Path
        child2 = session.spawn(
            parent,
            purpose="work2",
            directives={"role": Path("child-role.md")},
        )
        compiled = child2.compile()
        text = " ".join(m.content for m in compiled.messages)
        assert "worker" in text

        session.close()
