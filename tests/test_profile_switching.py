"""Tests for profile switching (Fix 5)."""
import pytest

from tract import Tract
from tract.toolkit.executor import ToolExecutor
from tract.toolkit.profiles import SELF_PROFILE, SUPERVISOR_PROFILE, FULL_PROFILE


class TestProfileSwitching:
    """Tests for ToolExecutor profile switching."""

    def test_set_profile_changes_available_tools(self):
        """set_profile() changes which tools are available."""
        t = Tract.open()
        executor = ToolExecutor(t)
        # Default: all tools
        all_tools = set(executor.available_tools())
        executor.set_profile("self")
        self_tools = set(executor.available_tools())
        assert len(self_tools) < len(all_tools)
        assert "commit" in self_tools
        assert "status" in self_tools

    def test_unlock_adds_tool_outside_profile(self):
        """unlock_tool() adds a tool not in the current profile."""
        t = Tract.open()
        executor = ToolExecutor(t)
        executor.set_profile("self")
        # gc is not in self profile
        assert "gc" not in executor.available_tools()
        executor.unlock_tool("gc")
        assert "gc" in executor.available_tools()

    def test_lock_removes_tool_in_profile(self):
        """lock_tool() removes a tool that's in the current profile."""
        t = Tract.open()
        executor = ToolExecutor(t)
        executor.set_profile("self")
        assert "commit" in executor.available_tools()
        executor.lock_tool("commit")
        assert "commit" not in executor.available_tools()

    def test_profile_switch_clears_overrides(self):
        """Switching profiles clears previous overrides."""
        t = Tract.open()
        executor = ToolExecutor(t)
        executor.set_profile("self")
        executor.unlock_tool("gc")
        assert "gc" in executor.available_tools()
        executor.set_profile("self")
        assert "gc" not in executor.available_tools()

    def test_facade_switch_profile(self):
        """Tract.switch_profile() delegates to executor."""
        t = Tract.open()
        t.switch_profile("self")
        # Should not raise

    def test_facade_unlock_lock(self):
        """Tract.unlock_tool() and lock_tool() delegate correctly."""
        t = Tract.open()
        t.switch_profile("self")
        t.unlock_tool("gc")
        t.lock_tool("commit")
        # Should not raise
