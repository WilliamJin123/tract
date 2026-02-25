"""Tests for tool query API: find_tool_results, find_tool_calls, find_tool_turns."""

from __future__ import annotations

import pytest

from tract import Tract, CommitInfo
from tract.protocols import ToolTurn


class TestFindToolResults:
    """Tests for Tract.find_tool_results()."""

    def test_find_all(self):
        """Finds all tool result commits."""
        with Tract.open() as t:
            t.user("hello")
            t.tool_result("c1", "grep", "found stuff")
            t.tool_result("c2", "read_file", "file content")
            results = t.find_tool_results()
            assert len(results) == 2
            names = [r.metadata["name"] for r in results]
            assert names == ["grep", "read_file"]

    def test_find_by_name(self):
        """Filters by tool name."""
        with Tract.open() as t:
            t.tool_result("c1", "grep", "found stuff")
            t.tool_result("c2", "read_file", "file content")
            t.tool_result("c3", "grep", "more stuff")
            results = t.find_tool_results(name="grep")
            assert len(results) == 2
            assert all(r.metadata["name"] == "grep" for r in results)

    def test_find_after(self):
        """Filters by position (after a hash)."""
        with Tract.open() as t:
            r1 = t.tool_result("c1", "grep", "first")
            r2 = t.tool_result("c2", "grep", "second")
            r3 = t.tool_result("c3", "grep", "third")
            results = t.find_tool_results(after=r1.commit_hash)
            assert len(results) == 2
            assert results[0].commit_hash == r2.commit_hash

    def test_find_empty(self):
        """No tool results returns empty list."""
        with Tract.open() as t:
            t.user("hello")
            t.assistant("hi")
            results = t.find_tool_results()
            assert results == []


class TestFindToolCalls:
    """Tests for Tract.find_tool_calls()."""

    def test_find_all(self):
        """Finds assistant commits with tool_calls metadata."""
        with Tract.open() as t:
            t.assistant("thinking...", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            t.tool_result("c1", "grep", "found it")
            t.assistant("done")
            calls = t.find_tool_calls()
            assert len(calls) == 1

    def test_find_by_name(self):
        """Filters to calls containing a specific tool name."""
        with Tract.open() as t:
            t.assistant("", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            t.assistant("", metadata={"tool_calls": [{"id": "c2", "name": "read_file", "arguments": {}}]})
            calls = t.find_tool_calls(name="grep")
            assert len(calls) == 1

    def test_find_empty(self):
        """No tool calls returns empty list."""
        with Tract.open() as t:
            t.user("hello")
            calls = t.find_tool_calls()
            assert calls == []


class TestFindToolTurns:
    """Tests for Tract.find_tool_turns()."""

    def test_groups_correctly(self):
        """Call + results are paired correctly."""
        with Tract.open() as t:
            t.assistant("", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            t.tool_result("c1", "grep", "found it")
            turns = t.find_tool_turns()
            assert len(turns) == 1
            assert len(turns[0].results) == 1
            assert turns[0].tool_names == ["grep"]

    def test_multi_tool(self):
        """Assistant calls 2 tools, both results grouped."""
        with Tract.open() as t:
            t.assistant("", metadata={"tool_calls": [
                {"id": "c1", "name": "grep", "arguments": {}},
                {"id": "c2", "name": "read_file", "arguments": {}},
            ]})
            t.tool_result("c1", "grep", "grep result")
            t.tool_result("c2", "read_file", "file content")
            turns = t.find_tool_turns()
            assert len(turns) == 1
            assert len(turns[0].results) == 2
            assert set(turns[0].tool_names) == {"grep", "read_file"}

    def test_filter_by_name(self):
        """Filters to turns containing specific tool."""
        with Tract.open() as t:
            t.assistant("", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            t.tool_result("c1", "grep", "result")
            t.assistant("", metadata={"tool_calls": [{"id": "c2", "name": "read_file", "arguments": {}}]})
            t.tool_result("c2", "read_file", "content")
            turns = t.find_tool_turns(name="grep")
            assert len(turns) == 1

    def test_empty(self):
        """No tool calls returns empty list."""
        with Tract.open() as t:
            t.user("hello")
            turns = t.find_tool_turns()
            assert turns == []

    def test_all_hashes_property(self):
        """all_hashes includes call + result hashes."""
        with Tract.open() as t:
            call_ci = t.assistant("", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            result_ci = t.tool_result("c1", "grep", "found it")
            turns = t.find_tool_turns()
            assert turns[0].all_hashes == [call_ci.commit_hash, result_ci.commit_hash]

    def test_result_hashes_property(self):
        """result_hashes includes only result hashes."""
        with Tract.open() as t:
            t.assistant("", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            result_ci = t.tool_result("c1", "grep", "found it")
            turns = t.find_tool_turns()
            assert turns[0].result_hashes == [result_ci.commit_hash]

    def test_total_tokens_property(self):
        """total_tokens sums call and result tokens."""
        with Tract.open() as t:
            t.assistant("some text for tokens", metadata={"tool_calls": [{"id": "c1", "name": "grep", "arguments": {}}]})
            t.tool_result("c1", "grep", "found it with some token content")
            turns = t.find_tool_turns()
            assert turns[0].total_tokens > 0
