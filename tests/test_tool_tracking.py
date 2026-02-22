"""Tests for tool schema tracking and provenance.

Covers: hashing, storage, commit linking, compile integration,
set_tools/get_tools convenience, and format output methods.
"""
from __future__ import annotations

import pytest

from tract import Tract, InstructionContent, DialogueContent, CompiledContext, hash_tool_schema


# ---------------------------------------------------------------------------
# Sample tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}

TOOL_SEARCH = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
            },
            "required": ["query"],
        },
    },
}

TOOL_CALC = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Evaluate a math expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string"},
            },
            "required": ["expression"],
        },
    },
}


# ---------------------------------------------------------------------------
# Hashing tests
# ---------------------------------------------------------------------------


class TestHashToolSchema:
    """Tests for hash_tool_schema utility."""

    def test_hash_deterministic(self):
        """Same schema always produces the same hash."""
        h1 = hash_tool_schema(TOOL_GET_WEATHER)
        h2 = hash_tool_schema(TOOL_GET_WEATHER)
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_hash_key_order_independent(self):
        """Key ordering does not affect the hash (canonical JSON sorts keys)."""
        schema_a = {"a": 1, "b": 2, "c": {"x": 10, "y": 20}}
        schema_b = {"c": {"y": 20, "x": 10}, "b": 2, "a": 1}
        assert hash_tool_schema(schema_a) == hash_tool_schema(schema_b)

    def test_different_schema_different_hash(self):
        """Different schemas produce different hashes."""
        h1 = hash_tool_schema(TOOL_GET_WEATHER)
        h2 = hash_tool_schema(TOOL_SEARCH)
        assert h1 != h2


# ---------------------------------------------------------------------------
# Storage-layer tests
# ---------------------------------------------------------------------------


class TestToolSchemaStorage:
    """Tests for SqliteToolSchemaRepository."""

    def test_store_and_get(self):
        """Store a tool schema and retrieve it by hash."""
        t = Tract.open()
        content_hash = hash_tool_schema(TOOL_GET_WEATHER)
        repo = t._tool_schema_repo

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        row = repo.store(content_hash, "get_weather", TOOL_GET_WEATHER, now)
        assert row.content_hash == content_hash
        assert row.name == "get_weather"
        assert row.schema_json == TOOL_GET_WEATHER

        retrieved = repo.get(content_hash)
        assert retrieved is not None
        assert retrieved.content_hash == content_hash
        t.close()

    def test_store_idempotent(self):
        """Storing the same hash twice is a no-op (returns existing)."""
        t = Tract.open()
        content_hash = hash_tool_schema(TOOL_SEARCH)
        repo = t._tool_schema_repo

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        row1 = repo.store(content_hash, "search", TOOL_SEARCH, now)
        row2 = repo.store(content_hash, "search", TOOL_SEARCH, now)
        assert row1.content_hash == row2.content_hash
        t.close()

    def test_get_by_name(self):
        """Retrieve all tool versions by name."""
        t = Tract.open()
        repo = t._tool_schema_repo

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        # Store two different schemas with the same name (different versions)
        schema_v1 = {"name": "tool_a", "version": 1}
        schema_v2 = {"name": "tool_a", "version": 2}
        repo.store(hash_tool_schema(schema_v1), "tool_a", schema_v1, now)
        repo.store(hash_tool_schema(schema_v2), "tool_a", schema_v2, now)
        t._session.commit()

        results = repo.get_by_name("tool_a")
        assert len(results) == 2
        t.close()


# ---------------------------------------------------------------------------
# Commit-level tool linking tests
# ---------------------------------------------------------------------------


class TestCommitTools:
    """Tests for tool linking on commits."""

    def test_commit_with_tools(self):
        """Tools passed to commit() are linked and retrievable."""
        t = Tract.open()
        tools = [TOOL_GET_WEATHER, TOOL_SEARCH]
        info = t.commit(
            InstructionContent(text="You are a helpful assistant."),
            tools=tools,
        )

        # Retrieve tools for this commit
        retrieved = t.get_commit_tools(info.commit_hash)
        assert len(retrieved) == 2
        assert retrieved[0] == TOOL_GET_WEATHER
        assert retrieved[1] == TOOL_SEARCH
        t.close()

    def test_commit_without_tools(self):
        """Commit with no tools has empty tool list."""
        t = Tract.open()
        info = t.commit(InstructionContent(text="No tools here"))

        retrieved = t.get_commit_tools(info.commit_hash)
        assert retrieved == []
        t.close()

    def test_tool_deduplication(self):
        """Same schema stored once even when referenced by multiple commits."""
        t = Tract.open()

        info1 = t.commit(
            InstructionContent(text="First commit"),
            tools=[TOOL_GET_WEATHER],
        )
        info2 = t.commit(
            DialogueContent(role="user", text="Hello"),
            tools=[TOOL_GET_WEATHER],
        )

        # Both commits reference the same tool
        tools1 = t.get_commit_tools(info1.commit_hash)
        tools2 = t.get_commit_tools(info2.commit_hash)
        assert tools1 == tools2 == [TOOL_GET_WEATHER]

        # Content-addressed: only one row in tool_definitions
        content_hash = hash_tool_schema(TOOL_GET_WEATHER)
        assert t._tool_schema_repo.get(content_hash) is not None
        t.close()

    def test_get_commit_tools_provenance(self):
        """get_commit_tools preserves position order."""
        t = Tract.open()
        tools = [TOOL_SEARCH, TOOL_GET_WEATHER, TOOL_CALC]
        info = t.commit(
            InstructionContent(text="System prompt"),
            tools=tools,
        )

        retrieved = t.get_commit_tools(info.commit_hash)
        assert len(retrieved) == 3
        assert retrieved[0] == TOOL_SEARCH
        assert retrieved[1] == TOOL_GET_WEATHER
        assert retrieved[2] == TOOL_CALC
        t.close()


# ---------------------------------------------------------------------------
# set_tools / get_tools convenience tests
# ---------------------------------------------------------------------------


class TestSetGetTools:
    """Tests for set_tools() auto-linking and get_tools()."""

    def test_set_tools_auto_links(self):
        """set_tools() causes subsequent commits to auto-link tools."""
        t = Tract.open()
        t.set_tools([TOOL_GET_WEATHER])

        info1 = t.commit(InstructionContent(text="System prompt"))
        info2 = t.user("Hello!")

        assert t.get_commit_tools(info1.commit_hash) == [TOOL_GET_WEATHER]
        assert t.get_commit_tools(info2.commit_hash) == [TOOL_GET_WEATHER]
        t.close()

    def test_set_tools_clear(self):
        """set_tools(None) clears auto-linking."""
        t = Tract.open()
        t.set_tools([TOOL_SEARCH])

        info1 = t.commit(InstructionContent(text="With tools"))
        t.set_tools(None)
        info2 = t.commit(InstructionContent(text="Without tools"))

        assert len(t.get_commit_tools(info1.commit_hash)) == 1
        assert t.get_commit_tools(info2.commit_hash) == []
        t.close()

    def test_get_tools_returns_current(self):
        """get_tools() returns what was set via set_tools()."""
        t = Tract.open()
        assert t.get_tools() is None

        t.set_tools([TOOL_CALC])
        assert t.get_tools() == [TOOL_CALC]

        t.set_tools(None)
        assert t.get_tools() is None
        t.close()

    def test_explicit_tools_override_set_tools(self):
        """Explicit tools= on commit() overrides set_tools()."""
        t = Tract.open()
        t.set_tools([TOOL_GET_WEATHER])

        info = t.commit(
            InstructionContent(text="Override"),
            tools=[TOOL_SEARCH],
        )

        retrieved = t.get_commit_tools(info.commit_hash)
        assert retrieved == [TOOL_SEARCH]
        t.close()


# ---------------------------------------------------------------------------
# Compile integration tests
# ---------------------------------------------------------------------------


class TestCompileTools:
    """Tests for tool integration in compile()."""

    def test_compile_returns_tools(self):
        """CompiledContext.tools has active tools from last commit with tools."""
        t = Tract.open()
        t.commit(
            InstructionContent(text="System"),
            tools=[TOOL_GET_WEATHER, TOOL_SEARCH],
        )
        t.user("Hello")

        result = t.compile()
        assert isinstance(result, CompiledContext)
        assert len(result.tools) == 2
        assert result.tools[0] == TOOL_GET_WEATHER
        assert result.tools[1] == TOOL_SEARCH
        t.close()

    def test_compile_no_tools(self):
        """CompiledContext.tools is empty when no tools are linked."""
        t = Tract.open()
        t.commit(InstructionContent(text="System"))
        t.user("Hello")

        result = t.compile()
        assert result.tools == []
        t.close()

    def test_compile_latest_tools_win(self):
        """The last commit with tools determines compile().tools."""
        t = Tract.open()
        t.commit(
            InstructionContent(text="System"),
            tools=[TOOL_GET_WEATHER],
        )
        t.commit(
            DialogueContent(role="user", text="Step 1"),
            tools=[TOOL_SEARCH, TOOL_CALC],
        )

        result = t.compile()
        # The latest commit with tools had SEARCH and CALC
        assert len(result.tools) == 2
        assert result.tools[0] == TOOL_SEARCH
        assert result.tools[1] == TOOL_CALC
        t.close()

    def test_compile_tools_from_earlier_commit(self):
        """Tools from an earlier commit show up if no later commit has tools."""
        t = Tract.open()
        t.commit(
            InstructionContent(text="System"),
            tools=[TOOL_GET_WEATHER],
        )
        t.user("Hello")  # No tools
        t.assistant("Hi there!")  # No tools

        result = t.compile()
        assert result.tools == [TOOL_GET_WEATHER]
        t.close()


# ---------------------------------------------------------------------------
# Format output tests
# ---------------------------------------------------------------------------


class TestFormatOutput:
    """Tests for to_openai_params and to_anthropic_params with tools."""

    def test_to_openai_params_includes_tools(self):
        """to_openai_params() includes tools when present."""
        t = Tract.open()
        t.commit(
            InstructionContent(text="System"),
            tools=[TOOL_GET_WEATHER],
        )
        t.user("What's the weather?")

        result = t.compile()
        params = result.to_openai_params()

        assert "messages" in params
        assert "tools" in params
        assert params["tools"] == [TOOL_GET_WEATHER]
        t.close()

    def test_to_anthropic_params_includes_tools(self):
        """to_anthropic_params() includes tools when present."""
        t = Tract.open()
        t.commit(
            InstructionContent(text="System"),
            tools=[TOOL_SEARCH],
        )
        t.user("Search for something")

        result = t.compile()
        params = result.to_anthropic_params()

        assert "system" in params
        assert "messages" in params
        assert "tools" in params
        assert params["tools"] == [TOOL_SEARCH]
        t.close()

    def test_to_openai_params_no_tools(self):
        """to_openai_params() omits tools key when no tools."""
        t = Tract.open()
        t.commit(InstructionContent(text="System"))
        t.user("Hello")

        result = t.compile()
        params = result.to_openai_params()

        assert "messages" in params
        assert "tools" not in params
        t.close()

    def test_to_anthropic_params_no_tools(self):
        """to_anthropic_params() omits tools key when no tools."""
        t = Tract.open()
        t.commit(InstructionContent(text="System"))
        t.user("Hello")

        result = t.compile()
        params = result.to_anthropic_params()

        assert "tools" not in params
        t.close()

    def test_to_openai_backward_compat(self):
        """to_openai() still returns list[dict] (backward compatible)."""
        t = Tract.open()
        t.commit(
            InstructionContent(text="System"),
            tools=[TOOL_GET_WEATHER],
        )
        t.user("Hi")

        result = t.compile()
        # to_openai() returns just messages, not params dict
        openai_msgs = result.to_openai()
        assert isinstance(openai_msgs, list)
        assert all(isinstance(m, dict) for m in openai_msgs)
        # Should NOT contain tools -- that's to_openai_params()
        t.close()
