"""Tests for @t.runtime.tools.tool decorator and callable_to_tool conversion."""

from __future__ import annotations

from typing import Literal, Optional

import pytest
from pydantic import BaseModel

from tract import Tract
from tract.toolkit.callables import callable_to_tool, _parse_param_docs
from tract.toolkit.models import ToolDefinition


class SearchParams(BaseModel):
    """Test model for Pydantic integration."""
    query: str
    limit: int = 10


# ---------------------------------------------------------------------------
# callable_to_tool unit tests
# ---------------------------------------------------------------------------


class TestCallableToTool:
    """Tests for callable_to_tool() conversion."""

    def test_basic_function(self):
        def greet(name: str) -> str:
            """Say hello to someone."""
            return f"Hello, {name}!"

        td = callable_to_tool(greet)
        assert isinstance(td, ToolDefinition)
        assert td.name == "greet"
        assert td.description == "Say hello to someone."
        assert td.parameters["properties"]["name"]["type"] == "string"
        assert td.parameters["required"] == ["name"]

    def test_multiple_params(self):
        def search(query: str, limit: int = 10, fuzzy: bool = False) -> str:
            """Search for items."""
            return ""

        td = callable_to_tool(search)
        props = td.parameters["properties"]
        assert props["query"]["type"] == "string"
        assert props["limit"]["type"] == "integer"
        assert props["limit"]["default"] == 10
        assert props["fuzzy"]["type"] == "boolean"
        assert props["fuzzy"]["default"] is False
        assert td.parameters["required"] == ["query"]

    def test_name_override(self):
        def my_func(x: str) -> str:
            """Original."""
            return x

        td = callable_to_tool(my_func, name="custom_name")
        assert td.name == "custom_name"

    def test_description_override(self):
        def my_func(x: str) -> str:
            """Original description."""
            return x

        td = callable_to_tool(my_func, description="Custom description")
        assert td.description == "Custom description"

    def test_no_docstring_uses_name(self):
        def mystery(x: str) -> str:
            return x

        td = callable_to_tool(mystery)
        assert td.description == "mystery"

    def test_multiline_docstring_uses_first_line(self):
        def documented(x: str) -> str:
            """First line summary.

            Extended description here.
            More details.
            """
            return x

        td = callable_to_tool(documented)
        assert td.description == "First line summary."

    def test_no_type_hints_defaults_to_string(self):
        def untyped(x) -> str:
            """Untyped params."""
            return str(x)

        td = callable_to_tool(untyped)
        assert td.parameters["properties"]["x"]["type"] == "string"

    def test_all_types(self):
        def typed(s: str, i: int, f: float, b: bool, d: dict, l: list) -> str:
            """All types."""
            return ""

        td = callable_to_tool(typed)
        props = td.parameters["properties"]
        assert props["s"]["type"] == "string"
        assert props["i"]["type"] == "integer"
        assert props["f"]["type"] == "number"
        assert props["b"]["type"] == "boolean"
        assert props["d"]["type"] == "object"
        assert props["l"]["type"] == "array"

    def test_list_of_str(self):
        def search(tags: list[str]) -> str:
            """Search by tags."""
            return ""

        td = callable_to_tool(search)
        prop = td.parameters["properties"]["tags"]
        assert prop["type"] == "array"
        assert prop["items"] == {"type": "string"}

    def test_list_of_int(self):
        def pick(ids: list[int]) -> str:
            """Pick by IDs."""
            return ""

        td = callable_to_tool(pick)
        prop = td.parameters["properties"]["ids"]
        assert prop["type"] == "array"
        assert prop["items"] == {"type": "integer"}

    def test_bare_list_no_items(self):
        def process(data: list) -> str:
            """Process data."""
            return ""

        td = callable_to_tool(process)
        assert td.parameters["properties"]["data"]["type"] == "array"
        assert "items" not in td.parameters["properties"]["data"]

    def test_optional_str(self):
        def greet(name: str, title: Optional[str] = None) -> str:
            """Greet someone."""
            return ""

        td = callable_to_tool(greet)
        # Optional[str] should resolve to string (the inner type)
        assert td.parameters["properties"]["title"]["type"] == "string"

    def test_union_none(self):
        def fetch(url: str, timeout: int | None = None) -> str:
            """Fetch URL."""
            return ""

        td = callable_to_tool(fetch)
        assert td.parameters["properties"]["timeout"]["type"] == "integer"

    def test_literal_strings(self):
        def sort(order: Literal["asc", "desc"] = "asc") -> str:
            """Sort items."""
            return ""

        td = callable_to_tool(sort)
        prop = td.parameters["properties"]["order"]
        assert prop["type"] == "string"
        assert prop["enum"] == ["asc", "desc"]

    def test_literal_ints(self):
        def pick(priority: Literal[1, 2, 3]) -> str:
            """Pick priority."""
            return ""

        td = callable_to_tool(pick)
        prop = td.parameters["properties"]["priority"]
        assert prop["type"] == "integer"
        assert prop["enum"] == [1, 2, 3]

    def test_pydantic_model(self):
        def search(params: SearchParams) -> str:
            """Search with structured params."""
            return ""

        td = callable_to_tool(search)
        prop = td.parameters["properties"]["params"]
        # Should be a full schema dict with properties
        assert prop["type"] == "object"
        assert "query" in prop["properties"]
        assert prop["properties"]["query"]["type"] == "string"
        assert prop["properties"]["limit"]["type"] == "integer"

    def test_dict_with_types(self):
        def process(data: dict[str, int]) -> str:
            """Process data."""
            return ""

        td = callable_to_tool(process)
        assert td.parameters["properties"]["data"]["type"] == "object"

    def test_handler_is_callable(self):
        def adder(a: int, b: int) -> str:
            return str(a + b)

        td = callable_to_tool(adder)
        assert td.handler(a=2, b=3) == "5"

    def test_not_callable_raises(self):
        with pytest.raises(TypeError, match="Expected a callable"):
            callable_to_tool("not a function")

    def test_to_openai_format(self):
        def ping(host: str) -> str:
            """Ping a host."""
            return ""

        td = callable_to_tool(ping)
        openai = td.to_openai()
        assert openai["type"] == "function"
        assert openai["function"]["name"] == "ping"
        assert openai["function"]["description"] == "Ping a host."
        assert "host" in openai["function"]["parameters"]["properties"]

    def test_to_anthropic_format(self):
        def ping(host: str) -> str:
            """Ping a host."""
            return ""

        td = callable_to_tool(ping)
        anthropic = td.to_anthropic()
        assert anthropic["name"] == "ping"
        assert "host" in anthropic["input_schema"]["properties"]

    def test_google_style_param_docs(self):
        def search(query: str, limit: int = 10) -> str:
            """Search for items.

            Args:
                query: The search query string.
                limit: Maximum results to return.
            """
            return ""

        td = callable_to_tool(search)
        props = td.parameters["properties"]
        assert props["query"].get("description") == "The search query string."
        assert props["limit"].get("description") == "Maximum results to return."


class TestParseParamDocs:
    """Tests for _parse_param_docs helper."""

    def test_empty(self):
        assert _parse_param_docs("") == {}

    def test_no_args_section(self):
        assert _parse_param_docs("Just a description.") == {}

    def test_google_style(self):
        doc = """Do something.

        Args:
            x: The x value.
            y: The y value.
        """
        result = _parse_param_docs(doc)
        assert result["x"] == "The x value."
        assert result["y"] == "The y value."

    def test_typed_params(self):
        doc = """Do something.

        Args:
            x (str): The x value.
            y (int): The y value.
        """
        result = _parse_param_docs(doc)
        assert result["x"] == "The x value."
        assert result["y"] == "The y value."

    def test_multiline_description(self):
        doc = """Do something.

        Args:
            x: First line of x description.
                Continuation of x description.
            y: Y value.
        """
        result = _parse_param_docs(doc)
        assert "First line" in result["x"]
        assert "Continuation" in result["x"]


# ---------------------------------------------------------------------------
# @t.runtime.tools.tool decorator integration tests
# ---------------------------------------------------------------------------


class TestToolDecorator:
    """Tests for the @t.runtime.tools.tool decorator on Tract instances."""

    def test_basic_registration(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def greet(name: str) -> str:
                """Say hello."""
                return f"Hello, {name}!"

            assert "greet" in t.runtime.tools.custom_tools
            assert t.runtime.tools.custom_tools["greet"].name == "greet"

    def test_with_name_override(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool(name="say_hi")
            def greet(name: str) -> str:
                """Say hello."""
                return f"Hello, {name}!"

            assert "say_hi" in t.runtime.tools.custom_tools
            assert "greet" not in t.runtime.tools.custom_tools

    def test_with_description_override(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool(description="Custom desc")
            def greet(name: str) -> str:
                """Original desc."""
                return f"Hello, {name}!"

            assert t.runtime.tools.custom_tools["greet"].description == "Custom desc"

    def test_function_unchanged(self):
        """The original function should be returned unmodified."""
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def add(a: int, b: int) -> str:
                """Add two numbers."""
                return str(a + b)

            # Function still works normally
            assert add(2, 3) == "5"

    def test_remove_tool(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def temp(x: str) -> str:
                """Temporary."""
                return x

            assert "temp" in t.runtime.tools.custom_tools
            t.runtime.tools.remove_tool("temp")
            assert "temp" not in t.runtime.tools.custom_tools

    def test_remove_nonexistent_raises(self):
        with Tract.open(":memory:") as t:
            with pytest.raises(KeyError, match="No custom tool 'nope'"):
                t.runtime.tools.remove_tool("nope")

    def test_multiple_tools(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def tool_a(x: str) -> str:
                """Tool A."""
                return x

            @t.runtime.tools.tool
            def tool_b(y: int) -> str:
                """Tool B."""
                return str(y)

            assert len(t.runtime.tools.custom_tools) == 2
            assert "tool_a" in t.runtime.tools.custom_tools
            assert "tool_b" in t.runtime.tools.custom_tools

    def test_appears_in_as_tools(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def my_search(query: str) -> str:
                """Search things."""
                return ""

            tools = t.runtime.tools.as_tools(profile="full", format="openai")
            names = [td["function"]["name"] for td in tools]
            assert "my_search" in names

    def test_appears_in_as_tools_anthropic(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def my_search(query: str) -> str:
                """Search things."""
                return ""

            tools = t.runtime.tools.as_tools(profile="full", format="anthropic")
            names = [td["name"] for td in tools]
            assert "my_search" in names

    def test_tool_names_filter_includes_custom(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def my_calc(expr: str) -> str:
                """Calculate."""
                return ""

            tools = t.runtime.tools.as_tools(
                profile="full",
                tool_names=["commit", "my_calc"],
                format="openai",
            )
            names = [td["function"]["name"] for td in tools]
            assert "my_calc" in names
            assert "commit" in names
            # Should not include other tools
            assert "status" not in names

    def test_tool_names_filter_excludes_custom(self):
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def excluded(x: str) -> str:
                """Excluded tool."""
                return ""

            tools = t.runtime.tools.as_tools(
                profile="full",
                tool_names=["commit"],
                format="openai",
            )
            names = [td["function"]["name"] for td in tools]
            assert "excluded" not in names

    def test_custom_tools_property_is_copy(self):
        """custom_tools should return a copy, not the internal dict."""
        with Tract.open(":memory:") as t:
            @t.runtime.tools.tool
            def my_tool(x: str) -> str:
                """Tool."""
                return x

            ct = t.runtime.tools.custom_tools
            ct.clear()  # mutating the copy
            assert "my_tool" in t.runtime.tools.custom_tools  # internal still has it

    def test_imperative_registration(self):
        """tool() can be called imperatively, not just as decorator."""
        with Tract.open(":memory:") as t:
            def my_func(x: str) -> str:
                """My function."""
                return x

            t.runtime.tools.tool(my_func)
            assert "my_func" in t.runtime.tools.custom_tools
