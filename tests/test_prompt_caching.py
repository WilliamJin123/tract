"""Tests for Anthropic prompt-caching support in CompiledContext.

Covers: cache_control parameter on to_anthropic() and to_anthropic_params(),
system prompt conversion, stable/volatile boundary marking, priority-based
boundary detection, backward compatibility, and edge cases.
"""

from tract.protocols import CompiledContext, Message, ToolCall


# -------------------------------------------------------------------
# Backward compatibility
# -------------------------------------------------------------------

class TestCacheControlDisabledByDefault:
    """cache_control=False (default) must produce no cache_control markers."""

    def test_no_markers_with_default(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="You are helpful."),
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi there!"),
            ],
            commit_count=3,
        )
        result = ctx.to_anthropic()
        # System is a plain string
        assert isinstance(result["system"], str)
        # No cache_control on any message
        for msg in result["messages"]:
            content = msg["content"]
            if isinstance(content, str):
                assert "cache_control" not in msg
            elif isinstance(content, list):
                for block in content:
                    assert "cache_control" not in block

    def test_no_markers_explicit_false(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="System prompt."),
                Message(role="user", content="Q"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic(cache_control=False)
        assert isinstance(result["system"], str)


# -------------------------------------------------------------------
# System prompt caching
# -------------------------------------------------------------------

class TestSystemPromptCaching:
    """cache_control=True converts system to block list with cache_control."""

    def test_system_string_becomes_block_list(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hi"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic(cache_control=True)
        system = result["system"]
        assert isinstance(system, list), f"Expected list, got {type(system)}"
        assert len(system) == 1
        block = system[0]
        assert block["type"] == "text"
        assert block["text"] == "You are a helpful assistant."
        assert block["cache_control"] == {"type": "ephemeral"}

    def test_multi_system_messages_joined(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Part 1."),
                Message(role="system", content="Part 2."),
                Message(role="user", content="Go"),
            ],
            commit_count=3,
        )
        result = ctx.to_anthropic(cache_control=True)
        system = result["system"]
        assert isinstance(system, list)
        assert system[0]["text"] == "Part 1.\n\nPart 2."
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_no_system_stays_none(self):
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="Hi"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic(cache_control=True)
        assert result["system"] is None


# -------------------------------------------------------------------
# Message boundary caching
# -------------------------------------------------------------------

class TestMessageBoundaryCaching:
    """cache_control=True adds a marker at the stable/volatile boundary."""

    def test_midpoint_boundary_default(self):
        """Without priorities, boundary is at midpoint of messages."""
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="msg1"),
                Message(role="assistant", content="resp1"),
                Message(role="user", content="msg2"),
                Message(role="assistant", content="resp2"),
            ],
            commit_count=4,
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # 4 messages -> midpoint = 4//2 - 1 = 1, so index 1 gets marker
        boundary_msg = msgs[1]
        content = boundary_msg["content"]
        assert isinstance(content, list), "Boundary message should be block list"
        assert content[-1]["cache_control"] == {"type": "ephemeral"}

        # Other messages should NOT have cache_control
        for i, msg in enumerate(msgs):
            if i == 1:
                continue
            c = msg["content"]
            if isinstance(c, str):
                pass  # no cache_control possible on raw string
            elif isinstance(c, list):
                for block in c:
                    assert "cache_control" not in block, (
                        f"Message {i} should not have cache_control"
                    )

    def test_single_message_gets_marker(self):
        """With only one message, it gets the marker at index 0."""
        ctx = CompiledContext(
            messages=[Message(role="user", content="only message")],
            commit_count=1,
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        assert len(msgs) == 1
        content = msgs[0]["content"]
        assert isinstance(content, list)
        assert content[0]["cache_control"] == {"type": "ephemeral"}

    def test_two_messages_boundary(self):
        """With two messages, boundary is at index 0."""
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="question"),
                Message(role="assistant", content="answer"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # 2 messages -> midpoint = 2//2 - 1 = 0
        content_0 = msgs[0]["content"]
        assert isinstance(content_0, list)
        assert content_0[-1]["cache_control"] == {"type": "ephemeral"}


# -------------------------------------------------------------------
# Priority-based boundary
# -------------------------------------------------------------------

class TestPriorityBoundary:
    """Priorities influence where the boundary marker is placed."""

    def test_pinned_priority_marks_boundary(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="msg1"),
                Message(role="assistant", content="resp1"),
                Message(role="user", content="msg2"),
                Message(role="assistant", content="resp2"),
            ],
            commit_count=5,
            priorities=["pinned", "normal", "normal", "normal", "normal"],
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # Priority index 0 is "pinned" -> maps to merged index 0
        # (system is extracted, so merged[0] is the first non-system msg)
        content_0 = msgs[0]["content"]
        assert isinstance(content_0, list)
        assert content_0[-1]["cache_control"] == {"type": "ephemeral"}

    def test_important_priority_marks_boundary(self):
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="msg1"),
                Message(role="assistant", content="resp1"),
                Message(role="user", content="msg2"),
                Message(role="assistant", content="resp2"),
            ],
            commit_count=4,
            priorities=["normal", "important", "normal", "normal"],
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # Last important priority is at index 1 -> merged[1]
        content_1 = msgs[1]["content"]
        assert isinstance(content_1, list)
        assert content_1[-1]["cache_control"] == {"type": "ephemeral"}

    def test_last_pinned_wins(self):
        """When multiple PINNED priorities exist, the last one is the boundary."""
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="u2"),
                Message(role="assistant", content="a2"),
            ],
            commit_count=4,
            priorities=["pinned", "normal", "pinned", "normal"],
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # Last pinned is index 2 -> merged[2]
        content_2 = msgs[2]["content"]
        assert isinstance(content_2, list)
        assert content_2[-1]["cache_control"] == {"type": "ephemeral"}

    def test_all_normal_falls_back_to_midpoint(self):
        """When all priorities are NORMAL, fall back to midpoint heuristic."""
        ctx = CompiledContext(
            messages=[
                Message(role="user", content="u1"),
                Message(role="assistant", content="a1"),
                Message(role="user", content="u2"),
                Message(role="assistant", content="a2"),
            ],
            commit_count=4,
            priorities=["normal", "normal", "normal", "normal"],
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # Midpoint = 4//2 - 1 = 1
        content_1 = msgs[1]["content"]
        assert isinstance(content_1, list)
        assert content_1[-1]["cache_control"] == {"type": "ephemeral"}


# -------------------------------------------------------------------
# Edge cases
# -------------------------------------------------------------------

class TestCacheControlEdgeCases:

    def test_empty_messages(self):
        """cache_control=True with no messages returns normally."""
        ctx = CompiledContext(messages=[], commit_count=0)
        result = ctx.to_anthropic(cache_control=True)
        assert result["system"] is None
        assert result["messages"] == []

    def test_system_only_no_messages(self):
        """System prompt with no other messages still gets cached."""
        ctx = CompiledContext(
            messages=[Message(role="system", content="You are a bot.")],
            commit_count=1,
        )
        result = ctx.to_anthropic(cache_control=True)
        system = result["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert result["messages"] == []

    def test_tool_calls_in_messages(self):
        """cache_control works with assistant tool_call messages."""
        tc = ToolCall(id="call_1", name="search", arguments={"q": "test"})
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="System."),
                Message(role="user", content="Find something."),
                Message(role="assistant", content="Searching...", tool_calls=[tc]),
                Message(role="tool", content="Result here", tool_call_id="call_1"),
                Message(role="user", content="Thanks"),
                Message(role="assistant", content="You're welcome."),
            ],
            commit_count=6,
        )
        result = ctx.to_anthropic(cache_control=True)

        # System is block list
        assert isinstance(result["system"], list)
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}

        # Messages should have exactly one cache_control marker
        msgs = result["messages"]
        cache_count = 0
        for msg in msgs:
            content = msg["content"]
            if isinstance(content, list):
                for block in content:
                    if "cache_control" in block:
                        cache_count += 1
            # string content can't have cache_control
        assert cache_count == 1, f"Expected 1 message marker, got {cache_count}"

    def test_block_content_gets_marker_on_last_block(self):
        """When boundary message has block-list content, last block gets marker."""
        tc = ToolCall(id="call_1", name="eval", arguments={})
        ctx = CompiledContext(
            messages=[
                Message(role="assistant", content="Let me check.", tool_calls=[tc]),
                Message(role="tool", content="42", tool_call_id="call_1"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic(cache_control=True)
        msgs = result["messages"]
        # Boundary at index 0 (midpoint: 2//2-1 = 0)
        boundary = msgs[0]
        blocks = boundary["content"]
        assert isinstance(blocks, list)
        # The last block (tool_use) should have cache_control
        assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
        # Earlier blocks should NOT
        if len(blocks) > 1:
            assert "cache_control" not in blocks[0]


# -------------------------------------------------------------------
# to_anthropic_params passthrough
# -------------------------------------------------------------------

class TestToAnthropicParamsPassthrough:
    """to_anthropic_params passes cache_control through to to_anthropic."""

    def test_params_default_no_cache(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Sys"),
                Message(role="user", content="Hi"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic_params()
        assert isinstance(result["system"], str)

    def test_params_with_cache_control(self):
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="Sys"),
                Message(role="user", content="Hi"),
            ],
            commit_count=2,
        )
        result = ctx.to_anthropic_params(cache_control=True)
        system = result["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    def test_params_with_tools_and_cache(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        ctx = CompiledContext(
            messages=[
                Message(role="system", content="You use tools."),
                Message(role="user", content="Search for X"),
            ],
            commit_count=2,
            tools=tools,
        )
        result = ctx.to_anthropic_params(cache_control=True)
        # System cached
        assert isinstance(result["system"], list)
        assert result["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Tools still present and converted
        assert "tools" in result
        assert result["tools"][0]["name"] == "search"
        assert "input_schema" in result["tools"][0]
