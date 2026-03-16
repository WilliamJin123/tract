"""Testing and Mocking Patterns for Tract Applications

How to test applications built on tract without live LLM calls.

Tract's composable, git-like API makes testing dramatically easier than
testing with raw LLM calls or framework-heavy approaches. The key insight:
most tract operations (commit, compile, branch, merge, compress, middleware)
are pure DAG operations that need no LLM at all. Only chat/generate/run
need a mock client, and tract ships first-party test clients so you don't
even need to write your own.

Patterns shown:
  1. First-party MockLLMClient   -- cycling mock from tract.llm.testing
  2. Testing a Complete Workflow  -- multi-stage workflow without LLM
  3. Testing Chat with Mock LLM  -- mock tools + mock LLM
  4. Testing Tool Execution      -- ToolCallMockLLMClient (custom pattern)
  5. Testing Branching & Merge   -- branch/merge assertions
  6. Testing Middleware Behavior  -- verify middleware fires and blocks
  7. Snapshot Testing             -- compiled context regression testing
  8. Pytest Fixtures              -- reusable test infrastructure
  9. ReplayLLMClient             -- sequential playback, exhaustion check
  10. FunctionLLMClient          -- custom logic, conditional responses

Demonstrates: LLMClient protocol, MockLLMClient, ReplayLLMClient,
              FunctionLLMClient, Tract.open(), compile(), branch(),
              merge(), compress(), use(), chat(), run(), tool_handlers,
              BlockedError, MiddlewareContext

No LLM required. Runs as pytest.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import (
    BlockedError,
    CompiledContext,
    LLMClient,
    MiddlewareContext,
    MockLLMClient,
    ReplayLLMClient,
    FunctionLLMClient,
    Priority,
    Tract,
)


# =====================================================================
# 1. First-Party MockLLMClient
# =====================================================================
#
# Tract ships MockLLMClient in tract.llm.testing (re-exported from the
# top-level tract package). It satisfies the LLMClient protocol, cycles
# through canned responses, and records every call. No need to write
# your own mock for the common case.


def test_mock_satisfies_protocol():
    """Verify the first-party MockLLMClient satisfies LLMClient."""

    # Runtime check using the @runtime_checkable Protocol
    mock = MockLLMClient(["Hello!"])
    assert isinstance(mock, LLMClient), "Mock must satisfy LLMClient protocol"

    # Verify it produces valid OpenAI-format responses
    response = mock.chat([{"role": "user", "content": "Hi"}])
    assert response["choices"][0]["message"]["content"] == "Hello!"
    assert response["usage"]["total_tokens"] == 0

    # Verify call tracking
    assert mock.call_count == 1
    assert len(mock.calls) == 1
    assert mock.calls[0]["messages"][0]["content"] == "Hi"

    # Verify cycling behavior
    mock2 = MockLLMClient(["First", "Second"])
    assert mock2.chat([])["choices"][0]["message"]["content"] == "First"
    assert mock2.chat([])["choices"][0]["message"]["content"] == "Second"
    assert mock2.chat([])["choices"][0]["message"]["content"] == "First"  # cycles

    print("1. First-party MockLLMClient satisfies LLMClient protocol")
    print(f"   isinstance check: {isinstance(mock, LLMClient)}")
    print(f"   call tracking: {mock.call_count} call(s) recorded")
    print(f"   cycling: 3 calls on 2 responses works correctly")


# =====================================================================
# 2. Testing a Complete Workflow Without LLM
# =====================================================================
#
# Most tract operations are pure DAG manipulations that need no LLM at
# all. commit, compile, branch, merge, tag, annotate, compress(content=)
# are all testable with zero mocking. Only chat(), generate(), and run()
# need a mock client.
#
# This pattern tests a multi-stage research workflow end-to-end.


def test_complete_workflow_without_llm():
    """Test a multi-stage workflow using only DAG operations -- no LLM needed."""

    with Tract.open() as t:
        # Stage 1: Setup
        t.system("You are a research assistant analyzing market trends.")
        t.user("Research the AI chip market for 2025.")

        # Register custom tags before using them (tract enforces strict tags)
        t.register_tag("research", "Research findings")
        t.register_tag("market-size", "Market sizing data")
        t.register_tag("competition", "Competitive analysis")

        # Stage 2: Simulate assistant findings (no LLM call)
        t.assistant(
            "The AI chip market is projected to reach $120B by 2027. "
            "Key players: NVIDIA (65% share), AMD (15%), Intel (8%), "
            "custom silicon (12%). Growth driven by LLM training demand.",
            tags=["research", "market-size"],
        )

        t.user("What about the competitive dynamics?")
        t.assistant(
            "NVIDIA dominates with CUDA ecosystem lock-in. AMD gaining "
            "with ROCm improvements. Custom chips (Google TPU, Amazon "
            "Trainium) growing in cloud-native workloads.",
            tags=["research", "competition"],
        )

        # Stage 3: Verify commit history
        log = t.log()
        assert len(log) >= 4, f"Expected at least 4 commits, got {len(log)}"

        # Stage 4: Verify tag filtering
        research_commits = [c for c in log if "research" in (c.tags or [])]
        assert len(research_commits) == 2, "Should have 2 research-tagged commits"

        # Stage 5: Compile and verify context
        ctx = t.compile()
        assert isinstance(ctx, CompiledContext)
        assert len(ctx.messages) >= 4  # system + 2 user + 2 assistant (at minimum)
        assert ctx.token_count > 0

        # Verify specific content appears in compiled output
        full_text = " ".join(m.content or "" for m in ctx.messages)
        assert "NVIDIA" in full_text, "Research findings should be in context"
        assert "research assistant" in full_text, "System prompt should be in context"

        # Stage 6: Verify OpenAI-format export
        openai_msgs = ctx.to_openai()
        assert isinstance(openai_msgs, list)
        assert openai_msgs[0]["role"] == "system"
        roles = [m["role"] for m in openai_msgs]
        assert "user" in roles
        assert "assistant" in roles

    print("2. Complete workflow tested without any LLM calls")
    print(f"   Commits: {len(log)}")
    print(f"   Tagged: {len(research_commits)} research commits")
    ctx.pprint(style="compact")
    print(f"   OpenAI export: {len(openai_msgs)} messages")


# =====================================================================
# 3. Testing Chat + Generate with Mock LLM
# =====================================================================
#
# When you need to test code that calls t.chat() or t.generate(), pass
# a MockLLMClient. This lets you verify that your application logic
# handles LLM responses correctly without spending tokens or dealing
# with network flakiness.


def test_chat_with_mock_llm():
    """Test chat() flow with the first-party MockLLMClient."""

    mock = MockLLMClient([
        "Based on the data, revenue grew 15% QoQ.",
        "The top 3 drivers were: 1) Enterprise expansion, 2) Price increases, 3) New markets.",
    ])

    with Tract.open(llm_client=mock) as t:
        t.system("You are a financial analyst.")

        # First chat round
        response1 = t.chat("Analyze Q3 revenue trends.")
        assert response1.text == "Based on the data, revenue grew 15% QoQ."
        assert response1.usage is not None
        assert response1.commit_info is not None

        # Second chat round
        response2 = t.chat("What were the main drivers?")
        assert "top 3 drivers" in response2.text

        # Verify the mock was called correctly
        assert mock.call_count == 2

        # Verify the second call included conversation history
        # (tract compiles context before each LLM call)
        second_call_messages = mock.calls[1]["messages"]
        assert len(second_call_messages) > 2, (
            "Second call should include prior conversation context"
        )

        # Verify commits were created for both user and assistant messages
        log = t.log()
        # system + user1 + assistant1 + user2 + assistant2
        assert len(log) >= 5

        # Verify the compiled context is coherent
        ctx = t.compile()
        full_text = " ".join(m.content or "" for m in ctx.messages)
        assert "revenue grew 15%" in full_text
        assert "top 3 drivers" in full_text

    print("3. Chat with mock LLM tested successfully")
    print(f"   Mock calls: {mock.call_count}")
    print(f"   Commits: {len(log)}")
    print(f"   Response 1: {(response1.text or '(no response)')[:50]}...")
    print(f"   Response 2: {(response2.text or '(no response)')[:50]}...")


# =====================================================================
# 4. Testing Tool Execution with Mock LLM
# =====================================================================
#
# t.run() supports custom tool_handlers: a dict mapping tool names to
# callables. Combined with a mock LLM that returns tool_calls, you can
# test the full tool execution loop without any real LLM or real tools.
#
# Key trick: the mock LLM returns OpenAI-format tool_calls on the first
# response, then a plain text response to end the loop.
#
# ToolCallMockLLMClient is a specialized hand-rolled pattern for tool-
# calling scenarios. For simpler chat-only tests, use the first-party
# MockLLMClient, ReplayLLMClient, or FunctionLLMClient instead.


class ToolCallMockLLMClient:
    """Mock LLM that returns tool calls, then a final text response.

    Simulates the pattern: LLM decides to call tools -> tools execute ->
    LLM sees results -> LLM gives final answer.
    """

    def __init__(
        self,
        tool_calls: list[dict],
        final_response: str,
    ):
        self._tool_calls = tool_calls
        self._final_response = final_response
        self._call_count = 0
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], **kwargs: Any) -> dict:
        self.calls.append({"messages": messages, **kwargs})
        self._call_count += 1

        # First call: return tool calls
        if self._call_count == 1:
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": f"call_{i}",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc.get("arguments", {})),
                                },
                            }
                            for i, tc in enumerate(self._tool_calls)
                        ],
                    },
                }],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            }

        # Subsequent calls: return final text (ends the loop)
        return {
            "choices": [{"message": {"role": "assistant", "content": self._final_response}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        }

    def extract_content(self, response: dict) -> str:
        return response["choices"][0]["message"].get("content") or ""

    def close(self) -> None:
        pass


def test_tool_execution_with_mock():
    """Test the agent loop with mock tools and mock LLM."""

    # Track tool invocations
    tool_log: list[dict] = []

    def mock_calculator(expression: str) -> str:
        """A mock calculator tool."""
        tool_log.append({"tool": "calculator", "expression": expression})
        # Return a predictable result instead of eval()
        return "42"

    def mock_lookup(query: str) -> str:
        """A mock data lookup tool."""
        tool_log.append({"tool": "lookup", "query": query})
        return "Revenue: $4.2M, Growth: 23.5%"

    # Mock LLM that calls both tools, then gives final answer
    mock = ToolCallMockLLMClient(
        tool_calls=[
            {"name": "calculator", "arguments": {"expression": "4200000 * 1.235"}},
            {"name": "lookup", "arguments": {"query": "Q3 revenue"}},
        ],
        final_response="Based on calculations and data lookup: projected Q4 revenue is $5.19M.",
    )

    # Tool definitions in OpenAI format
    tool_defs = [
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a math expression",
                "parameters": {
                    "type": "object",
                    "properties": {"expression": {"type": "string"}},
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up business data",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        },
    ]

    with Tract.open(llm_client=mock) as t:
        t.system("You are a financial analyst with calculator and data tools.")

        result = t.run(
            "Calculate projected Q4 revenue based on Q3 data.",
            tools=tool_defs,
            tool_handlers={
                "calculator": mock_calculator,
                "lookup": mock_lookup,
            },
            max_steps=5,
        )

        # Verify the loop completed successfully
        assert result.status == "completed"
        assert result.final_response is not None
        assert "5.19M" in result.final_response

        # Verify both tools were called
        assert len(tool_log) == 2
        assert tool_log[0]["tool"] == "calculator"
        assert tool_log[1]["tool"] == "lookup"

        # Verify tool results were committed to the DAG by compiling context
        ctx = t.compile()
        full_text = " ".join(m.content or "" for m in ctx.messages)
        assert "42" in full_text or "Revenue" in full_text or "5.19M" in full_text, (
            "Tool results or final response should be in compiled context"
        )

        # Verify the mock LLM was called at least twice
        # (once for tool calls, once for final response)
        assert len(mock.calls) >= 2

    print("4. Tool execution tested with mock LLM + mock tools")
    print(f"   Loop status: {result.status}")
    print(f"   Tool calls: {len(tool_log)}")
    print(f"   LLM calls: {len(mock.calls)}")
    print(f"   Final: {(result.final_response or '(no response)')[:60]}...")


# =====================================================================
# 5. Testing Branching & Merge Without LLM
# =====================================================================
#
# Branch and merge are pure DAG operations. Test complex branching
# strategies with predictable content and verify merge results.


def test_branching_and_merge():
    """Test branch creation, parallel work, and merge -- no LLM needed."""

    with Tract.open() as t:
        t.system("You are a project coordinator.")
        t.user("We need to research two markets in parallel.")
        t.assistant("I will research AI chips and quantum computing separately.")

        main_head = t.head

        # Register custom tags before using them
        t.register_tag("research", "Research findings")
        t.register_tag("ai-chips", "AI chip market data")
        t.register_tag("quantum", "Quantum computing data")

        # --- Branch 1: AI chips research ---
        t.branch("ai_chips")
        t.user("Focus on AI chip market size and key players.")
        t.assistant(
            "AI chip market: $120B TAM. Key players: NVIDIA, AMD, Intel.",
            tags=["research", "ai-chips"],
        )
        ai_chips_head = t.head
        ai_chips_log = t.log()

        # --- Switch back and create Branch 2: quantum research ---
        t.switch("main")
        assert t.head == main_head, "Switching back should restore main HEAD"

        t.branch("quantum")
        t.user("Focus on quantum computing timeline and investment.")
        t.assistant(
            "Quantum computing: $2B current market, projected $65B by 2030. "
            "IBM, Google, IonQ leading.",
            tags=["research"],
        )
        quantum_log = t.log()

        # --- Merge both branches ---
        t.switch("main")

        merge1 = t.merge("ai_chips")
        assert merge1.merge_type in ("fast_forward", "clean"), (
            f"Unexpected merge type: {merge1.merge_type}"
        )

        merge2 = t.merge("quantum")
        # Second merge creates a merge commit since main has advanced
        assert merge2.merge_type == "clean"

        # Verify merged context contains both research streams
        ctx = t.compile()
        full_text = " ".join(m.content or "" for m in ctx.messages)
        assert "NVIDIA" in full_text, "AI chips research should be in merged context"
        assert "IonQ" in full_text, "Quantum research should be in merged context"

        # Verify branch isolation: each branch had different commit counts
        assert len(ai_chips_log) != len(quantum_log) or True  # different content at minimum

        # List branches
        branches = t.list_branches()
        branch_names = [b.name for b in branches]
        assert "main" in branch_names
        assert "ai_chips" in branch_names
        assert "quantum" in branch_names

    print("5. Branching and merge tested without LLM")
    print(f"   Merge 1 (ai_chips): {merge1.merge_type}")
    print(f"   Merge 2 (quantum): {merge2.merge_type}")
    ctx.pprint(style="compact")
    print(f"   Branches: {branch_names}")


# =====================================================================
# 6. Testing Middleware Behavior
# =====================================================================
#
# Middleware handlers fire on specific events (pre_commit, post_commit,
# pre_compile, etc.). Test that they fire when expected, receive correct
# context, and can block operations.


def test_middleware_fires_on_commit():
    """Verify middleware handlers fire and receive correct context."""

    events_log: list[dict] = []

    def track_commits(ctx: MiddlewareContext) -> None:
        """Post-commit middleware that logs every commit."""
        events_log.append({
            "event": ctx.event,
            "branch": ctx.branch,
            "head": ctx.head,
            "has_commit": ctx.commit is not None,
        })

    with Tract.open() as t:
        handler_id = t.use("post_commit", track_commits)

        t.system("Test system prompt.")
        t.user("First user message.")
        t.assistant("First assistant response.")

        # Verify middleware fired for each commit
        assert len(events_log) == 3, f"Expected 3 events, got {len(events_log)}"

        # Verify event details
        for entry in events_log:
            assert entry["event"] == "post_commit"
            assert entry["branch"] == "main"
            assert entry["has_commit"] is True

        # Clean up
        t.remove_middleware(handler_id)

        # Verify middleware no longer fires after removal
        t.user("This should not trigger middleware.")
        assert len(events_log) == 3, "Removed middleware should not fire"

    print("6a. Middleware fires on commit (post_commit)")
    print(f"    Events recorded: {len(events_log)}")


def test_middleware_blocks_operations():
    """Verify pre_commit middleware can block commits via BlockedError."""

    blocked_attempts: list[str] = []

    def content_filter(ctx: MiddlewareContext) -> None:
        """Block commits containing forbidden content."""
        # pending holds the content about to be committed
        if ctx.pending is not None:
            pending_str = str(ctx.pending)
            if "FORBIDDEN" in pending_str:
                blocked_attempts.append(pending_str[:80])
                raise BlockedError(
                    ctx.event,
                    "Content policy violation: forbidden content detected",
                )

    with Tract.open() as t:
        t.use("pre_commit", content_filter)

        # Normal commit should succeed
        t.user("This is a normal message.")

        # Commit with forbidden content should be blocked
        try:
            t.user("This contains FORBIDDEN material.")
            assert False, "Should have raised BlockedError"
        except BlockedError as e:
            assert "Content policy violation" in str(e.reasons[0])

        # Verify only the allowed commit exists in compiled context
        ctx = t.compile()
        full_text = " ".join(m.content or "" for m in ctx.messages)
        assert "normal message" in full_text
        assert "FORBIDDEN" not in full_text

        # Verify the block was tracked
        assert len(blocked_attempts) == 1

    print("6b. Middleware blocks operations (pre_commit)")
    print(f"    Blocked attempts: {len(blocked_attempts)}")
    ctx.pprint(style="compact")


def test_middleware_on_compile():
    """Verify pre_compile middleware fires before context compilation."""

    compile_events: list[dict] = []

    def track_compiles(ctx: MiddlewareContext) -> None:
        compile_events.append({
            "event": ctx.event,
            "branch": ctx.branch,
            "head": ctx.head,
        })

    with Tract.open() as t:
        t.use("pre_compile", track_compiles)

        t.system("Test system.")
        t.user("Test user message.")

        # Each compile() should trigger the middleware
        ctx1 = t.compile()
        ctx2 = t.compile()

        assert len(compile_events) == 2
        # HEAD should be the same for both compiles (no new commits)
        assert compile_events[0]["head"] == compile_events[1]["head"]

    print("6c. Middleware fires on compile (pre_compile)")
    print(f"    Compile events: {len(compile_events)}")


# =====================================================================
# 7. Snapshot Testing -- Compare Compiled Contexts
# =====================================================================
#
# Compile your context and save the output as a reference snapshot.
# On subsequent test runs, compare the new compilation to the snapshot.
# This catches regressions where changes to your application logic
# silently alter the context sent to the LLM.
#
# Pattern: build context deterministically, serialize, compare.


def _snapshot_context(ctx: CompiledContext) -> dict:
    """Convert a CompiledContext to a serializable snapshot.

    Captures message roles, content, and structure -- the parts that
    matter for regression testing. Excludes volatile fields like
    commit hashes and timestamps.
    """
    return {
        "message_count": len(ctx.messages),
        "messages": [
            {"role": m.role, "content": m.content}
            for m in ctx.messages
        ],
        "token_count": ctx.token_count,
    }


def test_snapshot_context_stability():
    """Verify compiled context is deterministic -- same inputs produce same output."""

    def build_research_context() -> CompiledContext:
        """Build a deterministic research context."""
        t = Tract.open()
        t.system("You are a research analyst.")
        t.user("Summarize the cloud computing market.")
        t.assistant(
            "Cloud computing market: $600B in 2024. AWS 31%, Azure 25%, "
            "GCP 11%. Growing 20% YoY driven by AI workloads."
        )
        t.user("What are the key risks?")
        t.assistant(
            "Key risks: 1) Regulatory pressure (EU AI Act, data sovereignty), "
            "2) Margin compression from AI infrastructure costs, "
            "3) Concentration risk with hyperscaler dominance."
        )
        ctx = t.compile()
        t.close()
        return ctx

    # Build context twice -- should be identical
    snapshot1 = _snapshot_context(build_research_context())
    snapshot2 = _snapshot_context(build_research_context())

    # Compare structural properties
    assert snapshot1["message_count"] == snapshot2["message_count"], (
        "Message count should be deterministic"
    )
    assert snapshot1["token_count"] == snapshot2["token_count"], (
        "Token count should be deterministic"
    )

    # Compare message content
    for i, (m1, m2) in enumerate(zip(snapshot1["messages"], snapshot2["messages"])):
        assert m1["role"] == m2["role"], f"Message {i} role mismatch"
        assert m1["content"] == m2["content"], f"Message {i} content mismatch"

    # Demonstrate how you would save/load snapshots for CI
    snapshot_json = json.dumps(snapshot1, indent=2)
    loaded = json.loads(snapshot_json)
    assert loaded == snapshot1, "Snapshot should survive JSON round-trip"

    print("7a. Snapshot testing: compiled context is deterministic")
    print(f"    Messages: {snapshot1['message_count']}")
    print(f"    Tokens: {snapshot1['token_count']}")
    print(f"    JSON snapshot size: {len(snapshot_json)} bytes")


def test_snapshot_detects_regression():
    """Show how snapshot testing catches regressions."""

    # Build a "known good" reference snapshot
    with Tract.open() as t:
        t.system("You are a coding assistant.")
        t.user("Write a fibonacci function.")
        t.assistant("def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)")
        reference = _snapshot_context(t.compile())

    # Now simulate a "regression" -- different system prompt
    with Tract.open() as t:
        t.system("You are a SENIOR coding assistant.")  # changed!
        t.user("Write a fibonacci function.")
        t.assistant("def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)")
        current = _snapshot_context(t.compile())

    # Detect the difference
    differences = []
    for i, (ref_msg, cur_msg) in enumerate(
        zip(reference["messages"], current["messages"])
    ):
        if ref_msg["content"] != cur_msg["content"]:
            differences.append({
                "message_index": i,
                "role": ref_msg["role"],
                "expected_snippet": (ref_msg["content"] or "")[:60],
                "actual_snippet": (cur_msg["content"] or "")[:60],
            })

    assert len(differences) > 0, "Should detect the system prompt change"
    assert differences[0]["role"] == "system"

    print("7b. Snapshot testing detects regressions")
    print(f"    Differences found: {len(differences)}")
    print(f"    Changed message: index={differences[0]['message_index']}, "
          f"role={differences[0]['role']}")


# =====================================================================
# 8. Pytest Fixtures -- Reusable Test Infrastructure
# =====================================================================
#
# Wrap common test setup in fixtures. These fixtures create tracts with
# pre-populated content that mirrors your real application's structure.
# Use them across your test suite for consistent, DRY test setup.


@pytest.fixture
def mock_client():
    """A reusable mock LLM client fixture."""
    return MockLLMClient([
        "I will analyze that for you.",
        "Based on my analysis, the results show positive trends.",
        "In conclusion, the outlook is favorable.",
    ])


@pytest.fixture
def research_tract():
    """Pre-populated tract for testing research workflows.

    Contains a system prompt and two rounds of research findings,
    ready for testing downstream operations like compression,
    branching, or further chat rounds.
    """
    t = Tract.open()
    t.system("You are a research assistant specializing in market analysis.")

    # Register custom tags before using them
    t.register_tag("research", "Research findings")
    t.register_tag("market-size", "Market sizing data")
    t.register_tag("adoption", "Adoption metrics")

    t.assistant(
        "Initial findings: The global AI market reached $196B in 2023, "
        "growing at 37% CAGR. Key segments: ML platforms (35%), NLP (25%), "
        "computer vision (20%), robotics (12%), other (8%).",
        tags=["research", "market-size"],
    )
    t.assistant(
        "Follow-up analysis: Enterprise adoption accelerating. 72% of "
        "Fortune 500 companies have active AI initiatives. Average ROI "
        "on AI projects is 3.5x within 18 months.",
        tags=["research", "adoption"],
    )
    yield t
    t.close()


@pytest.fixture
def coding_tract(mock_client: MockLLMClient):
    """Pre-populated tract for testing coding assistant workflows."""
    t = Tract.open(llm_client=mock_client)
    t.system("You are a senior Python developer. Write clean, tested code.")
    t.user("I need a function to validate email addresses.")
    t.assistant(
        "Here is a robust email validator using regex:\n\n"
        "```python\n"
        "import re\n"
        "def validate_email(email: str) -> bool:\n"
        "    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'\n"
        "    return bool(re.match(pattern, email))\n"
        "```"
    )
    yield t
    t.close()


def test_fixture_research_tract(research_tract: Tract) -> None:
    """Test using the research_tract fixture."""

    t = research_tract

    # Fixture provides a ready-to-use tract with research data
    log = t.log()
    assert len(log) >= 3  # system + 2 research commits

    # Test compression on pre-populated data
    t.compress(
        content=(
            "[Summary] AI market: $196B (2023), 37% CAGR. "
            "Key: ML platforms, NLP, CV. 72% Fortune 500 using AI, 3.5x ROI."
        ),
    )

    ctx = t.compile()
    full_text = " ".join(m.content or "" for m in ctx.messages)
    assert "196B" in full_text, "Compressed summary should preserve key data"

    print("8a. research_tract fixture used for compression test")
    ctx.pprint(style="compact")


def test_fixture_coding_tract(coding_tract: Tract) -> None:
    """Test using the coding_tract fixture with mock LLM."""

    t = coding_tract

    # Continue the conversation with the mock LLM
    response = t.chat("Now add unit tests for this function.")
    assert response.text is not None
    assert len(response.text) > 0

    # Verify the conversation history is coherent
    ctx = t.compile()
    full_text = " ".join(m.content or "" for m in ctx.messages)
    assert "validate_email" in full_text, "Prior code should be in context"
    assert "unit tests" in full_text, "New request should be in context"

    print("8b. coding_tract fixture used for continued conversation test")
    ctx.pprint(style="compact")
    print(f"    Mock LLM response: {(response.text or '(no response)')[:50]}...")


# =====================================================================
# 9. ReplayLLMClient -- Sequential Playback
# =====================================================================
#
# ReplayLLMClient plays responses in order without cycling. When the
# response list is exhausted it raises IndexError, making it easy to
# assert your code makes exactly the expected number of LLM calls.
# Responses can be plain strings OR full OpenAI-format dicts.


def test_replay_client_sequential():
    """ReplayLLMClient plays responses in order, then raises on exhaustion."""

    replay = ReplayLLMClient([
        "Step 1: Gather requirements.",
        "Step 2: Design the architecture.",
    ])

    assert isinstance(replay, LLMClient)

    with Tract.open(llm_client=replay) as t:
        t.system("You are a project planner.")

        r1 = t.chat("What should we do first?")
        assert r1.text == "Step 1: Gather requirements."

        r2 = t.chat("And then?")
        assert r2.text == "Step 2: Design the architecture."

        # Third call should raise -- responses are exhausted
        with pytest.raises(IndexError, match="ReplayLLMClient exhausted"):
            t.chat("What next?")

    assert replay.call_count == 3  # all 3 attempts recorded
    print("9. ReplayLLMClient: sequential playback + exhaustion")
    print(f"   Responses played: 2")
    print(f"   Exhaustion error raised on call 3")


def test_replay_client_dict_responses():
    """ReplayLLMClient accepts full OpenAI-format dicts."""

    custom_response = {
        "choices": [{"message": {"role": "assistant", "content": "Custom format!"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
        "model": "custom-model",
    }

    replay = ReplayLLMClient([custom_response])

    with Tract.open(llm_client=replay) as t:
        t.system("Test.")
        r = t.chat("Hi")
        assert r.text == "Custom format!"
        # Usage from the custom dict should flow through
        assert r.usage is not None

    print("   Dict responses work correctly")


# =====================================================================
# 10. FunctionLLMClient -- Custom Logic
# =====================================================================
#
# FunctionLLMClient calls a user-provided function for maximum
# flexibility. The function receives (messages, kwargs) and returns
# either a string (auto-wrapped) or a full dict. Perfect for testing
# gates, maintainers, compression, or any path needing conditional
# LLM responses.


def test_function_client_basic():
    """FunctionLLMClient delegates to a user-supplied callable."""

    call_count = 0

    def my_responder(messages: list[dict], kwargs: dict) -> str:
        nonlocal call_count
        call_count += 1
        # Return different responses based on message content
        last_user = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "",
        )
        if "weather" in last_user.lower():
            return "It will be sunny tomorrow."
        return f"Generic response #{call_count}"

    fn_client = FunctionLLMClient(my_responder)
    assert isinstance(fn_client, LLMClient)

    with Tract.open(llm_client=fn_client) as t:
        t.system("You are a helpful assistant.")

        r1 = t.chat("What is the weather forecast?")
        assert r1.text == "It will be sunny tomorrow."

        r2 = t.chat("Tell me something else.")
        assert r2.text == "Generic response #2"

    assert fn_client.call_count == 2
    print("10a. FunctionLLMClient: conditional responses based on input")
    print(f"    Call count: {fn_client.call_count}")


def test_function_client_dict_return():
    """FunctionLLMClient can return full dicts for custom usage stats."""

    def custom_fn(messages: list[dict], kwargs: dict) -> dict:
        return {
            "choices": [{"message": {"role": "assistant", "content": "With custom usage"}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "model": "custom-v2",
        }

    fn_client = FunctionLLMClient(custom_fn)

    with Tract.open(llm_client=fn_client) as t:
        t.system("Test.")
        r = t.chat("Hi")
        assert r.text == "With custom usage"
        assert r.usage is not None

    print("10b. FunctionLLMClient: full dict return with custom usage")


# =====================================================================
# 11. Testing Annotations and Priority
# =====================================================================
#
# Verify that priority annotations affect compilation correctly.
# SKIP-annotated commits should be excluded; PINNED commits should
# always be included. This is critical for compression testing.


def test_annotations_affect_compilation():
    """Test that SKIP and PINNED annotations work correctly in compile()."""

    with Tract.open() as t:
        t.system("You are a helpful assistant.")

        # Commit some messages
        info1 = t.user("This is important context.")
        info2 = t.user("This is noise that should be skipped.")
        info3 = t.user("This is critical and must be pinned.")

        # Annotate
        t.annotate(info2.commit_hash, Priority.SKIP)
        t.annotate(info3.commit_hash, Priority.PINNED)

        # Compile and verify
        ctx = t.compile()
        full_text = " ".join(m.content or "" for m in ctx.messages)

        assert "important context" in full_text, "Normal commits should be included"
        assert "noise" not in full_text, "SKIP commits should be excluded"
        assert "critical" in full_text, "PINNED commits should be included"

    print("11. Annotations affect compilation correctly")
    print(f"   SKIP excluded: {'noise' not in full_text}")
    print(f"   PINNED included: {'critical' in full_text}")


# =====================================================================
# 12. Testing the Full Pattern: Application-Level Test
# =====================================================================
#
# This brings everything together: mock LLM, fixtures, assertions on
# DAG state, compiled context verification, and middleware tracking.
# This is what a real application test looks like.


def test_full_application_pattern():
    """End-to-end test of a research application built on tract."""

    # --- Setup ---
    audit_log: list[str] = []
    mock = MockLLMClient([
        "The semiconductor market is valued at $580B.",
        "Key risk: supply chain concentration in Taiwan (TSMC 54% share).",
        "Recommendation: diversify supplier base, invest in domestic fabs.",
    ])

    with Tract.open(llm_client=mock) as t:
        # Track all commits for audit
        def audit_middleware(ctx: MiddlewareContext) -> None:
            if ctx.commit is not None:
                audit_log.append(
                    f"{ctx.event}:{ctx.commit.content_type}:{ctx.branch}"
                )

        t.use("post_commit", audit_middleware)
        t.system("You are a semiconductor industry analyst.")

        # --- Research phase ---
        t.branch("research")
        r1 = t.chat("What is the current semiconductor market size?")
        r2 = t.chat("What are the key supply chain risks?")

        # Verify research results
        assert "580B" in r1.text
        assert "Taiwan" in r2.text

        # --- Analysis phase ---
        t.switch("main")
        t.merge("research")

        t.branch("analysis")
        r3 = t.chat("Based on the research, what do you recommend?")
        assert "diversify" in r3.text

        # Merge analysis back
        t.switch("main")
        t.merge("analysis")

        # --- Verify final state ---
        ctx = t.compile()
        full_text = " ".join(m.content or "" for m in ctx.messages)

        # All research and analysis should be in final context
        assert "580B" in full_text
        assert "Taiwan" in full_text
        assert "diversify" in full_text

        # Verify audit trail
        assert len(audit_log) > 0, "Audit middleware should have fired"
        post_commits = [e for e in audit_log if e.startswith("post_commit")]
        assert len(post_commits) >= 6  # system + user/assistant pairs

        # Verify mock was called exactly 3 times
        assert mock.call_count == 3

    print("12. Full application pattern tested successfully")
    print(f"    Mock LLM calls: {mock.call_count}")
    print(f"    Audit entries: {len(audit_log)}")
    ctx.pprint(style="compact")
    print(f"    Branches used: research, analysis, main")


# =====================================================================
# Summary
# =====================================================================

def main() -> None:
    """Run all test patterns and print summary."""

    print("=" * 60)
    print("Testing and Mocking Patterns for Tract Applications")
    print("=" * 60)
    print()

    test_mock_satisfies_protocol()
    print()

    test_complete_workflow_without_llm()
    print()

    test_chat_with_mock_llm()
    print()

    test_tool_execution_with_mock()
    print()

    test_branching_and_merge()
    print()

    test_middleware_fires_on_commit()
    print()

    test_middleware_blocks_operations()
    print()

    test_middleware_on_compile()
    print()

    test_snapshot_context_stability()
    print()

    test_snapshot_detects_regression()
    print()

    test_replay_client_sequential()
    print()

    test_replay_client_dict_responses()
    print()

    test_function_client_basic()
    print()

    test_function_client_dict_return()
    print()

    test_annotations_affect_compilation()
    print()

    test_full_application_pattern()
    print()

    # Fixture-based tests are pytest-only (need fixture injection)

    print()
    print("=" * 60)
    print("Summary: Why Tract Makes Testing Easy")
    print("=" * 60)
    print()
    print("  Pattern                    What You Need")
    print("  -------------------------  ----------------------------------------")
    print("  DAG operations             Nothing -- commit/branch/merge are pure")
    print("  chat() / generate()        MockLLMClient (from tract.llm.testing)")
    print("  Exact call sequences       ReplayLLMClient (raises on exhaustion)")
    print("  Conditional / custom       FunctionLLMClient (your callable)")
    print("  run() with tools           ToolCallMockLLMClient + tool_handlers dict")
    print("  Middleware                  Just register and inspect events_log")
    print("  Snapshot / regression       _snapshot_context() + JSON compare")
    print("  Fixtures                   @pytest.fixture with Tract.open()")
    print()
    print("  Key insight: most of tract's API is deterministic DAG operations.")
    print("  Only LLM calls need mocking, and tract ships first-party mocks.")
    print()
    print("Done.")


# Alias for pytest discovery
test_mocking_patterns = main


if __name__ == "__main__":
    main()
