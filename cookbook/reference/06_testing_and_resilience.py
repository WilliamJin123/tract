"""Testing and Resilience -- Mocking, Fixtures, Recovery, and Failover

How to test tract applications without live LLM calls, and how to build
resilient production workflows using tract's git-like primitives.

Sections:
  1.  MockLLMClient               -- first-party cycling mock
  2.  ReplayLLMClient             -- sequential playback, exhaustion check
  3.  FunctionLLMClient           -- custom logic, conditional responses
  4.  Pytest Fixtures             -- reusable test infrastructure
  5.  Snapshot Testing            -- compiled context regression testing
  6.  Checkpoint Recovery         -- tag + reset on failure
  7.  Branch-Isolated Retries     -- branch per attempt, merge on success
  8.  Compression Fallback Chain  -- progressively aggressive compression
  9.  Circuit Breaker             -- middleware blocks after N failures
  10. FallbackClient              -- automatic multi-provider failover

Demonstrates: MockLLMClient, ReplayLLMClient, FunctionLLMClient, LLMClient,
              FallbackClient, compile(), branch(), merge(), compress(),
              middleware, tags, BlockedError

No LLM required.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from tract import (
    BlockedError,
    CompiledContext,
    FunctionLLMClient,
    LLMClient,
    MiddlewareContext,
    MockLLMClient,
    ReplayLLMClient,
    Tract,
)
from tract.llm.fallback import FallbackClient
from tract.llm.protocols import LLMClient as LLMClientProtocol


# ===================================================================
# 1. MockLLMClient
# ===================================================================

def mock_llm_client_demo() -> None:
    """Cycling mock: satisfies LLMClient, tracks calls, cycles responses."""
    print("=" * 60)
    print("1. MockLLMClient")
    print("=" * 60)

    mock = MockLLMClient(["Hello!"])
    assert isinstance(mock, LLMClient)
    r = mock.chat([{"role": "user", "content": "Hi"}])
    assert r["choices"][0]["message"]["content"] == "Hello!"
    assert mock.call_count == 1

    # Cycling: wraps around
    m2 = MockLLMClient(["A", "B"])
    results = [m2.chat([])["choices"][0]["message"]["content"] for _ in range(3)]
    assert results == ["A", "B", "A"]

    # Use with Tract
    m3 = MockLLMClient(["Revenue grew 15%."])
    with Tract.open(llm_client=m3) as t:
        t.system("Analyst.")
        r = t.llm.chat("Analyze Q3.")
        assert r.text == "Revenue grew 15%." and r.commit_info is not None

    print(f"  Protocol: OK, cycling: OK, Tract integration: OK")
    print("PASSED\n")


# ===================================================================
# 2. ReplayLLMClient
# ===================================================================

def replay_client_demo() -> None:
    """Sequential playback; raises IndexError on exhaustion."""
    print("=" * 60)
    print("2. ReplayLLMClient")
    print("=" * 60)

    replay = ReplayLLMClient(["Step 1.", "Step 2."])
    assert isinstance(replay, LLMClient)

    with Tract.open(llm_client=replay) as t:
        t.system("Planner.")
        assert t.llm.chat("First?").text == "Step 1."
        assert t.llm.chat("Then?").text == "Step 2."
        try:
            t.llm.chat("Next?")
            assert False, "Should raise"
        except IndexError:
            pass

    assert replay.call_count == 3

    # Dict responses work too
    replay2 = ReplayLLMClient([{
        "choices": [{"message": {"role": "assistant", "content": "Custom!"}}],
        "usage": {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49},
    }])
    with Tract.open(llm_client=replay2) as t:
        t.system("Test.")
        assert t.llm.chat("Hi").text == "Custom!"

    print(f"  Sequential: OK, exhaustion: IndexError, dict responses: OK")
    print("PASSED\n")


# ===================================================================
# 3. FunctionLLMClient
# ===================================================================

def function_client_demo() -> None:
    """Delegates to a user-supplied callable for conditional responses."""
    print("=" * 60)
    print("3. FunctionLLMClient")
    print("=" * 60)

    count = 0

    def responder(messages: list[dict], kwargs: dict) -> str:
        nonlocal count
        count += 1
        last = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"), ""
        )
        return "Sunny tomorrow." if "weather" in last.lower() else f"Response #{count}"

    fn = FunctionLLMClient(responder)
    assert isinstance(fn, LLMClient)

    with Tract.open(llm_client=fn) as t:
        t.system("Assistant.")
        assert t.llm.chat("Weather?").text == "Sunny tomorrow."
        assert t.llm.chat("Other.").text == "Response #2"

    # Dict return for custom usage
    fn2 = FunctionLLMClient(lambda m, k: {
        "choices": [{"message": {"role": "assistant", "content": "With usage"}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    })
    with Tract.open(llm_client=fn2) as t:
        t.system("Test.")
        r = t.llm.chat("Hi")
        assert r.text == "With usage" and r.usage is not None

    print(f"  Conditional: OK, dict return: OK, call_count: {fn.call_count}")
    print("PASSED\n")


# ===================================================================
# 4. Pytest Fixtures
# ===================================================================

@pytest.fixture
def mock_client():
    """Reusable mock LLM client fixture."""
    return MockLLMClient(["Analysis result.", "Positive trends.", "Favorable."])


@pytest.fixture
def research_tract():
    """Pre-populated tract for testing research workflows."""
    t = Tract.open()
    t.system("You are a research assistant.")
    t.tags.register("research", "Research findings")
    t.assistant("AI market: $196B in 2023, 37% CAGR.", tags=["research"])
    t.assistant("72% Fortune 500 using AI. 3.5x ROI.", tags=["research"])
    yield t
    t.close()


def pytest_fixtures_demo() -> None:
    """Demonstrate fixture patterns (callable outside pytest too)."""
    print("=" * 60)
    print("4. Pytest Fixtures")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a research assistant.")
        t.tags.register("research", "Research findings")
        t.assistant("AI market: $196B, 37% CAGR.", tags=["research"])

        assert len(t.search.log()) >= 2
        t.compression.compress(content="AI market: $196B, 37% CAGR.")
        ctx = t.compile()
        assert "196B" in " ".join(m.content or "" for m in ctx.messages)

    print(f"  Pattern: @pytest.fixture + Tract.open() + yield + close()")
    print("PASSED\n")


# ===================================================================
# 5. Snapshot Testing
# ===================================================================

def _snapshot(ctx: CompiledContext) -> dict:
    return {
        "messages": [{"role": m.role, "content": m.content} for m in ctx.messages],
        "token_count": ctx.token_count,
    }


def snapshot_testing_demo() -> None:
    """Compiled context is deterministic -- detect regressions via snapshots."""
    print("=" * 60)
    print("5. Snapshot Testing")
    print("=" * 60)

    def build() -> CompiledContext:
        t = Tract.open()
        t.system("You are a research analyst.")
        t.user("Summarize cloud computing.")
        t.assistant("Cloud: $600B in 2024. AWS 31%, Azure 25%, GCP 11%.")
        ctx = t.compile()
        t.close()
        return ctx

    s1, s2 = _snapshot(build()), _snapshot(build())
    assert s1 == s2, "Same inputs must produce same output"
    assert json.loads(json.dumps(s1)) == s1, "Must survive JSON round-trip"

    # Detect regression
    with Tract.open() as t:
        t.system("You are a SENIOR analyst.")  # changed!
        t.user("Summarize cloud computing.")
        t.assistant("Cloud: $600B in 2024. AWS 31%, Azure 25%, GCP 11%.")
        current = _snapshot(t.compile())

    diffs = [i for i, (a, b) in enumerate(
        zip(s1["messages"], current["messages"])) if a != b]
    assert len(diffs) > 0
    print(f"  Deterministic: OK, regression detected: {len(diffs)} diff(s)")
    print("PASSED\n")


# ===================================================================
# 6. Checkpoint Recovery
# ===================================================================

def checkpoint_recovery() -> None:
    """Tag HEAD before risky ops, reset on failure."""
    print("=" * 60)
    print("6. Checkpoint Recovery")
    print("=" * 60)

    with Tract.open() as t:
        t.system("Data analyst.")
        t.user("Sales data.")
        t.assistant("Ready.")

        t.tags.register("checkpoint", "Safe rollback")
        cp = t.head
        t.tags.add(cp, "checkpoint")

        # Risky operation
        t.user("Complex regression with polynomial features.")
        t.assistant("Computing...")
        before = len(t.search.log())

        t.branches.reset(cp)  # Reset to checkpoint
        after = len(t.search.log())

        # Retry simply
        t.user("Top 3 sales trends.")
        t.assistant("1. Revenue +12%. 2. Enterprise led. 3. Churn down.")

        text = " ".join((m.content or "") for m in t.compile().messages)
        assert "polynomial" not in text and "trends" in text

    print(f"  Before: {before}, after reset: {after}, recovery: OK")
    print("PASSED\n")


# ===================================================================
# 7. Branch-Isolated Retries
# ===================================================================

def branch_isolated_retries() -> None:
    """Create a branch per attempt, merge only on success."""
    print("=" * 60)
    print("7. Branch-Isolated Retries")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are a financial modeler.")
        t.user("Build a revenue projection model.")
        t.assistant("Trying several approaches.")

        # Attempt 1 fails
        t.branches.create("attempt_1", switch=True)
        t.assistant("Monte Carlo... ERROR: Variance too high.")
        t.branches.switch("main")

        # Attempt 2 fails
        t.branches.create("attempt_2", switch=True)
        t.assistant("Linear regression... ERROR: R-squared 0.23.")
        t.branches.switch("main")

        # Attempt 3 succeeds
        t.branches.create("attempt_3", switch=True)
        t.assistant("3-month moving average: Q4 $2.1M. Robust.")
        t.branches.switch("main")
        mr = t.merge("attempt_3")

        ctx = t.compile()
        text = " ".join((m.content or "") for m in ctx.messages)
        assert "Variance" not in text and "R-squared" not in text
        assert "moving average" in text

    print(f"  Merge type: {mr.merge_type}, failed branches isolated: OK")
    print("PASSED\n")


# ===================================================================
# 8. Compression Fallback Chain
# ===================================================================

def compression_fallback_chain() -> None:
    """Try progressively aggressive compression until one succeeds."""
    print("=" * 60)
    print("8. Compression Fallback Chain")
    print("=" * 60)

    with Tract.open() as t:
        t.system("Tracking weekly metrics.")
        for w in range(1, 21):
            t.user(f"Week {w}: ${w*1000} rev, {w*50} users.")
            t.assistant(f"Week {w} recorded.")
        orig = t.compile().token_count

        won = None
        for name, content in [("sliding", "Last 5 weeks of data."),
                               ("summary", "20 weeks, $210k total, 1000 users."),
                               ("aggressive", "$210k revenue, 1000 users.")]:
            try:
                t.compression.compress(content=content)
                after = t.compile().token_count
                print(f"  '{name}': {orig}->{after} ({(1-after/orig)*100:.0f}% saved)")
                won = name
                break
            except Exception as e:
                print(f"  '{name}' failed: {e}")
        assert won is not None
    print("PASSED\n")


# ===================================================================
# 9. Circuit Breaker via Middleware
# ===================================================================

def circuit_breaker() -> None:
    """Middleware blocks commits after N consecutive failures."""
    print("=" * 60)
    print("9. Circuit Breaker")
    print("=" * 60)

    with Tract.open() as t:
        t.system("You are an API manager.")
        breaker = {"failures": 0, "threshold": 3, "open": False}

        def cb_mw(ctx: MiddlewareContext):
            if breaker["open"]:
                raise BlockedError(ctx.event, "Circuit breaker OPEN.")

        mw_id = t.middleware.add("pre_commit", cb_mw)
        blocked = 0
        for endpoint, ok in [("GET /users", True), ("GET /orders", True),
                              ("POST /pay", False), ("POST /pay", False),
                              ("POST /pay", False), ("GET /status", True)]:
            try:
                if not ok:
                    breaker["failures"] += 1
                    if breaker["failures"] >= breaker["threshold"]:
                        breaker["open"] = True
                t.user(f"Call {endpoint}")
                t.assistant(f"{'200' if ok else '500'}: {endpoint}")
                if ok:
                    breaker["failures"] = 0
            except BlockedError:
                blocked += 1

        assert blocked >= 1 and breaker["open"]
        breaker["open"] = False
        breaker["failures"] = 0
        t.user("GET /health")
        t.assistant("200 OK")
        t.middleware.remove(mw_id)

    print(f"  Threshold: 3, blocked: {blocked}, reset: OK")
    print("PASSED\n")


# ===================================================================
# 10. FallbackClient
# ===================================================================

class _MockClient:
    """Minimal LLMClient mock for FallbackClient demos."""
    def __init__(self, name: str, err: Exception | None = None):
        self.name, self._err, self.calls = name, err, []

    def chat(self, messages: list[dict[str, str]], **kw: Any) -> dict:
        self.calls.append(messages)
        if self._err:
            raise self._err
        return {"choices": [{"message": {"role": "assistant",
                "content": f"Hello from {self.name}"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15}}

    def close(self) -> None: pass
    def extract_content(self, r: dict) -> str:
        return r["choices"][0]["message"]["content"]
    def extract_usage(self, r: dict) -> dict | None: return r.get("usage")


def fallback_client_demo() -> None:
    """FallbackClient tries providers in priority order."""
    print("=" * 60)
    print("10. FallbackClient")
    print("=" * 60)

    # Primary succeeds
    c1 = FallbackClient(_MockClient("openai"), _MockClient("anthropic"))
    r = c1.chat([{"role": "user", "content": "Hi"}])
    assert c1.extract_content(r) == "Hello from openai"
    assert c1.last_client_index == 0
    c1.close()

    # Primary fails -> fallback
    c2 = FallbackClient(
        _MockClient("openai", ConnectionError("down")),
        _MockClient("anthropic"))
    r2 = c2.chat([{"role": "user", "content": "Hi"}])
    assert c2.extract_content(r2) == "Hello from anthropic"
    assert c2.last_client_index == 1
    c2.close()

    # All fail -> last exception re-raised
    c3 = FallbackClient(
        _MockClient("a", ConnectionError("down")),
        _MockClient("b", TimeoutError("timeout")))
    try:
        c3.chat([{"role": "user", "content": "Hi"}])
        assert False
    except TimeoutError:
        pass
    c3.close()

    # Protocol conformance
    assert isinstance(FallbackClient(_MockClient("x")), LLMClientProtocol)

    print(f"  Primary OK: index 0, failover: index 1, all-fail: last exc, "
          f"protocol: OK")
    print("PASSED\n")


# ===================================================================
# Main
# ===================================================================

def main() -> None:
    mock_llm_client_demo()
    replay_client_demo()
    function_client_demo()
    pytest_fixtures_demo()
    snapshot_testing_demo()
    checkpoint_recovery()
    branch_isolated_retries()
    compression_fallback_chain()
    circuit_breaker()
    fallback_client_demo()

    print("=" * 60)
    print("Summary: Testing and Resilience")
    print("=" * 60)
    print()
    print("  Testing: MockLLMClient (cycling), ReplayLLMClient (exhaustion),")
    print("  FunctionLLMClient (custom), fixtures, snapshot regression.")
    print("  Most tract ops are pure DAG -- no mocking needed.")
    print()
    print("  Resilience: checkpoint recovery (tags + reset), branch-isolated")
    print("  retries, compression fallback chains, circuit breakers via")
    print("  middleware, FallbackClient for multi-provider failover.")
    print()
    print("Done.")


test_testing_and_resilience = main

if __name__ == "__main__":
    main()
