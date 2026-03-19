"""Tests for tract.routing — fuzzy matching, semantic routing, and auto-config."""

from __future__ import annotations

import asyncio
import json

import pytest

from tract import Tract
from tract.llm.testing import MockLLMClient, ReplayLLMClient
from tract.routing import (
    Route,
    RoutingResult,
    RoutingTable,
    SemanticRouter,
)


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def table() -> RoutingTable:
    """Routing table with a handful of test routes."""
    t = RoutingTable()
    t.add_route("design", "Software design and architecture phase", "stage",
                keywords=["architecture", "plan", "design", "blueprint"])
    t.add_route("implement", "Code implementation stage", "stage",
                keywords=["code", "implement", "build", "write"])
    t.add_route("test", "Testing and validation stage", "stage",
                keywords=["test", "validate", "verify", "check"])
    t.add_route("research", "Deep research branch", "branch",
                keywords=["investigate", "research", "explore", "study"])
    t.add_route("coding", "Coding workflow profile", "workflow",
                keywords=["coding", "development", "software"])
    return t


@pytest.fixture
def tract_instance():
    """A minimal Tract for testing."""
    return Tract.open()


# ===================================================================
# Route dataclass
# ===================================================================


class TestRoute:
    def test_frozen(self):
        r = Route(target="main", route_type="branch", confidence=0.9, reasoning="test")
        with pytest.raises(AttributeError):
            r.target = "other"  # type: ignore[misc]

    def test_fields(self):
        r = Route(target="design", route_type="stage", confidence=0.75, reasoning="good match")
        assert r.target == "design"
        assert r.route_type == "stage"
        assert r.confidence == 0.75
        assert r.reasoning == "good match"


# ===================================================================
# RoutingResult dataclass
# ===================================================================


class TestRoutingResult:
    def test_frozen(self):
        route = Route(target="x", route_type="branch", confidence=0.5, reasoning="r")
        rr = RoutingResult(route=route, applied=False, tokens_used=0, method="fuzzy")
        with pytest.raises(AttributeError):
            rr.applied = True  # type: ignore[misc]

    def test_fields(self):
        route = Route(target="y", route_type="stage", confidence=0.8, reasoning="r")
        rr = RoutingResult(route=route, applied=True, tokens_used=42, method="semantic")
        assert rr.route is route
        assert rr.applied is True
        assert rr.tokens_used == 42
        assert rr.method == "semantic"


# ===================================================================
# RoutingTable
# ===================================================================


class TestRoutingTable:
    def test_add_and_list(self, table: RoutingTable):
        names = table.list_routes()
        assert "design" in names
        assert "implement" in names
        assert "test" in names
        assert "research" in names
        assert "coding" in names

    def test_add_duplicate_raises(self, table: RoutingTable):
        with pytest.raises(ValueError, match="already registered"):
            table.add_route("design", "duplicate", "stage")

    def test_add_invalid_type_raises(self):
        t = RoutingTable()
        with pytest.raises(ValueError, match="Invalid route_type"):
            t.add_route("bad", "bad route", "invalid_type")

    def test_remove_route(self, table: RoutingTable):
        table.remove_route("design")
        assert "design" not in table.list_routes()

    def test_remove_nonexistent_raises(self, table: RoutingTable):
        with pytest.raises(ValueError, match="not found"):
            table.remove_route("nonexistent")

    def test_match_exact_keyword(self, table: RoutingTable):
        results = table.match("design")
        assert len(results) > 0
        # "design" route should rank high because it's an exact keyword match
        targets = [r.target for r in results]
        assert "design" in targets
        # Should be in top position or near top
        assert results[0].target == "design"

    def test_match_keyword_substring(self, table: RoutingTable):
        """Keywords that appear as substrings in the query should get a boost."""
        results = table.match("I need to implement the feature")
        targets = [r.target for r in results]
        assert "implement" in targets
        # implement should rank highly
        implement_routes = [r for r in results if r.target == "implement"]
        assert implement_routes[0].confidence > 0.3

    def test_match_fuzzy_misspelled(self, table: RoutingTable):
        """Slightly misspelled keywords should still match via SequenceMatcher."""
        results = table.match("reserch")
        # "reserch" is close to "research" via SequenceMatcher
        targets = [r.target for r in results]
        assert "research" in targets

    def test_match_partial_keyword(self, table: RoutingTable):
        """Partial keywords should produce some fuzzy matches."""
        results = table.match("archit")
        # "archit" is a prefix of "architecture" — should have some similarity
        targets = [r.target for r in results]
        assert len(results) > 0

    def test_match_no_results(self):
        """Empty table should return empty list."""
        t = RoutingTable()
        assert t.match("anything") == []

    def test_match_returns_sorted_by_confidence(self, table: RoutingTable):
        results = table.match("code testing")
        if len(results) > 1:
            for i in range(len(results) - 1):
                assert results[i].confidence >= results[i + 1].confidence

    def test_match_with_regex_pattern(self):
        t = RoutingTable()
        t.add_route("urgent", "Urgent requests", "branch",
                     keywords=["urgent"], pattern=r"\burgent\b")
        t.add_route("normal", "Normal requests", "branch",
                     keywords=["normal"])

        results = t.match("this is urgent")
        assert results[0].target == "urgent"
        assert results[0].confidence > 0.5

    def test_confidence_capped_at_1(self, table: RoutingTable):
        """Confidence should never exceed 1.0."""
        results = table.match("design architecture plan blueprint")
        for r in results:
            assert r.confidence <= 1.0

    def test_all_routes_have_reasoning(self, table: RoutingTable):
        results = table.match("something")
        for r in results:
            assert r.reasoning


# ===================================================================
# SemanticRouter
# ===================================================================


class TestSemanticRouter:
    def test_route_with_mock_llm(self, table: RoutingTable, tract_instance):
        """SemanticRouter should parse LLM JSON and return a Route."""
        llm_response = json.dumps({
            "target": "research",
            "confidence": 0.92,
            "reasoning": "The query is about investigation, which matches the research branch."
        })
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("I want to investigate this topic deeply", tract_instance)

        assert isinstance(result, RoutingResult)
        assert result.route.target == "research"
        assert result.route.confidence == 0.92
        assert result.method == "semantic"
        assert result.applied is False
        assert mock.call_count == 1

    def test_route_with_invalid_target_falls_back_to_fuzzy(
        self, table: RoutingTable, tract_instance
    ):
        """If LLM returns a target not in the routing table, fall back to fuzzy."""
        llm_response = json.dumps({
            "target": "nonexistent_route",
            "confidence": 0.8,
            "reasoning": "Best match"
        })
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("investigate", tract_instance)

        # Should fall back to fuzzy since "nonexistent_route" is not registered
        assert result.route.target in table.list_routes() or result.route.target == ""
        assert result.method == "semantic"  # method is still semantic (LLM was called)

    def test_route_llm_failure_falls_back_to_fuzzy(
        self, table: RoutingTable, tract_instance
    ):
        """If LLM call raises, fall back to fuzzy matching."""
        class FailingClient:
            def chat(self, messages, **kwargs):
                raise ConnectionError("LLM is down")
            def close(self):
                pass
            def extract_content(self, response):
                return response["choices"][0]["message"]["content"]
            def extract_usage(self, response):
                return response.get("usage")

        tract_instance.config.configure_llm(FailingClient())

        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("research topic", tract_instance)

        assert result.method == "fuzzy"
        assert result.route.target  # should have a fuzzy match
        assert result.tokens_used == 0

    def test_route_no_llm_client_falls_back(self, table: RoutingTable):
        """If no LLM client is configured, fall back to fuzzy."""
        t = Tract.open()  # no LLM configured
        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("design something", t)

        assert result.method == "fuzzy"
        assert result.route.target == "design"

    def test_route_unparseable_response_falls_back(
        self, table: RoutingTable, tract_instance
    ):
        """If LLM returns garbage, fall back to fuzzy."""
        mock = MockLLMClient(["this is not json at all"])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("research", tract_instance)

        # Should still return something (fuzzy fallback from parse)
        assert result.method == "semantic"
        assert result.route is not None

    def test_route_with_instructions(self, table: RoutingTable, tract_instance):
        """Custom instructions should be included in the system prompt."""
        llm_response = json.dumps({
            "target": "implement",
            "confidence": 0.85,
            "reasoning": "User wants to code"
        })
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(
            name="test-router",
            routes=table,
            instructions="Prefer implementation stages for coding requests.",
        )
        result = router.route("let's write some code", tract_instance)

        assert result.route.target == "implement"
        # Check that instructions were in the system prompt
        system_msg = mock.calls[0]["messages"][0]["content"]
        assert "Prefer implementation stages" in system_msg

    def test_route_confidence_clamped(self, table: RoutingTable, tract_instance):
        """Confidence from LLM should be clamped to [0.0, 1.0]."""
        llm_response = json.dumps({
            "target": "design",
            "confidence": 1.5,  # over 1.0
            "reasoning": "very confident"
        })
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("design", tract_instance)

        assert result.route.confidence <= 1.0

    def test_aroute_async(self, table: RoutingTable, tract_instance):
        """Test async routing."""
        llm_response = json.dumps({
            "target": "test",
            "confidence": 0.88,
            "reasoning": "User wants to test"
        })
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        result = asyncio.run(
            router.aroute("run the tests", tract_instance)
        )

        assert result.route.target == "test"
        assert result.method == "semantic"

    def test_last_result_stored(self, table: RoutingTable, tract_instance):
        """Router should store last_result for inspection."""
        llm_response = json.dumps({
            "target": "design",
            "confidence": 0.7,
            "reasoning": "match"
        })
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        assert router.last_result is None

        result = router.route("design", tract_instance)
        assert router.last_result is result

    def test_route_code_fence_json(self, table: RoutingTable, tract_instance):
        """LLM response wrapped in code fences should be parsed correctly."""
        inner = json.dumps({
            "target": "research",
            "confidence": 0.9,
            "reasoning": "match"
        })
        llm_response = f"```json\n{inner}\n```"
        mock = MockLLMClient([llm_response])
        tract_instance.config.configure_llm(mock)

        router = SemanticRouter(name="test-router", routes=table)
        result = router.route("investigate", tract_instance)

        assert result.route.target == "research"


# ===================================================================
# Tract.routing.route() / add_route() / remove_route() integration
# ===================================================================


class TestTractRoutingIntegration:
    def test_add_and_remove_route(self):
        table = RoutingTable()
        table.add_route("feature", "Feature branch", "branch",
                     keywords=["feature", "new"])
        results = table.match("new feature")
        assert results[0].target == "feature"

        table.remove_route("feature")
        results = table.match("new feature")
        # After removal, should not match "feature" anymore
        assert len(results) == 0 or results[0].target != "feature"

    def test_route_fuzzy_no_router(self):
        """RoutingTable.match() should use fuzzy matching."""
        table = RoutingTable()
        table.add_route("design", "Design phase", "stage",
                     keywords=["design", "plan"])
        table.add_route("build", "Build phase", "stage",
                     keywords=["build", "implement"])

        results = table.match("let's plan the design")
        assert len(results) > 0
        assert results[0].target == "design"

    def test_route_with_semantic_router(self):
        """SemanticRouter.route() with a tract should use LLM."""
        t = Tract.open()
        table = RoutingTable()
        table.add_route("alpha", "Alpha branch", "branch",
                         keywords=["alpha"])

        llm_response = json.dumps({
            "target": "alpha",
            "confidence": 0.95,
            "reasoning": "direct match"
        })
        mock = MockLLMClient([llm_response])
        t.config.configure_llm(mock)

        router = SemanticRouter(name="r", routes=table)
        result = router.route("go to alpha", t)

        assert result.route.target == "alpha"
        assert result.method == "semantic"

    def test_route_empty_table_returns_no_matches(self):
        """Route with empty table should return no matches."""
        table = RoutingTable()
        results = table.match("anything at all")
        assert len(results) == 0

    def test_remove_route_no_table_raises(self):
        """Removing a route before any routes are added should raise."""
        table = RoutingTable()
        with pytest.raises(ValueError, match="not found"):
            table.remove_route("nonexistent")

    def test_aroute_async(self):
        """Async route should work."""
        t = Tract.open()
        table = RoutingTable()
        table.add_route("research", "Research", "branch",
                     keywords=["research"])

        router = SemanticRouter(name="test", routes=table)
        result = asyncio.run(
            router.aroute("research topic", t)
        )
        assert result.method == "fuzzy"
        assert result.route.target == "research"
