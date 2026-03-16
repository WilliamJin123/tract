"""Workflow routing for tract.

Provides fuzzy and LLM-powered routing of queries/intents to branches,
stages, or workflows.

Example (fuzzy only)::

    from tract.routing import RoutingTable

    table = RoutingTable()
    table.add_route("design", "Software design phase", "stage",
                     keywords=["architecture", "plan", "design"])
    matches = table.match("I need to plan the architecture")

Example (semantic)::

    from tract.routing import SemanticRouter, RoutingTable

    table = RoutingTable()
    table.add_route("research", "Deep research branch", "branch",
                     keywords=["investigate", "research"])
    router = SemanticRouter(name="main-router", routes=table)
    result = router.route("Let's do some investigation", tract=t)
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract._helpers import strip_fences as _strip_fences

if TYPE_CHECKING:
    from tract.tract import Tract

__all__: list[str] = [
    "Route",
    "RoutingTable",
    "SemanticRouter",
    "RoutingResult",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Route — immutable routing result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Route:
    """A single routing destination with confidence score.

    Attributes:
        target: Branch name, stage name, or workflow name.
        route_type: One of ``"branch"``, ``"stage"``, ``"workflow"``.
        confidence: Match confidence from 0.0 to 1.0.
        reasoning: Human-readable explanation of why this route matched.
    """

    target: str
    route_type: str  # "branch", "stage", "workflow"
    confidence: float  # 0.0–1.0
    reasoning: str


# ---------------------------------------------------------------------------
# RoutingResult — outcome of a route() call
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RoutingResult:
    """Result of a routing operation.

    Attributes:
        route: The selected Route.
        applied: Whether the route was applied (branch switched, stage applied, etc.).
        tokens_used: Tokens consumed by the LLM call (0 for fuzzy/exact).
        method: How the route was resolved — ``"semantic"``, ``"fuzzy"``, or ``"exact"``.
    """

    route: Route
    applied: bool
    tokens_used: int
    method: str  # "semantic", "fuzzy", "exact"


# ---------------------------------------------------------------------------
# Internal: registered route entry
# ---------------------------------------------------------------------------
@dataclass
class _RouteEntry:
    """Internal storage for a registered route."""

    name: str
    description: str
    route_type: str
    keywords: list[str]
    pattern: str | None


# ---------------------------------------------------------------------------
# RoutingTable — fuzzy matching registry
# ---------------------------------------------------------------------------
class RoutingTable:
    """Registry of routes with fuzzy matching support.

    Routes are registered with a name, description, type, and optional
    keywords/regex patterns.  The :meth:`match` method scores a query
    against all registered routes using substring matching and
    ``difflib.SequenceMatcher`` for keyword similarity.

    Example::

        table = RoutingTable()
        table.add_route("implement", "Implementation stage", "stage",
                         keywords=["code", "implement", "build"])
        routes = table.match("time to start coding")
    """

    def __init__(self) -> None:
        self._routes: dict[str, _RouteEntry] = {}

    def add_route(
        self,
        name: str,
        description: str,
        route_type: str,
        *,
        keywords: list[str] | None = None,
        pattern: str | None = None,
    ) -> None:
        """Register a route.

        Args:
            name: Unique route identifier.
            description: Human-readable description (used for fuzzy matching).
            route_type: ``"branch"``, ``"stage"``, or ``"workflow"``.
            keywords: Optional keywords that improve matching accuracy.
            pattern: Optional regex pattern for exact matching.

        Raises:
            ValueError: If *name* is already registered or *route_type* is
                not one of the valid types.
        """
        valid_types = {"branch", "stage", "workflow"}
        if route_type not in valid_types:
            raise ValueError(
                f"Invalid route_type '{route_type}'. Must be one of: {sorted(valid_types)}"
            )
        if name in self._routes:
            raise ValueError(f"Route '{name}' already registered. Remove it first.")
        self._routes[name] = _RouteEntry(
            name=name,
            description=description,
            route_type=route_type,
            keywords=keywords or [],
            pattern=pattern,
        )

    def remove_route(self, name: str) -> None:
        """Remove a registered route.

        Args:
            name: The route name to remove.

        Raises:
            ValueError: If no route with this name exists.
        """
        if name not in self._routes:
            raise ValueError(f"Route '{name}' not found.")
        del self._routes[name]

    def list_routes(self) -> list[str]:
        """Return names of all registered routes."""
        return list(self._routes.keys())

    def match(self, query: str) -> list[Route]:
        """Fuzzy-match a query against registered routes.

        Scoring (each component contributes 0.0–1.0, then averaged):

        1. **Regex pattern** — if a route has a ``pattern`` and the query
           matches, that route gets a 1.0 pattern score.
        2. **Keyword similarity** — best ``SequenceMatcher.ratio()`` between
           any query word and any route keyword.
        3. **Description similarity** — ``SequenceMatcher.ratio()`` between
           the full query and the route description.
        4. **Substring boost** — if any keyword appears as a substring in
           the query (case-insensitive), adds 0.3 to the score (capped at 1.0).

        Returns:
            Routes sorted by confidence (highest first).  Only routes with
            confidence > 0 are included.
        """
        if not self._routes:
            return []

        query_lower = query.lower()
        query_words = query_lower.split()
        results: list[Route] = []

        for entry in self._routes.values():
            score = self._score_entry(entry, query_lower, query_words)
            if score > 0.0:
                results.append(
                    Route(
                        target=entry.name,
                        route_type=entry.route_type,
                        confidence=round(min(score, 1.0), 4),
                        reasoning=f"Fuzzy match against '{entry.name}' ({entry.description})",
                    )
                )

        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    @staticmethod
    def _score_entry(
        entry: _RouteEntry, query_lower: str, query_words: list[str]
    ) -> float:
        """Score a single route entry against the query."""
        scores: list[float] = []

        # 1. Regex pattern match
        if entry.pattern:
            try:
                if re.search(entry.pattern, query_lower, re.IGNORECASE):
                    scores.append(1.0)
                else:
                    scores.append(0.0)
            except re.error:
                scores.append(0.0)

        # 2. Keyword similarity (best match between any query word and any keyword)
        keyword_score = 0.0
        substring_bonus = 0.0
        if entry.keywords:
            for kw in entry.keywords:
                kw_lower = kw.lower()
                # Substring check
                if kw_lower in query_lower:
                    substring_bonus = 0.3
                # SequenceMatcher against each query word
                for qw in query_words:
                    ratio = difflib.SequenceMatcher(None, qw, kw_lower).ratio()
                    keyword_score = max(keyword_score, ratio)
            scores.append(keyword_score)

        # 3. Description similarity
        desc_ratio = difflib.SequenceMatcher(
            None, query_lower, entry.description.lower()
        ).ratio()
        scores.append(desc_ratio)

        # 4. Name similarity (exact name match is strong signal)
        name_lower = entry.name.lower()
        for qw in query_words:
            name_ratio = difflib.SequenceMatcher(None, qw, name_lower).ratio()
            if name_ratio > 0.8:
                scores.append(name_ratio)
                break

        if not scores:
            return 0.0

        avg = sum(scores) / len(scores) + substring_bonus
        return avg


# ---------------------------------------------------------------------------
# SemanticRouter — LLM-powered routing
# ---------------------------------------------------------------------------

_ROUTER_SYSTEM_PROMPT = """\
You are a routing agent. Your job is to pick the best route for a user query.

You will receive a list of available routes (with descriptions and keywords) and a query.

Respond with JSON:
{"target": "<route_name>", "confidence": 0.0-1.0, "reasoning": "Brief explanation"}

Pick the single best matching route. If nothing matches well, set confidence below 0.3.
Only use route names from the provided list."""


@dataclass
class SemanticRouter:
    """LLM-powered query router with fuzzy fallback.

    Follows the same manifest-based, fail-open pattern as
    :class:`~tract.gate.SemanticGate` and
    :class:`~tract.maintain.SemanticMaintainer`.

    When the LLM call succeeds the method is ``"semantic"``; on failure
    the router falls back to :meth:`RoutingTable.match` (``"fuzzy"``).

    Attributes:
        name: Human-readable router identifier.
        routes: The routing table to draw candidates from.
        instructions: Optional extra instructions appended to the system prompt.
        model: Model override passed to ``client.chat()``.
        temperature: Sampling temperature for the routing call.
        max_tokens: Max tokens for the LLM response.
    """

    name: str
    routes: RoutingTable
    instructions: str | None = None
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int = 256

    # Last result for inspection
    last_result: RoutingResult | None = field(default=None, init=False, repr=False)

    # ------------------------------------------------------------------
    # Sync route
    # ------------------------------------------------------------------
    def route(self, query: str, tract: Tract) -> RoutingResult:
        """Route a query using the LLM, falling back to fuzzy matching.

        Args:
            query: The user query or intent string.
            tract: A Tract instance (used to resolve the LLM client).

        Returns:
            A :class:`RoutingResult` with the best route and metadata.
        """
        # Try semantic routing first
        try:
            client = tract._resolve_llm_client("route")
        except RuntimeError:
            logger.debug(
                "Router '%s': no LLM client configured; using fuzzy fallback.",
                self.name,
            )
            return self._fuzzy_fallback(query)

        messages = self._build_messages(query)
        llm_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            llm_kwargs["model"] = self.model
        if self.max_tokens is not None:
            llm_kwargs["max_tokens"] = self.max_tokens

        tokens_used = 0
        try:
            response = client.chat(messages, **llm_kwargs)
        except Exception:
            logger.warning(
                "Router '%s' LLM call failed; using fuzzy fallback (fail-open).",
                self.name,
                exc_info=True,
            )
            return self._fuzzy_fallback(query)

        try:
            raw_text = client.extract_content(response)
        except Exception:
            logger.warning(
                "Router '%s' failed to extract LLM response; using fuzzy fallback.",
                self.name,
                exc_info=True,
            )
            return self._fuzzy_fallback(query)

        # Track token usage
        try:
            usage = client.extract_usage(response) if hasattr(client, "extract_usage") else None
            if usage and isinstance(usage, dict):
                tokens_used = int(usage.get("total_tokens", 0))
        except Exception:
            pass

        # Parse response
        route = self._parse_response(raw_text, query)
        result = RoutingResult(
            route=route,
            applied=False,
            tokens_used=tokens_used,
            method="semantic",
        )
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Async route
    # ------------------------------------------------------------------
    async def aroute(self, query: str, tract: Tract) -> RoutingResult:
        """Async version of :meth:`route`.

        Uses ``achat()`` if the client supports it, otherwise wraps
        the sync ``chat()`` via ``asyncio.to_thread()``.
        """
        from tract.llm.protocols import acall_llm

        try:
            client = tract._resolve_llm_client("route")
        except RuntimeError:
            logger.debug(
                "Router '%s': no LLM client configured; using fuzzy fallback.",
                self.name,
            )
            return self._fuzzy_fallback(query)

        messages = self._build_messages(query)
        llm_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            llm_kwargs["model"] = self.model
        if self.max_tokens is not None:
            llm_kwargs["max_tokens"] = self.max_tokens

        tokens_used = 0
        try:
            response = await acall_llm(client, messages, **llm_kwargs)
        except Exception:
            logger.warning(
                "Router '%s' async LLM call failed; using fuzzy fallback.",
                self.name,
                exc_info=True,
            )
            return self._fuzzy_fallback(query)

        try:
            raw_text = client.extract_content(response)
        except Exception:
            logger.warning(
                "Router '%s' failed to extract async LLM response; using fuzzy fallback.",
                self.name,
                exc_info=True,
            )
            return self._fuzzy_fallback(query)

        try:
            usage = client.extract_usage(response) if hasattr(client, "extract_usage") else None
            if usage and isinstance(usage, dict):
                tokens_used = int(usage.get("total_tokens", 0))
        except Exception:
            pass

        route = self._parse_response(raw_text, query)
        result = RoutingResult(
            route=route,
            applied=False,
            tokens_used=tokens_used,
            method="semantic",
        )
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Fuzzy fallback
    # ------------------------------------------------------------------
    def _fuzzy_fallback(self, query: str) -> RoutingResult:
        """Fall back to fuzzy matching from the routing table."""
        matches = self.routes.match(query)
        if matches:
            route = matches[0]
        else:
            route = Route(
                target="",
                route_type="branch",
                confidence=0.0,
                reasoning="No matching routes found.",
            )
        result = RoutingResult(
            route=route,
            applied=False,
            tokens_used=0,
            method="fuzzy",
        )
        self.last_result = result
        return result

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def _build_messages(self, query: str) -> list[dict[str, str]]:
        """Build LLM messages for routing."""
        system = _ROUTER_SYSTEM_PROMPT
        if self.instructions:
            system += f"\n\nAdditional instructions:\n{self.instructions}"

        # Build routes manifest
        route_lines: list[str] = ["Available routes:"]
        for name in self.routes.list_routes():
            entry = self.routes._routes[name]
            kw_str = ", ".join(entry.keywords) if entry.keywords else "(none)"
            route_lines.append(
                f"  - {entry.name} [{entry.route_type}]: {entry.description} "
                f"(keywords: {kw_str})"
            )

        user_content = (
            f"{chr(10).join(route_lines)}\n\n"
            f"Query: {query}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    def _parse_response(self, text: str, query: str) -> Route:
        """Parse an LLM routing response into a Route.

        Falls back to fuzzy matching if the response cannot be parsed.
        """
        cleaned = _strip_fences(text)

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                target = str(data.get("target") or "").strip()
                confidence = float(data.get("confidence", 0.0))
                reasoning = str(data.get("reasoning") or "").strip() or "(no reasoning given)"

                # Validate target exists in routing table
                if target and target in self.routes._routes:
                    entry = self.routes._routes[target]
                    return Route(
                        target=target,
                        route_type=entry.route_type,
                        confidence=max(0.0, min(1.0, confidence)),
                        reasoning=reasoning,
                    )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback to fuzzy
        matches = self.routes.match(query)
        if matches:
            return matches[0]
        return Route(
            target="",
            route_type="branch",
            confidence=0.0,
            reasoning=f"Could not parse router response; no fuzzy matches. Raw: {text[:200]}",
        )
