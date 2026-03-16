"""Routing manager extracted from RoutingMixin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from typing import Any

    from tract.models.branch import BranchInfo
    from tract.routing import Route, RoutingResult, RoutingTable, SemanticRouter

logger = logging.getLogger(__name__)


class RoutingManager:
    """Routing operations: add, remove, route, aroute, and helpers.

    Standalone replacement for :class:`RoutingMixin` with explicit
    constructor dependencies instead of ``self`` on the Tract instance.
    """

    def __init__(
        self,
        tract_id: str,
        ref_repo: Any,
        commit_repo: Any,
        check_open: Callable[[], None],
        commit_session: Callable[[], Any],
        # Callbacks for apply actions:
        list_branches_fn: Callable[[], list[BranchInfo]],
        checkout_fn: Callable[[str], None],
        branch_fn: Callable[[str], None],
        apply_stage_fn: Callable[[str], None],
        load_profile_fn: Callable[[str], None],
    ) -> None:
        self._tract_id = tract_id
        self._ref_repo = ref_repo
        self._commit_repo = commit_repo
        self._check_open = check_open
        self._commit_session = commit_session
        self._list_branches_fn = list_branches_fn
        self._checkout_fn = checkout_fn
        self._branch_fn = branch_fn
        self._apply_stage_fn = apply_stage_fn
        self._load_profile_fn = load_profile_fn

        # Lazy-initialised state
        self._routing_table: RoutingTable | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        name: str,
        description: str,
        route_type: str,
        *,
        keywords: list[str] | None = None,
        pattern: str | None = None,
    ) -> None:
        """Register a route in the default routing table.

        Args:
            name: Unique route identifier.
            description: Human-readable description (used for fuzzy matching).
            route_type: ``"branch"``, ``"stage"``, or ``"workflow"``.
            keywords: Optional keywords that improve matching accuracy.
            pattern: Optional regex pattern for exact matching.

        Raises:
            ValueError: If *name* is already registered or *route_type* is invalid.

        Example::

            t.add_route("research", "Deep research branch", "branch",
                         keywords=["investigate", "research", "explore"])
        """
        self._check_open()
        self._ensure_table()
        self._routing_table.add_route(  # type: ignore[union-attr]
            name, description, route_type, keywords=keywords, pattern=pattern
        )

    def remove(self, name: str) -> None:
        """Remove a route from the default routing table.

        Args:
            name: The route name to remove.

        Raises:
            ValueError: If no route with this name exists or no routing table.
        """
        self._check_open()
        if self._routing_table is None:
            raise ValueError(f"Route '{name}' not found.")
        self._routing_table.remove_route(name)

    def route(
        self,
        query: str,
        *,
        router: SemanticRouter | None = None,
        apply: bool = False,
    ) -> RoutingResult:
        """Route a query to the best matching branch, stage, or workflow.

        Uses a :class:`~tract.routing.SemanticRouter` for LLM-powered
        routing, falling back to fuzzy matching from the default routing
        table.

        Args:
            query: The user query or intent string.
            router: An optional :class:`~tract.routing.SemanticRouter`.
                If ``None``, uses the default routing table with fuzzy matching only.
            apply: If ``True``, automatically apply the route (switch branch,
                apply stage, etc.).

        Returns:
            A :class:`~tract.routing.RoutingResult`.

        Example::

            result = t.route("time to start implementing")
            print(result.route.target, result.route.confidence)
        """
        self._check_open()
        from tract.routing import SemanticRouter

        if router is not None and isinstance(router, SemanticRouter):
            result = router.route(query, self)
        else:
            result = self._fallback(query)

        return self._apply_result(result, apply)

    async def aroute(
        self,
        query: str,
        *,
        router: SemanticRouter | None = None,
        apply: bool = False,
    ) -> RoutingResult:
        """Async version of :meth:`route`.

        Uses ``aroute()`` on the SemanticRouter if provided.
        """
        self._check_open()
        from tract.routing import SemanticRouter

        if router is not None and isinstance(router, SemanticRouter):
            result = await router.aroute(query, self)
        else:
            result = self._fallback(query)

        return self._apply_result(result, apply)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply(self, route: Route) -> bool:
        """Apply a route (switch branch, apply stage, etc.).

        Returns True if successfully applied, False otherwise.
        """
        from tract.routing import Route

        if not isinstance(route, Route):
            return False
        try:
            if route.route_type == "branch":
                existing = {b.name for b in self._list_branches_fn()}
                if route.target in existing:
                    self._checkout_fn(route.target)
                else:
                    self._branch_fn(route.target)
                return True
            elif route.route_type == "stage":
                self._apply_stage_fn(route.target)
                return True
            elif route.route_type == "workflow":
                self._load_profile_fn(route.target)
                return True
        except Exception:
            logger.warning(
                "Failed to apply route '%s' (%s): %s",
                route.target,
                route.route_type,
                exc_info=True,
            )
        return False

    def _ensure_table(self) -> None:
        """Lazily initialize the default routing table."""
        if self._routing_table is None:
            from tract.routing import RoutingTable

            self._routing_table = RoutingTable()

    def _fallback(self, query: str) -> RoutingResult:
        """Fuzzy-only routing when no SemanticRouter is provided."""
        from tract.routing import Route, RoutingResult

        self._ensure_table()
        matches = self._routing_table.match(query)  # type: ignore[union-attr]
        if matches:
            best = matches[0]
        else:
            best = Route(
                target="",
                route_type="branch",
                confidence=0.0,
                reasoning="No matching routes found.",
            )
        return RoutingResult(
            route=best,
            applied=False,
            tokens_used=0,
            method="fuzzy",
        )

    def _apply_result(self, result: RoutingResult, apply: bool) -> RoutingResult:
        """Optionally apply a routing result."""
        if apply and result.route.target and result.route.confidence > 0:
            from tract.routing import RoutingResult

            applied = self._apply(result.route)
            return RoutingResult(
                route=result.route,
                applied=applied,
                tokens_used=result.tokens_used,
                method=result.method,
            )
        return result
