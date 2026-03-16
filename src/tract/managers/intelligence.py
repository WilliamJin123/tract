"""Intelligence manager extracted from IntelligenceMixin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from tract.autonomous import AutoBranchResult, AutoRebaseResult, AutoSplitResult
    from tract.intelligence import CherryPickResult, DedupResult


class IntelligenceManager:
    """Context intelligence (cherry-pick, dedup) and autonomous ops (auto_split, auto_rebase, auto_branch).

    Standalone replacement for :class:`IntelligenceMixin` with explicit
    constructor dependencies.  All intelligence/autonomous functions take a
    tract instance as their first argument; this manager stores a reference
    to the owning tract and passes it through.
    """

    def __init__(
        self,
        tract_id: str,
        check_open: Callable[[], None],
        tract_ref: Any,  # The tract instance (intelligence functions take tract as arg)
    ) -> None:
        self._tract_id = tract_id
        self._check_open = check_open
        self._tract_ref = tract_ref

    # ------------------------------------------------------------------
    # Intelligence operations
    # ------------------------------------------------------------------

    def cherry_pick(
        self,
        query: str,
        *,
        limit: int = 10,
        **llm_kwargs: Any,
    ) -> CherryPickResult:
        """Select the most relevant commits for a task/query using LLM judgment.

        Builds a manifest of recent commits (with content previews) and asks
        the LLM to select the most relevant ones for the given query.

        Fail-open: on LLM error, returns all candidate commits (no filtering).

        Args:
            query: Natural-language task or query description.
            limit: Maximum number of commits to select (default 10).
            **llm_kwargs: Passed to the LLM client (``model``, ``temperature``,
                ``max_tokens``).

        Returns:
            :class:`~tract.intelligence.CherryPickResult` with selected
            commit hashes and reasoning.

        Example::

            result = t.cherry_pick("Implement the auth module", limit=5)
            for h in result.selected_hashes:
                print(t.get_content(h))
        """
        self._check_open()
        from tract.intelligence import cherry_pick as _cherry_pick

        return _cherry_pick(self._tract_ref, query, limit=limit, **llm_kwargs)

    async def acherry_pick(
        self,
        query: str,
        *,
        limit: int = 10,
        **llm_kwargs: Any,
    ) -> CherryPickResult:
        """Async version of :meth:`cherry_pick`."""
        self._check_open()
        from tract.intelligence import acherry_pick as _acherry_pick

        return await _acherry_pick(self._tract_ref, query, limit=limit, **llm_kwargs)

    def deduplicate(
        self,
        *,
        threshold: float = 0.8,
        auto_skip: bool = False,
        **llm_kwargs: Any,
    ) -> DedupResult:
        """Detect and optionally handle duplicate/overlapping commits using LLM judgment.

        Builds a manifest of recent commits with content previews and asks
        the LLM to identify groups of duplicate or highly overlapping content.

        If ``auto_skip=True``, annotates all but the newest commit in each
        duplicate group as SKIP.

        Fail-open: on LLM error, returns empty groups (no action taken).

        Args:
            threshold: Similarity threshold hint (0.0-1.0). Higher = stricter.
            auto_skip: If True, automatically mark older duplicates as SKIP.
            **llm_kwargs: Passed to the LLM client (``model``, ``temperature``,
                ``max_tokens``).

        Returns:
            :class:`~tract.intelligence.DedupResult` with duplicate groups
            and actions taken.

        Example::

            result = t.deduplicate(auto_skip=True)
            print(f"Found {len(result.duplicate_groups)} duplicate groups")
            print(f"Skipped {result.actions_taken} duplicate commits")
        """
        self._check_open()
        from tract.intelligence import deduplicate as _deduplicate

        return _deduplicate(
            self._tract_ref, threshold=threshold, auto_skip=auto_skip, **llm_kwargs
        )

    async def adeduplicate(
        self,
        *,
        threshold: float = 0.8,
        auto_skip: bool = False,
        **llm_kwargs: Any,
    ) -> DedupResult:
        """Async version of :meth:`deduplicate`."""
        self._check_open()
        from tract.intelligence import adeduplicate as _adeduplicate

        return await _adeduplicate(
            self._tract_ref, threshold=threshold, auto_skip=auto_skip, **llm_kwargs
        )

    # ------------------------------------------------------------------
    # Autonomous operations
    # ------------------------------------------------------------------

    def auto_split(self, commit_hash: str, **llm_kwargs: Any) -> AutoSplitResult:
        """Split a commit into smaller, logically coherent pieces using LLM judgment.

        Gets the commit content, asks an LLM to split it, then creates new
        APPEND commits for each piece and SKIPs the original.

        Fail-open: on LLM error, returns original hash unchanged.

        Args:
            commit_hash: Hash of the commit to split.
            **llm_kwargs: Forwarded to LLM (model, temperature, max_tokens).

        Returns:
            :class:`~tract.autonomous.AutoSplitResult`.
        """
        self._check_open()
        from tract.autonomous import auto_split as _auto_split

        return _auto_split(self._tract_ref, commit_hash, **llm_kwargs)

    async def aauto_split(self, commit_hash: str, **llm_kwargs: Any) -> AutoSplitResult:
        """Async version of :meth:`auto_split`."""
        self._check_open()
        from tract.autonomous import aauto_split as _aauto_split

        return await _aauto_split(self._tract_ref, commit_hash, **llm_kwargs)

    def auto_rebase(self, **llm_kwargs: Any) -> AutoRebaseResult:
        """Decide whether to rebase the current branch using LLM judgment.

        Builds a manifest of branch state and asks the LLM whether a rebase
        would be beneficial. If yes, executes the rebase.

        Fail-open: on error, returns rebased=False.

        Args:
            **llm_kwargs: Forwarded to LLM (model, temperature, max_tokens).

        Returns:
            :class:`~tract.autonomous.AutoRebaseResult`.
        """
        self._check_open()
        from tract.autonomous import auto_rebase as _auto_rebase

        return _auto_rebase(self._tract_ref, **llm_kwargs)

    async def aauto_rebase(self, **llm_kwargs: Any) -> AutoRebaseResult:
        """Async version of :meth:`auto_rebase`."""
        self._check_open()
        from tract.autonomous import aauto_rebase as _aauto_rebase

        return await _aauto_rebase(self._tract_ref, **llm_kwargs)

    def auto_branch(self, *, context: str = "", **llm_kwargs: Any) -> AutoBranchResult:
        """Decide whether to create a new branch using LLM judgment.

        Builds a manifest of current state and asks the LLM whether a new
        branch should be created. If yes, creates and switches to it.

        Fail-open: on error, returns branched=False.

        Args:
            context: Optional task/context description to inform the decision.
            **llm_kwargs: Forwarded to LLM (model, temperature, max_tokens).

        Returns:
            :class:`~tract.autonomous.AutoBranchResult`.
        """
        self._check_open()
        from tract.autonomous import auto_branch as _auto_branch

        return _auto_branch(self._tract_ref, context=context, **llm_kwargs)

    async def aauto_branch(self, *, context: str = "", **llm_kwargs: Any) -> AutoBranchResult:
        """Async version of :meth:`auto_branch`."""
        self._check_open()
        from tract.autonomous import aauto_branch as _aauto_branch

        return await _aauto_branch(self._tract_ref, context=context, **llm_kwargs)
