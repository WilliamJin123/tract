"""Annotation manager for Tract.

Extracted from AnnotationMixin (_annotations.py) into a standalone class
with explicit constructor dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tract.models.annotations import DEFAULT_TYPE_PRIORITIES, Priority, PriorityAnnotation, RetentionCriteria

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.engine.cache import CacheManager
    from tract.engine.commit import CommitEngine
    from tract.models.commit import CommitInfo
    from tract.storage.repositories import AnnotationRepository, CommitRepository


class AnnotationManager:
    """Priority annotations: set, get, counts, enrich."""

    def __init__(
        self,
        tract_id: str,
        annotation_repo: AnnotationRepository,
        commit_repo: CommitRepository,
        commit_engine: CommitEngine,
        cache: CacheManager,
        check_open: Callable[[], None],
        commit_session: Callable[[], None],
        get_head: Callable[[], str | None],
        log_fn: Callable,  # Tract.log (for annotation_counts)
    ) -> None:
        self._tract_id = tract_id
        self._annotation_repo = annotation_repo
        self._commit_repo = commit_repo
        self._commit_engine = commit_engine
        self._cache = cache
        self._check_open = check_open
        self._commit_session = commit_session
        self._get_head = get_head
        self._log_fn = log_fn

    def set(
        self,
        target_hash: str,
        priority: Priority,
        *,
        reason: str | None = None,
        retain: str | None = None,
        retain_match: list[str] | None = None,
        retain_match_mode: str = "substring",
    ) -> PriorityAnnotation:
        """Create a priority annotation on a commit.

        Args:
            target_hash: Hash of the commit to annotate.
            priority: Priority level (``SKIP``, ``NORMAL``, ``IMPORTANT``, ``PINNED``).
            reason: Optional reason for the annotation.
            retain: Fuzzy retention instructions (NL guidance for the LLM
                summarizer). Only meaningful for ``IMPORTANT`` priority.
            retain_match: Deterministic retention patterns -- substrings or
                regexes that MUST appear in compression summaries.
            retain_match_mode: How ``retain_match`` patterns are checked:
                ``"substring"`` (default) or ``"regex"``.

        Returns:
            :class:`PriorityAnnotation` model.
        """
        self._check_open()
        retention = None
        if retain is not None or retain_match is not None:
            retention = RetentionCriteria(
                instructions=retain,
                match_patterns=retain_match,
                match_mode=retain_match_mode,
            )
        annotation = self._commit_engine.annotate(
            target_hash, priority, reason, retention=retention
        )
        self._commit_session()

        # Annotations affect ALL cached snapshots that include the target commit.
        # Strategy: clear everything, then optionally re-add a patched current HEAD.
        # Exception: if patch returns the same snapshot object (NORMAL/PINNED on
        # already-included commit), the annotation is a no-op for compiled output
        # and we can skip the clear entirely.
        if self._cache.uses_default_compiler:
            current_head = self._get_head()
            patched = None
            if current_head:
                snapshot = self._cache.get(current_head)
                if snapshot is not None:
                    patched = self._cache.patch_for_annotate(
                        snapshot, target_hash, priority
                    )
            if patched is not None and patched is snapshot:
                pass  # No-op: annotation didn't change compiled output
            else:
                self._cache.clear()
                if patched is not None:
                    self._cache.put(current_head, patched)
        else:
            self._cache.clear()

        return annotation

    def get(self, target_hash: str) -> list[PriorityAnnotation]:
        """Get the full annotation history for a commit.

        Returns:
            List of :class:`PriorityAnnotation` in chronological order.
        """
        rows = self._annotation_repo.get_history(target_hash)
        return [
            PriorityAnnotation(
                id=row.id,
                tract_id=row.tract_id,
                target_hash=row.target_hash,
                priority=row.priority,
                reason=row.reason,
                retention=RetentionCriteria(**row.retention_json)
                if row.retention_json else None,
                created_at=row.created_at,
            )
            for row in rows
        ]

    def counts(self, limit: int = 500) -> dict[str, int]:
        """Count pinned and skipped annotations across recent commits.

        Args:
            limit: Maximum number of commits to scan. Default 500.

        Returns:
            Dict with ``"pinned"`` and ``"skip"`` integer counts.
        """
        entries = self._log_fn(limit=limit)
        commit_hashes = [e.commit_hash for e in entries]
        pinned = 0
        skip = 0
        if commit_hashes:
            annotations = self._annotation_repo.batch_get_latest(commit_hashes)
            for _hash, ann_row in annotations.items():
                if ann_row.priority == Priority.PINNED:
                    pinned += 1
                elif ann_row.priority == Priority.SKIP:
                    skip += 1
        return {"pinned": pinned, "skip": skip}

    def _enrich_with_priorities(self, entries: list[CommitInfo]) -> list[CommitInfo]:
        """Enrich CommitInfo entries with effective_priority.

        Resolves each commit's effective priority by checking explicit
        annotations first, then falling back to DEFAULT_TYPE_PRIORITIES.
        """
        if not entries:
            return entries
        hashes = [e.commit_hash for e in entries]
        annotations = self._annotation_repo.batch_get_latest(hashes)
        enriched: list[CommitInfo] = []
        for entry in entries:
            ann = annotations.get(entry.commit_hash)
            if ann is not None:
                priority = ann.priority
            else:
                priority = DEFAULT_TYPE_PRIORITIES.get(
                    entry.content_type, Priority.NORMAL,
                )
            enriched.append(entry.model_copy(update={"effective_priority": priority.value}))
        return enriched
