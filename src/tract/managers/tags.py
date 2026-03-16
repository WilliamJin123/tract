"""Tag manager for Tract.

Extracted from TagMixin (_tags.py) into a standalone class with explicit
constructor dependencies.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from tract.models.commit import CommitInfo, CommitOperation
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitRepository,
        ParentRepository,
        TagAnnotationRepository,
        TagRegistryRepository,
    )


class TagManager:
    """Tag operations: add, remove, get, register, list, query, seed, validate, classify."""

    def __init__(
        self,
        tract_id: str,
        get_tag_annotation_repo: Callable,  # lambda -> repo (late-bound)
        get_tag_registry_repo: Callable,  # lambda -> repo (late-bound)
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        annotation_repo: AnnotationRepository,
        parent_repo: ParentRepository,
        get_strict_tags: Callable,  # lambda -> bool (late-bound)
        check_open: Callable[[], None],
        commit_session: Callable[[], None],
        get_ancestors: Callable,  # was _get_merge_aware_ancestors
        row_to_info: Callable,  # was _commit_engine._row_to_info
        get_head: Callable[[], str | None],
    ) -> None:
        self._tract_id = tract_id
        self._get_tag_annotation_repo = get_tag_annotation_repo
        self._get_tag_registry_repo = get_tag_registry_repo
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._annotation_repo = annotation_repo
        self._parent_repo = parent_repo
        self._get_strict_tags = get_strict_tags
        self._check_open = check_open
        self._commit_session = commit_session
        self._get_ancestors = get_ancestors
        self._row_to_info = row_to_info
        self._get_head = get_head

    def add(self, target_hash: str, tag_name: str) -> None:
        """Add a mutable tag annotation to a commit.

        Unlike immutable tags set at commit time, annotation tags can be
        added retrospectively.

        Args:
            target_hash: Hash of the commit to tag.
            tag_name: Tag name to add.

        Raises:
            CommitNotFoundError: If the commit doesn't exist.
            TagNotRegisteredError: If strict mode is on and tag is not registered.
        """
        self._check_open()
        from tract.exceptions import CommitNotFoundError, TagNotRegisteredError

        commit = self._commit_repo.get(target_hash)
        if commit is None:
            raise CommitNotFoundError(target_hash)
        if self._get_strict_tags() and self._get_tag_registry_repo() is not None:
            if not self._get_tag_registry_repo().is_registered(self._tract_id, tag_name):
                raise TagNotRegisteredError(tag_name)
        if self._get_tag_annotation_repo() is not None:
            from datetime import timezone

            now = datetime.now(timezone.utc)
            self._get_tag_annotation_repo().add_tag(
                self._tract_id, target_hash, tag_name, now,
            )
            self._commit_session()

    def remove(self, target_hash: str, tag_name: str) -> bool:
        """Remove a mutable tag annotation from a commit.

        Args:
            target_hash: Hash of the commit to untag.
            tag_name: Tag name to remove.

        Returns:
            True if the tag was removed, False if it didn't exist.
        """
        self._check_open()
        if self._get_tag_annotation_repo() is None:
            return False
        result = self._get_tag_annotation_repo().remove_tag(
            self._tract_id, target_hash, tag_name,
        )
        self._commit_session()
        return result

    def get(self, target_hash: str) -> list[str]:
        """Get all tags for a commit (immutable + mutable combined).

        Args:
            target_hash: Hash of the commit.

        Returns:
            Deduplicated list of tag names.
        """
        tags: set[str] = set()
        # Immutable tags from CommitRow
        commit_row = self._commit_repo.get(target_hash)
        if commit_row is not None and commit_row.tags_json:
            tags.update(commit_row.tags_json)
        # Mutable annotation tags
        if self._get_tag_annotation_repo() is not None:
            annotation_tags = self._get_tag_annotation_repo().get_tags(target_hash)
            tags.update(annotation_tags)
        return sorted(tags)

    def register(self, name: str, description: str | None = None) -> None:
        """Register a new tag name.

        Registered tags can be used with ``add()`` and ``commit(tags=[...])``.
        In strict mode (default), only registered tags are allowed.

        Args:
            name: Tag name.
            description: Optional description of the tag.
        """
        if self._get_tag_registry_repo() is None:
            return
        from datetime import timezone

        now = datetime.now(timezone.utc)
        self._get_tag_registry_repo().register(
            self._tract_id, name, description,
            auto_created=False, created_at=now,
        )
        self._commit_session()

    def list(self) -> list[dict]:
        """List all registered tags with descriptions and usage counts.

        Returns:
            List of dicts with ``name``, ``description``, ``auto_created``,
            and ``count`` keys.
        """
        if self._get_tag_registry_repo() is None:
            return []
        rows = self._get_tag_registry_repo().list_all(self._tract_id)
        tag_names = [r.tag_name for r in rows]

        # Count annotation tags in batch
        annotation_counts: dict[str, int] = {}
        if self._get_tag_annotation_repo() is not None:
            for tn in tag_names:
                annotation_hashes = self._get_tag_annotation_repo().get_commits_by_tag(
                    self._tract_id, tn,
                )
                annotation_counts[tn] = len(annotation_hashes)

        # Walk ancestors ONCE, count immutable tags across all registered tags
        immutable_counts: dict[str, int] = {tn: 0 for tn in tag_names}
        head = self._get_head()
        if head is not None:
            tag_name_set = set(tag_names)
            ancestors = self._get_ancestors(head, limit=500)
            for ancestor in ancestors:
                if ancestor.tags_json:
                    for t in ancestor.tags_json:
                        if t in tag_name_set:
                            immutable_counts[t] = immutable_counts.get(t, 0) + 1

        result = []
        for row in rows:
            count = annotation_counts.get(row.tag_name, 0) + immutable_counts.get(row.tag_name, 0)
            result.append({
                "name": row.tag_name,
                "description": row.description,
                "auto_created": bool(row.auto_created),
                "count": count,
            })
        return result

    def query(
        self,
        tags: list[str],
        *,
        match: str = "any",
        limit: int = 100,
    ) -> list[CommitInfo]:
        """Query commits by tags (combining immutable and mutable tags).

        Args:
            tags: Tag names to filter by.
            match: ``"any"`` (OR -- commit has at least one tag) or
                ``"all"`` (AND -- commit has every listed tag).
            limit: Maximum results.

        Returns:
            List of :class:`CommitInfo` matching the tag criteria.
        """
        if not tags:
            return []

        head = self._get_head()
        if head is None:
            return []

        # Walk ancestors ONCE and reuse
        ancestors = self._get_ancestors(head, limit=500)

        # Batch-fetch annotation tags for all ancestors
        annotation_map: dict[str, list[str]] = {}
        if self._get_tag_annotation_repo() is not None:
            all_hashes = [r.commit_hash for r in ancestors]
            annotation_map = self._get_tag_annotation_repo().batch_get_tags(all_hashes)

        tag_set = set(tags)
        results: list[CommitInfo] = []
        for row in ancestors:
            # Combine immutable + mutable tags
            commit_tags = set(row.tags_json) if row.tags_json else set()
            commit_tags.update(annotation_map.get(row.commit_hash, []))

            if match == "any":
                if commit_tags & tag_set:
                    results.append(self._row_to_info(row))
            else:  # "all"
                if tag_set <= commit_tags:
                    results.append(self._row_to_info(row))

            if len(results) >= limit:
                break

        return results

    def _seed_base(self) -> None:
        """Seed the tag registry with base tags (idempotent)."""
        if self._get_tag_registry_repo() is None:
            return

        from datetime import timezone

        now = datetime.now(timezone.utc)

        base_tags = {
            "instruction": "System messages / instructions",
            "tool_call": "Messages containing tool calls",
            "tool_result": "Tool result messages",
            "reasoning": "Assistant reasoning without tool calls",
            "revision": "EDIT operations",
            "observation": "User messages with data / observations",
            "decision": "Assistant messages with explicit choices",
            "summary": "Compression output / summaries",
        }
        for tag_name, description in base_tags.items():
            self._get_tag_registry_repo().register(
                self._tract_id, tag_name, description,
                auto_created=True, created_at=now,
            )
        self._commit_session()

    def _validate(self, tags: list[str]) -> None:
        """Validate tags against registry in strict mode.

        Raises:
            TagNotRegisteredError: If any tag is not registered (reports all
                unregistered tags at once).
        """
        if not self._get_strict_tags() or self._get_tag_registry_repo() is None:
            return
        from tract.exceptions import TagNotRegisteredError

        registered = self._get_tag_registry_repo().batch_is_registered(self._tract_id, tags)
        unregistered = [tag for tag in tags if tag not in registered]
        if unregistered:
            raise TagNotRegisteredError(unregistered)

    def _classify(
        self,
        content_type: str,
        *,
        role: str | None = None,
        operation: CommitOperation | None = None,
        metadata: dict | None = None,
    ) -> list[str]:
        """Heuristic-based tag classification (no LLM call).

        Args:
            content_type: The content type discriminator.
            role: The message role (if dialogue).
            operation: The commit operation.
            metadata: The commit metadata.

        Returns:
            Deduplicated list of tag names.
        """
        from tract.models.commit import CommitOperation

        tags: list[str] = []

        # Classify based on content type and role
        if content_type == "instruction" or role == "system":
            tags.append("instruction")
        if role == "assistant":
            if metadata and metadata.get("tool_calls"):
                tags.append("tool_call")
            else:
                tags.append("reasoning")
        if role == "user":
            if metadata and metadata.get("tool_call_id"):
                tags.append("tool_result")
        if role == "tool":
            tags.append("tool_result")
        if content_type == "tool_io":
            if "tool_call" not in tags and "tool_result" not in tags:
                tags.append("tool_call")
        if operation == CommitOperation.EDIT:
            tags.append("revision")
        if content_type == "session" or (content_type and "session" in content_type):
            tags.append("observation")

        # Deduplicate preserving order
        seen: set[str] = set()
        unique_tags: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags
