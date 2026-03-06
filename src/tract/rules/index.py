"""Rule index: in-memory index of active rules built from DAG ancestry."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from tract.rules.ancestry import walk_ancestry
from tract.rules.models import RuleEntry

if TYPE_CHECKING:
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
    )


class RuleIndex:
    """In-memory index of active rules, built from DAG ancestry.

    Index structure: dict[(trigger, name) -> RuleEntry]
    When multiple rules share (trigger, name), closest to HEAD wins.
    """

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], RuleEntry] = {}
        self._head_hash: str | None = None
        self._stale: bool = False

    @classmethod
    def build(
        cls,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        head_hash: str,
        *,
        parent_repo: CommitParentRepository | None = None,
        annotation_repo: AnnotationRepository | None = None,
    ) -> RuleIndex:
        """Build index by walking ancestry and collecting RuleContent commits.

        Rules at distance 0 (closest to HEAD) override rules at distance N
        with the same (trigger, name) key.

        Distance computation: walk_ancestry() returns commits in root-to-head
        order. Distance is computed as: distance = len(rule_commits) - 1 - index.
        This gives distance 0 to the commit closest to HEAD (last in the list)
        and the highest distance to the root-most rule commit.
        """
        index = cls()
        index._head_hash = head_hash

        rule_commits = walk_ancestry(
            commit_repo,
            blob_repo,
            head_hash,
            content_type_filter={"rule"},
            parent_repo=parent_repo,
        )

        # Filter out SKIP-annotated rule commits
        if annotation_repo is not None and rule_commits:
            hashes = [c.commit_hash for c in rule_commits]
            annotations = annotation_repo.batch_get_latest(hashes)
            rule_commits = [
                c for c in rule_commits
                if c.commit_hash not in annotations
                or annotations[c.commit_hash].priority != "skip"
            ]

        if not rule_commits:
            return index

        num_rules = len(rule_commits)
        for i, commit in enumerate(rule_commits):
            distance = num_rules - 1 - i

            # Parse rule content from blob
            blob = blob_repo.get(commit.content_hash)
            if blob is None:
                continue

            try:
                payload = json.loads(blob.payload_json)
            except (json.JSONDecodeError, TypeError):
                continue

            name = payload.get("name", "")
            trigger = payload.get("trigger", "")
            condition = payload.get("condition")
            action = payload.get("action", {})

            # Extract provenance from commit metadata
            provenance = None
            if commit.metadata_json:
                provenance = commit.metadata_json.get("provenance")

            entry = RuleEntry(
                name=name,
                trigger=trigger,
                condition=condition,
                action=action,
                commit_hash=commit.commit_hash,
                dag_distance=distance,
                provenance=provenance,
            )

            key = (trigger, name)
            # Closer to HEAD (lower distance) wins
            if key not in index._entries or entry.dag_distance < index._entries[key].dag_distance:
                index._entries[key] = entry

        return index

    def get_by_trigger(self, trigger: str) -> list[RuleEntry]:
        """Get all rules matching a trigger, sorted by dag_distance ascending."""
        matches = [e for e in self._entries.values() if e.trigger == trigger]
        return sorted(matches, key=lambda e: e.dag_distance)

    def get_config(self, key: str) -> Any | None:
        """Resolve a config value from active rules.

        Looks for rules with trigger="active" and
        action={"type": "set_config", "key": key, ...}.
        Returns the value from the closest rule (lowest dag_distance).
        """
        candidates = []
        for entry in self._entries.values():
            if (
                entry.trigger == "active"
                and isinstance(entry.action, dict)
                and entry.action.get("type") == "set_config"
                and entry.action.get("key") == key
            ):
                candidates.append(entry)

        if not candidates:
            return None

        # Closest to HEAD wins (lowest dag_distance)
        closest = min(candidates, key=lambda e: e.dag_distance)
        return closest.action.get("value")

    def get_all_configs(self) -> dict[str, Any]:
        """Resolve all active config key-value pairs."""
        # Group by config key, pick closest for each
        config_entries: dict[str, list[RuleEntry]] = {}
        for entry in self._entries.values():
            if (
                entry.trigger == "active"
                and isinstance(entry.action, dict)
                and entry.action.get("type") == "set_config"
            ):
                config_key = entry.action.get("key")
                if config_key is not None:
                    config_entries.setdefault(config_key, []).append(entry)

        result: dict[str, Any] = {}
        for config_key, entries in config_entries.items():
            closest = min(entries, key=lambda e: e.dag_distance)
            result[config_key] = closest.action.get("value")
        return result

    def add_rule(self, entry: RuleEntry) -> None:
        """Add or update a rule entry (used for incremental maintenance)."""
        key = (entry.trigger, entry.name)
        if key not in self._entries or entry.dag_distance <= self._entries[key].dag_distance:
            self._entries[key] = entry

    def invalidate(self) -> None:
        """Mark index as stale (requires rebuild on next access)."""
        self._stale = True

    @property
    def is_stale(self) -> bool:
        return self._stale

    def __len__(self) -> int:
        """Number of unique (trigger, name) entries."""
        return len(self._entries)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._entries
