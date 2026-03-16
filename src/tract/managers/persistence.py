"""Persistence manager for Tract.

Extracted from Tract (tract.py) into a standalone class with explicit
constructor dependencies.  Handles snapshots, export/import, compile records,
token checkpoints, behavioral specs, workflow files, and .tract/ directory.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from tract.exceptions import TraceError
from tract.models.annotations import Priority
from tract.models.content import validate_content

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import Any

    from tract.models.commit import CommitInfo
    from tract.models.config import TractConfig
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitRepository,
        CompileRecordRepository,
        OperationEventRepository,
        ParentRepository,
        PersistenceRepository,
        RefRepository,
    )

logger = logging.getLogger(__name__)


class PersistenceManager:
    """Snapshots, export/import, compile records, behavioral specs, .tract/ directory."""

    def __init__(
        self,
        tract_id: str,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        ref_repo: RefRepository,
        annotation_repo: AnnotationRepository,
        parent_repo: ParentRepository,
        event_repo: OperationEventRepository,
        compile_record_repo: CompileRecordRepository | None,
        persistence_repo: PersistenceRepository | None,
        behavioral_spec_repo: Any | None,
        config: TractConfig,
        custom_type_registry: dict,
        db_path: str,
        check_open: Callable[[], None],
        commit_fn: Callable,  # Tract.commit
        log_fn: Callable,  # SearchManager.log
        branch_fn: Callable,  # BranchManager.create
        switch_fn: Callable,  # BranchManager.switch
        reset_fn: Callable,  # BranchManager.reset
        register_tag_fn: Callable,  # TagManager.register
        annotate_fn: Callable,  # AnnotationManager.set
        list_branches_fn: Callable,  # BranchManager.list
        commit_session: Callable[[], None],
        get_head: Callable[[], str | None],
        row_to_info: Callable,
    ) -> None:
        self._tract_id = tract_id
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._ref_repo = ref_repo
        self._annotation_repo = annotation_repo
        self._parent_repo = parent_repo
        self._event_repo = event_repo
        self._compile_record_repo = compile_record_repo
        self._persistence_repo = persistence_repo
        self._behavioral_spec_repo = behavioral_spec_repo
        self._config = config
        self._custom_type_registry = custom_type_registry
        self._db_path = db_path
        self._check_open = check_open
        self._commit_fn = commit_fn
        self._log_fn = log_fn
        self._branch_fn = branch_fn
        self._switch_fn = switch_fn
        self._reset_fn = reset_fn
        self._register_tag_fn = register_tag_fn
        self._annotate_fn = annotate_fn
        self._list_branches_fn = list_branches_fn
        self._commit_session = commit_session
        self._get_head = get_head
        self._row_to_info = row_to_info

        # Owned state
        self._quarantined: list[str] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def tract_dir(self) -> Path | None:
        """Path to .tract/ directory, or None for in-memory databases."""
        from pathlib import Path

        if self._db_path == ":memory:":
            return None
        db = Path(self._db_path)
        return db.parent / ".tract"

    @property
    def quarantined(self) -> list[str]:
        """List of modules that failed to load on startup."""
        return list(self._quarantined)

    # ------------------------------------------------------------------
    # .tract/ directory helpers
    # ------------------------------------------------------------------

    def _ensure_tract_dir(self, subdir: str | None = None) -> Path:
        """Create .tract/ (and optional subdir) lazily. Returns the dir path.

        Raises RuntimeError for in-memory databases.
        """
        td = self.tract_dir
        if td is None:
            raise RuntimeError(
                "File-based persistence is not available for in-memory databases."
            )
        if subdir:
            target = td / subdir
        else:
            target = td
        target.mkdir(parents=True, exist_ok=True)
        return target

    # ------------------------------------------------------------------
    # Persisted state loading
    # ------------------------------------------------------------------

    def _load_persisted_state(
        self,
        profile_registry: dict,
        template_registry: dict,
    ) -> None:
        """Load persisted configs and behavioral specs from DB.

        Called during Tract.open() after all repos are initialized.

        Args:
            profile_registry: The Tract's profile registry dict (mutated in-place).
            template_registry: The Tract's template registry dict (mutated in-place).
        """
        repo = self._persistence_repo
        if repo is None:
            return

        # Load operation configs from DB
        for config_row in repo.get_operation_configs(self._tract_id):
            try:
                json.loads(config_row.config_json)  # validate JSON
            except Exception:
                logger.warning(
                    "Failed to load config '%s': skipping.",
                    config_row.config_key,
                    exc_info=True,
                )
                self._quarantined.append(f"config:{config_row.config_key}")

        # Load behavioral specs from DB (profiles and templates)
        spec_repo = self._behavioral_spec_repo
        if spec_repo is None:
            return

        for row in spec_repo.list_specs(self._tract_id):
            try:
                data = json.loads(row.spec_json)
            except Exception:
                logger.warning(
                    "Failed to parse behavioral spec '%s/%s': skipping.",
                    row.spec_type,
                    row.spec_name,
                    exc_info=True,
                )
                self._quarantined.append(f"spec:{row.spec_type}:{row.spec_name}")
                continue

            try:
                if row.spec_type == "profile":
                    from tract.profiles import WorkflowProfile
                    profile = WorkflowProfile.from_spec(data)
                    profile_registry[profile.name] = profile
                elif row.spec_type == "template":
                    from tract.templates import DirectiveTemplate
                    template = DirectiveTemplate.from_spec(data)
                    template_registry[template.name] = template
                # gate and maintainer specs are loaded but NOT auto-wired
                # (callables cannot be restored; users must re-register)
            except Exception:
                logger.warning(
                    "Failed to restore behavioral spec '%s/%s': skipping.",
                    row.spec_type,
                    row.spec_name,
                    exc_info=True,
                )
                self._quarantined.append(f"spec:{row.spec_type}:{row.spec_name}")

    # ------------------------------------------------------------------
    # Compile records & token checkpoints
    # ------------------------------------------------------------------

    def compile_records(self, limit: int = 100) -> list:
        """Get compile records for this tract, newest first.

        Returns list of CompileRecordRow objects, or empty list if
        compile record repository is not available.
        """
        if self._compile_record_repo is None:
            return []
        records = self._compile_record_repo.get_all(self._tract_id)
        return list(reversed(records))[:limit]  # newest first

    def compile_record_commits(self, record_id: str) -> list[str]:
        """Get the ordered commit hashes for a compile record.

        Returns list of commit hashes in compilation order, or empty list
        if record not found or compile record repository not available.
        """
        if self._compile_record_repo is None:
            return []
        effectives = self._compile_record_repo.get_effectives(record_id)
        return [e.commit_hash for e in effectives]

    def token_checkpoints(self, limit: int = 100) -> list:
        """API-calibrated token checkpoints, newest first.

        Returns compile records where ``token_source`` starts with ``"api:"``,
        i.e. records created by :meth:`record_usage`. Use ``limit=0`` to
        return all matching records.
        """
        if self._compile_record_repo is None:
            return []
        all_records = list(reversed(self._compile_record_repo.get_all(self._tract_id)))
        api_records = [r for r in all_records if r.token_source.startswith("api:")]
        if limit == 0:
            return api_records
        return api_records[:limit]

    # ------------------------------------------------------------------
    # Snapshots
    # ------------------------------------------------------------------

    def snapshot(self, label: str = "", *, metadata: dict | None = None) -> str:
        """Create a named snapshot (restore point) at the current HEAD.

        Snapshots are implemented as specially-tagged commits with metadata.
        They record the current state for later restoration.

        Args:
            label: Human-readable snapshot label (e.g., "before-merge",
                "pre-compress").
            metadata: Optional extra metadata to store with the snapshot.

        Returns:
            The snapshot tag name (e.g., ``"snapshot:before-merge:abc123"``).

        Raises:
            TraceError: If there is no HEAD (empty tract).
        """
        self._check_open()
        import time

        current_head = self._get_head()
        if current_head is None:
            raise TraceError("Cannot create snapshot: no commits yet")

        head_short = current_head[:7]
        timestamp = int(time.time())
        tag_name = (
            f"snapshot:{label}:{head_short}"
            if label
            else f"snapshot:{timestamp}:{head_short}"
        )

        # Gather lightweight state (avoid expensive compile via status())
        branch_name = self._ref_repo.get_current_branch(self._tract_id)

        # Build snapshot metadata
        snap_meta: dict = {
            "snapshot": True,
            "label": label,
            "head": current_head,
            "branch": branch_name,
            "timestamp": timestamp,
            **(metadata or {}),
        }

        # Register the tag so strict-mode allows it
        self._register_tag_fn(tag_name, description=f"Snapshot: {label or 'unnamed'}")

        # Store as a metadata commit with the snapshot tag
        from tract.models.content import MetadataContent

        self._commit_fn(
            MetadataContent(kind="snapshot", data=snap_meta),
            message=f"Snapshot: {label or 'unnamed'}",
            metadata=snap_meta,
            tags=[tag_name],
        )

        return tag_name

    def list_snapshots(self) -> list[dict]:
        """List all snapshots for this tract.

        Returns:
            List of snapshot metadata dicts, newest first.  Each dict has
            keys: ``tag``, ``label``, ``head``, ``branch``, ``timestamp``,
            ``hash``.
        """
        snapshots: list[dict] = []
        for entry in self._log_fn(limit=500):
            meta = entry.metadata or {}
            if meta.get("snapshot"):
                snapshots.append({
                    "tag": next(
                        (t for t in entry.tags if t.startswith("snapshot:")), ""
                    ),
                    "label": meta.get("label", ""),
                    "head": meta.get("head", ""),
                    "branch": meta.get("branch", ""),
                    "timestamp": meta.get("timestamp", 0),
                    "hash": entry.commit_hash,
                })
        return snapshots

    def restore_snapshot(
        self,
        tag_or_label: str,
        *,
        create_branch: bool = True,
    ) -> str:
        """Restore to a previously created snapshot point.

        Args:
            tag_or_label: Snapshot tag name or label substring to match.
            create_branch: If ``True`` (default), create a recovery branch
                at the snapshot point (safe -- no history loss).  If
                ``False``, reset HEAD directly.

        Returns:
            The commit hash restored to.

        Raises:
            ValueError: If no matching snapshot is found.
        """
        self._check_open()
        snapshots = self.list_snapshots()
        match: dict | None = None
        for snap in snapshots:
            if snap["tag"] == tag_or_label or tag_or_label in snap.get("label", ""):
                match = snap
                break

        if match is None:
            raise ValueError(f"Snapshot not found: {tag_or_label}")

        target_head: str = match["head"]

        if create_branch:
            branch_name = f"restore/{match['label'] or 'snapshot'}"
            self._branch_fn(branch_name, source=target_head)
            self._switch_fn(branch_name)
        else:
            self._reset_fn(target_head)

        return target_head

    # ------------------------------------------------------------------
    # Export / Import
    # ------------------------------------------------------------------

    def export_state(self, *, include_blobs: bool = True) -> dict:
        """Export the current branch's DAG as a portable JSON-serializable dict.

        Creates a snapshot of all commits reachable from HEAD with their
        content, metadata, annotations, and branch info. The result can be
        saved to a file and loaded into a different tract via
        :meth:`load_state`.

        Note:
            The exported dict contains full commit details, but
            :meth:`load_state` only replays content payloads -- it does not
            reconstruct the original DAG. See :meth:`load_state` for the
            list of what is and is not preserved on import.

        Args:
            include_blobs: If True (default), include full content payloads.
                If False, include only commit metadata (smaller but not
                restorable).

        Returns:
            A dict with keys: version, tract_id, branch, head, commits,
            branches, exported_at.
        """
        self._check_open()
        from tract.operations.ancestry import walk_ancestry

        current_branch = self._ref_repo.get_current_branch(self._tract_id)
        head = self._get_head()
        if head is None:
            return {
                "version": 1,
                "tract_id": self._tract_id,
                "branch": current_branch,
                "head": None,
                "commits": [],
                "branches": {},
                "exported_at": datetime.now(timezone.utc).isoformat(),
            }

        # Walk full ancestry from HEAD
        commits_data = []
        ancestry = walk_ancestry(
            self._commit_repo, self._blob_repo, head,
            parent_repo=self._parent_repo,
        )

        for commit_row in ancestry:
            entry: dict = {
                "hash": commit_row.commit_hash,
                "content_type": commit_row.content_type,
                "operation": commit_row.operation.value if hasattr(commit_row.operation, "value") else str(commit_row.operation),
                "message": commit_row.message,
                "metadata": commit_row.metadata_json,
                "created_at": commit_row.created_at.isoformat() if commit_row.created_at else None,
            }

            # Parent hashes -- get_parents returns list[str]
            if self._parent_repo:
                parents = self._parent_repo.get_parents(commit_row.commit_hash)
                entry["parents"] = parents
            else:
                entry["parents"] = [commit_row.parent_hash] if commit_row.parent_hash else []

            # Blob content
            if include_blobs:
                blob = self._blob_repo.get(commit_row.content_hash)
                if blob:
                    entry["content_hash"] = commit_row.content_hash
                    entry["payload"] = blob.payload_json
                else:
                    entry["content_hash"] = commit_row.content_hash
                    entry["payload"] = None
            else:
                entry["content_hash"] = commit_row.content_hash

            # Annotations
            ann = self._annotation_repo.get_latest(commit_row.commit_hash)
            if ann:
                entry["priority"] = ann.priority.value if hasattr(ann.priority, "value") else str(ann.priority)

            commits_data.append(entry)

        # Branch info
        branches = {}
        for branch_info in self._list_branches_fn():
            branches[branch_info.name] = branch_info.commit_hash

        return {
            "version": 1,
            "tract_id": self._tract_id,
            "branch": current_branch,
            "head": head,
            "commits": commits_data,
            "branches": branches,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }

    def load_state(self, state: dict) -> int:
        """Load commits from an exported state dict into this tract.

        Replays the exported commits as new APPEND commits on the current
        branch. Does NOT overwrite existing commits.

        This is a **content replay** tool, not a structural backup/restore.

        Preserved on import:
            - Content payloads (the actual data in each commit)
            - ``content_type``
            - ``metadata``
            - ``message``
            - ``priority`` annotations (non-normal values)

        Not preserved on import:
            - DAG structure and parent links (commits are re-appended linearly)
            - Branches (all commits land on the current branch)
            - Operation types (EDIT operations become APPENDs)
            - Original timestamps (commits get new ``created_at`` values)
            - Tags
            - ``edit_target`` relationships
            - Original commit hashes

        Args:
            state: A dict previously returned by :meth:`export_state`.

        Returns:
            Number of commits loaded.

        Raises:
            ValueError: If the state dict is invalid or version unsupported.
        """
        self._check_open()
        if not isinstance(state, dict) or state.get("version") != 1:
            raise ValueError("Invalid or unsupported export state (expected version=1)")

        commits = state.get("commits", [])
        if not commits:
            return 0

        loaded = 0
        for entry in commits:
            payload = entry.get("payload")
            if payload is None:
                continue  # skip commits without content (include_blobs=False)

            content_type = entry.get("content_type", "")
            message = entry.get("message")
            metadata_json = entry.get("metadata")

            # Parse metadata
            metadata = None
            if metadata_json:
                try:
                    metadata = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
                except (ValueError, TypeError):
                    metadata = None

            # Reconstruct content from payload
            try:
                payload_dict = json.loads(payload) if isinstance(payload, str) else payload
            except (ValueError, TypeError):
                continue

            # Ensure content_type is present in payload dict
            if "content_type" not in payload_dict:
                payload_dict["content_type"] = content_type

            # Use validate_content to reconstruct the proper content model
            try:
                content = validate_content(
                    payload_dict,
                    custom_registry=self._custom_type_registry,
                )
            except Exception:
                # Content validation failed (unrecognized type, schema mismatch,
                # or corrupt payload); skip this entry rather than aborting import.
                logger.warning(
                    "load_state: skipping entry with content_type=%r: validation failed",
                    content_type, exc_info=True,
                )
                continue

            info = self._commit_fn(
                content,
                message=message or f"imported: {content_type}",
                metadata=metadata,
            )

            # Restore priority annotation if present
            priority_val = entry.get("priority")
            if priority_val and priority_val != "normal":
                try:
                    priority = Priority(priority_val)
                    self._annotate_fn(info.commit_hash, priority, reason="imported")
                except (ValueError, KeyError):
                    pass

            loaded += 1

        return loaded

    # ------------------------------------------------------------------
    # Behavioral spec persistence
    # ------------------------------------------------------------------

    def persist_behavioral_spec(
        self,
        spec_type: str,
        spec_name: str,
        spec_data: dict,
    ) -> None:
        """Save a behavioral spec to the database.

        Args:
            spec_type: One of ``"gate"``, ``"maintainer"``, ``"middleware"``,
                ``"profile"``, ``"template"``.
            spec_name: Unique name within the spec type.
            spec_data: JSON-serializable dict with the spec configuration.

        Raises:
            ValueError: If spec_type is invalid.
            RuntimeError: If no behavioral spec repository is available.
        """
        valid_types = {"gate", "maintainer", "middleware", "profile", "template"}
        if spec_type not in valid_types:
            raise ValueError(
                f"Invalid spec_type '{spec_type}'. "
                f"Valid types: {sorted(valid_types)}"
            )
        repo = self._behavioral_spec_repo
        if repo is None:
            return  # in-memory or no persistence

        from tract.storage.schema import BehavioralSpecRow

        now = datetime.now(timezone.utc)
        row = BehavioralSpecRow(
            tract_id=self._tract_id,
            spec_type=spec_type,
            spec_name=spec_name,
            spec_json=json.dumps(spec_data, default=str),
            created_at=now,
            updated_at=now,
        )
        repo.save(row)
        self._commit_session()

    def load_behavioral_specs(
        self,
        *,
        spec_type: str | None = None,
    ) -> list[dict]:
        """Load persisted behavioral specs from the database.

        Args:
            spec_type: Optional filter by type (``"gate"``, ``"maintainer"``,
                ``"profile"``, ``"template"``).

        Returns:
            List of dicts with keys: spec_type, spec_name, spec_data, created_at, updated_at.
        """
        repo = self._behavioral_spec_repo
        if repo is None:
            return []

        rows = repo.list_specs(self._tract_id, spec_type=spec_type)
        result = []
        for row in rows:
            try:
                data = json.loads(row.spec_json)
            except Exception:
                data = {}
            result.append({
                "spec_type": row.spec_type,
                "spec_name": row.spec_name,
                "spec_data": data,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            })
        return result

    def list_behavioral_specs(
        self,
        *,
        spec_type: str | None = None,
    ) -> list[dict]:
        """List persisted behavioral specs (summary view).

        Args:
            spec_type: Optional filter by type.

        Returns:
            List of dicts with keys: spec_type, spec_name, created_at, updated_at.
        """
        repo = self._behavioral_spec_repo
        if repo is None:
            return []

        rows = repo.list_specs(self._tract_id, spec_type=spec_type)
        return [
            {
                "spec_type": row.spec_type,
                "spec_name": row.spec_name,
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }
            for row in rows
        ]

    def remove_behavioral_spec(self, spec_type: str, spec_name: str) -> bool:
        """Remove a persisted behavioral spec.

        Args:
            spec_type: Spec type (``"gate"``, ``"maintainer"``, etc.).
            spec_name: Spec name.

        Returns:
            True if deleted, False if not found.
        """
        repo = self._behavioral_spec_repo
        if repo is None:
            return False
        deleted = repo.delete(self._tract_id, spec_type, spec_name)
        if deleted:
            self._commit_session()
        return deleted

    # ------------------------------------------------------------------
    # Workflow file persistence
    # ------------------------------------------------------------------

    def save_workflow(
        self,
        name: str,
        code: str,
        *,
        description: str = "",
    ) -> Path:
        """Write a workflow to .tract/workflows/{name}.py.

        Args:
            name: Workflow name (used as filename).
            code: Python source code.
            description: Human-readable description.

        Returns:
            Path to the written file.

        Raises:
            SyntaxError: If code has syntax errors (validated before writing).
            RuntimeError: If database is in-memory.
        """
        # Security: reject path separators and traversal in workflow name
        if "/" in name or "\\" in name or ".." in name:
            raise ValueError(
                f"Invalid workflow name {name!r}: must not contain "
                "'/', '\\', or '..'."
            )

        # Validate syntax
        compile(code, f"{name}.py", "exec")

        # Write file
        workflows_dir = self._ensure_tract_dir("workflows")
        file_path = workflows_dir / f"{name}.py"
        file_path.write_text(code, encoding="utf-8")

        return file_path
