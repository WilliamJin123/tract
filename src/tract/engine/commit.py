"""Commit engine for Trace.

Orchestrates commit creation with validation, parent chain management,
token budget enforcement, blob deduplication, and annotation auto-creation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import BaseModel

from tract.engine.hashing import commit_hash as compute_commit_hash
from tract.engine.hashing import content_hash as compute_content_hash
from tract.exceptions import (
    BudgetExceededError,
    CommitNotFoundError,
    EditTargetError,
)
from tract.models.annotations import DEFAULT_TYPE_PRIORITIES, Priority, PriorityAnnotation
from tract.models.commit import CommitInfo, CommitOperation
from tract.models.config import BudgetAction, TokenBudgetConfig
from tract.storage.schema import AnnotationRow, BlobRow, CommitRow

if TYPE_CHECKING:
    from tract.protocols import TokenCounter
    from tract.storage.repositories import (
        AnnotationRepository,
        BlobRepository,
        CommitParentRepository,
        CommitRepository,
        RefRepository,
    )

logger = logging.getLogger(__name__)


def extract_text_from_content(content: BaseModel) -> str:
    """Extract the primary text field from any content model.

    Handles the different field names used by different content types:
    - text: InstructionContent, DialogueContent, ReasoningContent, OutputContent
    - content: ArtifactContent
    - payload: ToolIOContent (dict -> JSON string), FreeformContent (dict -> JSON string)

    Args:
        content: A Pydantic content model instance.

    Returns:
        The text representation of the content.
    """
    if hasattr(content, "text"):
        return content.text  # type: ignore[return-value]
    if hasattr(content, "content") and isinstance(content.content, str):  # type: ignore[union-attr]
        return content.content  # type: ignore[return-value]
    if hasattr(content, "payload"):
        return json.dumps(content.payload, sort_keys=True)  # type: ignore[attr-defined]
    return ""


class CommitEngine:
    """Orchestrates commit creation with validation and storage.

    The commit engine is the primary write path for Trace. It enforces:
    - Content-addressable blob storage (dedup via hash)
    - Immutable commit chain (parent pointers)
    - Edit constraints (edits must target non-edit commits)
    - Token budget enforcement (warn, reject, callback)
    - Automatic priority annotations for certain content types
    """

    def __init__(
        self,
        commit_repo: CommitRepository,
        blob_repo: BlobRepository,
        ref_repo: RefRepository,
        annotation_repo: AnnotationRepository,
        token_counter: TokenCounter,
        tract_id: str,
        token_budget: TokenBudgetConfig | None = None,
        parent_repo: CommitParentRepository | None = None,
    ) -> None:
        self._commit_repo = commit_repo
        self._blob_repo = blob_repo
        self._ref_repo = ref_repo
        self._annotation_repo = annotation_repo
        self._token_counter = token_counter
        self._tract_id = tract_id
        self._token_budget = token_budget
        self._parent_repo = parent_repo

    def create_commit(
        self,
        content: BaseModel,
        operation: CommitOperation = CommitOperation.APPEND,
        message: str | None = None,
        response_to: str | None = None,
        metadata: dict | None = None,
        generation_config: dict | None = None,
    ) -> CommitInfo:
        """Create a new commit in the repository.

        Args:
            content: Pydantic content model instance (one of the 7 built-in types
                or a custom registered type).
            operation: APPEND (new content) or EDIT (replace existing).
            message: Optional human-readable commit message.
            response_to: For EDIT operations, the hash of the commit being edited.
            metadata: Optional arbitrary metadata dict.

        Returns:
            CommitInfo with the new commit's data.

        Raises:
            EditTargetError: If edit constraints are violated.
            BudgetExceededError: If token budget is exceeded in REJECT mode.
        """
        # 1. Serialize content to dict
        content_dict = content.model_dump(mode="json")
        content_type = content_dict.get("content_type", "unknown")

        # 2. Compute content hash
        c_hash = compute_content_hash(content_dict)

        # 3. Count tokens
        text = extract_text_from_content(content)
        token_count = self._token_counter.count_text(text)

        # 4. Store blob (content-addressable dedup)
        now = datetime.now(timezone.utc)
        blob = BlobRow(
            content_hash=c_hash,
            payload_json=json.dumps(content_dict, sort_keys=True, ensure_ascii=False),
            byte_size=len(json.dumps(content_dict).encode("utf-8")),
            token_count=token_count,
            created_at=now,
        )
        self._blob_repo.save_if_absent(blob)

        # 5. Get current HEAD
        parent_hash = self._ref_repo.get_head(self._tract_id)

        # 6. Check token budget
        if self._token_budget and self._token_budget.max_tokens is not None:
            total_tokens = token_count
            if parent_hash is not None:
                ancestors = self._commit_repo.get_ancestors(parent_hash)
                for ancestor in ancestors:
                    total_tokens += ancestor.token_count

            if total_tokens > self._token_budget.max_tokens:
                if self._token_budget.action == BudgetAction.REJECT:
                    raise BudgetExceededError(total_tokens, self._token_budget.max_tokens)
                elif self._token_budget.action == BudgetAction.WARN:
                    logger.warning(
                        "Token budget exceeded: %d tokens (max: %d)",
                        total_tokens,
                        self._token_budget.max_tokens,
                    )
                elif self._token_budget.action == BudgetAction.CALLBACK:
                    if self._token_budget.callback is not None:
                        self._token_budget.callback(total_tokens, self._token_budget.max_tokens)

        # 7. Generate timestamp
        timestamp = datetime.now(timezone.utc)
        timestamp_iso = timestamp.isoformat()

        # 8. Compute commit hash
        operation_value = operation.value if isinstance(operation, CommitOperation) else operation
        c_commit_hash = compute_commit_hash(
            content_hash=c_hash,
            parent_hash=parent_hash,
            content_type=content_type,
            operation=operation_value,
            timestamp_iso=timestamp_iso,
            response_to=response_to,
        )

        # 9. Validate edit constraints
        if operation == CommitOperation.EDIT:
            if response_to is None:
                raise EditTargetError("EDIT operation requires response_to to be set")
            target_commit = self._commit_repo.get(response_to)
            if target_commit is None:
                raise EditTargetError(
                    f"EDIT target commit not found: {response_to}"
                )
            if target_commit.operation == CommitOperation.EDIT:
                raise EditTargetError(
                    f"Cannot edit an EDIT commit: {response_to}"
                )

        # 10. Create CommitRow and save
        commit_row = CommitRow(
            commit_hash=c_commit_hash,
            tract_id=self._tract_id,
            parent_hash=parent_hash,
            content_hash=c_hash,
            content_type=content_type,
            operation=operation,
            response_to=response_to,
            message=message,
            token_count=token_count,
            metadata_json=metadata,
            generation_config_json=generation_config,
            created_at=timestamp,
        )
        self._commit_repo.save(commit_row)

        # 11. Update HEAD
        self._ref_repo.update_head(self._tract_id, c_commit_hash)

        # 12. Auto-create priority annotation if content type has non-NORMAL default
        default_priority = DEFAULT_TYPE_PRIORITIES.get(content_type, Priority.NORMAL)
        if default_priority != Priority.NORMAL:
            annotation = AnnotationRow(
                tract_id=self._tract_id,
                target_hash=c_commit_hash,
                priority=default_priority,
                reason=f"Default priority for {content_type}",
                created_at=timestamp,
            )
            self._annotation_repo.save(annotation)

        # 13. Return CommitInfo
        return CommitInfo(
            commit_hash=c_commit_hash,
            tract_id=self._tract_id,
            parent_hash=parent_hash,
            content_hash=c_hash,
            content_type=content_type,
            operation=operation,
            response_to=response_to,
            message=message,
            token_count=token_count,
            metadata=metadata,
            generation_config=generation_config,
            created_at=timestamp,
        )

    def create_merge_commit(
        self,
        content: BaseModel,
        parent_hashes: list[str],
        *,
        message: str | None = None,
        metadata: dict | None = None,
        generation_config: dict | None = None,
    ) -> CommitInfo:
        """Create a merge commit with multiple parents.

        Similar to create_commit() but:
        - Sets parent_hash to parent_hashes[0] (first parent = current branch).
        - Includes extra_parents in the commit hash computation.
        - Records all parents in the commit_parents table via parent_repo.
        - Operation is always APPEND.
        - No edit constraint checks.

        Args:
            content: Pydantic content model instance.
            parent_hashes: List of parent hashes. [0] = current branch tip,
                [1] = source branch tip.
            message: Optional commit message.
            metadata: Optional metadata dict.
            generation_config: Optional generation config dict.

        Returns:
            CommitInfo for the new merge commit.

        Raises:
            MergeError: If parent_repo is not configured.
        """
        if self._parent_repo is None:
            from tract.exceptions import MergeError

            raise MergeError("CommitEngine has no parent_repo -- cannot create merge commit")

        # 1. Serialize content
        content_dict = content.model_dump(mode="json")
        content_type = content_dict.get("content_type", "unknown")

        # 2. Compute content hash
        c_hash = compute_content_hash(content_dict)

        # 3. Count tokens
        text = extract_text_from_content(content)
        token_count = self._token_counter.count_text(text)

        # 4. Store blob
        now = datetime.now(timezone.utc)
        blob = BlobRow(
            content_hash=c_hash,
            payload_json=json.dumps(content_dict, sort_keys=True, ensure_ascii=False),
            byte_size=len(json.dumps(content_dict).encode("utf-8")),
            token_count=token_count,
            created_at=now,
        )
        self._blob_repo.save_if_absent(blob)

        # 5. Parent hashes
        first_parent = parent_hashes[0] if parent_hashes else None
        extra_parents = parent_hashes[1:] if len(parent_hashes) > 1 else None

        # 6. Generate timestamp and commit hash
        timestamp = datetime.now(timezone.utc)
        timestamp_iso = timestamp.isoformat()

        c_commit_hash = compute_commit_hash(
            content_hash=c_hash,
            parent_hash=first_parent,
            content_type=content_type,
            operation=CommitOperation.APPEND.value,
            timestamp_iso=timestamp_iso,
            extra_parents=extra_parents,
        )

        # 7. Create and save CommitRow
        commit_row = CommitRow(
            commit_hash=c_commit_hash,
            tract_id=self._tract_id,
            parent_hash=first_parent,
            content_hash=c_hash,
            content_type=content_type,
            operation=CommitOperation.APPEND,
            response_to=None,
            message=message,
            token_count=token_count,
            metadata_json=metadata,
            generation_config_json=generation_config,
            created_at=timestamp,
        )
        self._commit_repo.save(commit_row)

        # 8. Record all parents in commit_parents table
        self._parent_repo.add_parents(c_commit_hash, parent_hashes)

        # 9. Update HEAD
        self._ref_repo.update_head(self._tract_id, c_commit_hash)

        # 10. Return CommitInfo
        return CommitInfo(
            commit_hash=c_commit_hash,
            tract_id=self._tract_id,
            parent_hash=first_parent,
            content_hash=c_hash,
            content_type=content_type,
            operation=CommitOperation.APPEND,
            response_to=None,
            message=message,
            token_count=token_count,
            metadata=metadata,
            generation_config=generation_config,
            created_at=timestamp,
        )

    def get_commit(self, commit_hash: str) -> CommitInfo | None:
        """Fetch a commit by its hash.

        Args:
            commit_hash: The SHA-256 hash of the commit.

        Returns:
            CommitInfo if found, None otherwise.
        """
        row = self._commit_repo.get(commit_hash)
        if row is None:
            return None
        return self._row_to_info(row)

    def annotate(
        self,
        target_hash: str,
        priority: Priority,
        reason: str | None = None,
    ) -> PriorityAnnotation:
        """Create a priority annotation on a commit.

        Args:
            target_hash: Hash of the commit to annotate.
            priority: Priority level to set.
            reason: Optional reason for the annotation.

        Returns:
            PriorityAnnotation model.

        Raises:
            CommitNotFoundError: If the target commit doesn't exist.
        """
        target = self._commit_repo.get(target_hash)
        if target is None:
            raise CommitNotFoundError(target_hash)

        now = datetime.now(timezone.utc)
        annotation = AnnotationRow(
            tract_id=self._tract_id,
            target_hash=target_hash,
            priority=priority,
            reason=reason,
            created_at=now,
        )
        self._annotation_repo.save(annotation)

        return PriorityAnnotation(
            id=annotation.id,
            tract_id=self._tract_id,
            target_hash=target_hash,
            priority=priority,
            reason=reason,
            created_at=now,
        )

    def _row_to_info(self, row: CommitRow) -> CommitInfo:
        """Convert a CommitRow ORM object to a CommitInfo Pydantic model."""
        return CommitInfo(
            commit_hash=row.commit_hash,
            tract_id=row.tract_id,
            parent_hash=row.parent_hash,
            content_hash=row.content_hash,
            content_type=row.content_type,
            operation=row.operation,
            response_to=row.response_to,
            message=row.message,
            token_count=row.token_count,
            metadata=row.metadata_json,
            generation_config=row.generation_config_json,
            created_at=row.created_at,
        )
