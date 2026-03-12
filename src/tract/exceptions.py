"""Trace exception hierarchy.

All Trace-specific exceptions inherit from TraceError.
Each exception carries a ``hint`` attribute with an actionable suggestion
for LLM agents encountering the error.
"""

__all__: list[str] = [
    "TraceError",
    "CommitNotFoundError",
    "BlobNotFoundError",
    "ContentValidationError",
    "BudgetExceededError",
    "EditTargetError",
    "DetachedHeadError",
    "AmbiguousPrefixError",
    "BranchExistsError",
    "BranchNotFoundError",
    "InvalidBranchNameError",
    "UnmergedBranchError",
    "MergeError",
    "MergeConflictError",
    "NothingToMergeError",
    "RebaseError",
    "ImportCommitError",
    "SemanticSafetyError",
    "CompressionError",
    "GCError",
    "SpawnError",
    "SessionError",
    "RetryExhaustedError",
    "TagNotRegisteredError",
    "CurationError",
    "BlockedError",
]


class TraceError(Exception):
    """Base exception for all Trace errors."""

    hint: str = ""


class CommitNotFoundError(TraceError):
    """Raised when a commit hash lookup fails."""

    def __init__(self, commit_hash: str) -> None:
        self.commit_hash = commit_hash
        super().__init__(f"Commit not found: {commit_hash}")
        self.hint = "Use t.log() to see recent commits, or t.find() to search by content/tags."


class BlobNotFoundError(TraceError):
    """Raised when a blob hash lookup fails."""

    def __init__(self, content_hash: str) -> None:
        self.content_hash = content_hash
        super().__init__(f"Blob not found: {content_hash}")
        self.hint = "The blob may have been garbage collected. Use t.log() to find valid commits."


class ContentValidationError(TraceError):
    """Raised when content validation fails.

    Named ContentValidationError (not ValidationError) to avoid
    collision with pydantic.ValidationError.
    """

    hint: str = "Ensure content dict has a 'content_type' field matching a registered type. Use t.status() to check."


class BudgetExceededError(TraceError):
    """Raised when token budget is exceeded."""

    def __init__(self, current_tokens: int, max_tokens: int) -> None:
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Token budget exceeded: {current_tokens} tokens "
            f"(max: {max_tokens})"
        )
        self.hint = "Use t.compress() to reduce token usage, or t.configure(max_tokens=N) to raise the budget."


class EditTargetError(TraceError):
    """Raised when an edit targets an invalid commit.

    An edit must target an existing, non-edit commit. Edits targeting
    other edits or nonexistent commits are invalid.
    """

    hint: str = "Edit must target a non-edit commit. Use t.log() to find the original commit hash."


class DetachedHeadError(TraceError):
    """Raised when attempting to commit in detached HEAD state."""

    def __init__(self) -> None:
        super().__init__(
            "Cannot commit in detached HEAD state. "
            "Use 'tract checkout main' to return to your branch."
        )
        self.hint = "Run t.checkout('main') or t.switch('branch_name') to reattach."


class AmbiguousPrefixError(TraceError):
    """Raised when a commit hash prefix matches multiple commits."""

    def __init__(self, prefix: str, candidates: list[str]) -> None:
        self.prefix = prefix
        self.candidates = candidates
        candidate_str = ", ".join(c[:12] + "..." for c in candidates[:5])
        super().__init__(
            f"Ambiguous prefix '{prefix}'. Matches: {candidate_str}"
        )
        self.hint = "Provide more hash characters to disambiguate, or use t.log() to see full hashes."


class BranchExistsError(TraceError):
    """Raised when trying to create a branch that already exists."""

    def __init__(self, branch_name: str) -> None:
        self.branch_name = branch_name
        super().__init__(f"Branch already exists: {branch_name}")
        self.hint = "Use t.switch('name') to switch to the existing branch, or pick a different name."


class BranchNotFoundError(TraceError):
    """Raised when a branch lookup fails."""

    def __init__(self, branch_name: str) -> None:
        self.branch_name = branch_name
        super().__init__(f"Branch not found: {branch_name}")
        self.hint = "Use t.list_branches() to see available branches."


class InvalidBranchNameError(TraceError):
    """Raised when a branch name violates naming rules."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self.reason = reason
        super().__init__(f"Invalid branch name '{name}': {reason}")


class UnmergedBranchError(TraceError):
    """Raised when trying to delete a branch with unmerged commits."""

    def __init__(self, branch_name: str) -> None:
        self.branch_name = branch_name
        super().__init__(
            f"Branch '{branch_name}' has unmerged commits. "
            f"Use force=True to delete anyway."
        )
        self.hint = "Merge first with t.merge(), or use t.delete_branch(name, force=True)."


class MergeError(TraceError):
    """Base exception for all merge errors."""


class MergeConflictError(MergeError):
    """Raised when merge conflicts are detected and no resolver is available."""

    def __init__(self, conflict_count: int, details: str = "") -> None:
        self.conflict_count = conflict_count
        msg = f"Merge has {conflict_count} conflict(s) requiring resolution"
        if details:
            msg += f": {details}"
        super().__init__(msg)
        self.hint = "Use t.merge(source, strategy='theirs') or strategy='ours' to auto-resolve conflicts."


class NothingToMergeError(MergeError):
    """Raised when the source branch is already merged (up-to-date)."""

    def __init__(self, source_branch: str) -> None:
        self.source_branch = source_branch
        super().__init__(f"Branch '{source_branch}' is already up-to-date")
        self.hint = "Branch is already up-to-date. No action needed."


class RebaseError(TraceError):
    """Base exception for rebase errors."""

    hint: str = "Check that the target branch exists and has diverged. Use t.list_branches() to verify."


class ImportCommitError(TraceError):
    """Base exception for import-commit errors."""

    hint: str = "Verify the commit hash and source repository. Use t.log() on the source."


class SemanticSafetyError(TraceError):
    """Raised when a semantic safety check blocks and no resolver is available."""

    hint: str = "Review the content for safety issues, or configure a resolver to handle them."


class CompressionError(TraceError):
    """Raised when compression fails."""

    hint: str = "Ensure an LLM client is configured via t.configure_llm(). Check that context is non-empty."


class GCError(TraceError):
    """Raised when garbage collection fails."""

    hint: str = "GC may fail if there are active references. Try again after completing pending operations."


class SpawnError(TraceError):
    """Raised when spawn or collapse operations fail."""

    hint: str = "Check parent/child relationship. Use t.children() or t.parent() to inspect."


class SessionError(TraceError):
    """Raised when session operations fail."""

    hint: str = "Verify session state with t.status(). Sessions require proper initialization."


class RetryExhaustedError(TraceError):
    """All retry attempts failed."""

    def __init__(
        self, attempts: int, last_diagnosis: str, last_result: object = None
    ) -> None:
        self.attempts = attempts
        self.last_diagnosis = last_diagnosis
        self.last_result = last_result
        super().__init__(
            f"All {attempts} retry attempts failed. Last diagnosis: {last_diagnosis}"
        )
        self.hint = "Check LLM client configuration and API availability."


class TagNotRegisteredError(TraceError):
    """Raised when an unregistered tag is used in strict mode."""

    def __init__(self, tag_name: str | list[str]) -> None:
        if isinstance(tag_name, list):
            self.tag_name = tag_name[0] if len(tag_name) == 1 else tag_name[0]
            self.tag_names = tag_name
            quoted = ", ".join(f"'{t}'" for t in tag_name)
            super().__init__(
                f"Tags not registered: {quoted}. "
                f"Register them with t.register_tag(name, description) first, "
                f"or use t.list_tags() to see available tags."
            )
        else:
            self.tag_name = tag_name
            self.tag_names = [tag_name]
            super().__init__(
                f"Tag '{tag_name}' is not registered. "
                f"Use t.register_tag('{tag_name}', description) first, "
                f"or use t.list_tags() to see available tags."
            )


class CurationError(TraceError):
    """Raised when a curation operation fails during deploy()."""

    hint: str = "Review curation rules. Use t.log() to inspect commits targeted for curation."


class BlockedError(TraceError):
    """An operation was blocked by config enforcement or middleware.

    Attributes:
        event: The event that was blocked (e.g. "pre_commit", "pre_compile").
        reasons: List of human-readable block reasons.
    """

    def __init__(self, event: str, reasons: list[str] | str) -> None:
        self.event = event
        self.reasons = reasons if isinstance(reasons, list) else [reasons]
        reason_str = "; ".join(self.reasons) if self.reasons else "Blocked"
        super().__init__(f"{event} blocked: {reason_str}")
        self.hint = "Check middleware configuration. Use t.list_middleware() or review t.status() for active blocks."
