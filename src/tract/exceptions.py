"""Trace exception hierarchy.

All Trace-specific exceptions inherit from TraceError.
"""


class TraceError(Exception):
    """Base exception for all Trace errors."""


class CommitNotFoundError(TraceError):
    """Raised when a commit hash lookup fails."""

    def __init__(self, commit_hash: str) -> None:
        self.commit_hash = commit_hash
        super().__init__(f"Commit not found: {commit_hash}")


class BlobNotFoundError(TraceError):
    """Raised when a blob hash lookup fails."""

    def __init__(self, content_hash: str) -> None:
        self.content_hash = content_hash
        super().__init__(f"Blob not found: {content_hash}")


class ContentValidationError(TraceError):
    """Raised when content validation fails.

    Named ContentValidationError (not ValidationError) to avoid
    collision with pydantic.ValidationError.
    """


class BudgetExceededError(TraceError):
    """Raised when token budget is exceeded."""

    def __init__(self, current_tokens: int, max_tokens: int) -> None:
        self.current_tokens = current_tokens
        self.max_tokens = max_tokens
        super().__init__(
            f"Token budget exceeded: {current_tokens} tokens "
            f"(max: {max_tokens})"
        )


class EditTargetError(TraceError):
    """Raised when an edit targets an invalid commit.

    An edit must target an existing, non-edit commit. Edits targeting
    other edits or nonexistent commits are invalid.
    """


class DuplicateRefError(TraceError):
    """Raised when a ref already exists."""

    def __init__(self, ref_name: str) -> None:
        self.ref_name = ref_name
        super().__init__(f"Ref already exists: {ref_name}")


class DetachedHeadError(TraceError):
    """Raised when attempting to commit in detached HEAD state."""

    def __init__(self) -> None:
        super().__init__(
            "Cannot commit in detached HEAD state. "
            "Use 'tract checkout main' to return to your branch."
        )


class AmbiguousPrefixError(TraceError):
    """Raised when a commit hash prefix matches multiple commits."""

    def __init__(self, prefix: str, candidates: list[str]) -> None:
        self.prefix = prefix
        self.candidates = candidates
        candidate_str = ", ".join(c[:12] + "..." for c in candidates[:5])
        super().__init__(
            f"Ambiguous prefix '{prefix}'. Matches: {candidate_str}"
        )


class BranchExistsError(TraceError):
    """Raised when trying to create a branch that already exists."""

    def __init__(self, branch_name: str) -> None:
        self.branch_name = branch_name
        super().__init__(f"Branch already exists: {branch_name}")


class BranchNotFoundError(TraceError):
    """Raised when a branch lookup fails."""

    def __init__(self, branch_name: str) -> None:
        self.branch_name = branch_name
        super().__init__(f"Branch not found: {branch_name}")


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
