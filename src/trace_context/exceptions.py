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
