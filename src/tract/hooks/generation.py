"""PendingGeneration -- hook object for LLM generation operations.

Intercepts generate()/chat() responses before committing. Supports
validation, retry with steering (failed attempts committed with SKIP),
and retry metadata on the final commit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from tract.hooks.pending import Pending, PendingStatus
from tract.hooks.validation import ValidationResult

if TYPE_CHECKING:
    from tract.protocols import ChatResponse
    from tract.tract import Tract


@dataclass(repr=False)
class PendingGeneration(Pending):
    """An LLM generation that has been produced but not yet committed.

    The response text is available for inspection. Handlers can validate,
    retry (which commits the failed attempt with SKIP and re-generates),
    or approve (which commits the final response).

    Public fields:
        response_text: The generated response text
        validator: Optional validator callable (str) -> (bool, str|None)
        retry_prompt: Custom steering prompt template
        hide_retries: Whether to SKIP-annotate failed attempts (default True)
        retry_count: Number of retries performed so far
        retry_history: List of diagnosis strings from failed attempts

    Internal fields:
        _generate_fn: Closure that calls the LLM and returns (text, ChatResponse-building-info)
        _commit_fn: Closure that commits assistant response and returns CommitInfo
        _steer_fn: Closure that commits a steering user message (with SKIP annotation)
        _annotate_skip_fn: Closure that annotates a commit with SKIP priority
        _llm_kwargs: LLM parameters for re-generation
    """

    # Public fields
    response_text: str = ""
    validator: Callable[[str], tuple[bool, str | None]] | None = field(default=None, repr=False)
    retry_prompt: str | None = None
    hide_retries: bool = True
    retry_count: int = 0
    retry_history: list[str] = field(default_factory=list)

    # Internal: the ChatResponse data from the most recent generation
    _chat_response: Any = field(default=None, repr=False)

    # Internal: closures set by Tract.generate() for re-generation
    _generate_fn: Callable[[], Any] | None = field(default=None, repr=False)
    _steer_fn: Callable[[str], str] | None = field(default=None, repr=False)
    _annotate_skip_fn: Callable[[str], None] | None = field(default=None, repr=False)
    _commit_response_fn: Callable[[Any, dict | None], Any] | None = field(default=None, repr=False)

    # Whitelist for agent dispatch
    _public_actions: frozenset[str] = field(
        default_factory=lambda: frozenset({
            "approve",
            "reject",
            "retry",
            "validate",
        }),
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.operation:
            self.operation = "generate"

    def approve(self) -> Any:
        """Commit the current response text as the assistant message.

        If retries occurred and hide_retries is True, failed attempts
        are already SKIP-annotated. The final response is committed normally.

        Builds metadata with retry info if retries occurred.

        Returns:
            ChatResponse from the committed generation.
        """
        self._require_pending()
        self.status = PendingStatus.APPROVED

        # Build retry metadata if retries occurred
        metadata = None
        if self.retry_count > 0:
            metadata = {
                "retry_attempts": self.retry_count + 1,  # +1 for the initial attempt
                "retry_history": list(self.retry_history),
            }

        # Use _commit_response_fn to commit and get ChatResponse
        if self._commit_response_fn is not None:
            self._result = self._commit_response_fn(self._chat_response, metadata)
        elif self._execute_fn is not None:
            self._result = self._execute_fn(self)
        else:
            raise RuntimeError(
                "Cannot approve: no commit function set. "
                "This PendingGeneration was not created by Tract.generate()."
            )
        return self._result

    def reject(self, reason: str = "") -> None:
        """Reject the generation."""
        self._require_pending()
        self.status = PendingStatus.REJECTED
        self.rejection_reason = reason

    def retry(self, *, guidance: str = "", **llm_overrides: Any) -> None:
        """Reject current response and re-generate with steering.

        Flow:
        1. Record the failed response commit hash (already committed)
        2. Commit a steering user message with the diagnosis
        3. Re-generate via LLM (sees full context including failed attempt + steering)
        4. Update response_text with the new result

        SKIP annotations are deferred to approve() time so the LLM can see
        the failed attempt and steering during re-generation.

        Args:
            guidance: Diagnosis/feedback text to steer the next attempt.
            **llm_overrides: Override LLM parameters for this retry.
        """
        self._require_pending()

        if self._generate_fn is None or self._steer_fn is None:
            raise RuntimeError(
                "Cannot retry: generation functions not set. "
                "This PendingGeneration was not created by Tract.generate()."
            )

        # Record failed response hash + commit steering message.
        # SKIP annotations are deferred to _commit_response_fn (called by approve).
        steering_prompt = self.retry_prompt or "The previous response did not pass validation."
        full_guidance = f"{steering_prompt}\nDiagnosis: {guidance}" if guidance else steering_prompt
        self._steer_fn(full_guidance)

        # Track retry
        self.retry_count += 1
        if guidance:
            self.retry_history.append(guidance)

        # 3. Re-generate
        gen_result = self._generate_fn()
        self.response_text = gen_result.text if hasattr(gen_result, 'text') else str(gen_result)
        self._chat_response = gen_result

    def validate(self) -> ValidationResult:
        """Validate the current response text.

        Uses the validator callable if set. If no validator, always passes.

        Returns:
            ValidationResult indicating pass/fail.
        """
        if self.validator is None:
            return ValidationResult(passed=True)

        ok, diagnosis = self.validator(self.response_text)
        return ValidationResult(
            passed=ok,
            diagnosis=diagnosis,
        )

    # Display
    def __repr__(self):
        status = self.status.value if hasattr(self.status, 'value') else str(self.status)
        preview = self.response_text[:60] + "..." if len(self.response_text) > 60 else self.response_text
        return f"<PendingGeneration: {preview!r}, retries={self.retry_count}, {status}>"

    def _compact_detail(self) -> str:
        return f"{len(self.response_text)} chars, {self.retry_count} retries"

    def _pprint_details(self, console) -> None:
        if self.retry_count > 0:
            console.print(f"  Retries: {self.retry_count}")
            if self.retry_history:
                for i, diag in enumerate(self.retry_history):
                    console.print(f"    [{i+1}] {diag}")
