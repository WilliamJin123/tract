"""Layer 2: Presentation layer for LLM-optimized tool results.

Transforms raw tool execution output into LLM-friendly responses with:
- Overflow truncation with exploration hints
- Consistent metadata footers (timing, state signals)
- Error messages with actionable navigation hints
- Success output with state context

Inspired by the two-layer architecture pattern: Layer 1 (execution) stays
raw and lossless for correct operation semantics. Layer 2 (presentation)
optimizes the final output for LLM consumption.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tract.toolkit.models import ToolResult
    from tract.tract import Tract

logger = logging.getLogger(__name__)


@dataclass
class PresentationConfig:
    """Configuration for Layer 2 presentation."""

    max_output_lines: int = 200
    max_output_bytes: int = 50_000  # 50KB
    include_metadata: bool = True
    include_hints: bool = True


class ToolPresenter:
    """Transforms raw tool results into LLM-optimized responses.

    This is the Layer 2 presentation layer. It does NOT affect tool execution
    semantics (Layer 1) — only how results are communicated to the LLM.
    """

    def __init__(self, tract: Tract, config: PresentationConfig | None = None):
        self._tract = tract
        self._config = config or PresentationConfig()

    def present_success(self, output: str, tool_name: str, duration_ms: float) -> str:
        """Format a successful tool result for LLM consumption.

        Applies overflow truncation and appends metadata footer.
        """
        lines = output.split('\n')
        truncated = False
        original_lines = len(lines)
        original_bytes = len(output.encode('utf-8', errors='replace'))

        if (len(lines) > self._config.max_output_lines
                or original_bytes > self._config.max_output_bytes):
            truncated = True
            lines = lines[:self._config.max_output_lines]
            output = '\n'.join(lines)

        parts = [output]

        if truncated:
            parts.append(
                f"\n--- output truncated ({original_lines} lines, "
                f"{_format_bytes(original_bytes)}) ---"
            )
            parts.append(
                "Tip: Use more specific queries, filters, or limit "
                "parameters to narrow results."
            )

        if self._config.include_metadata:
            parts.append(self._metadata_footer(tool_name, duration_ms, success=True))

        return '\n'.join(parts)

    def present_error(
        self,
        error_msg: str,
        tool_name: str,
        duration_ms: float,
        exception: BaseException | None = None,
        hint: str = "",
    ) -> str:
        """Format a failed tool result with navigation hints.

        Always includes "what went wrong" and "what to do instead".

        Args:
            error_msg: The error message string.
            tool_name: Name of the tool that failed.
            duration_ms: Execution duration in milliseconds.
            exception: Optional exception instance (hint extracted from it).
            hint: Explicit hint string. Takes precedence over exception hint.
        """
        parts = [f"[error] {tool_name}: {error_msg}"]

        resolved_hint = hint
        if not resolved_hint and exception is not None and hasattr(exception, 'hint'):
            resolved_hint = getattr(exception, 'hint', '') or ''

        if resolved_hint and self._config.include_hints:
            parts.append(f"[hint] {resolved_hint}")

        if self._config.include_metadata:
            parts.append(self._metadata_footer(tool_name, duration_ms, success=False))

        return '\n'.join(parts)

    def present_result(self, result: ToolResult) -> str:
        """Apply Layer 2 presentation to a ToolResult.

        Convenience method that dispatches to :meth:`present_success` or
        :meth:`present_error` based on ``result.success``.

        Args:
            result: A raw ToolResult from the executor.

        Returns:
            LLM-optimized string representation.
        """
        if result.success:
            return self.present_success(result.output, result.tool_name, result.duration_ms)
        else:
            return self.present_error(
                result.error, result.tool_name, result.duration_ms,
                hint=result.hint,
            )

    def _metadata_footer(self, tool_name: str, duration_ms: float, success: bool) -> str:
        """Build consistent metadata footer.

        Intentionally lightweight — no compile() call. Token count is
        available via tract_inspect when the agent actually needs it.
        """
        status = "ok" if success else "error"
        duration_str = _format_duration(duration_ms)

        try:
            branch = self._tract.current_branch or "detached"
        except Exception:
            branch = "?"

        return f"[{status} | {duration_str} | {branch}]"


def _format_duration(ms: float) -> str:
    """Format milliseconds into human-readable duration."""
    if ms < 1:
        return "<1ms"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _format_bytes(n: int) -> str:
    """Format byte count into human-readable size."""
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def _format_tokens(n: int) -> str:
    """Format token count with k suffix for large numbers."""
    if n < 1000:
        return f"{n} tokens"
    return f"{n / 1000:.1f}k tokens"
