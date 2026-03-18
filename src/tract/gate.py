"""Semantic gates for tract middleware.

A SemanticGate is a callable that plugs into tract's middleware system
via ``t.middleware.add(event, gate_instance)``.  When fired, it builds a lightweight
manifest from the commit log and active config, sends it to an LLM with
a natural-language criterion, and raises :class:`BlockedError` if the
criterion is not met.

Example::

    from tract.gate import SemanticGate

    gate = SemanticGate(
        name="research-complete",
        check="At least 3 commits tagged 'key-finding' exist",
    )
    t.middleware.add("pre_transition", gate)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from tract._helpers import safe_llm_call as _safe_llm_call
from tract._helpers import strip_fences as _strip_fences
from tract.exceptions import BlockedError

if TYPE_CHECKING:
    from tract.context_view import ContextView
    from tract.middleware import MiddlewareContext
    from tract.tract import Tract

__all__: list[str] = [
    "SemanticGate",
    "GateResult",
    "build_manifest",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared manifest builder
# ---------------------------------------------------------------------------
def build_manifest(tract: Tract, max_log_entries: int = 30) -> str:
    """Build a text manifest from log entries and active config.

    Shared by :class:`SemanticGate` and
    :class:`~tract.maintain.SemanticMaintainer`.  Uses only ``t.search.log()``
    and ``t.config.get_all()`` — never ``t.search.status()`` or ``t.compile()``
    to avoid middleware recursion.
    """
    branch = tract.current_branch or "(detached)"
    head = tract.head
    head_short = head[:8] if head else "(empty)"

    entries = tract.search.log(limit=max_log_entries)
    shown = len(entries)

    lines: list[str] = [
        "=== CONTEXT MANIFEST ===",
        f"Branch: {branch} | HEAD: {head_short} | Commits shown: {shown}",
        "",
    ]

    # Commit log table
    if entries:
        lines.append("COMMIT LOG (newest first):")
        for entry in entries:
            h = entry.commit_hash[:8]
            ctype = entry.content_type
            tokens = entry.token_count
            tags_str = ",".join(entry.tags) if entry.tags else ""
            priority = entry.effective_priority or "normal"
            msg = entry.message if entry.message else "(no message)"
            if len(msg) > 60:
                msg = msg[:57] + "..."
            lines.append(
                f"  [{h}] {ctype:<12} | {tokens:>5} tok | "
                f"tags:[{tags_str}] | {priority:<9} | \"{msg}\""
            )
        lines.append("")

    # Active configuration
    try:
        config = tract.config.get_all()
    except Exception:
        config = {}
    if config:
        lines.append(f"ACTIVE CONFIG: {json.dumps(config, default=str)}")

    # Tag summary
    if entries:
        tag_counter: Counter[str] = Counter()
        for entry in entries:
            for tag in entry.tags:
                tag_counter[tag] += 1
        if tag_counter:
            tag_summary = ", ".join(
                f"{tag}({count})" for tag, count in tag_counter.most_common()
            )
            lines.append(f"TAGS: {tag_summary}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt for the gate LLM
# ---------------------------------------------------------------------------
_GATE_SYSTEM_PROMPT = """\
You are a quality gate evaluating whether a context meets a specific criterion.

You will receive a context manifest (metadata about commits, not full content) and a criterion to check.

Respond with JSON:
{"result": "pass", "reason": "Brief explanation"}
or
{"result": "fail", "reason": "Brief explanation of what's missing"}

Be strict. Only pass if the criterion is clearly met based on the available evidence."""


# ---------------------------------------------------------------------------
# GateResult — immutable result of a gate evaluation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateResult:
    """Result of a single gate evaluation."""

    gate_name: str
    passed: bool
    reason: str
    tokens_used: int
    consulted_hashes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# SemanticGate — middleware-compatible callable
# ---------------------------------------------------------------------------
@dataclass
class SemanticGate:
    """LLM-powered quality gate for tract middleware.

    Register with ``t.middleware.add(event, gate)`` where *event* is any
    :data:`~tract.middleware.MiddlewareEvent`.

    When invoked the gate:

    1. Runs an optional deterministic ``condition`` callback — returns
       early (passes) if the callback returns ``False``.
    2. Builds a lightweight manifest from ``t.search.log()`` and
       ``t.config.get_all()`` (never ``t.search.status()`` or ``t.compile()``
       which would cause recursion).
    3. Sends the manifest plus the ``check`` criterion to an LLM.
    4. Parses the LLM response for PASS / FAIL.
    5. On FAIL, raises :class:`~tract.exceptions.BlockedError`.

    Attributes:
        name: Human-readable gate identifier.
        check: Natural-language criterion the context must satisfy.
        model: Model override passed to ``client.chat()``.  Use the
            actual model ID (e.g. ``"gpt-4o"``), not an alias.
        condition: Optional deterministic pre-check.  Receives the
            :class:`~tract.middleware.MiddlewareContext` and returns
            ``True`` to proceed with the LLM check, or ``False`` to
            skip the gate entirely (auto-pass).
        temperature: Sampling temperature for the gate call.
        max_log_entries: Maximum number of commits to include in the
            manifest (newest first).
    """

    name: str
    check: str
    model: str | None = None
    condition: Callable[[Any], bool] | None = None
    temperature: float = 0.1
    max_log_entries: int = 30
    context: ContextView | None = None

    # Stored after each invocation so callers can inspect.
    last_result: GateResult | None = field(default=None, init=False, repr=False)

    def to_spec(self) -> dict[str, Any]:
        """Serialize gate configuration to a dict for persistence.

        Callables (``condition``) are NOT serialized -- only a flag
        indicating whether one was present.  The ``context`` field is
        serialized as a dict of non-default ContextView fields when set.

        Returns:
            Dict with all declarative gate configuration.
        """
        spec: dict[str, Any] = {
            "name": self.name,
            "check": self.check,
            "model": self.model,
            "has_condition": self.condition is not None,
            "temperature": self.temperature,
            "max_log_entries": self.max_log_entries,
        }
        if self.context is not None:
            spec["context"] = self.context.to_dict()
        return spec

    @classmethod
    def from_spec(cls, data: dict[str, Any]) -> SemanticGate:
        """Reconstruct a SemanticGate from a persisted spec dict.

        The ``condition`` callback is NOT restored (it is not serializable).
        Callers must re-register it manually if needed.
        """
        context = None
        if "context" in data and data["context"]:
            from tract.context_view import ContextView

            context = ContextView(**data["context"])

        return cls(
            name=data["name"],
            check=data["check"],
            model=data.get("model"),
            condition=None,  # not restorable
            temperature=data.get("temperature", 0.1),
            max_log_entries=data.get("max_log_entries", 30),
            context=context,
        )

    # ------------------------------------------------------------------
    # __call__ — middleware handler interface
    # ------------------------------------------------------------------
    def __call__(self, ctx: MiddlewareContext) -> None:
        """Evaluate the gate.  Raises :class:`BlockedError` on failure."""

        # 1. Deterministic pre-check
        if self.condition is not None:
            try:
                should_check = self.condition(ctx)
            except Exception:
                logger.warning(
                    "Gate '%s' condition callback raised; skipping gate (pass).",
                    self.name,
                    exc_info=True,
                )
                self.last_result = GateResult(
                    gate_name=self.name,
                    passed=True,
                    reason="Condition callback raised; defaulting to pass.",
                    tokens_used=0,
                    consulted_hashes=(),
                )
                return
            if not should_check:
                self.last_result = GateResult(
                    gate_name=self.name,
                    passed=True,
                    reason="Condition returned False; gate skipped.",
                    tokens_used=0,
                    consulted_hashes=(),
                )
                return

        # 2. Resolve LLM client
        tract = ctx.tract
        try:
            client = tract.config._resolve_llm_client("gate")
        except RuntimeError as exc:
            self.last_result = GateResult(
                gate_name=self.name,
                passed=False,
                reason="No LLM client configured; gate cannot evaluate.",
                tokens_used=0,
                consulted_hashes=(),
            )
            raise RuntimeError(
                f"SemanticGate '{self.name}' requires an LLM client but none "
                f"is configured.  Call t.config.configure_llm() or pass api_key= to "
                f"Tract.open()."
            ) from exc

        # 3. Build manifest and messages
        manifest, consulted_hashes = self._build_manifest(tract)
        prompt_override = tract.config.get_prompt("gate")
        messages = self._build_messages(manifest, ctx, system_prompt=prompt_override)

        # 4. LLM call — fail-open on infrastructure errors
        llm_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            llm_kwargs["model"] = self.model

        llm_result = _safe_llm_call(
            client, messages, llm_kwargs, caller=f"Gate '{self.name}'",
        )
        if llm_result is None:
            self.last_result = GateResult(
                gate_name=self.name,
                passed=True,
                reason="LLM call failed; fail-open default.",
                tokens_used=0,
                consulted_hashes=consulted_hashes,
            )
            return

        raw_text, tokens_used = llm_result

        # 5. Parse response
        passed, reason = self._parse_response(raw_text)
        self.last_result = GateResult(
            gate_name=self.name,
            passed=passed,
            reason=reason,
            tokens_used=tokens_used,
            consulted_hashes=consulted_hashes,
        )

        # 6. Block if failed
        if not passed:
            raise BlockedError(
                ctx.event,
                [f"Gate '{self.name}' FAILED: {reason}"],
            )

    # ------------------------------------------------------------------
    # Manifest construction
    # ------------------------------------------------------------------
    def _build_manifest(self, tract: Tract) -> tuple[str, tuple[str, ...]]:
        """Build context manifest via :func:`~tract.context_view.build_context`.

        Returns (manifest_text, consulted_hashes).
        """
        from tract.context_view import ContextView, build_context

        view = self.context or ContextView()
        built = build_context(view, tract, default_scope=self.max_log_entries)
        hashes = tuple(e["hash"] for e in built.commit_entries)
        return built.text, hashes

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def _build_messages(
        self, manifest: str, ctx: MiddlewareContext, system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Construct the LLM messages for the gate evaluation."""
        user_content = (
            f"=== CRITERION ===\n"
            f"{self.check}\n"
            f"\n"
            f"=== EVENT ===\n"
            f"{ctx.event}\n"
            f"\n"
            f"{manifest}"
        )
        return [
            {"role": "system", "content": system_prompt or _GATE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str) -> tuple[bool, str]:
        """Parse an LLM response into (passed, reason).

        Tries JSON first, then falls back to scanning for PASS/FAIL
        keywords in the raw text.
        """
        # Strip markdown code fences if present
        cleaned = _strip_fences(text)

        # Attempt JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                result = str(data.get("result") or "").lower().strip()
                reason = str(data.get("reason") or "").strip() or "(no reason given)"
                if result == "pass":
                    return True, reason
                if result == "fail":
                    return False, reason
                # Unrecognised result value — fall through to keyword scan
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Fallback: word-boundary keyword scan
        text_lower = text.lower()
        if re.search(r"\bfail\b", text_lower):
            return False, text.strip()[:200]
        if re.search(r"\bpass\b", text_lower):
            return True, text.strip()[:200]

        # Ambiguous — treat as pass (fail-open)
        return True, f"Could not parse gate response; defaulting to pass. Raw: {text[:200]}"
