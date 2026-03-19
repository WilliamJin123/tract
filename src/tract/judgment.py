"""Judgment -- unified primitive for LLM-powered decisions over context state.

Every LLM-powered operation in tract (gates, maintenance, cherry-pick,
deduplication, autonomous operations, routing) follows the same pattern:

    1. Build a view of context state (via ContextView)
    2. Send it to an LLM with typed instructions
    3. Parse the response into a typed result (via Pydantic)
    4. Fail-open on any error (return a safe default)

Judgment is that pattern, extracted and unified.

Example::

    from tract.judgment import Judgment, GateVerdict

    j = Judgment(
        instructions="Are there at least 3 commits tagged 'key-finding'?",
        response_model=GateVerdict,
        operation_name="gate",
    )
    result = j.evaluate(t)
    if result.succeeded and result.output:
        print(result.output.result, result.output.reason)

    # Custom response models work too:
    from pydantic import BaseModel

    class Priority(BaseModel):
        level: str
        reasoning: str

    j = Judgment(
        instructions="What priority should this commit receive?",
        response_model=Priority,
        context=ContextView(scope=5, detail="full"),
    )
    result = j.evaluate(t)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from tract._helpers import (
    async_safe_llm_call as _async_safe_llm_call,
    resolve_llm_client as _resolve_llm_client,
    safe_llm_call as _safe_llm_call,
    strip_fences as _strip_fences,
)
from tract.context_view import BuiltContext, ContextView, build_context

if TYPE_CHECKING:
    from tract.tract import Tract

__all__: list[str] = [
    "Judgment",
    "JudgmentResult",
    # Preset response models
    "GateVerdict",
    "MaintenanceAction",
    "MaintenancePlan",
    "SelectionResult",
    "DedupGroups",
    "SplitPlan",
    "BooleanDecision",
    "RouteSelection",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default system prompt
# ---------------------------------------------------------------------------

_DEFAULT_SYSTEM_PROMPT = """\
You are an evaluation engine for a context management system. \
You will receive context state (commit metadata, configuration, etc.) and instructions. \
Analyze the context and respond with a JSON object matching the requested schema. \
Be precise and concise. Only include fields specified in the schema."""


# ---------------------------------------------------------------------------
# JudgmentResult -- immutable evaluation output
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgmentResult:
    """Immutable result of a Judgment evaluation.

    Attributes:
        output: Parsed ``response_model`` instance, or ``None`` on failure.
        succeeded: Whether the LLM call and response parse both succeeded.
        reasoning: Extracted reasoning from the response, or an error description.
        tokens_used: LLM token usage (0 if unavailable or on failure).
        consulted_hashes: Commit hashes the LLM saw in the built context.
        raw_response: Raw LLM text for debugging.
    """

    output: BaseModel | None
    succeeded: bool
    reasoning: str
    tokens_used: int
    consulted_hashes: tuple[str, ...] = ()
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Judgment -- typed request for LLM evaluation
# ---------------------------------------------------------------------------

@dataclass
class Judgment:
    """A typed request for LLM evaluation over tract context state.

    Build a Judgment, then call :meth:`evaluate` (sync) or :meth:`aevaluate`
    (async) to run the LLM call and get a :class:`JudgmentResult`.

    Attributes:
        instructions: What to evaluate or decide. This is the core prompt
            sent to the LLM describing the task.
        response_model: Pydantic model class for the expected JSON output.
            The model's schema is included in the prompt so the LLM knows
            what fields to return.
        system_prompt: Override the default system prompt. ``None`` uses
            :data:`_DEFAULT_SYSTEM_PROMPT`.
        context: What context to show the LLM. ``None`` uses a default
            manifest of the last 30 commits.
        model: Override the LLM model name. ``None`` uses the client default.
        temperature: LLM temperature. Low by default for deterministic decisions.
        max_tokens: Maximum response tokens. ``None`` uses the client default.
        fail_open_default: Value to use as ``JudgmentResult.output`` when the
            LLM call or parse fails. ``None`` by default.
        operation_name: Name used for LLM client resolution cascade. The
            resolved client for this operation name is tried first, then
            ``"chat"`` as fallback.
    """

    instructions: str
    response_model: type[BaseModel]
    system_prompt: str | None = None
    context: ContextView | None = None
    model: str | None = None
    temperature: float = 0.1
    max_tokens: int | None = None
    fail_open_default: Any = None
    operation_name: str = "judgment"

    def evaluate(self, tract: Tract, *, llm_client: Any = None) -> JudgmentResult:
        """Synchronous evaluation.

        Args:
            tract: The Tract instance to read context from and resolve
                LLM clients against.
            llm_client: Explicit LLM client to use. If ``None``, resolves
                via ``tract.config`` using :attr:`operation_name`.

        Returns:
            A :class:`JudgmentResult`. On any failure (no client, LLM error,
            parse error), returns a fail-open result with ``succeeded=False``.
        """
        # 1. Resolve LLM client
        client = llm_client or _resolve_llm_client(
            tract, self.operation_name, "chat",
        )
        if client is None:
            return self._fail_open(
                f"No LLM client available for '{self.operation_name}' or 'chat'",
            )

        # 2. Build context
        view = self.context or ContextView()
        built = build_context(view, tract, default_scope=30)

        # 3. Build messages
        messages = self._build_messages(built)

        # 4. Build LLM kwargs
        llm_kwargs = self._build_llm_kwargs()

        # 5. Call LLM
        result = _safe_llm_call(
            client, messages, llm_kwargs,
            caller=f"Judgment({self.operation_name})",
        )
        if result is None:
            return self._fail_open("LLM call failed")

        raw_text, tokens_used = result

        # 6. Parse response
        return self._parse_response(raw_text, tokens_used, built)

    async def aevaluate(
        self,
        tract: Tract,
        *,
        llm_client: Any = None,
    ) -> JudgmentResult:
        """Async evaluation.

        Same flow as :meth:`evaluate` but uses ``async_safe_llm_call``
        for the LLM call, supporting ``achat()`` on async-capable clients.

        Args:
            tract: The Tract instance to read context from.
            llm_client: Explicit LLM client. If ``None``, resolved via config.

        Returns:
            A :class:`JudgmentResult`.
        """
        # 1. Resolve LLM client
        client = llm_client or _resolve_llm_client(
            tract, self.operation_name, "chat",
        )
        if client is None:
            return self._fail_open(
                f"No LLM client available for '{self.operation_name}' or 'chat'",
            )

        # 2. Build context
        view = self.context or ContextView()
        built = build_context(view, tract, default_scope=30)

        # 3. Build messages
        messages = self._build_messages(built)

        # 4. Build LLM kwargs
        llm_kwargs = self._build_llm_kwargs()

        # 5. Call LLM (async)
        result = await _async_safe_llm_call(
            client, messages, llm_kwargs,
            caller=f"Judgment({self.operation_name})",
        )
        if result is None:
            return self._fail_open("Async LLM call failed")

        raw_text, tokens_used = result

        # 6. Parse response
        return self._parse_response(raw_text, tokens_used, built)

    # -- Internal helpers --------------------------------------------------

    def _build_messages(self, built: BuiltContext) -> list[dict[str, str]]:
        """Construct the system + user messages for the LLM call."""
        system_msg = self.system_prompt or _DEFAULT_SYSTEM_PROMPT
        user_msg = (
            f"=== INSTRUCTIONS ===\n{self.instructions}\n\n"
            f"{built.text}\n\n"
            f"=== RESPONSE FORMAT ===\n{_schema_instructions(self.response_model)}"
        )
        return [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

    def _build_llm_kwargs(self) -> dict[str, Any]:
        """Construct keyword arguments for the LLM client."""
        kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            kwargs["model"] = self.model
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        return kwargs

    def _parse_response(
        self,
        raw_text: str,
        tokens_used: int,
        built: BuiltContext,
    ) -> JudgmentResult:
        """Parse raw LLM text into a typed JudgmentResult."""
        consulted = tuple(
            entry["hash"] for entry in built.commit_entries
        )

        data = _extract_json(raw_text)
        if data is None:
            return JudgmentResult(
                output=self.fail_open_default,
                succeeded=False,
                reasoning=f"Failed to extract JSON from LLM response",
                tokens_used=tokens_used,
                consulted_hashes=consulted,
                raw_response=raw_text,
            )

        try:
            parsed = self.response_model.model_validate(data)
        except Exception as exc:
            return JudgmentResult(
                output=self.fail_open_default,
                succeeded=False,
                reasoning=f"Failed to validate response against {self.response_model.__name__}: {exc}",
                tokens_used=tokens_used,
                consulted_hashes=consulted,
                raw_response=raw_text,
            )

        # Extract reasoning from the parsed model if it has a reasoning field
        reasoning = ""
        for attr in ("reasoning", "reason"):
            if hasattr(parsed, attr):
                reasoning = str(getattr(parsed, attr, ""))
                break

        return JudgmentResult(
            output=parsed,
            succeeded=True,
            reasoning=reasoning,
            tokens_used=tokens_used,
            consulted_hashes=consulted,
            raw_response=raw_text,
        )

    def _fail_open(self, reason: str) -> JudgmentResult:
        """Return a fail-open result with no output."""
        logger.warning(
            "Judgment(%s) fail-open: %s", self.operation_name, reason,
        )
        return JudgmentResult(
            output=self.fail_open_default,
            succeeded=False,
            reasoning=reason,
            tokens_used=0,
        )


# ---------------------------------------------------------------------------
# Schema instruction builder
# ---------------------------------------------------------------------------

def _schema_instructions(model: type[BaseModel]) -> str:
    """Generate response format instructions from a Pydantic model schema.

    Produces a human-readable JSON schema description that tells the LLM
    exactly what fields to return, their types, and whether they are required.
    """
    schema = model.model_json_schema()
    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    lines: list[str] = ["Respond with JSON matching this schema:"]
    lines.append("{")
    for name, prop in props.items():
        type_str = _resolve_type(prop)
        desc = prop.get("description", "")
        req = " (required)" if name in required else " (optional)"
        if desc:
            lines.append(f'  "{name}": {type_str}{req}  // {desc}')
        else:
            lines.append(f'  "{name}": {type_str}{req}')
    lines.append("}")
    return "\n".join(lines)


def _resolve_type(prop: dict[str, Any]) -> str:
    """Resolve a JSON schema property to a readable type string."""
    if "anyOf" in prop:
        # Union types (e.g. str | None)
        types = []
        for variant in prop["anyOf"]:
            t = variant.get("type", "any")
            if t != "null":
                types.append(t)
        return " | ".join(types) if types else "any"
    if "enum" in prop:
        return " | ".join(f'"{v}"' for v in prop["enum"])
    if "type" in prop:
        t = prop["type"]
        if t == "array":
            items = prop.get("items", {})
            item_type = items.get("type", "any")
            return f"array[{item_type}]"
        return t
    if "$ref" in prop:
        ref = prop["$ref"]
        # Extract the model name from #/$defs/ModelName
        return ref.rsplit("/", 1)[-1] if "/" in ref else ref
    return "any"


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    """Try to extract a JSON object from potentially messy LLM output.

    Tries in order:
    1. Strip markdown fences, then direct ``json.loads``
    2. Find the first ``{...}`` block via brace-depth tracking
    """
    cleaned = _strip_fences(text)

    # Direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first { ... } block by tracking brace depth
    start = cleaned.find("{")
    if start >= 0:
        depth = 0
        for i, ch in enumerate(cleaned[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            if depth == 0:
                try:
                    return json.loads(cleaned[start : i + 1])
                except (json.JSONDecodeError, ValueError):
                    break
    return None


# ---------------------------------------------------------------------------
# Preset response models
# ---------------------------------------------------------------------------

class GateVerdict(BaseModel):
    """Response model for gate evaluations (pass/fail decisions)."""

    model_config = {"extra": "allow"}

    result: Literal["pass", "fail"]
    reason: str = ""


class MaintenanceAction(BaseModel):
    """A single maintenance action to execute against tract primitives.

    The ``type`` field selects the action kind.  Only the fields relevant
    to that action type need to be populated; the rest are ignored.
    """

    model_config = {"extra": "allow"}

    type: str  # annotate, compress, compress_range, edit, configure, directive, tag, gc, block

    target: str | None = None  # commit hash for targeted actions

    # annotate
    priority: str | None = None  # for annotate

    # shared
    reason: str | None = None

    # compress
    commits: list[str] | None = None  # for compress
    instructions: str | None = None  # for compress / compress_range / edit

    # compress_range
    from_commit: str | None = None
    to_commit: str | None = None

    # edit
    content: str | None = None

    # configure
    key: str | None = None
    value: str | None = None  # for configure (Any serialized as str)

    # directive / tag
    name: str | None = None  # for directive / tag
    text: str | None = None  # for directive
    tag: str | None = None  # for tag


class MaintenancePlan(BaseModel):
    """Response model for maintenance evaluations.

    Contains zero or more :class:`MaintenanceAction` items and an
    optional reasoning summary.
    """

    model_config = {"extra": "allow"}

    reasoning: str = ""
    actions: list[MaintenanceAction] = []


class SelectionResult(BaseModel):
    """Response model for cherry-pick / selection judgments.

    ``selected`` contains the commit hashes that the LLM chose.
    """

    model_config = {"extra": "allow"}

    reasoning: str = ""
    selected: list[str] = []


class DedupGroups(BaseModel):
    """Response model for deduplication judgments.

    Each inner list in ``groups`` contains commit hashes that the LLM
    identified as duplicates of each other.
    """

    model_config = {"extra": "allow"}

    reasoning: str = ""
    groups: list[list[str]] = []


class SplitPlan(BaseModel):
    """Response model for content splitting judgments.

    Each piece is a dict with ``"content"`` and ``"message"`` keys
    describing how to split a commit into smaller parts.
    """

    model_config = {"extra": "allow"}

    reasoning: str = ""
    pieces: list[dict[str, str]] = []


class BooleanDecision(BaseModel):
    """Response model for yes/no decisions with optional parameters.

    ``params`` carries operation-specific data (e.g. target_branch,
    branch_name) when the decision implies an action.
    """

    model_config = {"extra": "allow"}

    reasoning: str = ""
    decision: bool = False
    params: dict[str, Any] = {}


class RouteSelection(BaseModel):
    """Response model for routing judgments.

    Selects a ``target`` (branch, stage, handler, etc.) with a
    ``confidence`` score.
    """

    model_config = {"extra": "allow"}

    target: str = ""
    confidence: float = 0.0
    reasoning: str = ""
