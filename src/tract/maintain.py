"""Semantic maintainers for tract middleware.

A SemanticMaintainer is a callable that plugs into tract's middleware system
via ``t.middleware.add(event, maintainer_instance)``.  When fired, it builds a lightweight
manifest from the commit log and active config, sends it to an LLM with
maintenance instructions, and executes returned actions against existing
tract primitives.

Unlike gates (which only block), maintainers perform maintenance actions:
annotate, compress, compress_range, configure, directive, edit, tag, gc, block.

When ``max_peeks > 0``, the maintainer runs a two-pass flow: it first asks
the LLM which commits (if any) need full content inspection, fetches those
contents, then makes a second LLM call with the enriched context to decide
on actions.  This keeps the common case cheap (manifest only) while allowing
content-aware decisions when metadata alone is insufficient.

Example::

    from tract.maintain import SemanticMaintainer

    maintainer = SemanticMaintainer(
        name="cleanup",
        instructions="Mark stale tool_io commits as SKIP and compress old dialogue",
        actions=["annotate", "compress"],
        max_peeks=3,  # allow inspecting up to 3 commits
    )
    t.middleware.add("post_commit", maintainer)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from tract._helpers import safe_llm_call as _safe_llm_call
from tract._helpers import strip_fences
from tract.exceptions import BlockedError
from tract.gate import build_manifest as _build_manifest

if TYPE_CHECKING:
    from tract.context_view import ContextView
    from tract.middleware import MiddlewareContext
    from tract.tract import Tract

__all__: list[str] = [
    "SemanticMaintainer",
    "MaintainResult",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_MAINTAINER_SYSTEM_PROMPT = """\
You are a context maintenance agent. Your job is to keep the context clean, relevant, and well-organized.

You will receive maintenance instructions, a context manifest, and a list of allowed actions.

Respond with JSON:
{
  "reasoning": "Brief explanation of what you're doing and why",
  "actions": [
    {"type": "annotate", "target": "<commit_hash_prefix>", "priority": "skip"},
    {"type": "compress", "commits": ["<hash1>", "<hash2>"], "instructions": "Summarize these"},
    {"type": "compress_range", "from": "<oldest_hash>", "to": "<newest_hash>", "instructions": "Summarize this range"},
    {"type": "edit", "target": "<commit_hash_prefix>", "content": "Replacement content (shorter/cleaner)"},
    {"type": "configure", "key": "stage", "value": "implementation"},
    {"type": "directive", "name": "current-phase", "text": "Focus on implementation"},
    {"type": "tag", "target": "<hash>", "tag": "key-finding"},
    {"type": "gc"},
    {"type": "block", "reason": "Explain why this operation should be blocked"}
  ]
}

Only use action types from the allowed list. If no maintenance is needed, return {"reasoning": "...", "actions": []}.

Valid priority values for annotate: "skip", "normal", "important", "pinned".
Use "compress_range" to summarize a contiguous range of commits (easier than listing individual hashes).
Use "edit" to rewrite a single commit's content (e.g., shorten a verbose tool result).
"""

_PEEK_SYSTEM_PROMPT = """\
You are a context maintenance agent reviewing a manifest of commits.

The manifest shows metadata (hash, type, tokens, tags, message) but NOT the full content.
If you need to see the full content of specific commits to make your maintenance decisions, list their hashes.
If the manifest provides enough information, you may skip peeking and provide your actions directly.

Respond with ONE of:
1. A peek request: {"peek": ["<hash1>", "<hash2>"]}
2. Direct actions (if no peeking needed): {"reasoning": "...", "actions": [...]}

Maximum commits you may peek at: {max_peeks}
"""


def build_manifest(tract: Tract, max_log_entries: int = 30) -> str:
    """Build a text manifest from log entries and active config.

    Delegates to :func:`tract.gate.build_manifest`.
    """
    return _build_manifest(tract, max_log_entries)


# ---------------------------------------------------------------------------
# MaintainResult -- result of a single maintenance run
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MaintainResult:
    """Result of a single maintenance run."""

    maintainer_name: str
    actions_requested: int
    actions_executed: int
    actions_failed: int
    tokens_used: int
    reasoning: str
    errors: tuple[str, ...] = ()
    peeks_requested: int = 0
    peeks_performed: int = 0
    consulted_hashes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# SemanticMaintainer -- middleware-compatible callable
# ---------------------------------------------------------------------------
@dataclass
class SemanticMaintainer:
    """LLM-powered context maintenance for tract middleware.

    Register with ``t.middleware.add(event, maintainer)`` or ``t.middleware.maintain()``.
    When triggered, builds a manifest, makes one LLM call with
    maintenance instructions, and executes returned actions against
    existing tract primitives.

    When ``max_peeks > 0``, the maintainer uses a two-pass flow:

    1. **Peek pass**: Sends the manifest and asks the LLM which commits
       (if any) need full content inspection.
    2. **Action pass**: Sends the manifest enriched with peeked content
       and asks for maintenance actions.

    If the LLM decides no peeking is needed, it can return actions
    directly in the first pass (skipping the second call).

    .. note:: **Recursion guard interaction**

       Actions like ``configure`` and ``directive`` internally call
       ``t.commit()``, which fires ``post_commit`` middleware.  If this
       maintainer is registered on ``post_commit``, the tract's recursion
       guard silently drops the inner ``post_commit`` event to prevent
       infinite loops.  The inner commits *are* persisted and visible in
       the log, but other ``post_commit`` handlers will not see them.

    Attributes:
        name: Human-readable maintainer identifier.
        instructions: Natural-language instructions for what maintenance
            to perform.
        actions: List of allowed action types. Subset of:
            ``"annotate"``, ``"compress"``, ``"compress_range"``,
            ``"configure"``, ``"directive"``, ``"edit"``,
            ``"tag"``, ``"gc"``, ``"block"``.
        model: Model override passed to ``client.chat()``.
        condition: Optional deterministic pre-check. Receives the
            :class:`~tract.middleware.MiddlewareContext` and returns
            ``True`` to proceed with the LLM check, or ``False`` to
            skip entirely.
        temperature: Sampling temperature for the maintainer call.
        max_log_entries: Maximum number of commits to include in the
            manifest (newest first).
        max_peeks: Maximum commits the LLM may inspect for full content.
            0 (default) disables peeking — single LLM call only.
    """

    name: str
    instructions: str
    actions: list[str]
    model: str | None = None
    condition: Callable[[Any], bool] | None = None
    temperature: float = 0.1
    max_log_entries: int = 30
    max_peeks: int = 0
    context: ContextView | None = None

    # Stored after each invocation so callers can inspect.
    last_result: MaintainResult | None = field(default=None, init=False, repr=False)

    # Valid action types
    VALID_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "annotate", "compress", "compress_range", "configure", "directive",
        "edit", "tag", "gc", "block",
    })

    def __post_init__(self) -> None:
        """Validate action types."""
        invalid = set(self.actions) - self.VALID_ACTIONS
        if invalid:
            raise ValueError(
                f"Invalid action types: {sorted(invalid)}. "
                f"Valid types: {sorted(self.VALID_ACTIONS)}"
            )
        if not self.actions:
            raise ValueError("At least one action type must be specified.")

    def to_spec(self) -> dict[str, Any]:
        """Serialize maintainer configuration to a dict for persistence.

        Callables (``condition``) are NOT serialized -- only a flag
        indicating whether one was present.

        Returns:
            Dict with all declarative maintainer configuration.
        """
        spec: dict[str, Any] = {
            "name": self.name,
            "instructions": self.instructions,
            "actions": list(self.actions),
            "model": self.model,
            "has_condition": self.condition is not None,
            "temperature": self.temperature,
            "max_log_entries": self.max_log_entries,
            "max_peeks": self.max_peeks,
        }
        if self.context is not None:
            spec["context"] = self.context.to_dict()
        return spec

    @classmethod
    def from_spec(cls, data: dict[str, Any]) -> SemanticMaintainer:
        """Reconstruct a SemanticMaintainer from a persisted spec dict.

        The ``condition`` callback is NOT restored (it is not serializable).
        Callers must re-register it manually if needed.
        """
        from tract.context_view import ContextView

        ctx_data = data.get("context")
        ctx_view = ContextView(**ctx_data) if ctx_data is not None else None

        return cls(
            name=data["name"],
            instructions=data["instructions"],
            actions=data.get("actions", sorted(cls.VALID_ACTIONS)),
            model=data.get("model"),
            condition=None,  # not restorable
            temperature=data.get("temperature", 0.1),
            max_log_entries=data.get("max_log_entries", 30),
            max_peeks=data.get("max_peeks", 0),
            context=ctx_view,
        )

    # ------------------------------------------------------------------
    # __call__ -- middleware handler interface
    # ------------------------------------------------------------------
    def __call__(self, ctx: MiddlewareContext) -> None:
        """Execute the maintainer. Runs actions against the tract."""

        # 1. Deterministic pre-check
        if self.condition is not None:
            try:
                should_run = self.condition(ctx)
            except Exception:
                logger.warning(
                    "Maintainer '%s' condition callback raised; skipping (no-op).",
                    self.name,
                    exc_info=True,
                )
                self.last_result = MaintainResult(
                    maintainer_name=self.name,
                    actions_requested=0,
                    actions_executed=0,
                    actions_failed=0,
                    tokens_used=0,
                    reasoning="Condition callback raised; skipping.",
                    errors=(),
                    consulted_hashes=(),
                )
                return
            if not should_run:
                self.last_result = MaintainResult(
                    maintainer_name=self.name,
                    actions_requested=0,
                    actions_executed=0,
                    actions_failed=0,
                    tokens_used=0,
                    reasoning="Condition returned False; maintainer skipped.",
                    errors=(),
                    consulted_hashes=(),
                )
                return

        # 2. Resolve LLM client
        tract = ctx.tract
        try:
            client = tract.config._resolve_llm_client("maintain")
        except RuntimeError as exc:
            self.last_result = MaintainResult(
                maintainer_name=self.name,
                actions_requested=0, actions_executed=0, actions_failed=0,
                tokens_used=0,
                reasoning="No LLM client configured; maintainer cannot evaluate.",
                errors=("No LLM client configured",),
                consulted_hashes=(),
            )
            raise RuntimeError(
                f"SemanticMaintainer '{self.name}' requires an LLM client but none "
                f"is configured.  Call t.config.configure_llm() or pass api_key= to "
                f"Tract.open()."
            ) from exc

        # 3. Build manifest via ContextView
        from tract.context_view import ContextView, build_context

        view = self.context or ContextView()
        built = build_context(view, tract, default_scope=self.max_log_entries)
        manifest = built.text
        consulted_hashes = tuple(e["hash"] for e in built.commit_entries)
        llm_kwargs: dict[str, Any] = {"temperature": self.temperature}
        if self.model is not None:
            llm_kwargs["model"] = self.model

        # 3b. Resolve prompt overrides
        maintain_prompt = tract.config.get_prompt("maintain")
        peek_prompt = tract.config.get_prompt("maintain_peek")

        # 4. Run LLM flow (single-pass or two-pass with peeking)
        tokens_used = 0
        peeks_requested = 0
        peeks_performed = 0

        if self.max_peeks > 0:
            peek_result = self._run_with_peeking(
                ctx, tract, client, manifest, llm_kwargs,
                maintain_prompt=maintain_prompt, peek_prompt=peek_prompt,
            )
            if peek_result is None:
                return  # fail-open already stored last_result
            reasoning, action_list, tokens_used, peeks_requested, peeks_performed = peek_result
        else:
            single_result = self._run_single_pass(
                ctx, tract, client, manifest, llm_kwargs,
                maintain_prompt=maintain_prompt,
            )
            if single_result is None:
                return  # fail-open already stored last_result
            reasoning, action_list, tokens_used = single_result

        # 5. Filter out disallowed action types
        allowed = set(self.actions)
        filtered_actions = []
        skipped_actions = []
        for action in action_list:
            action_type = action.get("type", "")
            if action_type in allowed:
                filtered_actions.append(action)
            else:
                skipped_actions.append(action_type)

        if skipped_actions:
            logger.info(
                "Maintainer '%s' skipped disallowed action types: %s",
                self.name,
                skipped_actions,
            )

        # 6. Sort: execute block actions LAST so maintenance completes first
        block_actions = [a for a in filtered_actions if a.get("type") == "block"]
        non_block_actions = [a for a in filtered_actions if a.get("type") != "block"]

        # 7. Execute non-block actions
        executed = 0
        failed = 0
        errors: list[str] = []

        for action in non_block_actions:
            try:
                self._execute_action(tract, action)
                executed += 1
            except Exception as exc:
                failed += 1
                action_type = action.get("type", "unknown")
                error_msg = f"Action '{action_type}' failed: {exc}"
                errors.append(error_msg)
                logger.warning(
                    "Maintainer '%s' action failed: %s",
                    self.name,
                    error_msg,
                    exc_info=True,
                )

        # 8. Store result (before potential block raise)
        self.last_result = MaintainResult(
            maintainer_name=self.name,
            actions_requested=len(filtered_actions),
            actions_executed=executed,
            actions_failed=failed,
            tokens_used=tokens_used,
            reasoning=reasoning,
            errors=tuple(errors),
            peeks_requested=peeks_requested,
            peeks_performed=peeks_performed,
            consulted_hashes=consulted_hashes,
        )

        # 9. Execute block actions (raises BlockedError)
        if block_actions:
            reasons = [
                a.get("reason") or "(no reason given)" for a in block_actions
            ]
            raise BlockedError(ctx.event, reasons)

    # ------------------------------------------------------------------
    # Single-pass flow (no peeking)
    # ------------------------------------------------------------------
    def _run_single_pass(
        self,
        ctx: MiddlewareContext,
        tract: Tract,
        client: Any,
        manifest: str,
        llm_kwargs: dict[str, Any],
        maintain_prompt: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], int] | None:
        """Single LLM call. Returns (reasoning, actions, tokens) or None on failure."""
        messages = self._build_messages(manifest, ctx, system_prompt=maintain_prompt)
        return self._safe_llm_call(client, messages, llm_kwargs)

    # ------------------------------------------------------------------
    # Two-pass flow with peeking
    # ------------------------------------------------------------------
    def _run_with_peeking(
        self,
        ctx: MiddlewareContext,
        tract: Tract,
        client: Any,
        manifest: str,
        llm_kwargs: dict[str, Any],
        maintain_prompt: str | None = None,
        peek_prompt: str | None = None,
    ) -> tuple[str, list[dict[str, Any]], int, int, int] | None:
        """Two-pass flow: peek selection → enriched action decision.

        Returns (reasoning, actions, tokens, peeks_requested, peeks_performed)
        or None on failure.
        """
        # Pass 1: Ask what to peek (or get direct actions)
        peek_messages = self._build_peek_messages(manifest, ctx, system_prompt=peek_prompt)
        pass1 = self._safe_llm_call_raw(client, peek_messages, llm_kwargs)
        if pass1 is None:
            return None
        raw_text, pass1_tokens = pass1

        # Parse pass 1 response
        peek_hashes, direct_result = self._parse_peek_or_actions(raw_text)

        if direct_result is not None:
            # LLM went straight to actions (no peeking needed)
            reasoning, action_list = direct_result
            return reasoning, action_list, pass1_tokens, 0, 0

        if not peek_hashes:
            # Empty peek list — fall through to normal action call
            messages = self._build_messages(manifest, ctx, system_prompt=maintain_prompt)
            result = self._safe_llm_call(client, messages, llm_kwargs)
            if result is None:
                return None
            reasoning, action_list, pass2_tokens = result
            return reasoning, action_list, pass1_tokens + pass2_tokens, 0, 0

        # Cap at max_peeks
        requested = len(peek_hashes)
        peek_hashes = peek_hashes[:self.max_peeks]

        # Fetch content for requested commits
        peeked_content: dict[str, str] = {}
        for h in peek_hashes:
            try:
                full_hash = tract.branches.resolve(h)
                content = tract.search.get_content(full_hash)
                if content is None:
                    peeked_content[h] = "(content not found)"
                elif isinstance(content, dict):
                    peeked_content[h] = json.dumps(content, default=str)[:2000]
                else:
                    peeked_content[h] = str(content)[:2000]
            except Exception as exc:
                peeked_content[h] = f"(could not retrieve: {str(exc)[:200]})"

        performed = len(peeked_content)

        # Pass 2: Enriched action call
        enriched_messages = self._build_enriched_messages(
            manifest, ctx, peeked_content, system_prompt=maintain_prompt,
        )
        result = self._safe_llm_call(client, enriched_messages, llm_kwargs)
        if result is None:
            return None
        reasoning, action_list, pass2_tokens = result
        return reasoning, action_list, pass1_tokens + pass2_tokens, requested, performed

    # ------------------------------------------------------------------
    # Safe LLM call helpers (fail-open)
    # ------------------------------------------------------------------
    def _safe_llm_call_raw(
        self, client: Any, messages: list[dict[str, str]], llm_kwargs: dict[str, Any]
    ) -> tuple[str, int] | None:
        """Make an LLM call, return (raw_text, tokens_used) or None on failure.

        Delegates to :func:`tract._helpers.safe_llm_call` and stores a
        fail-open :class:`MaintainResult` on ``self.last_result`` when the
        helper returns ``None``.
        """
        result = _safe_llm_call(
            client, messages, llm_kwargs,
            caller=f"Maintainer '{self.name}'",
        )
        if result is None:
            self.last_result = MaintainResult(
                maintainer_name=self.name,
                actions_requested=0, actions_executed=0, actions_failed=0,
                tokens_used=0,
                reasoning="LLM call failed; fail-open default.",
                errors=(),
                consulted_hashes=(),
            )
            return None
        return result

    def _safe_llm_call(
        self, client: Any, messages: list[dict[str, str]], llm_kwargs: dict[str, Any]
    ) -> tuple[str, list[dict[str, Any]], int] | None:
        """Make an LLM call, parse response. Returns (reasoning, actions, tokens) or None."""
        result = self._safe_llm_call_raw(client, messages, llm_kwargs)
        if result is None:
            return None
        raw_text, tokens_used = result
        reasoning, action_list = self._parse_response(raw_text)
        return reasoning, action_list, tokens_used

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def _build_messages(
        self, manifest: str, ctx: MiddlewareContext, system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Construct the LLM messages for the maintainer (single-pass)."""
        allowed_actions_str = ", ".join(sorted(self.actions))
        user_content = (
            f"=== MAINTENANCE INSTRUCTIONS ===\n"
            f"{self.instructions}\n"
            f"\n"
            f"=== ALLOWED ACTIONS ===\n"
            f"{allowed_actions_str}\n"
            f"\n"
            f"=== EVENT ===\n"
            f"{ctx.event}\n"
            f"\n"
            f"{manifest}"
        )
        return [
            {"role": "system", "content": system_prompt or _MAINTAINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _build_peek_messages(
        self, manifest: str, ctx: MiddlewareContext, system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Construct messages for the peek selection pass."""
        system = system_prompt or _PEEK_SYSTEM_PROMPT.replace(
            "{max_peeks}", str(self.max_peeks)
        )
        allowed_actions_str = ", ".join(sorted(self.actions))
        user_content = (
            f"=== MAINTENANCE INSTRUCTIONS ===\n"
            f"{self.instructions}\n"
            f"\n"
            f"=== ALLOWED ACTIONS ===\n"
            f"{allowed_actions_str}\n"
            f"\n"
            f"=== EVENT ===\n"
            f"{ctx.event}\n"
            f"\n"
            f"{manifest}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

    def _build_enriched_messages(
        self,
        manifest: str,
        ctx: MiddlewareContext,
        peeked_content: dict[str, str],
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Construct messages with peeked content for the action pass."""
        allowed_actions_str = ", ".join(sorted(self.actions))

        # Build the peeked content section
        peek_lines = ["=== PEEKED COMMIT CONTENTS ==="]
        for h, content in peeked_content.items():
            peek_lines.append(f"\n--- [{h}] ---")
            peek_lines.append(content)
        peek_section = "\n".join(peek_lines)

        user_content = (
            f"=== MAINTENANCE INSTRUCTIONS ===\n"
            f"{self.instructions}\n"
            f"\n"
            f"=== ALLOWED ACTIONS ===\n"
            f"{allowed_actions_str}\n"
            f"\n"
            f"=== EVENT ===\n"
            f"{ctx.event}\n"
            f"\n"
            f"{manifest}\n"
            f"\n"
            f"{peek_section}\n"
            f"\n"
            f"Now provide your maintenance actions based on the manifest and peeked content above."
        )
        return [
            {"role": "system", "content": system_prompt or _MAINTAINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    # Delegate to the shared helper; kept as a static method for
    # backward compatibility (tests call SemanticMaintainer._strip_fences).
    _strip_fences = staticmethod(strip_fences)

    @staticmethod
    def _parse_response(text: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse an LLM response into (reasoning, actions_list).

        Returns a tuple of (reasoning_string, list_of_action_dicts).
        On parse failure, returns empty actions (fail-open).
        """
        cleaned = SemanticMaintainer._strip_fences(text)

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                reasoning = str(data.get("reasoning") or "").strip() or "(no reasoning given)"
                actions = data.get("actions", [])
                if not isinstance(actions, list):
                    actions = []
                valid_actions = []
                for action in actions:
                    if isinstance(action, dict) and "type" in action:
                        valid_actions.append(action)
                return reasoning, valid_actions
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        return f"Could not parse maintainer response; no actions taken. Raw: {text[:200]}", []

    @staticmethod
    def _parse_peek_or_actions(
        text: str,
    ) -> tuple[list[str], tuple[str, list[dict[str, Any]]] | None]:
        """Parse a peek-pass response.

        The LLM can return either:
        - ``{"peek": ["hash1", "hash2"]}`` — wants to peek
        - ``{"reasoning": "...", "actions": [...]}`` — direct actions

        Returns:
            (peek_hashes, direct_actions_or_none)
            If peek_hashes is non-empty, direct_actions is None.
            If direct_actions is not None, peek_hashes is empty.
        """
        cleaned = SemanticMaintainer._strip_fences(text)

        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                # Check for peek request
                if "peek" in data:
                    peeks = data["peek"]
                    if isinstance(peeks, list):
                        hashes = [str(h) for h in peeks if h]
                        return hashes, None
                    return [], None

                # Check for direct actions
                if "actions" in data:
                    reasoning = str(data.get("reasoning") or "").strip() or "(no reasoning given)"
                    actions = data.get("actions", [])
                    if not isinstance(actions, list):
                        actions = []
                    valid = [a for a in actions if isinstance(a, dict) and "type" in a]
                    return [], (reasoning, valid)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Unparseable — return empty peeks, no direct actions
        return [], None

    # ------------------------------------------------------------------
    # Action execution
    # ------------------------------------------------------------------
    def _execute_action(self, tract: Tract, action: dict[str, Any]) -> None:
        """Execute a single maintenance action against the tract."""
        action_type = action["type"]

        if action_type == "annotate":
            self._exec_annotate(tract, action)
        elif action_type == "compress":
            self._exec_compress(tract, action)
        elif action_type == "compress_range":
            self._exec_compress_range(tract, action)
        elif action_type == "configure":
            self._exec_configure(tract, action)
        elif action_type == "directive":
            self._exec_directive(tract, action)
        elif action_type == "edit":
            self._exec_edit(tract, action)
        elif action_type == "tag":
            self._exec_tag(tract, action)
        elif action_type == "gc":
            self._exec_gc(tract, action)
        elif action_type == "block":
            pass  # Handled separately in __call__ (executed last)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    @staticmethod
    def _exec_annotate(tract: Tract, action: dict[str, Any]) -> None:
        """Execute an annotate action: t.annotations.set(hash, priority)."""
        from tract.models.annotations import Priority

        target = action.get("target", "")
        priority_str = str(action.get("priority", "")).lower().strip()

        priority_map = {
            "skip": Priority.SKIP,
            "normal": Priority.NORMAL,
            "important": Priority.IMPORTANT,
            "pinned": Priority.PINNED,
        }
        priority = priority_map.get(priority_str)
        if priority is None:
            raise ValueError(
                f"Invalid priority '{priority_str}'. "
                f"Valid values: {list(priority_map.keys())}"
            )

        full_hash = tract.branches.resolve(target)
        reason = action.get("reason")
        tract.annotations.set(full_hash, priority, reason=reason)

    @staticmethod
    def _exec_compress(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a compress action: t.compression.compress(commits=..., instructions=...)."""
        commits = action.get("commits", [])
        instructions = action.get("instructions")

        if not commits:
            raise ValueError("Compress action requires 'commits' list.")

        resolved = [tract.branches.resolve(c) for c in commits]
        tract.compression.compress(commits=resolved, instructions=instructions)

    @staticmethod
    def _exec_compress_range(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a compress_range action: compress a contiguous range of commits."""
        from_hash = action.get("from", "")
        to_hash = action.get("to", "")
        instructions = action.get("instructions")

        if not from_hash or not to_hash:
            raise ValueError("compress_range requires 'from' and 'to' commit hashes.")

        resolved_from = tract.branches.resolve(from_hash)
        resolved_to = tract.branches.resolve(to_hash)
        tract.compression.compress(
            from_commit=resolved_from,
            to_commit=resolved_to,
            instructions=instructions,
        )

    @staticmethod
    def _exec_edit(tract: Tract, action: dict[str, Any]) -> None:
        """Execute an edit action: rewrite a commit's content in-place."""
        from tract.models.commit import CommitOperation

        target = action.get("target", "")
        content = action.get("content", "")

        if not target:
            raise ValueError("Edit action requires a 'target' commit hash.")
        if not content:
            raise ValueError("Edit action requires 'content'.")

        full_hash = tract.branches.resolve(target)
        tract.commit(
            content={"content_type": "freeform", "text": content},
            operation=CommitOperation.EDIT,
            edit_target=full_hash,
            message=f"Maintainer edit: {target[:8]}",
        )

    @staticmethod
    def _exec_configure(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a configure action: t.config.set(**{key: value})."""
        key = action.get("key", "")
        value = action.get("value")

        if not key:
            raise ValueError("Configure action requires a 'key'.")

        tract.config.set(**{key: value})

    @staticmethod
    def _exec_directive(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a directive action: t.directive(name, text)."""
        name = action.get("name", "")
        text = action.get("text", "")

        if not name:
            raise ValueError("Directive action requires a 'name'.")
        if not text:
            raise ValueError("Directive action requires 'text'.")

        tract.directive(name, text)

    def _exec_tag(self, tract: Tract, action: dict[str, Any]) -> None:
        """Execute a tag action: t.tags.add(hash, tag_name)."""
        target = action.get("target", "")
        tag_name = action.get("tag", "")

        if not target:
            raise ValueError("Tag action requires a 'target'.")
        if not tag_name:
            raise ValueError("Tag action requires a 'tag'.")

        full_hash = tract.branches.resolve(target)

        # Auto-register the tag (idempotent — no-ops if already registered).
        tract.tags.register(tag_name, f"Auto-registered by maintainer '{self.name}'")
        tract.tags.add(full_hash, tag_name)

    @staticmethod
    def _exec_gc(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a gc action: t.compression.gc()."""
        tract.compression.gc()
