"""Semantic maintainers for tract middleware.

A SemanticMaintainer is a callable that plugs into tract's middleware system
via ``t.use(event, maintainer_instance)``.  When fired, it builds a lightweight
manifest from the commit log and active config, sends it to an LLM with
maintenance instructions, and executes returned actions against existing
tract primitives.

Unlike gates (which block), maintainers perform maintenance actions:
annotate, compress, configure, directive, tag, gc.

Example::

    from tract.maintain import SemanticMaintainer

    maintainer = SemanticMaintainer(
        name="cleanup",
        instructions="Mark stale tool_io commits as SKIP and compress old dialogue",
        actions=["annotate", "compress"],
    )
    t.use("post_commit", maintainer)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from tract.gate import SemanticGate

if TYPE_CHECKING:
    from tract.middleware import MiddlewareContext
    from tract.tract import Tract

__all__: list[str] = [
    "SemanticMaintainer",
    "MaintainResult",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt for the maintainer LLM
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
    {"type": "configure", "key": "stage", "value": "implementation"},
    {"type": "directive", "name": "current-phase", "text": "Focus on implementation"},
    {"type": "tag", "target": "<hash>", "tag": "key-finding"},
    {"type": "gc"}
  ]
}

Only use action types from the allowed list. If no maintenance is needed, return {"reasoning": "...", "actions": []}.

Valid priority values for annotate: "skip", "normal", "important", "pinned".
"""


def build_manifest(tract: Tract, max_log_entries: int = 30) -> str:
    """Build a text manifest from log entries and active config.

    Shared between SemanticGate and SemanticMaintainer.
    Uses only ``t.log()`` and ``t.get_all_configs()`` -- never
    ``t.status()`` or ``t.compile()`` to avoid recursion.
    """
    # Delegate to the gate's manifest builder which has the same logic
    # We create a temporary gate instance just to reuse the method
    _gate = SemanticGate(name="_manifest_builder", check="_", max_log_entries=max_log_entries)
    return _gate._build_manifest(tract)


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
    errors: list[str]  # descriptions of failed actions


# ---------------------------------------------------------------------------
# SemanticMaintainer -- middleware-compatible callable
# ---------------------------------------------------------------------------
@dataclass
class SemanticMaintainer:
    """LLM-powered context maintenance for tract middleware.

    Register with ``t.use(event, maintainer)`` or ``t.maintain()``.
    When triggered, builds a manifest, makes one LLM call with
    maintenance instructions, and executes returned actions against
    existing tract primitives.

    Attributes:
        name: Human-readable maintainer identifier.
        instructions: Natural-language instructions for what maintenance
            to perform.
        actions: List of allowed action types. Subset of:
            ``"annotate"``, ``"compress"``, ``"configure"``,
            ``"directive"``, ``"tag"``, ``"gc"``.
        model: Model override passed to ``client.chat()``.
        condition: Optional deterministic pre-check. Receives the
            :class:`~tract.middleware.MiddlewareContext` and returns
            ``True`` to proceed with the LLM check, or ``False`` to
            skip entirely.
        temperature: Sampling temperature for the maintainer call.
        max_log_entries: Maximum number of commits to include in the
            manifest (newest first).
    """

    name: str
    instructions: str
    actions: list[str]
    model: str | None = None
    condition: Callable[[Any], bool] | None = None
    temperature: float = 0.1
    max_log_entries: int = 30

    # Stored after each invocation so callers can inspect.
    last_result: MaintainResult | None = field(default=None, init=False, repr=False)

    # Valid action types
    VALID_ACTIONS: ClassVar[frozenset[str]] = frozenset({
        "annotate", "compress", "configure", "directive", "tag", "gc",
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
                    errors=[],
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
                    errors=[],
                )
                return

        # 2. Resolve LLM client
        tract = ctx.tract
        try:
            client = tract._resolve_llm_client("maintain")
        except RuntimeError:
            raise RuntimeError(
                f"SemanticMaintainer '{self.name}' requires an LLM client but none "
                f"is configured.  Call t.configure_llm() or pass api_key= to "
                f"Tract.open()."
            )

        # 3. Build manifest and messages
        manifest = build_manifest(tract, self.max_log_entries)
        messages = self._build_messages(manifest, ctx)

        # 4. LLM call -- fail-open on infrastructure errors
        tokens_used = 0
        try:
            llm_kwargs: dict[str, Any] = {"temperature": self.temperature}
            if self.model is not None:
                llm_kwargs["model"] = self.model

            response = client.chat(messages, **llm_kwargs)
        except Exception:
            logger.warning(
                "Maintainer '%s' LLM call failed; skipping (fail-open).",
                self.name,
                exc_info=True,
            )
            self.last_result = MaintainResult(
                maintainer_name=self.name,
                actions_requested=0,
                actions_executed=0,
                actions_failed=0,
                tokens_used=0,
                reasoning="LLM call failed; fail-open default.",
                errors=[],
            )
            return

        try:
            raw_text = client.extract_content(response)
        except Exception:
            logger.warning(
                "Maintainer '%s' failed to extract LLM response; skipping (fail-open).",
                self.name,
                exc_info=True,
            )
            self.last_result = MaintainResult(
                maintainer_name=self.name,
                actions_requested=0,
                actions_executed=0,
                actions_failed=0,
                tokens_used=0,
                reasoning="Failed to extract LLM response; fail-open default.",
                errors=[],
            )
            return

        # Track token usage if available
        try:
            usage = client.extract_usage(response) if hasattr(client, "extract_usage") else None
            if usage and isinstance(usage, dict):
                tokens_used = usage.get("total_tokens", 0)
        except Exception:
            pass  # Usage tracking is best-effort

        # 5. Parse response
        reasoning, action_list = self._parse_response(raw_text)

        # 6. Filter out disallowed action types
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

        # 7. Execute each action
        executed = 0
        failed = 0
        errors: list[str] = []

        for action in filtered_actions:
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

        # 8. Store result
        self.last_result = MaintainResult(
            maintainer_name=self.name,
            actions_requested=len(filtered_actions),
            actions_executed=executed,
            actions_failed=failed,
            tokens_used=tokens_used,
            reasoning=reasoning,
            errors=errors,
        )

    # ------------------------------------------------------------------
    # Message construction
    # ------------------------------------------------------------------
    def _build_messages(
        self, manifest: str, ctx: MiddlewareContext
    ) -> list[dict[str, str]]:
        """Construct the LLM messages for the maintainer."""
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
            {"role": "system", "content": _MAINTAINER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(text: str) -> tuple[str, list[dict[str, Any]]]:
        """Parse an LLM response into (reasoning, actions_list).

        Returns a tuple of (reasoning_string, list_of_action_dicts).
        On parse failure, returns empty actions (fail-open).
        """
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.index("\n") if "\n" in cleaned else len(cleaned)
            cleaned = cleaned[first_newline + 1:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

        # Attempt JSON parse
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                reasoning = str(data.get("reasoning") or "").strip() or "(no reasoning given)"
                actions = data.get("actions", [])
                if not isinstance(actions, list):
                    actions = []
                # Validate each action is a dict with a "type" key
                valid_actions = []
                for action in actions:
                    if isinstance(action, dict) and "type" in action:
                        valid_actions.append(action)
                return reasoning, valid_actions
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Failed to parse -- return empty actions (fail-open)
        return f"Could not parse maintainer response; no actions taken. Raw: {text[:200]}", []

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
        elif action_type == "configure":
            self._exec_configure(tract, action)
        elif action_type == "directive":
            self._exec_directive(tract, action)
        elif action_type == "tag":
            self._exec_tag(tract, action)
        elif action_type == "gc":
            self._exec_gc(tract, action)
        else:
            raise ValueError(f"Unknown action type: {action_type}")

    @staticmethod
    def _exec_annotate(tract: Tract, action: dict[str, Any]) -> None:
        """Execute an annotate action: t.annotate(hash, priority)."""
        from tract.models.annotations import Priority

        target = action.get("target", "")
        priority_str = str(action.get("priority", "")).lower().strip()

        # Map string to Priority enum
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

        # Resolve the commit hash (supports prefixes)
        full_hash = tract.resolve_commit(target)
        reason = action.get("reason")
        tract.annotate(full_hash, priority, reason=reason)

    @staticmethod
    def _exec_compress(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a compress action: t.compress(commits=..., instructions=...)."""
        commits = action.get("commits", [])
        instructions = action.get("instructions")

        if not commits:
            raise ValueError("Compress action requires 'commits' list.")

        # Resolve each commit hash
        resolved = [tract.resolve_commit(c) for c in commits]
        tract.compress(commits=resolved, instructions=instructions)

    @staticmethod
    def _exec_configure(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a configure action: t.configure(**{key: value})."""
        key = action.get("key", "")
        value = action.get("value")

        if not key:
            raise ValueError("Configure action requires a 'key'.")

        tract.configure(**{key: value})

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

    @staticmethod
    def _exec_tag(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a tag action: t.tag(hash, tag_name)."""
        target = action.get("target", "")
        tag_name = action.get("tag", "")

        if not target:
            raise ValueError("Tag action requires a 'target'.")
        if not tag_name:
            raise ValueError("Tag action requires a 'tag'.")

        full_hash = tract.resolve_commit(target)

        # Try to register the tag (ignore if already registered or not strict)
        try:
            tract.register_tag(tag_name, f"Auto-registered by maintainer '{__name__}'")
        except Exception:
            pass  # Tag may already be registered, or strict mode off

        tract.tag(full_hash, tag_name)

    @staticmethod
    def _exec_gc(tract: Tract, action: dict[str, Any]) -> None:
        """Execute a gc action: t.gc()."""
        tract.gc()
