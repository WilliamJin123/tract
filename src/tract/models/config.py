"""Configuration models for Trace.

TractConfig holds per-tract settings.
TokenBudgetConfig controls token budget enforcement behavior.
LLMConfig holds fully-typed LLM configuration used everywhere:
operation defaults, call-time overrides, and commit-level storage.
"""

from __future__ import annotations

import enum
import types
from dataclasses import dataclass, field, fields as dc_fields
from typing import Callable, Literal, Optional

# Supported comparison operators for query_by_config.
# "between" / "not between" take a 2-element [low, high] value for inclusive range.
# "in" / "not in" take a list of values for set membership.
Operator = Literal["=", "!=", ">", "<", ">=", "<=", "in", "not in", "between", "not between"]

from pydantic import BaseModel


class BudgetAction(str, enum.Enum):
    """Action to take when token budget is exceeded."""

    WARN = "warn"
    REJECT = "reject"
    CALLBACK = "callback"


class TokenBudgetConfig(BaseModel):
    """Configuration for token budget enforcement."""

    model_config = {"arbitrary_types_allowed": True}

    max_tokens: Optional[int] = None  # None = unlimited
    action: BudgetAction = BudgetAction.WARN
    callback: Optional[Callable[[int, int], None]] = None  # (current, max) -> None


_ALIASES: dict[str, str] = {
    "stop": "stop_sequences",
    "max_completion_tokens": "max_tokens",
}

_IGNORED: frozenset[str] = frozenset({
    "messages", "tools", "tool_choice", "stream",
    "response_format", "n", "logprobs", "top_logprobs",
    "functions", "function_call",
    "system", "metadata",
})


class TractConfig(BaseModel):
    """Per-tract configuration."""

    model_config = {"arbitrary_types_allowed": True}

    db_path: str = ":memory:"
    db_url: Optional[str] = None
    tokenizer_encoding: str = "o200k_base"
    token_budget: Optional[TokenBudgetConfig] = None
    default_branch: str = "main"
    compile_cache_maxsize: int = 8
    delete_branch_on_merge: bool = False


@dataclass(frozen=True)
class LLMConfig:
    """Fully-typed LLM configuration.

    All fields are Optional -- None means 'not set / inherit from higher level.'
    Used everywhere: operation defaults, call-time overrides, commit-level storage.

    Example::

        from tract import LLMConfig
        config = LLMConfig(model="gpt-4o", temperature=0.7)
    """

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stop_sequences: tuple[str, ...] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    top_k: int | None = None
    seed: int | None = None
    extra: dict | None = None

    def __post_init__(self) -> None:
        if self.extra is not None:
            object.__setattr__(self, "extra", types.MappingProxyType(dict(self.extra)))
        if self.stop_sequences is not None and not isinstance(self.stop_sequences, tuple):
            object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))

    def __hash__(self) -> int:
        extra_hashable = tuple(sorted(self.extra.items())) if self.extra else ()
        return hash((
            self.model, self.temperature, self.top_p, self.max_tokens,
            self.stop_sequences, self.frequency_penalty, self.presence_penalty,
            self.top_k, self.seed, extra_hashable,
        ))

    @classmethod
    def from_dict(cls, d: dict | None) -> LLMConfig | None:
        """Create LLMConfig from a dict, routing unknown keys to extra.

        Applies cross-framework aliases (e.g. ``stop`` -> ``stop_sequences``)
        and drops API plumbing keys (e.g. ``messages``, ``tools``) before
        routing known/unknown fields.

        Returns None if d is None.
        """
        if d is None:
            return None
        # Copy to avoid mutating caller's dict
        d = dict(d)
        # Apply aliases: alias -> canonical (canonical wins if both present)
        for alias, canonical in _ALIASES.items():
            if alias in d:
                if canonical not in d:
                    d[canonical] = d.pop(alias)
                else:
                    del d[alias]  # canonical wins, drop alias
        # Drop API plumbing keys
        for key in _IGNORED:
            d.pop(key, None)
        known = {f.name for f in dc_fields(cls)} - {"extra"}
        known_kwargs: dict = {}
        extra_kwargs: dict = {}
        for k, v in d.items():
            if k in known:
                known_kwargs[k] = v
            else:
                extra_kwargs[k] = v
        if "stop_sequences" in known_kwargs and isinstance(known_kwargs["stop_sequences"], list):
            known_kwargs["stop_sequences"] = tuple(known_kwargs["stop_sequences"])
        return cls(**known_kwargs, extra=extra_kwargs if extra_kwargs else None)

    def to_dict(self) -> dict:
        """Convert to a flat dict, merging extra keys at top level.

        Only includes non-None fields. Tuples are converted to lists
        for JSON compatibility.
        """
        result: dict = {}
        for f in dc_fields(self):
            if f.name == "extra":
                continue
            val = getattr(self, f.name)
            if val is not None:
                if isinstance(val, tuple):
                    val = list(val)
                result[f.name] = val
        if self.extra:
            result.update(dict(self.extra))
        return result

    def non_none_fields(self) -> dict:
        """Return dict of only the named (non-extra) fields that are set."""
        result: dict = {}
        for f in dc_fields(self):
            if f.name == "extra":
                continue
            val = getattr(self, f.name)
            if val is not None:
                result[f.name] = val
        return result

    @classmethod
    def from_obj(cls, obj: object) -> LLMConfig | None:
        """Extract LLMConfig from an arbitrary object.

        Handles dataclasses (via fields), Pydantic models (via model_dump),
        and plain objects (via vars). Pipes through from_dict() for alias
        handling and field routing.

        Returns None if obj is None.
        """
        if obj is None:
            return None
        import dataclasses as _dc
        if _dc.is_dataclass(obj) and not isinstance(obj, type):
            d = {f.name: getattr(obj, f.name) for f in _dc.fields(obj)}
        elif hasattr(obj, "model_dump"):
            d = obj.model_dump()
        else:
            d = vars(obj)
        return cls.from_dict(d)


@dataclass(frozen=True)
class OperationConfigs:
    """Per-operation LLM configuration defaults.

    Each field corresponds to an LLM-powered operation on Tract.
    None means 'no operation-level override -- use tract default.'
    Frozen for safety -- use dataclasses.replace() to create modified copies.
    """

    chat: LLMConfig | None = None
    merge: LLMConfig | None = None
    compress: LLMConfig | None = None
    orchestrate: LLMConfig | None = None


@dataclass(frozen=True)
class OperationClients:
    """Per-operation LLM client overrides.

    Each field holds an LLM client (conforming to the LLMClient protocol)
    for a specific operation.  None means 'use the tract-level default client.'
    Frozen for safety -- use dataclasses.replace() to create modified copies.

    Example::

        from tract import OperationClients
        t.configure_clients(OperationClients(
            chat=openai_client,
            compress=ollama_client,
        ))
    """

    chat: object | None = None
    merge: object | None = None
    compress: object | None = None
    orchestrate: object | None = None


@dataclass(frozen=True)
class ToolSummarizationConfig:
    """Configuration for automatic tool result summarization.

    Used by :meth:`Tract.configure_tool_summarization` to set up
    a tool_result hook that summarizes results based on per-tool
    instructions and/or token count thresholds.

    Attributes:
        instructions: Per-tool summarization instructions. Keys are tool
            names, values are instruction strings passed to the LLM.
        auto_threshold: Token count threshold. Results exceeding this
            are automatically summarized.
        default_instructions: Fallback instructions for tools not in
            the ``instructions`` dict but over the threshold.
    """
    instructions: dict[str, str] = field(default_factory=dict)
    auto_threshold: int | None = None
    default_instructions: str | None = None
    include_context: bool = False
    system_prompt: str | None = None
