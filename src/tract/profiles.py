"""Pre-built workflow profiles for common use cases.

A workflow profile bundles configuration, directives, tool profile, and
stage definitions into a single reusable package. Load a profile with
``t.load_profile("coding")`` then advance stages with ``t.apply_stage("test")``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__: list[str] = [
    "WorkflowProfile",
    "CODING",
    "RESEARCH",
    "ECOMMERCE",
    "BUILT_IN_PROFILES",
    "_DEFAULT_PROFILES",
    "get_profile",
    "list_profiles",
    "register_profile",
]


@dataclass(frozen=True)
class WorkflowProfile:
    """A pre-built configuration bundle for a specific workflow type.

    Attributes:
        name: Short identifier (e.g. ``"coding"``, ``"research"``).
        description: Human-readable summary of what this profile is for.
        config: Settings applied via ``t.configure()`` when the profile is loaded.
        directive_templates: Template names to apply via ``t.apply_template()``.
            Maps ``template_name -> {param: value}``.
        directives: Raw directives (``name -> content``) committed via ``t.directive()``.
        tool_profile: Tool profile name to use (``"self"``, ``"supervisor"``, etc.).
        stages: Stage definitions mapping ``stage_name -> config overrides``.
    """

    name: str
    description: str
    # Config settings to apply via t.configure()
    config: dict[str, object] = field(default_factory=dict)
    # Directive template names to apply (from templates.py)
    directive_templates: dict[str, dict[str, str]] = field(default_factory=dict)  # template_name -> {param: value}
    # Raw directives (name -> content) for workflow-specific instructions
    directives: dict[str, str] = field(default_factory=dict)
    # Tool profile name
    tool_profile: str = "self"
    # Stage definitions (name -> config overrides for that stage)
    stages: dict[str, dict[str, object]] = field(default_factory=dict)

    def to_spec(self) -> dict[str, Any]:
        """Serialize profile to a dict for persistence.

        All fields are JSON-serializable (no callables).

        Returns:
            Dict with all profile configuration.
        """
        return {
            "name": self.name,
            "description": self.description,
            "config": dict(self.config),
            "directive_templates": {k: dict(v) for k, v in self.directive_templates.items()},
            "directives": dict(self.directives),
            "tool_profile": self.tool_profile,
            "stages": {k: dict(v) for k, v in self.stages.items()},
        }

    @classmethod
    def from_spec(cls, data: dict[str, Any]) -> WorkflowProfile:
        """Reconstruct a WorkflowProfile from a persisted spec dict."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            config=data.get("config", {}),
            directive_templates=data.get("directive_templates", {}),
            directives=data.get("directives", {}),
            tool_profile=data.get("tool_profile", "self"),
            stages=data.get("stages", {}),
        )


# ---------------------------------------------------------------------------
# Built-in profiles
# ---------------------------------------------------------------------------

CODING = WorkflowProfile(
    name="coding",
    description="Software engineering workflow: design, implement, test, review",
    config={
        "temperature": 0.3,
        "compile_strategy": "messages",
    },
    directives={
        "methodology": (
            "Follow test-driven development: write failing tests first, then implement.\n"
            "Break complex tasks into small, verifiable steps.\n"
            "Always consider edge cases and error handling."
        ),
        "code_quality": (
            "Write clean, readable code following language idioms.\n"
            "Include meaningful variable names and minimal comments.\n"
            "Prefer simple solutions over clever ones."
        ),
    },
    tool_profile="self",
    stages={
        "design": {"temperature": 0.5, "compile_strategy": "full"},
        "implement": {"temperature": 0.2, "compile_strategy": "messages"},
        "test": {"temperature": 0.1, "compile_strategy": "messages"},
        "review": {"temperature": 0.4, "compile_strategy": "adaptive"},
    },
)

RESEARCH = WorkflowProfile(
    name="research",
    description="Research pipeline: ingest, organize, synthesize, validate",
    config={
        "temperature": 0.5,
        "compile_strategy": "full",
    },
    directives={
        "methodology": (
            "Investigate systematically from multiple perspectives.\n"
            "Track source reliability and recency.\n"
            "Identify contradictions and knowledge gaps.\n"
            "Tag findings: pro, con, risk, opportunity."
        ),
        "synthesis": (
            "Synthesize findings with explicit confidence levels.\n"
            "Distinguish between established facts and interpretations.\n"
            "Note areas of consensus and disagreement."
        ),
    },
    tool_profile="self",
    stages={
        "ingest": {"temperature": 0.3, "compile_strategy": "full"},
        "organize": {"temperature": 0.4, "compile_strategy": "messages"},
        "synthesize": {"temperature": 0.6, "compile_strategy": "adaptive"},
        "validate": {"temperature": 0.2, "compile_strategy": "full"},
    },
)

ECOMMERCE = WorkflowProfile(
    name="ecommerce",
    description="E-commerce optimization: research, create, campaign, analyze, optimize",
    config={
        "temperature": 0.6,
        "compile_strategy": "messages",
    },
    directives={
        "brand_consistency": (
            "Maintain consistent brand voice across all content.\n"
            "Lead with customer benefits, not features.\n"
            "Include clear calls-to-action in all customer-facing content."
        ),
        "data_driven": (
            "Base decisions on metrics: conversion rate, CTR, engagement.\n"
            "A/B test all major creative decisions.\n"
            "Track ROI for every campaign variant."
        ),
    },
    tool_profile="self",
    stages={
        "research": {"temperature": 0.4, "compile_strategy": "full"},
        "creative": {"temperature": 0.8, "compile_strategy": "messages"},
        "campaign": {"temperature": 0.5, "compile_strategy": "messages"},
        "analysis": {"temperature": 0.2, "compile_strategy": "adaptive"},
        "optimize": {"temperature": 0.5, "compile_strategy": "messages"},
    },
)

# ---------------------------------------------------------------------------
# Default registry (seed data for per-instance registries)
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES: dict[str, WorkflowProfile] = {
    "coding": CODING,
    "research": RESEARCH,
    "ecommerce": ECOMMERCE,
}

# Backward-compat alias
BUILT_IN_PROFILES = _DEFAULT_PROFILES


# ---------------------------------------------------------------------------
# Module-level public API (operates on the global _DEFAULT_PROFILES dict)
# ---------------------------------------------------------------------------


def get_profile(name: str) -> WorkflowProfile:
    """Get a workflow profile by name from the global registry.

    Raises:
        KeyError: If no profile with that name is registered.
    """
    if name not in _DEFAULT_PROFILES:
        available = ", ".join(sorted(_DEFAULT_PROFILES.keys()))
        raise KeyError(f"Profile '{name}' not found. Available: {available}")
    return _DEFAULT_PROFILES[name]


def list_profiles() -> list[WorkflowProfile]:
    """List all available workflow profiles from the global registry."""
    return list(_DEFAULT_PROFILES.values())


def register_profile(profile: WorkflowProfile) -> None:
    """Register a custom workflow profile in the global registry."""
    _DEFAULT_PROFILES[profile.name] = profile


def default_profile_registry() -> dict[str, WorkflowProfile]:
    """Return a fresh copy of the default profiles for per-instance use."""
    return dict(_DEFAULT_PROFILES)
