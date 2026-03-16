"""Built-in directive templates for common workflow patterns."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__: list[str] = [
    "DirectiveTemplate",
    "BUILT_IN_TEMPLATES",
    "_DEFAULT_TEMPLATES",
    "list_templates",
    "get_template",
    "register_template",
]


@dataclass(frozen=True)
class DirectiveTemplate:
    """A reusable, parameterized directive template."""

    name: str
    description: str
    content: str  # May contain {placeholder} variables
    parameters: dict[str, str] = field(default_factory=dict)  # param_name -> description

    def render(self, **kwargs: object) -> str:
        """Render the template with provided parameters.

        Raises ValueError if any declared parameters remain unresolved
        after substitution.
        """
        content = self.content
        for key, value in kwargs.items():
            content = content.replace(f"{{{key}}}", str(value))
        # Check for unresolved declared parameters (not arbitrary braces)
        missing = [p for p in self.parameters if p not in kwargs]
        if missing:
            raise ValueError(f"Unresolved template parameters: {missing}")
        return content

    def to_spec(self) -> dict[str, Any]:
        """Serialize template to a dict for persistence.

        Returns:
            Dict with all template configuration.
        """
        return {
            "name": self.name,
            "description": self.description,
            "content": self.content,
            "parameters": dict(self.parameters),
        }

    @classmethod
    def from_spec(cls, data: dict[str, Any]) -> DirectiveTemplate:
        """Reconstruct a DirectiveTemplate from a persisted spec dict."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            content=data["content"],
            parameters=data.get("parameters", {}),
        )


# ---------------------------------------------------------------------------
# Built-in templates (seed data for per-instance registries)
# ---------------------------------------------------------------------------

_DEFAULT_TEMPLATES: dict[str, DirectiveTemplate] = {}


def _register(t: DirectiveTemplate) -> DirectiveTemplate:
    _DEFAULT_TEMPLATES[t.name] = t
    return t


# --- Quality & Review ---

_register(
    DirectiveTemplate(
        name="review_protocol",
        description="Structured review process with criteria",
        content=(
            "Review Protocol:\n"
            "1. Check for correctness and completeness\n"
            "2. Evaluate {criteria} against requirements\n"
            "3. Flag any issues with severity (critical/major/minor)\n"
            "4. Provide actionable improvement suggestions\n"
            "5. Rate overall quality on a 1-10 scale"
        ),
        parameters={"criteria": "What specific criteria to evaluate"},
    )
)

_register(
    DirectiveTemplate(
        name="safety_guardrails",
        description="Safety constraints for LLM outputs",
        content=(
            "Safety Requirements:\n"
            "- Never generate harmful, misleading, or biased content\n"
            "- Always cite sources when making factual claims\n"
            "- Flag uncertainty explicitly with confidence levels\n"
            "- Respect {domain} regulatory requirements\n"
            "- Escalate to human review when confidence < {threshold}"
        ),
        parameters={
            "domain": "Domain for regulatory compliance",
            "threshold": "Confidence threshold for escalation (e.g., 0.7)",
        },
    )
)

_register(
    DirectiveTemplate(
        name="output_format",
        description="Structured output formatting requirements",
        content=(
            "Output Format:\n"
            "- Use {format} format for all responses\n"
            "- Include a summary section (max {max_words} words)\n"
            "- Tag key findings with severity/priority\n"
            "- Include actionable next steps\n"
            "- End with confidence assessment"
        ),
        parameters={
            "format": "Output format (markdown/json/plain)",
            "max_words": "Maximum words for summary",
        },
    )
)

# --- Research ---

_register(
    DirectiveTemplate(
        name="research_protocol",
        description="Systematic research methodology",
        content=(
            "Research Protocol:\n"
            "- Investigate {topic} systematically\n"
            "- Gather evidence from multiple perspectives\n"
            "- Track source reliability and recency\n"
            "- Identify contradictions and knowledge gaps\n"
            "- Synthesize findings with confidence levels\n"
            "- Tag findings: pro, con, risk, opportunity"
        ),
        parameters={"topic": "Research topic or question"},
    )
)

_register(
    DirectiveTemplate(
        name="citation_required",
        description="Require source attribution",
        content=(
            "Citation Requirements:\n"
            "- Every factual claim must include a source reference\n"
            "- Rate source reliability: primary/secondary/tertiary\n"
            "- Note publication date and relevance to {context}\n"
            "- Flag claims with no available source as [UNVERIFIED]"
        ),
        parameters={"context": "The context for evaluating source relevance"},
    )
)

# --- Coding ---

_register(
    DirectiveTemplate(
        name="code_review",
        description="Code review standards",
        content=(
            "Code Review Standards:\n"
            "- Check for {language} best practices and idioms\n"
            "- Verify error handling covers edge cases\n"
            "- Assess test coverage for new/changed code\n"
            "- Flag security vulnerabilities (OWASP top 10)\n"
            "- Evaluate performance implications\n"
            "- Ensure backward compatibility"
        ),
        parameters={"language": "Programming language"},
    )
)

_register(
    DirectiveTemplate(
        name="implementation_plan",
        description="Structured implementation planning",
        content=(
            "Implementation Protocol:\n"
            "- Break {task} into discrete, testable steps\n"
            "- Identify dependencies and blockers\n"
            "- Estimate complexity per step (low/medium/high)\n"
            "- Define acceptance criteria for each step\n"
            "- Plan rollback strategy for risky changes"
        ),
        parameters={"task": "The implementation task"},
    )
)

# --- E-commerce ---

_register(
    DirectiveTemplate(
        name="brand_voice",
        description="Brand consistency guidelines",
        content=(
            "Brand Voice Guidelines:\n"
            "- Tone: {tone}\n"
            "- Target audience: {audience}\n"
            "- Key messaging pillars: {pillars}\n"
            "- Avoid: jargon, hyperbole, unsubstantiated claims\n"
            "- Always include a clear call-to-action"
        ),
        parameters={
            "tone": "Brand tone (e.g., professional, casual, authoritative)",
            "audience": "Target audience description",
            "pillars": "Key messaging themes",
        },
    )
)

_register(
    DirectiveTemplate(
        name="conversion_optimization",
        description="Conversion-focused content guidelines",
        content=(
            "Conversion Optimization:\n"
            "- Lead with the primary benefit for {segment}\n"
            "- Include social proof and credibility signals\n"
            "- Create urgency without false scarcity\n"
            "- A/B test headlines and CTAs\n"
            "- Track: click-through, engagement, conversion metrics"
        ),
        parameters={"segment": "Target customer segment"},
    )
)


# ---------------------------------------------------------------------------
# Backward-compat alias: BUILT_IN_TEMPLATES points to _DEFAULT_TEMPLATES
# ---------------------------------------------------------------------------

BUILT_IN_TEMPLATES = _DEFAULT_TEMPLATES


# ---------------------------------------------------------------------------
# Module-level public API (operates on the global _DEFAULT_TEMPLATES dict)
# ---------------------------------------------------------------------------


def list_templates() -> list[DirectiveTemplate]:
    """Return all registered templates from the global registry."""
    return list(_DEFAULT_TEMPLATES.values())


def get_template(name: str) -> DirectiveTemplate:
    """Get a template by name from the global registry. Raises KeyError if not found."""
    if name not in _DEFAULT_TEMPLATES:
        available = ", ".join(sorted(_DEFAULT_TEMPLATES.keys()))
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return _DEFAULT_TEMPLATES[name]


def register_template(template: DirectiveTemplate) -> None:
    """Register a custom directive template in the global registry."""
    _DEFAULT_TEMPLATES[template.name] = template


def default_template_registry() -> dict[str, DirectiveTemplate]:
    """Return a fresh copy of the default templates for per-instance use."""
    return dict(_DEFAULT_TEMPLATES)
