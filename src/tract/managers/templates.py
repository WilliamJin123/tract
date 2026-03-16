"""Template and workflow profile manager extracted from TemplateMixin."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Callable

    from tract.models.commit import CommitInfo
    from tract.profiles import WorkflowProfile
    from tract.templates import DirectiveTemplate


class TemplateManager:
    """Directive templates and workflow profiles."""

    def __init__(
        self,
        check_open: Callable | None = None,  # Callable (for consistency)
        directive_fn: Callable | None = None,  # Callable - Tract.directive
        configure_fn: Callable | None = None,  # Callable - ConfigManager.set (was configure)
    ) -> None:
        self._check_open_fn = check_open or (lambda: None)
        self._directive_fn = directive_fn
        self._configure_fn = configure_fn

        # Initialize template and profile registries from defaults
        from tract.profiles import default_profile_registry
        from tract.templates import default_template_registry

        self._template_registry = default_template_registry()
        self._profile_registry = default_profile_registry()
        self._active_profile: WorkflowProfile | None = None

    # ------------------------------------------------------------------
    # Directive templates
    # ------------------------------------------------------------------

    def apply(
        self, name: str, *, directive_name: str | None = None, **params: object
    ) -> CommitInfo:
        """Apply a directive template by name with parameters.

        Uses the per-instance template registry (seeded from defaults).

        Args:
            name: Template name (built-in or custom registered on this instance)
            directive_name: Override the directive name (defaults to template name)
            **params: Template parameters to fill in placeholders

        Returns:
            CommitInfo for the directive commit
        """
        if name not in self._template_registry:
            available = ", ".join(sorted(self._template_registry.keys()))
            raise KeyError(f"Template '{name}' not found. Available: {available}")
        template = self._template_registry[name]
        content = template.render(**params)
        return self._directive_fn(directive_name or template.name, content)

    def register(self, template: DirectiveTemplate) -> None:
        """Register a custom directive template on this instance.

        Args:
            template: A :class:`DirectiveTemplate` instance.
        """
        self._template_registry[template.name] = template

    def get(self, name: str) -> DirectiveTemplate:
        """Get a template by name from this instance's registry.

        Raises:
            KeyError: If the template name is not found.
        """
        if name not in self._template_registry:
            available = ", ".join(sorted(self._template_registry.keys()))
            raise KeyError(f"Template '{name}' not found. Available: {available}")
        return self._template_registry[name]

    def list(self) -> list:
        """List all available directive templates from this instance's registry."""
        return list(self._template_registry.values())

    # ------------------------------------------------------------------
    # Workflow profiles
    # ------------------------------------------------------------------

    def load_profile(self, name: str, *, apply_directives: bool = True) -> None:
        """Load a workflow profile, applying its config and directives.

        Uses the per-instance profile registry (seeded from defaults).

        Args:
            name: Profile name (``"coding"``, ``"research"``, ``"ecommerce"``,
                or a custom-registered name).
            apply_directives: Whether to apply the profile's directives
                (default ``True``).

        Raises:
            KeyError: If the profile name is not found.
        """
        if name not in self._profile_registry:
            available = ", ".join(sorted(self._profile_registry.keys()))
            raise KeyError(f"Profile '{name}' not found. Available: {available}")
        profile = self._profile_registry[name]

        # Apply config
        if profile.config:
            self._configure_fn(**profile.config)

        # Apply directives
        if apply_directives:
            for dir_name, content in profile.directives.items():
                self._directive_fn(dir_name, content)

        # Apply directive templates
        if profile.directive_templates:
            for tmpl_name, params in profile.directive_templates.items():
                self.apply(tmpl_name, **params)

        # Store profile reference for stage transitions
        self._active_profile = profile

    def apply_stage(self, stage_name: str) -> None:
        """Apply stage-specific config from the active workflow profile.

        Overrides configuration values for the given stage while keeping
        non-overridden settings from the base profile config.

        Args:
            stage_name: Stage name (must exist in the profile's ``stages`` dict).

        Raises:
            ValueError: If no profile is loaded or the stage name is unknown.
        """
        if self._active_profile is None:
            raise ValueError("No workflow profile loaded. Call load_profile() first.")
        profile = self._active_profile
        if stage_name not in profile.stages:
            available = ", ".join(sorted(profile.stages.keys()))
            raise ValueError(
                f"Stage '{stage_name}' not in profile '{profile.name}'. "
                f"Available: {available}"
            )
        stage_config = profile.stages[stage_name]
        self._configure_fn(**stage_config)

    @property
    def active_profile(self) -> WorkflowProfile | None:
        """The currently loaded workflow profile, or None."""
        return self._active_profile

    def register_profile(self, profile: WorkflowProfile) -> None:
        """Register a custom workflow profile on this instance.

        Args:
            profile: A :class:`WorkflowProfile` instance.
        """
        self._profile_registry[profile.name] = profile

    def get_profile(self, name: str) -> WorkflowProfile:
        """Get a workflow profile by name from this instance's registry.

        Raises:
            KeyError: If the profile name is not found.
        """
        if name not in self._profile_registry:
            available = ", ".join(sorted(self._profile_registry.keys()))
            raise KeyError(f"Profile '{name}' not found. Available: {available}")
        return self._profile_registry[name]

    def list_profiles(self) -> list:
        """List all available workflow profiles from this instance's registry."""
        return list(self._profile_registry.values())
