"""Operation registry for dynamic operations.

Tracks registered OperationSpecs and their compiled Pending classes.
Owned by a Tract instance.
"""

from __future__ import annotations

import copy

from tract.hooks.dynamic import (
    OperationSpec,
    make_dynamic_pending_class,
    spec_from_dict,
    spec_to_dict,
)

# Built-in operation names that cannot be overridden
_BUILTIN_OPS = frozenset({"compress", "gc", "rebase", "merge", "trigger", "tool_result"})


class OperationRegistry:
    """Registry of dynamic operations. Owned by a Tract instance."""

    def __init__(self) -> None:
        self._specs: dict[str, OperationSpec] = {}
        self._classes: dict[str, type] = {}

    def register(self, spec: OperationSpec) -> type:
        """Register an OperationSpec. Compiles actions, creates Pending subclass.

        Deep-copies spec.fields and spec.actions dicts at registration time
        (copy-on-input) so post-registration mutations don't corrupt the class.

        Raises ValueError if name conflicts with built-in ops or is already registered.
        Returns the generated Pending subclass.
        """
        if spec.name in _BUILTIN_OPS:
            raise ValueError(
                f"Cannot register '{spec.name}': conflicts with built-in operation."
            )
        if spec.name in self._specs:
            raise ValueError(
                f"Operation '{spec.name}' is already registered. "
                f"Unregister it first to re-register."
            )

        # Deep copy for immutability
        safe_spec = OperationSpec(
            name=spec.name,
            description=spec.description,
            fields=copy.deepcopy(spec.fields),
            actions=copy.deepcopy(spec.actions),
            version=spec.version,
        )

        cls = make_dynamic_pending_class(safe_spec)
        self._specs[spec.name] = safe_spec
        self._classes[spec.name] = cls
        return cls

    def unregister(self, name: str) -> None:
        """Remove a dynamic operation."""
        self._specs.pop(name, None)
        self._classes.pop(name, None)

    def get_class(self, name: str) -> type | None:
        """Get the compiled Pending subclass for an operation name."""
        return self._classes.get(name)

    def get_spec(self, name: str) -> OperationSpec | None:
        """Get the original spec."""
        return self._specs.get(name)

    def is_registered(self, name: str) -> bool:
        return name in self._specs

    @property
    def operation_names(self) -> set[str]:
        """All registered dynamic operation names."""
        return set(self._specs.keys())

    def to_config(self) -> list[dict]:
        """Serialize all specs for persistence."""
        return [spec_to_dict(s) for s in self._specs.values()]

    def from_config(self, configs: list[dict]) -> None:
        """Load specs from persisted config. Re-compiles all actions."""
        for data in configs:
            spec = spec_from_dict(data)
            if spec.name not in self._specs:
                self.register(spec)
