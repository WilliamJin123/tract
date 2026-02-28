"""Dynamic operation infrastructure for runtime-defined hookable operations.

Allows LLMs and applications to define new operations at runtime with
custom fields and compiled action handlers. Action code is Python strings
compiled once at registration time via exec(compile(...)).

Security model: register_operation() is a privileged SDK call -- the caller
is responsible for vetting code strings before registration.
"""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Mapping from string type names to Python types
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "list": list,
    "dict": dict,
    "list[str]": list,
    "list[int]": list,
    "dict[str, str]": dict,
    "Any": object,
    "None": type(None),
}


@dataclass(frozen=True)
class ActionDef:
    """Definition of a single action on a dynamic operation."""

    name: str
    description: str
    params: dict[str, str]  # {"strict": "bool", "urls": "list"}
    code: str
    required: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class OperationSpec:
    """Complete specification for a dynamic hookable operation."""

    name: str
    description: str
    fields: dict[str, dict]  # {"urls": {"type": "list[str]", "description": "...", "default": None}}
    actions: dict[str, ActionDef]
    version: str = "1"


def compile_action(action_def: ActionDef) -> Callable:
    """Compile an ActionDef's code string into a callable function.

    Builds a function: def action_name(pending, param1: type1, ...): <code>

    Type annotations are emitted from params dict using _TYPE_MAP so that
    inspect.signature() returns proper annotations for to_tools() JSON Schema.

    Required params are emitted without defaults; optional params get default=None.
    A wrapper validates that all required params are provided before calling.

    The compiled function receives the Pending instance as first arg (like self),
    plus declared params. Source name is '<action:action_name>' for tracebacks.

    Exec namespace: {"__builtins__": __builtins__} -- action code has access
    to all Python builtins including __import__. No additional modules are
    pre-imported; action code must import what it needs explicitly.
    """
    name = action_def.name
    params = action_def.params
    required = set(action_def.required)
    code = action_def.code

    # Build parameter list with type annotations
    param_parts = []
    annotations: dict[str, type] = {}
    for pname, ptype_str in params.items():
        resolved_type = _TYPE_MAP.get(ptype_str, object)
        annotations[pname] = resolved_type
        if pname in required:
            param_parts.append(pname)
        else:
            param_parts.append(f"{pname}=None")

    params_str = ", ".join(param_parts)

    # Build the function source
    # Indent code body under the def
    indented_code = "\n".join(
        f"    {line}" if line.strip() else ""
        for line in code.strip().splitlines()
    )

    func_source = f"def {name}(pending, {params_str}):\n{indented_code}\n"

    # Compile
    compiled = compile(func_source, f"<action:{name}>", "exec")
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    exec(compiled, namespace)  # noqa: S102

    raw_fn = namespace[name]

    # Apply type annotations (exec'd functions don't get them automatically)
    raw_fn.__annotations__ = annotations
    raw_fn.__doc__ = action_def.description

    # Build a wrapper that validates required params
    if required:
        def _make_wrapper(fn: Callable, req: set[str]) -> Callable:
            def wrapper(pending: Any, **kwargs: Any) -> Any:
                missing = req - set(kwargs.keys())
                # Also count params that are explicitly None as missing if required
                for k in req:
                    if k in kwargs and kwargs[k] is None:
                        missing.add(k)
                if missing:
                    raise TypeError(
                        f"Action '{name}' requires parameters: {sorted(missing)}"
                    )
                return fn(pending, **kwargs)

            # Preserve signature for introspection
            wrapper.__signature__ = inspect.signature(fn)
            wrapper.__annotations__ = fn.__annotations__
            wrapper.__doc__ = fn.__doc__
            wrapper.__name__ = fn.__name__
            wrapper.__qualname__ = fn.__qualname__
            return wrapper

        return _make_wrapper(raw_fn, required)

    return raw_fn


def make_dynamic_pending_class(spec: OperationSpec) -> type:
    """Create a Pending subclass from an OperationSpec using type().

    The generated class:
    - Inherits from Pending (NOT a dataclass -- uses __init__ override)
    - Has a `fields: dict` attribute for runtime-defined fields
    - Has compiled action methods bound as instance methods
    - Has _public_actions frozenset including all spec actions + approve/reject/pass_through
    - Has __doc__ set from spec.description
    - Has _compact_detail() and _pprint_details() for display
    """
    from tract.hooks.pending import Pending, PendingStatus, _format_value_for_display

    # Compile all actions
    compiled_actions: dict[str, Callable] = {}
    for aname, adef in spec.actions.items():
        compiled_actions[aname] = compile_action(adef)

    # Deep copy spec fields for closure safety
    spec_fields = copy.deepcopy(spec.fields)
    op_name = spec.name

    # Build the __init__
    def __init__(self, *, tract, fields=None, **kwargs):
        # Call Pending.__init__ via dataclass-generated init
        Pending.__init__(self, operation=op_name, tract=tract, **kwargs)
        # Override _public_actions (Pending's dataclass default overwrites
        # the class-level attribute set by type())
        self._public_actions = action_names
        self.fields = dict(fields) if fields else {}
        # Apply defaults from spec
        for fname, fdef in spec_fields.items():
            if fname not in self.fields and "default" in fdef:
                self.fields[fname] = copy.deepcopy(fdef["default"])

    # Build action methods that bind compiled functions as instance methods
    def _make_method(compiled_fn):
        def method(self, **kwargs):
            return compiled_fn(self, **kwargs)
        # Copy introspection attributes
        method.__doc__ = compiled_fn.__doc__
        method.__name__ = compiled_fn.__name__
        # Build proper signature: replace 'pending' param with 'self'
        sig = inspect.signature(compiled_fn)
        params = list(sig.parameters.values())
        if params and params[0].name == "pending":
            params[0] = params[0].replace(name="self")
        method.__signature__ = sig.replace(parameters=params)
        method.__annotations__ = compiled_fn.__annotations__
        return method

    # Build class dict
    action_names = frozenset({"approve", "reject", "pass_through"} | set(spec.actions.keys()))
    class_dict = {
        "__init__": __init__,
        "_public_actions": action_names,
        "__doc__": spec.description,
    }

    # Add action methods
    for aname, compiled_fn in compiled_actions.items():
        class_dict[aname] = _make_method(compiled_fn)

    # Display methods
    def _compact_detail(self):
        return f"fields={list(self.fields.keys())}"

    def _pprint_details(self, console, *, verbose=False):
        if self.fields:
            from rich.table import Table

            table = Table(title="Dynamic Fields", show_header=True, header_style="bold")
            table.add_column("Field", style="cyan")
            table.add_column("Value")
            for k, v in self.fields.items():
                table.add_row(k, _format_value_for_display(v))
            console.print(table)

    class_dict["_compact_detail"] = _compact_detail
    class_dict["_pprint_details"] = _pprint_details

    # Generate class name: citation_check -> PendingCitationCheck
    class_name = "Pending" + "".join(word.capitalize() for word in op_name.split("_"))

    return type(class_name, (Pending,), class_dict)


def spec_to_dict(spec: OperationSpec) -> dict:
    """Serialize an OperationSpec to a JSON-safe dict for persistence."""
    return {
        "name": spec.name,
        "description": spec.description,
        "version": spec.version,
        "fields": copy.deepcopy(spec.fields),
        "actions": {
            aname: {
                "name": adef.name,
                "description": adef.description,
                "params": dict(adef.params),
                "code": adef.code,
                "required": list(adef.required),
            }
            for aname, adef in spec.actions.items()
        },
    }


def spec_from_dict(data: dict) -> OperationSpec:
    """Deserialize an OperationSpec from a dict."""
    actions = {}
    for aname, adata in data.get("actions", {}).items():
        actions[aname] = ActionDef(
            name=adata["name"],
            description=adata["description"],
            params=adata["params"],
            code=adata["code"],
            required=adata.get("required", []),
        )
    return OperationSpec(
        name=data["name"],
        description=data["description"],
        fields=data.get("fields", {}),
        actions=actions,
        version=data.get("version", "1"),
    )
