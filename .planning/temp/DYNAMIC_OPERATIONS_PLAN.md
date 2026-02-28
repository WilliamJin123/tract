# Dynamic Operations — Plan

## Context

The hook system has a fixed set of hookable operations (`_HOOKABLE_OPS = {"compress", "gc", "rebase", "merge", "policy", "tool_result"}`). Each has a hand-crafted Pending subclass with bespoke fields and actions.

We want LLMs to define new operations at runtime — e.g., a research agent defines a "citation_check" operation, or a code agent defines a "test_gate" operation. The LLM writes the action handler code as Python strings, compiled once at registration time via `exec(compile(...))`.

This builds ON TOP of the existing system. No consolidation of policies, orchestrator, or existing hooks. Existing tests stay green. The new system extends `_HOOKABLE_OPS`, the Pending class hierarchy, and the toolkit registry.

### Design decisions (from prior conversation)
- **Option B (code-gen)**: Action handlers are Python strings, compiled once at registration
- **Security model**: `register_operation()` is a *privileged SDK call* — the caller is responsible for vetting code strings before registration. When an LLM generates an OperationSpec, the host application must review it before calling `register_operation()`. This is analogous to how `exec()` in Jupyter notebooks is the user's responsibility, not the kernel's. The `review=True` parameter on `register_operation()` returns the compiled spec for inspection without activating it.
- **`pending.fields` dict bag**: Dynamic operations use a mutable dict for runtime-defined fields (dataclass fields require class-definition-time knowledge)
- **`type()` metaprogramming**: Create real Pending subclasses at runtime so introspection (to_tools, to_dict, describe_api) works unchanged
- **Persistence**: OperationSpecs can be saved/loaded so custom operations survive Tract.close()/open()

---

## Files to create

### 1. `src/tract/hooks/dynamic.py` (~250 lines)

Core module for dynamic operation infrastructure.

```python
@dataclass(frozen=True)
class ActionDef:
    """Definition of a single action on a dynamic operation."""
    name: str                      # "verify", "flag", etc.
    description: str               # One-line docstring for introspection
    params: dict[str, str]         # {"strict": "bool", "urls": "list"} — string type names
    code: str                      # Python function body (indented into def)
    required: list[str] = field(default_factory=list)  # Params that must be provided (validated at call time)

@dataclass(frozen=True)
class OperationSpec:
    """Complete specification for a dynamic hookable operation."""
    name: str                      # "citation_check" — becomes the hook event name
    description: str               # Docstring for the generated Pending subclass
    fields: dict[str, dict]        # {"urls": {"type": "list[str]", "description": "...", "default": None}}
    actions: dict[str, ActionDef]  # {"verify": ActionDef(...)}
    version: str = "1"             # For persistence schema evolution
```

Key functions:

```python
_TYPE_MAP: dict[str, type] = {
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "list[str]": list, "list[int]": list,
    "dict[str, str]": dict, "Any": object, "None": type(None),
}

def compile_action(action_def: ActionDef) -> Callable:
    """Compile an ActionDef's code string into a callable function.

    Builds: def action_name(pending, param1: type1, param2: type2, ...): <code>
    Type annotations are emitted from params dict using _TYPE_MAP so that
    inspect.signature() returns proper annotations for to_tools() JSON Schema.

    Required params (from action_def.required) are emitted without defaults;
    optional params get default=None. A wrapper validates that all required
    params are provided before calling the compiled body.

    Compiles once at registration time. The compiled function receives
    the Pending instance as first arg (like self), plus declared params.
    Source name is '<action:action_name>' for tracebacks.

    Exec namespace: {"__builtins__": __builtins__} — action code has access
    to all Python builtins including __import__. No additional modules are
    pre-imported; action code must import what it needs explicitly.
    """

def make_dynamic_pending_class(spec: OperationSpec) -> type:
    """Create a Pending subclass from an OperationSpec using type().

    The generated class:
    - Inherits from Pending (NOT a dataclass — uses __init__ override)
    - Has a `fields: dict` attribute for runtime-defined fields
    - Has compiled action methods bound as instance methods
    - Has _public_actions frozenset including all spec actions + approve/reject/pass_through
    - Has __doc__ set from spec.description
    - Has _compact_detail() and _pprint_details() for display
    """

def spec_to_dict(spec: OperationSpec) -> dict:
    """Serialize an OperationSpec to a JSON-safe dict for persistence."""

def spec_from_dict(data: dict) -> OperationSpec:
    """Deserialize an OperationSpec from a dict."""
```

**Design detail — the generated class:**

```python
# What make_dynamic_pending_class produces (conceptually):
class PendingCitationCheck(Pending):
    """Verify source URLs survive compression."""

    _public_actions = frozenset({"approve", "reject", "pass_through", "verify", "flag"})

    def __init__(self, *, tract, fields=None, **kwargs):
        super().__init__(operation="citation_check", tract=tract, **kwargs)
        self.fields = fields or {}
        # Apply defaults from spec
        for fname, fdef in spec.fields.items():
            if fname not in self.fields and "default" in fdef:
                self.fields[fname] = fdef["default"]

    def verify(self, strict: bool = False):
        """Check all URLs are present."""
        # <compiled from ActionDef.code>

    def flag(self, reason: str):
        """Flag for manual review."""
        # <compiled from ActionDef.code>

    def _compact_detail(self):
        return f"fields={list(self.fields.keys())}"

    def _pprint_details(self, console, verbose=False):
        # Print fields dict
```

**Why NOT a dataclass:** Dynamic fields aren't known at class definition time. Using a `fields: dict` attribute instead of dataclass fields means `dataclasses.fields()` won't see them, but `to_dict()` in introspection.py will need a small addition to also check for a `fields` attribute. This is a 3-line change.

**Introspection compatibility:**
- `to_tools()` → works unchanged — compiled functions have proper type annotations (emitted from `params` dict via `_TYPE_MAP`), so `inspect.signature()` returns correct annotations and `_python_type_to_json_schema()` maps them correctly
- `to_dict()` → needs small patch: if `hasattr(pending, 'fields')`, include `pending.fields` in the output dict
- `describe_api()` → works unchanged (reads methods via `getattr` + `inspect`)
- `pprint()` → works via `_compact_detail()` and `_pprint_details()` overrides on the generated class

### 2. `src/tract/hooks/registry.py` (~120 lines)

Operation registry — tracks registered OperationSpecs and their compiled Pending classes.

```python
class OperationRegistry:
    """Registry of dynamic operations. Owned by a Tract instance."""

    def __init__(self):
        self._specs: dict[str, OperationSpec] = {}          # name -> spec
        self._classes: dict[str, type] = {}                 # name -> compiled Pending subclass

    def register(self, spec: OperationSpec) -> type:
        """Register an OperationSpec. Compiles actions, creates Pending subclass.

        Deep-copies spec.fields and spec.actions dicts at registration time
        (copy-on-input) so post-registration mutations don't corrupt the class.
        This follows the project's established immutability convention (Phase 1.3).

        Raises ValueError if name conflicts with built-in ops or is already registered.
        Returns the generated Pending subclass.
        """

    def unregister(self, name: str) -> None:
        """Remove a dynamic operation."""

    def get_class(self, name: str) -> type | None:
        """Get the compiled Pending subclass for an operation name."""

    def get_spec(self, name: str) -> OperationSpec | None:
        """Get the original spec."""

    def is_registered(self, name: str) -> bool

    @property
    def operation_names(self) -> set[str]:
        """All registered dynamic operation names."""

    def to_config(self) -> list[dict]:
        """Serialize all specs for persistence."""

    def from_config(self, configs: list[dict]) -> None:
        """Load specs from persisted config. Re-compiles all actions."""
```

---

## Files to modify

### 3. `src/tract/tract.py` — 4 changes (~80 lines added)

**3a. Add `_operation_registry` and `_custom_hookable_ops` in `__init__` (~5 lines)**
```python
# In __init__:
self._operation_registry = OperationRegistry()
self._custom_hookable_ops: set[str] = set()
```

**3b. Add `register_operation()` and `unregister_operation()` methods (~35 lines)**
```python
def register_operation(
    self,
    spec: OperationSpec,
    *,
    review: bool = False,
) -> type | None:
    """Register a dynamic hookable operation.

    SECURITY: The caller is responsible for vetting code strings in the spec
    before registration. When LLMs generate specs, the host application should
    inspect the spec (especially ActionDef.code) before calling this method.

    Args:
        spec: The operation specification with fields and action code.
        review: If True, compile and return the Pending subclass for inspection
            without activating the operation. Call again with review=False to activate.

    After registration (when review=False):
    - The operation name is added to hookable ops (instance-level set)
    - Hook handlers can be registered via t.on(spec.name, handler)
    - The operation can be fired via t.fire(spec.name, fields={...})
    - The operation appears in as_tools() output
    """
    # Validate name doesn't conflict with built-in ops
    # Compile and register (deep-copies spec internals)
    cls = self._operation_registry.register(spec)
    if review:
        # Undo registration — just return the compiled class for inspection
        self._operation_registry.unregister(spec.name)
        return cls
    self._custom_hookable_ops.add(spec.name)
    return None

def unregister_operation(self, name: str) -> None:
    """Remove a dynamic operation and clean up all associated state.

    Removes from registry, hookable ops set, and any registered handlers.
    """
    self._operation_registry.unregister(name)
    self._custom_hookable_ops.discard(name)
    self.off(name)  # Remove all handlers for this operation
```

**3c. Add `fire()` method (~50 lines)**
```python
def fire(
    self,
    operation: str,
    fields: dict | None = None,
    *,
    execute_fn: Callable | None = None,
    review: bool = False,
    triggered_by: str | None = None,
) -> Any:
    """Fire a dynamic operation.

    Creates a Pending instance from the registered OperationSpec,
    attaches execute_fn (wrapped with provenance recording),
    and routes through _fire_hook().

    Args:
        operation: Registered operation name.
        fields: Dict of operation-specific fields (validated against spec).
        execute_fn: Optional finalizer called on approve().
            If None, approve() just sets status (no side effect).
        review: If True, return the Pending for manual review.
        triggered_by: Provenance string.

    Returns:
        - If review=True: the Pending instance (for manual inspection)
        - If approved with execute_fn: the execute_fn's return value
        - If approved without execute_fn: the Pending (status=APPROVED)
        - If rejected/unresolved: the Pending (status=REJECTED/PENDING)
    """
    spec = self._operation_registry.get_spec(operation)
    cls = self._operation_registry.get_class(operation)
    if cls is None or spec is None:
        raise ValueError(f"Unknown dynamic operation: {operation!r}")

    # Validate fields against spec (basic type checking)
    resolved_fields = _validate_dynamic_fields(fields or {}, spec.fields)

    pending = cls(tract=self, fields=resolved_fields, triggered_by=triggered_by)

    # Wrap execute_fn with provenance recording
    original_fn = execute_fn
    if original_fn is not None:
        def _wrapped_execute(p):
            result = original_fn(p)
            self._event_repo.create(OperationEventRow(
                event_id=uuid4().hex,
                tract_id=self._tract_id,
                event_type=f"custom:{operation}",
                params_json=json.dumps(p.fields),
            ))
            return result
        pending._execute_fn = _wrapped_execute
    # No execute_fn: approve() sets status only, no _execute_fn needed
    # (Pending base class already handles _execute_fn=None)

    if review:
        return pending

    self._fire_hook(pending)
    # Return based on outcome:
    # - Approved with execute_fn: return the result (may be None — that's valid)
    # - Approved without execute_fn: return the pending (status=APPROVED)
    # - Rejected/unresolved: return the pending
    if pending.status == PendingStatus.APPROVED and original_fn is not None:
        return pending._result
    return pending
```

**3c-helper. Add `_validate_dynamic_fields()` (~15 lines)**
```python
_FIELD_TYPE_CHECK: dict[str, type | tuple[type, ...]] = {
    "str": str, "int": (int,), "float": (int, float), "bool": bool,
    "list": list, "dict": dict, "list[str]": list, "list[int]": list,
}

def _validate_dynamic_fields(fields: dict, field_specs: dict) -> dict:
    """Validate and apply defaults for dynamic operation fields.

    Raises ValueError if a provided field value doesn't match its declared type.
    Applies defaults from spec for missing fields.
    """
    result = dict(fields)
    for fname, fdef in field_specs.items():
        if fname not in result:
            if "default" in fdef:
                result[fname] = fdef["default"]
            # Missing with no default is OK — not all fields are required
        elif fdef.get("type") in _FIELD_TYPE_CHECK:
            expected = _FIELD_TYPE_CHECK[fdef["type"]]
            if not isinstance(result[fname], expected):
                raise ValueError(
                    f"Field {fname!r}: expected {fdef['type']}, got {type(result[fname]).__name__}"
                )
    return result
```

**3d. Modify `on()` validation to accept dynamic ops (~3 lines)**
```python
# In on(), change the validation:
# BEFORE:
if operation != "*" and operation not in self._HOOKABLE_OPS:
    raise ValueError(...)

# AFTER:
_all_hookable = self._HOOKABLE_OPS | self._custom_hookable_ops
if operation != "*" and operation not in _all_hookable:
    raise ValueError(...)
```

### 4. `src/tract/hooks/introspection.py` — 1 change (~5 lines)

In `pending_to_dict()`, after iterating dataclass fields, also include dynamic fields:

```python
# After the dataclass fields loop:
if hasattr(pending, 'fields') and isinstance(pending.fields, dict):
    for k, v in pending.fields.items():
        fields[k] = _serialize_value(_truncate_for_llm(v))
```

### 5. `src/tract/toolkit/definitions.py` — 1 change (~15 lines)

In `get_all_tools()`, after building the static 15 tools, append tools for dynamic operations:

```python
# At end of get_all_tools():
# _operation_registry is always initialized in __init__, so no hasattr check needed
for op_name in tract._operation_registry.operation_names:
    spec = tract._operation_registry.get_spec(op_name)
    tools.append(ToolDefinition(
        name=f"fire_{op_name}",
        description=spec.description,
        parameters=_spec_fields_to_json_schema(spec.fields),
        handler=lambda fields=None, _name=op_name: tract.fire(_name, fields=fields),
    ))
```

Note: Tool lists from `get_all_tools()` are snapshots — if operations are registered/unregistered after the call, the tool list is stale. Callers should call `get_all_tools()` fresh when the registry changes.

Plus a small helper `_spec_fields_to_json_schema()` (~10 lines) that converts the spec's field definitions to JSON Schema format.

### 6. `src/tract/hooks/__init__.py` — add exports (~3 lines)

```python
from tract.hooks.dynamic import ActionDef, OperationSpec
from tract.hooks.registry import OperationRegistry
```

---

## Persistence (optional, can defer)

Store operation specs in `_trace_meta` table (same pattern as policy config persistence):

```python
# On Tract:
def save_operation_specs(self) -> None:
    """Persist all dynamic operation specs to _trace_meta."""
    config = self._operation_registry.to_config()
    self._meta_repo.set("operation_specs", json.dumps(config))

def load_operation_specs(self) -> None:
    """Load persisted operation specs and re-register them."""
    raw = self._meta_repo.get("operation_specs")
    if raw:
        self._operation_registry.from_config(json.loads(raw))
```

Can be wired into `Tract.open()` / `Tract.close()` later, same as policy config. **Defer to a follow-up.**

---

## Provenance

Provenance recording is integrated directly into the `fire()` method (see 3c above). When an `execute_fn` is provided, `fire()` wraps it with `_wrapped_execute` that writes an `OperationEventRow` after successful execution. The `event_type` prefix `"custom:"` distinguishes dynamic operations from built-in ones in queries.

When no `execute_fn` is provided (approve-only, no side effect), no provenance event is written — there's no operation to record.

---

## Example usage

```python
from tract import Tract
from tract.hooks.dynamic import ActionDef, OperationSpec

with Tract.open(api_key=..., model=...) as t:
    # Define a custom operation
    t.register_operation(OperationSpec(
        name="citation_check",
        description="Verify source URLs survive compression",
        fields={
            "urls": {"type": "list", "description": "URLs to verify"},
            "verified": {"type": "bool", "default": False},
        },
        actions={
            "verify": ActionDef(
                name="verify",
                description="Check all URLs are present in context",
                params={"strict": "bool"},
                required=[],
                code='''
compiled = pending.tract.compile()
text = " ".join(m["content"] for m in compiled.to_dicts())
missing = [u for u in pending.fields["urls"] if u not in text]
if missing and strict:
    pending.reject(f"Missing {len(missing)} URLs: {missing}")
else:
    pending.fields["verified"] = True
    pending.approve()
''',
            ),
        },
    ))

    # Register a handler (optional)
    t.on("citation_check", lambda p: p.verify(strict=True))

    # Fire it
    result = t.fire("citation_check", fields={"urls": ["https://example.com"]})

    # Or with review=True for manual inspection
    pending = t.fire("citation_check", fields={"urls": [...]}, review=True)
    pending.pprint()
    pending.verify(strict=False)
```

**LLM-generated at runtime (with review gate):**
```python
# The LLM generates this spec during a conversation
spec = OperationSpec(
    name="quality_check",
    description="Validate context quality before compression",
    fields={
        "min_tokens": {"type": "int", "default": 100},
        "passed": {"type": "bool", "default": False},
    },
    actions={
        "check": ActionDef(
            name="check",
            description="Verify context meets quality threshold",
            params={"threshold": "int"},
            required=["threshold"],
            code='''
compiled = pending.tract.compile()
total = sum(len(m.get("content", "")) for m in compiled.to_dicts())
pending.fields["passed"] = total >= threshold
if pending.fields["passed"]:
    pending.approve()
else:
    pending.reject(f"Context too short: {total} chars < {threshold}")
''',
        ),
    },
)

# IMPORTANT: review=True lets the host inspect compiled code before activation
cls = t.register_operation(spec, review=True)
# Inspect cls, its methods, etc. — then activate:
t.register_operation(spec)  # Now it's live
```

**Security note:** Action code runs with full Python builtins (including `__import__`). When LLMs generate OperationSpecs, the host application **must** vet the code strings before calling `register_operation()`. Use `review=True` to inspect without activating. Never blindly register LLM-generated specs that use `subprocess`, `os`, `shutil`, or network I/O without human review.

---

## Test plan

### New test file: `tests/test_dynamic_operations.py` (~300 lines, ~25 tests)

**ActionDef / OperationSpec:**
- test_action_def_frozen
- test_operation_spec_frozen
- test_spec_validation (name conflicts with built-ins rejected)

**compile_action:**
- test_compile_simple_action (no params)
- test_compile_action_with_params
- test_compile_action_with_defaults
- test_compile_action_syntax_error_raises
- test_compiled_function_has_correct_signature (for introspection)
- test_compiled_function_has_type_annotations (inspect.signature returns proper annotations)
- test_compiled_function_traceback_shows_source_name
- test_compiled_action_runtime_error_propagates (KeyError/TypeError in action code)
- test_compiled_action_required_param_missing_raises (required validation)

**make_dynamic_pending_class:**
- test_generated_class_inherits_pending
- test_generated_class_has_fields_dict
- test_generated_class_has_public_actions
- test_generated_class_approve_reject_work
- test_generated_class_pprint_works
- test_generated_class_to_dict_includes_fields
- test_generated_class_to_tools_includes_actions

**OperationRegistry:**
- test_register_and_get
- test_register_duplicate_raises
- test_register_builtin_name_raises
- test_unregister
- test_to_config_from_config_roundtrip

**Tract integration:**
- test_register_operation_makes_hookable
- test_register_operation_review_mode (review=True returns class without activating)
- test_unregister_operation_removes_from_hookable
- test_unregister_operation_removes_handlers
- test_fire_with_handler
- test_fire_with_review
- test_fire_auto_approve_no_handler (returns Pending with status=APPROVED)
- test_fire_with_execute_fn_returns_result
- test_fire_without_execute_fn_returns_pending
- test_fire_field_type_validation (wrong type raises ValueError)
- test_fire_field_defaults_applied
- test_fire_unknown_operation_raises
- test_fire_provenance_event_written (OperationEventRow with "custom:" prefix)
- test_on_accepts_dynamic_operation_name
- test_dynamic_op_appears_in_as_tools
- test_spec_fields_deep_copied_on_register (mutation after register doesn't affect class)

### Verification
```bash
python -m pytest tests/test_dynamic_operations.py -v
python -m pytest tests/ -x  # full suite still green
```

---

## Scope boundaries

**In scope:**
- OperationSpec + ActionDef dataclasses
- compile_action() code-gen (with type annotations for introspection, required param validation)
- make_dynamic_pending_class() factory (with deep-copied spec internals)
- OperationRegistry (with copy-on-input for spec.fields/actions)
- Tract.register_operation() (with review=True gate) + Tract.unregister_operation() + Tract.fire()
- Field type validation at fire() time (_validate_dynamic_fields)
- Introspection compatibility (to_dict patch)
- Toolkit integration (dynamic tools in as_tools)
- Provenance (OperationEventRow for custom ops, integrated into fire())
- Tests

**Deferred:**
- Persistence (save/load specs across sessions)
- LLM-driven spec generation (prompt templates for generating OperationSpecs)
- Trigger integration (auto-firing dynamic ops based on conditions)
- GuidanceMixin on dynamic ops (two-stage for custom operations)
- Orchestrator awareness (orchestrator using dynamic tools)
