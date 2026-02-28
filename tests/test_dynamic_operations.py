"""Tests for dynamic operations: ActionDef, OperationSpec, compile_action,
make_dynamic_pending_class, OperationRegistry, and Tract integration.
"""

from __future__ import annotations

import inspect

import pytest

from tract import Tract
from tract.hooks.dynamic import (
    ActionDef,
    OperationSpec,
    compile_action,
    make_dynamic_pending_class,
    spec_from_dict,
    spec_to_dict,
)
from tract.hooks.pending import Pending, PendingStatus
from tract.hooks.registry import OperationRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_spec(name: str = "test_op") -> OperationSpec:
    """Create a minimal OperationSpec for testing."""
    return OperationSpec(
        name=name,
        description="A test dynamic operation",
        fields={
            "items": {"type": "list", "description": "Items to check"},
            "passed": {"type": "bool", "default": False},
        },
        actions={
            "check": ActionDef(
                name="check",
                description="Run the check",
                params={"strict": "bool"},
                code="pending.fields['passed'] = True\npending.approve()",
            ),
        },
    )


def _spec_with_required() -> OperationSpec:
    """Spec with a required param on one of its actions."""
    return OperationSpec(
        name="gated_op",
        description="Operation with required param",
        fields={
            "score": {"type": "int", "default": 0},
        },
        actions={
            "evaluate": ActionDef(
                name="evaluate",
                description="Evaluate with threshold",
                params={"threshold": "int"},
                required=["threshold"],
                code="pending.fields['score'] = threshold\npending.approve()",
            ),
        },
    )


# ===========================================================================
# ActionDef / OperationSpec dataclass tests
# ===========================================================================


class TestDataclasses:
    def test_action_def_frozen(self):
        ad = ActionDef(name="x", description="y", params={}, code="pass")
        with pytest.raises(AttributeError):
            ad.name = "z"  # type: ignore[misc]

    def test_operation_spec_frozen(self):
        spec = _simple_spec()
        with pytest.raises(AttributeError):
            spec.name = "other"  # type: ignore[misc]

    def test_spec_default_version(self):
        spec = _simple_spec()
        assert spec.version == "1"


# ===========================================================================
# compile_action tests
# ===========================================================================


class TestCompileAction:
    def test_compile_simple_action_no_params(self):
        ad = ActionDef(
            name="noop",
            description="Does nothing",
            params={},
            code="pass",
        )
        fn = compile_action(ad)
        assert callable(fn)
        # Should accept pending arg
        fn(object())

    def test_compile_action_with_params(self):
        ad = ActionDef(
            name="greet",
            description="Greet someone",
            params={"name": "str"},
            code="return f'hello {name}'",
        )
        fn = compile_action(ad)
        result = fn(None, name="world")
        assert result == "hello world"

    def test_compile_action_with_defaults(self):
        ad = ActionDef(
            name="inc",
            description="Increment",
            params={"n": "int"},
            code="return (n or 0) + 1",
        )
        fn = compile_action(ad)
        # n defaults to None when not required
        assert fn(None, n=5) == 6

    def test_compile_action_syntax_error_raises(self):
        ad = ActionDef(
            name="bad",
            description="Bad code",
            params={},
            code="def def def",
        )
        with pytest.raises(SyntaxError):
            compile_action(ad)

    def test_compiled_function_has_correct_signature(self):
        ad = ActionDef(
            name="check",
            description="Check items",
            params={"strict": "bool", "limit": "int"},
            code="pass",
        )
        fn = compile_action(ad)
        sig = inspect.signature(fn)
        param_names = list(sig.parameters.keys())
        assert "pending" in param_names
        assert "strict" in param_names
        assert "limit" in param_names

    def test_compiled_function_has_type_annotations(self):
        ad = ActionDef(
            name="typed",
            description="Typed action",
            params={"name": "str", "count": "int"},
            code="pass",
        )
        fn = compile_action(ad)
        annotations = fn.__annotations__
        assert annotations["name"] is str
        assert annotations["count"] is int

    def test_compiled_function_traceback_shows_source_name(self):
        ad = ActionDef(
            name="failing",
            description="Will fail",
            params={},
            code="raise ValueError('test error')",
        )
        fn = compile_action(ad)
        with pytest.raises(ValueError, match="test error"):
            fn(None)

    def test_compiled_action_runtime_error_propagates(self):
        ad = ActionDef(
            name="bad_access",
            description="Bad key access",
            params={},
            code="d = {}\nreturn d['missing']",
        )
        fn = compile_action(ad)
        with pytest.raises(KeyError):
            fn(None)

    def test_compiled_action_required_param_missing_raises(self):
        ad = ActionDef(
            name="req",
            description="Requires threshold",
            params={"threshold": "int"},
            required=["threshold"],
            code="return threshold",
        )
        fn = compile_action(ad)
        with pytest.raises(TypeError, match="requires parameters"):
            fn(None)  # No threshold provided

    def test_compiled_action_required_param_provided(self):
        ad = ActionDef(
            name="req",
            description="Requires threshold",
            params={"threshold": "int"},
            required=["threshold"],
            code="return threshold",
        )
        fn = compile_action(ad)
        assert fn(None, threshold=42) == 42


# ===========================================================================
# make_dynamic_pending_class tests
# ===========================================================================


class TestMakeDynamicPendingClass:
    def test_generated_class_inherits_pending(self):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        assert issubclass(cls, Pending)

    def test_generated_class_has_fields_dict(self):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()
        instance = cls(tract=t, fields={"items": [1, 2, 3]})
        assert instance.fields == {"items": [1, 2, 3], "passed": False}  # default applied

    def test_generated_class_has_public_actions(self):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()
        instance = cls(tract=t)
        assert "check" in instance._public_actions
        assert "approve" in instance._public_actions
        assert "reject" in instance._public_actions
        assert "pass_through" in instance._public_actions

    def test_generated_class_approve_reject_work(self):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()

        # Test reject
        p = cls(tract=t)
        p._execute_fn = lambda p: None
        p.reject("bad")
        assert p.status == PendingStatus.REJECTED
        assert p.rejection_reason == "bad"

    def test_generated_class_pprint_works(self, capsys):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()
        instance = cls(tract=t, fields={"items": ["a", "b"]})
        # Should not raise
        instance.pprint()

    def test_generated_class_to_dict_includes_fields(self):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()
        instance = cls(tract=t, fields={"items": ["a", "b"]})
        d = instance.to_dict()
        assert d["operation"] == "test_op"
        assert "items" in d["fields"]
        assert d["fields"]["items"] == ["a", "b"]

    def test_generated_class_to_tools_includes_actions(self):
        spec = _simple_spec()
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()
        instance = cls(tract=t)
        tools = instance.to_tools()
        tool_names = [tool["function"]["name"] for tool in tools]
        assert "check" in tool_names
        assert "approve" in tool_names
        assert "reject" in tool_names

    def test_generated_class_name(self):
        spec = OperationSpec(
            name="citation_check",
            description="Check citations",
            fields={},
            actions={},
        )
        cls = make_dynamic_pending_class(spec)
        assert cls.__name__ == "PendingCitationCheck"

    def test_generated_class_docstring(self):
        spec = OperationSpec(
            name="test",
            description="A custom description",
            fields={},
            actions={},
        )
        cls = make_dynamic_pending_class(spec)
        assert cls.__doc__ == "A custom description"

    def test_action_method_accesses_pending_fields(self):
        spec = OperationSpec(
            name="field_test",
            description="Test field access",
            fields={"counter": {"type": "int", "default": 0}},
            actions={
                "increment": ActionDef(
                    name="increment",
                    description="Increment counter",
                    params={},
                    code="pending.fields['counter'] += 1",
                ),
            },
        )
        cls = make_dynamic_pending_class(spec)
        t = Tract.open()
        instance = cls(tract=t)
        assert instance.fields["counter"] == 0
        instance.increment()
        assert instance.fields["counter"] == 1


# ===========================================================================
# OperationRegistry tests
# ===========================================================================


class TestOperationRegistry:
    def test_register_and_get(self):
        reg = OperationRegistry()
        spec = _simple_spec()
        cls = reg.register(spec)
        assert cls is not None
        assert issubclass(cls, Pending)
        assert reg.get_class("test_op") is cls
        assert reg.get_spec("test_op") is not None
        assert reg.is_registered("test_op")

    def test_register_duplicate_raises(self):
        reg = OperationRegistry()
        spec = _simple_spec()
        reg.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(spec)

    def test_register_builtin_name_raises(self):
        reg = OperationRegistry()
        for name in ["compress", "gc", "rebase", "merge", "policy", "tool_result"]:
            spec = OperationSpec(name=name, description="x", fields={}, actions={})
            with pytest.raises(ValueError, match="built-in"):
                reg.register(spec)

    def test_unregister(self):
        reg = OperationRegistry()
        spec = _simple_spec()
        reg.register(spec)
        reg.unregister("test_op")
        assert not reg.is_registered("test_op")
        assert reg.get_class("test_op") is None

    def test_operation_names(self):
        reg = OperationRegistry()
        reg.register(_simple_spec("op_a"))
        reg.register(_simple_spec("op_b"))
        assert reg.operation_names == {"op_a", "op_b"}

    def test_to_config_from_config_roundtrip(self):
        reg = OperationRegistry()
        spec = _simple_spec()
        reg.register(spec)
        config = reg.to_config()

        reg2 = OperationRegistry()
        reg2.from_config(config)
        assert reg2.is_registered("test_op")
        assert reg2.get_spec("test_op").description == spec.description

    def test_spec_fields_deep_copied_on_register(self):
        """Mutation after register doesn't affect the registered class."""
        fields = {"counter": {"type": "int", "default": 0}}
        spec = OperationSpec(
            name="copy_test",
            description="Test copy",
            fields=fields,
            actions={},
        )
        reg = OperationRegistry()
        reg.register(spec)
        # Mutate original
        fields["counter"]["default"] = 999
        # Registered spec should be unaffected
        stored = reg.get_spec("copy_test")
        assert stored.fields["counter"]["default"] == 0


# ===========================================================================
# spec_to_dict / spec_from_dict tests
# ===========================================================================


class TestSpecSerialization:
    def test_roundtrip(self):
        spec = _simple_spec()
        d = spec_to_dict(spec)
        restored = spec_from_dict(d)
        assert restored.name == spec.name
        assert restored.description == spec.description
        assert restored.version == spec.version
        assert set(restored.actions.keys()) == set(spec.actions.keys())
        assert restored.fields == spec.fields

    def test_roundtrip_with_required(self):
        spec = _spec_with_required()
        d = spec_to_dict(spec)
        restored = spec_from_dict(d)
        assert restored.actions["evaluate"].required == ["threshold"]


# ===========================================================================
# Tract integration tests
# ===========================================================================


class TestTractIntegration:
    def test_register_operation_makes_hookable(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)
        # Should be able to register a hook handler
        t.on("test_op", lambda p: p.approve())

    def test_register_operation_review_mode(self):
        t = Tract.open()
        spec = _simple_spec()
        cls = t.register_operation(spec, review=True)
        assert cls is not None
        assert issubclass(cls, Pending)
        # Operation should NOT be activated
        with pytest.raises(ValueError, match="not a hookable operation"):
            t.on("test_op", lambda p: p.approve())

    def test_unregister_operation_removes_from_hookable(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)
        t.unregister_operation("test_op")
        with pytest.raises(ValueError, match="not a hookable operation"):
            t.on("test_op", lambda p: p.approve())

    def test_unregister_operation_removes_handlers(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)
        t.on("test_op", lambda p: p.approve())
        t.unregister_operation("test_op")
        assert "test_op" not in t.hooks

    def test_fire_with_handler(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        decisions = []
        t.on("test_op", lambda p: (decisions.append("fired"), p.approve()))

        result = t.fire("test_op", fields={"items": [1, 2, 3]})
        assert len(decisions) == 1
        assert result.status == PendingStatus.APPROVED

    def test_fire_with_review(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        pending = t.fire("test_op", fields={"items": ["a"]}, review=True)
        assert pending.status == PendingStatus.PENDING
        assert pending.fields["items"] == ["a"]
        assert pending.operation == "test_op"

    def test_fire_auto_approve_no_handler(self):
        """No handler registered -> auto-approve via _fire_hook."""
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        result = t.fire("test_op", fields={"items": []})
        assert result.status == PendingStatus.APPROVED

    def test_fire_with_execute_fn_returns_result(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        def my_execute(pending):
            return {"result": "done", "items": pending.fields.get("items")}

        result = t.fire("test_op", fields={"items": [1]}, execute_fn=my_execute)
        assert result == {"result": "done", "items": [1]}

    def test_fire_without_execute_fn_returns_pending(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        result = t.fire("test_op")
        assert isinstance(result, Pending)
        assert result.status == PendingStatus.APPROVED

    def test_fire_field_type_validation(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        with pytest.raises(ValueError, match="expected bool"):
            t.fire("test_op", fields={"passed": "not-a-bool"})

    def test_fire_field_defaults_applied(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        result = t.fire("test_op", review=True)
        assert result.fields["passed"] is False

    def test_fire_unknown_operation_raises(self):
        t = Tract.open()
        with pytest.raises(ValueError, match="Unknown dynamic operation"):
            t.fire("nonexistent")

    def test_fire_provenance_event_written(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        events_written = []

        def my_execute(pending):
            events_written.append(True)
            return "ok"

        t.fire("test_op", fields={"items": [1]}, execute_fn=my_execute)
        assert len(events_written) == 1

    def test_on_accepts_dynamic_operation_name(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)
        # Should not raise
        t.on("test_op", lambda p: p.approve())

    def test_on_rejects_unregistered_dynamic_name(self):
        t = Tract.open()
        with pytest.raises(ValueError, match="not a hookable operation"):
            t.on("unknown_op", lambda p: p.approve())

    def test_dynamic_op_appears_in_as_tools(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        tools = t.as_tools(profile="full")
        tool_names = [
            tool.get("function", {}).get("name") or tool.get("name")
            for tool in tools
        ]
        assert "fire_test_op" in tool_names

    def test_fire_handler_calls_custom_action(self):
        """Handler calls a compiled action method on the Pending."""
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        def handler(p):
            p.check(strict=True)

        t.on("test_op", handler)
        result = t.fire("test_op", fields={"items": ["x"]})
        assert result.status == PendingStatus.APPROVED
        assert result.fields["passed"] is True

    def test_fire_triggered_by(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        result = t.fire("test_op", triggered_by="policy:auto", review=True)
        assert result.triggered_by == "policy:auto"

    def test_fire_rejected_by_handler(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        t.on("test_op", lambda p: p.reject("not allowed"))
        result = t.fire("test_op")
        assert result.status == PendingStatus.REJECTED
        assert result.rejection_reason == "not allowed"

    def test_multiple_operations_coexist(self):
        t = Tract.open()
        t.register_operation(_simple_spec("op_a"))
        t.register_operation(_simple_spec("op_b"))

        a_result = t.fire("op_a", review=True)
        b_result = t.fire("op_b", review=True)
        assert a_result.operation == "op_a"
        assert b_result.operation == "op_b"

    def test_wildcard_handler_catches_dynamic_op(self):
        t = Tract.open()
        spec = _simple_spec()
        t.register_operation(spec)

        caught = []
        t.on("*", lambda p: (caught.append(p.operation), p.approve()))
        t.fire("test_op")
        assert "test_op" in caught
