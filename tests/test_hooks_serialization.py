"""Tests for Pending serialization and agent interface (Phase 3).

Tests to_dict(), to_tools(), describe_api(), apply_decision(), execute_tool()
for both the base Pending class and concrete subclasses (PendingCompress, etc.).
Also tests pprint() smoke test and introspection edge cases.
"""

from __future__ import annotations

import json

import pytest

from tract import Tract
from tract.hooks.compress import PendingCompress
from tract.hooks.gc import PendingGC
from tract.hooks.merge import PendingMerge
from tract.hooks.pending import Pending
from tract.hooks.trigger import PendingTrigger
from tract.hooks.rebase import PendingRebase
from tract.hooks.introspection import (
    method_to_tool_schema,
    pending_to_dict,
    pending_to_tools,
    pending_describe_api,
    _python_type_to_json_schema,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_compressible_tract() -> Tract:
    """Create an in-memory Tract with 3 commits (system + user + assistant)."""
    t = Tract.open(":memory:")
    t.system("You are a helpful assistant.")
    t.user("Hello, how are you?")
    t.assistant("I'm doing well, thank you!")
    return t


def _make_pending_compress() -> tuple[Tract, PendingCompress]:
    """Create a PendingCompress via the compress pipeline."""
    t = _make_compressible_tract()
    pending = t.compress(content="A concise summary.", review=True)
    assert isinstance(pending, PendingCompress)
    return t, pending


def _make_standalone_gc() -> PendingGC:
    """Create a standalone PendingGC for unit testing (no Tract wiring)."""
    t = Tract.open(":memory:")
    return PendingGC(
        operation="gc",
        tract=t,
        commits_to_remove=["abc123", "def456", "ghi789"],
        tokens_to_free=1500,
    )


def _make_standalone_rebase() -> PendingRebase:
    """Create a standalone PendingRebase for unit testing."""
    t = Tract.open(":memory:")
    return PendingRebase(
        operation="rebase",
        tract=t,
        replay_plan=["commit1", "commit2", "commit3"],
        target_base="base_hash",
        warnings=["Warning: divergent history"],
    )


def _make_standalone_merge() -> PendingMerge:
    """Create a standalone PendingMerge for unit testing."""
    t = Tract.open(":memory:")
    return PendingMerge(
        operation="merge",
        tract=t,
        resolutions={"conflict_a": "resolved content A", "conflict_b": "resolved content B"},
        source_branch="feature",
        target_branch="main",
        conflicts=["conflict_a", "conflict_b"],
        guidance="Prefer the feature branch changes.",
        guidance_source="user",
    )


def _make_standalone_trigger() -> PendingTrigger:
    """Create a standalone PendingTrigger for unit testing."""
    t = Tract.open(":memory:")
    return PendingTrigger(
        operation="trigger",
        tract=t,
        trigger_name="auto_compress",
        action_type="compress",
        action_params={"target_tokens": 1000},
        reason="Token count exceeded threshold.",
    )


# ===========================================================================
# 1. to_dict() Structure
# ===========================================================================


class TestToDict:
    """Tests for to_dict() on various Pending subclasses."""

    def test_to_dict_has_required_keys(self):
        """to_dict() returns dict with operation, pending_id, status, fields, available_actions."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert "operation" in d
        assert "pending_id" in d
        assert "status" in d
        assert "fields" in d
        assert "available_actions" in d
        t.close()

    def test_to_dict_operation_matches(self):
        """to_dict()['operation'] matches the pending's operation."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert d["operation"] == "compress"
        t.close()

    def test_to_dict_pending_id_matches(self):
        """to_dict()['pending_id'] matches the pending's pending_id."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert d["pending_id"] == pending.pending_id
        t.close()

    def test_to_dict_status_matches(self):
        """to_dict()['status'] matches the pending's status."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert d["status"] == "pending"
        t.close()

    def test_to_dict_fields_contain_summaries(self):
        """to_dict()['fields'] contains 'summaries' for PendingCompress."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert "summaries" in d["fields"]
        assert isinstance(d["fields"]["summaries"], list)
        t.close()

    def test_to_dict_fields_contain_source_commits(self):
        """to_dict()['fields'] contains 'source_commits' for PendingCompress."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert "source_commits" in d["fields"]
        t.close()

    def test_to_dict_available_actions_sorted(self):
        """to_dict()['available_actions'] is a sorted list of action names."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        actions = d["available_actions"]
        assert isinstance(actions, list)
        assert actions == sorted(actions)
        t.close()

    def test_to_dict_available_actions_match_public_actions(self):
        """to_dict()['available_actions'] matches _public_actions."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert set(d["available_actions"]) == pending._public_actions
        t.close()

    def test_to_dict_excludes_internal_fields(self):
        """to_dict()['fields'] does not contain fields starting with _."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        for key in d["fields"]:
            assert not key.startswith("_"), f"Internal field {key!r} in to_dict() fields"
        t.close()

    def test_to_dict_excludes_metadata_fields(self):
        """to_dict()['fields'] does not contain operation, pending_id, status, tract."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        for meta_key in ("operation", "pending_id", "status", "tract"):
            assert meta_key not in d["fields"], f"Metadata field {meta_key!r} in fields"
        t.close()

    def test_to_dict_round_trip_summaries(self):
        """to_dict()['fields']['summaries'] matches actual pending.summaries."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert d["fields"]["summaries"] == pending.summaries
        t.close()

    def test_to_dict_round_trip_original_tokens(self):
        """to_dict()['fields']['original_tokens'] matches pending.original_tokens."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        assert d["fields"]["original_tokens"] == pending.original_tokens
        t.close()

    def test_to_dict_is_json_serializable(self):
        """to_dict() output can be serialized to JSON."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert isinstance(json_str, str)
        t.close()

    def test_to_dict_gc(self):
        """to_dict() works for PendingGC."""
        gc = _make_standalone_gc()
        d = gc.to_dict()
        assert d["operation"] == "gc"
        assert "commits_to_remove" in d["fields"]
        assert d["fields"]["commits_to_remove"] == ["abc123", "def456", "ghi789"]
        assert d["fields"]["tokens_to_free"] == 1500
        gc.tract.close()

    def test_to_dict_rebase(self):
        """to_dict() works for PendingRebase."""
        rb = _make_standalone_rebase()
        d = rb.to_dict()
        assert d["operation"] == "rebase"
        assert "replay_plan" in d["fields"]
        assert "target_base" in d["fields"]
        assert d["fields"]["target_base"] == "base_hash"
        rb.tract.close()

    def test_to_dict_merge(self):
        """to_dict() works for PendingMerge."""
        m = _make_standalone_merge()
        d = m.to_dict()
        assert d["operation"] == "merge"
        assert "resolutions" in d["fields"]
        assert "source_branch" in d["fields"]
        assert d["fields"]["source_branch"] == "feature"
        m.tract.close()

    def test_to_dict_trigger(self):
        """to_dict() works for PendingTrigger."""
        p = _make_standalone_trigger()
        d = p.to_dict()
        assert d["operation"] == "trigger"
        assert "trigger_name" in d["fields"]
        assert "action_type" in d["fields"]
        assert d["fields"]["trigger_name"] == "auto_compress"
        p.tract.close()


# ===========================================================================
# 2. to_tools() Schema Validity
# ===========================================================================


class TestToTools:
    """Tests for to_tools() JSON Schema tool definitions."""

    def test_to_tools_returns_list(self):
        """to_tools() returns a list."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        assert isinstance(tools, list)
        t.close()

    def test_to_tools_each_has_type_function(self):
        """Each tool definition has 'type': 'function'."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        for tool in tools:
            assert tool["type"] == "function", f"Tool missing type=function: {tool}"
        t.close()

    def test_to_tools_each_has_function_key(self):
        """Each tool definition has a 'function' key."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        for tool in tools:
            assert "function" in tool, f"Tool missing 'function' key: {tool}"
        t.close()

    def test_to_tools_function_has_name(self):
        """Each tool's function has a 'name' key."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        for tool in tools:
            func = tool["function"]
            assert "name" in func, f"Function missing 'name': {func}"
        t.close()

    def test_to_tools_function_has_parameters(self):
        """Each tool's function has a 'parameters' key."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        for tool in tools:
            func = tool["function"]
            assert "parameters" in func, f"Function missing 'parameters': {func}"
        t.close()

    def test_to_tools_function_has_description(self):
        """Each tool's function has a 'description' key."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        for tool in tools:
            func = tool["function"]
            assert "description" in func, f"Function missing 'description': {func}"
            assert len(func["description"]) > 0
        t.close()

    def test_to_tools_includes_all_public_actions(self):
        """to_tools() includes a tool for each method in _public_actions."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert tool_names == pending._public_actions
        t.close()

    def test_to_tools_approve_no_required_params(self):
        """The 'approve' tool has no required parameters."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        approve_tool = next(t for t in tools if t["function"]["name"] == "approve")
        params = approve_tool["function"]["parameters"]
        assert "required" not in params or len(params.get("required", [])) == 0
        t.close()

    def test_to_tools_reject_has_reason_param(self):
        """The 'reject' tool has a 'reason' parameter."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        reject_tool = next(t for t in tools if t["function"]["name"] == "reject")
        params = reject_tool["function"]["parameters"]
        assert "reason" in params["properties"]
        t.close()

    def test_to_tools_edit_summary_has_required_params(self):
        """The 'edit_summary' tool has required 'index' and 'new_text' params."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        edit_tool = next(t for t in tools if t["function"]["name"] == "edit_summary")
        params = edit_tool["function"]["parameters"]
        assert "index" in params["properties"]
        assert "new_text" in params["properties"]
        assert "index" in params.get("required", [])
        assert "new_text" in params.get("required", [])
        t.close()

    def test_to_tools_edit_summary_index_is_integer(self):
        """The 'edit_summary' tool's 'index' param has type 'integer'."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        edit_tool = next(t for t in tools if t["function"]["name"] == "edit_summary")
        index_prop = edit_tool["function"]["parameters"]["properties"]["index"]
        assert index_prop["type"] == "integer"
        t.close()

    def test_to_tools_is_json_serializable(self):
        """to_tools() output can be serialized to JSON."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()
        json_str = json.dumps(tools)
        assert isinstance(json_str, str)
        t.close()

    def test_to_tools_gc(self):
        """to_tools() works for PendingGC and includes exclude tool."""
        gc = _make_standalone_gc()
        tools = gc.to_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert "exclude" in tool_names
        assert "approve" in tool_names
        assert "reject" in tool_names
        gc.tract.close()

    def test_to_tools_gc_exclude_has_commit_hash_param(self):
        """PendingGC.exclude tool has 'commit_hash' parameter."""
        gc = _make_standalone_gc()
        tools = gc.to_tools()
        exclude_tool = next(t for t in tools if t["function"]["name"] == "exclude")
        params = exclude_tool["function"]["parameters"]
        assert "commit_hash" in params["properties"]
        gc.tract.close()

    def test_to_tools_trigger(self):
        """to_tools() works for PendingTrigger and includes modify_params."""
        p = _make_standalone_trigger()
        tools = p.to_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert "modify_params" in tool_names
        p.tract.close()

    def test_to_tools_merge_includes_edit_resolution(self):
        """to_tools() for PendingMerge includes edit_resolution tool."""
        m = _make_standalone_merge()
        tools = m.to_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert "edit_resolution" in tool_names
        m.tract.close()


# ===========================================================================
# 3. describe_api()
# ===========================================================================


class TestDescribeApi:
    """Tests for describe_api() output."""

    def test_describe_api_returns_nonempty_string(self):
        """describe_api() returns a non-empty string."""
        t, pending = _make_pending_compress()
        api_desc = pending.describe_api()
        assert isinstance(api_desc, str)
        assert len(api_desc) > 0
        t.close()

    def test_describe_api_contains_class_name(self):
        """describe_api() contains the class name."""
        t, pending = _make_pending_compress()
        api_desc = pending.describe_api()
        assert "PendingCompress" in api_desc
        t.close()

    def test_describe_api_contains_action_names(self):
        """describe_api() lists all public action names."""
        t, pending = _make_pending_compress()
        api_desc = pending.describe_api()
        for action in pending._public_actions:
            assert action in api_desc, f"Action {action!r} not in describe_api() output"
        t.close()

    def test_describe_api_contains_fields_section(self):
        """describe_api() has a Fields section."""
        t, pending = _make_pending_compress()
        api_desc = pending.describe_api()
        assert "### Fields" in api_desc
        t.close()

    def test_describe_api_contains_actions_section(self):
        """describe_api() has an Actions section."""
        t, pending = _make_pending_compress()
        api_desc = pending.describe_api()
        assert "### Actions" in api_desc
        t.close()

    def test_describe_api_contains_summaries_field(self):
        """describe_api() mentions the 'summaries' field for PendingCompress."""
        t, pending = _make_pending_compress()
        api_desc = pending.describe_api()
        assert "summaries" in api_desc
        t.close()

    def test_describe_api_gc(self):
        """describe_api() works for PendingGC."""
        gc = _make_standalone_gc()
        api_desc = gc.describe_api()
        assert "PendingGC" in api_desc
        assert "commits_to_remove" in api_desc
        assert "exclude" in api_desc
        gc.tract.close()

    def test_describe_api_trigger(self):
        """describe_api() works for PendingTrigger."""
        p = _make_standalone_trigger()
        api_desc = p.describe_api()
        assert "PendingTrigger" in api_desc
        assert "modify_params" in api_desc
        p.tract.close()


# ===========================================================================
# 4. apply_decision() Routing
# ===========================================================================


class TestApplyDecision:
    """Tests for apply_decision() routing through execute_tool()."""

    def test_apply_decision_approve(self):
        """apply_decision({'action': 'approve'}) calls approve()."""
        t, pending = _make_pending_compress()
        pending.apply_decision({"action": "approve"})
        assert pending.status == "approved"
        t.close()

    def test_apply_decision_reject_with_reason(self):
        """apply_decision({'action': 'reject', 'args': {'reason': '...'}}) works."""
        t, pending = _make_pending_compress()
        pending.apply_decision({"action": "reject", "args": {"reason": "bad quality"}})
        assert pending.status == "rejected"
        assert pending.rejection_reason == "bad quality"
        t.close()

    def test_apply_decision_blocks_execute_fn(self):
        """apply_decision({'action': '_execute_fn'}) raises ValueError."""
        t, pending = _make_pending_compress()
        with pytest.raises(ValueError, match="_execute_fn"):
            pending.apply_decision({"action": "_execute_fn"})
        t.close()

    def test_apply_decision_blocks_require_pending(self):
        """apply_decision({'action': '_require_pending'}) raises ValueError."""
        t, pending = _make_pending_compress()
        with pytest.raises(ValueError):
            pending.apply_decision({"action": "_require_pending"})
        t.close()

    def test_apply_decision_blocks_nonexistent_action(self):
        """apply_decision({'action': 'nonexistent'}) raises ValueError."""
        t, pending = _make_pending_compress()
        with pytest.raises(ValueError, match="nonexistent"):
            pending.apply_decision({"action": "nonexistent"})
        t.close()

    def test_apply_decision_edit_summary(self):
        """apply_decision can route to edit_summary."""
        t, pending = _make_pending_compress()
        pending.apply_decision({
            "action": "edit_summary",
            "args": {"index": 0, "new_text": "Updated via decision"},
        })
        assert pending.summaries[0] == "Updated via decision"
        t.close()

    def test_apply_decision_missing_action_key_raises(self):
        """apply_decision({}) raises KeyError for missing 'action'."""
        t, pending = _make_pending_compress()
        with pytest.raises(KeyError):
            pending.apply_decision({})
        t.close()


# ===========================================================================
# 5. execute_tool() Whitelist
# ===========================================================================


class TestExecuteTool:
    """Tests for execute_tool() whitelist enforcement."""

    def test_execute_tool_approve(self):
        """execute_tool('approve', {}) calls approve()."""
        t, pending = _make_pending_compress()
        pending.execute_tool("approve", {})
        assert pending.status == "approved"
        t.close()

    def test_execute_tool_reject(self):
        """execute_tool('reject', {'reason': 'bad'}) works."""
        t, pending = _make_pending_compress()
        pending.execute_tool("reject", {"reason": "bad"})
        assert pending.status == "rejected"
        assert pending.rejection_reason == "bad"
        t.close()

    def test_execute_tool_blocks_private_method(self):
        """execute_tool('_execute_fn', {}) raises ValueError."""
        t, pending = _make_pending_compress()
        with pytest.raises(ValueError, match="Cannot execute private"):
            pending.execute_tool("_execute_fn", {})
        t.close()

    def test_execute_tool_blocks_commit_fn(self):
        """execute_tool('_commit_fn', {}) raises ValueError."""
        t, pending = _make_pending_compress()
        with pytest.raises(ValueError, match="Cannot execute private"):
            pending.execute_tool("_commit_fn", {})
        t.close()

    def test_execute_tool_blocks_unlisted_public(self):
        """execute_tool with a public method not in _public_actions raises ValueError."""
        t, pending = _make_pending_compress()
        # to_dict is a real method but not in _public_actions
        with pytest.raises(ValueError, match="not in the allowed"):
            pending.execute_tool("to_dict", {})
        t.close()

    def test_execute_tool_gc_exclude(self):
        """execute_tool('exclude', {'commit_hash': '...'}) on PendingGC works."""
        gc = _make_standalone_gc()
        gc.execute_tool("exclude", {"commit_hash": "abc123"})
        assert "abc123" not in gc.commits_to_remove
        gc.tract.close()

    def test_execute_tool_trigger_modify_params(self):
        """execute_tool('modify_params', {'params': {...}}) on PendingTrigger works."""
        p = _make_standalone_trigger()
        p.execute_tool("modify_params", {"params": {"target_tokens": 2000}})
        assert p.action_params["target_tokens"] == 2000
        p.tract.close()

    def test_execute_tool_none_args(self):
        """execute_tool('approve') with no args defaults to empty dict."""
        t, pending = _make_pending_compress()
        pending.execute_tool("approve")
        assert pending.status == "approved"
        t.close()


# ===========================================================================
# 6. Dynamic _public_actions
# ===========================================================================


class TestDynamicPublicActions:
    """Tests that adding a method to _public_actions makes it visible."""

    def test_adding_action_appears_in_to_dict(self):
        """Using register_action() makes it appear in to_dict()."""
        gc = _make_standalone_gc()
        # Dynamically add a method via register_action()
        gc.custom_action = lambda: "custom"
        gc.register_action("custom_action")

        d = gc.to_dict()
        assert "custom_action" in d["available_actions"]
        gc.tract.close()

    def test_adding_action_appears_in_to_tools(self):
        """Using register_action() makes it appear in to_tools()."""
        gc = _make_standalone_gc()

        def my_custom_method(text: str) -> None:
            """Do something custom with text."""
            pass

        gc.my_custom_method = my_custom_method
        gc.register_action("my_custom_method")

        tools = gc.to_tools()
        tool_names = {t["function"]["name"] for t in tools}
        assert "my_custom_method" in tool_names
        gc.tract.close()

    def test_adding_action_appears_in_describe_api(self):
        """Using register_action() makes it appear in describe_api()."""
        gc = _make_standalone_gc()
        gc.my_action = lambda: None
        gc.register_action("my_action")

        desc = gc.describe_api()
        assert "my_action" in desc
        gc.tract.close()


# ===========================================================================
# 7. pprint() Smoke Test
# ===========================================================================


class TestPprintSmoke:
    """Smoke tests for pprint() -- it shouldn't crash."""

    def test_pprint_pending_compress(self, capsys):
        """pprint() on PendingCompress doesn't crash."""
        t, pending = _make_pending_compress()
        pending.pprint()
        captured = capsys.readouterr()
        assert "PendingCompress" in captured.out
        t.close()

    def test_pprint_gc(self, capsys):
        """pprint() on PendingGC doesn't crash."""
        gc = _make_standalone_gc()
        gc.pprint()
        captured = capsys.readouterr()
        assert "PendingGC" in captured.out
        gc.tract.close()

    def test_pprint_rebase(self, capsys):
        """pprint() on PendingRebase doesn't crash."""
        rb = _make_standalone_rebase()
        rb.pprint()
        captured = capsys.readouterr()
        assert "PendingRebase" in captured.out
        rb.tract.close()

    def test_pprint_merge(self, capsys):
        """pprint() on PendingMerge doesn't crash."""
        m = _make_standalone_merge()
        m.pprint()
        captured = capsys.readouterr()
        assert "PendingMerge" in captured.out
        m.tract.close()

    def test_pprint_trigger(self, capsys):
        """pprint() on PendingTrigger doesn't crash."""
        p = _make_standalone_trigger()
        p.pprint()
        captured = capsys.readouterr()
        assert "PendingTrigger" in captured.out
        p.tract.close()

    def test_pprint_shows_status(self, capsys):
        """pprint() shows the pending status."""
        t, pending = _make_pending_compress()
        pending.pprint()
        captured = capsys.readouterr()
        assert "pending" in captured.out
        t.close()

    def test_pprint_shows_available_actions(self, capsys):
        """pprint() shows available actions."""
        t, pending = _make_pending_compress()
        pending.pprint()
        captured = capsys.readouterr()
        assert "approve" in captured.out
        assert "reject" in captured.out
        t.close()

    def test_pprint_rejected_shows_reason(self, capsys):
        """pprint() on a rejected pending shows the rejection reason."""
        gc = _make_standalone_gc()
        gc.status = "rejected"
        gc.rejection_reason = "Not needed"
        gc.pprint()
        captured = capsys.readouterr()
        assert "Not needed" in captured.out
        gc.tract.close()


# ===========================================================================
# 8. Introspection Module Utilities
# ===========================================================================


class TestIntrospectionUtils:
    """Tests for low-level introspection utilities."""

    def test_python_type_to_json_schema_str(self):
        """str maps to 'string'."""
        assert _python_type_to_json_schema(str) == {"type": "string"}

    def test_python_type_to_json_schema_int(self):
        """int maps to 'integer'."""
        assert _python_type_to_json_schema(int) == {"type": "integer"}

    def test_python_type_to_json_schema_float(self):
        """float maps to 'number'."""
        assert _python_type_to_json_schema(float) == {"type": "number"}

    def test_python_type_to_json_schema_bool(self):
        """bool maps to 'boolean'."""
        assert _python_type_to_json_schema(bool) == {"type": "boolean"}

    def test_python_type_to_json_schema_dict(self):
        """dict maps to 'object'."""
        assert _python_type_to_json_schema(dict) == {"type": "object"}

    def test_python_type_to_json_schema_list(self):
        """list maps to 'array'."""
        assert _python_type_to_json_schema(list) == {"type": "array"}

    def test_python_type_to_json_schema_unknown_falls_back(self):
        """Unknown types fall back to 'string'."""
        import inspect
        result = _python_type_to_json_schema(inspect.Parameter.empty)
        assert result == {"type": "string"}

    def test_method_to_tool_schema_basic(self):
        """method_to_tool_schema produces valid structure."""

        def my_method(x: int, y: str = "default") -> None:
            """Do something useful."""
            pass

        schema = method_to_tool_schema(my_method, "my_method")
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "my_method"
        assert "Do something useful" in schema["function"]["description"]
        props = schema["function"]["parameters"]["properties"]
        assert "x" in props
        assert props["x"]["type"] == "integer"
        assert "y" in props
        assert props["y"]["type"] == "string"
        assert "x" in schema["function"]["parameters"]["required"]
        assert "y" not in schema["function"]["parameters"].get("required", [])

    def test_method_to_tool_schema_no_type_hints(self):
        """method_to_tool_schema handles methods with no type hints."""

        def untyped_method(a, b):
            """An untyped method."""
            pass

        schema = method_to_tool_schema(untyped_method, "untyped_method")
        assert schema["type"] == "function"
        props = schema["function"]["parameters"]["properties"]
        assert "a" in props
        assert "b" in props

    def test_method_to_tool_schema_no_docstring(self):
        """method_to_tool_schema handles methods with no docstring."""

        def no_doc(x: int) -> None:
            pass

        schema = method_to_tool_schema(no_doc, "no_doc")
        assert schema["type"] == "function"
        # Should have some description (fallback)
        assert "description" in schema["function"]

    def test_method_to_tool_schema_skips_self(self):
        """method_to_tool_schema skips 'self' parameter."""

        class Dummy:
            def my_method(self, x: int) -> None:
                """Test method."""
                pass

        d = Dummy()
        schema = method_to_tool_schema(d.my_method, "my_method")
        props = schema["function"]["parameters"]["properties"]
        assert "self" not in props

    def test_method_to_tool_schema_skips_kwargs(self):
        """method_to_tool_schema skips **kwargs parameter."""

        def with_kwargs(x: int, **kwargs) -> None:
            """Test method."""
            pass

        schema = method_to_tool_schema(with_kwargs, "with_kwargs")
        props = schema["function"]["parameters"]["properties"]
        assert "kwargs" not in props
        assert "x" in props


# ===========================================================================
# 9. PendingCompress End-to-End
# ===========================================================================


class TestPendingCompressEndToEnd:
    """End-to-end tests combining serialization with real compress pipeline."""

    def test_to_dict_then_apply_decision_approve(self):
        """Serialize to dict, inspect, then apply approve decision."""
        t, pending = _make_pending_compress()
        d = pending.to_dict()

        # Verify we can inspect the dict
        assert d["status"] == "pending"
        assert len(d["fields"]["summaries"]) > 0

        # Apply approval
        pending.apply_decision({"action": "approve"})
        assert pending.status == "approved"
        t.close()

    def test_to_tools_then_execute_tool(self):
        """Generate tools, find edit_summary tool, execute it."""
        t, pending = _make_pending_compress()
        tools = pending.to_tools()

        # Find the edit_summary tool
        edit_tool = next(
            t for t in tools if t["function"]["name"] == "edit_summary"
        )
        assert edit_tool is not None

        # Execute it
        pending.execute_tool("edit_summary", {"index": 0, "new_text": "Tool-driven edit"})
        assert pending.summaries[0] == "Tool-driven edit"
        t.close()

    def test_describe_api_then_use_api(self):
        """Generate API description, verify it mentions edit_summary, then use it."""
        t, pending = _make_pending_compress()
        api = pending.describe_api()

        assert "edit_summary" in api
        assert "index" in api

        # Use the method
        pending.edit_summary(0, "Manually edited")
        assert pending.summaries[0] == "Manually edited"
        t.close()

    def test_to_dict_reflects_edits(self):
        """After editing, to_dict() reflects the updated state."""
        t, pending = _make_pending_compress()
        pending.edit_summary(0, "Edited summary")
        d = pending.to_dict()
        assert d["fields"]["summaries"][0] == "Edited summary"
        t.close()

    def test_to_dict_reflects_status_after_reject(self):
        """After rejecting, to_dict() shows status='rejected'."""
        t, pending = _make_pending_compress()
        pending.reject("Not good enough")
        d = pending.to_dict()
        assert d["status"] == "rejected"
        t.close()
