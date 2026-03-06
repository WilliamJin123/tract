"""Action handlers for the rule engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from tract.llm.protocols import LLMClient
    from tract.rules.models import ActionResult, EvalContext


class ActionHandler(Protocol):
    """Protocol for action type handlers."""

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult: ...


class SetConfigAction:
    """Set a configuration parameter. Override semantics.

    Returns config data in ActionResult for the caller to apply. The engine
    deduplicates set_config results by key (override semantics: closest to
    HEAD wins). The caller (_fire_rules in tract.py) extracts the deduped
    set_config results from EvalResult and applies them.
    """

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.models import ActionResult

        return ActionResult(
            action_type="set_config",
            success=True,
            data={"key": params["key"], "value": params["value"]},
        )


class OperationAction:
    """Run a substrate operation. Accumulate semantics."""

    SUPPORTED_OPS = {
        "compress",
        "branch",
        "annotate",
        "edit",
        "merge",
        "rebase",
        "delete",
        "cherry_pick",
        "gc",
    }

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.models import ActionResult

        op = params["op"]
        op_params = params.get("params", {})
        t = ctx.tract

        if op not in self.SUPPORTED_OPS:
            raise ValueError(
                f"Unknown operation: {op!r}. Supported: {sorted(self.SUPPORTED_OPS)}"
            )

        if op == "compress":
            result = t.compress(**op_params)
            return ActionResult("operation", True, {"op": "compress", "result": str(result)})
        elif op == "branch":
            result = t.list_branches()
            # create_branch doesn't exist as a method, use the operations module
            from tract.operations.branch import create_branch

            branch_name = op_params.get("name", op_params.get("branch_name", ""))
            create_branch(
                branch_name,
                t._tract_id,
                t._ref_repo,
                t._commit_repo,
            )
            t._session.commit()
            return ActionResult("operation", True, {"op": "branch", "result": branch_name})
        elif op == "annotate":
            t.annotate(**op_params)
            return ActionResult("operation", True, {"op": "annotate"})
        elif op == "edit":
            content = op_params.pop("content", None)
            if content is None:
                return ActionResult("operation", False, reason="edit requires 'content' param")
            from tract.models.commit import CommitOperation

            result = t.commit(content, operation=CommitOperation.EDIT, **op_params)
            return ActionResult("operation", True, {"op": "edit", "result": str(result)})
        elif op == "merge":
            result = t.merge(**op_params)
            return ActionResult("operation", True, {"op": "merge", "result": str(result)})
        elif op == "rebase":
            result = t.rebase(**op_params)
            return ActionResult("operation", True, {"op": "rebase", "result": str(result)})
        elif op == "delete":
            # delete_commit is not a direct method; skip for now
            return ActionResult("operation", False, reason="delete not implemented yet")
        elif op == "cherry_pick":
            result = t.import_commit(**op_params)
            return ActionResult("operation", True, {"op": "cherry_pick", "result": str(result)})
        elif op == "gc":
            result = t.gc(**op_params)
            return ActionResult("operation", True, {"op": "gc", "result": str(result)})

        return ActionResult("operation", False, reason=f"Unhandled op: {op}")


class BlockAction:
    """Prevent the triggering operation/transition. Accumulate semantics."""

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.models import ActionResult

        reason = params.get("reason", "Blocked by rule")
        return ActionResult("block", True, {"blocked": True}, reason=reason)


class RequireAction:
    """Block until embedded condition is met. Accumulate semantics."""

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.conditions import evaluate_condition
        from tract.rules.models import ActionResult

        inner_condition = params.get("condition")
        met = evaluate_condition(inner_condition, ctx)
        if met:
            return ActionResult("require", True, {"met": True})
        return ActionResult(
            "require",
            False,
            {"met": False},
            reason=f"Requirement not met: {inner_condition}",
        )


class CompileFilterAction:
    """Configure transition handoff compilation. Override semantics."""

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.models import ActionResult

        return ActionResult("compile_filter", True, data=params)


class LLMAction:
    """LLM-evaluated action. Accumulate semantics.

    Sends instruction + recent context to the configured LLM and returns
    the response. Falls back to failure when no LLM client is available.
    """

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.models import ActionResult

        instruction = params.get("instruction", "")
        llm = getattr(ctx.tract, "_llm_client", None)
        if llm is None:
            return ActionResult("llm", False, reason="No LLM client available")

        try:
            compiled = ctx.tract.compile(strategy="adaptive", strategy_k=3)
            messages = compiled.to_dicts()
            messages.append({"role": "user", "content": instruction})

            response = llm.chat(messages=messages)
            content = _extract_llm_text(response, llm)

            return ActionResult("llm", True, {"response": content})
        except Exception as exc:
            return ActionResult("llm", False, reason=f"LLM action failed: {exc}")


class CreateRuleAction:
    """Commit a new RuleContent. Accumulate semantics.

    Does NOT commit during event processing. Returns the template in
    ActionResult.data for deferred commitment after the pipeline completes.
    """

    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        from tract.rules.models import ActionResult

        template = params.get("template", {})
        if not all(k in template for k in ("name", "trigger", "action")):
            return ActionResult(
                "create_rule",
                False,
                reason="Template missing required fields: name, trigger, action",
            )
        return ActionResult("create_rule", True, {"template": template, "deferred": True})


# Registry
BUILTIN_ACTIONS: dict[str, ActionHandler] = {
    "set_config": SetConfigAction(),
    "operation": OperationAction(),
    "block": BlockAction(),
    "require": RequireAction(),
    "compile_filter": CompileFilterAction(),
    "llm": LLMAction(),
    "create_rule": CreateRuleAction(),
}

# Semantics metadata
ACTION_SEMANTICS: dict[str, str] = {
    "set_config": "override",
    "compile_filter": "override",
    "block": "accumulate",
    "require": "accumulate",
    "llm": "accumulate",
    "operation": "accumulate",
    "create_rule": "accumulate",
}

# Category ordering
ACTION_CATEGORIES: dict[str, str] = {
    "require": "gate",
    "block": "gate",
    "llm": "work",
    "operation": "work",
    "compile_filter": "handoff",
    "set_config": "work",
    "create_rule": "post",
}


def _extract_llm_text(response: object, client: LLMClient | None = None) -> str:
    """Extract text content from an LLM response (provider-agnostic)."""
    if client is not None and hasattr(client, "extract_content"):
        try:
            return client.extract_content(response)
        except (ValueError, KeyError, TypeError):
            pass
    if isinstance(response, dict):
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            return str(response.get("content", ""))
    if isinstance(response, str):
        return response
    return str(response)
