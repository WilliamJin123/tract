# Phase R4: Cookbooks + Validation + Registries

> **Status: SUPERSEDED by Phase 14 (Config + Directives + Middleware).** Rule engine was implemented (R0-R4, commit 7a86b94) then replaced (commit 23a89eb). Kept as historical reference.

## Goal

Rewrite core cookbooks for the new API, validate on cheap models (Cerebras/Groq),
and add the extensibility registries for custom conditions/actions/metrics/triggers.

**Depends on:** R3 (loop, toolkit, and full rule engine working)

## Task Breakdown

### Task 4.1: Extensibility Registries (`rules/registries.py`)

```python
"""Extension registries for custom conditions, actions, metrics, and triggers.

Follows the same pattern as tract's content type registry:
protocol-based registration with per-engine instances.
"""

from __future__ import annotations
from typing import Protocol, Any

from tract.rules.models import EvalContext, ActionResult


class ConditionEvaluator(Protocol):
    def evaluate(self, params: dict, ctx: EvalContext) -> bool: ...

class ActionHandler(Protocol):
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult: ...

class MetricProvider(Protocol):
    def compute(self, ctx: EvalContext) -> float: ...

class TriggerSource(Protocol):
    def check(self, ctx: EvalContext) -> bool: ...


class Registry:
    """Unified registry for extensibility points."""

    def __init__(self) -> None:
        self._conditions: dict[str, ConditionEvaluator] = {}
        self._actions: dict[str, ActionHandler] = {}
        self._metrics: dict[str, MetricProvider] = {}
        self._triggers: dict[str, TriggerSource] = {}

    def register_condition(self, name: str, evaluator: ConditionEvaluator) -> None:
        """Register a custom condition evaluator."""
        self._conditions[name] = evaluator

    def register_action(self, name: str, handler: ActionHandler) -> None:
        """Register a custom action handler."""
        self._actions[name] = handler

    def register_metric(self, name: str, provider: MetricProvider) -> None:
        """Register a custom metric for threshold conditions."""
        self._metrics[name] = provider

    def register_trigger(self, name: str, source: TriggerSource) -> None:
        """Register a custom trigger source."""
        self._triggers[name] = source

    @property
    def conditions(self) -> dict[str, ConditionEvaluator]:
        return dict(self._conditions)

    @property
    def actions(self) -> dict[str, ActionHandler]:
        return dict(self._actions)

    @property
    def metrics(self) -> dict[str, MetricProvider]:
        return dict(self._metrics)

    @property
    def triggers(self) -> dict[str, TriggerSource]:
        return dict(self._triggers)
```

**Wire into Tract:**

```python
# In tract.py
def register_condition(self, name: str, evaluator) -> None:
    """Register a custom condition type for the rule engine."""
    self._registry.register_condition(name, evaluator)

def register_action(self, name: str, handler) -> None:
    """Register a custom action type for the rule engine."""
    self._registry.register_action(name, handler)

def register_metric(self, name: str, provider) -> None:
    """Register a custom metric for threshold conditions."""
    self._registry.register_metric(name, provider)
```

### Task 4.2: LLM Condition/Action Implementation

Now that the loop exists and LLM clients are wired, implement real LLM
evaluation for conditions and actions.

```python
class LLMCondition:
    def evaluate(self, params: dict, ctx: EvalContext) -> bool:
        instruction = params["instruction"]
        llm = ctx.tract._llm_client
        if llm is None:
            return True  # permissive when no LLM

        # Build evaluation prompt
        messages = [
            {"role": "system", "content": (
                "You are evaluating a condition. Respond with exactly "
                "'true' or 'false', nothing else."
            )},
            {"role": "user", "content": instruction},
        ]

        response = llm.chat(messages=messages)
        content = _extract_text(response).strip().lower()
        return content == "true"


class LLMAction:
    def execute(self, params: dict, ctx: EvalContext) -> ActionResult:
        instruction = params["instruction"]
        llm = ctx.tract._llm_client
        if llm is None:
            return ActionResult("llm", False, reason="No LLM client available")

        # Build context with recent history
        compiled = ctx.tract.compile(strategy="adaptive", strategy_k=3)
        messages = compiled.to_dicts()
        messages.append({"role": "user", "content": instruction})

        response = llm.chat(messages=messages)
        content = _extract_text(response)

        return ActionResult("llm", True, {"response": content})
```

### Task 4.3: Rewrite Core Cookbooks

**New cookbook structure:**

```
cookbook/
    getting_started/
        01_basics.py            # commit, compile, chat
        02_rules.py             # create rules, config resolution
        03_agent_loop.py        # run_loop with tools
    developer/
        config/                 # LLMConfig, operation configs (mostly unchanged)
        conversations/          # shorthand, batch, chat (mostly unchanged)
        history/                # log, diff, reset (unchanged)
        metadata/               # tags, priority, MetadataContent (update)
        operations/             # compress, branch, merge (minor updates)
    rules/
        01_basic_rules.py       # create rules, condition types
        02_config_rules.py      # active rules as config
        03_event_rules.py       # commit/compile/compress triggers
        04_transition_rules.py  # stage transitions
        05_compile_strategy.py  # full/messages/adaptive
        06_data_preservation.py # block compress for tagged data
        07_quality_gates.py     # require conditions, approval gates
    workflows/
        01_coding_workflow.py   # design -> implementation -> validation
        02_research_workflow.py # ingest -> organize -> synthesize
        03_ecommerce_workflow.py # product_research -> lander_pages -> ads -> metrics
    integrations/
        langchain.py            # LangChain integration
        agno.py                 # Agno integration
        crewai.py               # CrewAI integration
```

**Key cookbooks to write:**

#### `getting_started/02_rules.py`
```python
"""Rules: Behavior as Data

Rules replace hooks, triggers, policies, and the orchestrator with
one unified model: trigger + condition + action, scoped by DAG placement.
"""
from tract import Tract

with Tract.open() as t:
    # Config rules -- always active, scoped by branch
    t.rule("temperature", trigger="active",
           action={"type": "set_config", "key": "temperature", "value": 0.3})

    # Event rules -- fire on specific operations
    t.rule("auto_compress", trigger="commit",
           condition={"type": "threshold", "metric": "total_tokens",
                      "op": ">", "value": 8000},
           action={"type": "operation", "op": "compress",
                   "params": {"target_tokens": 4000}})

    # Data preservation -- protect tagged data from compression
    t.rule("keep_pricing", trigger="compress",
           condition={"type": "tag", "tag": "pricing", "present": True},
           action={"type": "block"})

    # Resolve config from rules
    print(t.get_config("temperature"))  # 0.3
```

#### `getting_started/03_agent_loop.py`
```python
"""Agent Loop: compile -> LLM -> tools -> repeat

The default loop ships with tract, like the default LLM client.
Easily replaced by LangChain, Agno, CrewAI, etc.
"""
from tract import Tract
from tract.loop import run_loop, LoopConfig

with Tract.open(api_key="...") as t:
    # Set up rules
    t.rule("context_strategy", trigger="active",
           action={"type": "set_config", "key": "compile_strategy",
                   "value": "adaptive", "k": 5})

    # Run the loop
    result = run_loop(t, task="Research the latest trends in AI agents",
                     config=LoopConfig(max_steps=10))

    print(f"Status: {result.status}")
    print(f"Steps: {result.steps}")
    print(f"Tool calls: {result.tool_calls}")
```

#### `workflows/01_coding_workflow.py`
```python
"""Coding Workflow: design -> implementation -> validation

Demonstrates multi-stage workflows using rules for stage configuration,
transition gates, and compile filters.
"""
from tract import Tract
from tract.loop import run_loop, LoopConfig

with Tract.open(api_key="...") as t:
    # -- Design stage --
    t.create_branch("design")
    t.switch("design")

    t.rule("design_temp", trigger="active",
           action={"type": "set_config", "key": "temperature", "value": 0.7})

    t.rule("to_impl", trigger="transition:implementation",
           action={"type": "compile_filter", "mode": "selective",
                   "include_tags": ["design_decision", "api_contract", "spec"]})

    run_loop(t, task="Design a REST API for a todo app",
            config=LoopConfig(max_steps=5))

    # -- Transition --
    t.transition("implementation")

    # -- Implementation stage --
    t.rule("impl_temp", trigger="active",
           action={"type": "set_config", "key": "temperature", "value": 0.2})

    t.rule("to_validation", trigger="transition:validation",
           action={"type": "compile_filter", "mode": "same_context"})

    run_loop(t, task="Implement the TODO API from the design",
            config=LoopConfig(max_steps=10))
```

### Task 4.4: POC Validation on Cheap Models

**Test with Cerebras and Groq free tiers:**

1. Configure `_providers.py` for Cerebras (gpt-oss-120b) and Groq (kimi-instruct):
```python
# Cerebras
client = OpenAIClient(
    api_key=os.environ["CEREBRAS_API_KEY"],
    base_url="https://api.cerebras.ai/v1",
    model="llama3.1-70b",
)

# Groq
client = OpenAIClient(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
)
```

2. Run each core cookbook on both providers
3. Validate: does the agent follow rules? Do transitions work? Does the loop
   terminate cleanly?

**Success criteria for validation:**
- Agent can create and follow rules (config resolved correctly)
- Multi-stage workflow: transitions happen at the right time
- Compile strategy: agent can switch between full/adaptive/messages
- Quality gates: blocks fire when conditions are met
- The architecture is doing real work, not the model

### Task 4.5: Clean Up __init__.py

Final cleanup of the public API surface:

```python
# New exports
from tract.rules import RuleEngine, RuleIndex, EvalContext, RuleEntry, ActionResult, EvalResult
from tract.rules.conditions import evaluate_condition, BUILTIN_CONDITIONS
from tract.rules.actions import BUILTIN_ACTIONS
from tract.rules.registries import Registry
from tract.loop import run_loop, LoopConfig, LoopResult
from tract.models.content import RuleContent, MetadataContent
```

Update `__all__` with new symbols, verify no dead imports remain.

---

## Test Plan

### `tests/test_registries.py` (~15 tests)

- `test_register_condition` -- custom condition callable
- `test_register_action` -- custom action handler
- `test_register_metric` -- custom metric provider
- `test_register_trigger` -- custom trigger source
- `test_custom_condition_in_rule` -- rule with custom condition evaluates
- `test_custom_action_in_rule` -- rule with custom action executes
- `test_custom_metric_in_threshold` -- threshold uses custom metric
- `test_registry_properties` -- conditions, actions, metrics, triggers
- `test_registry_through_facade` -- t.register_condition(...)
- `test_registry_isolation` -- per-engine instances

### `tests/test_llm_conditions.py` (~8 tests, require mock LLM)

- `test_llm_condition_true` -- mock returns "true"
- `test_llm_condition_false` -- mock returns "false"
- `test_llm_condition_no_client` -- permissive default
- `test_llm_action_executes` -- mock LLM returns response
- `test_llm_action_no_client` -- returns failure
- `test_llm_condition_in_rule` -- end-to-end with rule engine
- `test_llm_condition_sorted_last` -- evaluated after deterministic
- `test_short_circuit_skips_llm` -- deterministic gate blocks, LLM skipped

### Cookbook validation (manual, not pytest)

Run each cookbook file and verify output:
- `getting_started/01_basics.py` -- basic commit/compile
- `getting_started/02_rules.py` -- rule creation and config
- `getting_started/03_agent_loop.py` -- full loop (needs API key)
- `rules/01_basic_rules.py` through `rules/07_quality_gates.py`
- `workflows/01_coding_workflow.py` (needs API key)

---

## Acceptance Criteria

1. Custom conditions, actions, metrics, and triggers register and execute
2. LLM conditions evaluate via the configured LLM client
3. At least 10 working cookbooks demonstrating the new API
4. POC validation: at least one workflow cookbook runs successfully on Cerebras or Groq
5. `__init__.py` exports clean, no orphaned imports
6. Full test suite passes (`python -m pytest tests/`)
7. All ~23 new tests pass
8. Clean `git status` -- no leftover dead files

## End State

After R4, the reconceptualization is complete (except the promotion loop, deferred to R5):
- Substrate operations untouched and fully tested
- Rule engine: create rules, evaluate conditions, process events, resolve configs
- Default loop: compile -> LLM -> tools -> repeat
- Compile strategy: full/messages/adaptive(k)
- RuleContent and MetadataContent as first-class content types
- Extensible: custom conditions, actions, metrics, triggers
- Validated on cheap models
- Working cookbooks demonstrating all patterns
