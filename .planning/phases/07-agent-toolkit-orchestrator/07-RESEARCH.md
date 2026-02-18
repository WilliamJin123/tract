# Phase 7: Agent Toolkit & Orchestrator - Research

**Researched:** 2026-02-18
**Domain:** Agent toolkit (tool schema generation), orchestrator (LLM agent loop with policy integration), callback-based proposal flow
**Confidence:** HIGH

## Summary

Phase 7 adds two complementary layers to the existing Tract codebase: (1) a **toolkit** that exposes Tract operations as tool schemas consumable by any LLM agent, and (2) a **built-in orchestrator** that wires those tools into an LLM-driven agent loop with policy integration. Research focused on the existing Tract API surface (which operations become tools), the standard tool schema formats used by OpenAI and Anthropic APIs (since the built-in LLM client is OpenAI-compatible), the agent loop architecture, and how to integrate with the Phase 6 policy engine for triggers and constraints.

The codebase already has every building block needed. The `Tract` class exposes ~20 public methods that map directly to tools (commit, compile, annotate, branch, merge, compress, etc.). The OpenAI-compatible `OpenAIClient` already supports `**kwargs` passthrough, meaning `tools=` can be passed to the chat completion call today. The policy engine already has triggers ("compile", "commit"), autonomy levels ("autonomous", "collaborative", "supervised"), and a proposal flow with callbacks (`on_proposal`). The orchestrator is therefore a composition layer: it generates tool schemas from Tract methods, sends them to the LLM along with context assessment prompts, and dispatches the resulting tool calls back through the Tract public API -- all while respecting policy constraints.

The key architectural insight is the **toolkit-first** design: `Tract.as_tools()` generates tool definitions as plain dicts (JSON-Schema-style), the orchestrator consumes them via the same interface any external agent would, and the orchestrator dogfoods the toolkit (no special internal APIs). This means the toolkit is independently useful even without the orchestrator.

**Primary recommendation:** Use plain dict tool definitions following the OpenAI function calling format (`{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}`), since the built-in LLM client is OpenAI-compatible. Profiles are dataclasses that curate tool subsets and provide profile-specific descriptions. The orchestrator is a synchronous loop that calls the LLM with tools, executes tool calls, and repeats until the LLM stops calling tools -- with policy checks and proposal callbacks at each step.

## Standard Stack

No new external dependencies required. Phase 7 uses only what's already in the project.

### Core (existing dependencies, no additions)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| sqlalchemy | >=2.0.46,<2.2 | No new tables needed for toolkit/orchestrator | Already used for all storage |
| pydantic | >=2.10,<3.0 | Possible use for ToolProfile config models | Already used for all models |
| httpx | >=0.27,<1.0 | LLM client already supports tool calling via kwargs | Already the LLM transport |
| tenacity | >=8.2,<10 | Retry for LLM calls during orchestrator loop | Already used in OpenAIClient |
| tiktoken | >=0.12.0 | Token counting for context assessment | Already used for compilation |

### Supporting (no new additions)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| threading (stdlib) | N/A | `threading.Event` for stop/pause signals | Orchestrator lifecycle control |
| dataclasses (stdlib) | N/A | ToolDefinition, OrchestratorConfig, Proposal, StepResult | Result types and config |
| enum (stdlib) | N/A | AutonomyLevel, OrchestratorState enums | Autonomy ceiling, state tracking |
| logging (stdlib) | N/A | Orchestrator step logging | Same pattern as PolicyEvaluator |
| uuid (stdlib) | N/A | Proposal IDs, step IDs | Same pattern as policy proposals |
| json (stdlib) | N/A | Tool call argument parsing | LLM returns JSON in tool calls |
| copy (stdlib) | N/A | Deep copy of tool definitions for overrides | Profile customization safety |
| inspect (stdlib) | N/A | Could extract method signatures for schema gen | Tool schema auto-generation |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Plain dict tool schemas | Pydantic models for schemas | Plain dicts are what the API expects; Pydantic adds validation but also complexity. Dicts are simpler and match the OpenAI `tools=` parameter directly. |
| Synchronous orchestrator loop | asyncio event loop | Tract is explicitly not async ("Not thread-safe in v1. Each thread should open its own Tract."). Sync loop is consistent. Async is a future enhancement. |
| threading.Event for stop/pause | asyncio.Event | Sync codebase; threading.Event is the stdlib solution for sync code |
| Manual tool schema definitions | Auto-generate from method signatures via inspect | Auto-generation produces poor descriptions. Manual definitions with profile-specific descriptions are what CONTEXT.md requires. Hybrid approach: define schemas manually but validate against actual method signatures. |

**Installation:**
```bash
# No new dependencies -- everything is already in pyproject.toml
```

## Architecture Patterns

### Recommended Project Structure
```
src/tract/
  toolkit/
    __init__.py           # Public exports: ToolDefinition, ToolProfile, as_tools()
    definitions.py        # Per-operation tool definitions (name, description, parameters)
    profiles.py           # Built-in profiles: SelfProfile, SupervisorProfile
    executor.py           # ToolExecutor: dispatches tool calls to Tract methods
  orchestrator/
    __init__.py           # Public exports: Orchestrator, OrchestratorConfig
    config.py             # OrchestratorConfig, AutonomyLevel, TriggerConfig
    loop.py               # Core agent loop: assess -> propose -> execute
    assessment.py         # Context health assessment prompts and parsing
    proposals.py          # Proposal, ProposalResponse, built-in callbacks
  prompts/
    orchestrator.py       # System prompts for context assessment and orchestration
```

### Pattern 1: Toolkit -- Tool Definitions as Plain Dicts
**What:** Each Tract operation gets a `ToolDefinition` (frozen dataclass) containing the OpenAI-format tool schema dict plus an executor function.
**When to use:** Always -- this is the core toolkit format.
**Why plain dicts:** The OpenAI chat completions API expects `tools=[{"type": "function", "function": {...}}]`. Anthropic uses a similar format (`{"name": ..., "description": ..., "input_schema": {...}}`). Plain dicts are the lowest common denominator and can be trivially adapted to any provider format.

```python
# Source: Follows OpenAI function calling format (verified via official docs)

@dataclass(frozen=True)
class ToolDefinition:
    """A single tool definition for LLM consumption."""

    name: str
    description: str
    parameters: dict  # JSON Schema for parameters
    handler: Callable[..., object]  # The actual function to call

    def to_openai(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic(self) -> dict:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
```

### Pattern 2: Profiles -- Curated Tool Subsets with Descriptions
**What:** A profile selects which tools are available and provides scenario-appropriate descriptions.
**When to use:** `Tract.as_tools(profile="self")` or `Tract.as_tools(profile="supervisor")`.
**Key insight:** Profiles are NOT about different tools -- they're about different *descriptions* and *subsets* of the same tools. A "self" profile describes `compress` as "Compress your own context when it gets too large" while a "supervisor" profile describes it as "Compress the managed agent's context to free up token budget."

```python
# Source: CONTEXT.md decisions -- profiles curate which tools and descriptions

@dataclass
class ToolProfile:
    """A named set of tool configurations.

    Each profile specifies which tools to include and provides
    scenario-appropriate descriptions for each tool.
    """

    name: str
    tool_configs: dict[str, ToolConfig]  # tool_name -> config

    def filter_tools(
        self, all_tools: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        """Return tools filtered and customized for this profile."""
        result = []
        for tool in all_tools:
            if tool.name in self.tool_configs:
                config = self.tool_configs[tool.name]
                if config.enabled:
                    # Override description if profile provides one
                    desc = config.description or tool.description
                    result.append(replace(tool, description=desc))
        return result


@dataclass(frozen=True)
class ToolConfig:
    """Per-tool configuration within a profile."""

    enabled: bool = True
    description: str | None = None  # Override default description
```

### Pattern 3: ToolExecutor -- Dispatch Tool Calls to Tract Methods
**What:** A class that maps tool names to Tract method calls and handles argument parsing/validation.
**When to use:** Both the orchestrator and external agents call tools through this executor.
**Key insight:** The executor is what makes the toolkit *usable*. Without it, you'd have tool schemas but no way to execute them. The executor is the bridge between LLM tool call JSON and actual Tract method invocations.

```python
# Source: Follows PolicyEvaluator._dispatch_action pattern

class ToolExecutor:
    """Executes tool calls against a Tract instance.

    Maps tool names to Tract methods and handles argument
    parsing, validation, and result formatting.
    """

    def __init__(self, tract: Tract) -> None:
        self._tract = tract
        self._handlers: dict[str, Callable] = self._build_handler_map()

    def execute(self, tool_name: str, arguments: dict) -> ToolResult:
        """Execute a tool call and return the result.

        Args:
            tool_name: The name of the tool to execute.
            arguments: The arguments dict from the LLM's tool call.

        Returns:
            ToolResult with success/error status and output.
        """
        handler = self._handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"Unknown tool: {tool_name}",
            )
        try:
            result = handler(**arguments)
            return ToolResult(
                tool_name=tool_name,
                success=True,
                output=self._format_result(result),
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )
```

### Pattern 4: Orchestrator Agent Loop
**What:** A synchronous loop that: (1) assesses context health, (2) sends tools + assessment to LLM, (3) executes resulting tool calls, (4) repeats until LLM stops calling tools or policy constraints halt execution.
**When to use:** `orchestrator.run()` or triggered by policy evaluation.
**Key insight from CONTEXT.md:** "The orchestrator is literally what you'd build if you wired the toolkit into an agent yourself." This means the loop should be transparent, customizable at every step, and NOT opaque.

```python
# Source: Standard LLM tool-calling agent loop pattern
# (verified via OpenAI agents SDK docs, Anthropic tool use docs)

class Orchestrator:
    """Built-in context management orchestrator.

    A transparent, customizable agent loop that uses Tract tools
    to manage context health. Policy-integrated: policies trigger
    it, it reasons via LLM, policies constrain it.
    """

    def __init__(
        self,
        tract: Tract,
        config: OrchestratorConfig | None = None,
        llm_callable: Callable | None = None,
    ) -> None:
        self._tract = tract
        self._config = config or OrchestratorConfig()
        self._executor = ToolExecutor(tract)
        self._llm = llm_callable  # or use tract's built-in LLM client
        self._state = OrchestratorState.IDLE
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()

    def run(self) -> OrchestratorResult:
        """Run the orchestrator loop once.

        Steps:
        1. Build context assessment (token counts, recent commits, etc.)
        2. Send assessment + tools to LLM
        3. Parse LLM response for tool calls
        4. For each tool call:
           a. Check against policy constraints
           b. In collaborative mode: create proposal, await callback
           c. In autonomous mode: execute directly
        5. Send tool results back to LLM
        6. Repeat until LLM returns no tool calls or stop signal
        """
        ...

    def stop(self) -> None:
        """Immediate halt -- abort in-flight actions."""
        self._stop_event.set()
        self._state = OrchestratorState.STOPPED

    def pause(self) -> None:
        """Graceful wind-down -- finish current step, then stop."""
        self._pause_event.set()
        self._state = OrchestratorState.PAUSING
```

### Pattern 5: Proposal Flow -- Callback-Based Review
**What:** In collaborative mode, each tool call becomes a Proposal sent to a callback function. The callback returns approve/reject/modify.
**When to use:** Default mode (collaborative). The orchestrator proposes, human approves.
**Key insight from CONTEXT.md:** "Callback-based: User provides a callback function that receives proposals and returns approve/reject/modify." This follows the existing `on_proposal` pattern from PolicyEvaluator.

```python
# Source: Follows PolicyProposal pattern from models/policy.py

@dataclass
class OrchestratorProposal:
    """A proposed action from the orchestrator awaiting review.

    Contains the recommended action, LLM reasoning, and alternatives.
    """

    proposal_id: str
    recommended_action: ToolCall  # What the LLM wants to do
    reasoning: str  # Why the LLM recommends this
    alternatives: list[ToolCall]  # Other actions considered
    context_summary: str  # Brief context state description

    # Reviewer response
    decision: str = "pending"  # "approved", "rejected", "modified"
    modified_action: ToolCall | None = None  # If reviewer modifies


# Built-in callbacks
def auto_approve(proposal: OrchestratorProposal) -> ProposalResponse:
    """Approve all proposals automatically (autonomous mode)."""
    return ProposalResponse(decision="approved")


def cli_prompt(proposal: OrchestratorProposal) -> ProposalResponse:
    """Interactive CLI prompt for proposal review."""
    # Uses Rich for display (already a dependency)
    ...


def log_and_approve(proposal: OrchestratorProposal) -> ProposalResponse:
    """Log the proposal and auto-approve (audit trail mode)."""
    logger.info("Proposal: %s -- %s", proposal.recommended_action, proposal.reasoning)
    return ProposalResponse(decision="approved")
```

### Pattern 6: Policy Integration -- Policies Trigger, Orchestrator Reasons, Policies Constrain
**What:** The orchestrator is invoked by policies (new trigger type: "orchestrator") and constrained by the autonomy ceiling.
**When to use:** When a policy fires and needs LLM reasoning to decide *how* to act.
**Key insight from CONTEXT.md:** "Policies define 'when' and 'what' -- orchestrator handles 'how' via LLM reasoning."

```python
# The integration between policies and orchestrator:
#
# 1. Policy evaluates: "token usage at 85%, should compress"
# 2. Policy returns PolicyAction with action_type="orchestrate"
#    (or a new action type that invokes the orchestrator)
# 3. Orchestrator receives the trigger, builds context assessment
# 4. LLM reasons about HOW to compress (which commits, target tokens, etc.)
# 5. Orchestrator proposes/executes via the normal tool call flow
# 6. Autonomy ceiling constrains: if ceiling=collaborative,
#    even autonomous policy actions go through proposal review

# Autonomy ceiling check:
def _check_autonomy(self, action: ToolCall) -> str:
    """Determine effective autonomy for an action.

    Returns the lower of the orchestrator ceiling and the policy's autonomy.
    """
    policy_autonomy = action.metadata.get("autonomy", "collaborative")
    ceiling = self._config.autonomy_ceiling

    # Hierarchy: manual < collaborative < autonomous
    levels = {"manual": 0, "collaborative": 1, "autonomous": 2}
    effective = min(levels[ceiling], levels[policy_autonomy])
    return {v: k for k, v in levels.items()}[effective]
```

### Pattern 7: Context Health Assessment -- LLM Holistic Judgment
**What:** Instead of numeric scoring, the orchestrator sends the compiled context to the LLM and asks for qualitative assessment.
**When to use:** At the start of each orchestrator run.
**Key insight from CONTEXT.md:** "No arbitrary heuristic scores. Token pressure is still math. Everything else is LLM-assessed holistically."

```python
# The assessment prompt includes:
# 1. Token pressure (quantitative): current_tokens / max_tokens
# 2. Recent commit history (what's been happening)
# 3. User-provided task context (if available)
# 4. Available tools (what the orchestrator can do)
#
# The LLM responds with tool calls to act on its assessment.
# No intermediate scoring layer -- reasoning goes directly to actions.

ORCHESTRATOR_SYSTEM_PROMPT = """You are a context management assistant for an LLM conversation.
Your job is to review the current context state and take actions to maintain context health.

Context state:
- Token usage: {token_count}/{max_tokens} ({pct:.0f}%)
- Commits: {commit_count} total
- Current branch: {branch_name}
- Recent activity: {recent_summary}

{task_context}

Review the context and determine if any maintenance actions are needed.
If the context is healthy, respond with a brief assessment and no tool calls.
If action is needed, use the available tools to improve context health.

Guidelines:
- Compress when token pressure is high (>80% of budget)
- Pin important context (system prompts, key decisions, constraints)
- Branch when the conversation has diverged into a tangent
- Prioritize actions by impact: compression > pinning > branching
- Explain your reasoning before taking action
"""
```

### Anti-Patterns to Avoid
- **Framework lock-in:** Tool schemas must NOT require a specific LLM framework. They're plain dicts that any framework can consume.
- **Opaque orchestrator:** The loop must be transparent. Users should be able to inspect/customize every step (what to review, when to review, instructions for how to act).
- **Numeric scoring:** Do NOT build a relevance/coherence scoring system. Token pressure is the only numeric metric. Everything else is LLM judgment.
- **Special internal APIs:** The orchestrator must use the same tool execution path as any external agent. No backdoor Tract._internal_compress() methods.
- **Async without need:** Tract is synchronous. The orchestrator should be synchronous too. Async is a future enhancement.
- **Modifying existing operations:** Phase 7 must NOT change compress(), annotate(), branch(), or any existing method. It composes on top of them.
- **Heavy prompt engineering in code:** System prompts should be in a dedicated `prompts/orchestrator.py` module, not embedded in loop logic. Users can override them.

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Tool call argument parsing | Custom JSON parser | `json.loads()` + dict unpacking | LLM returns valid JSON in tool calls |
| Token pressure calculation | Custom token counter | `tract.status().token_count / tract.config.token_budget.max_tokens` | Already computed, cached |
| Context compilation for assessment | Custom chain walker | `tract.compile()` | Full compile pipeline with cache |
| Operation dispatch | Giant if/elif chain | Dict mapping tool_name -> Tract method | Same pattern as PolicyEvaluator._dispatch_action |
| Proposal lifecycle | Custom state machine | Extend PolicyProposal pattern (status field + callbacks) | Phase 6 already built this |
| Stop/pause signals | Custom flag checking | `threading.Event` with `Event.wait(timeout)` | Standard Python pattern for graceful shutdown |
| Tool schema validation | Custom validator | JSON Schema validation or just let the LLM/API handle it | The API validates tool calls; let it |
| Retry on LLM failure | Custom retry logic | `tenacity` (already a dependency) | Built-in to OpenAIClient |
| Audit logging | Custom logging | PolicyEvaluator audit log pattern (PolicyLogRow) | Extend existing audit infrastructure |
| Config persistence | Custom file I/O | `_trace_meta` table (key-value) | Already used for policy config |

**Key insight:** Phase 7 is ~70% orchestration code wiring existing components together, ~20% tool definition boilerplate, and ~10% new logic (assessment prompts, proposal flow). Almost nothing is built from scratch.

## Common Pitfalls

### Pitfall 1: Tool Schema Description Quality
**What goes wrong:** LLM makes poor tool call decisions because descriptions are vague or misleading.
**Why it happens:** Auto-generated descriptions from docstrings are too technical ("Create a new commit with operation APPEND") instead of scenario-appropriate ("Record a new piece of context in the conversation history").
**How to avoid:** Write profile-specific descriptions manually. "Self" profile: describe tools from the agent's perspective. "Supervisor" profile: describe from the manager's perspective. Test descriptions by sending them to the LLM and seeing if it uses tools correctly.
**Warning signs:** LLM calls wrong tools, passes wrong arguments, or ignores available tools.

### Pitfall 2: Infinite Orchestrator Loop
**What goes wrong:** Orchestrator calls tool -> tool changes context -> LLM wants to call more tools -> never terminates.
**Why it happens:** No max steps limit or convergence check.
**How to avoid:** Add `max_steps` to OrchestratorConfig (default: 10). After each step, check if the action actually changed anything meaningful. If the LLM is repeating the same tool call, stop. Use `_stop_event` and `_pause_event` as circuit breakers.
**Warning signs:** Orchestrator runs for minutes, burns LLM tokens, or produces repeated compress/annotate cycles.

### Pitfall 3: Policy-Orchestrator Recursion
**What goes wrong:** Policy triggers orchestrator -> orchestrator calls compress() -> compress triggers compile() -> compile triggers policy evaluation -> policy triggers orchestrator again.
**Why it happens:** Same recursion issue as Phase 6 Pitfall 1, but now with the orchestrator in the chain.
**How to avoid:** Use the existing `_evaluating` guard on PolicyEvaluator. Additionally, the orchestrator should set a `_orchestrating` flag on Tract (or use the existing `_in_batch` pattern) that prevents policy re-evaluation during orchestrator execution.
**Warning signs:** Stack overflow, infinite loops, rapid token consumption.

### Pitfall 4: Tool Call Argument Mismatch
**What goes wrong:** LLM provides arguments that don't match Tract method signatures (wrong types, missing required params, extra params).
**Why it happens:** Tool schemas don't exactly match method signatures, or LLM hallucinates arguments.
**How to avoid:** Validate arguments before calling Tract methods. Use try/except around every tool execution. Return clear error messages that help the LLM correct its call. Consider using `strict: true` in OpenAI tool definitions for guaranteed schema compliance.
**Warning signs:** TypeError exceptions during tool execution, LLM retrying the same broken call.

### Pitfall 5: Stop/Pause Data Loss
**What goes wrong:** User calls `stop()` mid-operation, partially committed state is left in the database.
**Why it happens:** Stop signal arrives between a Tract method call and its session commit.
**How to avoid:** Check stop/pause signals BETWEEN tool calls, not during them. Each tool call is atomic (Tract methods call `_session.commit()` at the end). The orchestrator should: (1) check signal before each tool call, (2) execute tool call atomically, (3) record result, (4) check signal again before next tool call. Never interrupt a Tract method mid-execution.
**Warning signs:** Inconsistent database state after stop(), orphaned partial operations.

### Pitfall 6: Autonomy Ceiling vs Policy Autonomy Confusion
**What goes wrong:** A policy marked "autonomous" executes without review even though the global ceiling is "collaborative."
**Why it happens:** Ceiling enforcement is missing or applied at the wrong layer.
**How to avoid:** Enforce ceiling in the orchestrator at the point where it decides to execute vs. propose. `effective_autonomy = min(ceiling, policy_autonomy)`. Always default to collaborative. Test: set ceiling=collaborative, configure autonomous policy, verify proposal is created.
**Warning signs:** Actions executing without user approval when ceiling is collaborative.

### Pitfall 7: Large Context in Assessment Prompt
**What goes wrong:** Sending the full compiled context to the LLM for assessment blows the orchestrator's own context budget.
**Why it happens:** Context can be thousands of tokens; sending it all in the assessment prompt is wasteful.
**How to avoid:** Send a summary/metadata view, not the full compiled context. The assessment prompt should include: token count, commit count, recent N commit summaries (not full content), branch info. Only send full content if the LLM requests it via a "read_context" tool.
**Warning signs:** Orchestrator LLM calls timing out, excessive token usage for orchestration itself.

## Code Examples

### Example 1: Tract.as_tools() -- Toolkit Entry Point
```python
# Source: Follows existing Tract facade pattern (public method on Tract class)

def as_tools(
    self,
    *,
    profile: str | ToolProfile = "self",
    overrides: dict[str, str] | None = None,
    format: str = "openai",
) -> list[dict]:
    """Get tool definitions for this Tract instance.

    Args:
        profile: Built-in profile name or custom ToolProfile.
            "self" = tools for an agent managing its own context.
            "supervisor" = tools for managing another agent's context.
        overrides: Dict of tool_name -> description to override
            profile descriptions. Applied on top of the profile.
        format: Output format. "openai" (default) or "anthropic".

    Returns:
        List of tool definition dicts ready for LLM API consumption.
    """
    from tract.toolkit import get_all_tools, get_profile

    # Get all tool definitions bound to this Tract instance
    all_tools = get_all_tools(self)

    # Apply profile filtering and descriptions
    if isinstance(profile, str):
        profile_obj = get_profile(profile)
    else:
        profile_obj = profile
    tools = profile_obj.filter_tools(all_tools)

    # Apply user overrides on top of profile
    if overrides:
        tools = [
            replace(t, description=overrides.get(t.name, t.description))
            for t in tools
        ]

    # Convert to requested format
    if format == "anthropic":
        return [t.to_anthropic() for t in tools]
    return [t.to_openai() for t in tools]
```

### Example 2: Tool Definition for "compress"
```python
# Source: Follows OpenAI function calling JSON Schema format (verified via official docs)

def _make_compress_tool(tract: Tract) -> ToolDefinition:
    """Build the compress tool definition."""
    return ToolDefinition(
        name="compress_context",
        description=(
            "Compress a range of commits in the context history into a concise summary. "
            "Use this when token usage is high and older context can be summarized "
            "without losing critical information. Pinned commits are preserved verbatim. "
            "Specify target_tokens to control summary length."
        ),
        parameters={
            "type": "object",
            "properties": {
                "target_tokens": {
                    "type": "integer",
                    "description": "Target token count for the compressed summary. "
                    "If not specified, uses the LLM's judgment for appropriate length.",
                },
                "from_commit": {
                    "type": "string",
                    "description": "Start of commit range to compress (hash prefix). "
                    "If not specified, compresses from the earliest commit.",
                },
                "to_commit": {
                    "type": "string",
                    "description": "End of commit range to compress (hash prefix). "
                    "If not specified, compresses up to the current HEAD.",
                },
                "instructions": {
                    "type": "string",
                    "description": "Additional instructions for the compression LLM. "
                    "E.g., 'Preserve all code snippets' or 'Focus on decisions made'.",
                },
            },
            "required": [],
        },
        handler=lambda **kwargs: tract.compress(**kwargs, auto_commit=True),
    )
```

### Example 3: OrchestratorConfig
```python
# Source: Follows TractConfig pattern from models/config.py

from enum import Enum

class AutonomyLevel(str, Enum):
    """Autonomy levels for the orchestrator."""
    MANUAL = "manual"
    COLLABORATIVE = "collaborative"
    AUTONOMOUS = "autonomous"

class OrchestratorState(str, Enum):
    """Orchestrator lifecycle states."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSING = "pausing"
    STOPPED = "stopped"

@dataclass
class TriggerConfig:
    """When the orchestrator should run."""
    on_commit_count: int | None = None  # Run every N commits
    on_token_threshold: float | None = None  # Run when tokens exceed % of budget
    on_compile: bool = False  # Run on every compile
    on_schedule_seconds: float | None = None  # Run every N seconds

@dataclass
class OrchestratorConfig:
    """Configuration for the built-in orchestrator."""
    autonomy_ceiling: AutonomyLevel = AutonomyLevel.COLLABORATIVE
    max_steps: int = 10  # Max tool calls per run
    profile: str = "self"  # Default tool profile
    system_prompt: str | None = None  # Override default assessment prompt
    task_context: str | None = None  # User-provided task description
    triggers: TriggerConfig | None = None
    model: str | None = None  # LLM model override
    temperature: float = 0.0  # Low temperature for deterministic actions
    on_proposal: Callable | None = None  # Callback for collaborative mode
    on_step: Callable | None = None  # Callback for each step (observability)
```

### Example 4: Orchestrator.run() -- Core Agent Loop
```python
# Source: Standard LLM tool-calling loop pattern

def run(self) -> OrchestratorResult:
    """Execute the orchestrator loop."""
    self._state = OrchestratorState.RUNNING
    steps: list[StepResult] = []

    # 1. Build initial context assessment
    assessment = self._build_assessment()
    messages = [
        {"role": "system", "content": self._get_system_prompt()},
        {"role": "user", "content": assessment},
    ]

    # 2. Get tool definitions
    tools = self._tract.as_tools(profile=self._config.profile)

    for step_num in range(self._config.max_steps):
        # Check stop/pause signals
        if self._stop_event.is_set():
            self._state = OrchestratorState.STOPPED
            break
        if self._pause_event.is_set():
            self._state = OrchestratorState.STOPPED
            break

        # 3. Call LLM with tools
        response = self._call_llm(messages, tools)

        # 4. Check if LLM returned tool calls
        tool_calls = self._extract_tool_calls(response)
        if not tool_calls:
            # LLM done -- no more actions needed
            break

        # 5. Process each tool call
        tool_results = []
        for tc in tool_calls:
            # Check autonomy ceiling
            effective_autonomy = self._check_autonomy(tc)

            if effective_autonomy == "collaborative":
                # Create proposal, await callback
                proposal = self._create_proposal(tc, response)
                decision = self._await_decision(proposal)
                if decision.decision == "rejected":
                    tool_results.append(ToolResult(
                        tool_name=tc.name, success=False,
                        error="Action rejected by reviewer",
                    ))
                    continue
                if decision.modified_action:
                    tc = decision.modified_action

            elif effective_autonomy == "manual":
                # Skip execution entirely
                continue

            # Execute the tool call
            result = self._executor.execute(tc.name, tc.arguments)
            tool_results.append(result)
            steps.append(StepResult(step=step_num, tool_call=tc, result=result))

        # 6. Send tool results back to LLM
        messages.append({"role": "assistant", "content": response})
        messages.append(self._format_tool_results(tool_results))

    self._state = OrchestratorState.IDLE
    return OrchestratorResult(steps=steps, state=self._state)
```

### Example 5: Built-in Callbacks
```python
# Source: CONTEXT.md -- "Built-in callbacks: auto_approve, cli_prompt, log_and_approve"

@dataclass
class ProposalResponse:
    """Response from a proposal review callback."""
    decision: str  # "approved", "rejected", "modified"
    modified_action: ToolCall | None = None
    reason: str = ""


def auto_approve(proposal: OrchestratorProposal) -> ProposalResponse:
    """Auto-approve all proposals (for autonomous mode)."""
    return ProposalResponse(decision="approved")


def log_and_approve(proposal: OrchestratorProposal) -> ProposalResponse:
    """Log proposal details and auto-approve (audit trail)."""
    logger.info(
        "Orchestrator proposal: %s -- %s",
        proposal.recommended_action.name,
        proposal.reasoning,
    )
    return ProposalResponse(decision="approved")


def cli_prompt(proposal: OrchestratorProposal) -> ProposalResponse:
    """Interactive CLI prompt for proposal review (requires [cli] extra)."""
    from rich.console import Console
    from rich.panel import Panel

    console = Console()
    console.print(Panel(
        f"[bold]Action:[/] {proposal.recommended_action.name}\n"
        f"[bold]Reasoning:[/] {proposal.reasoning}\n"
        f"[bold]Arguments:[/] {proposal.recommended_action.arguments}",
        title="Orchestrator Proposal",
    ))

    choice = console.input("[a]pprove / [r]eject / [m]odify: ").strip().lower()
    if choice.startswith("a"):
        return ProposalResponse(decision="approved")
    elif choice.startswith("r"):
        reason = console.input("Reason: ").strip()
        return ProposalResponse(decision="rejected", reason=reason)
    else:
        # Modify flow would need parameter editing UI
        return ProposalResponse(decision="approved")
```

### Example 6: LLM Tool Call Handling with OpenAI Format
```python
# Source: OpenAI chat completions API tool calling format (verified)

def _call_llm(self, messages: list[dict], tools: list[dict]) -> dict:
    """Call the LLM with tool definitions.

    Uses the built-in OpenAIClient or user-provided callable.
    The OpenAIClient.chat() already supports **kwargs, so tools=
    passes through directly to the API.
    """
    if self._llm is not None:
        # User-provided callable
        return self._llm(messages=messages, tools=tools)

    # Built-in LLM client
    client = getattr(self._tract, "_llm_client", None)
    if client is None:
        raise OrchestratorError(
            "No LLM client configured. Call tract.configure_llm() "
            "or provide llm_callable to the orchestrator."
        )
    return client.chat(
        messages,
        model=self._config.model,
        temperature=self._config.temperature,
        tools=tools,  # Passed via **kwargs to OpenAIClient.chat()
    )


def _extract_tool_calls(self, response: dict) -> list[ToolCall]:
    """Extract tool calls from an LLM response.

    OpenAI format: response["choices"][0]["message"]["tool_calls"]
    Each tool_call has: id, type, function.name, function.arguments (JSON string)
    """
    message = response.get("choices", [{}])[0].get("message", {})
    raw_calls = message.get("tool_calls", [])

    calls = []
    for raw in raw_calls:
        func = raw.get("function", {})
        calls.append(ToolCall(
            id=raw.get("id", ""),
            name=func.get("name", ""),
            arguments=json.loads(func.get("arguments", "{}")),
        ))
    return calls
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Manual SDK calls | Policy-triggered automatic actions | Phase 6 (just completed) | Policies define rules, but "how" is hardcoded |
| Hardcoded policy actions | LLM-reasoned context management | Phase 7 (this phase) | LLM decides HOW to act, not just WHEN |
| No tool schema export | `Tract.as_tools()` with profiles | Phase 7 (this phase) | Any external agent can consume Tract tools |
| No proposal review | Callback-based proposal flow | Phase 7 (this phase) | Humans can review, modify, or reject actions |
| Binary on/off for automation | Autonomy spectrum with runtime ceiling | Phase 7 (this phase) | Progressive trust: manual -> collaborative -> autonomous |

**Tool calling format landscape (2025-2026):**
- **OpenAI:** `{"type": "function", "function": {"name": ..., "description": ..., "parameters": {...}}}` with `tools=` parameter and `tool_calls` in response. Supports `strict: true` for guaranteed schema compliance.
- **Anthropic:** `{"name": ..., "description": ..., "input_schema": {...}}` with `tools=` parameter and `tool_use` content blocks in response.
- **Both use JSON Schema** for parameter definitions. The core schema format is identical; only the wrapper differs.
- **Decision:** Use OpenAI format as default (matches built-in LLM client), provide `.to_anthropic()` conversion. Framework adapters (LangChain, etc.) are deferred per CONTEXT.md.

**Existing codebase conventions that MUST be followed:**
- `from __future__ import annotations` at top of every module
- `if TYPE_CHECKING:` for heavy import guards
- `logger = logging.getLogger(__name__)` for module loggers
- `str, enum.Enum` for string enums (not plain Enum)
- Frozen dataclasses for result types (`@dataclass(frozen=True)`)
- Mutable dataclasses for proposal/config types (`@dataclass`)
- Pydantic BaseModel for user-facing config models (TractConfig pattern)
- Imports from `tract.xyz` (not relative imports)
- `self._session.commit()` at Tract facade level after operation completes
- Tools parameter format consistent with OpenAI function calling schema

## Open Questions

Things that couldn't be fully resolved and are marked as "Claude's Discretion" in CONTEXT.md:

1. **Exact profile names and default tool subsets per profile**
   - What we know: "self" and "supervisor" are mentioned. Self = agent managing own context. Supervisor = managing other agents' context.
   - What's unclear: Exact tool subsets for each profile. Should "self" include branch/merge? Should "supervisor" include spawn/collapse?
   - Recommendation: **"self" profile** includes: compile, commit, annotate, compress, status, log (core context management tools). Excludes: branch, merge, rebase, cherry-pick (advanced operations the agent is unlikely to use on itself). **"supervisor" profile** includes: all tools from "self" plus branch, merge, status on child tracts. A third **"full" profile** includes everything.

2. **Tool schema format details**
   - What we know: CONTEXT.md says "Claude's discretion on the exact format."
   - What's unclear: Whether to auto-generate from method signatures or hand-write each schema.
   - Recommendation: Hand-write tool schemas in `toolkit/definitions.py`. Auto-generation produces schemas that are too mechanical. The descriptions are the most important part and must be human-crafted per profile. Consider having a test that validates hand-written schemas against actual method signatures to catch drift.

3. **Orchestrator internal loop architecture (async, threading, etc.)**
   - What we know: CONTEXT.md says "Claude's discretion." Tract is synchronous.
   - What's unclear: Whether the orchestrator loop should use threading for the stop/pause signals.
   - Recommendation: Synchronous loop with `threading.Event` for stop/pause signals. The loop itself runs in the calling thread. `stop()` and `pause()` set events from any thread. The loop checks events between tool calls. No background threads, no asyncio.

4. **How the LLM structures its holistic context review internally**
   - What we know: "Free-form reasoning into actions." No scoring.
   - What's unclear: How much context to include in the assessment prompt.
   - Recommendation: Include token pressure (quantitative), recent N commit summaries (configurable, default 5), current branch info, and user-provided task context. Do NOT include full compiled context -- too expensive. If the LLM needs to read specific content, provide a `read_context` tool that returns the compiled messages.

5. **Built-in trigger condition implementations**
   - What we know: "Both periodic and event-triggered."
   - What's unclear: How triggers integrate with the existing policy evaluation hooks.
   - Recommendation: Triggers are checked in a new `_check_orchestrator_triggers()` method called from `compile()` and `commit()` (alongside policy evaluation). Periodic triggers need a simple timestamp-based check (not a background timer). Event triggers check conditions (token threshold, commit count). Configuration stored in `_trace_meta` alongside policy config.

6. **Whether ToolDefinition.handler should store the callable or use a dispatch map**
   - What we know: The PolicyEvaluator uses `_dispatch_action()` with a big if/elif block.
   - Recommendation: Use a dict mapping approach (`_handlers: dict[str, Callable]`) built once at init. This is more maintainable than if/elif and allows dynamic tool registration. Each ToolDefinition stores its handler callable directly.

## Sources

### Primary (HIGH confidence)
- **Existing codebase analysis** -- All patterns derived directly from reading source files:
  - `src/tract/tract.py` -- 1886 lines, full public API surface (20+ methods that become tools)
  - `src/tract/policy/protocols.py` -- Policy ABC pattern
  - `src/tract/policy/evaluator.py` -- Sidecar pattern, dispatch, proposal flow, recursion guard
  - `src/tract/policy/builtin/` -- Built-in policy implementations (compress, pin, branch, archive)
  - `src/tract/models/policy.py` -- PolicyAction, PolicyProposal, EvaluationResult dataclasses
  - `src/tract/llm/client.py` -- OpenAIClient with **kwargs passthrough (tools= works today)
  - `src/tract/llm/protocols.py` -- LLMClient, ResolverCallable protocols
  - `src/tract/protocols.py` -- CompiledContext, Message, TokenCounter protocols
  - `src/tract/models/config.py` -- TractConfig, TokenBudgetConfig patterns
  - `src/tract/exceptions.py` -- Exception hierarchy pattern
  - `src/tract/prompts/summarize.py` -- System prompt pattern for LLM operations
  - `pyproject.toml` -- Existing dependencies, optional extras pattern

### Secondary (MEDIUM confidence)
- **Anthropic tool use documentation** (https://platform.claude.com/docs/en/agents-and-tools/tool-use/implement-tool-use) -- Tool definition format (name, description, input_schema), tool_use/tool_result message flow, parallel tool use, error handling. Verified current as of 2025.
- **OpenAI function calling** (https://platform.openai.com/docs/guides/function-calling) -- Tool definition format (type:function, function:{name, description, parameters}), tool_calls response format, strict mode. URL accessible but content verified via secondary search.
- **Python threading.Event** (https://docs.python.org/3/library/threading.html) -- Standard library approach for graceful shutdown with Event.wait(timeout).

### Tertiary (LOW confidence)
- **Agent loop architecture patterns** -- WebSearch-only findings about LangGraph, Swarm, Langroid patterns. Not directly applicable since we're building a simple synchronous loop, not a full framework. Useful for confirming the "call LLM with tools, execute calls, repeat" pattern is standard.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - No new dependencies, all patterns from existing codebase
- Toolkit (tool definitions, profiles): HIGH - OpenAI/Anthropic tool formats are well-documented standards
- Orchestrator (agent loop): HIGH - Standard tool-calling loop pattern, well-understood
- Policy integration: HIGH - Extends existing PolicyEvaluator with minimal changes
- Proposal flow: HIGH - Extends existing PolicyProposal/PendingCompression patterns
- Stop/pause lifecycle: MEDIUM - threading.Event is standard but interaction with Tract session needs careful implementation
- Assessment prompts: MEDIUM - Prompt engineering is iterative; initial prompts will need tuning
- Profile tool subsets: MEDIUM - Discretionary decisions, recommendations provided

**Research date:** 2026-02-18
**Valid until:** Indefinite (internal codebase patterns + stable API formats)
