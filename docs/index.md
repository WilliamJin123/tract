# Tract

**Git-like version control for LLM context windows.**

Agents produce better outputs when their context is clean, coherent, and
relevant. Tract makes context a managed, version-controlled resource —
with branches, compression, middleware, and multi-agent coordination.

```bash
pip install tract-ai            # Core library
pip install tract-ai[runner]    # + LLM clients, agent loop, toolkit
```

## Quickstart

```python
from tract import Tract

with Tract.open() as t:
    # Build context through commits
    t.system("You are a helpful research assistant.")
    t.user("Summarize recent ML papers on context management.")
    t.assistant("Here are the key papers from 2024-2025...")

    # Compile into LLM-ready messages
    ctx = t.compile()
    print(f"{ctx.commit_count} commits, {ctx.token_count} tokens")
    for msg in ctx.messages:
        print(f"  [{msg.role}] {msg.content[:60]}")

    # Full history
    for c in t.log():
        print(f"  {c.commit_hash[:8]} [{c.content_type}] {c.message}")
```

## Agent Loop

With the runner extra, tract provides a built-in agent loop:

```python
from tract import Tract

with Tract.open(api_key="sk-...", model="gpt-4o") as t:
    t.system("You are a coding assistant.")
    t.user("Write a Python function to check if a number is prime.")

    result = t.run()  # compile -> LLM -> tool calls -> repeat
    print(result.text)
```

## Why Tract?

LLM context windows are the most valuable resource in agent systems, but
they're treated as throwaway strings. Tract treats context like source code:

| Problem | Tract solution |
|---------|---------------|
| Context grows unbounded | **Compression** — summarize old commits, preserve pinned content |
| No structure to history | **Typed commits** — dialogue, instructions, tool I/O, reasoning, artifacts |
| Can't explore alternatives | **Branches** — fork context, try approaches, merge results |
| Multi-agent chaos | **Sessions** — spawn/collapse child agents with shared storage |
| No quality control | **Middleware** — gates, maintainers, and hooks on 12 lifecycle events |
| Config scattered everywhere | **Config system** — token budgets, compile strategies, LLM settings per-branch |

## Core Concepts

### Commits and Content Types

Every piece of context is an immutable commit with a typed payload:

```python
t.system("You are a code reviewer.")          # InstructionContent
t.user("Review this function.")               # DialogueContent (role=user)
t.assistant("Found 3 issues...")              # DialogueContent (role=assistant)
t.reasoning("The user wants a thorough...")   # ReasoningContent
t.metadata({"language": "python"})            # MetadataContent
t.commit(ArtifactContent(                     # ArtifactContent
    artifact_type="code", text="def foo(): ..."
))
```

Ten built-in types: `instruction`, `dialogue`, `tool_io`, `reasoning`,
`artifact`, `output`, `freeform`, `session`, `config`, `metadata`.
Register custom types with `t.register_content_type()`.

### Compile

Reconstruct LLM-ready messages from the commit chain:

```python
ctx = t.compile()
ctx.to_openai()      # [{"role": "system", "content": "..."}]
ctx.to_anthropic()   # Anthropic message format with content blocks
ctx.to_dicts()       # Generic list[dict]
```

Respects priority annotations (SKIP removes, PINNED survives compression),
token budgets, and compile strategies (`latest`, `sliding_window`).

### Branches and Merge

```python
t.branch("experiment")        # Create + switch
t.user("Try approach A...")
t.assistant("Result: ...")

t.checkout("main")            # Switch back
t.merge("experiment")         # Merge results in
```

### Compression

Collapse long histories into summaries while preserving pinned content:

```python
t.compress()                              # LLM-powered summarization
t.compress(strategy="sliding_window")     # Window-based compression
t.compress(content="Manual summary...")   # Provide your own
```

### Configuration

```python
t.configure(
    token_budget_max=8000,
    compile_strategy="latest",
    model="gpt-4o",
    temperature=0.7,
)
```

Config is branch-scoped and persisted as commits — it survives across sessions.

### Directives

Persistent instructions that survive compression:

```python
t.directive("tone", "Always respond in a professional tone.")
t.directive("format", "Use markdown with headers for structure.")
```

### Middleware

12 lifecycle events with handler registration:

```python
from tract.middleware import MiddlewareContext

def log_commits(ctx: MiddlewareContext):
    print(f"New commit: {ctx.commit.content_type}")

t.use("post_commit", log_commits)
```

Events: `pre_commit`, `post_commit`, `pre_compile`, `pre_compress`,
`pre_merge`, `pre_gc`, `pre_transition`, `post_transition`,
`pre_generate`, `post_generate`, `pre_tool_execute`, `post_tool_execute`.

### Semantic Gates

LLM-powered quality gates that enforce natural-language criteria:

```python
from tract.gate import SemanticGate

gate = SemanticGate(
    name="research-complete",
    check="At least 3 commits tagged 'key-finding' exist",
)
t.use("pre_transition", gate)
```

### Semantic Maintainers

LLM-powered context maintenance — automatic tagging, annotation, compression:

```python
from tract.maintain import SemanticMaintainer

maintainer = SemanticMaintainer(
    name="auto-tagger",
    instructions="Tag commits containing code as 'code-snippet'",
    actions=["tag"],
)
t.use("post_commit", maintainer)
```

### Templates and Profiles

Reusable directive templates and workflow profiles:

```python
# Templates — parameterized directives
t.apply_template("persona", role="senior engineer", domain="backend systems")

# Profiles — bundled config + directives for common workflows
t.load_profile("coding")    # Sets up coding-optimized config
t.apply_stage("implement")  # Apply stage-specific settings
```

Three built-in profiles: `coding`, `research`, `ecommerce`.

## Multi-Agent

```python
from tract import Session

with Session.open("project.db") as session:
    parent = session.create_tract(display_name="orchestrator")
    parent.system("Build a web application.")
    parent.user("Start with the backend.")

    # Spawn child for subtask
    child = session.spawn(parent, purpose="Design the database schema")
    child.assistant("Schema: users, posts, comments with indexes.")

    # Collapse back
    session.collapse(child, into=parent, content="DB schema complete.")
```

## Persistence and Recovery

Tract uses SQLite — everything persists automatically:

```python
# Session 1: do work
with Tract.open("project.db") as t:
    t.system("Build an API.")
    t.user("Start with auth.")
    t.assistant("Implementing JWT auth...")
    t.tag(t.head, "checkpoint")

# Session 2: pick up where you left off
with Tract.open("project.db") as t:
    ctx = t.compile()  # Full context restored
    print(f"Resuming with {ctx.token_count} tokens")
```

## Formatting and Inspection

Rich terminal output for debugging:

```python
from tract.formatting import pprint

t.compile().pprint()           # Pretty-print compiled context
t.compile().pprint(style="compact")  # Compact view
t.status().pprint()            # Current tract status
```

## Cookbook

40+ runnable examples in [`cookbook/`](cookbook/):

| Category | Examples |
|----------|---------|
| **Getting Started** | Quick start, config, tools, streaming, async, agent loop |
| **Config & Middleware** | Strategies, events, gates, transitions, observability |
| **Workflows** | Coding assistant, customer support, e-commerce, adversarial review |
| **Agent Patterns** | Branching, self-correction, staged workflows, multi-agent, supervisor-worker |
| **Reference** | Content types, branching, compression, metadata, validation, batch ops |
| **Persistence** | Checkpoints, portability, snapshots |
| **Optimization** | Budget management, production monitoring |
| **Error Handling** | Recovery strategies, graceful degradation |
| **Testing** | Mocking patterns |

## Architecture

```
tract/
  tract.py          # Public SDK facade (Tract class)
  models/           # Pydantic content types, config, annotations
  operations/       # DAG operations (branch, merge, rebase, compress, diff)
  storage/          # SQLite + SQLAlchemy persistence
  middleware.py     # Event system (12 lifecycle hooks)
  gate.py           # SemanticGate (LLM-powered quality gates)
  maintain.py       # SemanticMaintainer (LLM-powered context maintenance)
  templates.py      # Directive templates (9 built-in)
  profiles.py       # Workflow profiles (coding, research, ecommerce)
  protocols.py      # CompiledContext, Message, TokenCounter
  formatting.py     # Rich terminal output
  session.py        # Multi-agent Session coordinator
  llm/              # LLM client protocols and implementations
  toolkit/          # Tool definitions, executor, profiles
  loop.py           # Agent loop (run/arun)
```

## Development

```bash
git clone https://github.com/WilliamJin123/tract.git
cd tract
pip install -e ".[dev]"
python -m pytest tests/ -x -q   # 2726 tests
```

## License

MIT
