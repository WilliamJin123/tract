# Tract Agent Labs

Experimental scenarios testing how autonomous agents use Tract to manage their own context. Unlike `cookbook/` (which teaches the API from a developer's perspective), labs test **agent decision quality** — did the LLM make good choices about its own context?

Every lab requires real LLM calls. Results are non-deterministic. Each lab captures metrics and decision traces for evaluation.

## How Labs Differ from Cookbooks

| Dimension | `cookbook/` | `labs/` |
|-----------|-----------|---------|
| Perspective | Developer calling API | Agent managing itself |
| Determinism | Mostly deterministic | Non-deterministic (LLM decisions) |
| Output | "Here's what the API does" | "Here's how well the agent decided" |
| Success criteria | Code runs, shows feature | Metrics: tokens saved, info preserved, decision quality |
| LLM required | Some (chat, compress) | All |

## Conventions

### File Structure

```python
"""Lab Title

Setup: What context state the agent starts with.
Decision: What the agent must decide or accomplish.
Evaluates: What metrics we capture.

Demonstrates: tract APIs exercised
Compares: what axes are being contrasted (if any)
"""

def main():
    # --- Setup: build initial state ---
    # --- Agent loop: LLM makes decisions ---
    # --- Evaluate: measure outcomes ---
    # --- Report: print metrics ---
    pass

if __name__ == "__main__":
    main()
```

### Metrics Capture

Every lab should print a summary block at the end:

```
=== Lab Results ===
Tokens before: 6200
Tokens after:  3100
Reduction:     50%
Facts preserved: 8/8 (100%)
Decisions made: compress(3 msgs), skip(2 msgs), pin(1 msg)
```

### Fixtures

Shared test data lives in `labs/fixtures/`. Labs should use these for reproducible setups rather than generating conversations inline.

---

## File Tree

```
labs/
├── SCENARIOS.md
├── 01_tool_interfaces/
│   ├── 01_programmatic.py
│   ├── 02_mcp_server.py
│   ├── 03_cli_skill.py
│   └── 04_paradigm_comparison.py
├── 02_self_curation/
│   ├── 01_autonomous_compress.py
│   ├── 02_selective_pinning.py
│   ├── 03_edit_own_context.py
│   └── 04_policy_self_config.py
├── 03_subagent_patterns/
│   ├── 01_branch_delegate_merge.py
│   ├── 02_shared_tract.py
│   ├── 03_independent_tracts.py
│   └── 04_hierarchical.py
├── 04_autonomous_loops/
│   ├── 01_research_session.py
│   ├── 02_exploration_branching.py
│   ├── 03_self_correcting.py
│   └── 04_long_running.py
├── 05_evaluation/
│   ├── 01_token_efficiency.py
│   ├── 02_information_retention.py
│   ├── 03_decision_quality.py
│   └── 04_paradigm_benchmark.py
├── fixtures/
│   ├── long_conversation.json
│   ├── research_corpus.md
│   └── evaluation_rubrics.md
└── results/
    └── .gitkeep
```

---

# Part 1: Tool Interface Labs

## 01 — Tool Interfaces

How does an agent invoke Tract operations? Three paradigms, each with different trade-offs for latency, flexibility, and LLM awareness.

### 01/01 — Programmatic Tool Calling

**Setup:** An agent loop (Python) with Tract operations exposed as OpenAI-format tool definitions via `as_tools()`. The LLM receives tool schemas and decides which operations to call.

**Decision:** Given a conversation at 85% token budget, the LLM must choose which Tract tool to invoke (compress, skip, branch, etc.) and with what arguments.

**Success criteria:**
- LLM correctly identifies budget pressure from status output
- Selected tool is appropriate for the situation
- Tool arguments are valid (real commit hashes, valid priorities)
- Context stays under budget after the operation

**Compares:** Baseline for 02 (MCP) and 03 (CLI) — same scenario, different interface.

> `as_tools(format="openai")`, `ToolExecutor`, tool call parsing, `chat()` with tools

### 01/02 — MCP Server

**Setup:** Tract exposed as an MCP (Model Context Protocol) server. An MCP-compatible client connects and discovers available tools dynamically.

**Decision:** Same budget pressure scenario as 01/01, but the agent discovers and invokes tools through MCP's tool discovery protocol instead of static schemas.

**Success criteria:**
- MCP server starts and exposes Tract tools
- Client discovers tools correctly
- Tool invocation produces same results as programmatic path
- Latency overhead of MCP transport is measured

**Compares:** MCP transport overhead vs direct programmatic calls.

> MCP server setup, tool discovery, tool invocation, transport latency

### 01/03 — CLI + Skill

**Setup:** An agent that interacts with Tract through CLI commands (shell execution) or natural language skill invocations that map to CLI operations.

**Decision:** Same budget pressure scenario. The agent must formulate correct CLI commands or skill invocations to manage context.

**Success criteria:**
- Agent generates valid CLI commands
- Commands execute successfully
- Output parsing is correct (agent understands CLI output)
- End state matches programmatic and MCP paths

**Compares:** Natural language command generation vs structured tool calling.

> CLI commands (`tract status`, `tract compress`, etc.), output parsing, skill mapping

### 01/04 — Paradigm Comparison

**Setup:** Run the identical scenario (budget pressure → decide action → execute → verify) through all three interfaces. Capture decision quality, latency, error rate, and token overhead per paradigm.

**Evaluates:**
- **Decision equivalence:** Do all three paradigms reach the same conclusion?
- **Latency:** Wall-clock time per paradigm
- **Token overhead:** How many tokens does each interface consume for the meta-operations?
- **Error rate:** How often does the agent generate invalid tool calls per paradigm?
- **Robustness:** Which paradigm degrades most gracefully under ambiguous situations?

> Side-by-side metrics table, statistical comparison across N runs

---

# Part 2: Self-Curation Labs

## 02 — Self-Curation

The agent manages its own context: deciding what to compress, pin, skip, edit, and how to configure policies — without human guidance.

### 02/01 — Autonomous Compression

**Setup:** A 40-message research conversation loaded from `fixtures/long_conversation.json`. Token budget is 4000, current usage is ~6200. The agent has Tract toolkit tools available.

**Decision:** The agent must:
1. Detect that it's over budget (via `status()` or compile metrics)
2. Decide which messages to target for compression
3. Choose compression parameters (target_tokens, range)
4. Execute the compression
5. Verify the result stays under budget

**Success criteria:**
- Stays under budget after compression
- Preserves all named entities and numerical facts from IMPORTANT messages
- Compression ratio is reasonable (not over-aggressive)
- Agent's reasoning for what to compress is logged and reviewable

**Evaluates:** Token reduction ratio, fact preservation score (automated check against known facts in fixture), number of LLM calls to complete.

> `status()`, `compile()`, `compress_range()`, IMPORTANT/retain awareness

### 02/02 — Selective Pinning

**Setup:** A 25-message conversation where 5 messages contain critical reference information (API keys format, schema definitions, business rules) and 20 are routine back-and-forth. No messages are pinned yet.

**Decision:** The agent reviews the conversation and decides which messages deserve PINNED status to protect them from future compression.

**Success criteria:**
- Agent pins the 5 critical messages (precision)
- Agent does not pin routine messages (recall)
- Agent provides reasoning for each pin decision
- Pinned messages survive a subsequent compression pass

**Evaluates:** Precision/recall against ground truth labels, reasoning quality (LLM-as-judge).

> `compile()`, `annotate(hash, PINNED)`, `log()`, message content analysis

### 02/03 — Edit Own Context

**Setup:** A conversation where the agent made a factual error 8 messages ago (e.g., stated "the API rate limit is 100 req/s" when it's actually 60 req/s). The agent later receives a correction from the user.

**Decision:** The agent must:
1. Identify the erroneous message in history
2. Decide whether to edit-in-place vs skip+re-state
3. Execute the correction
4. Verify the compiled context reflects the fix

**Success criteria:**
- Agent correctly identifies the erroneous commit
- Uses EDIT operation (not just appending a correction)
- Corrected content is factually accurate
- Compiled context shows corrected version, not original

**Evaluates:** Error identification accuracy, correction strategy quality, compiled context correctness.

> `log()`, `commit(operation=EDIT, edit_target=hash)`, `compile()`, `diff()`

### 02/04 — Policy Self-Configuration

**Setup:** The agent starts with no policies configured. It's given a meta-instruction: "You'll be running for a long session. Configure your own context management policies based on what you think will keep you effective."

**Decision:** The agent must:
1. Assess its own operational context (budget, expected session length, task type)
2. Choose appropriate policies (compress threshold, pin rules, etc.)
3. Configure them via Tract's policy API
4. Justify its choices

**Success criteria:**
- Agent configures at least 2 policies
- Chosen policies are appropriate for the described scenario
- Agent can explain why it chose each policy
- Policies actually fire when their conditions are met (verified in subsequent turns)

**Evaluates:** Policy selection appropriateness (LLM-as-judge), policy effectiveness (do they fire correctly?), configuration correctness.

> `configure_policies()`, `CompressPolicy`, `PinPolicy`, policy triggers, `status()`

---

# Part 3: Subagent Pattern Labs

## 03 — Subagent Patterns

How Tract operates in multi-agent architectures. The core question: when does branch+edit within one Tract replace the need for a subagent, and when is a separate Tract genuinely needed?

### 03/01 — Branch, Delegate, Merge

**Setup:** A parent agent has a 20-message conversation and needs a focused research task done. Instead of spawning a separate process, it branches within its own Tract.

**Decision:** The parent agent:
1. Branches from current HEAD
2. Switches to the branch
3. Performs focused research work (multiple commits)
4. Compresses the research into a summary
5. Switches back to main
6. Merges or cherry-picks the summary

**Success criteria:**
- Main branch is untouched during research
- Research branch has focused, relevant commits
- Summary accurately captures research findings
- Merge produces clean result on main
- Token count on main is lower than if research was done inline

**Evaluates:** Context isolation quality, summary fidelity, token efficiency vs inline approach.

> `branch()`, `switch()`, `compress()`, `merge()` or `cherry_pick()`, `compile()` on both branches

### 03/02 — Shared Tract

**Setup:** Two agent personas (researcher and editor) share a single Tract instance. They take turns committing, and each needs to see the other's contributions.

**Decision:** How to coordinate turn-taking and context awareness when both agents write to the same timeline.

**Success criteria:**
- Both agents can read each other's commits
- No lost updates or race conditions (sequential access)
- Each agent can distinguish its own commits from the other's (via role or name)
- Final compiled context is coherent

**Evaluates:** Context coherence, agent awareness of each other's contributions, coordination overhead.

> Shared `Tract` instance, `commit(name="researcher")`, `log()`, `compile()`

### 03/03 — Independent Tracts

**Setup:** Three specialist agents (researcher, coder, reviewer) each own their own Tract. They work independently and sync results at defined checkpoints.

**Decision:** How to transfer context between independent Tracts without sharing the full history. Options:
1. Compress and commit summary into receiving Tract
2. Cherry-pick specific commits
3. Compile and commit the full message list

**Success criteria:**
- Each agent maintains focused context for its specialty
- Sync points transfer essential information without bloating receiving Tract
- No information loss at sync boundaries
- Total token usage across all Tracts is lower than a single shared Tract would be

**Evaluates:** Per-agent token efficiency, information transfer fidelity, sync overhead.

> Multiple `Tract.open()` instances, `compile()` → `commit()` cross-Tract, `compress()`

### 03/04 — Hierarchical Orchestration

**Setup:** An orchestrator agent owns a top-level Tract. It spawns child tasks, each getting a child Tract linked to the parent. Children work autonomously and report back.

**Decision:** The orchestrator must:
1. Decide when to spawn a child (vs handle inline)
2. Provision the child's initial context (what subset of parent context?)
3. Monitor child progress
4. Ingest child results back into parent context

**Success criteria:**
- Orchestrator makes reasonable spawn vs inline decisions
- Child context is appropriately scoped (not full parent dump)
- Child results are cleanly integrated into parent
- Provenance chain is intact (parent → child → result → parent)

**Evaluates:** Spawn decision quality, context scoping, integration cleanliness, provenance completeness.

> `parent()`, `children()`, child Tract provisioning, result ingestion, provenance chain

---

# Part 4: Autonomous Loop Labs

## 04 — Autonomous Loops

End-to-end agent sessions where the LLM drives Tract operations over multiple turns without human intervention.

### 04/01 — Research Session

**Setup:** Agent is given a research question and access to web search tool (simulated or real). It must research, accumulate findings, manage growing context, and produce a final summary.

**Decision points (multiple, across the session):**
1. When to search vs synthesize from existing context
2. When context is getting bloated — compress or skip?
3. Which findings are important enough to pin?
4. When is the research "done" — produce final output?

**Success criteria:**
- Agent produces a factually grounded research summary
- Context stays within budget throughout the session
- Agent actively manages context (at least 2 curation operations)
- Final summary references specific findings (not generic)

**Evaluates:** Research quality (LLM-as-judge), context management activity, token efficiency over session, curation decision quality.

> `chat()`, `status()`, `annotate()`, `compress()`, tool use, multi-turn loop

### 04/02 — Exploration Branching

**Setup:** Agent faces an ambiguous design decision with 2-3 viable approaches. It must explore each approach, evaluate outcomes, and choose the best one.

**Decision:** The agent must:
1. Identify that the problem has multiple viable solutions
2. Branch for each approach
3. Explore each branch (3-5 commits of reasoning per branch)
4. Compare results across branches
5. Choose the best branch and merge/cherry-pick to main

**Success criteria:**
- Agent creates branches for distinct approaches (not duplicates)
- Each branch contains genuine exploration (not token-wasting)
- Agent's comparison is substantive (references specific branch content)
- Final choice is well-reasoned

**Evaluates:** Branch diversity, exploration depth, comparison quality, final choice justification.

> `branch()`, `switch()`, `compile()` per branch, `diff()` across branches, `merge()` or `cherry_pick()`

### 04/03 — Self-Correcting Agent

**Setup:** Agent has a structured output task (generate valid JSON API response). The first attempt will likely have errors. The agent must detect errors, use Tract's edit to fix them, and produce correct output.

**Decision:** The agent must:
1. Generate initial output
2. Validate it (schema check, business rule check)
3. On failure: edit the bad commit or append a correction
4. Re-validate
5. Loop until correct or retries exhausted

**Success criteria:**
- Agent detects validation failures
- Uses EDIT (not just appending "actually, here's the fix")
- Compiled context shows only the corrected version
- Final output passes all validators
- History preserves the correction chain for audit

**Evaluates:** Error detection rate, correction strategy (edit vs append), final correctness, retry count.

> `chat(validator=)`, `commit(operation=EDIT)`, `compile()`, validation loop, `log()` for audit

### 04/04 — Long-Running Session

**Setup:** Agent runs a 100+ turn session simulating a day-long coding assistant. Context will exceed budget multiple times. Policies and orchestrator are available.

**Decision points (ongoing):**
- When to compress (manual? policy-triggered? orchestrator?)
- What to pin as session progresses
- When to branch for tangential tasks
- When to GC old archives

**Success criteria:**
- Session completes all 100+ turns without context overflow
- Agent maintains coherent awareness of early-session decisions
- Compression preserves critical context (tested via targeted questions)
- Performance doesn't degrade in later turns (measured by response relevance)

**Evaluates:** Context survival rate, coherence over time (LLM-as-judge), compression frequency, total tokens consumed.

> Full Tract API, policies, orchestrator triggers, `compress()`, `gc()`, `status()`, 100+ turns

---

# Part 5: Evaluation Labs

## 05 — Evaluation and Benchmarks

Systematic measurement of agent context management quality. These labs don't test features — they test whether the agent makes good decisions.

### 05/01 — Token Efficiency

**Setup:** Same 50-message conversation run through three strategies:
1. No management (just accumulate)
2. Programmatic rules (compress at 80%, skip tool outputs)
3. Agent-driven (LLM decides all curation)

**Evaluates:**
- Total tokens consumed per strategy
- Tokens per useful output token (efficiency ratio)
- Budget violations per strategy
- Cost estimate per strategy

> Controlled comparison, same input, three strategies, metrics table

### 05/02 — Information Retention

**Setup:** A conversation with 20 planted "key facts" (names, numbers, dates, decisions). Run compression with three aggressiveness levels. After each, quiz the agent on all 20 facts.

**Evaluates:**
- Fact recall rate per compression level
- Which fact types survive best (numbers vs names vs decisions)
- Does IMPORTANT + retain_match improve retention?
- False memory rate (agent "remembers" things not in context)

> `compress(target_tokens=...)` at 75%/50%/25% reduction, fact quiz, ground truth comparison

### 05/03 — Decision Quality

**Setup:** Present the agent with 10 context management scenarios (budget pressure, stale context, noisy tool outputs, etc.). For each, record what action the agent takes and compare to expert-labeled "best action."

**Evaluates:**
- Agreement rate with expert labels
- Severity of disagreements (minor vs catastrophic)
- Does the agent explain its reasoning?
- Consistency across runs (same scenario, different random seeds)

> 10 scenarios, expert labels, LLM-as-judge for reasoning quality, multi-run consistency

### 05/04 — Paradigm Benchmark

**Setup:** Run the full evaluation suite (05/01 + 05/02 + 05/03) across all three tool interface paradigms (programmatic, MCP, CLI). Same model, same scenarios, different interfaces.

**Evaluates:**
- Does the interface paradigm affect decision quality?
- Token overhead per paradigm (meta-operations cost)
- Error rate per paradigm (invalid tool calls, parse failures)
- Composite score: decision quality × efficiency × reliability

> Cross-paradigm matrix, composite scoring, statistical significance
