# Requirements: Trace

**Defined:** 2026-02-10
**Core Value:** Agents produce better outputs when their context is clean, coherent, and relevant. Trace makes context a managed, version-controlled resource.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Operations

- [ ] **CORE-01**: User can initialize a new trace (empty context with metadata)
- [ ] **CORE-02**: User can commit a context snapshot with message, timestamp, parent pointer, and commit type (append/edit/pin)
- [ ] **CORE-03**: User can view commit history (log) with token counts per commit and cumulative
- [ ] **CORE-04**: User can view current state (status): HEAD, current branch, token budget usage
- [ ] **CORE-05**: User can compare two commits (diff) with textual difference output
- [ ] **CORE-06**: User can reset HEAD to a previous commit (soft: keep content accessible, hard: discard forward)
- [ ] **CORE-07**: User can checkout a specific commit for read-only inspection
- [ ] **CORE-08**: Commits support structured content types: plain text, conversation messages (role: system/user/assistant/tool), and tool call results
- [ ] **CORE-09**: Materialization preserves message structure (role, content) when rendering context for LLM consumption

### Branching & Merging

- [ ] **BRNC-01**: User can create a named branch from current HEAD (pointer-based, not copy)
- [ ] **BRNC-02**: User can switch active branch
- [ ] **BRNC-03**: User can merge branch into current (fast-forward when possible, merge commit otherwise)
- [ ] **BRNC-04**: User can trigger LLM-mediated semantic merge for conflicting/overlapping context
- [ ] **BRNC-05**: User can rebase current branch onto target with semantic safety checks
- [ ] **BRNC-06**: User can cherry-pick/inject specific commits from one branch to another

### Compression

- [ ] **COMP-01**: User can compress a range of commits into a summary commit with token budget target
- [ ] **COMP-02**: Pinned commits survive compression verbatim (hash verification)
- [ ] **COMP-03**: User can reorder commits with semantic safety checks (warn when order change affects meaning)
- [ ] **COMP-04**: User can run garbage collection to clean up unreachable commits with configurable retention policies

### Multi-Agent

- [ ] **MAGT-01**: User can spawn a subagent trace linked to current commit (spawn pointer)
- [ ] **MAGT-02**: Each subagent gets its own full trace repository
- [ ] **MAGT-03**: User can collapse subagent trace into parent (summary + compressed trace + provenance pointer)
- [ ] **MAGT-04**: User can expand a collapse commit to inspect subagent history for debugging
- [ ] **MAGT-05**: All agent traces persist durably in storage
- [ ] **MAGT-06**: User can resume from last committed state after process restart (crash recovery)
- [ ] **MAGT-07**: User can query across repos within a session

### Infrastructure

- [ ] **INFR-01**: SQLite storage via SQLAlchemy with content-addressable blobs and structured metadata
- [ ] **INFR-02**: Token counting on every commit and operation (tiktoken default)
- [ ] **INFR-03**: Pluggable tokenizer protocol for model-specific counters
- [ ] **INFR-04**: API response token count extraction when available
- [ ] **INFR-05**: Pluggable materializer with simple concatenation default
- [ ] **INFR-06**: Custom materializer support for structured prompts (message arrays, XML tags, etc.)

### Interfaces

- [ ] **INTF-01**: Python SDK as primary interface (Repo.open(), commit(), branch(), merge(), etc.)
- [ ] **INTF-02**: CLI wrapper for inspection/debugging (trace log, status, diff, etc.) via Click + Rich
- [ ] **INTF-03**: Built-in LLM client (httpx-based) for compression and semantic merge
- [ ] **INTF-04**: User-provided callable support for custom LLM operations
- [ ] **INTF-05**: pip-installable package with documentation and examples

## v2 Requirements

Deferred to future release. Tracked but not in current roadmap.

### Virtual Context

- **VCTX-01**: Virtual context with tiered storage — active window vs archival, with paging between tiers
- **VCTX-02**: Automatic promotion/demotion of context between tiers based on relevance

### Observability

- **SMDIFF-01**: Semantic diff — LLM-powered "what meaning changed" between states
- **AUDIT-01**: Trace blame — track origin of specific context content
- **AUDIT-02**: Cross-agent audit queries ("what did agent-3 know when agent-1 decided X?")

### Ecosystem

- **ADAPT-01**: Framework adapters for LangChain, CrewAI, OpenAI Agents SDK
- **GUI-01**: Web/desktop visual DAG explorer for multi-agent traces (inspired by OpenTelemetry viewers)

### Automation

- **AUTO-01**: Policy engine for automatic context management (auto-branch on tangents, auto-compress at threshold)
- **AUTO-02**: Context management agent that monitors and manages another agent's Trace

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| RAG system / vector database | Well-solved by LlamaIndex, Pinecone, etc. Trace manages context windows, not knowledge retrieval |
| Agent framework / orchestration | CrewAI, LangGraph, etc. own orchestration. Trace is a composable library |
| Real-time streaming VCS | Version-control completed content, not partial streaming tokens |
| Framework-specific adapters in v1 | Need stable SDK first. Adapters are v2 |
| GUI/visualization in v1 | SDK must be proven first. GUI is v2 |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| CORE-01 | Phase 1 | Pending |
| CORE-02 | Phase 1 | Pending |
| CORE-03 | Phase 2 | Pending |
| CORE-04 | Phase 2 | Pending |
| CORE-05 | Phase 2 | Pending |
| CORE-06 | Phase 2 | Pending |
| CORE-07 | Phase 2 | Pending |
| CORE-08 | Phase 1 | Pending |
| CORE-09 | Phase 1 | Pending |
| BRNC-01 | Phase 3 | Pending |
| BRNC-02 | Phase 3 | Pending |
| BRNC-03 | Phase 3 | Pending |
| BRNC-04 | Phase 3 | Pending |
| BRNC-05 | Phase 3 | Pending |
| BRNC-06 | Phase 3 | Pending |
| COMP-01 | Phase 4 | Pending |
| COMP-02 | Phase 4 | Pending |
| COMP-03 | Phase 4 | Pending |
| COMP-04 | Phase 4 | Pending |
| MAGT-01 | Phase 5 | Pending |
| MAGT-02 | Phase 5 | Pending |
| MAGT-03 | Phase 5 | Pending |
| MAGT-04 | Phase 5 | Pending |
| MAGT-05 | Phase 5 | Pending |
| MAGT-06 | Phase 5 | Pending |
| MAGT-07 | Phase 5 | Pending |
| INFR-01 | Phase 1 | Pending |
| INFR-02 | Phase 1 | Pending |
| INFR-03 | Phase 1 | Pending |
| INFR-04 | Phase 1 | Pending |
| INFR-05 | Phase 1 | Pending |
| INFR-06 | Phase 1 | Pending |
| INTF-01 | Phase 1 | Pending |
| INTF-02 | Phase 2 | Pending |
| INTF-03 | Phase 3 | Pending |
| INTF-04 | Phase 3 | Pending |
| INTF-05 | Phase 5 | Pending |

**Coverage:**
- v1 requirements: 37 total
- Mapped to phases: 37
- Unmapped: 0

---
*Requirements defined: 2026-02-10*
*Last updated: 2026-02-10 after roadmap creation (phases renumbered 1-5)*
