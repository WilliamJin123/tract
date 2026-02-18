# Requirements: Trace

**Defined:** 2026-02-10
**Core Value:** Agents produce better outputs when their context is clean, coherent, and relevant. Trace makes context a managed, version-controlled resource.

## v1 Requirements

Requirements for initial release. Each maps to roadmap phases.

### Core Operations

- [x] **CORE-01**: User can initialize a new trace (empty context with metadata)
- [x] **CORE-02**: User can commit a context snapshot with message, timestamp, parent pointer, and operation (append/edit); priority annotations (skip/normal/pinned) are separate metadata
- [x] **CORE-03**: User can view commit history (log) with token counts per commit
- [x] **CORE-04**: User can view current state (status): HEAD, current branch, token budget usage
- [x] **CORE-05**: User can compare two commits (diff) with textual difference output
- [x] **CORE-06**: User can reset HEAD to a previous commit (soft: keep content accessible, hard: discard forward)
- [x] **CORE-07**: User can checkout a specific commit for read-only inspection
- [x] **CORE-08**: Commits support structured content types: plain text, conversation messages (role: system/user/assistant/tool), and tool call results
- [x] **CORE-09**: Materialization preserves message structure (role, content) when rendering context for LLM consumption

### Branching & Merging

- [x] **BRNC-01**: User can create a named branch from current HEAD (pointer-based, not copy)
- [x] **BRNC-02**: User can switch active branch
- [x] **BRNC-03**: User can merge branch into current (fast-forward when possible, merge commit otherwise)
- [x] **BRNC-04**: User can trigger LLM-mediated semantic merge for conflicting/overlapping context
- [x] **BRNC-05**: User can rebase current branch onto target with semantic safety checks
- [x] **BRNC-06**: User can cherry-pick/inject specific commits from one branch to another

### Compression

- [x] **COMP-01**: User can compress a range of commits into a summary commit with token budget target
- [x] **COMP-02**: Pinned commits survive compression verbatim (hash verification)
- [x] **COMP-03**: User can reorder commits with semantic safety checks (warn when order change affects meaning)
- [x] **COMP-04**: User can run garbage collection to clean up unreachable commits with configurable retention policies

### Multi-Agent

- [x] **MAGT-01**: User can spawn a subagent trace linked to current commit (spawn pointer)
- [x] **MAGT-02**: Each subagent gets its own full trace repository
- [x] **MAGT-03**: User can collapse subagent trace into parent (summary + compressed trace + provenance pointer)
- [x] **MAGT-04**: User can expand a collapse commit to inspect subagent history for debugging
- [x] **MAGT-05**: All agent traces persist durably in storage
- [x] **MAGT-06**: User can resume from last committed state after process restart (crash recovery)
- [x] **MAGT-07**: User can query across repos within a session

### Infrastructure

- [x] **INFR-01**: SQLite storage via SQLAlchemy with content-addressable blobs and structured metadata
- [x] **INFR-02**: Token counting on every commit and operation (tiktoken default)
- [x] **INFR-03**: Pluggable tokenizer protocol for model-specific counters
- [x] **INFR-04**: API response token count extraction when available
- [x] **INFR-05**: Pluggable materializer with simple concatenation default
- [x] **INFR-06**: Custom materializer support for structured prompts (message arrays, XML tags, etc.)

### Interfaces

- [x] **INTF-01**: Python SDK as primary interface (Tract.open(), commit(), branch(), merge(), etc.)
- [x] **INTF-02**: CLI wrapper for inspection/debugging (tract log, status, diff, etc.) via Click + Rich
- [x] **INTF-03**: Built-in LLM client (httpx-based) for compression and semantic merge
- [x] **INTF-04**: User-provided callable support for custom LLM operations
- [x] **INTF-05**: pip-installable package with documentation and examples

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

### Automation — Policy Engine (Phase 6)

- [x] **AUTO-01**: User can define declarative policies (rules with triggers, thresholds, and actions) that the system evaluates automatically
- [x] **AUTO-02**: Auto-compress fires when token budget threshold is exceeded, producing valid compression commits that preserve pinned content
- [x] **AUTO-03**: Auto-pin applies heuristics (content type matching for InstructionContent/SessionContent) to automatically protect critical context from compression
- [x] **AUTO-04**: Auto-branch detects tangential exploration (content type switching patterns) and proposes branches to isolate it
- [x] **AUTO-05**: Auto-rebase detects stale branches (configurable staleness threshold + commit count) and proposes archiving
- [x] **AUTO-06**: Every automatic operation can be intercepted, reviewed, or overridden by a human (collaborative mode proposals, pause/resume, approve/reject)

### Automation — Context Management Agent (Phase 7)

- **AUTO-07**: Context management agent monitors context health (relevance, coherence, token pressure)
- **AUTO-08**: Agent proposes context operations with explanations humans can review
- **AUTO-09**: In autonomous mode, agent executes operations end-to-end without human intervention
- **AUTO-10**: Human can override or disable the context agent at any point without data loss

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
| CORE-01 | Phase 1 | Complete |
| CORE-02 | Phase 1 | Complete (updated: pin is annotation, not operation) |
| CORE-03 | Phase 2 | Complete |
| CORE-04 | Phase 2 | Complete |
| CORE-05 | Phase 2 | Complete |
| CORE-06 | Phase 2 | Complete |
| CORE-07 | Phase 2 | Complete |
| CORE-08 | Phase 1 | Complete |
| CORE-09 | Phase 1 | Complete |
| BRNC-01 | Phase 3 | Complete |
| BRNC-02 | Phase 3 | Complete |
| BRNC-03 | Phase 3 | Complete |
| BRNC-04 | Phase 3 | Complete |
| BRNC-05 | Phase 3 | Complete |
| BRNC-06 | Phase 3 | Complete |
| COMP-01 | Phase 4 | Complete |
| COMP-02 | Phase 4 | Complete |
| COMP-03 | Phase 4 | Complete |
| COMP-04 | Phase 4 | Complete |
| MAGT-01 | Phase 5 | Complete |
| MAGT-02 | Phase 5 | Complete |
| MAGT-03 | Phase 5 | Complete |
| MAGT-04 | Phase 5 | Complete |
| MAGT-05 | Phase 5 | Complete |
| MAGT-06 | Phase 5 | Complete |
| MAGT-07 | Phase 5 | Complete |
| INFR-01 | Phase 1 | Complete |
| INFR-02 | Phase 1 | Complete |
| INFR-03 | Phase 1 | Complete |
| INFR-04 | Phase 1 | Complete |
| INFR-05 | Phase 1 | Complete |
| INFR-06 | Phase 1 | Complete |
| INTF-01 | Phase 1 | Complete |
| INTF-02 | Phase 2 | Complete |
| INTF-03 | Phase 3 | Complete |
| INTF-04 | Phase 3 | Complete |
| INTF-05 | Phase 5 | Complete |

| AUTO-01 | Phase 6 | Complete |
| AUTO-02 | Phase 6 | Complete |
| AUTO-03 | Phase 6 | Complete |
| AUTO-04 | Phase 6 | Complete |
| AUTO-05 | Phase 6 | Complete |
| AUTO-06 | Phase 6 | Complete |
| AUTO-07 | Phase 7 | Pending |
| AUTO-08 | Phase 7 | Pending |
| AUTO-09 | Phase 7 | Pending |
| AUTO-10 | Phase 7 | Pending |

**Coverage:**
- v1 requirements: 37 total, all Complete
- v2 requirements: 10 total (6 Complete, 4 Pending)
- Mapped to phases: 47
- Unmapped: 0

---
*Requirements defined: 2026-02-10*
*Last updated: 2026-02-17 after Phase 6 completion (v2 AUTO requirements added)*
