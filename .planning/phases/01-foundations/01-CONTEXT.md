# Phase 1: Foundations - Context

**Gathered:** 2026-02-10
**Status:** Ready for planning

<domain>
## Phase Boundary

SDK library for creating traces, committing structured context snapshots, and materializing them for LLM consumption with accurate token counts. Users can initialize a trace, commit typed content, materialize context as structured messages, and track token usage. Branching, merging, compression, and multi-agent features are out of scope.

</domain>

<decisions>
## Implementation Decisions

### Content Modeling

#### Type System
- Extensible type system with built-in defaults
- Users can register custom types for specialized protocols (e.g., custom mutation rules, boilerplate reduction strategies)
- Custom types are for power users designing specific protocols; built-in types should be comprehensive enough for most use cases

#### Built-in Content Types
- Six built-in types: **instruction**, **dialogue**, **tool_io**, **reasoning**, **artifact**, **output**
- Types describe the *nature of the content*, not conversation roles (a single user message may span multiple commits)
- Each type carries default behavioral hints: compression priority, materialization ordering, reduction rules

#### Content Schema
- Built-in types have enforced schemas (validated on commit)
- Freeform escape hatch: a generic content type accepts any payload without schema enforcement
- Custom types *optionally* define schemas; schemas enable protocol behavior (compression, mutation rules)

#### Commit Granularity
- Single type per commit (one commit = one content item)
- `tool_io` was considered as a full I/O cycle but decided: keep granular and consistent with other types (tool_call and tool_result are separate commits)
- Related commits linked via `reply_to` pointer (e.g., tool_result → tool_call, answer → question)

#### Linking Model
- Simple `reply_to` field: optional hash pointer to another commit
- Commit types carry the relationship semantics (a tool_result with reply_to pointing at a tool_call — the types tell you it's a call-result pair)
- No group IDs or typed relationships in Phase 1; add group_id in Phase 4 if multi-step grouping needs it

### Commit Semantics

#### Operations
- Three commit operations: **append**, **edit**, **delete**
- Append: add new content to history (default)
- Edit: create a new commit that supersedes a previous one (reply_to points at original). Full commit replacement, not partial/field-level edits
- Delete: soft-delete — marks a commit as excluded from materialization but preserved in history
- Operations are orthogonal to content types (an edit of a dialogue commit is still type dialogue)

#### Immutable History
- History is immutable — edits always create new commits, never mutate existing ones
- Content-addressable hashing stays clean (content never mutates, hashes always valid)
- Diffs between original and edited versions are trivially available

#### Priority Annotations (Unified Pin/Delete System)
- Unified three-tier priority system instead of separate pin/delete mechanisms
- **Skip** = soft delete: excluded from materialization entirely. Exists in history for provenance only
- **Normal** = standard: included in materialization, eligible for compression
- **Pinned** = protected: included in materialization, survives compression verbatim
- Priority tracked as lightweight annotations (like git tags), not commits: target hash, priority level, timestamp, optional reason message
- Full provenance: a commit can go normal → pinned → normal → skip over time, with history of changes and reasons
- Content types define default priority (e.g., instruction defaults to pinned, reasoning defaults to normal). Users can override

#### Commit Metadata
- Each commit carries an optional open metadata dict (no schema enforcement in Phase 1)
- Intended for compression directives, custom hints, and future extensibility
- Phase 4 will formalize the compression directive contract; Phase 1 just stores whatever the user puts there

### Materialization Output

#### Output Format
- Default materializer outputs structured message list [{role, content}, ...] ready for LLM APIs
- Configurable type-to-role mapping (instruction → system, dialogue → user/assistant, tool_io → tool messages, etc.)
- Multiple commits of the same type aggregate into combined messages (e.g., multiple instruction commits → one system message)
- RAG output is just committed content (likely instruction type or custom retrieval type) that maps through the same system

#### Edit Handling
- Silent replacement by default (materialization uses latest version, no indication of edits)
- Opt-in annotation mode for provenance-aware agents that want edit markers

#### Time-Travel
- Materialization supports "as of" a specific commit/timestamp — shows context as it existed at that point, only seeing commits and edits up to that point

#### Priority Respect
- Materializer respects priority: skip commits excluded, normal and pinned commits rendered

#### Pluggable Protocol
- Two layers of customization:
  1. Override hooks on default materializer (type-to-role mapping, aggregation strategy, ordering, edit handling)
  2. Full protocol replacement (implement the complete materializer interface for total control)

### Token Accounting

#### Counting Strategy
- Both eager (on commit) and lazy (on materialize) with caching
- Default to reading API-reported token counts from LLM API response payloads as source of truth
- Local tokenizer as fallback when API counts unavailable
- Callback/hook for extracting token usage from API responses, with default hooks for common providers (Anthropic, OpenAI) — research during planning to confirm payload standardization

#### Default Tokenizer
- Ship with a built-in default tokenizer for out-of-the-box experience
- User can swap for a model-specific tokenizer via pluggable protocol

#### Storage
- Per-commit token counts stored with each commit
- Running cumulative total maintained for current branch/HEAD — O(1) budget checks

#### Budget Enforcement
- Optional token budget with configurable behavior on exceed:
  - **Warn** — log warning, allow commit (default)
  - **Reject** — block the commit
  - **Callback** — call a user-provided function (Phase 4 plugs in smart compression here)
- Creates self-managing context: agents commit freely, compression triggers automatically when budget is exceeded

### Type-Level Compression Protocols
- Each content type defines default compression behavior as part of its behavioral hints
- Includes aggregation rules: how groups of same-type commits compress together (e.g., 100 tool_io commits → "called tool X 100 times, 50 failures, key results: ...")
- Compression strategies are ultimately LLM prompts — the type protocol defines the prompt template
- Per-commit metadata can override type-level defaults for specific commits
- Actual compression implementation is Phase 4; Phase 1 ensures the data model carries protocol definitions and metadata

</decisions>

<specifics>
## Specific Ideas

- Tool I/O should support boilerplate reduction — repeated tool calls are a primary compression target
- "I want agents to just keep committing and Trace handles the pressure" — self-managing context via auto-compression callback
- The type system should be comprehensive enough that custom types are only for specialized protocols, not everyday use
- Content types describe nature of content, not conversation roles — don't model around user/assistant message boundaries
- Pin reasons and delete reasons should be queryable — useful for debugging context management decisions

</specifics>

<deferred>
## Deferred Ideas

- Group IDs for multi-step interaction clustering — evaluate need in Phase 4 (compression)
- Gradation within "normal" priority (low-priority = compress first) — Phase 4 if needed
- Compression directive schema formalization — Phase 4 when compressor exists
- Cross-type group compression (mix of dialogue + tool_io from one task) — Phase 4

</deferred>

---

*Phase: 01-foundations*
*Context gathered: 2026-02-10*
