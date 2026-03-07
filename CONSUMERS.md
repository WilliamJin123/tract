# Tract Consumers

Projects that depend on tract-ai as an editable install.

## Active consumers

| Repo | Path | Key APIs | Status |
|------|------|----------|--------|
| tract-coding | `../tract-coding` | Session, run_loop, ToolSummarizationConfig, branching | Scaffold |
| tract-ecomm | `../tract-ecomm` | Session.spawn/collapse, branching (A/B), compression, tool compaction | Scaffold |
| tract-research | `../tract-research` | Session.spawn/collapse, branching (hypotheses), compression, knowledge persistence | Scaffold |

## API stability

### Stable (safe to depend on)
- `Tract.open()`, `.commit()`, `.compile()`, `.close()`
- `Tract.system()`, `.user()`, `.assistant()` shorthand
- `CompiledContext.to_dicts()`, `.to_openai()`, `.to_anthropic()`
- `Tract.branch()`, `.switch()`, `.merge()`
- `Session.open()`, `.create_tract()`, `.spawn()`, `.collapse()`
- All content types (InstructionContent, DialogueContent, etc.)
- CommitInfo, CommitOperation, Priority

### In flux (may change)
- `run_loop()` — API shape still being validated
- `ToolSummarizationConfig` — compaction behavior being tuned
- `OperationConfigs` / `OperationClients` — wiring may simplify
- Orchestrator module — may be replaced by run_loop

### Not yet validated by consumers
- `Tract.chat()` / `.generate()` — convenience methods untested in real workflows
- `ConfigIndex` — query_by_config patterns
- Middleware system — `t.on()` / `t.off()` hooks

## Breaking changes

| Date | Change | Consumers affected | Migration |
|------|--------|--------------------|-----------|
| (none yet) | | | |
