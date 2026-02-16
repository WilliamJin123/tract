---
phase: "04"
plan: "01"
subsystem: "compression-storage"
tags: ["schema", "repository", "migration", "models", "prompts"]
dependency-graph:
  requires: ["03-05"]
  provides: ["compression-schema", "compression-repository", "compression-models", "summarization-prompt"]
  affects: ["04-02", "04-03"]
tech-stack:
  added: []
  patterns: ["repository-pattern-for-compression", "schema-migration-chain"]
key-files:
  created:
    - "src/tract/storage/schema.py (CompressionRow, CompressionSourceRow, CompressionResultRow)"
    - "src/tract/models/compression.py"
    - "src/tract/prompts/__init__.py"
    - "src/tract/prompts/summarize.py"
    - "tests/test_compression_storage.py"
  modified:
    - "src/tract/storage/repositories.py (CompressionRepository ABC)"
    - "src/tract/storage/sqlite.py (SqliteCompressionRepository)"
    - "src/tract/storage/engine.py (v2->v3 migration)"
    - "src/tract/exceptions.py (CompressionError, GCError)"
    - "tests/test_storage/test_schema.py (schema version assertion update)"
decisions:
  - id: "04-01-01"
    description: "Schema version bumped 2->3 with auto-migration for compression tables"
  - id: "04-01-02"
    description: "CompressionRepository follows same ABC+SQLite pattern as other repositories"
  - id: "04-01-03"
    description: "PendingCompression is mutable (not frozen) to allow summary editing before approval"
  - id: "04-01-04"
    description: "v1->v2 migration chain extended: v1->v2->v3 runs sequentially for v1 databases"
metrics:
  duration: "6m"
  completed: "2026-02-16"
---

# Phase 4 Plan 1: Compression Storage Foundation Summary

**One-liner:** 3 ORM tables with v2->v3 migration, repository ABC+SQLite impl, 4 domain models, summarization prompt module

## What Was Built

### Storage Layer
- **CompressionRow** table: tracks compression records with tract_id, branch_name, token counts, target_tokens, and user instructions
- **CompressionSourceRow** table: associates compression records with their source (compressed) commits, preserving position order
- **CompressionResultRow** table: associates compression records with their result (summary) commits, preserving position order
- **Schema migration v2->v3**: auto-creates compression tables for existing v2 databases; v1 databases chain through v1->v2->v3
- **CompressionRepository ABC**: 8 abstract methods (save_record, add_source, add_result, get_record, get_sources, get_results, is_source_of, get_all_source_hashes)
- **SqliteCompressionRepository**: full SQLAlchemy implementation of all 8 methods

### Domain Models
- **CompressResult** (frozen): immutable result of completed compression with token counts, commit lists, compression ratio, new HEAD
- **PendingCompression** (mutable): editable draft with summaries list, edit_summary() for revision, approve() for finalization via callback
- **GCResult** (frozen): garbage collection stats (commits/blobs removed, tokens freed, duration)
- **ReorderWarning** (frozen): warnings about commit reordering issues during compression

### Exceptions
- **CompressionError(TraceError)**: raised when compression fails
- **GCError(TraceError)**: raised when garbage collection fails

### Prompts
- **DEFAULT_SUMMARIZE_SYSTEM**: system prompt for LLM summarization (third-person prose, preserve specifics, omit pleasantries)
- **build_summarize_prompt()**: builds user prompt with optional target_tokens and instructions

## Test Results

| Category | Tests | Status |
|----------|-------|--------|
| Schema (tables, migration, roundtrip) | 5 | Pass |
| Repository (CRUD, queries) | 8 | Pass |
| Domain models (frozen, mutable, errors) | 5 | Pass |
| Prompts (content, builder) | 3 | Pass |
| **New tests total** | **21** | **Pass** |
| **Existing tests** | **489** | **Pass** |
| **Full suite** | **510** | **Pass** |

## Commits

| Hash | Type | Description |
|------|------|-------------|
| bce5d87 | feat | Schema tables, repository pattern, v2->v3 migration |
| 5187f31 | feat | Domain models, exceptions, summarization prompt, 21 tests |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Updated existing schema version test assertion**
- **Found during:** Task 1 verification
- **Issue:** `test_trace_meta_has_schema_version` asserted schema_version=="2" but we bumped to "3"
- **Fix:** Updated assertion from "2" to "3" in tests/test_storage/test_schema.py
- **Commit:** bce5d87

## Decisions Made

| ID | Decision | Rationale |
|----|----------|-----------|
| 04-01-01 | Schema version 2->3 with auto-migration | Follows established pattern from v1->v2 in Phase 3 |
| 04-01-02 | CompressionRepository follows ABC+SQLite pattern | Consistent with CommitRepository, BlobRepository, etc. |
| 04-01-03 | PendingCompression is mutable (not frozen) | Users need to edit summaries before approving |
| 04-01-04 | v1->v2->v3 migration chain | v1 databases auto-upgrade through both steps |

## Next Phase Readiness

**Ready for 04-02 (Compression Engine):**
- All storage tables and repository methods are in place
- Domain models (CompressResult, PendingCompression) define the API contract
- Summarization prompt module provides LLM integration content
- CompressionError exception ready for error paths
- SqliteCompressionRepository NOT yet wired into Tract.open() (deferred to 04-02)
