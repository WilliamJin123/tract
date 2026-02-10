# Phase 1: Foundations - Research

**Researched:** 2026-02-10
**Domain:** Python library internals -- data model, storage, commit/materialize cycle, token accounting, SDK entry point
**Confidence:** HIGH (primary patterns verified with official docs; implementation details cross-referenced)

## Summary

Phase 1 builds the load-bearing foundation of Trace: the data model (SQLAlchemy ORM), content-addressable storage, the commit engine with six built-in content types, a priority annotation system, a materializer that produces structured message lists, token accounting via tiktoken, and the public SDK entry point (`Repo.open()`). The project-level research (STACK.md, ARCHITECTURE.md, PITFALLS.md) already locked major technology choices. This phase-level research focuses on **implementation specifics** -- exact SQLAlchemy 2.0 patterns for the models, the Pydantic v2 discriminated union approach for content types, the TypeDecorator bridge for storing Pydantic models in JSON columns, tiktoken integration details, the materialization algorithm, the priority annotation storage model, deterministic hashing for content-addressable commits, and testing patterns.

The core challenge of Phase 1 is getting the data model right. Every subsequent phase (branching, compression, multi-agent) depends on the commit model, content type system, and storage schema. The content type system must be extensible (users register custom types) while enforcing schemas for built-in types. The priority annotation system must be lightweight (like git tags, not commits) while maintaining full provenance history. The materializer must handle edit resolution, time-travel, priority filtering, and type-to-role mapping while remaining pluggable.

**Primary recommendation:** Build storage layer and data model first, then commit engine with content types, then materialization, then token accounting, then the SDK surface. Each layer should be independently testable via the repository pattern.

## Standard Stack

Phase 1 uses a subset of the project-wide stack. CLI (Click, Rich) and HTTP (httpx) are NOT needed in Phase 1.

### Core (Phase 1 Dependencies)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| SQLAlchemy | >=2.0.46, <2.2 | ORM for storage layer | 2.0-style DeclarativeBase with Mapped[], mapped_column(). Full type safety. |
| Pydantic | >=2.10, <3.0 | SDK-facing types, content schemas, validation | Discriminated unions for content type system. model_dump()/model_validate() for JSON serialization. |
| tiktoken | >=0.12.0 | Default BPE tokenizer | Rust-backed, covers OpenAI models (cl100k_base, o200k_base). encoding_for_model() with built-in caching. |
| typing-extensions | >=4.12 | Backported typing features | Required for Protocol on Python 3.10. Pydantic depends on it anyway. |

### Supporting (Phase 1 Dev Dependencies)

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| pytest | >=8.0 | Test runner | All tests |
| hypothesis | >=6.150 | Property-based testing | DAG invariants, hash collision resistance, content type validation |
| pytest-cov | >=7.0 | Coverage | CI gate at 80%+ coverage |
| ruff | >=0.15 | Lint + format | Pre-commit and CI |
| mypy | >=1.14 | Type checking | CI with SQLAlchemy plugin enabled |

### Not Needed in Phase 1

| Library | Why Not Yet |
|---------|-------------|
| Click, Rich, rich-click | CLI is Phase 2 |
| httpx | LLM client is Phase 3 |
| aiosqlite | Sync-first in Phase 1; async is v2 |
| pytest-asyncio | No async code in Phase 1 |

**Installation (Phase 1 only):**
```bash
uv add sqlalchemy pydantic tiktoken typing-extensions
uv add --dev pytest hypothesis pytest-cov ruff mypy
```

## Architecture Patterns

### Recommended Project Structure (Phase 1)

```
src/trace/
    __init__.py              # Public API: Repo, ContentType, etc.
    _version.py              # Version string

    models/                  # Pydantic models (SDK-facing types)
        __init__.py
        content.py           # Content type system (discriminated union)
        commit.py            # CommitInfo, CommitOperation enum
        annotations.py       # PriorityAnnotation model
        materialized.py      # MaterializedContext, Message
        config.py            # RepoConfig, TokenBudgetConfig

    storage/                 # SQLAlchemy models + repository pattern
        __init__.py
        schema.py            # ORM models (Base, CommitRow, BlobRow, RefRow, AnnotationRow)
        types.py             # Custom TypeDecorator for Pydantic<->JSON bridge
        repositories.py      # Abstract repository interfaces
        sqlite.py            # SQLite implementation of repositories
        engine.py            # Engine/session factory, schema init

    engine/                  # Core business logic (no SQLAlchemy imports)
        __init__.py
        commit.py            # Commit creation, hashing, validation
        materialize.py       # Materialization algorithm
        tokens.py            # Token counting protocol + tiktoken impl

    protocols.py             # All Protocol definitions in one place
    exceptions.py            # Custom exception hierarchy
    repo.py                  # Repo class (public SDK entry point)

tests/
    conftest.py              # Shared fixtures (in-memory DB, sample commits)
    test_storage/
        test_schema.py
        test_repositories.py
    test_engine/
        test_commit.py
        test_materialize.py
        test_tokens.py
    test_models/
        test_content.py
        test_annotations.py
    test_repo.py             # Integration tests via public API
    strategies.py            # Hypothesis custom strategies
```

### Pattern 1: Pydantic TypeDecorator Bridge (Storage <-> Domain)

**What:** A custom SQLAlchemy `TypeDecorator` that transparently converts Pydantic models to/from JSON columns in SQLite.

**When to use:** Any SQLAlchemy column that stores structured data with a Pydantic schema (content payloads, metadata dicts, annotation records).

**Why:** Keeps the domain layer pure Pydantic while the storage layer handles serialization automatically. No manual model_dump()/model_validate() calls at every DB interaction.

**Example:**
```python
# Source: SQLAlchemy TypeDecorator docs + community pattern
# https://docs.sqlalchemy.org/en/20/core/custom_types.html

import json
from typing import Any, Type
from pydantic import BaseModel
from sqlalchemy import JSON, TypeDecorator, Dialect

class PydanticJSON(TypeDecorator):
    """Store Pydantic models as JSON in SQLite."""
    impl = JSON
    cache_ok = True

    def __init__(self, pydantic_type: Type[BaseModel]) -> None:
        super().__init__()
        self.pydantic_type = pydantic_type

    def process_bind_param(self, value: BaseModel | None, dialect: Dialect) -> dict | None:
        if value is None:
            return None
        return value.model_dump(mode="json")

    def process_result_value(self, value: dict | str | None, dialect: Dialect) -> BaseModel | None:
        if value is None:
            return None
        if isinstance(value, str):
            return self.pydantic_type.model_validate_json(value)
        return self.pydantic_type.model_validate(value)

    def coerce_compared_value(self, op, value):
        return self.impl.coerce_compared_value(op, value)
```

**Usage in ORM model:**
```python
from sqlalchemy.orm import Mapped, mapped_column

class CommitRow(Base):
    __tablename__ = "commits"
    content_payload: Mapped[dict] = mapped_column(
        PydanticJSON(ContentPayload), nullable=False
    )
```

**Confidence:** HIGH -- TypeDecorator is official SQLAlchemy API; Pydantic bridge pattern verified across multiple sources.

### Pattern 2: Discriminated Union for Content Types

**What:** Use Pydantic v2 discriminated unions with a `Literal` type field to validate content payloads against the correct schema based on content type.

**When to use:** Validating commit content on ingestion. Each of the six built-in types (instruction, dialogue, tool_io, reasoning, artifact, output) plus the freeform escape hatch has its own schema.

**Example:**
```python
# Source: https://docs.pydantic.dev/latest/concepts/unions/

from typing import Annotated, Literal, Union
from pydantic import BaseModel, Field

class InstructionContent(BaseModel):
    content_type: Literal["instruction"] = "instruction"
    text: str

class DialogueContent(BaseModel):
    content_type: Literal["dialogue"] = "dialogue"
    role: Literal["user", "assistant", "system"]
    text: str
    name: str | None = None

class ToolIOContent(BaseModel):
    content_type: Literal["tool_io"] = "tool_io"
    tool_name: str
    direction: Literal["call", "result"]
    payload: dict  # tool input args or output
    status: Literal["success", "error"] | None = None

class ReasoningContent(BaseModel):
    content_type: Literal["reasoning"] = "reasoning"
    text: str

class ArtifactContent(BaseModel):
    content_type: Literal["artifact"] = "artifact"
    artifact_type: str  # e.g., "code", "document", "config"
    content: str
    language: str | None = None  # for code artifacts

class OutputContent(BaseModel):
    content_type: Literal["output"] = "output"
    text: str
    format: Literal["text", "markdown", "json"] = "text"

class FreeformContent(BaseModel):
    content_type: Literal["freeform"] = "freeform"
    payload: dict  # No schema enforcement

# The discriminated union:
ContentPayload = Annotated[
    Union[
        InstructionContent,
        DialogueContent,
        ToolIOContent,
        ReasoningContent,
        ArtifactContent,
        OutputContent,
        FreeformContent,
    ],
    Field(discriminator="content_type"),
]
```

**Registry for custom types:** Custom user-defined types cannot be added to the discriminated union at import time. Instead, use a registry dict that maps type name to Pydantic model class, and validate custom types via `TypeAdapter` lookup at runtime:

```python
from pydantic import TypeAdapter

_custom_type_registry: dict[str, type[BaseModel]] = {}

def register_content_type(name: str, model: type[BaseModel]) -> None:
    """Register a custom content type with optional schema."""
    _custom_type_registry[name] = model

def validate_content(data: dict) -> BaseModel:
    """Validate content against built-in or custom type schema."""
    content_type = data.get("content_type", "freeform")
    if content_type in _custom_type_registry:
        adapter = TypeAdapter(_custom_type_registry[content_type])
        return adapter.validate_python(data)
    # Fall through to built-in discriminated union
    adapter = TypeAdapter(ContentPayload)
    return adapter.validate_python(data)
```

**Confidence:** HIGH -- Pydantic v2 discriminated unions are official, well-documented API. Registry pattern is standard Python.

### Pattern 3: Content-Addressable Hashing (Deterministic)

**What:** SHA-256 hash of canonical JSON representation for both content blobs and commits.

**When to use:** Computing commit_hash and content_hash for content-addressable storage.

**Critical requirement:** The hash must be deterministic -- same data always produces the same hash. This requires canonical JSON serialization.

**Example:**
```python
# Source: hashlib stdlib docs + deterministic JSON pattern
# https://death.andgravity.com/stable-hashing

import hashlib
import json
from typing import Any

def canonical_json(data: Any) -> bytes:
    """Deterministic JSON serialization for content-addressable hashing.

    Rules:
    - Keys sorted alphabetically
    - No whitespace (compact separators)
    - ensure_ascii=False for consistent Unicode handling
    - Pydantic models converted via model_dump(mode='json') first
    """
    return json.dumps(
        data,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def content_hash(payload: dict) -> str:
    """Hash content payload for blob deduplication."""
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def commit_hash(
    content_hash: str,
    parent_hash: str | None,
    content_type: str,
    operation: str,
    timestamp_iso: str,
    reply_to: str | None = None,
) -> str:
    """Hash commit for unique identification.

    Includes content + structural context so identical content
    at different points in history gets different commit hashes.
    """
    data = {
        "content_hash": content_hash,
        "parent_hash": parent_hash,
        "content_type": content_type,
        "operation": operation,
        "timestamp": timestamp_iso,
    }
    if reply_to is not None:
        data["reply_to"] = reply_to
    return hashlib.sha256(canonical_json(data)).hexdigest()
```

**Confidence:** HIGH -- hashlib is stdlib, canonical JSON pattern is well-established (RFC 8785 JCS is the formal standard, but sort_keys + compact separators is sufficient for this use case).

### Pattern 4: Repository Pattern for Storage

**What:** Abstract all database access behind repository interfaces. Engine/domain code uses the interface, never SQLAlchemy sessions directly.

**When to use:** All storage operations. This enables testing with in-memory SQLite and future backend swaps.

**Example:**
```python
from abc import ABC, abstractmethod
from typing import Sequence

class CommitRepository(ABC):
    @abstractmethod
    def get(self, commit_hash: str) -> "CommitRow | None": ...

    @abstractmethod
    def save(self, commit: "CommitRow") -> None: ...

    @abstractmethod
    def get_ancestors(self, commit_hash: str, limit: int | None = None) -> Sequence["CommitRow"]: ...

    @abstractmethod
    def get_by_type(self, content_type: str, repo_id: str) -> Sequence["CommitRow"]: ...

class BlobRepository(ABC):
    @abstractmethod
    def get(self, content_hash: str) -> "BlobRow | None": ...

    @abstractmethod
    def save_if_absent(self, blob: "BlobRow") -> None:
        """Content-addressable: only store if hash not already present."""
        ...

class RefRepository(ABC):
    @abstractmethod
    def get_head(self, repo_id: str) -> str | None: ...

    @abstractmethod
    def update_head(self, repo_id: str, commit_hash: str) -> None: ...

    @abstractmethod
    def get_branch(self, repo_id: str, branch_name: str) -> str | None: ...

class AnnotationRepository(ABC):
    @abstractmethod
    def get_latest(self, target_hash: str) -> "AnnotationRow | None": ...

    @abstractmethod
    def save(self, annotation: "AnnotationRow") -> None: ...

    @abstractmethod
    def get_history(self, target_hash: str) -> Sequence["AnnotationRow"]: ...
```

**Confidence:** HIGH -- standard architecture pattern, recommended by SQLAlchemy documentation and project-level ARCHITECTURE.md.

### Pattern 5: Materialization Algorithm

**What:** Convert the commit chain (root to HEAD) into a structured message list `[{role, content}, ...]` ready for LLM APIs.

**Algorithm (default materializer):**

```
materialize(head_hash, as_of=None):
    1. Walk commits from root to HEAD (or up to as_of timestamp/hash)
    2. Build edit resolution map:
       - For each commit with operation=EDIT, map: {reply_to_hash -> edit_commit}
       - If multiple edits target same commit, latest edit wins
    3. Build priority map:
       - For each commit, get latest annotation (or use content type default)
       - Skip commits with priority=SKIP
    4. Build effective commit list:
       - For each non-EDIT, non-DELETE commit in order:
         a. If it is an edit target, substitute the edit's content
         b. If it has priority=SKIP, exclude
         c. Otherwise include
    5. Map content types to roles using type_to_role_mapping:
       - instruction -> system
       - dialogue -> user/assistant (from role field)
       - tool_io -> tool (call) / tool (result)
       - reasoning -> assistant (or custom)
       - artifact -> assistant
       - output -> assistant
    6. Aggregate consecutive same-role messages (optional, configurable)
    7. Return MaterializedContext(messages=List[Message], token_count=int)
```

**Confidence:** HIGH for the algorithm structure. MEDIUM for the specific aggregation rules (may need tuning during implementation).

### Anti-Patterns to Avoid

- **Storing materialized context as source of truth:** Always derive from commits. Cache keyed by HEAD hash, invalidate on any mutation.
- **Mixing SQLAlchemy session logic into domain code:** Domain code (engine/) must never import from sqlalchemy. Only storage/ touches the ORM.
- **Using Pydantic BaseModel for SQLAlchemy ORM classes:** Do NOT use SQLModel or MappedAsDataclass for ORM classes. Keep them separate: SQLAlchemy models in storage/schema.py, Pydantic models in models/. Bridge via TypeDecorator for JSON columns and explicit conversion functions for the rest.
- **Mutable content hashes:** Never mutate a blob or commit after storing. Edits create new commits. This is the core immutability invariant.
- **Global/singleton Repo:** Each Repo is an explicit instance. No global state.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| JSON serialization/validation for content types | Custom dict validation with isinstance checks | Pydantic v2 discriminated unions with `Field(discriminator=...)` | Pydantic handles type coercion, error messages, nested validation, and JSON Schema generation. Hand-rolling this for 7+ types is bug-prone. |
| SHA-256 hashing | Custom hash implementation | `hashlib.sha256()` from stdlib | Stdlib, C-backed, no dependency. |
| BPE token counting | Regex-based tokenizer approximation | tiktoken `encoding_for_model()` | Rust-backed, exact tokenizer for OpenAI models, caches encoding files locally. |
| JSON column type for SQLAlchemy | Manual json.dumps/loads in every query | `TypeDecorator` with `impl = JSON` wrapping Pydantic model_dump/model_validate | One-time setup, transparent throughout codebase. |
| Canonical JSON for hashing | Custom serialization | `json.dumps(sort_keys=True, separators=(',',':'), ensure_ascii=False)` | Stdlib, well-understood, deterministic. |
| Content type schema validation | Manual field checking per type | Pydantic BaseModel per content type with `model_validate()` | Type-safe, generates error messages, supports optional fields and defaults. |
| Enum storage in SQLite | String constants with manual validation | Python `enum.Enum` + SQLAlchemy `Mapped[MyEnum]` | SQLAlchemy natively maps Python Enums to VARCHAR. Type-safe at both Python and SQL level. |

**Key insight:** Phase 1 has many moving parts (6 content types, 3 operations, 3 priority levels, annotations, edit resolution, time-travel) but each individual piece has a well-established pattern. The complexity is in the interactions, not the individual components. Use standard tools for each piece; invest implementation effort in the integration logic (materialization algorithm, commit engine workflow).

## Common Pitfalls

### Pitfall 1: Non-Deterministic Commit Hashes

**What goes wrong:** Using `datetime.now()` or Python's default dict iteration order in hash computation produces different hashes for logically identical commits across runs or platforms.

**Why it happens:** Timestamps have platform-dependent precision. Dict ordering in Python 3.7+ is insertion-order-dependent, which is deterministic within a single program but not across serialization boundaries. Floating-point representations in JSON can differ.

**How to avoid:**
- Always use `json.dumps(sort_keys=True, separators=(',',':'))` for hash input
- Pass timestamps as ISO 8601 strings with fixed precision (microseconds)
- Never include mutable or non-deterministic data (Python object ids, random values) in hash input
- Write a property test: `forall(commit_data) => hash(serialize(data)) == hash(serialize(data))`

**Warning signs:** Same content producing different hashes on re-commit. Content deduplication failing.

### Pitfall 2: Pydantic v2 Discriminated Union Gotchas

**What goes wrong:** Discriminated union validation fails unexpectedly, or the wrong variant is selected.

**Why it happens:** Pydantic v2 discriminated unions require the discriminator field to be a `Literal` type with a default value. If the field is missing from input data, or if two variants share the same Literal value, validation fails or selects the wrong variant.

**How to avoid:**
- Every content type model MUST have `content_type: Literal["typename"] = "typename"` as its first field
- Test that each Literal value is unique across all union members
- For custom types (not in the union), use the registry fallback pattern, not dynamic union modification
- Test validation with both dict input and model instance input (Pydantic handles both)

**Warning signs:** `ValidationError` mentioning "unable to discriminate" or "no match for discriminator."

### Pitfall 3: SQLAlchemy JSON Column Mutation Tracking

**What goes wrong:** Modifying a Pydantic model stored in a JSON column does not trigger SQLAlchemy's change detection. The `UPDATE` is never issued.

**Why it happens:** SQLAlchemy tracks mutations on scalar columns automatically, but for JSON/dict columns it uses shallow equality checks. If you mutate the dict in-place (e.g., `row.metadata["key"] = "value"`), SQLAlchemy may not detect the change.

**How to avoid:**
- Use `MutableDict.as_mutable()` or `MutableList.as_mutable()` from `sqlalchemy.ext.mutable` for any JSON column that might be updated in-place
- Better: treat stored Pydantic models as immutable. To change, create a new model instance and assign it to the column (triggers change detection)
- For the metadata dict on commits, use `MutableDict` since it is an open dict that users might modify

**Warning signs:** Metadata changes not persisting after session commit. "Stale" data on re-read.

### Pitfall 4: Token Count Disagreements

**What goes wrong:** Trace's tiktoken-based counts diverge from the LLM provider's actual consumption. Users hit context window limits unexpectedly.

**Why it happens:** tiktoken is only accurate for OpenAI models. Even within OpenAI, chat message formatting adds overhead (3 tokens per message, 3 tokens for response primer) that raw text counting misses. Different models use different encodings (cl100k_base vs o200k_base).

**How to avoid:**
- Default encoding should be `o200k_base` (covers GPT-4o, o1, o3 -- the current generation)
- Expose the encoding name in config: `RepoConfig(tokenizer_encoding="o200k_base")`
- The token count stored per-commit is the RAW CONTENT count (no message overhead). Message overhead is computed at materialization time by the materializer
- Clearly label counts as "estimated (tiktoken)" vs "provider-reported" in the returned `MaterializedContext`
- Design the `TokenCounter` protocol to accept both `str` and `list[dict]` (messages) inputs

**Warning signs:** Token budget violations despite Trace reporting under-budget. Users reporting 5-15% discrepancy from provider invoices.

### Pitfall 5: Annotation Table Growing Unbounded

**What goes wrong:** Every priority change creates a new annotation row. Over a long trace with frequent pin/unpin operations, the annotation table grows large and queries for "current priority of commit X" become slow.

**Why it happens:** The design stores annotations as an append-only history (for provenance). Without an index or caching strategy, finding the latest annotation for a commit requires scanning all annotations for that target_hash.

**How to avoid:**
- Index: `CREATE INDEX ix_annotations_target ON annotations(target_hash, created_at DESC)`
- Query pattern: `SELECT ... WHERE target_hash = ? ORDER BY created_at DESC LIMIT 1`
- Consider a denormalized `current_priority` column on the commit table that is updated whenever a new annotation is created (write-time optimization for read-heavy materialization)
- Materialization should batch-fetch all relevant annotations in one query, not N+1

**Warning signs:** Materialization slowing down as trace grows. N+1 query patterns in SQLAlchemy logs.

### Pitfall 6: Edit Chain Cycles

**What goes wrong:** An edit targets another edit, which targets another edit, creating a chain that the materializer must resolve. Worse: a cycle where edit A targets B and edit B targets A.

**Why it happens:** The `reply_to` field on edit commits is an arbitrary hash. Without validation, a user could create circular edit references.

**How to avoid:**
- On commit with operation=EDIT: validate that `reply_to` points to an existing commit
- Validate that `reply_to` does NOT point to another EDIT commit (edits target originals, not other edits). If you want to "re-edit," the new edit targets the same original
- Alternatively, if chained edits are desired, resolve them at materialization time by following the chain to find the latest edit (but detect cycles with a visited set)
- Add a constraint: edit commits MUST have `reply_to` set; append/delete commits MAY have it set

**Warning signs:** Infinite loop in materialization. Stack overflow during edit resolution.

## Code Examples

### SQLAlchemy ORM Models (Phase 1 Schema)

```python
# storage/schema.py
# Source: SQLAlchemy 2.0 DeclarativeBase docs
# https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html

import enum
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON, DateTime, ForeignKey, Index, Integer, String, Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class CommitOperation(enum.Enum):
    APPEND = "append"
    EDIT = "edit"
    DELETE = "delete"


class Priority(enum.Enum):
    SKIP = "skip"
    NORMAL = "normal"
    PINNED = "pinned"


class BlobRow(Base):
    """Content-addressable blob storage. Keyed by SHA-256 of content."""
    __tablename__ = "blobs"

    content_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)  # Canonical JSON
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class CommitRow(Base):
    """A commit in the context DAG."""
    __tablename__ = "commits"

    commit_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    repo_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    parent_hash: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), nullable=True
    )
    content_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("blobs.content_hash"), nullable=False
    )
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)  # "instruction", "dialogue", etc.
    operation: Mapped[CommitOperation] = mapped_column(nullable=False)
    reply_to: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), nullable=True
    )
    message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)  # Denormalized from blob
    cumulative_tokens: Mapped[int] = mapped_column(Integer, nullable=False)  # Running total
    metadata_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)  # Open metadata
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    # Relationships
    blob: Mapped["BlobRow"] = relationship("BlobRow", lazy="select")
    parent: Mapped[Optional["CommitRow"]] = relationship(
        "CommitRow", remote_side="CommitRow.commit_hash",
        foreign_keys=[parent_hash],
    )

    __table_args__ = (
        Index("ix_commits_repo_time", "repo_id", "created_at"),
        Index("ix_commits_repo_type", "repo_id", "content_type"),
        Index("ix_commits_reply_to", "reply_to"),
    )


class RefRow(Base):
    """Mutable named pointer to a commit (branch, HEAD)."""
    __tablename__ = "refs"

    repo_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    ref_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    commit_hash: Mapped[Optional[str]] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), nullable=True
    )
    symbolic_target: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)


class AnnotationRow(Base):
    """Lightweight priority annotation (like git tags).

    Append-only: each change creates a new row for provenance.
    The latest row for a given target_hash is the current annotation.
    """
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repo_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    target_hash: Mapped[str] = mapped_column(
        String(64), ForeignKey("commits.commit_hash"), nullable=False
    )
    priority: Mapped[Priority] = mapped_column(nullable=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)

    __table_args__ = (
        Index("ix_annotations_target_time", "target_hash", "created_at"),
    )
```

**Design notes on the schema:**

1. **Single parent_hash instead of CommitParent association table:** Phase 1 is linear history only (no merge commits with multiple parents). The association table from ARCHITECTURE.md is for Phase 3 (branching/merging). In Phase 1, a simple nullable `parent_hash` FK is sufficient and simpler. Phase 3 will add the `commit_parents` association table for multi-parent support.

2. **content_type as String, not Enum:** Stored as a string rather than a DB enum to support user-registered custom types without schema migration. Validation happens at the Pydantic layer, not the DB layer.

3. **cumulative_tokens on CommitRow:** The running total of tokens from root to this commit. Computed at commit time as `parent.cumulative_tokens + this.token_count`. Enables O(1) budget checks.

4. **Separate AnnotationRow table:** Annotations are NOT part of the commit itself. They are lightweight, mutable metadata (like git tags). A commit's priority can change without modifying the commit. The annotation table is append-only for provenance (who changed priority, when, why).

5. **metadata_json as plain JSON column:** No Pydantic validation on commit metadata in Phase 1 (per CONTEXT.md: "no schema enforcement in Phase 1"). Use `MutableDict.as_mutable(JSON)` if in-place mutation is needed.

### Token Counter Protocol and Implementation

```python
# engine/tokens.py
from typing import Protocol, runtime_checkable

@runtime_checkable
class TokenCounter(Protocol):
    """Protocol for pluggable token counting."""

    def count_text(self, text: str) -> int:
        """Count tokens in a plain text string."""
        ...

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens in a structured message list (including overhead)."""
        ...


class TiktokenCounter:
    """Default token counter using tiktoken (OpenAI models)."""

    def __init__(self, model: str = "gpt-4o", encoding_name: str | None = None):
        import tiktoken
        if encoding_name:
            self._enc = tiktoken.get_encoding(encoding_name)
        else:
            try:
                self._enc = tiktoken.encoding_for_model(model)
            except KeyError:
                self._enc = tiktoken.get_encoding("o200k_base")
        self._model = model

    def count_text(self, text: str) -> int:
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        """Count tokens including chat message overhead.

        Based on OpenAI cookbook:
        https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken/
        """
        tokens_per_message = 3  # Standard for GPT-4o and similar
        tokens_per_name = 1

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                if isinstance(value, str):
                    num_tokens += len(self._enc.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # Assistant response primer
        return num_tokens
```

**tiktoken caching note:** tiktoken downloads encoding data on first use and caches in a temp directory. Set `TIKTOKEN_CACHE_DIR` for persistence across reboots. The cache is ~10MB for all encodings. The `encoding_for_model()` call itself is near-instant after first download. Cache the `Encoding` object instance (as shown above in `self._enc`) -- do NOT re-create it per call.

**Confidence:** HIGH -- based on tiktoken docs and OpenAI cookbook, verified.

### Content Type Behavioral Hints

```python
# models/content.py (partial -- behavioral hints registry)

from dataclasses import dataclass
from typing import Callable

@dataclass(frozen=True)
class ContentTypeHints:
    """Default behavioral hints for a content type."""
    default_priority: Priority = Priority.NORMAL
    materialization_order: int = 50  # Lower = earlier in output. 0=system, 50=middle, 100=end
    default_role: str = "assistant"  # Default LLM message role
    compression_priority: int = 50  # Lower = compress first (0=compress first, 100=protect)
    aggregation_rule: str = "concatenate"  # How same-type commits combine

# Built-in type hints registry
BUILTIN_TYPE_HINTS: dict[str, ContentTypeHints] = {
    "instruction": ContentTypeHints(
        default_priority=Priority.PINNED,
        materialization_order=0,  # System messages first
        default_role="system",
        compression_priority=90,  # Resist compression
    ),
    "dialogue": ContentTypeHints(
        default_priority=Priority.NORMAL,
        materialization_order=50,
        default_role="user",  # Overridden by role field in content
        compression_priority=50,
    ),
    "tool_io": ContentTypeHints(
        default_priority=Priority.NORMAL,
        materialization_order=60,
        default_role="tool",
        compression_priority=30,  # Compress aggressively (boilerplate)
    ),
    "reasoning": ContentTypeHints(
        default_priority=Priority.NORMAL,
        materialization_order=70,
        default_role="assistant",
        compression_priority=40,
    ),
    "artifact": ContentTypeHints(
        default_priority=Priority.NORMAL,
        materialization_order=80,
        default_role="assistant",
        compression_priority=60,  # Moderate protection
    ),
    "output": ContentTypeHints(
        default_priority=Priority.NORMAL,
        materialization_order=90,
        default_role="assistant",
        compression_priority=70,  # Protect outputs
    ),
}
```

### Materializer Protocol and Default Implementation

```python
# protocols.py (partial)

from typing import Protocol
from dataclasses import dataclass

@dataclass(frozen=True)
class Message:
    role: str
    content: str

@dataclass(frozen=True)
class MaterializedContext:
    messages: list[Message]
    token_count: int         # Total tokens in materialized output
    commit_count: int        # Number of commits included
    token_source: str        # "tiktoken:o200k_base" or "provider:gpt-4o"

class Materializer(Protocol):
    def materialize(
        self,
        commits: list,  # Ordered list of commit data
        *,
        as_of: str | None = None,
        include_edit_annotations: bool = False,
    ) -> MaterializedContext: ...
```

### Token Budget Configuration

```python
# models/config.py

from enum import Enum
from typing import Callable, Optional
from pydantic import BaseModel

class BudgetAction(str, Enum):
    WARN = "warn"
    REJECT = "reject"
    CALLBACK = "callback"

class TokenBudgetConfig(BaseModel):
    max_tokens: Optional[int] = None  # None = unlimited
    action: BudgetAction = BudgetAction.WARN
    callback: Optional[Callable[[int, int], None]] = None  # (current, max) -> None

    model_config = {"arbitrary_types_allowed": True}
```

### SQLite Engine Initialization

```python
# storage/engine.py

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from .schema import Base

def create_trace_engine(db_path: str = ":memory:"):
    """Create SQLAlchemy engine with SQLite optimizations."""
    url = f"sqlite:///{db_path}" if db_path != ":memory:" else "sqlite://"
    engine = create_engine(url, echo=False)

    # SQLite pragmas for performance and safety
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables
    Base.metadata.create_all(engine)

    return engine

def create_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)
```

**Note:** `expire_on_commit=False` is important -- prevents lazy-load issues when accessing attributes after commit, especially relevant if we add async later.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `declarative_base()` function | `class Base(DeclarativeBase)` | SQLAlchemy 2.0 (2023) | All ORM models must use new style |
| `Column()` without type hints | `Mapped[T] = mapped_column()` | SQLAlchemy 2.0 (2023) | Full type inference, mypy support |
| `Pydantic v1 .dict()` | `Pydantic v2 .model_dump(mode='json')` | Pydantic 2.0 (2023) | 5-50x faster, mode='json' for JSON-safe output |
| Pydantic v1 Hypothesis plugin | No built-in Hypothesis plugin in v2 | Pydantic 2.0 (2023) | Use `st.builds()` with explicit strategies instead |
| tiktoken `cl100k_base` default | tiktoken `o200k_base` for current models | GPT-4o launch (2024) | Must use correct encoding for model |
| `asyncio_mode = "strict"` | `asyncio_mode = "auto"` | pytest-asyncio 1.0 (2025) | Simpler async test configuration |

**Deprecated/outdated:**
- `declarative_base()`: Use `DeclarativeBase` class inheritance instead
- `Column()` without `Mapped[]`: Loses type safety
- Pydantic `.dict()`, `.parse_obj()`: Use `.model_dump()`, `.model_validate()`
- `session.query(Model)`: Use `select(Model)` with `session.execute()` for 2.0-style queries
- tiktoken assuming cl100k_base: Current-gen models (GPT-4o, o1, o3) use o200k_base

## Testing Patterns for Phase 1

### Fixture Strategy

```python
# tests/conftest.py

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from trace.storage.schema import Base
from trace.storage.engine import create_trace_engine

@pytest.fixture
def engine():
    """In-memory SQLite engine for fast tests."""
    eng = create_trace_engine(":memory:")
    yield eng
    eng.dispose()

@pytest.fixture
def session(engine):
    """Session with automatic rollback after each test."""
    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    sess = SessionLocal()
    yield sess
    sess.rollback()
    sess.close()

@pytest.fixture
def sample_repo_id():
    return "test-repo-001"
```

### Hypothesis Strategies

```python
# tests/strategies.py

from hypothesis import strategies as st
from trace.models.content import (
    InstructionContent, DialogueContent, ToolIOContent,
    ReasoningContent, ArtifactContent, OutputContent,
)

# Text that is valid for content (non-empty, reasonable size)
content_text = st.text(min_size=1, max_size=5000, alphabet=st.characters(
    whitelist_categories=("L", "N", "P", "Z", "S"),
))

instruction_content = st.builds(
    InstructionContent,
    text=content_text,
)

dialogue_content = st.builds(
    DialogueContent,
    role=st.sampled_from(["user", "assistant", "system"]),
    text=content_text,
    name=st.one_of(st.none(), st.text(min_size=1, max_size=50)),
)

tool_io_content = st.builds(
    ToolIOContent,
    tool_name=st.text(min_size=1, max_size=100),
    direction=st.sampled_from(["call", "result"]),
    payload=st.dictionaries(
        keys=st.text(min_size=1, max_size=20),
        values=st.text(max_size=200),
        max_size=10,
    ),
    status=st.one_of(st.none(), st.sampled_from(["success", "error"])),
)

any_content = st.one_of(
    instruction_content,
    dialogue_content,
    tool_io_content,
    st.builds(ReasoningContent, text=content_text),
    st.builds(ArtifactContent, artifact_type=st.sampled_from(["code", "document"]), content=content_text),
    st.builds(OutputContent, text=content_text),
)
```

**Note on Pydantic v2 + Hypothesis:** Pydantic v2 dropped its built-in Hypothesis plugin. Use `st.builds()` with explicit field strategies instead. This actually gives more control over test data generation. Do NOT use `st.from_type()` with Pydantic v2 models -- it will not work without the plugin.

### Property Tests to Write

| Property | What It Verifies | Priority |
|----------|-----------------|----------|
| `hash(content) == hash(content)` | Deterministic hashing | P0 |
| `hash(A) != hash(B)` for `A != B` (probabilistic) | Collision resistance | P0 |
| `materialize(commits).token_count >= 0` | Token counts non-negative | P0 |
| `materialize(commits_with_edit).messages` does not contain original content | Edit resolution works | P0 |
| `materialize(commits_with_skip).messages` does not include skipped commits | Priority filtering works | P0 |
| `cumulative_tokens == sum(all ancestor token_counts)` | Running total is correct | P1 |
| `content_type validates for all built-in types` | Schema validation | P1 |
| `round_trip(pydantic_model) == pydantic_model` | TypeDecorator serialization | P1 |

## Open Questions

Things that could not be fully resolved during research:

1. **Aggregation rules for same-type message combining**
   - What we know: The CONTEXT.md says "Multiple commits of the same type aggregate into combined messages." The default materializer should support this.
   - What's unclear: Exact aggregation behavior for each type. Do consecutive `instruction` commits merge into one system message? Or only when they are adjacent in the commit chain?
   - Recommendation: Start with "adjacent same-role messages concatenate with double newline." Make it configurable via the materializer. Defer complex aggregation to later phases.

2. **API-reported token count extraction hook**
   - What we know: CONTEXT.md says "Callback/hook for extracting token usage from API responses, with default hooks for common providers."
   - What's unclear: In Phase 1 there is no LLM client yet (Phase 3). How should the hook be designed before there are API calls to extract from?
   - Recommendation: Define the `TokenUsageExtractor` protocol in Phase 1 but do NOT implement provider-specific hooks until Phase 3 when httpx is added. The protocol should accept a generic dict (API response) and return `{prompt_tokens: int, completion_tokens: int}`.

3. **Freeform content type and custom type serialization**
   - What we know: Freeform accepts any dict payload. Custom types optionally define schemas.
   - What's unclear: Should freeform payloads be stored as-is in the JSON blob, or should they be wrapped in a standard envelope?
   - Recommendation: All content is stored in a standard envelope: `{content_type: str, ...fields}`. Freeform just has `{content_type: "freeform", payload: dict}`. Custom types follow the same envelope pattern. This keeps the blob schema uniform and enables type-based queries.

4. **Time-travel materialization with annotations**
   - What we know: "as of" a specific commit/timestamp. Only sees commits and edits up to that point.
   - What's unclear: Should annotations also be time-bounded? If a commit was pinned after the as_of point, should the time-travel view see it as normal priority?
   - Recommendation: Yes, annotations should be time-bounded. Query annotations with `created_at <= as_of_timestamp`. This preserves true time-travel semantics.

5. **Commit metadata and MutableDict**
   - What we know: Open metadata dict, no schema in Phase 1.
   - What's unclear: Will users mutate metadata after commit creation? If commits are immutable, metadata should be immutable too.
   - Recommendation: Metadata is set at commit time and NOT mutable after. This preserves immutability. If metadata needs to change, create a new commit (edit operation). Use plain `JSON` column without `MutableDict` since mutation is not supported.

## Sources

### Primary (HIGH confidence)
- [SQLAlchemy 2.0 Declarative Tables](https://docs.sqlalchemy.org/en/20/orm/declarative_tables.html) -- Mapped[], mapped_column(), Enum column patterns
- [SQLAlchemy 2.0 Custom Types / TypeDecorator](https://docs.sqlalchemy.org/en/20/core/custom_types.html) -- TypeDecorator API, cache_ok, process_bind_param/process_result_value
- [Pydantic v2 Unions / Discriminated Unions](https://docs.pydantic.dev/latest/concepts/unions/) -- Literal discriminator, Field(discriminator=...), callable discriminator, Tag
- [tiktoken GitHub / Caching Strategy](https://deepwiki.com/openai/tiktoken/5.1-caching-strategy-and-configuration) -- Three-level cache, TIKTOKEN_CACHE_DIR, atomic writes
- [tiktoken Encoding Registry](https://deepwiki.com/openai/tiktoken/2.3-encoding-registry-and-model-mapping) -- cl100k_base vs o200k_base, model-to-encoding mapping
- [OpenAI Cookbook: Token Counting](https://developers.openai.com/cookbook/examples/how_to_count_tokens_with_tiktoken/) -- num_tokens_from_messages, per-message overhead constants
- [hashlib Python stdlib docs](https://docs.python.org/3/library/hashlib.html) -- SHA-256 API
- [Pydantic v2 Hypothesis Integration](https://docs.pydantic.dev/latest/integrations/hypothesis/) -- Plugin dropped in v2, use st.builds() instead

### Secondary (MEDIUM confidence)
- [Pydantic in SQLAlchemy fields (Imankulov)](https://roman.pt/posts/pydantic-in-sqlalchemy-fields/) -- TypeDecorator pattern for Pydantic<->JSON bridge
- [Pydantic in SQLAlchemy fields (Gist)](https://gist.github.com/imankulov/4051b7805ad737ace7d8de3d3f934d6b) -- Full implementation code
- [Deterministic Hashing of Python Data Objects](https://death.andgravity.com/stable-hashing) -- Canonical JSON pattern, sort_keys, separators
- [Pydantic Discriminated Unions Deep Dive](https://blog.dataengineerthings.org/pydantic-for-experts-discriminated-unions-in-pydantic-v2-2d9ca965b22f) -- Expert patterns, fallback handling
- [SQLAlchemy Enum Discussion](https://github.com/sqlalchemy/sqlalchemy/discussions/10615) -- Mapped[Enum] patterns in 2.0
- [Token Counting Guide: tiktoken, Anthropic, Gemini](https://www.propelcode.ai/blog/token-counting-tiktoken-anthropic-gemini-guide-2025) -- Cross-provider tokenizer differences

### Tertiary (LOW confidence)
- [Pydantic v2 + SQLAlchemy + Alembic (Medium, Jan 2026)](https://asim-poptani.medium.com/pydantic-v2-sqlalchemy-alembic-the-proper-way-56aed7847b5b) -- Recent blog post on integration patterns
- [Registry Pattern with Decorators (Medium, Dec 2025)](https://medium.com/@tihomir.manushev/implementing-the-registry-pattern-with-decorators-in-python-de8daf4a452a) -- Plugin registry pattern

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all libraries verified via project-level research, versions confirmed
- Architecture (schema/models): HIGH -- SQLAlchemy 2.0 patterns verified against official docs, TypeDecorator pattern cross-verified
- Content type system: HIGH -- Pydantic v2 discriminated unions are official, well-documented API
- Priority annotations: MEDIUM -- design pattern is sound (append-only annotations table) but specific query patterns need validation during implementation
- Materialization algorithm: MEDIUM -- algorithm structure is clear from requirements, but aggregation rules and edge cases need implementation-time tuning
- Token counting: HIGH -- tiktoken API verified, encoding mapping confirmed, overhead constants from OpenAI cookbook
- Testing patterns: MEDIUM -- Hypothesis + Pydantic v2 lacks plugin (confirmed), st.builds() workaround is standard but not as well documented

**Research date:** 2026-02-10
**Valid until:** 2026-03-10 (stable domain -- SQLAlchemy 2.0 and Pydantic v2 are mature)
