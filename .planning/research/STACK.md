# Technology Stack

**Project:** Trace -- Git for Context
**Researched:** 2026-02-10
**Dimension:** Stack (Python library for git-like LLM context version control)

---

## Recommended Stack

### Python Version & Runtime

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Python | >=3.10, <3.14 | Runtime | 3.10 is the floor for SQLAlchemy 2.1 and modern typing features (ParamSpec, TypeAlias). 3.14 is still alpha. Target 3.11+ for users (performance improvements from 3.11 are substantial for IO-bound workloads). | HIGH |

**Rationale:** Python 3.10 gives us `match` statements and modern `typing` features including `ParamSpec` and `TypeAlias`. SQLAlchemy 2.1 (beta, targeting stable in 2026) requires Python 3.10+. The 3.11 runtime is 10-60% faster for common operations. Avoid requiring 3.12+ to maximize library adoption -- many production environments are still on 3.10/3.11.

### Project Tooling & Build System

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| uv | latest (>=0.5) | Package management, virtual environments, build | The 2026 standard for Python projects. 10-100x faster than pip. Handles venv creation, dependency resolution, locking, and building. Single tool replaces pip + pip-tools + virtualenv + build. Written in Rust by Astral (same team as ruff). | HIGH |
| pyproject.toml | PEP 621 | Project metadata & config | The only modern option. All tool configuration (ruff, pytest, mypy) goes here. No setup.py, no setup.cfg. | HIGH |
| hatchling | >=1.25 | Build backend | Lightweight, fast build backend. Default for `uv init --lib`. Simpler than setuptools, more mature than uv_build (still new). Good PEP 621 support. | MEDIUM |

**Rationale for hatchling over alternatives:**
- `setuptools`: Legacy, complex configuration, slower. Still works but no reason to choose it for a new project.
- `uv_build`: Astral's own build backend, very new (2025). Fast but less ecosystem testing. Consider switching to it once it matures. Flag for re-evaluation if uv_build reaches 1.0 during development.
- `flit-core`: Minimal but lacks some features (no dynamic version from `__init__.py` without plugins). Good but hatchling is more flexible.
- `pdm-backend`: Tied to PDM ecosystem. No advantage if using uv.

**Project structure:** Use `src/` layout (`src/trace/`) as scaffolded by `uv init --lib`. This ensures the library is properly isolated and import paths work correctly in both development and installed modes.

```bash
uv init --lib trace
# Creates: src/trace/__init__.py, pyproject.toml, README.md, .python-version
```

### Database & Storage

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| SQLAlchemy | >=2.0.46, <2.2 | ORM & database abstraction | Modern 2.0-style with `DeclarativeBase`, `Mapped[]`, `mapped_column()`. Full type annotation support. Async support via extension. Pin to 2.0.x stable for now; 2.1.0b1 is in beta. | HIGH |
| aiosqlite | >=0.22.0 | Async SQLite driver | Required for SQLAlchemy async with SQLite. Uses background thread (SQLite is not truly async) but provides asyncio-compatible interface. Version 0.22.0+ has architecture changes that SQLAlchemy 2.0.38+ handles correctly. | HIGH |
| sqlite3 | stdlib | Sync SQLite driver | Used by SQLAlchemy by default for synchronous operations. No extra dependency. Python's stdlib pysqlite is sufficient. | HIGH |

**SQLAlchemy patterns to use (2.0 style):**
```python
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy import String, Integer, ForeignKey, DateTime
from typing import Optional

class Base(DeclarativeBase):
    pass

class Commit(Base):
    __tablename__ = "commits"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA-256 hash
    message: Mapped[str] = mapped_column(String)
    content: Mapped[str] = mapped_column(String)
    parent_id: Mapped[Optional[str]] = mapped_column(ForeignKey("commits.id"), nullable=True)
    token_count: Mapped[int] = mapped_column(Integer)
```

**SQLAlchemy patterns to AVOID:**
- `declarative_base()` -- deprecated, use `DeclarativeBase` class inheritance
- `Column()` without `Mapped[]` -- loses type safety
- Legacy `Session` patterns without context managers
- `expire_on_commit=True` in async sessions (causes lazy loads that fail in async)

**Alembic decision:** Do NOT include Alembic for schema migrations in v1. Trace is an embedded library, not a web service. The library should manage its own schema creation on init (via `Base.metadata.create_all()`). Migrations add complexity inappropriate for a library. If the schema changes between versions, provide a manual migration utility or version-check-and-migrate on open. Revisit if schema stabilization becomes an issue.

### Data Modeling & Serialization

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Pydantic | >=2.10, <3.0 | API surface data models, validation, serialization | The standard for validated data structures in Python. 5-50x faster than v1 (Rust core). `.model_dump()` and `.model_dump_json()` for serialization. Use for SDK-facing types (CommitInfo, BranchInfo, etc.) and configuration objects. | HIGH |
| dataclasses | stdlib | Internal lightweight data containers | Zero-dependency, 6x faster instance creation than Pydantic. Use for internal objects that don't need validation (e.g., DAG traversal state, internal caches). | HIGH |
| hashlib | stdlib | SHA-256 content hashing for commit IDs | Standard library, no dependency needed. Use `hashlib.sha256()` for content-addressable commit IDs, matching git's model. | HIGH |

**Pydantic vs dataclasses decision:**
- **Pydantic** for anything crossing the SDK boundary (user-facing types, config objects, serialized/deserialized data). Validation matters here.
- **dataclasses** for internal data containers where you control all inputs. Performance matters here, especially in DAG traversal.
- **Do NOT use attrs.** It adds a dependency for minimal benefit over dataclasses for internal types, and Pydantic is strictly better for validated types. One less thing to learn.

**Serialization strategy:**
- Commit content: stored as text in SQLite (context windows are text)
- Commit metadata: Pydantic models serialized to JSON via `.model_dump_json()`
- DAG structure: stored relationally in SQLite (parent pointers, branch refs)
- No need for msgpack/protobuf -- the data is text-heavy and JSON is human-debuggable

### Token Counting

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| tiktoken | >=0.12.0 | BPE tokenizer for OpenAI models | OpenAI's official tokenizer. Covers GPT-4, GPT-4o, GPT-3.5-turbo. Fast (Rust-backed). cl100k_base and o200k_base encodings. Required dependency. | HIGH |

**Token counting architecture:**
- **tiktoken as default:** Covers OpenAI models (the majority use case). Make it a required dependency.
- **API response extraction:** Parse `usage.prompt_tokens` / `usage.completion_tokens` from LLM API responses when available. This is the authoritative count.
- **Pluggable tokenizer protocol:** Define a `Protocol` (or ABC) for token counting so users can provide tokenizers for Anthropic, Google, etc. Don't bundle all tokenizers -- let users bring their own.
- **Do NOT depend on:** `transformers` (huge dependency, 500MB+), `anthropic` tokenizer (not publicly available as standalone), or `token-counter` (small package with uncertain maintenance).

```python
from typing import Protocol

class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...

class TiktokenCounter:
    def __init__(self, model: str = "gpt-4o"):
        import tiktoken
        self.enc = tiktoken.encoding_for_model(model)

    def count(self, text: str) -> int:
        return len(self.enc.encode(text))
```

### CLI Framework

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| Click | >=8.1 | CLI framework | Battle-tested, huge ecosystem, used by major tools (Flask, pip, AWS CLI). More control than Typer for complex CLI patterns. Decorator-based is fine for a debugging/inspection tool. | HIGH |
| rich | >=14.0 | Terminal output formatting | Beautiful tables, syntax highlighting, tree views, progress bars. Essential for `trace log`, `trace diff`, `trace status` output. Works perfectly with Click via rich-click. | HIGH |
| rich-click | >=1.8 | Rich-formatted Click help | Drop-in replacement for Click's help formatter. Zero effort, beautiful help output. | MEDIUM |

**Why Click over Typer:**
- Trace's CLI is a debugging/inspection tool, not the primary interface. It doesn't need to be the absolute simplest to write.
- Click gives more control over command groups, custom types, and output formatting.
- Click is a direct dependency (not wrapping another library). Typer wraps Click, adding an abstraction layer with no real benefit for a developer tool.
- Click has rich-click for beautiful output. Typer bundles Rich but the integration is less mature for custom formatting.
- The target audience (developer tool builders) is deeply familiar with Click.

**Why NOT argparse:**
- Verbose, manual help formatting, no built-in support for command groups, poor DX for subcommand-heavy CLIs.
- Trace needs subcommands (`trace log`, `trace commit`, `trace branch list`) -- Click handles this naturally.

### Async & HTTP

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| asyncio | stdlib | Async runtime | Standard library. No dependency. Use for async SDK methods. | HIGH |
| httpx | >=0.28.0 | HTTP client for LLM API calls | Supports both sync and async with the same API. Modern, well-maintained. Better than aiohttp (async-only) or requests (sync-only). Type-annotated. Built-in timeout/retry support. | HIGH |

**Async architecture decision:**
- Provide **both sync and async** SDK interfaces. The sync interface wraps async internally or uses separate sync paths.
- Use `httpx.AsyncClient` for the built-in LLM client. Offer `httpx.Client` for sync path.
- Do NOT force users into async. Many agent frameworks are sync. The SDK should work seamlessly in both modes.
- Pattern: `trace.commit(...)` (sync) and `await trace.acommit(...)` (async), or use a single interface that detects the event loop.

**Why httpx over alternatives:**
- `requests`: No async support. Would need a separate async library.
- `aiohttp`: Async-only. Can't serve sync users without `asyncio.run()` wrappers.
- `urllib3`: Low-level, no async. Used under the hood by httpx anyway.

### Logging

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| structlog | >=25.0 | Structured logging | JSON-ready structured logs, perfect for a library. Context variables support. Integrates with stdlib logging. Production-proven since 2013. | MEDIUM |
| logging | stdlib | Fallback / integration | structlog integrates with stdlib logging. Users who don't want structlog get standard logging behavior. | HIGH |

**Rationale:** A library should use structured logging so consumers can integrate it into their own logging pipeline. structlog is the standard for this in Python. However, it IS an extra dependency. Alternative: use stdlib `logging` only and let users configure it. **Recommendation: Start with stdlib `logging` only in Phase 0, add structlog as an optional enhancement later if structured output proves valuable during development.** This keeps initial dependencies minimal.

### Testing Infrastructure

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| pytest | >=8.0 | Test runner | The standard. No debate. Fixtures, parametrize, plugins. | HIGH |
| pytest-asyncio | >=1.0 | Async test support | Major 1.0 release (May 2025) simplified the API. Removed event_loop fixture complexity. Required for testing async SDK methods. | HIGH |
| pytest-cov | >=7.0 | Coverage reporting | Coverage.py 7.13+ integration. Supports Python 3.10-3.15. | HIGH |
| hypothesis | >=6.150 | Property-based testing | Invaluable for DAG operations (merge, rebase, compression). Finds edge cases in graph algorithms that unit tests miss. Use for core data model invariants. | HIGH |
| pytest-xdist | >=3.0 | Parallel test execution | Run tests in parallel as the suite grows. Optional but recommended for CI. | MEDIUM |

**Testing strategy notes:**
- **hypothesis is critical** for this project. DAG operations (merge, rebase, reorder) have complex invariants that are hard to enumerate manually. Property-based testing can verify: "merging two branches produces a valid DAG," "compression preserves pinned commits," "token counts are always non-negative."
- **pytest-asyncio 1.0+** is a significant improvement. The old event_loop fixture was a common source of test flakiness. Pin to >=1.0.
- **No mocking framework needed** beyond pytest's built-in `monkeypatch` and `unittest.mock`. Don't add `pytest-mock` -- it's a thin wrapper that adds a dependency for minimal benefit.

### Code Quality & Linting

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| ruff | >=0.15 | Linting + formatting | Replaces flake8, black, isort, pyupgrade, autoflake. 10-100x faster (Rust). Single tool. 800+ built-in rules. Astral ecosystem (same as uv). | HIGH |
| mypy | >=1.14 | Static type checking | Industry standard for library type checking. Slower than pyright but has plugin support (important for SQLAlchemy). SQLAlchemy ships a mypy plugin. | MEDIUM |

**Why mypy over pyright:**
- SQLAlchemy has an official mypy plugin that understands ORM patterns (`Mapped[]`, relationships). Pyright does not have equivalent plugin support.
- mypy is the standard for library projects that ship type stubs.
- pyright is faster (3-5x) but the SQLAlchemy plugin is a dealbreaker.
- **Emerging option: ty** (Astral's Rust-based type checker, 10-60x faster than mypy). Too new for production use as of Feb 2026. Watch this space.

**Ruff configuration (in pyproject.toml):**
```toml
[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "W",    # pycodestyle warnings
    "F",    # pyflakes
    "I",    # isort
    "N",    # pep8-naming
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "TCH",  # flake8-type-checking
    "RUF",  # ruff-specific rules
]
```

### Type Checking

| Technology | Version | Purpose | Why | Confidence |
|------------|---------|---------|-----|------------|
| typing-extensions | >=4.12 | Backported typing features | Required for supporting Python 3.10 with newer typing constructs. Pydantic depends on it anyway. | HIGH |

---

## Alternatives Considered

| Category | Recommended | Alternative | Why Not |
|----------|-------------|-------------|---------|
| Package manager | uv | poetry, pip+pip-tools, pdm | Poetry is slower and uses a non-standard lock format. pip-tools is manual. pdm is good but less adoption than uv in 2026. |
| Build backend | hatchling | setuptools, flit-core, uv_build | setuptools is legacy/complex. flit-core is minimal but hatchling is more flexible. uv_build is too new. |
| ORM | SQLAlchemy 2.0 | Tortoise ORM, peewee, raw sqlite3 | SQLAlchemy is the standard. Tortoise is async-only (can't serve sync users). Peewee is simpler but less powerful for complex queries. Raw sqlite3 loses type safety and migration path. |
| CLI | Click + rich-click | Typer, argparse | Typer adds abstraction for minimal benefit in a dev tool. argparse is verbose and lacks subcommand ergonomics. |
| HTTP client | httpx | requests, aiohttp, urllib3 | requests lacks async. aiohttp lacks sync. urllib3 is low-level. httpx does both with identical API. |
| Data validation | Pydantic v2 | attrs, dataclasses-only, marshmallow | attrs is redundant when you have Pydantic + dataclasses. Marshmallow is older/slower. dataclasses-only lacks validation. |
| Token counting | tiktoken | transformers, tokenizers (HF) | transformers is 500MB+. HF tokenizers is good but tiktoken covers the primary use case (OpenAI) with minimal footprint. |
| Formatter | ruff | black, autopep8 | ruff replaces black and is 100x faster. Same formatting output. |
| Linter | ruff | flake8, pylint | ruff replaces flake8+plugins. pylint is slow and opinionated beyond what's useful. |
| Type checker | mypy | pyright, ty | pyright lacks SQLAlchemy plugin. ty is too new (2025). |
| Logging | stdlib logging (Phase 0) | structlog, loguru | Keep dependencies minimal initially. Add structlog later if needed. loguru is opinionated and less suited for libraries. |
| Schema migration | None (create_all) | Alembic | Trace is a library, not a service. Libraries manage their own schema. Alembic is overkill. |
| DAG library | Custom (no NetworkX) | NetworkX, rustworkx | Trace's DAG is simple (parent pointers, branches, HEAD refs). NetworkX is a 10MB+ dependency for features we don't need. Build a thin DAG layer on top of SQLAlchemy relationships. |

---

## Full Dependency List

### Core Dependencies (required)

```toml
[project]
dependencies = [
    "sqlalchemy>=2.0.46,<2.2",
    "aiosqlite>=0.22.0",
    "pydantic>=2.10,<3.0",
    "tiktoken>=0.12.0",
    "click>=8.1",
    "rich>=14.0",
    "rich-click>=1.8",
    "httpx>=0.28.0",
    "typing-extensions>=4.12",
]
```

### Development Dependencies

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=1.0",
    "pytest-cov>=7.0",
    "pytest-xdist>=3.0",
    "hypothesis>=6.150",
    "mypy>=1.14",
    "ruff>=0.15",
]
```

### Installation Commands

```bash
# Initialize project
uv init --lib trace

# Add core dependencies
uv add sqlalchemy aiosqlite pydantic tiktoken click rich rich-click httpx typing-extensions

# Add dev dependencies
uv add --dev pytest pytest-asyncio pytest-cov pytest-xdist hypothesis mypy ruff
```

---

## Key Architecture Decisions Encoded in Stack

### 1. Dual Sync/Async Support
The stack (SQLAlchemy + aiosqlite + httpx) supports both sync and async paths. This is deliberate -- agent frameworks are split between sync and async, and a library must serve both.

### 2. Minimal Dependency Footprint
Core dependencies: 9 packages. Every dependency earns its place:
- SQLAlchemy + aiosqlite: storage (the core feature)
- Pydantic: validation/serialization (SDK quality)
- tiktoken: token counting (the domain requirement)
- Click + rich + rich-click: CLI (the debugging interface)
- httpx: LLM calls (the intelligence layer)
- typing-extensions: compatibility

### 3. Content-Addressable Storage
Using hashlib (stdlib) for SHA-256 commit IDs. This mirrors git's object model and enables deduplication, integrity checking, and deterministic references. No extra dependency needed.

### 4. No Heavy ML Dependencies
tiktoken (small, Rust-backed) is the only ML-adjacent dependency. No transformers, no torch, no sentence-transformers. The LLM operations (compression, semantic merge) go through httpx to external APIs. This keeps install size small and install time fast.

### 5. Plugin Architecture via Protocols
Python `Protocol` types (PEP 544) for extensibility points:
- `TokenCounter` protocol for custom tokenizers
- `Materializer` protocol for context rendering
- `LLMCallable` protocol for user-provided LLM functions
This avoids framework coupling while providing type safety.

---

## pyproject.toml Skeleton

```toml
[project]
name = "trace-context"
version = "0.1.0"
description = "Git-like version control for LLM context windows"
readme = "README.md"
license = "MIT"
requires-python = ">=3.10"
authors = [
    { name = "Your Name", email = "you@example.com" }
]
keywords = ["llm", "context", "version-control", "agent"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
    "Topic :: Software Development :: Libraries",
]
dependencies = [
    "sqlalchemy>=2.0.46,<2.2",
    "aiosqlite>=0.22.0",
    "pydantic>=2.10,<3.0",
    "tiktoken>=0.12.0",
    "click>=8.1",
    "rich>=14.0",
    "rich-click>=1.8",
    "httpx>=0.28.0",
    "typing-extensions>=4.12",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=1.0",
    "pytest-cov>=7.0",
    "pytest-xdist>=3.0",
    "hypothesis>=6.150",
    "mypy>=1.14",
    "ruff>=0.15",
]

[project.scripts]
trace = "trace.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/trace"]

[tool.ruff]
target-version = "py310"
line-length = 100
src = ["src"]

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "UP", "B", "SIM", "TCH", "RUF"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[tool.mypy]
python_version = "3.10"
strict = true
plugins = ["pydantic.mypy", "sqlalchemy.ext.mypy.plugin"]

[tool.coverage.run]
source = ["trace"]
branch = true

[tool.coverage.report]
show_missing = true
fail_under = 80
```

---

## Version Verification Summary

| Package | Verified Version | Source | Date Checked |
|---------|-----------------|--------|--------------|
| SQLAlchemy | 2.0.46 (stable), 2.1.0b1 (beta) | PyPI + SQLAlchemy blog | 2026-02-10 |
| aiosqlite | 0.22.1 | PyPI | 2026-02-10 |
| Pydantic | 2.12.x | PyPI | 2026-02-10 |
| tiktoken | 0.12.0 | PyPI | 2026-02-10 |
| Click | 8.x | PyPI (version not explicitly verified) | 2026-02-10 |
| Rich | 14.1.0 | PyPI + readthedocs | 2026-02-10 |
| httpx | 0.28.1 | PyPI | 2026-02-10 |
| Typer | 0.20.0 | PyPI | 2026-02-10 |
| pytest | 8.x | PyPI (version not explicitly verified) | 2026-02-10 |
| pytest-asyncio | 1.3.0 (docs), 1.0+ (major release May 2025) | PyPI + GitHub | 2026-02-10 |
| pytest-cov | 7.0 | PyPI + readthedocs | 2026-02-10 |
| hypothesis | 6.151.4 | PyPI | 2026-02-10 |
| coverage.py | 7.13.4 | readthedocs | 2026-02-10 |
| ruff | 0.15.x | PyPI + Astral blog | 2026-02-10 |
| mypy | 1.14+ | WebSearch (version approximate) | 2026-02-10 |
| uv | latest | Astral docs | 2026-02-10 |
| Alembic | 1.18.3 | PyPI | 2026-02-10 |
| structlog | 25.5.0 | PyPI + readthedocs | 2026-02-10 |

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Python version floor (3.10) | HIGH | Verified via SQLAlchemy 2.1 docs requiring 3.10+ |
| SQLAlchemy 2.0 patterns | HIGH | Official docs confirm DeclarativeBase/Mapped/mapped_column as standard |
| uv as package manager | HIGH | Widely documented, 2026 "golden path" per multiple sources |
| Click over Typer for CLI | MEDIUM | Opinionated choice. Both are valid. Rationale is sound but Typer is also fine. |
| Pydantic for SDK types | HIGH | Industry standard for validated Python data models |
| tiktoken for token counting | HIGH | PyPI verified, actively maintained, covers primary use case |
| httpx for HTTP | HIGH | Verified async+sync support, well-maintained |
| hatchling as build backend | MEDIUM | Good default but uv_build or flit-core would also work |
| No Alembic (schema management) | MEDIUM | Correct for a library, but may need revisiting if schema evolves significantly |
| No NetworkX for DAG | HIGH | Trace's DAG is too simple to justify a 10MB dependency |
| mypy over pyright | MEDIUM | SQLAlchemy plugin is the tiebreaker, but ty (Astral) may change this landscape soon |
| hypothesis for testing | HIGH | Perfect fit for DAG invariant testing |

---

## Sources

### Verified (HIGH confidence)
- [SQLAlchemy 2.0 Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [SQLAlchemy 2.1 Documentation](https://docs.sqlalchemy.org/en/21/dialects/sqlite.html)
- [SQLAlchemy 2.1.0b1 Release Blog](https://www.sqlalchemy.org/blog/2026/01/21/sqlalchemy-2.1.0b1-released/)
- [SQLAlchemy Declarative Mapping Styles](https://docs.sqlalchemy.org/en/20/orm/declarative_styles.html)
- [uv Project Documentation](https://docs.astral.sh/uv/guides/projects/)
- [uv Build Backend Documentation](https://docs.astral.sh/uv/concepts/build-backend/)
- [uv Project Init Documentation](https://docs.astral.sh/uv/concepts/projects/init/)
- [Pydantic Serialization Documentation](https://docs.pydantic.dev/latest/concepts/serialization/)
- [pytest Good Integration Practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html)
- [pytest-asyncio Documentation](https://pytest-asyncio.readthedocs.io/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Rich Documentation](https://rich.readthedocs.io/en/stable/introduction.html)
- [HTTPX Async Documentation](https://www.python-httpx.org/async/)
- [hashlib Python Documentation](https://docs.python.org/3/library/hashlib.html)
- [Typer Documentation](https://typer.tiangolo.com/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/en/latest/)
- [Python Packaging - pyproject.toml Guide](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/)
- [structlog Documentation](https://www.structlog.org/)
- [NetworkX DAG Documentation](https://networkx.org/documentation/stable/reference/algorithms/dag.html)

### WebSearch-informed (MEDIUM confidence)
- [Managing Python Projects With uv - Real Python](https://realpython.com/python-uv/)
- [Embracing Modern SQLAlchemy 2.0 - Medium](https://medium.com/@azizmarzouki/embracing-modern-sqlalchemy-2-0-declarativebase-mapped-and-beyond-ef8bcba1e79c)
- [2026 Golden Path: Building Python Packages with uv - Medium](https://medium.com/@diwasb54/the-2026-golden-path-building-and-publishing-python-packages-with-a-single-tool-uv-b19675e02670)
- [Asynchronous LLM API Calls Guide - Unite.AI](https://www.unite.ai/asynchronous-llm-api-calls-in-python-a-comprehensive-guide/)
- [Python HTTP Clients Comparison - Speakeasy](https://www.speakeasy.com/blog/python-http-clients-requests-vs-httpx-vs-aiohttp)
- [HTTPX vs Requests vs AIOHTTP - Oxylabs](https://oxylabs.io/blog/httpx-vs-requests-vs-aiohttp)
- [Mypy vs Pyright Performance - Medium](https://medium.com/@asma.shaikh_19478/python-type-checking-mypy-vs-pyright-performance-battle-fce38c8cb874)
- [Pyright Mypy Comparison - GitHub](https://github.com/microsoft/pyright/blob/main/docs/mypy-comparison.md)
- [ty Type Checker Announcement - Astral](https://astral.sh/blog/ty)
- [Dataclasses vs Pydantic Performance - Medium](https://levelup.gitconnected.com/i-benchmarked-pythons-3-data-libraries-the-results-surprised-me-821dbc7c440e)

### Package version verification (PyPI)
- [SQLAlchemy on PyPI](https://pypi.org/project/SQLAlchemy)
- [tiktoken on PyPI](https://pypi.org/project/tiktoken/)
- [pydantic on PyPI](https://pypi.org/project/pydantic/)
- [typer on PyPI](https://pypi.org/project/typer/)
- [httpx on PyPI](https://pypi.org/project/httpx/)
- [aiosqlite on PyPI](https://pypi.org/project/aiosqlite/)
- [ruff on PyPI](https://pypi.org/project/ruff/)
- [pytest-asyncio on PyPI](https://pypi.org/project/pytest-asyncio/)
- [pytest-cov on PyPI](https://pypi.org/project/pytest-cov/)
- [hypothesis on PyPI](https://pypi.org/project/hypothesis/)
- [structlog on PyPI](https://pypi.org/project/structlog/)
- [alembic on PyPI](https://pypi.org/project/alembic/)
- [rich-click on PyPI](https://pypi.org/project/rich-click/)
