"""Engine and session factory for Trace storage.

Provides SQLite engine creation with performance pragmas,
session factory creation, and database initialization.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from tract.storage.schema import Base, TraceMetaRow


def create_trace_engine(
    db_path: str = ":memory:",
    *,
    url: str | None = None,
) -> Engine:
    """Create a SQLAlchemy engine for Trace storage.

    Supports three modes:

    1. **SQLite shorthand** (default): pass a file path or ``":memory:"``.
    2. **Full URL**: pass any SQLAlchemy connection URL via *url=*.
    3. **Pre-built engine**: callers who need full control can create
       their own ``Engine`` and pass it directly to :meth:`Tract.open`.

    SQLite performance pragmas (WAL, busy_timeout, foreign keys) are
    applied automatically when the engine dialect is SQLite.

    Args:
        db_path: Path to SQLite database file, or ``":memory:"`` for
            in-memory.  Ignored when *url* is provided.
        url: Full SQLAlchemy database URL, e.g.
            ``"postgresql://user:pass@host/db"`` or
            ``"sqlite:///path/to/file.db"``.

    Returns:
        Configured SQLAlchemy Engine.
    """
    if url is not None:
        engine = create_engine(url, echo=False)
    elif db_path == ":memory:":
        engine = create_engine("sqlite://", echo=False)
    else:
        engine = create_engine(f"sqlite:///{db_path}", echo=False)

    # Only apply SQLite pragmas when the backend is SQLite
    if engine.dialect.name == "sqlite":

        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):  # type: ignore[no-untyped-def]
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    return engine


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create a session factory bound to the given engine.

    Uses expire_on_commit=False to prevent lazy-load issues
    when accessing attributes after commit.
    """
    return sessionmaker(bind=engine, expire_on_commit=False)


def _migrate_compressions_v5_to_v6(engine: Engine) -> None:
    """Migrate compression data from old tables to operation_events/operation_commits.

    Reads all rows from compressions, compression_sources, compression_results
    and inserts corresponding rows into operation_events and operation_commits.
    Uses raw SQL to avoid dependency on removed ORM classes.
    """
    from sqlalchemy import text

    with engine.connect() as conn:
        # Check if compressions table exists (might not if DB went v2->v3 path)
        tables = [
            r[0]
            for r in conn.execute(
                text("SELECT name FROM sqlite_master WHERE type='table'")
            ).fetchall()
        ]
        if "compressions" not in tables:
            return

        # Read all compression records
        rows = conn.execute(text("SELECT * FROM compressions")).fetchall()
        for row in rows:
            compression_id = row[0]
            tract_id = row[1]
            branch_name = row[2]
            created_at = row[3]
            original_tokens = row[4]
            compressed_tokens = row[5]
            target_tokens = row[6] if len(row) > 6 else None
            instructions = row[7] if len(row) > 7 else None

            # Build params_json
            import json

            params = {}
            if target_tokens is not None:
                params["target_tokens"] = target_tokens
            if instructions is not None:
                params["instructions"] = instructions
            params_json = json.dumps(params) if params else None

            conn.execute(
                text(
                    "INSERT INTO operation_events "
                    "(event_id, tract_id, event_type, branch_name, created_at, "
                    "original_tokens, compressed_tokens, params_json) "
                    "VALUES (:eid, :tid, 'compress', :bn, :ca, :ot, :ct, :pj)"
                ),
                {
                    "eid": compression_id,
                    "tid": tract_id,
                    "bn": branch_name,
                    "ca": created_at,
                    "ot": original_tokens,
                    "ct": compressed_tokens,
                    "pj": params_json,
                },
            )

        # Migrate compression_sources -> operation_commits (role="source")
        if "compression_sources" in tables:
            sources = conn.execute(
                text("SELECT compression_id, commit_hash, position FROM compression_sources")
            ).fetchall()
            for src in sources:
                conn.execute(
                    text(
                        "INSERT INTO operation_commits "
                        "(event_id, commit_hash, role, position) "
                        "VALUES (:eid, :ch, 'source', :pos)"
                    ),
                    {"eid": src[0], "ch": src[1], "pos": src[2]},
                )

        # Migrate compression_results -> operation_commits (role="result")
        if "compression_results" in tables:
            results = conn.execute(
                text("SELECT compression_id, commit_hash, position FROM compression_results")
            ).fetchall()
            for res in results:
                conn.execute(
                    text(
                        "INSERT INTO operation_commits "
                        "(event_id, commit_hash, role, position) "
                        "VALUES (:eid, :ch, 'result', :pos)"
                    ),
                    {"eid": res[0], "ch": res[1], "pos": res[2]},
                )

        conn.commit()


def init_db(engine: Engine) -> None:
    """Initialize the database: create all tables and set schema version.

    Creates all tables defined in Base.metadata, then sets schema_version.
    For new databases, schema_version is set to "9".
    For existing v1 databases, migrates v1->v2->...->v9.
    For existing v2 databases, migrates v2->v3->...->v9.
    For existing v3 databases, migrates v3->v4->...->v9.
    For existing v4 databases, migrates v4->v5->...->v9 (policy tables).
    For existing v5 databases, migrates v5->v6->...->v9 (unified operation events).
    For existing v6 databases, migrates v6->v7->v8->v9 (retention_json on annotations).
    For existing v7 databases, migrates v7->v8->v9 (tool tracking tables).
    For existing v8 databases, migrates v8->v9 (instruction columns on operation_events).
    """
    from sqlalchemy import text

    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with SessionLocal() as session:
        existing = session.execute(
            select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        ).scalar_one_or_none()

        if existing is None:
            # New database: set schema version to 9
            session.add(TraceMetaRow(key="schema_version", value="9"))
            session.commit()
        elif existing.value == "1":
            # Migrate v1 -> v2: create commit_parents table
            Base.metadata.tables["commit_parents"].create(engine, checkfirst=True)
            existing.value = "2"
            session.commit()
            # Fall through to v2->v3 migration
            existing = session.execute(
                select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
            ).scalar_one()
        if existing is not None and existing.value == "2":
            # Migrate v2 -> v3: create compression record tables (raw SQL
            # because the ORM classes were removed in v6)
            with engine.connect() as conn:
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS compressions (
                        compression_id VARCHAR(64) PRIMARY KEY,
                        tract_id VARCHAR(64) NOT NULL,
                        branch_name VARCHAR(255),
                        created_at DATETIME NOT NULL,
                        original_tokens INTEGER NOT NULL DEFAULT 0,
                        compressed_tokens INTEGER NOT NULL DEFAULT 0,
                        target_tokens INTEGER,
                        instructions TEXT
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS compression_sources (
                        compression_id VARCHAR(64) NOT NULL
                            REFERENCES compressions(compression_id),
                        commit_hash VARCHAR(64) NOT NULL
                            REFERENCES commits(commit_hash),
                        position INTEGER NOT NULL,
                        PRIMARY KEY (compression_id, commit_hash)
                    )
                """))
                conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS compression_results (
                        compression_id VARCHAR(64) NOT NULL
                            REFERENCES compressions(compression_id),
                        commit_hash VARCHAR(64) NOT NULL
                            REFERENCES commits(commit_hash),
                        position INTEGER NOT NULL,
                        PRIMARY KEY (compression_id, commit_hash)
                    )
                """))
                conn.commit()
            existing.value = "3"
            session.commit()
        if existing is not None and existing.value == "3":
            # Migrate v3 -> v4: create spawn_pointers table
            Base.metadata.tables["spawn_pointers"].create(engine, checkfirst=True)
            existing.value = "4"
            session.commit()
        if existing is not None and existing.value == "4":
            # Migrate v4 -> v5: create policy tables
            for table_name in ["policy_proposals", "policy_log"]:
                Base.metadata.tables[table_name].create(engine, checkfirst=True)
            existing.value = "5"
            session.commit()
        if existing is not None and existing.value == "5":
            # Migrate v5 -> v6: unified operation events + compile records
            # Create new tables
            for table_name in [
                "operation_events",
                "operation_commits",
                "compile_records",
                "compile_effectives",
            ]:
                Base.metadata.tables[table_name].create(engine, checkfirst=True)
            # Migrate existing compression data to operation_events
            _migrate_compressions_v5_to_v6(engine)
            # Drop old compression tables (order matters: children first)
            with engine.connect() as conn:
                conn.execute(text("DROP TABLE IF EXISTS compression_results"))
                conn.execute(text("DROP TABLE IF EXISTS compression_sources"))
                conn.execute(text("DROP TABLE IF EXISTS compressions"))
                conn.commit()
            existing.value = "6"
            session.commit()
        if existing is not None and existing.value == "6":
            # Migrate v6 -> v7: add retention_json column to annotations
            with engine.connect() as conn:
                # Check if column already exists (idempotent)
                columns = [
                    r[1]
                    for r in conn.execute(
                        text("PRAGMA table_info(annotations)")
                    ).fetchall()
                ]
                if "retention_json" not in columns:
                    conn.execute(
                        text("ALTER TABLE annotations ADD COLUMN retention_json TEXT")
                    )
                    conn.commit()
            existing.value = "7"
            session.commit()
        if existing is not None and existing.value == "7":
            # Migrate v7 -> v8: add tool tracking tables
            for table_name in ["tool_definitions", "commit_tools"]:
                Base.metadata.tables[table_name].create(engine, checkfirst=True)
            existing.value = "8"
            session.commit()
        if existing is not None and existing.value == "8":
            # Migrate v8 -> v9: add instruction columns to operation_events
            with engine.connect() as conn:
                columns = [
                    r[1]
                    for r in conn.execute(
                        text("PRAGMA table_info(operation_events)")
                    ).fetchall()
                ]
                if "original_instructions" not in columns:
                    conn.execute(
                        text("ALTER TABLE operation_events ADD COLUMN original_instructions TEXT")
                    )
                if "effective_instructions" not in columns:
                    conn.execute(
                        text("ALTER TABLE operation_events ADD COLUMN effective_instructions TEXT")
                    )
                conn.commit()
            existing.value = "9"
            session.commit()
