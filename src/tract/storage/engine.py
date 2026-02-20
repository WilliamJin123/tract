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


def init_db(engine: Engine) -> None:
    """Initialize the database: create all tables and set schema version.

    Creates all tables defined in Base.metadata, then sets schema_version.
    For new databases, schema_version is set to "5".
    For existing v1 databases, migrates v1->v2->v3->v4->v5.
    For existing v2 databases, migrates v2->v3->v4->v5.
    For existing v3 databases, migrates v3->v4->v5.
    For existing v4 databases, migrates v4->v5 (policy tables).
    """
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with SessionLocal() as session:
        existing = session.execute(
            select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        ).scalar_one_or_none()

        if existing is None:
            # New database: set schema version to 5
            session.add(TraceMetaRow(key="schema_version", value="5"))
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
            # Migrate v2 -> v3: create compression record tables
            for table_name in ["compressions", "compression_sources", "compression_results"]:
                Base.metadata.tables[table_name].create(engine, checkfirst=True)
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
