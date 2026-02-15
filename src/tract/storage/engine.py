"""Engine and session factory for Trace storage.

Provides SQLite engine creation with performance pragmas,
session factory creation, and database initialization.
"""

from __future__ import annotations

from sqlalchemy import Engine, create_engine, event, select
from sqlalchemy.orm import Session, sessionmaker

from tract.storage.schema import Base, TraceMetaRow


def create_trace_engine(db_path: str = ":memory:") -> Engine:
    """Create a SQLAlchemy engine with SQLite optimizations.

    Args:
        db_path: Path to SQLite database file, or ":memory:" for in-memory.

    Returns:
        Configured SQLAlchemy Engine.
    """
    url = "sqlite://" if db_path == ":memory:" else f"sqlite:///{db_path}"
    engine = create_engine(url, echo=False)

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
    For new databases, schema_version is set to "2".
    For existing v1 databases, migrates to v2 by creating commit_parents table.
    """
    Base.metadata.create_all(engine)

    SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
    with SessionLocal() as session:
        existing = session.execute(
            select(TraceMetaRow).where(TraceMetaRow.key == "schema_version")
        ).scalar_one_or_none()

        if existing is None:
            # New database: set schema version to 2
            session.add(TraceMetaRow(key="schema_version", value="2"))
            session.commit()
        elif existing.value == "1":
            # Migrate v1 -> v2: create commit_parents table
            Base.metadata.tables["commit_parents"].create(engine, checkfirst=True)
            existing.value = "2"
            session.commit()
