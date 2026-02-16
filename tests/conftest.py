"""Shared test fixtures for Trace.

Provides in-memory SQLite engine, session, and repository fixtures.
"""

import pytest
from sqlalchemy.orm import Session, sessionmaker

from tract.storage.engine import create_trace_engine, init_db
from tract.storage.sqlite import (
    SqliteAnnotationRepository,
    SqliteBlobRepository,
    SqliteCommitRepository,
    SqliteRefRepository,
)


@pytest.fixture
def engine():
    """In-memory SQLite engine with all tables created."""
    eng = create_trace_engine(":memory:")
    init_db(eng)
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
def sample_tract_id() -> str:
    return "test-tract-001"


@pytest.fixture
def blob_repo(session: Session) -> SqliteBlobRepository:
    return SqliteBlobRepository(session)


@pytest.fixture
def commit_repo(session: Session) -> SqliteCommitRepository:
    return SqliteCommitRepository(session)


@pytest.fixture
def ref_repo(session: Session) -> SqliteRefRepository:
    return SqliteRefRepository(session)


@pytest.fixture
def annotation_repo(session: Session) -> SqliteAnnotationRepository:
    return SqliteAnnotationRepository(session)


# ------------------------------------------------------------------
# Shared test helpers (used by test_navigation.py, test_operations.py)
# ------------------------------------------------------------------

def make_tract(**kwargs) -> "Tract":
    """Create an in-memory Tract for testing."""
    from tract import Tract
    return Tract.open(":memory:", **kwargs)


def populate_tract(t: "Tract", n: int = 3) -> list[str]:
    """Commit n messages and return their hashes."""
    from tract import InstructionContent, DialogueContent
    hashes = []
    for i in range(n):
        if i == 0:
            info = t.commit(InstructionContent(text=f"System prompt {i}"))
        else:
            info = t.commit(DialogueContent(role="user", text=f"Message {i}"))
        hashes.append(info.commit_hash)
    return hashes


def make_tract_with_commits(n_commits=5, texts=None):
    """Create a Tract with n dialogue commits and return (tract, commit_hashes).

    Shared helper used by compression, reorder, and GC tests.
    """
    from tract import Tract, DialogueContent
    t = Tract.open()
    hashes = []
    texts = texts or [f"Message {i+1}" for i in range(n_commits)]
    for text in texts:
        info = t.commit(DialogueContent(role="user", text=text))
        hashes.append(info.commit_hash)
    return t, hashes
