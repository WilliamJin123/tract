"""Deterministic hashing utilities for Trace.

Provides canonical JSON serialization and SHA-256 hashing for content
and commits. All hashing is deterministic: same input always produces
same output, regardless of dict key ordering.

IMPORTANT: Pydantic models must be converted to dicts via
model_dump(mode="json") BEFORE passing to these functions.
These functions operate on plain dicts/primitives only.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(data: Any) -> bytes:
    """Serialize data to canonical JSON bytes.

    Uses sorted keys, compact separators, and UTF-8 encoding
    to ensure deterministic output.

    Args:
        data: Any JSON-serializable Python object (dict, list, str, int, etc.).

    Returns:
        UTF-8 encoded bytes of the canonical JSON string.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def content_hash(payload: dict) -> str:
    """Compute SHA-256 hash of a content payload dict.

    Args:
        payload: Content dict (already converted from Pydantic model).

    Returns:
        Hex digest of SHA-256 hash.
    """
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def commit_hash(
    content_hash: str,
    parent_hash: str | None,
    content_type: str,
    operation: str,
    timestamp_iso: str,
    reply_to: str | None = None,
) -> str:
    """Compute SHA-256 hash of structured commit data.

    The commit hash is computed from a canonical JSON dict containing
    all identity-relevant fields. reply_to is only included when not None.

    Args:
        content_hash: SHA-256 hex digest of the content blob.
        parent_hash: Hash of parent commit, or None for root.
        content_type: Content type discriminator string.
        operation: Commit operation ("append" or "edit").
        timestamp_iso: ISO 8601 timestamp string.
        reply_to: Hash of the commit being edited, or None.

    Returns:
        Hex digest of SHA-256 hash.
    """
    data: dict[str, Any] = {
        "content_hash": content_hash,
        "parent_hash": parent_hash,
        "content_type": content_type,
        "operation": operation,
        "timestamp_iso": timestamp_iso,
    }
    if reply_to is not None:
        data["reply_to"] = reply_to

    return hashlib.sha256(canonical_json(data)).hexdigest()
