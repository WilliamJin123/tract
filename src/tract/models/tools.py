"""Tool schema models and utilities for Tract."""
from __future__ import annotations

import hashlib
import json


def hash_tool_schema(schema: dict) -> str:
    """Compute SHA-256 content hash of a tool schema.

    Canonicalizes by sorting keys, no-space separators.

    Args:
        schema: Tool definition dict (typically a JSON Schema).

    Returns:
        Hex-encoded SHA-256 hash of the canonical JSON representation.
    """
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()
