"""Backward-compatibility alias: RebasePolicy -> ArchivePolicy.

The policy was renamed to ArchivePolicy to better reflect its behavior
(archiving stale branches, not rebasing).  Import from archive.py instead.
"""

from tract.policy.builtin.archive import ArchivePolicy

# Backward-compatibility alias
RebasePolicy = ArchivePolicy

__all__ = ["RebasePolicy"]
