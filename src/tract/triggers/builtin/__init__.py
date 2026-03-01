"""Built-in triggers for Trace's trigger engine.

Provides seven ready-to-use triggers:
- CompressTrigger: Auto-compress when token usage exceeds threshold
- PinTrigger: Auto-pin commits based on content type
- BranchTrigger: Detect content type tangents and propose branching
- MergeTrigger: Detect branch completion and propose merge
- RebaseTrigger: Detect branch divergence and propose rebase
- GCTrigger: Detect dead commits and propose garbage collection
- ArchiveTrigger: Detect stale branches and propose archiving
"""

from tract.triggers.builtin.archive import ArchiveTrigger
from tract.triggers.builtin.branch import BranchTrigger
from tract.triggers.builtin.compress import CompressTrigger
from tract.triggers.builtin.gc import GCTrigger
from tract.triggers.builtin.merge import MergeTrigger
from tract.triggers.builtin.pin import PinTrigger
from tract.triggers.builtin.rebase import RebaseTrigger

__all__ = [
    "CompressTrigger",
    "PinTrigger",
    "BranchTrigger",
    "MergeTrigger",
    "RebaseTrigger",
    "GCTrigger",
    "ArchiveTrigger",
]
