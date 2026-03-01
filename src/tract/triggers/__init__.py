"""Trigger engine package -- automatic context management triggers.

Provides the Trigger ABC, TriggerEvaluator, and built-in triggers for
registering and evaluating triggers against a Tract instance.
"""

from tract.triggers.protocols import Trigger
from tract.triggers.evaluator import TriggerEvaluator
from tract.triggers.builtin import (
    ArchiveTrigger,
    BranchTrigger,
    CompressTrigger,
    GCTrigger,
    MergeTrigger,
    PinTrigger,
    RebaseTrigger,
)

__all__ = [
    "Trigger",
    "TriggerEvaluator",
    "CompressTrigger",
    "PinTrigger",
    "BranchTrigger",
    "MergeTrigger",
    "RebaseTrigger",
    "GCTrigger",
    "ArchiveTrigger",
]
