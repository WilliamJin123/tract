"""Policy engine package -- automatic context management policies.

Provides the Policy ABC, PolicyEvaluator, and built-in policies for
registering and evaluating policies against a Tract instance.
"""

from tract.policy.protocols import Policy
from tract.policy.evaluator import PolicyEvaluator
from tract.policy.builtin import ArchivePolicy, BranchPolicy, CompressPolicy, PinPolicy, RebasePolicy

__all__ = [
    "Policy",
    "PolicyEvaluator",
    "CompressPolicy",
    "PinPolicy",
    "BranchPolicy",
    "ArchivePolicy",
    "RebasePolicy",
]
