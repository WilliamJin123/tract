"""Policy engine package -- automatic context management policies.

Provides the Policy ABC and PolicyEvaluator for registering and
evaluating policies against a Tract instance.
"""

from tract.policy.protocols import Policy
from tract.policy.evaluator import PolicyEvaluator

__all__ = ["Policy", "PolicyEvaluator"]
