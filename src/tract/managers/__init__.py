"""Manager classes extracted from Tract mixins.

Each manager is a standalone class with explicit constructor dependencies,
replacing the mixin-based inheritance pattern.
"""

from __future__ import annotations

from tract.managers.annotations import AnnotationManager
from tract.managers.branches import BranchManager
from tract.managers.compression import CompressionManager
from tract.managers.config import ConfigManager
from tract.managers.intelligence import IntelligenceManager
from tract.managers.llm import LLMManager
from tract.managers.middleware import MiddlewareManager
from tract.managers.persistence import PersistenceManager
from tract.managers.routing import RoutingManager
from tract.managers.search import SearchManager
from tract.managers.spawn import SpawnManager
from tract.managers.state import LLMState
from tract.managers.tags import TagManager
from tract.managers.templates import TemplateManager
from tract.managers.toolkit import ToolkitManager
from tract.managers.tools import ToolManager

__all__ = [
    "AnnotationManager",
    "BranchManager",
    "CompressionManager",
    "ConfigManager",
    "IntelligenceManager",
    "LLMManager",
    "LLMState",
    "MiddlewareManager",
    "PersistenceManager",
    "RoutingManager",
    "SearchManager",
    "SpawnManager",
    "TagManager",
    "TemplateManager",
    "ToolkitManager",
    "ToolManager",
]
