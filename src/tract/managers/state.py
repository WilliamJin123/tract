"""Shared mutable LLM configuration state and sentinels."""
from __future__ import annotations

# Shared sentinel objects used by Tract, LLMManager, and ToolkitManager
# for detecting "not provided" vs None in tool/profile parameters.
TOOLS_SENTINEL = object()
PROFILE_SENTINEL = object()

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from tract.models.config import OperationClients, OperationConfigs, OperationPrompts

if TYPE_CHECKING:
    from tract.llm.protocols import LLMClient, ResolverCallable
    from tract.models.config import LLMConfig, RetryConfig, ToolSummarizationConfig


@dataclass
class LLMState:
    """Shared mutable LLM configuration state.

    Created by ``Tract.__init__``, then passed to LLM/Compression/Intelligence/Config managers.
    ConfigManager is the writer; others are readers.
    """

    llm_client: LLMClient | None = None
    default_config: LLMConfig | None = None
    operation_configs: OperationConfigs = field(default_factory=OperationConfigs)
    operation_prompts: OperationPrompts = field(default_factory=OperationPrompts)
    operation_clients: OperationClients = field(default_factory=OperationClients)
    retry_config: RetryConfig | None = None
    default_resolver: ResolverCallable | None = None
    commit_reasoning: bool = True
    auto_message_enabled: bool = False
    tool_summarization_config: ToolSummarizationConfig | None = None
    owns_llm_client: bool = False
