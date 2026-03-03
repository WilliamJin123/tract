"""Agno Integration -- Tract Tools in an Agno Agent

Use tract's context management alongside Agno's agent framework.
The Agno agent gets tract tools (compress, branch, status, etc.) as
native Agno tools alongside its own tools (web search, etc.).

Two integration depths:
  1. as_callable_tools() -- inject tract tools into any Agno agent
  2. AgentLoop adapter -- swap tract's orchestrator for Agno's loop

Demonstrates: as_callable_tools() with Agno Agent, TractToolkit pattern,
              message sync via pre/post hooks, AgentLoop adapter

Requires: agno (pip install agno)
"""

# TODO: Implement when agno integration is built
#
# Planned sections:
#
# Part 1 -- Callable tools injection
#   - tools = t.as_callable_tools()
#   - Agent(model=..., tools=[DuckDuckGoTools(), *tools])
#   - Agent can search web AND manage tract context
#
# Part 2 -- Message sync
#   - Pre-hook: compile tract context into Agno messages
#   - Post-hook: commit Agno response back to tract
#   - Triggers fire naturally on post-hook commit
#
# Part 3 -- TractToolkit (native Agno Toolkit)
#   - Subclass agno.tools.Toolkit
#   - @tool decorated methods that call tract operations
#   - Agent(tools=[DuckDuckGoTools(), TractToolkit(t)])
#
# Part 4 -- AgentLoop adapter
#   - Wrap Agno Agent as AgentLoop protocol
#   - t.orchestrate(agent_loop=AgnoAdapter(agent))
#   - Provenance extraction from RunOutput.metrics
