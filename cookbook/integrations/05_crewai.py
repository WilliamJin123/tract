"""CrewAI Integration -- Tract Tools in a CrewAI Agent

Use tract's context management alongside CrewAI's agent framework.
Tract tools become CrewAI tools via as_callable_tools(), letting
agents self-manage their context windows during multi-agent workflows.

Demonstrates: as_callable_tools() with CrewAI Agent/Task,
              per-agent tract instances, delegation with context sync

Requires: crewai (pip install crewai)
"""

# TODO: Implement when crewai integration is built
#
# Planned sections:
#
# Part 1 -- Callable tools with CrewAI Agent
#   - tools = t.as_callable_tools()
#   - Agent(tools=[SerperDevTool(), *tools])
#
# Part 2 -- Multi-agent with per-agent tracts
#   - Each CrewAI agent gets its own tract instance
#   - Agents manage their own context independently
#
# Part 3 -- Delegation with context sync
#   - Parent agent delegates to child
#   - import_commit() to sync relevant context across tracts
