"""LangChain Integration -- Tract Tools in a LangChain Agent

Use tract's context management alongside LangChain's agent framework.
Tract tools become LangChain tools via as_callable_tools(), and the
AgentLoop protocol lets you swap tract's orchestrator for a LangGraph
agent.

Demonstrates: as_callable_tools() with LangChain AgentExecutor,
              LangGraph tool nodes, AgentLoop adapter,
              callback-based provenance extraction

Requires: langchain, langgraph (pip install langchain langgraph)
"""

# TODO: Implement when langchain integration is built
#
# Planned sections:
#
# Part 1 -- Callable tools with AgentExecutor
#   - tools = t.as_callable_tools()
#   - AgentExecutor(tools=[TavilySearch(), *tools])
#
# Part 2 -- LangGraph tool nodes
#   - Add tract tools as graph tool nodes
#   - Agent can route between task tools and context tools
#
# Part 3 -- AgentLoop adapter
#   - Wrap LangGraph graph as AgentLoop protocol
#   - Provenance from get_openai_callback()
