"""Callable Tools -- Framework-Agnostic Tool Export

Export tract's context management tools as typed Python callables that
any agent framework can introspect.  Every major framework (Agno, LangChain,
CrewAI, LangGraph) accepts plain functions and reads their __name__,
__doc__, __signature__, and type annotations to build tool schemas.

as_callable_tools() does the conversion once -- no per-framework adapters.

Demonstrates: as_callable_tools(), profile filtering, description overrides,
              inspect.signature() introspection

Requires: no external dependencies (framework-agnostic)
"""

# TODO: Implement when integration examples are built
#
# Planned sections:
#
# Part 1 -- Basic export
#   - t.as_callable_tools() returns typed callables
#   - Inspect __name__, __doc__, __signature__
#   - Call a tool directly (status, log)
#
# Part 2 -- Profile and override control
#   - as_callable_tools(profile="self") vs "full"
#   - Description overrides for framework-specific hints
#
# Part 3 -- Framework integration sketch
#   - Show the pattern: tools = t.as_callable_tools()
#   - Pseudocode for passing to any framework
#   - Verify introspection matches framework expectations
