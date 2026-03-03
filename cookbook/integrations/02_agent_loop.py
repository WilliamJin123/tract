"""Agent Loop Protocol -- Pluggable Orchestrator

Swap tract's built-in Orchestrator for any external agent loop.
The AgentLoop protocol parallels LLMClient: LLMClient swaps the LLM
transport layer, AgentLoop swaps the orchestration layer.

Tract prepares everything (assessment, tools, execute_tool callable)
and hands off to the loop.  The loop runs however it wants -- its own
LLM, its own tool dispatch -- and returns an AgentLoopResult with
optional provenance (generation_config, usage per step).

Demonstrates: AgentLoop protocol, AgentLoopResult, configure_agent_loop(),
              Tract.open(agent_loop=), orchestrate(agent_loop=),
              provenance recording from external loops

Requires: no external dependencies (uses mock adapter)
"""

# TODO: Implement when integration examples are built
#
# Planned sections:
#
# Part 1 -- Custom AgentLoop
#   - Define a minimal class with run() and stop()
#   - run() receives messages, tools, execute_tool
#   - Return AgentLoopResult with steps
#
# Part 2 -- Wiring into Tract
#   - Tract.open(agent_loop=my_loop)
#   - configure_agent_loop() post-open
#   - orchestrate(agent_loop=) per-call override
#
# Part 3 -- Provenance
#   - Return total_usage and model in AgentLoopResult
#   - Per-step generation_config and usage on StepResult
#   - Verify tract records the provenance
#
# Part 4 -- execute_tool bridge
#   - Loop calls execute_tool("compress", {...})
#   - Tract's ToolExecutor dispatches to real operations
#   - Results flow back through the loop
