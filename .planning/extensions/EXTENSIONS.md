# Tract Extensions & Future Ideas

## Tool Compaction
- Contextualized tool compaction (tool compaction with more injected context)

## Multi-Session
- Multi-session stuff equivalent to Agno sessions
- DB integration with agent SDKs — consuming their context?
- All the messy parameters that affect context downstream (esp for Agno)
- Cross-session preservations

## Routing & Config
- Fuzzy AND workflow routing
- Self-managing configs

## Evaluation
- Long memory eval 

## Subagent Protocols
- Subagent context delegation protocols (super detailed definitions)
- Defining clean protocol for granular subagent context, config, and tools
- Similar to tool usage, subagent deployments are expensive — how to summarize
- A2A communication? Parent agent looks at subagent commits, asks clarifying questions / peeks into full context?

## LLM-Driven Context Intelligence
- LLM-driven context cherry picking (e.g., eliminating duplicate reads, duplicate summaries / level 1 conclusions)

## Branch & Rebase
- Branch + rebase

## Autonomous Tract Operations
- Able to commit automatically in a granular way (splitting up one convo turn)
- Able to separate commits with identifiers / useful info
- Able to compact automatically
- Able to peek at its own commit logs / make high-level decisions like skipping, pinning, etc.
- Able to rebase itself
- Able to branch
- Able to rebase + branch into a subagent
- Able to peek at subagent commit logs and ask clarifying questions
- Able to use hooks

## Persistence
- Persisting custom LLM-built workflows / hooks / policies / configs
