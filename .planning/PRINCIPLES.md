1. Granular Control 
- Interjections at any point
- context window manipulation
- prompting and meta-prompting (policies)
- EVERYTHING should be configurable together or separately (LLM [Models, Hyperparameters, Prompts], strategies / triggers for any operation)
- Deterministic (hard coded strategies) OR fuzzy (LLM-based sentiment analysis / judgement / reasoning)
2. Anything a human can do, an agent / LLM can also do 
- HITL prompt interjection ==> AITL monitoring
- human-triggered context operations (rebase, reset, edits + appends, branching, etc.) ==> agent-triggered is possible
- Any human-exposed interface (apis, python functions, etc.) can also be exposed to an LLM (toolkits, customizable prompts, etc.)
- Crucially the agents not only automate processes / real-world tasks, but they must also be able to manage / automate their own configurations / select protocols
- etc.
3. Full History
- Provenance on everything
- How things (context window, config) change over time
- How subagent delegation happened over commits and / or timestamps
- Tracing everything back to triggers (manual vs automated), 