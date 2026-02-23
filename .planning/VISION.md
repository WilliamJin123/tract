# Tract Vision & Drift Analysis

## Core Principles (from PRINCIPLES.md)

1. **Granular Control** — Interjections at any point, context window manipulation, prompting and meta-prompting (policies)
2. **Human/Agent Symmetry** — Anything a human can do, an agent/LLM can also do. HITL prompt interjection maps to AITL monitoring. Human-triggered operations map to agent-triggered operations. Any human-exposed interface can also be exposed to an LLM.

## What We Built (v2 Complete)

A powerful, git-like version control system for LLM context windows:
- 888 tests, 7 phases, full branching/merging/compression/policy/orchestrator
- Content type system with 8 built-in types
- Compile cache with LRU, incremental extension, snapshot patching
- Branch/merge/rebase with LLM-powered conflict resolution
- Compression engine with PINNED preservation
- Policy engine with autonomy spectrum
- Agent toolkit (15 tools) and orchestrator loop

## Where We've Drifted

The cookbook examples are the clearest signal. Every example that talks to an LLM repeats the same boilerplate:

### Problem 1: Manual LLM Call Boilerplate

Every example writes its own `call_llm()` function from scratch:

```python
def call_llm(messages: list[dict]) -> dict:
    with httpx.Client(timeout=30.0) as client:
        response = client.post(
            f"{TRACT_OPENAI_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {API_KEY}", ...},
            json={"model": MODEL, "messages": messages},
        )
        response.raise_for_status()
        return response.json()
```

We *have* `OpenAIClient` in `tract.llm.client` — but nothing connects it to the Tract facade for the simple path. `configure_llm()` exists but only powers merge/rebase conflict resolution, not the core chat loop.

### Problem 2: Message Format Conversion

Every single example does this:

```python
compiled = t.compile()
messages = [{"role": m.role, "content": m.content} for m in compiled.messages]
```

`CompiledContext` returns frozen `Message` dataclasses but every LLM API wants `list[dict]`. There's no `.to_dicts()` or `.to_openai()` method. Users always have to transform the output.

### Problem 3: Content Type Verbosity

The most common operations are painfully verbose:

```python
t.commit(InstructionContent(text="You are helpful."), message="system prompt")
t.commit(DialogueContent(role="user", text="Hello"), message="user greeting")
t.commit(DialogueContent(role="assistant", text=response), message="assistant reply")
```

For the #1 use case (chat), this is 3 imports + long constructor calls for what should be:

```python
t.system("You are helpful.")
t.user("Hello")
t.assistant(response)
```

### Problem 4: No Integrated Chat Loop

The core pattern — commit user message, compile, call LLM, commit response — is ~15 lines every time:

```python
t.commit(DialogueContent(role="user", text=question), message="user question")
compiled = t.compile()
messages = [{"role": m.role, "content": m.content} for m in compiled.messages]
response = call_llm(messages)
assistant_text = response["choices"][0]["message"]["content"]
t.commit(DialogueContent(role="assistant", text=assistant_text), message="answer")
# Optional: record usage
if response.get("usage"):
    t.record_usage(response["usage"])
```

This should be 1-2 lines:

```python
response = t.chat("What is Python?")
# or with more control:
t.user("What is Python?")
response = t.generate()  # compile + call + commit + record_usage
```

### Problem 5: record_usage() is Disconnected

Token tracking requires manual extraction and a separate call. When using our own client, this should be automatic.

### Problem 6: generation_config is Manual

Users have to manually construct and pass `generation_config={"model": ..., "temperature": ...}` on every commit. When using our client, this should be captured automatically.

### Problem 7: Batch + LLM is Awkward

The atomic_batch example shows the pain: you compile BEFORE the batch, then manually append messages inside the batch, then call the LLM. The pre-batch compile + manual message building is confusing.

## The Gap

**What we have:** A powerful low-level system (git-for-context) where every operation is explicit.

**What's missing:** A high-level layer that makes the common path (multi-turn LLM conversation with version control) trivially easy while still allowing drop-down to the low-level API.

The principles say "granular control" — we have that. But granular doesn't mean verbose-by-default. Git has both `git add -p` (granular) and `git commit -am "msg"` (convenient). We only have the former.

## What "Right" Looks Like

### Tier 1: Zero-Boilerplate Chat

```python
from tract import Tract

with Tract.open() as t:
    t.system("You are a helpful assistant.")
    reply = t.chat("What is Python?")
    print(reply.text)
    reply = t.chat("Tell me more about decorators.")
    print(reply.text)
```

Under the hood: `chat()` commits the user message, compiles, calls the configured LLM, commits the response with auto-captured generation_config, records usage, and returns a rich response object.

### Tier 2: Control When You Want It

```python
with Tract.open() as t:
    t.system("You are a helpful assistant.")
    t.user("What is Python?")

    # Inspect/modify context before calling
    compiled = t.compile()
    print(f"Sending {compiled.token_count} tokens")

    # Call LLM with explicit control
    reply = t.generate(temperature=0.7, model="gpt-4o")
    print(reply.text)

    # Or bring your own LLM call
    messages = compiled.to_dicts()
    raw_response = my_custom_llm(messages)
    t.assistant(raw_response["text"])
```

### Tier 3: Full Power (Current API)

Everything we have today remains available. `commit()`, `compile()`, `annotate()`, `branch()`, `merge()`, etc.

## Specific Changes Needed

### Must-Have (Eliminates cookbook boilerplate)

1. **`CompiledContext.to_dicts()`** — Returns `list[dict]` with `role`/`content` keys. Trivial but eliminates the #1 repeated pattern.

2. **Convenience commit methods on Tract** — `t.system(text)`, `t.user(text)`, `t.assistant(text)` as shortcuts for the DialogueContent/InstructionContent ceremony. These just call `commit()` under the hood.

3. **`t.chat(text)` method** — The "happy path" that does commit-user → compile → call-LLM → commit-assistant → record-usage in one call. Requires LLM to be configured.

4. **`t.generate()` method** — Like `chat()` but assumes the user message is already committed. Just compile → call → commit-response → record-usage.

5. **Auto generation_config capture** — When using the built-in client, automatically populate generation_config from the request params (model, temperature, etc.).

6. **Auto usage recording** — When using the built-in client via `chat()`/`generate()`, automatically call `record_usage()` with the API response.

### Nice-to-Have (Polish)

7. **Rich response object from `chat()`/`generate()`** — Returns object with `.text`, `.usage`, `.commit_info`, `.generation_config`, etc.

8. **`Tract.open()` accepts LLM config directly** — `Tract.open(llm="openai", api_key="...", model="gpt-4o")` so users don't need a separate `configure_llm()` call.

9. **`CompiledContext.to_openai()` / `.to_anthropic()`** — Format-specific output methods for different providers.

## Design Constraints

- **Backward compatible** — All existing API stays. New methods are additive.
- **LLM-optional** — Core version control features must work without any LLM configured. The convenience methods raise clear errors if no LLM is set up.
- **No magic** — `chat()` and `generate()` are explicit about what they do. Users can always drop down to `commit()` + `compile()` for full control.
- **Content types stay** — The typed content system is correct for the low-level API. Convenience methods are sugar, not replacements.

## Priority Order

1. `CompiledContext.to_dicts()` — 5 minutes, eliminates the #1 boilerplate
2. `t.system()`, `t.user()`, `t.assistant()` — 30 minutes, eliminates import ceremony
3. `t.chat()` + `t.generate()` — 2-3 hours, eliminates the core loop boilerplate
4. Auto generation_config + usage recording — folded into #3
5. Rich response object — folded into #3
6. LLM config on `Tract.open()` — 1 hour, eliminates `configure_llm()` ceremony
