# Phase 9: Conversation Layer - Research

**Researched:** 2026-02-19
**Domain:** LLM conversation integration, response objects, auto-config capture
**Confidence:** HIGH

## Summary

Phase 9 adds a conversation layer to the Tract SDK that eliminates boilerplate for the most common use case: multi-turn LLM chat with version control. The existing infrastructure is extensive -- `OpenAIClient` with retry logic, `configure_llm()`, `record_usage()`, `generation_config` on commits, `to_dicts()`/`to_openai()`/`to_anthropic()` on `CompiledContext`, and shorthand `system()`/`user()`/`assistant()` methods. Phase 9 wires these together into `chat()` and `generate()` convenience methods, plus allows LLM configuration directly on `Tract.open()`.

The research confirms this is a pure facade layer -- no new storage, schema, or engine changes. All building blocks already exist. The implementation is a matter of composing existing methods into higher-level operations and adding a response dataclass to carry results.

**Primary recommendation:** Implement in a single plan with 3 sections: (1) Response dataclass + LLM config on `Tract.open()`, (2) `chat()` method, (3) `generate()` method with auto-config/usage capture. Total new code is ~150-200 lines in `tract.py` plus ~30 lines for a response model, plus ~200 lines of tests.

## Standard Stack

### Core

No new libraries needed. Phase 9 uses only existing infrastructure:

| Component | Location | Purpose | Already Exists |
|-----------|----------|---------|----------------|
| `OpenAIClient` | `src/tract/llm/client.py` | HTTP client for OpenAI-compatible APIs | Yes |
| `LLMClient` protocol | `src/tract/llm/protocols.py` | Pluggable client interface | Yes |
| `Tract.configure_llm()` | `src/tract/tract.py:1229` | Stores LLM client on instance | Yes |
| `Tract.record_usage()` | `src/tract/tract.py:1732` | Records API token usage | Yes |
| `Tract.commit()` | `src/tract/tract.py:417` | Creates commits with generation_config | Yes |
| `Tract.compile()` | `src/tract/tract.py:595` | Compiles context to messages | Yes |
| `CompiledContext.to_dicts()` | `src/tract/protocols.py:40` | Converts to LLM-ready format | Yes (Phase 8) |
| `Tract.user()` / `Tract.assistant()` | `src/tract/tract.py:534,563` | Shorthand commit methods | Yes (Phase 8) |
| `OpenAIClient.extract_content()` | `src/tract/llm/client.py:216` | Extracts text from response | Yes |
| `OpenAIClient.extract_usage()` | `src/tract/llm/client.py:236` | Extracts usage from response | Yes |

### New Components Needed

| Component | Purpose | Location |
|-----------|---------|----------|
| `ChatResponse` | Frozen dataclass for chat/generate return value | `src/tract/protocols.py` or new file |
| `Tract.chat()` | One-call conversation turn | `src/tract/tract.py` |
| `Tract.generate()` | Compile + LLM + commit | `src/tract/tract.py` |
| LLM params on `Tract.open()` | api_key, model, base_url kwargs | `src/tract/tract.py` |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Frozen dataclass for ChatResponse | Pydantic model | Overkill -- this is output-only, no validation needed. Dataclass matches existing `CompiledContext`, `TokenUsage`, `Message` patterns |
| New `conversation.py` module | Methods directly on `Tract` | Separate module adds indirection for simple methods. `system()`/`user()`/`assistant()` are already on `Tract` directly -- `chat()`/`generate()` should follow same pattern |

## Architecture Patterns

### Recommended Approach: Compose Existing Methods

`chat()` and `generate()` are pure orchestration methods that compose existing primitives in sequence. They do NOT introduce new concepts.

```
chat(text, **kwargs):
    1. self.user(text)              # commit user message
    2. return self.generate(**kwargs)  # delegate to generate()

generate(**kwargs):
    1. compiled = self.compile()     # compile context
    2. messages = compiled.to_dicts() # convert to LLM format
    3. response = self._llm_client.chat(messages, **kwargs)  # call LLM
    4. text = extract_content(response)    # extract text
    5. usage = extract_usage(response)     # extract usage
    6. gen_config = build_generation_config(response, kwargs)  # build config
    7. commit_info = self.assistant(text, generation_config=gen_config)  # commit
    8. if usage: self.record_usage(usage)  # record usage
    9. return ChatResponse(text, usage, commit_info, gen_config)  # return
```

### Pattern 1: ChatResponse Dataclass

**What:** A frozen dataclass returned by `chat()` and `generate()`.
**When to use:** Every `chat()` and `generate()` call.

```python
@dataclass(frozen=True)
class ChatResponse:
    """Response from chat() or generate()."""
    text: str
    usage: TokenUsage | None
    commit_info: CommitInfo
    generation_config: dict
```

This follows the project convention established by `CompiledContext`, `TokenUsage`, and `Message` -- all frozen dataclasses in `protocols.py`.

### Pattern 2: LLM Config on Tract.open()

**What:** Pass `api_key`, `model`, `base_url` as optional kwargs to `Tract.open()`.
**When to use:** Users who want zero-ceremony LLM setup.

```python
@classmethod
def open(
    cls,
    path: str = ":memory:",
    *,
    # ... existing params ...
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> Tract:
    tract = cls(...)
    # Auto-configure LLM if api_key provided (or env var exists)
    if api_key is not None:
        from tract.llm.client import OpenAIClient
        client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            default_model=model or "gpt-4o-mini",
        )
        tract.configure_llm(client)
        tract._default_model = model  # store for generate() default
    return tract
```

Key decision: `api_key=None` means "don't auto-configure LLM" (existing behavior preserved). The LLM is only set up when the user explicitly passes `api_key`. This maintains the LLM-optional principle.

Note: `OpenAIClient.__init__()` already falls back to `TRACT_OPENAI_API_KEY` env var. But `Tract.open()` should only auto-configure if the user opts in with an explicit parameter. Env-var-only LLM config would be too magical.

### Pattern 3: Generation Config Auto-Capture

**What:** Automatically build `generation_config` from the LLM request parameters and response.
**When to use:** Every `chat()`/`generate()` call.

```python
def _build_generation_config(
    self,
    response: dict,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> dict:
    config = {}
    # From response (authoritative -- tells us what model was actually used)
    if "model" in response:
        config["model"] = response["model"]
    elif model is not None:
        config["model"] = model
    # From request params
    if temperature is not None:
        config["temperature"] = temperature
    if max_tokens is not None:
        config["max_tokens"] = max_tokens
    # Any extra kwargs passed to generate()
    for k, v in kwargs.items():
        if v is not None:
            config[k] = v
    return config
```

### Pattern 4: Error Handling for Missing LLM

**What:** Clear error when `chat()`/`generate()` called without LLM configured.
**When to use:** Guard at the top of `generate()`.

```python
def generate(self, **kwargs) -> ChatResponse:
    if not hasattr(self, "_llm_client"):
        from tract.llm.errors import LLMConfigError
        raise LLMConfigError(
            "No LLM client configured. Call configure_llm(client) or "
            "pass api_key to Tract.open() before using chat()/generate()."
        )
    ...
```

Use `LLMConfigError` (already exists in `tract.llm.errors`). This is consistent with how `OpenAIClient.__init__()` raises the same error for missing API key.

### Anti-Patterns to Avoid

- **Separate conversation module:** Don't create `src/tract/conversation.py`. The methods belong on `Tract` directly, following the pattern of `system()`/`user()`/`assistant()`.
- **Async support:** Don't add async `achat()`/`agenerate()` in Phase 9. The entire codebase is sync. Async can be a future phase.
- **Streaming:** Don't add streaming support. `OpenAIClient.chat()` is sync/non-streaming. Streaming is a separate concern.
- **Provider detection:** Don't try to auto-detect whether to use `to_openai()` or `to_anthropic()`. Always use `to_dicts()` (which is the OpenAI-compatible format). The `OpenAIClient` speaks OpenAI format. Users with Anthropic clients can use `generate()` with a custom client.
- **Closing LLM client on Tract.close():** When `Tract.open()` auto-creates the client, it should also close it on `Tract.close()`. Track this with a `_owns_llm_client: bool` flag.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Extracting text from LLM response | Manual `response["choices"][0]["message"]["content"]` | `OpenAIClient.extract_content(response)` | Already handles edge cases, raises proper errors |
| Extracting usage from LLM response | Manual `response.get("usage")` | `OpenAIClient.extract_usage(response)` | Consistent extraction |
| Normalizing usage dict to TokenUsage | Custom normalization | `Tract._normalize_usage_dict()` (line 1798) | Already handles OpenAI and Anthropic formats |
| Retry logic for LLM calls | Custom retry | `OpenAIClient.chat()` with tenacity | Already has exponential backoff, jitter, retryable status detection |
| Message format conversion | Manual list comprehension | `CompiledContext.to_dicts()` | Phase 8 already built this |

**Key insight:** Every building block for chat()/generate() already exists. The value of Phase 9 is purely in composition and the thin response object.

## Common Pitfalls

### Pitfall 1: LLM Client Lifecycle Management

**What goes wrong:** If `Tract.open(api_key=...)` creates an `OpenAIClient`, but `Tract.close()` doesn't close it, the httpx client leaks.
**Why it happens:** The existing `configure_llm(client)` takes an externally-owned client, so `Tract.close()` rightly doesn't close it. But `Tract.open(api_key=...)` creates an internally-owned client.
**How to avoid:** Track ownership with `self._owns_llm_client: bool`. In `Tract.close()`, if `_owns_llm_client` is True and `_llm_client` exists, call `self._llm_client.close()`.
**Warning signs:** ResourceWarning about unclosed httpx.Client in tests.

### Pitfall 2: Detached HEAD Guard on chat()/generate()

**What goes wrong:** `chat()` calls `self.user()` which calls `self.commit()` which raises `DetachedHeadError`. The error message doesn't mention chat().
**Why it happens:** The existing guard is on `commit()` (line 442). When called through chat(), the user gets a confusing error.
**How to avoid:** Either (a) add a guard at the top of chat()/generate() with a clearer message, or (b) let the existing guard work -- `DetachedHeadError` is already descriptive enough. Option (b) is simpler and avoids duplicating guards.
**Recommendation:** Let existing guard work. The error says "Cannot commit in detached HEAD state" which is clear enough.

### Pitfall 3: record_usage() During Batch

**What goes wrong:** If `chat()` or `generate()` is called inside a `batch()` context manager, `record_usage()` might not work correctly because the compile cache was cleared on batch entry.
**Why it happens:** `batch()` clears the cache and defers session commits.
**How to avoid:** Document that `chat()`/`generate()` should NOT be called inside `batch()`. They are high-level convenience methods. Add a guard: `if self._in_batch: raise TraceError("chat()/generate() cannot be used inside batch()")`.

### Pitfall 4: Policy/Orchestrator Triggers During chat()

**What goes wrong:** `chat()` calls `user()` (which is `commit()`), then `generate()` calls `compile()` and then `assistant()` (which is `commit()` again). Each of these triggers policy evaluation and orchestrator checks. That's 2 commits + 1 compile = 3 trigger evaluations per chat() call.
**Why it happens:** The trigger system fires on every commit() and compile().
**How to avoid:** This is actually correct behavior -- policies should evaluate after each step. But if it causes performance issues, consider wrapping the internal operations with `self._orchestrating = True` to suppress intermediate triggers. For Phase 9, let triggers fire naturally -- they're designed for this.
**Recommendation:** Let triggers fire naturally. The orchestrator has its own recursion guard (`_orchestrating` flag). If users want to suppress triggers, they can pause policies.

### Pitfall 5: generate() kwargs Collision with LLM Client kwargs

**What goes wrong:** `generate(temperature=0.7)` -- the `temperature` kwarg needs to be forwarded to `self._llm_client.chat()`, but other kwargs like `message=` or `metadata=` should NOT be forwarded.
**Why it happens:** `generate()` accepts both LLM-specific kwargs (temperature, model, max_tokens) and potentially tract-specific kwargs.
**How to avoid:** Use explicit parameters for known LLM args. Don't use `**kwargs` passthrough. Define: `generate(*, model=None, temperature=None, max_tokens=None)` as explicit params. If users need exotic LLM kwargs, they can use the lower-level `compile() + client.chat()` path.

### Pitfall 6: Auto-Generated Commit Message for Assistant Response

**What goes wrong:** The auto-message for assistant commits might be too long or reveal sensitive content.
**Why it happens:** Phase 8's `_auto_message()` generates messages like `dialogue: The answer to your question is...` which truncates at 72 chars.
**How to avoid:** The existing auto-message system already handles this with truncation. `assistant()` delegates to `commit()` which calls `_auto_message()`. No special handling needed.

## Code Examples

### Example 1: chat() Method Implementation

```python
# Source: Direct composition of existing Tract methods
def chat(
    self,
    text: str,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    message: str | None = None,
    metadata: dict | None = None,
) -> ChatResponse:
    """Send a user message and get an LLM response in one call.

    Commits the user message, compiles context, calls the LLM,
    commits the assistant response, and records usage.

    Args:
        text: The user message text.
        model: Model override for this call.
        temperature: Temperature override.
        max_tokens: Max tokens override.
        message: Optional commit message for the user commit.
        metadata: Optional metadata for the user commit.

    Returns:
        ChatResponse with .text, .usage, .commit_info, .generation_config.

    Raises:
        LLMConfigError: If no LLM client is configured.
        DetachedHeadError: If HEAD is detached.
    """
    # Commit user message
    self.user(text, message=message, metadata=metadata)
    # Delegate to generate()
    return self.generate(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

### Example 2: generate() Method Implementation

```python
# Source: Direct composition of existing Tract methods
def generate(
    self,
    *,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> ChatResponse:
    """Compile context, call LLM, commit response, record usage.

    Assumes the user message has already been committed.
    Use chat() for the all-in-one path.

    Returns:
        ChatResponse with .text, .usage, .commit_info, .generation_config.

    Raises:
        LLMConfigError: If no LLM client is configured.
    """
    if not hasattr(self, "_llm_client"):
        from tract.llm.errors import LLMConfigError
        raise LLMConfigError(
            "No LLM client configured. Pass api_key to Tract.open() "
            "or call configure_llm(client)."
        )

    if self._in_batch:
        raise TraceError("chat()/generate() cannot be used inside batch()")

    # 1. Compile context
    compiled = self.compile()
    messages = compiled.to_dicts()

    # 2. Call LLM
    llm_kwargs = {}
    if model is not None:
        llm_kwargs["model"] = model
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if max_tokens is not None:
        llm_kwargs["max_tokens"] = max_tokens

    response = self._llm_client.chat(messages, **llm_kwargs)

    # 3. Extract content and usage
    from tract.llm.client import OpenAIClient
    text = OpenAIClient.extract_content(response)
    usage_dict = OpenAIClient.extract_usage(response)

    # 4. Build generation_config from response + request params
    gen_config = self._build_generation_config(
        response, model=model, temperature=temperature, max_tokens=max_tokens
    )

    # 5. Commit assistant response
    commit_info = self.assistant(text, generation_config=gen_config)

    # 6. Record usage
    usage = None
    if usage_dict:
        usage = self._normalize_usage_dict(usage_dict)
        self.record_usage(usage)

    return ChatResponse(
        text=text,
        usage=usage,
        commit_info=commit_info,
        generation_config=gen_config,
    )
```

### Example 3: ChatResponse Dataclass

```python
from dataclasses import dataclass
from tract.models.commit import CommitInfo
from tract.protocols import TokenUsage

@dataclass(frozen=True)
class ChatResponse:
    """Response from Tract.chat() or Tract.generate().

    Attributes:
        text: The assistant's response text.
        usage: Token usage from the API, or None if not reported.
        commit_info: CommitInfo for the assistant's commit.
        generation_config: The generation config captured from the request/response.
    """
    text: str
    usage: TokenUsage | None
    commit_info: CommitInfo
    generation_config: dict
```

### Example 4: LLM Config on Tract.open()

```python
# In Tract.open():
@classmethod
def open(
    cls,
    path: str = ":memory:",
    *,
    tract_id: str | None = None,
    config: TractConfig | None = None,
    tokenizer: TokenCounter | None = None,
    compiler: ContextCompiler | None = None,
    verify_cache: bool = False,
    # New LLM config params
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> Tract:
    # ... existing initialization ...

    # Auto-configure LLM if api_key provided
    if api_key is not None:
        from tract.llm.client import OpenAIClient
        client = OpenAIClient(
            api_key=api_key,
            base_url=base_url,
            default_model=model or "gpt-4o-mini",
        )
        tract.configure_llm(client)
        tract._owns_llm_client = True
        tract._default_model = model

    return tract
```

### Example 5: User-Facing Usage

```python
from tract import Tract

# Zero-ceremony chat
with Tract.open(api_key="sk-...") as t:
    t.system("You are a helpful assistant.")
    r = t.chat("What is Python?")
    print(r.text)
    print(r.usage)  # TokenUsage(prompt_tokens=..., ...)

# With explicit control
with Tract.open(api_key="sk-...") as t:
    t.system("You are a helpful assistant.")
    t.user("What is Python?")
    # Can inspect context here
    compiled = t.compile()
    print(f"Sending {compiled.token_count} tokens")
    # Then generate
    r = t.generate(temperature=0.7)
    print(r.text)
```

## State of the Art

| Old Approach (Before Phase 9) | New Approach (After Phase 9) | Impact |
|-------------------------------|------------------------------|--------|
| `from tract.llm import OpenAIClient; client = OpenAIClient(...); t.configure_llm(client)` | `t = Tract.open(api_key="...")` | 3 lines -> 1 parameter |
| `t.user(q); c = t.compile(); msgs = c.to_dicts(); resp = client.chat(msgs); text = resp["choices"][0]["message"]["content"]; t.assistant(text); t.record_usage(resp["usage"])` | `r = t.chat(q)` | ~7 lines -> 1 line |
| Manual `generation_config={"model": ..., "temperature": ...}` on every assistant commit | Automatic from LLM response | Eliminates manual config tracking |
| Manual `record_usage()` after every LLM call | Automatic inside `chat()`/`generate()` | Eliminates forgetting to record usage |

**Not deprecated:**
- `configure_llm()` remains for users who want to inject custom LLM clients
- `compile()` + manual client call remains for users who need full control
- `record_usage()` remains for users recording usage from external LLM calls

## Design Decisions to Make

### D1: Where to put ChatResponse

**Options:**
1. In `src/tract/protocols.py` alongside `CompiledContext`, `TokenUsage`, `Message`
2. In a new `src/tract/models/conversation.py`

**Recommendation:** Option 1. `protocols.py` is the established home for frozen dataclass output types. `ChatResponse` is the same kind of thing -- a frozen output container. No new file needed.

### D2: Should generate() accept message= and metadata= for the assistant commit?

**Options:**
1. Yes -- `generate(message="my custom msg", metadata={...})`
2. No -- keep generate() focused on LLM params only

**Recommendation:** Option 1. The `assistant()` shorthand already supports these, and generate() should expose them. Users might want to annotate auto-generated commits.

### D3: Should chat() accept name= for the user message?

**Options:**
1. Yes -- `chat("hello", name="Alice")`
2. No -- use `user("hello", name="Alice"); generate()` for that

**Recommendation:** Option 1. Minor but useful. chat() should accept all user()-compatible params.

### D4: How to handle extract_content for non-OpenAI clients

**What:** `OpenAIClient.extract_content()` is a static method that assumes OpenAI response format. If a user provides a custom `LLMClient` via `configure_llm()`, their response might have a different structure.

**Recommendation:** Use `OpenAIClient.extract_content()` as default. Since the `LLMClient` protocol requires returning a dict with `choices` key (per protocol definition), this is safe. Custom clients that don't follow this format are not conforming to the protocol.

### D5: Should Tract.open() also accept env vars for LLM config?

**What:** Should `Tract.open()` auto-detect `TRACT_OPENAI_API_KEY` env var even without `api_key=` parameter?

**Recommendation:** No. Auto-configuration from env vars alone would be too magical. Users should explicitly opt in by passing `api_key=...` (even if they pass the env var value). This maintains the "no magic" principle. The env var fallback in `OpenAIClient.__init__()` is fine because the user already explicitly chose to create a client.

However: consider a convenience of `api_key="env"` or a sentinel that means "read from env var". This could be Phase 10 territory. For Phase 9, keep it simple: explicit value only.

### D6: _build_generation_config as a private method vs standalone function

**Recommendation:** Private method on Tract (`self._build_generation_config()`). It needs access to `self._default_model` for fallback. Method is cleaner than passing the fallback as a parameter.

## Open Questions

1. **Should chat()/generate() work inside batch()?**
   - What we know: `batch()` clears cache and defers commits. `generate()` needs compile() which needs cache. `record_usage()` needs cache.
   - What's unclear: Whether there's a valid use case for chat() inside batch().
   - Recommendation: Raise `TraceError` if called inside batch(). This is a convenience method -- batch() users should use the low-level API.

2. **Should the LLM client created by Tract.open() be closed on Tract.close()?**
   - What we know: `configure_llm()` takes external clients that should not be closed. `Tract.open(api_key=...)` creates internal clients that should be.
   - What's unclear: Whether tracking ownership adds too much complexity.
   - Recommendation: Track with `self._owns_llm_client: bool`. Close in `Tract.close()` when True. This is 4 lines of code and prevents resource leaks.

3. **Should generate() forward extra kwargs to the LLM client?**
   - What we know: `OpenAIClient.chat()` accepts `**kwargs` for extra payload params (like `top_p`, `frequency_penalty`, etc.).
   - What's unclear: Whether to expose these on generate().
   - Recommendation: Use explicit params only (model, temperature, max_tokens) for Phase 9. Extra kwargs can be added in Phase 10 (per-operation config). This keeps the API clean and avoids confusion.

## Testing Strategy

### Mock LLM Pattern (Existing)

The codebase already has a well-established mock LLM pattern used in 20+ test files:

```python
class MockLLMClient:
    def __init__(self, responses=None):
        self.responses = responses or ["Default response"]
        self._call_count = 0
        self.last_messages = None

    def chat(self, messages, **kwargs):
        self.last_messages = messages
        text = self.responses[min(self._call_count, len(self.responses) - 1)]
        self._call_count += 1
        return {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": kwargs.get("model", "mock-model"),
        }

    def close(self):
        pass
```

This same pattern should be used for Phase 9 tests.

### Test Categories Needed

1. **ChatResponse model:** Frozen, correct attributes, repr
2. **Tract.open() with LLM params:** api_key creates client, base_url/model forwarded, close() closes client, without api_key leaves no client
3. **chat() happy path:** user commit created, LLM called with compiled messages, assistant commit created, usage recorded, ChatResponse returned
4. **generate() happy path:** same as chat() but without user commit
5. **Auto generation_config:** model/temperature/max_tokens captured from request, model from response takes precedence
6. **Auto usage recording:** usage dict from response auto-recorded, TokenUsage in ChatResponse matches
7. **Error cases:** no LLM configured, LLM error propagation, inside batch(), detached HEAD
8. **Integration:** multi-turn chat (chat -> chat -> chat), generate after user(), chat() after system()

## Sources

### Primary (HIGH confidence)
- Direct code reading of `src/tract/tract.py` -- all existing methods verified
- Direct code reading of `src/tract/llm/client.py` -- OpenAIClient API verified
- Direct code reading of `src/tract/llm/protocols.py` -- LLMClient protocol verified
- Direct code reading of `src/tract/protocols.py` -- dataclass patterns verified
- Direct code reading of `src/tract/exceptions.py` -- error hierarchy verified
- `.planning/VISION.md` -- original design intent for chat()/generate()
- `.planning/ROADMAP.md` -- success criteria and requirements
- `.planning/REQUIREMENTS.md` -- CONV-01, CONV-02, CONV-03, LLM-01, LLM-02, LLM-03

### Secondary (MEDIUM confidence)
- [OpenAI ChatCompletion response format](https://platform.openai.com/docs/api-reference/chat) -- response.choices[0].message.content pattern
- [LiteLLM response design](https://docs.litellm.ai/docs/completion/output) -- unified response object patterns

### Tertiary (LOW confidence)
- None -- all findings verified from source code

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all components verified in source code, no new libraries
- Architecture: HIGH -- pure composition of existing methods, patterns established by Phase 8
- Pitfalls: HIGH -- identified from direct code analysis of batch(), triggers, and lifecycle
- Response object design: HIGH -- follows existing frozen dataclass convention

**Research date:** 2026-02-19
**Valid until:** indefinite (internal architecture research, not external library)
