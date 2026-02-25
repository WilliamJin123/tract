# Design: Reasoning Trace Handling

**Date**: 2026-02-25
**Status**: Approved

## Problem

LLM reasoning models (OpenAI o1/o3, Anthropic Claude thinking, Cerebras GLM/GPT-OSS, DeepSeek R1) produce intermediate thinking tokens before their final response. These reasoning traces explain *why* the model gave a particular answer.

Tract currently discards reasoning traces. The `generate()` flow extracts only the final text content and tool calls from the LLM response. The raw response is stored ephemerally on `ChatResponse.raw_response` but never committed. Once the `ChatResponse` goes out of scope, the model's chain of thought is lost.

This is like git storing only the final file but not the diff that produced it.

## Why It Matters

- **Provenance**: "Why did the model say X?" is the #1 debugging question. Reasoning answers it.
- **Token accounting**: Reasoning tokens count toward completion tokens even when hidden. Without the text, token costs are unexplainable.
- **Compression optionality**: Separate reasoning commits let the compression engine treat reasoning differently from dialogue (compress reasoning but keep conclusion, or vice versa).
- **Time-travel**: `tract.log()` can show "what was the model thinking at step N?"
- **Multi-agent replay**: Agent B can inspect Agent A's reasoning, not just its output.

## Provider Landscape

| Provider | Format | Where reasoning lives |
|----------|--------|-----------------------|
| Cerebras (parsed) | `reasoning` field on `message` | `response.choices[0].message.reasoning` |
| Cerebras/DeepSeek (raw) | `<think>...</think>` tags in content | Mixed into `message.content` |
| Anthropic Claude | `thinking` content blocks | `response.content[].type == "thinking"` |
| OpenAI o1/o3 | `reasoning_content` field | `response.choices[0].message.reasoning_content` |
| Cerebras (hidden) | Dropped entirely | Tokens counted, text lost |

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auto-commit in generate()? | Yes, by default | Reasoning is first-class content; opt-out via `reasoning=False` per-call or `commit_reasoning=False` on Tract.open() |
| Include in compile()? | Excluded by default (SKIP priority) | Avoids bloating the next LLM call; `compile(include_reasoning=True)` overrides |
| Commit order? | Before assistant | Matches execution order (think, then answer); commit chain reads `...user -> reasoning -> assistant` |
| Extraction mechanism? | Duck-typed optional on LLM client | `extract_reasoning()` is not a protocol requirement; if client doesn't have it, reasoning handling is skipped entirely |
| Shorthand method? | `t.reasoning(text)` | Matches `t.user()` / `t.assistant()` pattern for manual LLM call workflows |
| Format tracking? | `format` field on ReasoningContent | Records extraction source for debugging and round-tripping |
| Visual distinction? | Dedicated pprint color (dim/muted) | Reasoning is "behind the scenes"; distinct from dialogue in log/diff/pprint |

## Data Model

### ReasoningContent (extended)

Existing model in `src/tract/models/content.py`, extended with a `format` field:

```python
class ReasoningContent(BaseModel):
    content_type: Literal["reasoning"] = "reasoning"
    text: str
    format: Literal["parsed", "raw", "think_tags", "anthropic"] = "parsed"
```

### BUILTIN_TYPE_HINTS update

Change `default_priority` from `"normal"` to `"skip"`:

```python
"reasoning": ContentTypeHints(
    default_priority="skip",       # was "normal"
    default_role="assistant",
    compression_priority=40,
),
```

### ChatResponse (extended)

Two new fields:

```python
@dataclass(frozen=True)
class ChatResponse:
    text: str
    reasoning: str | None = None               # extracted reasoning text
    reasoning_commit: CommitInfo | None = None  # committed reasoning (None if reasoning=False or no reasoning found)
    usage: TokenUsage | None
    commit_info: CommitInfo
    generation_config: LLMConfig
    prompt: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw_response: dict | None = None
```

### Tract.open() / TractConfig

New optional parameter:

```python
Tract.open(
    ...,
    commit_reasoning: bool = True,  # default: auto-commit reasoning traces
)
```

Stored on Tract instance as `self._commit_reasoning: bool`.

## Extraction Protocol

`extract_reasoning()` is a **duck-typed optional** on the LLM client. It is NOT added to the `LLMClient` Protocol class.

```python
# In _generate_once():
reasoning_text = None
reasoning_format = "parsed"
if hasattr(chat_client, "extract_reasoning"):
    result = chat_client.extract_reasoning(response)
    # result is (text, format) tuple or just text string
```

### Built-in auto-detect (on OpenAIClient)

The built-in `OpenAIClient` implements `extract_reasoning()` with a priority chain:

1. **Parsed field**: `response["choices"][0]["message"].get("reasoning")` (Cerebras parsed)
2. **OpenAI reasoning_content**: `response["choices"][0]["message"].get("reasoning_content")` (o1/o3)
3. **Anthropic thinking blocks**: `response["content"]` list with `type="thinking"` blocks
4. **`<think>` tags**: Regex extraction from content text, stripping tags from the content

When `<think>` tags are detected in the content, the extractor strips them from the content so `ChatResponse.text` is clean. The reasoning goes to `ChatResponse.reasoning`, the answer goes to `ChatResponse.text`.

Custom clients that don't implement `extract_reasoning()` skip all reasoning handling silently.

## Commit Flow

### Current flow in _generate_once()

```
compile -> LLM call -> extract text -> extract usage -> commit assistant -> record usage -> return ChatResponse
```

### New flow

```
compile -> LLM call -> extract text -> extract reasoning -> extract usage
  -> [if reasoning_text and commit_reasoning]:
       commit ReasoningContent (APPEND, default SKIP priority)
  -> commit assistant (parent is now reasoning commit if it exists)
  -> record usage -> return ChatResponse
```

The reasoning commit is a regular APPEND. Its parent is the previous HEAD (typically the user message). The assistant commit's parent becomes the reasoning commit. The chain reads:

```
...user -> [reasoning] -> assistant
```

### Per-call opt-out

```python
t.generate(reasoning=False)   # skip reasoning commit for this call
t.chat("hello", reasoning=False)  # same for chat()
```

When `reasoning=False`: reasoning is still extracted onto `ChatResponse.reasoning` (if the client supports it), but NOT committed. `ChatResponse.reasoning_commit` is None.

## Compile Behavior

### Default: reasoning excluded

`compile()` excludes reasoning commits because `BUILTIN_TYPE_HINTS["reasoning"].default_priority` is `"skip"`.

### Override: include_reasoning flag

```python
compiled = t.compile(include_reasoning=True)
```

Implementation: the compiler's `_build_priority_map` checks `include_reasoning`. When True, commits with `content_type="reasoning"` that would get default SKIP are promoted to NORMAL instead. Explicit annotations (user called `t.annotate()`) still take precedence.

### Per-commit override

```python
t.annotate(reasoning_hash, Priority.PINNED, reason="important step")
```

This always works, regardless of `include_reasoning` flag.

## Shorthand Method

```python
t.reasoning("Let me think about this...", format="parsed")
```

Commits a `ReasoningContent` with APPEND operation. Follows the same pattern as `t.user()` / `t.assistant()`:

```python
def reasoning(
    self,
    text: str,
    *,
    format: str = "parsed",
    message: str | None = None,
    metadata: dict | None = None,
) -> CommitInfo:
    content = ReasoningContent(text=text, format=format)
    return self.commit(content, message=message or _auto_message("reasoning", text), metadata=metadata)
```

## Formatting

Reasoning commits get dedicated visual treatment:

- **log()**: `[reasoning]` tag prefix, dim/muted color (e.g., dim cyan)
- **pprint()**: Distinct style from dialogue commits
- **diff()**: Reasoning content shown in muted color

This makes reasoning visually distinct as "behind the scenes" content while remaining inspectable.

## Non-goals (deferred)

- **Query API** (`find_reasoning_traces()`): Users can use `log()` and filter by `content_type` for now.
- **Compression-specific handling**: Reasoning compresses with the normal engine. Custom reasoning compression prompts can be a future enhancement.
- **Hook system** (`t.on("reasoning", handler)`): Can be added later following the existing hook pattern.
- **Reasoning context retention**: Re-sending reasoning back to the same provider (the Cerebras "Reasoning Context Retention" pattern) is out of scope. Users can use `compile(include_reasoning=True)` for this manually.

## Files Changed

| File | Change |
|------|--------|
| `src/tract/models/content.py` | Add `format` field to `ReasoningContent`; update `BUILTIN_TYPE_HINTS` default_priority to "skip" |
| `src/tract/protocols.py` | Add `reasoning` and `reasoning_commit` fields to `ChatResponse` |
| `src/tract/tract.py` | Add `t.reasoning()` shorthand; update `_generate_once()` extraction + commit flow; add `commit_reasoning` param to `open()` and `generate()`/`chat()`; add `reasoning` param to `generate()`/`chat()` |
| `src/tract/llm/client.py` | Add `extract_reasoning()` to `OpenAIClient` with auto-detect |
| `src/tract/engine/compiler.py` | Add `include_reasoning` param to `compile()`; override SKIP for reasoning when set |
| `src/tract/cli/formatting.py` (or `src/tract/formatting.py`) | Distinct color/style for reasoning commits in log/diff/pprint |
| `tests/` | New test file or additions to existing test files |
