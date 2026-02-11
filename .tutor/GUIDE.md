# Trace Tutorial Writing Guide

This guide defines how tutorials for the Trace project should be written. It is generated from `.tutor/config.yaml` and tailored to this project's specific audience and priorities.

## Audience

**Profile**: Solo developer building Trace. Tutorials serve as your own future reference — a way to crystallize design thinking and ensure you can re-enter the codebase with full context after time away.

**Technical level**: Intermediate. Assume fluency in Python (no need to explain classes, decorators, context managers, type hints, dataclasses, etc.).

**What to explain**: SQLAlchemy patterns, Pydantic model design, Git/DAG concepts, and tiktoken usage should be introduced when first used in a tutorial. One or two sentences of context is sufficient — don't write a primer, but don't assume deep familiarity either.

**Adaptation rules**:
- Assume Python language fluency — never explain Python syntax or stdlib patterns
- Introduce project-specific libraries (SQLAlchemy, Pydantic, tiktoken) on first use with a brief sentence of context
- Focus heavily on project-specific design decisions — these are the things you'll forget
- Code fragments are fine if surrounding context makes them clear; full runnable examples for key APIs
- Explain the *why* before the *what* — future-you remembers code faster than reasoning

## Section Weights

Every tutorial should cover these areas at the specified depth:

| Focus Area | Weight | What to Write |
|---|---|---|
| **Conceptual** | heavy | Dedicated section. Explain the problem being solved, the mental model, and the key abstractions. Use analogies to Git concepts where appropriate. Define all Trace-specific terms. |
| **Design choices** | heavy | Dedicated section. Document alternatives considered, why each was rejected, and the tradeoffs of the chosen approach. Include links to planning docs where relevant. This is the highest-value content for future reference. |
| **Implementation** | heavy | Dedicated section. Walk through key functions, data flow, and class interactions. Show how the pieces connect at the code level. Reference specific files and line ranges. |
| **Connections** | heavy | Dedicated section. Map how this piece fits into the overall Trace architecture. What depends on it? What does it depend on? What changes in one place would ripple here? |
| **Examples** | heavy | Dedicated section. Show concrete usage with explanations. Include both simple "hello world" examples and realistic multi-step scenarios. Show expected output. |

## Style Rules (Narrative)

Tutorials use a **narrative walkthrough** style:

- Write in second person ("you") or first-person plural ("we")
- Follow a logical progression: "First we need to... because... then we..."
- Use transitions that explain causality: "Because X works this way, we need Y"
- Group related concepts into a story arc, not a reference list
- Start each major section with *why* it exists before *how* it works
- End sections with a bridge to what comes next
- Avoid bullet-point-heavy sections — prefer flowing paragraphs for explanations, with code blocks interspersed
- Use bullet points only for enumerating concrete items (API methods, config options, error types)

## Frontmatter Requirements

Every tutorial must start with this exact YAML frontmatter format:

```yaml
---
date: YYYY-MM-DD
summary: "One-line description of what this tutorial covers"
audience: intermediate
---
```

## Naming Convention

Pattern: `{nn}-{slug}.md` where `{nn}` is a zero-padded phase number.

Use letter suffixes for sub-topic deep dives within a phase:

```
tutorials/
  01-foundations-overview.md              # Phase 1 overview
  01a-data-models-and-storage.md         # Phase 1 sub-topic
  01b-engine-layer.md                    # Phase 1 sub-topic
  01c-repo-api-and-design-patterns.md    # Phase 1 sub-topic
  01.1-incremental-compile-cache-and-token-tracking.md  # Sub-phase
  02-linear-history.md                   # Phase 2 overview
  02a-checkout-and-head.md               # Phase 2 sub-topic
```

## Quality Bar

After reading a tutorial, you (the author, returning after weeks away) should be able to:

- **Understand WHY** each design decision was made without re-reading planning docs
- **Follow the implementation logic** without opening the source code first
- **Ask informed questions** about alternatives and tradeoffs — the tutorial should surface enough context to reason about changes
- **Know where to look** in the codebase if you need to modify something
- **Remember the constraints** that shaped the design (what you tried and rejected, what edge cases matter)

Prioritize **clarity and depth over brevity**. A tutorial that's too long but complete is more valuable than one that's concise but leaves gaps.

## Presets

When invoking `/tutor:write`, you can use presets:

- **quick**: Light conceptual coverage, skip implementation details. Good for minor features or config changes.
- **deep-dive**: Maximum depth on everything. Use for new phases, major refactors, or architecturally significant work.

Default (no preset): Uses the standard heavy weights defined above.
