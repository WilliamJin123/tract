# Applications: Patterns Built from Substrate

Applications are common workflow patterns expressed as rules over the commit
DAG. They ship as pre-built rule templates but could be authored from scratch
by a developer or LLM. None require special-cased code paths.

## Classification

| Application | Substrate expression |
|---|---|
| Workflow stage | Branch + set of stage-scoped rules committed to it |
| Stage transition | Decomposed rules on source branch (gates → work → handoff) |
| Compression trigger | Rule: condition=token_count > threshold, action=collapse subgraph |
| Tool compaction | Rule: auto-summarize tool result commits above N tokens |
| Quality gate | Rule: condition checks output, action=approve/retry/reject |
| Secret redaction | Rule: condition=pattern match, action=edit commit content |
| Data preservation | Rule: scoped to specific commits, action=block compress/gc |
| Promotion loop | Meta-rule: observe repeated LLM decisions → create deterministic rule |
| GC policy | Rule: condition=orphaned + age > retention, action=delete |
| Priority (PINNED) | Rule: scoped to commit, action=never compress/skip |
| Priority (SKIP) | Rule: scoped to commit, action=exclude from compiled output |
| Cherry-pick | Just append with content sourced from existing node |
| Fan-out / fan-in | Transition rule spawns N branches; merge rule consolidates on completion |
| Urgent broadcast | Rule: on tag=urgent, cherry-pick commit to all active sibling branches |
| Human approval gate | Rule: condition=requires_tag("approved"), block transition until present |

## Concrete Schema Examples

A few rules in their actual RuleContent form:

```python
# Config — committed on product_research branch
RuleContent(name="temp", trigger="active",
    condition=None,
    action={"type": "set_config", "key": "temperature", "value": 0.3})

# Data preservation — committed on workflow root (inherited everywhere)
RuleContent(name="preserve_pricing", trigger="compress",
    condition={"type": "tag", "tag": "pricing", "present": True},
    action={"type": "block"})

# Approval gate — committed on lander_pages branch
RuleContent(name="client_gate", trigger="transition:ads",
    condition=None,
    action={"type": "require",
            "condition": {"type": "tag", "tag": "client_approved", "present": True}})

# Transition handoff — committed on product_research branch
RuleContent(name="to_lander", trigger="transition:lander_pages",
    condition=None,
    action={"type": "compile_filter", "mode": "selective",
            "include_tags": ["key_finding", "pricing", "brand_voice"]})

# Secret redaction — committed on workflow root
RuleContent(name="redact_secrets", trigger="commit",
    condition={"type": "pattern", "regex": "API_KEY=\\S+"},
    action={"type": "operation", "op": "edit",
            "params": {"replacement": "API_KEY=***"}})

# Urgent broadcast — committed on a shared ancestor branch
RuleContent(name="urgent_broadcast", trigger="commit",
    condition={"type": "tag", "tag": "urgent", "present": True},
    action={"type": "operation", "op": "cherry_pick",
            "params": {"to": "active_siblings"}})
```

## Example: Ecommerce Workflow

The entire ecommerce pipeline is a collection of rules on branches:

```
workflow root (main branch):
  rules:
    - redact_secrets: trigger=commit, pattern match → edit
    - preserve_pricing: trigger=compress, tag=pricing → block
    - preserve_competitor: trigger=compress, tag=competitor_data → block

stage: product_research (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.3)
    - compaction: trigger=active, set(tool_compaction, true)
  rules:
    - exit_audit: trigger=transition, action=llm("audit precompaction commits")
    - on_error: trigger=commit, condition=tag(error), action=set(auto_summarize, false)
  transition → lander_pages:
    - to_lander: trigger=transition:lander_pages,
      action=compile_filter(selective, include=[key_finding, pricing, brand_voice])

stage: lander_pages (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.7)
    - no_summarize: trigger=active, set(auto_summarize, false)
  rules:
    - brand_check: trigger=compile, condition=llm("follows brand guidelines?"),
      action=block
    - client_gate: trigger=transition:ads, action=require(tag=client_approved)
  transition → ads:
    - to_ads: trigger=transition:ads,
      action=compile_filter(summarized, target_tokens=300)

stage: ads (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.6)
  rules:
    - platform_specs: trigger=commit, condition=llm("verify platform specs"),
      action=block("spec violation")
    - budget_cap: trigger=commit,
      condition=threshold(campaign_spend > 1000), action=block("over budget")
  transition → metric_analysis:
    - to_metrics: trigger=transition:metric_analysis,
      action=compile_filter(selective, include=[campaign_id, spend, creative_variant])

stage: metric_analysis (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.1)
  rules:
    - preserve_numbers: trigger=compress, condition=tag(numerical_data),
      action=block
  transition → lander_pages:
    - to_lander_v2: trigger=transition:lander_pages,
      action=compile_filter(summarized, target_tokens=500,
                            preserve_tags=[conversion_data])
  transition → ads:
    - to_ads_v2: trigger=transition:ads,
      action=compile_filter(selective, include=[ctr_data, audience_segment, budget])
```

The macro workflow graph emerges from the transition rules. No separate
"workflow graph" data structure.

## Example: Coding Workflow

```
stage: design (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.7)
    - summarize: trigger=active, set(auto_summarize, true)
  rules:
    - explore: trigger=commit, condition=llm("explored multiple approaches?"),
      action=llm("flag if committing to approach too early")
  transition → implementation:
    - to_impl: trigger=transition:implementation,
      action=compile_filter(selective, include=[design_decision, api_contract, spec])

stage: implementation (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.2)
    - summarize: trigger=active, set(auto_summarize, true)
  rules:
    - test_after_change: trigger=commit, condition=llm("significant code change?"),
      action=operation(run_tests)
    - design_gap: trigger=commit, condition=llm("design gap found?"),
      action=operation(branch, {name: "design_rethink"})
  transition → validation:
    - to_validation: trigger=transition:validation,
      action=compile_filter(same_context)

stage: design_rethink (branch off IMPLEMENTATION — inherits its rules)
  configs:
    - temp: trigger=active, set(temperature, 0.7)  # overrides impl's 0.2
  rules:
    - (inherits test_after_change from implementation via DAG ancestry)
  transition → implementation:
    - back_to_impl: trigger=transition:implementation,
      action=compile_filter(selective, include=[revised_design, updated_spec])

stage: validation (same_context from implementation — same branch)
  rules:
    - preserve_errors: trigger=commit, condition=tag(error_trace),
      action=set_config(auto_summarize, false)
    - on_pass: trigger=commit, condition=tag(all_tests_pass),
      action=llm("prepare completion summary")
    - on_fail: trigger=commit, condition=tag(test_failure),
      action=operation(branch, {name: "implementation",
                                handoff: "failure context"})
```

Note: design_rethink branches off implementation (not root), so it inherits
implementation's rules automatically. Overrides only what's different.

## Example: AI Research Workflow

```
stage: ingest (branch off root)
  configs:
    - compaction: trigger=active, set(tool_compaction, false)
  rules:
    - verbatim: trigger=commit, action=set_config(auto_summarize, false)
    - auto_tag: trigger=commit, condition=pattern("arxiv|doi"),
      action=llm("extract and tag: topic, author, date, methodology")
  transition → organize:
    - to_organize: trigger=transition:organize,
      action=compile_filter(selective, include=[abstract, key_finding, metadata])

stage: organize (branch off root)
  configs:
    - summarize: trigger=active, set(auto_summarize, true)
    - compaction: trigger=active, set(tool_compaction, true)
  rules:
    - cross_ref: trigger=commit, condition=llm("related to existing papers?"),
      action=llm("create cross-reference annotations")
  transition → synthesize:
    - to_synth: trigger=transition:synthesize,
      action=compile_filter(summarized, target_tokens=2000,
                            preserve_tags=[cross_reference, theme, gap_identified])

stage: synthesize (branch off root)
  configs:
    - temp: trigger=active, set(temperature, 0.8)
  rules:
    - cite: trigger=compile, condition=llm("every claim has citation?"),
      action=block("uncited claim")
    - new_insight: trigger=commit, condition=tag(novel_insight),
      action=llm("tag and evaluate if this should feed back to organize")
```

Multi-stage concurrency (ingest during organize): agent switches HEAD to
ingest branch, works there with ingest rules, switches back. Or use a
second agent on the ingest branch (fan-out pattern).

## Key Insight

None of these applications require code changes to tract's substrate.
They're all rules — different triggers, conditions, and actions composed
over the same graph primitives. A new workflow (e.g., customer support,
legal document review) is just a new set of rules, not new operations.
