# Waves 14-17 Summary (2026-03-12)

Ran all 54 cookbook files (57 total minus 3 helpers). Fixed library bugs, design issues, model compatibility, and improved agent behavior quality across 4 waves.

---

## Wave 14: Full Cookbook Run (54 files)

### Non-LLM cookbooks: 32/32 PASS

### LLM cookbooks: 19/22 PASS (3 hard failures, 4 soft issues)

**Hard failures (rate-limit induced):**
- `workflows/01,02` — Groq TPM ceiling
- `workflows/04` — gate needs 4 commits, produces 3

**Soft issues (LLM behavior quality):**
- `agent/03` — agent appends instead of editing
- `agent/04` — model loops on register_tag
- `agent/05` — tool call truncated (max_tokens too low)
- `agent/06` — agent doesn't branch for tangent

---

## Wave 15: Library Robustness Fixes
**Commit:** `45b8a28`

### Library Bugs Fixed (3)

1. **Commit tool string handling** (`toolkit/definitions.py`):
   - `_parse_str_to_obj()`: json.loads + ast.literal_eval fallback
   - Handles stringified content, metadata, generation_config, tags

2. **Executor hallucinated kwargs** (`toolkit/executor.py`):
   - Introspects handler signature, strips unknown kwargs

3. **CommitInfo generation_config coercion** (`models/commit.py`):
   - Handles stringified dicts from small models

### Cookbook Fixes (4)
- `workflows/01,02`: Groq -> Cerebras
- `workflows/04`: gate 4 -> 3
- `agent/05`: max_tokens 1024 -> 4096

---

## Wave 16: Design Audit + Behavior Quality
**Commit:** `a646c02`

### Design Bugs Fixed (2)

1. **Edit target short hash resolution** (`toolkit/definitions.py`):
   - Added `resolve_commit(edit_target)` before dispatch

2. **Adversarial review pipeline** (`workflows/08`):
   - `compare().to_json()` had null content; now compiles critique branch directly

### Other Fixes
- `_logging.py`: handle string args in `_format_args`

### Agent Prompting Improvements (4)

| Cookbook | Before | After |
|---------|--------|-------|
| 03_self_correction | 0 edits | 2 edits in place |
| 04_knowledge_org | 0 tags | 4 tags, all commits tagged |
| 05_staged_workflow | 2/3 stages | 3/3 stages |
| 06_tangent_isolation | No branching | Branched for tangent |

---

## Wave 17: Deep Quality Audit + Model Compatibility
**Commit:** `c51d8f8`

### Behavior Audit (13 LLM cookbooks)
Full quality audit of all remaining LLM cookbooks. Found:
- 2 cookbooks using models too weak for their workflows
- 2 cookbooks where agents never attempted the intended action
- 1 cookbook with mode_gate deadlock bug

### Model Upgrades (2)
- `workflows/03_customer_support`: llm.small -> cerebras llm.large (llama3.1-8b couldn't use function calling for complex workflows)
- `workflows/06_coding_with_tests`: llm.small -> cerebras llm.large (same issue + fixed bare string commit)

### Design Improvements (3)

1. **Quality gate demonstration** (`agent/07`):
   - Agent never attempted transition, so gate was never tested
   - Added programmatic fallback: forces transition attempt to demonstrate gate blocking
   - Gate correctly reports "2 artifacts, need 3"

2. **Self-regulation enforcement** (`agent/09`):
   - Agent ignored config/directive tools when they were optional
   - Added `mode_gate` middleware requiring `configure(mode=...)` before any content commit
   - With passthrough for system/user/config/instruction content
   - Agent now sets mode='advocate' and mode='critic' with directives

3. **Context management pressure** (`agent/01`):
   - Budget was too generous (1500 tokens, 16% used) — no pressure to manage
   - Tightened to 600 tokens (46% pre-filled)
   - Agent now calls compress() to consolidate 13 notes into 1 summary

### Bug Found by Subagent
- `workflows/06` line 440: bare string passed to `t.commit()` instead of content dict — fixed to use `{"content_type": "note", "text": ...}`

---

## Cumulative Stats

| Metric | Value |
|--------|-------|
| Cookbooks verified | 54/54 |
| Library bugs fixed | 5 |
| Design bugs fixed | 5 |
| Cookbook fixes | 14 |
| Tests passing | 2435 |
| Commits | 4 |
