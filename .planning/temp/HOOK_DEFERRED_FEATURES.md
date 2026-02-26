# Hook System — Deferred Features

Three features were designed but never wired. All prerequisites exist (prompts, helpers, schema columns). Each is independent.

## 1. two_stage=True on compress()
- **What:** Generate guidance before summaries (judgment/execution split)
- **Prerequisites ready:** `prompts/guidance.py` (COMPRESS_GUIDANCE_SYSTEM, build_compress_guidance_prompt), GuidanceMixin on PendingCompress, `_two_stage` attr stored on pending
- **Wire:** In `compress_range()`, before summary generation loop: if two_stage, call LLM with guidance prompt, store result on `pending.guidance`/`guidance_source="llm"`, thread guidance into `build_summarize_prompt()` as instructions
- **Size:** ~30 lines in `operations/compression.py`, ~10 lines in `tract.py`

## 2. improve=True on user()/assistant()/system()
- **What:** Commit original, then LLM-improve and EDIT
- **Prerequisites ready:** `prompts/improve.py` (IMPROVE_CONTENT_SYSTEM, build_improve_prompt), `hooks/improve.py` (_improve_content helper, fixed), schema columns for original/effective instructions
- **Wire:** Add `improve: bool = False` to shorthand methods. When True: commit original, call LLM with improve prompt, call `_improve_content(tract, original_hash, improved_text)`. LLM failure → keep original, emit warning.
- **Size:** ~20 lines per method (3 methods), ~10 lines shared helper

## 3. retry()/validate() on PendingCompress + PendingMerge
- **What:** LLM-powered re-generation of individual summaries/resolutions
- **Prerequisites ready:** `auto_retry()` loop in `hooks/retry.py` (full logic, just needs stubs replaced), `_public_actions` ready to re-add methods
- **Wire:** `retry(index, guidance)` needs the LLM client from `pending.tract`. Call `_summarize_group()` with guidance injected, replace `self.summaries[index]`. `validate()` needs quality criteria (token count check, retention check). Re-add to `_public_actions` after wiring.
- **Size:** ~40 lines for retry, ~20 lines for validate, per subclass
- **Dependency:** Merge retry also needs conflict resolution re-generation

## Execution order
1 → 2 → 3 (increasing complexity, each independent)
