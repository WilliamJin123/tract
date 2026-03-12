"""Config Resolution Chain

Quick reference for the 4-level LLM config resolution:
  Level 1 (highest): Sugar params — chat(temperature=0.9, model="...")
  Level 2: llm_config= — LLMConfig object per call
  Level 3: Operation config — configure_operations(chat=LLMConfig(...))
  Level 4 (lowest): Tract default — Tract.open(default_config=LLMConfig(...))

Each field resolves independently: temperature from level 1, model from level 4, etc.
Every resolved config is auto-captured on assistant commits for provenance.

Also covers: per-operation clients, cross-framework aliases, token budget config.
All examples below are structural — no LLM calls. See source files for live demos.
"""

from tract import LLMConfig, OperationConfigs, TractConfig, TokenBudgetConfig, Tract


def main():
    # =================================================================
    # 1. LEVEL 4 — Tract-level default (lowest priority)
    # =================================================================
    # Every operation inherits this unless overridden at a higher level.

    tract_default = LLMConfig(model="gpt-4o", temperature=0.5, top_p=0.95)

    # Two ways to set the default:
    #   Tract.open(default_config=tract_default)          # explicit LLMConfig
    #   Tract.open(api_key="...", model="gpt-4o")         # sugar on open()

    # =================================================================
    # 2. LEVEL 3 — Per-operation config
    # =================================================================
    # Override settings per operation type (chat, compress, merge, message).

    t = Tract.open(default_config=tract_default)

    t.configure_operations(
        chat=LLMConfig(temperature=0.8),             # creative chat
        compress=LLMConfig(temperature=0.1, seed=42), # deterministic compression
        # merge and message inherit tract_default
    )

    # Inspect current operation configs:
    ops = t.operation_configs  # OperationConfigs dataclass
    print(f"chat config: {ops.chat}")           # LLMConfig(temperature=0.8)
    print(f"compress config: {ops.compress}")    # LLMConfig(temperature=0.1, seed=42)
    print(f"merge config: {ops.merge}")          # None -> falls through to default
    t.close()

    # =================================================================
    # 3. LEVEL 2 — Per-call LLMConfig object
    # =================================================================
    # Pass llm_config= to chat()/generate() for one-off overrides.

    precise = LLMConfig(temperature=0.2, seed=123)
    # t.chat("Explain X.", llm_config=precise)
    # t.generate(llm_config=precise)

    # =================================================================
    # 4. LEVEL 1 — Sugar params (highest priority)
    # =================================================================
    # Pass temperature=, model=, max_tokens= directly on chat()/generate().

    # t.chat("Hello", temperature=0.9)
    # t.generate(temperature=0.1, max_tokens=100)

    # Sugar overrides llm_config for the same field:
    # t.chat("Hello", llm_config=LLMConfig(temperature=0.3), temperature=0.8)
    # -> temperature=0.8 (sugar wins), other fields from llm_config

    # =================================================================
    # 5. RESOLUTION EXAMPLE — all 4 levels
    # =================================================================
    # Given:
    #   Level 4: model="gpt-4o", temperature=0.5, top_p=0.95
    #   Level 3: chat -> temperature=0.8
    #   Level 2: llm_config=LLMConfig(temperature=0.2, seed=123)
    #   Level 1: temperature=0.9
    #
    # Resolved per field:
    #   model      = "gpt-4o"  (level 4, nothing overrides)
    #   temperature = 0.9      (level 1 sugar wins)
    #   top_p      = 0.95      (level 4, nothing overrides)
    #   seed       = 123       (level 2 llm_config)

    # =================================================================
    # 6. CROSS-FRAMEWORK ALIASES — LLMConfig.from_dict()
    # =================================================================
    # Accepts OpenAI/Anthropic param names and normalizes them.

    config = LLMConfig.from_dict({
        "model": "gpt-4o",
        "temperature": 0.3,
        "max_completion_tokens": 200,  # alias -> max_tokens
        "stop": ["\n\n"],             # alias -> stop_sequences
        "messages": ["..."],          # API plumbing -> auto-ignored
    })
    print(f"max_tokens={config.max_tokens}, stop={config.stop_sequences}")

    # =================================================================
    # 7. TOKEN BUDGET CONFIG
    # =================================================================
    # Set a token limit and track usage against it.

    budget_config = TractConfig(
        token_budget=TokenBudgetConfig(max_tokens=4096),
    )
    t = Tract.open(config=budget_config)
    t.system("You are helpful.")

    status = t.status()
    if status.token_budget_max:
        pct = status.token_count / status.token_budget_max * 100
        if pct > 80:
            print(f"WARNING: {pct:.0f}% budget used — consider compressing")

    t.close()

    # =================================================================
    # 8. PROVENANCE — generation_config on commits
    # =================================================================
    # After chat()/generate(), the response carries the resolved config:
    #   r = t.chat("Hello")
    #   r.generation_config          # LLMConfig used for this call
    #   r.generation_config.to_dict()  # serializable dict
    #
    # Query by config later:
    #   t.log() -> entry.generation_config  # LLMConfig or None

    print("Config resolution reference complete.")


if __name__ == "__main__":
    main()
