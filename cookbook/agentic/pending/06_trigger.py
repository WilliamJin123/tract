"""PendingTrigger -- LLM agent controlling trigger actions.

An LLM agent autonomously intercepts trigger-fired actions via the hook
system and exercises ALL PendingTrigger actions: approve, reject, and
modify_params.  The agent inspects the pending's to_dict() output to
understand what triggered, decides whether to adjust parameters before
approving, or rejects outright.

Flow overview:

    trigger fires -> PendingTrigger created -> t.on("trigger", handler)
    -> handler asks LLM -> LLM decides: modify_params + approve, or reject

Scenario A: Agent modifies target_tokens then approves a compress trigger.
Scenario B: Agent decides the context is too important to compress and rejects.

Tools exercised: modify_params, approve, reject
Demonstrates: PendingTrigger lifecycle, LLM-driven approval, param editing,
              rejection with reasoning, hook interception of triggers
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm

from tract import CompressTrigger, Tract, TractConfig, TokenBudgetConfig
from tract.hooks.trigger import PendingTrigger

MODEL_ID = llm.large


# =====================================================================
# Helper: ask the LLM a question and get a JSON decision
# =====================================================================

def ask_agent(client, system_prompt: str, user_prompt: str) -> dict:
    """Send a prompt to the LLM via tract's built-in client and parse a JSON decision."""
    from tract.llm.client import OpenAIClient

    response = client.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=512,
    )
    text = OpenAIClient.extract_content(response)
    # Extract JSON from the response (handle markdown code fences)
    text = text.strip()
    if text.startswith("```"):
        # Strip ```json ... ``` fences
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)


# =====================================================================
# SCENARIO A: Agent modifies params then approves
# =====================================================================

APPROVE_SYSTEM = """\
You are a context management agent. A compression trigger has fired and you \
must decide how to handle it.

You will receive a JSON description of the pending trigger action. Inspect \
the action_params and reason, then respond with a JSON object:

{
  "action": "modify_and_approve",
  "target_tokens": <integer: the target token count after compression>,
  "reasoning": "<one sentence explaining your choice>"
}

Rules:
- Set target_tokens to roughly 60% of the current threshold to leave headroom.
- Always respond with valid JSON only, no extra text."""


def scenario_a_approve():
    """Agent intercepts a compress trigger, adjusts target_tokens, and approves."""
    print("=" * 60)
    print("SCENARIO A -- Agent Modifies Params and Approves")
    print("=" * 60)

    if not llm.api_key:
        print("\n  SKIPPED (no API key)")
        print("=" * 60)
        return

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=200))

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        config=config,
    ) as t:
        # Track what the agent decided
        decisions: list[dict] = []

        def agent_handler(pending: PendingTrigger) -> None:
            """Hook handler: ask the LLM to decide on the trigger action."""
            info = pending.to_dict()
            print(f"\n  [hook] Trigger fired: {pending.trigger_name}")
            print(f"  [hook] Action type:   {pending.action_type}")
            print(f"  [hook] Reason:        {pending.reason}")
            print(f"  [hook] Params:        {pending.action_params}")

            # Use tract's built-in LLM client (configured via Tract.open())
            prompt = (
                f"A trigger has fired. Here is the pending action:\n\n"
                f"{json.dumps(info, indent=2)}\n\n"
                f"What should we do?"
            )
            decision = ask_agent(pending.tract._llm_client, APPROVE_SYSTEM, prompt)
            decisions.append(decision)
            print(f"  [agent] Decision: {json.dumps(decision)}")

            # Apply the agent's decision: modify params then approve
            target = decision.get("target_tokens", 100)
            pending.modify_params({"target_tokens": target})
            print(f"  [agent] Modified target_tokens -> {target}")
            print(f"  [agent] Params after modify: {pending.action_params}")
            pending.approve()
            print(f"  [agent] Status: {pending.status}")

        # Register the hook and trigger
        t.on("trigger", agent_handler, name="agent-approve")
        trigger = CompressTrigger(threshold=0.3, summary_content="Condensed context.")
        t.configure_triggers([trigger])

        # Seed enough messages to fire the trigger
        t.system("You are a research assistant tracking experiment results.")
        for i in range(8):
            t.user(f"Experiment {i}: measured value = {i * 3.14:.2f} units.")

        print(f"\n  Commits before compile: {len(t.log(limit=100))}")
        status = t.status()
        print(f"  Tokens before compile: {status.token_count}/{status.token_budget_max}")

        # compile() triggers evaluation -> fires hook -> agent decides
        ctx = t.compile()
        print(f"\n  After compile: {ctx.token_count} tokens, {len(ctx.messages)} messages")

        if decisions:
            print(f"\n  Agent reasoning: {decisions[0].get('reasoning', 'N/A')}")
        else:
            print("\n  (trigger did not fire -- threshold not reached)")

    print()


# =====================================================================
# SCENARIO B: Agent rejects the trigger
# =====================================================================

REJECT_SYSTEM = """\
You are a context management agent. A compression trigger has fired but \
you must evaluate whether compression is appropriate right now.

You will receive a JSON description of the pending trigger action. The \
context contains critical experiment data that should NOT be compressed \
yet because the user is actively analyzing it.

Respond with a JSON object:

{
  "action": "reject",
  "reason": "<one sentence explaining why compression should be blocked>"
}

Rules:
- Always reject. The data is too important to compress right now.
- Always respond with valid JSON only, no extra text."""


def scenario_b_reject():
    """Agent intercepts a compress trigger and rejects it to protect data."""
    print("=" * 60)
    print("SCENARIO B -- Agent Rejects the Trigger")
    print("=" * 60)

    if not llm.api_key:
        print("\n  SKIPPED (no API key)")
        print("=" * 60)
        return

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=200))

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
        config=config,
    ) as t:
        rejections: list[dict] = []

        def agent_handler(pending: PendingTrigger) -> None:
            """Hook handler: agent rejects compression to protect data."""
            info = pending.to_dict()
            print(f"\n  [hook] Trigger fired: {pending.trigger_name}")
            print(f"  [hook] Action type:   {pending.action_type}")
            print(f"  [hook] Reason:        {pending.reason}")

            # Use tract's built-in LLM client (configured via Tract.open())
            prompt = (
                f"A compression trigger wants to fire:\n\n"
                f"{json.dumps(info, indent=2)}\n\n"
                f"The user is actively analyzing this experiment data. "
                f"Should we compress?"
            )
            decision = ask_agent(pending.tract._llm_client, REJECT_SYSTEM, prompt)
            rejections.append(decision)
            print(f"  [agent] Decision: {json.dumps(decision)}")

            # Apply: reject with the agent's reasoning
            reason = decision.get("reason", "Agent rejected without reason.")
            pending.reject(reason)
            print(f"  [agent] Status: {pending.status}")
            print(f"  [agent] Rejection reason: {pending.rejection_reason}")

        # Register hook and trigger
        t.on("trigger", agent_handler, name="agent-reject")
        trigger = CompressTrigger(threshold=0.3, summary_content="Should not appear.")
        t.configure_triggers([trigger])

        # Seed critical experiment data
        t.system("You are tracking a live particle physics experiment.")
        for i in range(8):
            t.user(f"Detector reading {i}: {(i + 1) * 42} GeV, significance {i + 1} sigma.")

        status_before = t.status()
        print(f"\n  Tokens before compile: {status_before.token_count}/{status_before.token_budget_max}")

        # compile() triggers evaluation -> agent rejects -> no compression
        ctx = t.compile()
        status_after = t.status()

        print(f"\n  After compile: {ctx.token_count} tokens, {len(ctx.messages)} messages")
        print(f"  Tokens unchanged: {status_before.token_count == status_after.token_count}")

        if rejections:
            print(f"\n  Agent reasoning: {rejections[0].get('reason', 'N/A')}")
            print("  Result: compression was BLOCKED -- all data preserved.")
        else:
            print("\n  (trigger did not fire)")

    print()


# =====================================================================
# SCENARIO C: Manual walkthrough (no LLM required)
# =====================================================================

def scenario_c_manual():
    """Manual demonstration of all three PendingTrigger actions without LLM."""
    print("=" * 60)
    print("SCENARIO C -- Manual: All Three Actions (no LLM)")
    print("=" * 60)

    config = TractConfig(token_budget=TokenBudgetConfig(max_tokens=200))

    # -- 1. modify_params + approve ------------------------------------
    print("\n  1. modify_params + approve:")
    with Tract.open(config=config) as t:
        captured: list[PendingTrigger] = []

        def modify_and_approve(pending: PendingTrigger) -> None:
            captured.append(pending)
            print(f"     trigger_name:  {pending.trigger_name}")
            print(f"     action_type:   {pending.action_type}")
            print(f"     reason:        {pending.reason}")
            print(f"     action_params: {pending.action_params}")

            # Modify target_tokens before approving
            pending.modify_params({"target_tokens": 80})
            print(f"     after modify:  {pending.action_params}")
            pending.approve()
            print(f"     status:        {pending.status}")

        t.on("trigger", modify_and_approve, name="modify-approve")
        t.configure_triggers([CompressTrigger(threshold=0.3, summary_content="Summary.")])
        t.system("Instructions for the assistant.")
        for i in range(8):
            t.user(f"Message {i} with enough text to grow token count.")
        t.compile()

        if captured:
            print(f"     result:        approved with target_tokens={captured[0].action_params.get('target_tokens')}")

    # -- 2. reject -----------------------------------------------------
    print("\n  2. reject:")
    with Tract.open(config=config) as t:
        rejected_pending: list[PendingTrigger] = []

        def reject_handler(pending: PendingTrigger) -> None:
            rejected_pending.append(pending)
            pending.reject("Data is critical, do not compress.")
            print(f"     status:           {pending.status}")
            print(f"     rejection_reason: {pending.rejection_reason}")

        t.on("trigger", reject_handler, name="reject")
        t.configure_triggers([CompressTrigger(threshold=0.3, summary_content="Nope.")])
        t.system("Critical data tracking.")
        for i in range(8):
            t.user(f"Critical reading {i}.")
        ctx = t.compile()
        print(f"     messages after:   {len(ctx.messages)} (all preserved)")

    # -- 3. to_dict() inspection ---------------------------------------
    print("\n  3. to_dict() inspection:")
    with Tract.open(config=config) as t:
        def inspect_handler(pending: PendingTrigger) -> None:
            info = pending.to_dict()
            print(f"     keys: {sorted(info.keys())}")
            print(f"     operation:     {info['operation']}")
            print(f"     status:        {info['status']}")

            # Fields contain the trigger-specific data
            fields = info.get("fields", {})
            print(f"     trigger_name:  {fields.get('trigger_name')}")
            print(f"     action_type:   {fields.get('action_type')}")
            print(f"     action_params: {fields.get('action_params')}")
            print(f"     reason:        {fields.get('reason')}")

            # Available actions show what the agent can do
            print(f"     actions:       {info.get('available_actions')}")
            pending.approve()

        t.on("trigger", inspect_handler, name="inspect")
        t.configure_triggers([CompressTrigger(threshold=0.3, summary_content="Inspected.")])
        t.system("System setup.")
        for i in range(8):
            t.user(f"Filler message {i}.")
        t.compile()

    print()


def main():
    scenario_c_manual()
    scenario_a_approve()
    scenario_b_reject()


if __name__ == "__main__":
    main()


# --- See also ---
# cookbook/agentic/sidecar/01_triggers.py         -- All trigger types and autonomy spectrum
# cookbook/hooks/01_routing/01_three_tier.py       -- Three-tier hook routing
# cookbook/hooks/02_pending/01_compress_lifecycle.py -- PendingCompress lifecycle
