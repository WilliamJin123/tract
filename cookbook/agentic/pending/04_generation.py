"""Agentic generation control: LLM agent autonomously manages PendingGeneration.

Demonstrates:
    - approve: commit the generated response as-is
    - reject: refuse the generation (raises RetryExhaustedError)
    - retry: re-generate with steering guidance from the agent
    - validate: check response quality against a validator function

PendingGeneration is created internally by generate() when a validator is
provided -- NOT via review=True. You intercept it by registering a handler
on the "generate" hook. Inside the handler, the agent uses to_dict() +
to_tools() + a SEPARATE httpx LLM call to decide what to do.

Scenario A: Quality gate agent
    Validator rejects short responses. The hook handler validates, retries
    with guidance if needed, then approves.

Scenario B: Agent-driven rejection
    Handler inspects response_text via to_dict() and rejects based on content.
    Rejection propagates as RetryExhaustedError from generate().
"""

import json
import sys
from pathlib import Path

import httpx

from tract import Tract
from tract.exceptions import RetryExhaustedError
from tract.hooks.generation import PendingGeneration
from tract.hooks.validation import ValidationResult

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _providers import groq as llm  # noqa: E402

MODEL_ID = llm.large


# ---------------------------------------------------------------------------
# Helper: ask the agent LLM to decide on a pending generation
# ---------------------------------------------------------------------------

def _ask_agent(pending: PendingGeneration, instruction: str) -> dict:
    """Send pending state + tools to a SEPARATE LLM call for a decision.

    This is the agent's "brain" -- completely independent from the tract
    generate() LLM call that produced the response being evaluated.

    Returns:
        A decision dict with "action" and "args" keys.
    """
    tools = pending.to_tools()
    ctx_info = pending.to_dict()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a quality-gate agent. You receive a pending LLM "
                "generation and must decide what to do with it. Use the "
                "provided tools to take exactly one action."
            ),
        },
        {
            "role": "user",
            "content": f"{instruction}\n\nPending state:\n{json.dumps(ctx_info, indent=2)}",
        },
    ]

    resp = httpx.post(
        f"{llm.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {llm.api_key}"},
        json={"model": MODEL_ID, "messages": messages, "tools": tools},
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()

    tc_list = raw["choices"][0]["message"].get("tool_calls", [])
    if tc_list:
        tc = tc_list[0]
        return {
            "action": tc["function"]["name"],
            "args": json.loads(tc["function"].get("arguments", "{}")),
        }
    # Fallback: if LLM responds without a tool call, default to approve
    return {"action": "approve", "args": {}}


# ---------------------------------------------------------------------------
# Scenario A: Quality gate agent
# ---------------------------------------------------------------------------

def quality_gate_agent() -> None:
    """Hook handler validates, retries with guidance if needed, then approves."""
    print("=" * 60)
    print("SCENARIO A -- Quality Gate Agent")
    print("=" * 60)
    print()
    print("  A validator checks response length. The hook handler:")
    print("    1. validate() -- check against the quality gate")
    print("    2. retry(guidance=...) if validation fails")
    print("    3. re-validate, then approve or reject")

    # Validator: response must be more than 20 characters
    def length_validator(text: str) -> tuple[bool, str | None]:
        if len(text) > 20:
            return (True, None)
        return (False, "Response too short")

    decisions_log: list[dict] = []

    def quality_handler(pending: PendingGeneration) -> None:
        """Agent-driven quality gate: validate -> retry -> approve."""
        print(f"\n  [hook] Received PendingGeneration")
        print(f"    response_text preview: {pending.response_text[:80]!r}")
        print(f"    retry_count: {pending.retry_count}")
        pending.pprint()

        # Step 1: validate
        result: ValidationResult = pending.validate()
        print(f"\n  [hook] validate(): passed={result.passed}, diagnosis={result.diagnosis}")

        if not result.passed:
            # Step 2: retry with the diagnosis as guidance
            print(f"  [hook] Validation failed -- retrying with guidance")
            pending.retry(guidance=result.diagnosis or "Improve response quality")
            print(f"    New response_text preview: {pending.response_text[:80]!r}")
            print(f"    retry_count: {pending.retry_count}")

            # Step 3: re-validate after retry
            result = pending.validate()
            print(f"\n  [hook] Re-validate: passed={result.passed}, diagnosis={result.diagnosis}")

        # Use the agent LLM to make the final decision
        decision = _ask_agent(
            pending,
            f"The response {'passed' if result.passed else 'FAILED'} validation. "
            f"Response text: {pending.response_text[:200]!r}. "
            f"Retries so far: {pending.retry_count}. "
            f"{'Approve the response.' if result.passed else 'Reject with reason: quality gate failed.'}",
        )
        decisions_log.append(decision)
        print(f"\n  [hook] Agent decision: {json.dumps(decision)}")

        # Dispatch the agent's decision
        pending.apply_decision(decision)
        print(f"  [hook] Final status: {pending.status}")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant. Always give detailed answers.")
        t.user("Explain the difference between TCP and UDP in networking.")

        # Register the hook handler BEFORE calling generate()
        t.on("generate", quality_handler, name="quality_gate")

        # generate() with a validator creates PendingGeneration internally
        # and routes it through our hook handler
        response = t.generate(
            validator=length_validator,
            max_tokens=500,
        )

        print(f"\n  generate() returned ChatResponse:")
        print(f"    text preview: {response.text[:80]!r}")
        print(f"    commit_hash:  {response.commit_info.commit_hash[:12]}")

        # Show the full conversation after generation
        ctx = t.compile()
        print(f"\n  Compiled context: {len(ctx.messages)} messages, {ctx.token_count} tokens")
        ctx.pprint(style="compact")

    print(f"\n  Agent decisions: {len(decisions_log)}")
    for i, d in enumerate(decisions_log):
        print(f"    [{i}] {json.dumps(d)}")


# ---------------------------------------------------------------------------
# Scenario B: Agent-driven rejection
# ---------------------------------------------------------------------------

def agent_rejection() -> None:
    """Handler inspects response_text and rejects based on content."""
    print("\n" + "=" * 60)
    print("SCENARIO B -- Agent-Driven Rejection")
    print("=" * 60)
    print()
    print("  The handler inspects response_text via to_dict() and asks the")
    print("  agent LLM to reject. Rejection raises RetryExhaustedError.")

    # Trivial validator (always passes) -- we still need one to trigger
    # the PendingGeneration hook path
    def always_pass(text: str) -> tuple[bool, str | None]:
        return (True, None)

    def rejection_handler(pending: PendingGeneration) -> None:
        """Agent inspects response and decides to reject."""
        ctx_info = pending.to_dict()
        print(f"\n  [hook] Received PendingGeneration")
        print(f"    response_text from to_dict(): {ctx_info['fields'].get('response_text', '')[:80]!r}")

        # Ask the agent LLM to reject
        decision = _ask_agent(
            pending,
            "This generation must be REJECTED. The response does not meet "
            "our safety policy. Use the reject tool with a clear reason.",
        )
        print(f"  [hook] Agent decision: {json.dumps(decision)}")

        # If the agent didn't pick reject, force it
        if decision.get("action") != "reject":
            print(f"  [hook] Agent did not reject -- forcing rejection")
            pending.reject(reason="Policy violation: response flagged by safety agent")
        else:
            pending.apply_decision(decision)

        print(f"  [hook] Final status: {pending.status}")
        if pending.rejection_reason:
            print(f"  [hook] Rejection reason: {pending.rejection_reason}")

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("Tell me a joke.")

        t.on("generate", rejection_handler, name="safety_gate")

        # generate() will raise RetryExhaustedError because the hook rejects
        try:
            response = t.generate(
                validator=always_pass,
                max_tokens=200,
            )
            # If we reach here, the LLM agent approved instead of rejecting
            print(f"\n  Unexpected: generate() succeeded with: {response.text[:60]!r}")
        except RetryExhaustedError as e:
            print(f"\n  RetryExhaustedError caught (expected):")
            print(f"    attempts:       {e.attempts}")
            print(f"    last_diagnosis: {e.last_diagnosis}")
            print(f"    last_result:    {str(e.last_result)[:60]!r}")

        # Conversation still has the committed messages up to the rejection
        ctx = t.compile()
        print(f"\n  Context after rejection: {len(ctx.messages)} messages, {ctx.token_count} tokens")
        ctx.pprint(style="compact")


# ---------------------------------------------------------------------------
# PendingGeneration field reference
# ---------------------------------------------------------------------------

def field_reference() -> None:
    """Show PendingGeneration fields, to_dict(), and to_tools() output."""
    print("\n" + "=" * 60)
    print("REFERENCE -- PendingGeneration Fields & Agent Interface")
    print("=" * 60)

    def dummy_validator(text: str) -> tuple[bool, str | None]:
        return (True, None)

    inspected: list[PendingGeneration] = []

    def inspect_handler(pending: PendingGeneration) -> None:
        inspected.append(pending)

        # Show all public fields
        print(f"\n  PendingGeneration public fields:")
        print(f"    response_text:  {pending.response_text[:60]!r}...")
        print(f"    validator:      {pending.validator}")
        print(f"    retry_prompt:   {pending.retry_prompt}")
        print(f"    hide_retries:   {pending.hide_retries}")
        print(f"    retry_count:    {pending.retry_count}")
        print(f"    retry_history:  {pending.retry_history}")

        # to_dict() for LLM consumption
        d = pending.to_dict()
        print(f"\n  to_dict() keys: {list(d.keys())}")
        print(f"    operation:         {d['operation']}")
        print(f"    status:            {d['status']}")
        print(f"    available_actions: {d['available_actions']}")
        print(f"    fields.response_text: {str(d['fields'].get('response_text', ''))[:60]!r}")

        # to_tools() for function calling
        tools = pending.to_tools()
        print(f"\n  to_tools() -- {len(tools)} tool definitions:")
        for tool in tools:
            fn = tool["function"]
            params = fn["parameters"].get("properties", {})
            print(f"    {fn['name']}({', '.join(params.keys())})")

        # Approve so generate() can return
        pending.approve()

    with Tract.open(
        api_key=llm.api_key,
        base_url=llm.base_url,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("What is 2 + 2?")

        t.on("generate", inspect_handler, name="inspector")
        t.generate(validator=dummy_validator, max_tokens=100)

    print(f"\n  Hook fired: {len(inspected)} time(s)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    quality_gate_agent()
    agent_rejection()
    field_reference()

    print("\n" + "=" * 60)
    print("KEY TAKEAWAYS")
    print("=" * 60)
    print("""
  1. PendingGeneration is created internally by generate() when a validator
     is provided. It is NOT created via review=True.

  2. Register t.on("generate", handler) to intercept. The handler receives
     the PendingGeneration directly.

  3. Inside the handler, use to_dict() + to_tools() + httpx to ask a
     SEPARATE LLM to decide (approve, reject, retry, validate).

  4. retry() commits a steering message, re-generates via LLM, and updates
     response_text in place. Failed attempts get SKIP-annotated on approve().

  5. validate() returns ValidationResult(passed, diagnosis, index).

  6. Rejection from the hook raises RetryExhaustedError in the caller.

  7. The agent LLM call (via httpx) is completely independent from the
     tract generate() LLM call that produced the response.
""")


if __name__ == "__main__":
    main()
