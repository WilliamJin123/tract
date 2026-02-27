"""PendingMerge retry and validate: use validate() / retry() / auto_retry()
on PendingMerge to check resolutions, re-resolve via LLM when validation
fails, and run the automated retry loop. Shows HookRejection on exhaustion.
"""

import os

from dotenv import load_dotenv

from tract import Tract
from tract.hooks.merge import PendingMerge
from tract.hooks.retry import auto_retry
from tract.hooks.validation import HookRejection, ValidationResult

load_dotenv()

TRACT_OPENAI_API_KEY = os.environ["TRACT_OPENAI_API_KEY"]
TRACT_OPENAI_BASE_URL = os.environ["TRACT_OPENAI_BASE_URL"]
MODEL_ID = "gpt-oss-120b"


def merge_retry_and_validate():
    print("=" * 60)
    print("PendingMerge: Retry and Validate")
    print("=" * 60)
    print()
    print("  validate() checks all resolutions are present and non-empty.")
    print("  retry() re-resolves all conflicts via LLM with optional guidance.")
    print("  auto_retry() is the automated validate->retry loop.")

    # -- Manual validate / retry cycle --
    print(f"\n  --- Manual validate/retry cycle ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        # Build a conflict: both branches EDIT the same message
        sys_ci = t.system("You are a helpful assistant.")
        user_ci = t.user("Explain machine learning.")
        t.assistant("Machine learning is a subset of AI.")

        # Feature branch edits the assistant message
        t.branch("feature-ml")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Machine learning uses statistical models to learn patterns from data.",
        )

        # Main also edits the same message -> conflict
        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Machine learning enables computers to improve through experience.",
        )

        # --- review=True: get PendingMerge ---
        pending: PendingMerge = t.merge("feature-ml", review=True)

        # pprint shows conflicts and resolution status
        pending.pprint()

        # --- Clear resolutions to demonstrate validate() failing ---
        # (In real use, resolutions might be missing if review=True
        #  was called without a resolver, or if the resolver failed.)
        saved_resolutions = dict(pending.resolutions)
        pending.resolutions.clear()

        print(f"\n  Cleared resolutions to simulate missing data:")
        print(f"    resolutions: {len(pending.resolutions)} keys")

        # --- validate() should fail: no resolutions ---
        result: ValidationResult = pending.validate()
        print(f"\n  validate() #1:")
        print(f"    passed:    {result.passed}")
        print(f"    diagnosis: {result.diagnosis}")
        print(f"    index:     {result.index}")

        # --- retry() re-resolves all conflicts via LLM ---
        print(f"\n  Calling retry(guidance='Be concise')...")
        pending.retry(guidance="Be concise")
        print(f"    resolutions after retry: {len(pending.resolutions)} keys")
        for key, resolution in pending.resolutions.items():
            print(f"    {key[:12]}: {resolution[:80]}")

        # --- validate() again: should pass now ---
        result2: ValidationResult = pending.validate()
        print(f"\n  validate() #2:")
        print(f"    passed:    {result2.passed}")
        print(f"    diagnosis: {result2.diagnosis}")

        # --- Approve ---
        merge_result = pending.approve()
        print(f"\n  Approved! Merge complete")
        pending.pprint()

    # -- auto_retry(): automated validate->retry loop --
    print(f"\n  --- auto_retry(): automated loop ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("What is deep learning?")
        t.assistant("Deep learning uses neural networks.")

        t.branch("feature-dl")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Deep learning uses multi-layered neural networks for representation learning.",
        )

        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Deep learning is a branch of ML with layered architectures.",
        )

        pending2: PendingMerge = t.merge("feature-dl", review=True)

        pending2.pprint()
        print(f"\n  Calling auto_retry(pending, max_retries=3)...")

        result = auto_retry(pending2, max_retries=3)
        print(f"\n  auto_retry returned: {type(result).__name__}")

        if isinstance(result, HookRejection):
            print(f"    REJECTED: {result.reason}")
        else:
            print(f"    Merge approved and complete")
            pending2.pprint()

    # -- HookRejection on exhausted retries --
    print(f"\n  --- HookRejection when retries exhaust ---")

    with Tract.open(
        api_key=TRACT_OPENAI_API_KEY,
        base_url=TRACT_OPENAI_BASE_URL,
        model=MODEL_ID,
    ) as t:
        t.system("You are a helpful assistant.")
        t.user("Explain transformers.")
        t.assistant("Transformers are a neural network architecture.")

        t.branch("feature-tx")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Transformers use self-attention to process sequences in parallel.",
        )

        t.switch("main")
        t.assistant(
            edit=t.log()[-1].commit_hash,
            text="Transformers replaced RNNs for most NLP tasks.",
        )

        pending3: PendingMerge = t.merge("feature-tx", review=True)

        # Force resolutions to be empty so every retry's validate() fails
        # (simulates a resolver that keeps producing bad output)
        original_retry = pending3.retry

        def broken_retry(**kwargs):
            """Simulate a resolver that always produces empty resolutions."""
            original_retry(**kwargs)
            # Wipe resolutions after each retry to force failure
            pending3.resolutions.clear()

        pending3.retry = broken_retry

        print(f"\n  Simulating a resolver that always fails...")
        print(f"  Calling auto_retry(pending, max_retries=2)...")

        result = auto_retry(pending3, max_retries=2)
        print(f"\n  auto_retry returned: {type(result).__name__}")

        if isinstance(result, HookRejection):
            print(f"    reason:           {result.reason}")
            print(f"    rejection_source: {result.rejection_source}")
            print(f"    pending status:   {pending3.status}")
            if result.metadata:
                print(f"    metadata:         {result.metadata}")


if __name__ == "__main__":
    merge_retry_and_validate()
