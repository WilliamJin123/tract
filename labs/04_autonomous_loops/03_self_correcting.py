"""Self-Correcting Agent

Setup: Agent has a structured output task (generate valid JSON API response).
       First attempt will likely have errors. Agent must detect, edit-fix, and
       re-validate in a loop.
Decision: Detect validation failures, choose edit vs append, correct, re-validate.
Evaluates: Error detection rate, correction strategy, final correctness, retry count.

Demonstrates: chat(validator=), commit(operation=EDIT), compile(), validation loop
"""


def main():
    # --- Setup: define JSON schema, validation rules ---
    # --- Attempt 1: agent generates output ---
    # --- Validate: check against schema ---
    # --- On failure: agent inspects error, decides correction ---
    # --- Edit: agent uses EDIT to fix the bad commit ---
    # --- Re-validate: loop until correct or retries exhausted ---
    # --- Audit: log() shows full correction chain ---
    # --- Evaluate: detection rate, strategy, correctness ---
    # --- Report: print metrics ---
    pass


if __name__ == "__main__":
    main()
