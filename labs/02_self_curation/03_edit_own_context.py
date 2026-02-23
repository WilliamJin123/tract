"""Edit Own Context

Setup: A conversation where the agent made a factual error 8 messages ago
       (stated wrong API rate limit). Agent later receives a correction.
Decision: Agent must identify the erroneous message, choose edit-in-place vs
          skip+re-state, execute the correction, and verify.
Evaluates: Error identification accuracy, correction strategy, compiled correctness.

Demonstrates: log(), commit(operation=EDIT, edit_target=hash), compile(), diff()
"""


def main():
    # --- Setup: build conversation with planted factual error ---
    # --- Trigger: user points out the error ---
    # --- Agent: search history for the wrong message ---
    # --- Agent: decide correction strategy (EDIT vs append) ---
    # --- Execute: agent performs the correction ---
    # --- Verify: compile() shows corrected content ---
    # --- Evaluate: identification accuracy, strategy quality ---
    # --- Report: print metrics ---
    pass


if __name__ == "__main__":
    main()
