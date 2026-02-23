"""Selective Pinning

Setup: A 25-message conversation where 5 messages contain critical reference
       information and 20 are routine. No messages are pinned.
Decision: Agent reviews the conversation and decides which messages deserve
          PINNED status to protect from future compression.
Evaluates: Precision/recall against ground truth, reasoning quality.

Demonstrates: compile(), annotate(hash, PINNED), log(), content analysis
"""


def main():
    # --- Setup: build conversation with 5 known-critical messages ---
    # --- Agent: review all messages, decide what to pin ---
    # --- Execute: agent calls annotate() for chosen messages ---
    # --- Verify: run compression, check pinned messages survive ---
    # --- Evaluate: precision/recall vs ground truth labels ---
    # --- Report: print metrics ---
    pass


if __name__ == "__main__":
    main()
