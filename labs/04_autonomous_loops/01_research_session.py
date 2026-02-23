"""Research Session

Setup: Agent is given a research question and access to web search tool
       (simulated or real). Must research, accumulate, manage context, and
       produce a final summary.
Decision: Multiple — when to search vs synthesize, when to compress, what to
          pin, when research is "done."
Evaluates: Research quality, context management activity, token efficiency.

Demonstrates: chat(), status(), annotate(), compress(), tool use, multi-turn
"""


def main():
    # --- Setup: research question, search tool, budget ---
    # --- Loop: agent researches (search, read, synthesize) ---
    #   --- Monitor: track context growth per turn ---
    #   --- Curation: log any compress/pin/skip decisions ---
    # --- Final: agent produces research summary ---
    # --- Evaluate: quality, management activity, efficiency ---
    # --- Report: print metrics ---
    pass


if __name__ == "__main__":
    main()
