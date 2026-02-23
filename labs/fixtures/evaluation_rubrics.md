# Evaluation Rubrics

Scoring criteria for agent decision quality assessment.

## Compression Decision Rubric

| Score | Criteria |
|-------|----------|
| 5 | Optimal: compressed exactly the right messages, preserved all critical info |
| 4 | Good: reasonable compression target, minor info loss on non-critical content |
| 3 | Acceptable: budget met but over-aggressive or too conservative |
| 2 | Poor: lost critical information or barely reduced tokens |
| 1 | Failure: budget violated, critical data lost, or invalid operation |

## Pinning Decision Rubric

| Score | Criteria |
|-------|----------|
| 5 | All critical messages pinned, no false positives |
| 4 | Most critical messages pinned, 1 false positive |
| 3 | Some critical messages missed OR 2+ false positives |
| 2 | Majority of critical messages missed |
| 1 | Pinning decisions appear random |

## Edit vs Append Decision Rubric

| Score | Criteria |
|-------|----------|
| 5 | Used EDIT for factual corrections, compiled context is clean |
| 4 | Used EDIT but minor issues (wrong target, partial fix) |
| 3 | Used append when EDIT was better, but result is correct |
| 2 | Correction is incomplete or introduces new errors |
| 1 | Failed to identify or correct the error |

## Research Quality Rubric

| Score | Criteria |
|-------|----------|
| 5 | Comprehensive, well-sourced, no hallucinations, addresses all aspects |
| 4 | Thorough, minor gaps, well-structured |
| 3 | Adequate coverage, some unsupported claims |
| 2 | Superficial, significant gaps or unsupported claims |
| 1 | Off-topic, mostly hallucinated, or incoherent |

## Context Management Activity Rubric

| Score | Criteria |
|-------|----------|
| 5 | Proactive management: compressed before hitting limit, pinned key info early |
| 4 | Reactive but effective: managed context when pressure arose |
| 3 | Minimal management: only acted when forced by budget |
| 2 | Late management: context overflow before acting |
| 1 | No management: ignored context pressure entirely |
