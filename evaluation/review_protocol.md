# Detector validation review protocol

This protocol implements the human review in [the analysis plan](../analysis_plan.md)
(Implementation assessment). Review time is at most 10 hours per reviewer.

## Reviewers

Two infectious-disease modellers who did not implement `evaluation/detectors.py`,
reviewing independently. Record their experience with each language and
framework before review. Reviewers use no LLM assistance. Condition may be
apparent from language and package structure; conceal model, paraphrase and run
identity, truth, scores, detector outputs and the other reviewer's ratings.

The project lead adjudicates disagreements between the two reviewers after both
sheets are submitted and preserved, to form a consensus rating per component and
submission. Adjudication does not change either reviewer's recorded initial
rating.

## Sample

One randomly selected canonical run with reviewable code in each
scenario–condition–commercial-model cell for scenarios 2 and 3: 18 submissions.
Draw with a recorded seed after all runs complete. If a cell has no run with
reviewable code, record this and leave the cell empty. Both reviewers assess the
same 18 submissions, each in an independently randomised order.

## Materials and timing

Each sample folder contains the source code, the scientific prompt, a fixed data
summary including the supplied GI, delay and ascertainment information, and a
blank sheet. Stop at 30 minutes per submission; record the review as incomplete
and keep the ratings made so far. Eighteen submissions at the cap take at most
9 hours per reviewer.

## Review sheet

```text
Sample ID:
Reviewer ID:
Elapsed minutes:
Review complete: yes / no

Component | correct / incorrect / absent / unclear / not applicable | one-line reason
Reporting-delay convolution                   |  |
Delay discretisation and finite support       |  |
Generation-interval discretisation            |  |
Weekly reporting variation                    |  |
Supplied ascertainment used on report date    |  |
Smoothing or prior on the reproduction number |  |
Shared infections across streams (scenario 3) |  |

Prompt compliance concern (optional, one line):
```

The component list is frozen with the detector version before registration and
matches the detector flags one to one.

## Scoring

Report raw agreement and Cohen's kappa between the two reviewers, per component,
from their initial independent ratings. Record the adjudicated consensus rating
separately from each reviewer's initial rating.

Compare each detector flag with the adjudicated consensus, treating **incorrect**
and **absent** as a defect and **correct** as no defect. Exclude **unclear** and
**not applicable** from sensitivity and specificity and report their counts.
Report per-component sensitivity, specificity and raw agreement with counts and
exact binomial intervals. With 18 submissions these intervals are wide and
indicate gross detector failures rather than precise accuracy. Ratings from
incomplete reviews are used where given.
