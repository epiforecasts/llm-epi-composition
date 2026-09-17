# Detector validation review protocol

This protocol implements the human review in [the analysis plan](../analysis_plan.md)
(Implementation assessment). Total human review time is at most 10 hours.

## Reviewer

One infectious-disease modeller who did not implement `evaluation/detectors.py`.
Record their experience with each language and framework before review. The
reviewer uses no LLM assistance. Condition may be apparent from language and
package structure; conceal model, paraphrase and run identity, truth, scores and
detector outputs.

## Sample

One randomly selected canonical run with reviewable code in each
scenario–condition–commercial-model cell for scenarios 2 and 3: 18 submissions.
Draw with a recorded seed after all runs complete. If a cell has no run with
reviewable code, record this and leave the cell empty. Present submissions in
random order.

## Materials and timing

Each sample folder contains the source code, the scientific prompt, a fixed data
summary including the supplied GI, delay and ascertainment information, and a
blank sheet. Stop at 30 minutes; record the review as incomplete and keep the
ratings made so far.

## Review sheet

```text
Sample ID:
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

Compare each detector flag with the reviewer's rating, treating **incorrect** and
**absent** as a defect and **correct** as no defect. Exclude **unclear** and
**not applicable** from sensitivity and specificity and report their counts.
Report per-component sensitivity, specificity and raw agreement with counts and
exact binomial intervals. Ratings from incomplete reviews are used where given.
