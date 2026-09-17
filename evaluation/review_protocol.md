# Expert review protocol

This protocol implements the review outcomes in [the analysis plan](../analysis_plan.md).
It covers implementation review, method identification, detector validation and
H3 defect detection. The review allocation and scoring are specified here;
package-specific mutation validation remains outstanding until the
ComposableTuringIDModels migration is complete.

## Roles and independence

Two infectious-disease modellers work independently. Record their experience
with each language and framework before review. The coordinator prepares samples
and assigns random IDs but does not assess samples they prepared. A validator
who did not implement the automated detectors performs detector validation.

Condition may be apparent from language and package structure. Conceal model,
paraphrase and run identity, mutation status, pair identities, truth and detector
outputs. Do not claim condition blinding or absence of familiarity bias.
Reviewers use no LLM assistance. Discussion and adjudication occur only after
initial sheets have been submitted and preserved.

The full protocol includes coordinator-only allocation and scoring instructions.
Give reviewers only their task instructions, scenario information and sheets;
do not distribute the mutation catalogue, sealed allocation or hypotheses.

## General implementation review

Randomly select four of 20 canonical runs from every scenario–condition–commercial-
model cell (144 submissions), and one of three runs from every sensitivity-panel
cell (180 submissions), using recorded seeds. Both reviewers independently assess
the same original code in this pool. Include incomplete or failed code where
review is possible; retain missing-code counts without replacing them with
successful submissions. Keep panels separate in prevalence estimates.

Assess compliance with the prompt, correctness of the stated implementation and
scientific adequacy separately. Use these component categories:

- Reproduction-number definition and infection process.
- Generation-interval discretisation.
- Reporting-delay convolution and daily interval integration.
- Finite delay support and boundary treatment, assessed separately.
- Weekly reporting variation.
- Ascertainment inputs, timing convention and identifiability assumptions.
- Infection-process variation and conditional observation likelihood.
- Regularisation or smoothing assumptions.
- Shared infections and stream-specific observation models, where required.

For each applicable component, record **correct**, **incorrect**, **unclear** or
**absent**, with a short explanation; use **not applicable** otherwise. Presence
alone does not establish correctness. Accept scientifically equivalent methods.
Scenario 1a allows open method choice; record its estimand rather than requiring
a renewal model. A Poisson observation model is not itself an error. Assess
where variation occurs and whether this matches the task and stated model.
Known ascertainment need not be estimated, and an unknown smooth ascertainment
curve does not establish identifiability.

Classify implementation departures as equivalent alternative, minor error,
major error or fundamental model/estimand error, with a scientific justification.
Record semantic concerns about reproduction-number definitions, likelihood and
serial versus generation intervals. Identify the method for scenario 1a.

The independent detector-validation sample contains 60 submissions, five per
scenario–condition cell, distributed across models. Its selection is independent
of this review pool; overlap is allowed and recorded. Compare detector flags with
human classifications using confusion matrices, agreement and kappa. Screening
flags do not certify defect-free code. Additional purposively selected cases do
not enter error-prevalence estimates.

## H3 sample allocation — coordinator only

H3 concerns eligible clean generated code in canonical scenarios 2 and 3, for
`julia` and `composable`, across the three commercial models. It excludes legacy
EpiAware runs and reference solutions.

Within each of six scenario–model cells:

1. Select four independently checked clean submissions per condition by random
   candidate order among valid runs. Keep all candidate exclusions and reasons.
   Static-detector success is insufficient evidence of clean code.
2. Match the four submissions across conditions to form four pairs. Assign the
   same defect type within each pair, using a balanced schedule frozen before
   candidate-code inspection. Independently check that the mutation creates the
   intended error. Record inapplicable candidates and take the next eligible one.
   Verify both scenario and condition applicability: each pair must express the
   same scientific defect in `julia` and `composable`, even if the code edits differ.
   One-condition-only defects are excluded from H3. Broad catalogue tags do not
   extend the pool beyond scenarios 2 and 3. Record these checks in the manifest.
3. Produce one clean and one mutated version per base. Reviewer A sees one and
   reviewer B the other, never both. In each cell, one reviewer receives both
   mutated members of two pairs and the other reviewer both mutated members of
   the remaining two pairs. Each reviewer then sees two clean and two mutated
   versions per condition within each cell, and each pair is compared within one
   reviewer and one defect type. Balance defect types across reviewers in the
   frozen schedule.
4. Assign unrelated random IDs and randomise presentation order. Keep the map of
   base, pair, condition, source run, mutation and reviewer assignment sealed.

Totals are **24 matched pairs, 48 base submissions and 96 reviews**: 48 mutated
versions and 48 clean versions, with 24 of each per condition. Each exact version
has one reviewer; general implementation review supplies the dual assessments
used for inter-rater agreement. Do not compute agreement between clean and
mutated versions as though reviewers assessed identical code.

H3 must precede any general review that could expose a reviewer to its base code.
Build both allocations before review and record overlap. The independent clean-
code check must be performed by someone other than the two H3 reviewers.

If the planned pool cannot be assembled, record H3 as not estimable under this
protocol. Do not substitute reference mutations or select a new pool after seeing
detection results. If two reviewers cannot be recruited, H3 is not conducted.
General implementation review may continue with one reviewer, with no inter-rater
agreement claim.

## Workload

The analysis plan's Review workload section gives assessment counts and hours at
the time cap. Pilot both review types on development submissions before the
allocation is fixed, record committed reviewer hours, and apply the plan's
reduction order if the hours are insufficient.

## Reviewer materials and timing

Each sample folder contains source code, a fixed scenario data summary, the
scientific prompt and a blank sheet. Supply the frozen GI/delay and ascertainment
information required to judge the task. Do not include truth or mutation records.

Start timing when opening the sample and stop when submitting the defect list.
Stop at 45 minutes, retain findings made so far and record the review as incomplete.
Time to complete an unfinished review is right-censored; it is not a completed
review at 45 minutes. Record completeness separately from elapsed time.

## Review sheet

Use this sheet for both pools. The coordinator supplies the applicable component
list for general implementation review. H3 reviewers list defects without access
to mutation status or catalogue.

```text
Sample ID:
Reviewer ID:
Elapsed minutes:
Review complete: yes / no
If incomplete: time cap / other reason (specify)

Defects identified:
For each, record component, code location and a short scientific explanation.
Leave blank if none identified; an empty list does not imply review completion.

Component assessment (general implementation pool):
Component | correct / incorrect / unclear / absent / not applicable | explanation

Prompt compliance concerns:
Scientific adequacy or identifiability concerns:
Departure classification and justification:
Semantic concerns (estimand / likelihood / serial versus generation interval):
Method and target (scenario 1a):
Optional notes:
```

## Scoring — coordinator and analyst only

Preserve initial defect lists. Adjudicate disputed matches and genuine uninjected
errors independently; reviewers do not revise initial detections after learning
the mutation record. Apply the same rules in both conditions.

- **Sensitivity:** injected defects identified / all validated injected defects.
  A timeout without detection is a missed defect. Detection before timeout counts.
- **Specificity among completed clean reviews:** completed clean reviews without
  a false defect report / all completed clean reviews.
- **Successful clean-review proportion:** completed clean reviews without a false
  report / all assigned clean reviews. Incomplete reviews are unsuccessful here.
- **Other false reports:** report spurious defect counts on mutated samples
  separately. Do not call genuine uninjected defects false positives.
- **Detection precision:** matched injected-defect reports / (matched reports plus
  adjudicated spurious reports), with duplicate descriptions collapsed. Report
  genuine uninjected findings separately and disclose the pool's mutation prevalence.

Report all numerators, denominators, timeouts and exclusions by condition, cell
and reviewer. If an unexpected genuine defect invalidates a nominally clean
control, mark it invalid, retain its record and disclose the denominator change;
do not replace it after observing reviews. If a mutation/base validation failure
breaks the required complete H3 allocation, the planned contrast is not estimable.

H3 is the equally weighted mean of the six cell sensitivity differences,
`composable` minus `julia`, which equals the mean within-pair detection difference
over the 24 pairs. Inference concerns the fixed defect schedule, reviewer
assignment and the two reviewers. Use the paired-proportion interval specified in
the plan; do not resample pairs, which would alter the fixed margins.

A valid, wide interval that includes zero leaves the pre-specified H3 unresolved.
Show its counts and interval; do not change its status because the result is
uninformative. If the required pool is unavailable, report descriptive counts and the
reason no confirmatory conclusion is available. A degenerate interval does not
establish certainty. Keep H3's share of the three-hypothesis Bonferroni allocation
reserved even if H3 is not conducted or cannot support inference. Do not reuse
that allocation for H1/H2 or introduce a more favourable analysis after inspection.

General implementation agreement is computed from initial independent ratings
on identical samples; adjudicated results are reported separately. Consult a
third reviewer where disagreement persists. Remaining submissions receive a
single-reviewer semantic-concern pass with a justification for each flagged case.
