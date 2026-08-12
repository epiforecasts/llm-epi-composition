# Expert review protocol

This document defines the review pass that produces the reviewability,
interpretability, and component-correctness data used in the paper. Every
submission in the stratified subsample plus every submission flagged as a
semantic departure is reviewed under this protocol.

Reviewers work independently. A shared discussion happens only after each
reviewer has finished the subsample.

## Setup per reviewer

Each reviewer is given:

- A **review folder** on disk. Each submission is a subfolder named
  `submission_{ID}/` where `{ID}` is an anonymised integer identifier
  assigned by the review coordinator. The mapping from `{ID}` to (scenario,
  condition, model, paraphrase, run) is kept sealed by the coordinator and
  not accessible to reviewers during the pass.
- Inside each `submission_{ID}/`:
  - `code/`: the agent's final scripts, after passing through
    `evaluation/blind_submission.py`. Package imports and package-namespaced
    calls are stripped or genericised; filename comments referring to a
    condition are removed. What remains is the modelling logic in
    condition-neutral form.
  - `output.csv`: the agent's `outputs/rt_estimates.csv` (columns
    `date, Rt_median, Rt_lower, Rt_upper` or a subset).
  - `data_summary.txt`: summary statistics of the data the agent saw
    (number of observations, streams if any, structural features to reproduce
    if any). Same summary text for every submission on the same scenario, so
    reviewers can orient themselves without seeing the DGP.
  - `review_sheet.md`: a blank copy of the sheet below, for the reviewer to
    fill in.
- A copy of the **scenario prompt** the agent received (without the
  paraphrase-specific wording; the shared common statement of what the
  scenario asked for).
- The four **departure classification** labels (A, B, C, D) with their
  definitions.
- The list of expected modelling components per scenario (below).

Reviewers do not have access to:

- Model identity, paraphrase identity, or condition.
- The DGP script, true `Rt` trajectory, or generated infections.
- Any automated detector outputs.
- Other reviewers' sheets until the discussion round.

## Time recording

The reviewer starts a timer at the moment they open `submission_{ID}/`, and
stops it at the moment they write "confident assessment reached" or "gave up"
on the sheet. If the reviewer is interrupted, they pause the timer. The
reviewer records elapsed minutes on the sheet.

If a reviewer cannot reach a confident assessment within 60 minutes for a
single submission, they mark it "not-confident within 60 min" and move on;
the time is recorded as ≥ 60 min for the reviewability distribution.

## Review sheet

Reviewers fill in this sheet per submission. All fields are required.

```
Submission ID: {ID}
Reviewer: {A|B}

Time to confident assessment (minutes): __

Confidence in the reconstruction (1–5): __
  1 = "I cannot say what this model is doing"
  2 = "I have a guess but I would not defend it"
  3 = "I have a working reconstruction I could defend under mild challenge"
  4 = "I am confident in the reconstruction and know its main defects"
  5 = "I am confident in every component and could re-implement it from
       memory"

Readability (1–5): __
  1 = "Incomprehensible; would refuse to review as a paper"
  2 = "Barely readable; heavy interpretation required"
  3 = "Readable but not clean; standard PhD-student output"
  4 = "Clean and well-organised; a colleague could pick this up quickly"
  5 = "Crisp; every step is signposted and every decision is legible"

Component identification. For each component listed under the scenario, mark
one of: PRESENT / ABSENT / UNCLEAR. If PRESENT, note briefly whether the
implementation looks correct, buggy, or ambiguous.

  Renewal-equation infection process:              [PRESENT/ABSENT/UNCLEAR]  __
  Discretised generation interval:                 [PRESENT/ABSENT/UNCLEAR]  __
  Reporting-delay convolution:                     [PRESENT/ABSENT/UNCLEAR]  __
  Delay handled with censoring/truncation:         [PRESENT/ABSENT/UNCLEAR]  __
  Day-of-week / weekend effect:                    [PRESENT/ABSENT/UNCLEAR]  __
  Time-varying ascertainment structure:            [PRESENT/ABSENT/UNCLEAR]  __
  Overdispersed observation likelihood:            [PRESENT/ABSENT/UNCLEAR]  __
  Smoothing prior on Rt (AR/RW/GP/spline):         [PRESENT/ABSENT/UNCLEAR]  __
  Shared latent across streams (scenario 3):       [PRESENT/ABSENT/UNCLEAR]  __

Departure classification (single choice):
  A = Equivalent alternative: different but equally valid
  B = Minor error: unlikely to substantially affect results
  C = Major error: would bias results
  D = Fundamental misunderstanding: lack of grasp of underlying epi/stats

Justification for departure classification (1–2 sentences):

Free-text notes (optional, 3 sentences max):

Semantic departure flags. Check any that apply:
  [ ] confused_rt_r         (confusion between R fixed and Rt time-varying)
  [ ] wrong_likelihood      (something other than Poisson or NegBin, not just Normal)
  [ ] si_not_gi             (serial interval used where generation interval was
                             specified, or vice versa)
```

## Scenario expectations

Reviewers are told, per scenario, which components a fully-specified
submission would include. This information is on the scenario prompt itself;
reviewers do not need to guess.

- **Scenario 1a — Rt from cases, open method.** Renewal or renewal-equivalent
  method; discretised GI; delay handling.
- **Scenario 1b — Rt from cases with renewal.** As 1a, plus explicit renewal
  equation.
- **Scenario 2 — Cases with DoW, time-varying ascertainment, overdispersion.**
  As 1b, plus day-of-week effect, time-varying ascertainment structure, and
  overdispersed likelihood (typically NegBin).
- **Scenario 3 — Multi-stream (cases, hospitalisations, deaths).** As 2, plus
  a single shared latent (Rt or infection process) generating three streams
  through separate observation models.

Components not required for a scenario should be marked ABSENT without
counting against the submission.

## Scenario 1a method identification

For scenario-1a submissions, reviewers additionally classify the modelling
method as one of:

- Renewal equation (any Bayesian renewal implementation)
- Wallinga–Teunis
- Bettencourt–Ribeiro (SIR-based Rt)
- Naive ratio (cases[t] / cases[t-1] or similar)
- Other (with a one-line note)

## Semantic departures across all submissions

For submissions not in the stratified subsample, only three flags are
checked (this is a light-touch pass that touches every submission):

- `confused_rt_r`
- `wrong_likelihood` (beyond Poisson / NegBin)
- `si_not_gi`

The lightweight pass has its own single-page sheet: submission ID, three
tick boxes, and one-sentence justifications only if a box is ticked. No
timing, no readability, no per-component identification.

## Coordination and blinding integrity

The **review coordinator** (see Roles & Responsibilities in the analysis
plan) is responsible for:

- Assembling `submission_{ID}/` folders from `runs/` with the blinding
  preprocessor applied.
- Assigning IDs so that neither reviewer sees the same submission twice under
  different IDs.
- Distributing folders to reviewers in a randomised order (not grouped by
  scenario or condition).
- Recording the ID → (scenario, condition, model, paraphrase, run) mapping in
  a sealed file, opened only after both reviewers have submitted.
- Running the **blinding-integrity check**: on a calibration set of 24
  submissions (2 per (scenario × condition) cell), reviewers guess the
  condition. The blinding-failure rate is reported as a study limitation.

## After both reviewers finish

- **Inter-rater kappa** is computed per outcome (departure classification,
  each component's PRESENT/ABSENT judgment, readability rating).
- **Detector validation**: the reviewers' component-identification calls are
  compared against `evaluation/detectors.py` output on the same submissions
  to produce the per-detector confusion matrix and Cohen's kappa per
  detector.
- **Disagreements** on departure classification and component identification
  are resolved by discussion between the two reviewers. If discussion does
  not resolve a case, a third reviewer is consulted; their independent call
  is treated as tie-breaking.
- **Semantic departure pass**: the coordinator merges the lightweight-pass
  results, resolves single-reviewer disagreements by discussion if any, and
  writes them into the analysis dataset.

## No LLM assistance

Reviewers may not use any LLM (Claude, GPT, Gemini, or otherwise) to help
with reading the code, interpreting components, or filling in the sheet.
Framing sensitivity in LLM judges is a known concern; using them here would
compromise the review's independence from the primary evaluation.
