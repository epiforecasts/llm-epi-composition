# Expert review protocol

This protocol produces the reviewability, component-correctness, and
scenario-1a method-identification data used in the paper. Reviewers see
raw code — blinding to condition is not attempted, because language and
package structure identify the condition on inspection. Instead the review
pass uses **injected defects** to give reviewability a ground-truth signal
that doesn't depend on blinding.

Reviewers work independently. A shared discussion happens only after each
reviewer has finished the review pool.

## What the review measures

The review pool contains a mix of:

- **Defect-free samples** — reference solutions and a curated set of real
  LLM submissions that pass every automated detector.
- **Mutated samples** — reference solutions with one defect from the
  mutation catalogue (`evaluation/mutation_catalogue.md`) injected by the
  review coordinator.

For each sample, reviewers list the defects they identify. Scoring per
condition (analysis plan §Reviewability):

- **Sensitivity** = injected defects correctly flagged / total injected
  defects. The headline reviewability outcome.
- **Specificity** = clean samples correctly unmarked / total clean samples.
- **Precision** = injected defects correctly flagged / (injected + spurious).

The reviewer knows which condition each sample is from (this is visible
from the code). Sensitivity is still meaningful because the reviewer has
to actually read the code to find the injected defect; knowing the
condition doesn't tell them where the defect is.

The pool also includes each scenario-1a real submission, for method
identification (renewal vs Wallinga-Teunis vs …). Method identification
is factual and unaffected by blinding.

## What the review no longer does

The earlier draft asked reviewers for subjective readability and
confidence ratings on a 1–5 scale. Those are dropped from the confirmatory
pass because knowing the condition would bias the ratings: reviewers who
recognise a package they consider well-designed would inflate readability,
and familiarity would inflate confidence. Both remain optional descriptive
fields on the sheet but are not reported as confirmatory outcomes.

## Setup per reviewer

Each reviewer is given:

- A **review folder** on disk containing samples named `sample_{ID}/` where
  `{ID}` is a random integer assigned by the coordinator. The mapping from
  `{ID}` to (source: reference or submission; condition; scenario; mutation
  applied if any) is kept sealed by the coordinator until both reviewers
  have submitted.
- Inside each `sample_{ID}/`:
  - `code/`: the source file(s) for the sample. If the sample is a
    mutated reference, the mutation has been applied and no marker is left
    behind indicating that.
  - `data_summary.txt`: a fixed short description of the data the model is
    supposed to consume. Same text across samples for the same scenario.
  - `review_sheet.md`: blank sheet to fill in.
- The **scenario prompt** for orientation (the shared statement of what
  the scenario asks for; not the specific paraphrase text).
- The **component list per scenario** (see below). This tells reviewers
  which components a fully-specified submission would include.
- The **departure classification** definitions (A/B/C/D).
- The **semantic-departure flag list** (`confused_rt_r`, `wrong_likelihood`,
  `si_not_gi`).

Reviewers do **not** have access to:

- Which samples are clean and which are mutated.
- The mutation catalogue itself.
- Model, paraphrase, or run identity (for real submissions).
- The DGP script, true `Rt`, or generated infections.
- Any automated detector outputs.
- Other reviewers' sheets until the discussion round.

## Time recording

The reviewer starts a timer at the moment they open `sample_{ID}/` and
stops it at the moment they finalise the defect list. If a sample stalls
past 45 minutes, the reviewer stops, records "gave up at 45 min", and
moves on; time for that sample is recorded as ≥ 45 min.

## Review sheet

Filled in per sample. All fields are required.

```
Sample ID: {ID}
Reviewer: {A|B}

Time to complete review (minutes): __

Defects identified. List every defect you see. Do not restrict yourself to
the component checklist. For each defect, note the component (if it maps
onto one) and a one-sentence description.

  Format per defect:
    [component_flag] short description
  where component_flag is one of:
    dow, ascertainment, delay, censoring, truncation, poisson_only,
    multistream, smoothing, other

  Example:
    [dow] no day-of-week term in the observation model
    [other] generation-interval vector appears reversed relative to the
            convolution

  Enter one line per defect. Leave blank if none.

Component checklist. Independently of the defect list above, tick the box
for each component you would say is PRESENT and correct in this sample.
This is the factual-identification pass. Missing / unclear / present-but-
broken all count as "not ticked".

  [ ] Renewal-equation infection process
  [ ] Discretised generation interval
  [ ] Reporting-delay convolution
  [ ] Delay handled with censoring / interval integration
  [ ] Delay support truncated with renormalisation
  [ ] Day-of-week / weekend effect
  [ ] Time-varying ascertainment structure
  [ ] Overdispersed observation likelihood
  [ ] Smoothing prior on Rt (AR / RW / GP / spline)
  [ ] Shared latent across streams (scenario 3 only)

Departure classification (single choice, for real LLM submissions only):
  A = Equivalent alternative
  B = Minor error
  C = Major error
  D = Fundamental misunderstanding

Justification (1–2 sentences):

Semantic departure flags (tick any that apply):
  [ ] confused_rt_r
  [ ] wrong_likelihood
  [ ] si_not_gi

Method identification (scenario 1a real submissions only):
  [ ] Renewal equation
  [ ] Wallinga–Teunis
  [ ] Bettencourt–Ribeiro
  [ ] Naive ratio
  [ ] Other: __

Free-text notes (optional, 3 sentences max):
```

## Scenario expectations

- **Scenario 1a — Rt from cases, open method.** Renewal or renewal-equivalent
  method; discretised GI; delay handling; truncation.
- **Scenario 1b — Rt from cases with renewal.** As 1a, plus explicit
  renewal equation.
- **Scenario 2 — Cases with DoW, time-varying ascertainment, overdispersion.**
  As 1b, plus day-of-week effect, time-varying ascertainment structure, and
  overdispersed likelihood.
- **Scenario 3 — Multi-stream.** As 2, plus a single shared latent
  generating three streams through separate observation models.

Components not required for a scenario should not be ticked on the
checklist even if they happen to be present.

## Coordinator role

The **review coordinator** (see Roles & Responsibilities in the plan):

- Builds the review pool from reference solutions and clean real
  submissions.
- Applies mutations per `mutation_catalogue.md`, one per mutated sample,
  covering the difficulty and component distribution the catalogue
  requires.
- Assigns random IDs, distributes samples in randomised order (not grouped
  by condition, scenario, or clean/mutated status).
- Keeps the sealed ID → (source, condition, mutation) mapping until both
  reviewers have submitted.
- Does not review any samples themselves.

## After both reviewers finish

- **Sensitivity, specificity, and precision** are computed per condition
  from the defect lists against the sealed mutation record.
- **Detector validation**: reviewers' component-checklist ticks are
  compared against `evaluation/detectors.py` output on the same samples,
  giving a confusion matrix and Cohen's kappa per detector.
- **Inter-rater kappa** is computed on the component checklist, the
  departure classification, and the semantic-departure flags.
- **Disagreements** are resolved by discussion. If discussion fails, a
  third reviewer is consulted; their independent call is tie-breaking.

## Semantic-departure lightweight pass

The review pool covers the stratified subsample plus mutation-injected
controls. Every remaining real submission (outside the subsample) receives
a lightweight pass:

- One reviewer per submission.
- One-page sheet: submission ID, three tick boxes (`confused_rt_r`,
  `wrong_likelihood`, `si_not_gi`), one-sentence justification per ticked
  box, no timing, no checklist.

## No LLM assistance

Reviewers may not use any LLM (Claude, GPT, Gemini, or otherwise) to help
with reading code, filling the checklist, or writing the defect list.
Framing sensitivity in LLM judges is a known concern; using them would
compromise the independence of expert review from the primary evaluation.
