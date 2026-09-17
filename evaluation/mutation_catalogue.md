# Mutation catalogue

This catalogue is not used in the current study design, which does not assess
reviewability. It is retained for a possible follow-up study of defect detection.

This document lists candidate defects for an injected-defect reviewability outcome.
Package-specific operations below are legacy development examples and require
validation against the pinned ComposableTuringIDModels API and revised scientific
tasks before use. A mutation must introduce an actual modelling error; changing
a conditional Poisson observation model or using supplied ascertainment is not
intrinsically a defect. The frozen matched-defect schedule includes only validated,
applicable operations.

The catalogue is internal to the review-coordinator role and **not**
distributed to reviewers. Reviewers know the space of possible defects
(because they know what components each scenario requires) but not which
sample has which injected defect.

## Migration and eligibility checks

Retain the EpiAware operations below as a translation checklist. They are not
executable instructions for ComposableTuringIDModels. For each mutation selected
for the frozen schedule, record the legacy operation, replacement operation,
pinned package commit, applicable scenarios and applicable conditions. Validate
the replacement on both an unmodified and a mutated sample before approving it.

`evaluation/mutate.py` currently implements legacy concrete patterns. Its regex
helpers may be reusable, but each selected pattern needs implementation and
validation against the pinned API and generated-code forms. A successful text
replacement is not proof that exactly one intended scientific defect was created.
Record the tool version, diff and independent validation result; manual mutations
require the same checks.

For H3, eligibility is the intersection of three requirements:

- The mutation applies to scenario 2 or 3, as assigned in the schedule.
- The same scientific defect can be introduced in both `julia` and `composable`.
- Both selected base submissions support a validated implementation of that defect.

Catalogue tags such as “all scenarios” describe broader potential use; they do
not expand the H3 pool. A condition-specific operation can implement a shared
defect, but a defect applicable to only one condition cannot enter an H3 matched
pair. Such defects may be retained for separate descriptive or detector-validation
work. Do not substitute a different defect in the other condition to fill a pair.
Check these requirements when freezing the schedule and again when constructing
the sealed manifest. If the planned pool cannot be assembled, follow the protocol's
not-estimable rule rather than changing the schedule after review outcomes appear.

## Convention

Each mutation has:

- **ID** — a short handle used in logs and score sheets.
- **Component** — the plan's component-correctness flag the mutation targets.
  Ground truth for detector calibration.
- **Difficulty** — subjective class: *obvious* / *moderate* / *subtle*.
- **Applies to** — scenarios and conditions the mutation can be applied to.
- **Operation** — the concrete change made to the source.

Detectability class is a coordinator judgement, not a study outcome; it's
used to balance the review pool so reviewers see a mix of difficulties.

## Mutations targeting `flag_no_dow` (day-of-week effect)

**M_dow_remove_obvious** · difficulty: obvious · scenarios 2, 3
- Delete the day-of-week term entirely. In EpiAware submissions this means
  removing `ascertainment_dayofweek(...)` from the observation block; in
  raw Turing / numpyro / PyMC it means deleting the `dow_effect[dow_idx]`
  multiplier from the expected-count expression.
- Component: `flag_no_dow`.

**M_dow_broadcast_wrong** · difficulty: moderate · scenarios 2, 3
- Keep the DoW parameter block but broadcast it against the wrong axis
  (e.g. use `dow[weekday]` where `weekday` indexes a *replicate* rather
  than the day of the week). Model runs; posterior has no DoW signal.
- Component: `flag_no_dow` (detector should NOT catch this; expert
  reviewers should).

## Mutations targeting `flag_no_ascertainment`

**M_asc_remove_obvious** · difficulty: obvious · scenarios 2, 3
- Drop the supplied ascertainment trajectory from the expected-count expression,
  so reports are modelled as if every infection were observed. Raw code: replace
  `alpha[t] * ...` with `...`; package code: remove the ascertainment component
  wrapping the observation model.
- Component: `flag_no_ascertainment`.

**M_asc_wrong_date** · difficulty: moderate · scenarios 2, 3
- Apply the supplied ascertainment on the infection date (inside the delay
  convolution) where the task specifies the report date. The model runs and the
  multiplier is present, but it is misaligned by the reporting delay.
- Component: `flag_no_ascertainment` (detector will not catch; reviewer should
  notice on inspection).

## Mutations targeting `flag_no_delay_handling`

**M_delay_remove** · difficulty: obvious · all scenarios
- Delete the reporting-delay convolution. In EpiAware: remove
  `LatentDelay(...)` wrapping the observation model. In raw code: replace
  `sum(f[e] * I[t-e] for e in 0:D)` with `I[t]` (immediate observation).
- Component: `flag_no_delay_handling`.

## Mutations targeting `flag_no_censoring`

**M_censor_replace_with_pdf** · difficulty: moderate · all scenarios
- Replace `censored_pmf(...)` (or the CDF-difference discretisation) with
  a naïve `pdf(dist, 1:D)` lookup at integer points.
- Component: `flag_no_censoring` and `flag_naive_density_at_integers`.

## Mutations targeting `flag_no_truncation`

**M_trunc_remove** · difficulty: moderate · all scenarios
- Remove PMF renormalisation and the `Truncated(...)` wrapper. Model uses
  an unrenormalised PMF over the finite window.
- Component: `flag_no_truncation`.

**M_trunc_shorten_wrong** · difficulty: subtle · all scenarios
- Shorten the delay support so it excludes substantial probability mass and
  renormalise over the shortened window (e.g. `D_max = 30` to `D_max = 6` for a
  delay with mean 5 days). The model runs; the effective delay is too short.
- Component: `flag_no_truncation` (detector will not catch; reviewer might).

## Mutations targeting `flag_no_multistream_latent` (scenario 3 only)

**M_stream_independent** · difficulty: obvious · scenario 3
- Break shared-latent structure: fit three independent Rt latent processes,
  one per stream, and glue their outputs together at the CSV stage
  (e.g. average the three posteriors). Removes `StackObservationModels`
  and any shared `Renewal` block.
- Component: `flag_no_multistream_latent`.

**M_stream_cases_only** · difficulty: obvious · scenario 3
- Ignore two of the three streams. Only `cases.csv` is used; the model is
  scenario-2-shaped.
- Component: `flag_no_multistream_latent`.

## Mutations targeting `flag_no_smoothing_term`

**M_smooth_remove** · difficulty: obvious · all scenarios
- Replace the AR(1) / random-walk / GP latent process on log-Rt with an
  independent-day prior (each day of Rt drawn from the same Normal, no
  temporal coupling).
- Component: `flag_no_smoothing_term`.

**M_smooth_scale_wrong** · difficulty: subtle · all scenarios
- Keep the smoothing structure but change the innovation scale by 10× in
  either direction. Model looks correct; Rt trajectory is over- or under-
  smoothed.
- Component: `flag_no_smoothing_term` (detector won't catch; reviewer
  might).

## Mutations targeting subtle numerical / structural errors

**M_gi_reversed** · difficulty: moderate · all scenarios
- Reverse the generation-interval vector in the convolution (`g[e]` used
  as `g[end - e + 1]`). Off-by-window; recovery degrades noticeably.
- Component: no specific detector. Expert only.

**M_conv_offbyone** · difficulty: subtle · all scenarios
- Introduce an off-by-one in the delay convolution index (`I[t - e]` used
  as `I[t - e + 1]` throughout). Rt estimate shifted by one day.
- Component: no specific detector. Expert only.

**M_prior_wrong** · difficulty: subtle · all scenarios
- Change the `initialisation_prior` for the log-Rt AR(1) to something
  nonsensical (`Normal(10, 5)` when it should be `Normal(0, 0.5)`).
  Sampler may diverge or explore weird posterior modes.
- Component: no specific detector. Expert only.

## Injection and scoring

Use [the expert review protocol](review_protocol.md) for the authoritative sample
allocation, masking, timing and scoring rules. H3 uses 24 matched pairs of generated
base submissions (48 bases), each with one clean and one mutated version. The
96 versions each receive one review; a reviewer never sees both versions of a base.
Reference mutations are separate detector-validation materials and do not enter H3.

Record mutation ID, base and pair IDs, source run, applicability checks, validation
result and reviewer assignment in the sealed manifest. Match defect type across
conditions within pairs. Do not infer clean status from passing automated detectors.

Report detections and denominators even when H3 cannot support a confirmatory
conclusion. A wide valid interval leaves the hypothesis unresolved; a degenerate
or invalid interval does not establish certainty. H3's multiplicity allocation
remains reserved. See the analysis plan and review protocol for the exact rules.

## Review workload

The H3 pool entails 96 reviews, up to 72 person-hours at the 45-minute cap.
See the analysis plan's Review workload section for the full review budget.
