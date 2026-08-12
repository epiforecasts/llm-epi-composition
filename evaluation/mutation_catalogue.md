# Mutation catalogue

This document defines the defects that can be injected into review samples for
the reviewability outcome (analysis plan, evaluation axis 4). Each mutation
takes a defect-free source file and produces a variant that omits, breaks,
or subtly perverts a modelling component. Reviewers try to identify the
defects on a mixed pool of clean and mutated samples. Detection sensitivity
per condition is a confirmatory outcome.

The catalogue is internal to the review-coordinator role and **not**
distributed to reviewers. Reviewers know the space of possible defects
(because they know what components each scenario requires) but not which
sample has which injected defect.

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
- Replace the time-varying ascertainment structure with a scalar constant.
  EpiAware: remove the `Ascertainment(...)` block wrapping the observation
  model. Raw code: replace `alpha[t]` with `alpha` (single scalar).
- Component: `flag_no_ascertainment`.

**M_asc_fixed_wrong** · difficulty: moderate · scenarios 2, 3
- Keep a vector `alpha[t]` but hard-code it to a constant across time
  (`alpha[t] = 0.5` for all t). Model appears time-varying but isn't.
- Component: `flag_no_ascertainment` (detector may not catch; reviewer
  should notice on inspection).

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

**M_trunc_extend_wrong** · difficulty: subtle · all scenarios
- Extend the truncation window to a value that includes near-zero mass
  (e.g. change `D_max = 30` to `D_max = 90` for a delay with mean 5).
  Doesn't break the model but adds costly zero-contribution convolution
  terms and slightly changes the discrete PMF's renormalisation.
- Component: `flag_no_truncation` (detector won't catch; reviewer might).

## Mutations targeting `flag_poisson_only`

**M_likelihood_poisson** · difficulty: obvious · scenarios 2, 3
- Replace `NegativeBinomialError(...)` (EpiAware) or the negative-binomial
  observation likelihood with a Poisson. Delete any dispersion parameter.
- Component: `flag_poisson_only`.

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

## Injection protocol

The **review coordinator** produces the review pool as follows:

1. Start with defect-free base samples. First-pass source: the three
   condition-specific reference solutions per scenario
   (`reference_solutions/*.jl`). Second-pass source (optional, only if
   time permits): a curated set of real LLM submissions that pass every
   automated detector.
2. For each (scenario, condition, base sample), generate up to 8 mutated
   variants, each with exactly one mutation from the catalogue. Choose
   mutations to cover a range of difficulties and to include at least one
   mutation per required-component per scenario.
3. Include unmutated controls: for each (scenario, condition), 3–5 samples
   with no injected defect.
4. Randomise the order in which each reviewer sees samples so that clean
   controls are interleaved with mutations across conditions.
5. Record, per sample, the mutation ID applied (or "clean"). Reviewers do
   not see this until after the review pass completes.

Blinding to which specific mutation was applied is achievable in a way
that blinding to condition is not: the mutation-selection is a coordinator
choice invisible to the reviewer, and mutations are applied
deterministically enough that no visual tell survives.

## Scoring

Per (reviewer, sample):

- If the sample is mutated with mutation M:
  - **True positive** if the reviewer's defect list includes M (or a
    reasonable paraphrase of it under the component ontology).
  - **False negative** if the reviewer's list omits M.
- If the sample is clean:
  - **True negative** if the reviewer's list is empty (or contains only
    component-not-required-here entries).
  - **False positive** for any spurious defect claim.

Per condition:

- **Sensitivity** = TP / (TP + FN) — the fraction of injected defects
  reviewers catch. This is the headline reviewability metric.
- **Specificity** = TN / (TN + FP)
- **Precision** = TP / (TP + FP)

Per mutation type:

- Detection rate = fraction of applications caught by any reviewer, and by
  both reviewers.
- Comparison against detector output for the corresponding
  component-correctness flag (calibration data for the automated detectors).

## Cost

For a review pool of 40 samples (roughly 3 mutations per scenario per
condition, 12 scenarios × 3 conditions × 3 mutations minus overlap, plus
4 clean controls), at 20–40 minutes per sample per reviewer, two reviewers,
the total reviewer time is 27–54 person-hours. Reasonable for an expert
review pass.
