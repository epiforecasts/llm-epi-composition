# Analysis Plan: AI-Assisted Epidemic Model Composition

## Research Question

**When task-specific packages are available (EpiEstim, EpiNow2, PyMC), LLMs retrieve rather than compose. When they are not available, can LLMs still produce correct models — and does providing validated primitives via a composable DSL help?**

Secondary questions:
- How do LLMs default to solving Rt estimation when unconstrained by language or framework?
- Does composition ability degrade with task complexity (scenarios 1 → 3)?

## Study History

This study was designed and run in two phases:

1. **Phase 1 (2024-12 to 2025):** Original design under the plan as committed on 2024-12-07 (see git history). Conditions were R / Python / Julia / EpiAware. Some scenario 1a runs were executed; scenario 1b, 2, and 3 runs were not completed. The phase was abandoned before the pre-specified analysis was carried out — no results from phase 1 are used in the confirmatory analysis below.
2. **Phase 2 (2026-04 onwards):** Design revised in response to methodological concerns raised by Omar et al. (*Nat Med* 2026) and observations during phase 1 (notably that "from-scratch" code generation collapses to task-specific packages when they exist). Phase 2 uses simulation with ground truth (new data, not previously queried) as the primary evaluation, plus additional methodological controls. The revised plan is time-stamped by commit to the public repository before any model is queried under it.

Phase 1 artefacts have been removed from the working tree. The original analysis plan, phase 1 prompts, and tracked expert-review scaffolding are preserved in git history and can be reconstructed from earlier commits. Phase 1 run outputs (agent conversation logs, generated code, in-progress expert-review drafts) were gitignored and have not been retained.

## Motivation for Current Design

Following methodological concerns raised by Omar et al. (*Nat Med* 2026) on the evaluation of LLMs, the design was revised to:

1. **Re-pose the primary question** around composition vs retrieval (see above).
2. **Ground-truth recovery on simulated data** as the primary outcome, addressing framing-dependent scoring.
3. **Adversarial DGPs** that stress specific modelling decisions, separating recall of textbook machinery from correct implementation.
4. **Automated correctness detectors** for mechanically detectable departures, with expert review reserved for irreducibly semantic judgments on a stratified subsample.
5. **Prompt-paraphrase randomisation**, with results reported as distributions.
6. **Internal role separation** and explicit blinding tests.
7. **Pre-specified predictions** so non-confirmation is informative.

## Study Design

### Conditions

Three conditions on a composition gradient:

| Condition | Specification | Docs provided | Composition forced? |
|---|---|---|---|
| **no-spec** | None — model chooses language, library, approach | None (implicit in training weights) | No (default path available) |
| **julia** | "Use Julia" | Turing.jl API reference | Partially (no EpiEstim/EpiNow2 equivalent) |
| **epiaware** | "Use EpiAware.jl" | EpiAware API reference (~900 lines) | Yes (DSL primitives) |

**Rationale.** The original R/Python/Julia axis confounded language with ecosystem maturity because EpiEstim/EpiNow2/PyMC are well represented in training data while Julia has no equivalent. The revised axis tests composition directly: default path → composition-forced (no package shortcut) → composition-scaffolded (DSL primitives).

#### Minimum working knowledge (MWK) principle

Documentation is provided to each condition to ensure API-level knowledge is roughly symmetric across conditions, not as a treatment. The test is of composition ability *given* working API knowledge, not of recall of specific APIs. Without this levelling, EpiAware submissions would fail predominantly due to hallucinating non-existent functions, which is uninformative about composition.

**Operational rule** for what may be included in a docs bundle:

- **May include**: function/type signatures, arguments, return types, brief primitive-level usage snippets (single-function examples).
- **May not include**: end-to-end Rt estimation examples, tutorials solving any of scenarios 1a/1b/2/3, vignettes walking through full model construction.

Applied per condition:

- **no-spec**: MWK is implicit in training weights. R/Python + EpiEstim/EpiNow2/PyMC are well-represented.
- **julia**: Turing.jl is in training but thin enough to hallucinate; a Turing.jl API reference is bundled. Nothing Rt-specific.
- **epiaware**: EpiAware is not well-represented in training; the EpiAware API reference is bundled. The current `prompts/epiaware_api_docs.md` is used after checking it complies with the operational rule above (strip any end-to-end Rt examples if present).

**Hallucination rate is recorded as a secondary outcome** (see Evaluation). If hallucination persists in the epiaware condition despite the bundled reference, that is itself a finding about how LLMs use in-context API docs.

### Observation, not arm

Within **no-spec**, record language and package selection. If (as predicted) most submissions default to R + EpiEstim/EpiNow2 for scenarios 1–2, report this distribution as a primary finding. The observation "when not constrained, LLMs overwhelmingly reach for task-specific packages" is itself a result about composition vs retrieval.

### Models Under Evaluation

| Model | Type | Rationale |
|---|---|---|
| Claude Sonnet 4.6 (Anthropic) | Commercial frontier | Current Anthropic frontier-class capability |
| GPT-5 (OpenAI) | Commercial frontier | Frontier-class comparison from a different model family; supports the cross-family generalisation claim |
| Llama 3.1 8B (Meta) | Open-source, small | Tertiary; included to demonstrate local inference on consumer hardware (LMIC accessibility) and to bound expected performance at the low end of the model-scale range |

Two frontier-class models from different families (Anthropic, OpenAI) form the primary basis for cross-family generalisation of any conclusion. Llama 3.1 8B is run with reduced sample size (see Sample size and crossing → Llama) and is reported as a tertiary comparison only; conclusions that hold only on Llama and not on the frontier pair are reported but not generalised.

Findings may not generalise to other model families (Gemini, Qwen, Mistral, DeepSeek, ...). This is a documented limitation.

### Scenarios

Structure unchanged from earlier draft; scenario 3 becomes the key test because all conditions are forced off well-memorised paths (EpiNow2 does not natively support multiple observation streams).

| Scenario | Description | Composition-forcing? |
|---|---|---|
| **1a** | Rt from cases, open method | Weak — package shortcut suffices |
| **1b** | Rt with renewal equation | Weak — package shortcut suffices |
| **2** | + DoW, time-varying ascertainment, NegBin | Moderate — requires observation model extensions |
| **3** | Multi-stream (cases, hospitalisations, deaths) | Strong — no package shortcut in any condition |

The comparison of 1a vs 1b also documents which methods LLMs choose when unconstrained.

### Data: Simulation with Ground Truth

Primary evaluation is on simulated data with known Rt trajectory. Real UK COVID data (UKHSA, England) is retained as a secondary realism check.

#### Primary (canonical) DGP

Individual-level Lloyd-Smith Bellman–Harris age-dependent branching process. The renewal equation is the expectation of this process (Mishra et al. 2020 derive this rigorously for arbitrary offspring distributions); we simulate the realised process directly rather than a moment-closed marginal approximation.

**Infection process (Lloyd-Smith branching).** Each infection at continuous time $t_i$:
- draws an individual reproduction number $\nu_i \sim \mathrm{Gamma}(k, R(t_i)/k)$ (Lloyd-Smith et al. 2005);
- produces $Z_i \sim \mathrm{Poisson}(\nu_i)$ offspring;
- the $j$-th offspring is born at $t_i + \tau_{ij}$ with $\tau_{ij} \overset{\text{iid}}{\sim} g(\tau)$ (continuous Gamma generation interval).

Marginally $Z_i \sim \mathrm{NegBin}(R(t_i), k)$. As $k \to \infty$, individual heterogeneity vanishes and the process reduces to a Poisson branching process whose expectation also satisfies the renewal equation (Cori et al. 2013).

**Observation process (incomplete observation; Poisson measurement):**
$$\mathbb{E}[C_t] = \alpha_t \cdot w_{\mathrm{dow}(t)} \cdot \sum_{e=0}^{E} f_e \cdot I_{t-e}$$
$$C_t \sim \mathrm{Poisson}(\mathbb{E}[C_t])$$

where $I_d$ is the realised count of branching-process infections born in day $d$ (the count is integer-valued by construction). The delay PMF $f_e$ is daily-discretised by double interval censoring of the continuous delay distribution. The recovery target is $R_t$ — the parameter $R(d)$ of the branching process, definition (c) in Funk, Abbott & Bracher (2022, *J R Stat Soc A*). See "Definition of $R_t$" below.

**Sources of overdispersion.** Real epidemic case counts are overdispersed relative to a homogeneous Poisson model. The DGP captures this through individual-level offspring heterogeneity (Lloyd-Smith et al. 2005): different infectious individuals produce widely different numbers of secondary cases. The population-level marginal $I_t \mid I_{<t}$ under this branching process is a compound (non-NegBin) distribution — a sum of NegBins with shared shape parameter but cohort-dependent means. NegBin renewal estimators (e.g. EpiNow2 with `cluster_factor`) fit a NegBinL approximation to this marginal; the approximation is good but not exact. Observations are Poisson — consistent with binomial-thinning of branching-process infections plus measurement noise — without an additional free observation-level dispersion knob.

**GT-vs-estimator structural mismatch.** All standard population-level renewal estimators (Poisson or NegBin) fit a moment-closed marginal to the data, while the GT is the realised individual-level branching process. The mismatch is fundamental: the exact population marginal of a Lloyd-Smith BP is non-NegBin, so even a correctly-specified NegBin renewal estimator is mildly misspecified relative to truth. This mismatch is a fixed mathematical property that applies equally to all submissions, so it does not bias relative comparisons across conditions, scenarios, or LLMs. The pre-registered predictions are framed as relative comparisons.

**Parameters (canonical):**
- $T = 150$ days, nominal start date 2023-01-01
- $R_t$ trajectory: piecewise-linear, $R_t(1)=0.8$ → $R_t(50)=1.5$ → $R_t(100)=0.8$ → $R_t(150)=0.8$ (rise, fall, plateau)
- Generation interval: Gamma, mean 5.5 days, SD 2 days; truncation at $\tau_{\max} = 20$ days
- Delay (infection → report): log-normal, mean 5 days, SD 2 days; truncation at $D_{\max} = 30$ days
- Ascertainment $\alpha_t$: $0.4 + 0.2\sin(2\pi t / T)$ (incomplete observation rate)
- Day-of-week multiplier $w_{\mathrm{dow}(t)}$: $\{1, 1, 1, 1, 1, 0.5, 0.5\}$ for Mon–Sun
- Offspring dispersion $k = 1.0$ (within Lloyd-Smith respiratory-pathogen range)
- Initialisation: seed individuals placed in the pre-observation window $t \in [-\tau_{\max}, 0]$ with continuous timestamps. The seed-count *expected rate* per day follows the Euler–Lotka equilibrium profile $\lambda(d) = c \cdot 100 \cdot \exp(r_0 \cdot d)$ (with $r_0$ the equilibrium growth rate matching $R_t(1) = 0.8$ under the daily-discretised GI; $c = 0.3$ chosen so that the BP's realised rate at $t = 1$ matches the moment-closed expectation). The realised seed *counts* per day are Poisson draws around that expected rate, and the seed individuals' sub-day timestamps are uniform within their day. Each seed individual then produces offspring per the Lloyd-Smith mechanism, and those offspring produce their own offspring recursively through the pre-observation window before the observation window begins. Seed individuals themselves do not count as new obs-window infections; their realised descendants do. The BP is thus stochastic from seed onwards, but the expected seed rate is a deterministic anchor (replacing it with a longer pure-BP burn-in starting from a single ancestor far in the past is feasible but adds inter-replicate variance not relevant to the LLM-composition test).

**Multi-stream parameters (scenario 3):**

Shared $R_t$ and shared latent $I_t$; each stream has its own delay distribution, ascertainment trajectory, and dispersion.

| Stream | Delay (log-normal) | Ascertainment $\alpha_{\text{stream}}(t)$ |
|---|---|---|
| Cases | mean 5d, SD 2d | $0.40 + 0.20\sin(2\pi t / T)$ |
| Hospitalisations | mean 10d, SD 3d | $0.040 + 0.020\sin(2\pi t / T + \pi/3)$ |
| Deaths | mean 20d, SD 5d | $0.008 + 0.004\cos(2\pi t / (1.5 T))$ |

Ascertainment trajectories are phase-shifted or use a different period across streams so that the three series carry partially independent ascertainment signals rather than a single sinusoidal structure scaled three ways. Values are plausible for a moderately severe respiratory pathogen; the DGP is not labelled as a specific disease. Per-stream observations share the same Poisson measurement model and inherit infection-level overdispersion through $k$.

**Disease labelling.** The simulation is not labelled as COVID-19 or any other specific disease. Data files, prompts, and metadata describe "an infectious disease outbreak" with no country or pathogen named. This prevents the LLM from leaning on disease-specific priors or memorised parameter values from training data.

#### Definition of $R_t$

In a stochastic data-generating process, "$R_t$" admits multiple legitimate definitions (Funk, Abbott & Bracher 2022). For a daily Poisson renewal model these include:

(a) the realised ratio $I_t / \Lambda_t$ where $\Lambda_t = \sum_s g_s I_{t-s}$ — a noisy quantity that fluctuates around the rate due to Poisson sampling;
(b) a per-step random multiplier (does not arise in the model used here, which has no random R per time step);
(c) the parameter $R(d)$ — the deterministic function plugged into the renewal equation as the rate's multiplier.

**The recovery target is (c) — the parameter $R(d)$.** This is the quantity targeted by EpiEstim, EpiNow2, and renewal-equation Bayesian methods (the dominant approaches in the literature and the predicted default for LLM submissions). Methods that target (a) — notably Wallinga–Teunis — recover a noisy version of $R(d)$ that converges to (c) at large counts but disagrees at finite counts; this disagreement does not reflect implementation error and is flagged in expert review on the scenario 1a method-identification subsample.

#### Generation procedure

For each DGP variant and replicate seed:

1. Compute the per-stream delay PMFs $f_e^{\text{stream}}$ for $e = 0, \ldots, D_{\max}$ by **double interval censoring** of the continuous delay distribution via numerical quadrature:
   $$P(D = d) = \int_0^1 \big[F(d + 1 - p) - F(\max(d - p, 0))\big] dp$$
   renormalised over the truncation window. Compute the daily-discretised GI PMF $g_s$ similarly (used only for the Euler–Lotka root). The GI itself is treated as a continuous distribution in the BP simulation.
2. Solve $1 = R_0 \sum_s g_s e^{-r(s-1)}$ for $r_0$ at $R_0 = R_t(1)$.
3. Place seed individuals in the pre-observation window with continuous timestamps. Per day $d \in [-(\tau_{\max} - 1), 0]$, draw $n_d \sim \mathrm{Poisson}\big((1 - R_0) \int_{d-1}^d 100 \exp(r_0 t) dt\big)$ and place each individual at a uniformly distributed sub-day time within the day. The $(1 - R_0)$ rescaling compensates for the geometric inflation that would otherwise arise from BP propagation of a seed whose rate is itself the equilibrium rate.
4. **BP propagation.** Each seed individual generates offspring per Lloyd-Smith: draw $\nu_i \sim \mathrm{Gamma}(k, R(t_i)/k)$, $Z_i \sim \mathrm{Poisson}(\nu_i)$, each offspring's continuous GI $\tau_{ij} \sim g(\tau)$, offspring birth at $t_i + \tau_{ij}$. Each offspring is added to a time-ordered queue and processed in turn by the same procedure, generating its own offspring, until the queue is exhausted or all timestamps exceed $T$. Seed individuals themselves do not count as new infections; their descendants do.
5. Aggregate realised infections to integer daily counts $I_d$ for $d = 1, \ldots, T$ (and similarly for the seed-window aggregate, which is used only as input to the observation convolution for early obs days).
6. For each observation stream, compute the deterministic expected report:
   $$\mathbb{E}[C_d^{\text{stream}}] = \alpha_{\text{stream}}(d) \cdot w_{\mathrm{dow}(d)} \cdot \sum_{e=0}^{D_{\max}} f_e^{\text{stream}} \cdot I_{d-e}.$$
7. Sample $C_d^{\text{stream}} \sim \mathrm{Poisson}(\mathbb{E}[C_d^{\text{stream}}])$ per day per stream.
8. Write `data/cases.csv` (+ `hospitalisations.csv`, `deaths.csv`), `truth/true_rt.csv`, `truth/true_infections.csv`, `truth/true_expected.csv`, `truth/params.json`, `truth/sim_script.jl`.

For scenarios 1a/1b/2, only `cases.csv` is copied into the agent sandbox; scenario 3 receives all three streams.

**Choices fixed in the plan:**
- Infections are realised samples from an individual-level Lloyd-Smith Bellman–Harris age-dependent branching process. The recovery target is the parameter $R(d)$ governing the offspring distribution, not any realisation-level quantity.
- Twenty independent replicates per variant, seeds $\{101, \ldots, 120\}$. Across replicates, both the realised branching trajectory and the observation noise vary. Within-cell variance across replicates is reported as a distribution.
- Dates are synthetic; no calendar features beyond day-of-week.
- Discretisation enters only at the observation step (delay PMF) via double interval censoring. The infection-process GI is continuous (per-individual draws).

**Sanity check before running any LLM condition.** The reference EpiAware implementations are applied to the canonical DGP across all 20 replicates and must satisfy:

- Median Rt RMSE on the evaluation window (days 25–125) < 0.10.
- Median 90% coverage on the same window ≥ 0.80 (i.e. not under-covering by more than 10pp).
- Median calibration |median(coverage) − 0.90| ≤ 0.10.

The criterion is **asymmetric**: under-coverage invalidates the reference (intervals too tight to contain the truth); over-coverage means the reference is conservatively calibrated (intervals wider than nominal). Conservative coverage is acceptable because LLM-composition analyses compare *relative* coverage across conditions, not absolute coverage against the 90% nominal.

The reference solutions use `HalfNormal(0.05)` for the AR(1) innovation std prior. Calibration on canonical (20 reps): median RMSE = 0.086, median coverage = 1.00, calibration error = 0.10. The reference passes the asymmetric criterion. Reps 1, 10, 17 cover at [0.82, 0.95] (rep 1 with `HalfNormal(0.025)`); reps 11, 13, 19, 20 just above 0.95; the remaining 13 reps over-cover at 1.00. Walking down to `HalfNormal(0.025)` does not substantially relax the over-coverage on the smooth-trajectory replicates (their R(t) is well-matched to an AR(1) prior over a wide range of std priors); the over-coverage is structural, not a tuning issue.

Rationale for accepting conservative coverage: the AR(1) prior with reasonable scale gives intervals that contain the truth on essentially every smooth trajectory of this length. The 90% nominal coverage is meaningful only when the data forces the credible interval to be informative; on these trajectories, the prior's natural width exceeds what 90% nominal requires. This is a property of any reasonable AR(1)-based reference and does not invalidate the reference for the study's purpose.

#### Adversarial DGPs

Each variant stresses a single modelling decision; a submission missing that component is expected to show scenario-specific bias.

Each variant differs from the canonical DGP in exactly one parameter:

| Variant | Perturbation | Stresses | Predicted failure if missing |
|---|---|---|---|
| Short GI | Gamma mean 2.5d, SD 1d | Discretisation; short-GI dynamics | Bias if continuous density evaluated at integers |
| Long delay | Log-normal mean 10d, SD 3d | Delay handling | Rt estimate lagged/compressed near end |
| Strong DoW | Weekend multiplier 0.25 | Observation model | Oscillating Rt |
| High ascertainment variability | $\alpha_t = 0.4 + 0.35\sin(2\pi t / T)$ | Ascertainment model | Spurious Rt trend |
| Low dispersion | $k = 1000$ (near-Poisson infections) | Likelihood / overdispersion | No effect — null condition |
| Extreme dispersion | $k = 0.1$ (stress test beyond observed pathogen ranges; Lloyd-Smith $k$ for SARS was ~0.16) | Likelihood / overdispersion | Overconfident intervals if Poisson used |
| Abrupt change | $R_t$ drops 1.5 → 0.5 over 3 days around day 75 | Smoothness prior | Over-smoothed estimators lag the drop |
| Sinusoidal Rt | $R(t) = 1.0 + 0.4\sin(2\pi t / 60)$ — three full cycles across 150 days | Smoothness prior choice | Estimators with priors that prefer piecewise-linear (or that over-smooth) recover Rt only in a low-pass-filtered form |

**Rationale for DGP selection.** Each variant stresses one of the components that appears in the canonical DGP and in the reference specification (GI, delay, observation model with DoW, ascertainment, dispersion / likelihood, smoothness prior). The adversarial DGPs are therefore not hand-picked to match failures we expect to find; they enumerate the modelling decisions that a correctly-specified renewal-equation model must handle. Any component *not* adversarially stressed would be a gap in the evaluation.

The low-dispersion (Poisson-like) variant is included deliberately as a null condition: Poisson submissions should perform comparably to NegBin submissions here, but diverge specifically on `extreme_dispersion`. This controls for the possibility that "bad" submissions just fail everywhere (making the adversarial panel uninformative) versus failing in component-specific ways (making it diagnostic).

**Short-GI caveat.** The short-GI perturbation changes both discretisation sensitivity (the intended stress) and epidemic dynamics (shorter GI → sharper rise and peak under the same $R_t$). The two effects cannot be fully separated within a single variant without departing from the "perturb one parameter" principle. Recovery on short_gi is therefore interpreted as a combined stress on discretisation handling and estimator robustness to faster dynamics; this is noted when reporting the adversarial-panel results.

**Mechanism of dispersion variants.** The `low_dispersion` and `extreme_dispersion` adversarial variants perturb the offspring dispersion $k$ (Lloyd-Smith parameter) at the infection level — varying the population-level NegBin mass-shape implied by individual-level offspring heterogeneity. Estimators that model NegBin observations (e.g. EpiNow2's `cluster_factor`) absorb this overdispersion through their observation likelihood; estimators that assume Poisson observations (e.g. EpiEstim) will show undercoverage on `extreme_dispersion`. The `extreme_dispersion` $k = 0.1$ value is *beyond* the range observed in nature; it is a deliberate stress test, not a realistic-pathogen scenario (Lloyd-Smith reported $k \approx 0.16$ for SARS).

**Why simulation-based evaluation addresses contamination.** The DGP is canonical (LLMs have seen renewal-equation structure in training data) but the specific data does not match any training example. Grading on recovery against truth detects cases where the model recalls textbook machinery but implements it with missing components, because missing components cause bias in scenario-specific ways.

#### Isolation of simulation parameters from agent runs

Simulation parameters, the true Rt trajectory, and the data-generation script are stored separately from the observed data files and are never copied into an agentic run's working directory. Repository layout:

```
simulations/
  generate.jl
  Project.toml
  Manifest.toml
  {variant}/                # canonical, short_gi, long_delay, strong_dow,
                            #   high_asc_var, low_dispersion, extreme_dispersion,
                            #   abrupt_change, sinusoidal_rt
    rep_{01..20}/
      truth/                # true_rt.csv, true_infections.csv, true_expected.csv,
                            #   params.json, sim_script.jl — never exposed to agent
      data/                 # cases.csv (+ hospitalisations.csv, deaths.csv for scenario 3)
                            #   — only files copied to the agent
```

`run_agentic.sh` copies `simulations/{variant}/rep_{rr}/data/` (and the prompt + docs bundle per condition) into a `mktemp` working directory; nothing else from the repository is visible to the agent. For scenarios 1a/1b/2 only `cases.csv` is copied; scenario 3 receives all three stream files. Observed files are formatted identically to real-data files so the agent cannot distinguish simulated from real data by file structure. Evaluation (RMSE, coverage) runs outside the agent's sandbox after the run completes, using the truth files the agent never saw.

#### Discretisation and Censoring

The generator operates at daily resolution and discretises continuous GI / delay distributions to daily PMFs by **double interval censoring**:

- **Primary censoring**: primary event (infection) occurs at an unknown time within its day
- **Secondary censoring**: secondary event (onset, reporting) also occurs at an unknown time within its day

Per-day mass: $P(D = d) = \int_0^1 [F(d + 1 - p) - F(\max(d - p, 0))] dp$, computed by numerical quadrature and renormalised to sum to 1 over the truncation window.

A discrete-time estimator operating on daily data requires a discretised GI and delay. Naive discretisation (evaluating the PDF at integer points) is inconsistent with the generator and biased for short means; the short-GI adversarial variant tests this directly. Estimators that use double interval censoring or another method that explicitly addresses the censoring problem are aligned with the generator and should recover $R(d)$ within tolerance.

Acceptable estimator-side approaches:
- Double interval censoring (e.g. EpiAware's `censored_pmf()`)
- Midpoint discretisation with appropriate justification
- Any method that explicitly addresses the censoring problem

#### Inference Approach

Prompts do not specify an inference approach. Bayesian MCMC, variational inference, maximum likelihood, and other approaches are all acceptable provided they produce point estimates and (ideally) uncertainty quantification for $R_t$ over time. Whether uncertainty is provided is recorded as an automated criterion.

### Reference Implementations

Reference implementations serve as a sanity check on the adversarial DGPs (they should recover truth) and as a benchmark for visualisation. They are **not** used as a grading target — grading is on recovery against simulation truth.

| Scenario | Reference implementation |
|---|---|
| 1a/1b | EpiAware: Renewal + AR(1) latent + NegBin obs with delay |
| 2 | As above + DoW + time-varying ascertainment |
| 3 | As above + `StackObservationModels` for multi-stream |

An EpiNow2 run is also provided on the same data for scenarios 1a, 1b, 2 (EpiNow2 does not support multi-stream, by design of this study).

## Evaluation

### Primary: Recovery against simulation truth

- **Rt RMSE** against true trajectory, averaged over the **evaluation window days 25–125**.
- **90% coverage** of credible/confidence intervals over the same window.
- **Calibration** of uncertainty (width of interval vs error magnitude) over the same window.

The evaluation window excludes the first 24 days (where the lookback into the seed window dominates and any reasonable estimator has insufficient information) and the last 25 days (where the longest reporting delay — 20 days for deaths, plus the truncation buffer — means the latest observed cases reflect infections from outside the simulation horizon). $T = 150$ days; window = days 25–125 inclusive (101 day points).

Each metric computed per submission per DGP variant. Primary result is the distribution of RMSE and coverage within each (scenario × condition) cell, across paraphrases × replicates × runs (canonical-DGP only for primary; full variant panel for adversarial fingerprint).

### Secondary: Hallucination and iteration behaviour

Recorded per run:

- **Hallucination rate**: number of iterations that failed due to "function X does not exist" / "no method matching" / undefined-symbol errors, as a fraction of total iterations.
- **Iterations to success**: number of iterations required to reach runnable code (or NA if not reached).
- **Error type distribution**: syntax, runtime, fitting failure, convergence failure.

Reported per condition. If hallucination rate is materially higher in `epiaware` despite the bundled API reference, or in `julia` despite the Turing.jl reference, that is a finding about in-context docs use independent of the composition result.

### Diagnostic: Automated structural-pattern detectors

Static-analysis detectors that **flag mechanically detectable structural patterns** in submitted code and outputs. Each flag identifies a pattern, not a "correctness departure" — the appropriate model for any given task is a judgment call that depends on the data-generating mechanism. Detectors are developed on a training subsample and calibrated against expert review (Cohen's kappa). They are diagnostic instruments for analysis, not graders. Implemented in `evaluation/detectors.py`.

| Detector flag | Pattern detected | Approach | Feasibility |
|---|---|---|---|
| `flag_poisson_only` | Poisson observation likelihood without NegBin alternative | AST/regex | Clean |
| `flag_no_smoothing_term` | No AR/RW/GP/spline term on Rt or its log | Regex (with known false negatives — e.g., custom multivariate Normal priors with smoothing covariance) | Clean |
| `flag_no_delay_handling` | No reporting-delay convolution or delay distribution use | Regex | Mostly clean |
| `flag_no_uncertainty` | Output file has no `Rt_lower`/`Rt_upper` columns | Output check | Clean |
| `flag_naive_density_at_integers` | Continuous density evaluated at integer points without integration | Regex (heuristic) | Partial |
| `flag_negative_rt` | Output contains negative Rt entries | Output check | Clean |
| `flag_normal_observation` | Observation modelled with `Normal`/`Gaussian` rather than count distribution | Regex (heuristic) | Partial |
| `flag_confused_rt_r` | Confusion between $R$ (fixed) and $R_t$ (time-varying) | Semantic — expert review only | Hard |

Dropped from the taxonomy given the revised design (GI and delay are provided as fixed known distributions):

- `fixed_gi` — no longer an error if the prompt specifies fixed values
- `wrong_gi`, `wrong_delay` — trivial typo checks, not interesting
- `si_not_gi` — implausible when the prompt explicitly labels the distribution as generation interval

### Diagnostic: Targeted expert review

Expert review is reserved for:

1. **Stratified subsample** (~20% of submissions, balanced across scenarios × conditions) — inter-rater reliability and calibration of automated detectors against human judgment.
2. **Semantic departures** across all submissions: `si_not_gi`, `confused_rt_r`, `wrong_likelihood` (beyond Poisson/NegBin).
3. **Scenario 1a method identification** — judgment call: renewal equation / Wallinga-Teunis / Bettencourt-Ribeiro / naive ratio / other.

**No LLM assistance in expert review.** The framing sensitivity of LLMs (including when used as judges) undermines the objectivity the expert review is supposed to provide.

#### Departure classification (for the subsample and semantic departures)

- **A** Equivalent alternative — different but equally valid
- **B** Minor error — unlikely to substantially affect results
- **C** Major error — would bias results
- **D** Fundamental misunderstanding — lack of grasp of underlying epi/stats

Classification is cross-referenced with recovery: a "C" classification with good recovery (or vice versa) is a case worth discussion, not a contradiction.

## Randomisation

### Prompt paraphrases

For each (scenario, condition), k=5 paraphrases. The three LLM-paraphrase slots are deliberately allocated across model families so that paraphraser-side wording bias does not coincide with the evaluator's family.

- **Slot 01 — Original.** The base prompt as written by the project authors.
- **Slot 02 — Internal manual rewrite.** Manual rewrite by a project author other than the original prompt author. This author is *not* blinded to the predictions or study design — they are an existing project member with full access to the plan — so this slot does not provide hypothesis-blinded paraphrasing. What it does provide is wording-decorrelation from the original prompt-author's house style, which is a real but weaker mitigation of the author-design critique.
- **Slot 03 — OpenAI** (GPT-5).
- **Slot 04 — Google** (Gemini 2.5 Flash).
- **Slot 05 — Anthropic** (Claude Sonnet 4.5; same family as the primary evaluator, included as a within-family sanity check).

LLM paraphrases are generated by `evaluation/generate_paraphrases.py`, which calls each provider's API directly with the paraphrasing instruction and `temperature = 0.7`. The paraphrasing call is API-mode (not Claude Code), so it does set temperature; this is independent of the evaluator runs (which still use Claude Code at its default temperature — see "Temperature is not a randomisation axis" above).

If a third-party infectious-disease modeller external to the project can be recruited, their two manual rewrites replace slots 03 and 04. Their brief is committed to the repository as `prompts/paraphrase_brief.md`. Recruitment is not a precondition for running the study.

Five paraphrases × 4 scenarios × 3 conditions = 60 prompt variants.

The choice of three frontier families (OpenAI, Google, Anthropic) is not exhaustive: Mistral, Qwen, DeepSeek, xAI, Cohere, and others are omitted on cost grounds. We document the choice as a limitation.

**API docs MWK validation.** Before pre-registration is finalised, the bundled API docs (`prompts/turing_api_docs.md`, `prompts/epiaware_api_docs.md`) are checked for any content that constitutes a worked Rt example or otherwise violates the MWK operational rule. This is done by:

1. The second internal author (who did not write the docs) reading every section of both files against the explicit checklist in `prompts/paraphrase_brief.md`. They have access to the predictions and study design — blinding is not necessary for this structured "is this a worked Rt example?" check.
2. Two separate LLM instances (different families, no study context) performing the same check as an additional cross-check.

Any flagged content is edited out and the validation re-run. Both validations and any edits are listed in the commit history.

### Runs

n=10 runs per (scenario, condition, paraphrase) for the primary recovery analysis on the canonical variant. n=5 runs per (scenario, condition, paraphrase, variant) for the adversarial-DGP fingerprint analysis. See "Sample size and crossing" below.

### Temperature is not a randomisation axis in this study

The Anthropic Claude Code CLI used to drive agent runs does not expose a `temperature` parameter at invocation time. Implementing a separate API-mode runner (re-implementing the agentic loop on top of the bare Messages API) would substantially expand scope. We therefore do *not* include temperature as a randomisation axis. Within-cell variability across runs of the same (scenario, condition, paraphrase) reflects only the model's intrinsic stochasticity at the API's default temperature plus any internal nondeterminism. Documented explicitly in Limitations.

### Sample size and crossing

These dimensions are crossed multiplicatively unless stated otherwise:

**Primary recovery analysis** (Tables 1, 5, 6; Figures 1, 5, 6):
- Variant: canonical only.
- Replicate: 3 of the 20 generated replicates per cell, seeds {101, 102, 103}.
- Paraphrase: 5.
- Runs: 10 per (scenario, condition, paraphrase, replicate) cell.
- Total per (scenario, condition): 5 × 3 × 10 = 150 runs.
- Total per model across all (scenario, condition) cells: 12 × 150 = 1800 runs.

**Adversarial-DGP fingerprint analysis** (Table 2, Figure 2):
- Variant: 8 (all of the panel).
- Replicate: 3 per variant, seeds {101, 102, 103}.
- Paraphrase: paraphrase 01 only.
- Runs: 5 per (scenario, condition, variant, replicate).
- Total per (scenario, condition): 8 × 3 × 5 = 120 runs.
- Total per model: 12 × 120 = 1440 runs.

**Across two models (Sonnet 4.6 + a second frontier model):** ~6500 runs total.

The primary and adversarial analyses are independent — the runs do not overlap. Running them sequentially, the harness cost is bounded.

For Llama 3.1 8B (if retained as a tertiary model), reduce within-cell replication to keep total wall-clock under one week of single-GPU compute; report the reduced sample size explicitly.

### Reporting

Every result reported as a distribution across paraphrases × replicates × runs, not a single pass rate. Primary figures show distributions (violin/box plots); point estimates as summaries only.

## Blinding

- **Expert review blinding.** Submissions are preprocessed deterministically to strip imports and package-specific syntax before review. Where full stripping is infeasible (e.g. scenario 3 multi-stream structure), the effective blinding is tested: reviewers guess the condition on a calibration subset; the blinding-failure rate is reported as a study limitation.
- **Internal role separation.** Prompt design, detector implementation, and review coordination are assigned to different team members. Documented explicitly in the paper.

## Protocol

### Prompt construction

Standardised prompts per (scenario, condition) contain:
- Clear problem statement (epidemiological task)
- Data description and format (no disease or country label)
- Language/framework specification per condition table
- Epidemiological parameters that would plausibly come from external studies (see below)
- For `julia`: Turing.jl API reference in working directory
- For `epiaware`: EpiAware API reference in working directory (conforming to the MWK operational rule — no end-to-end Rt examples)

**Information provided vs estimated.** To isolate composition from parameter-guessing, prompts give the LLM values that would realistically come from external epidemiological studies, and require the LLM to estimate everything else.

Provided in the prompt (as fixed, known distributions from an external study; LLMs are not expected to propagate uncertainty on these):
- Generation interval: family, mean, SD (e.g. "Gamma, mean 5.5 days, SD 2 days, from external study")
- Delay distribution(s) per stream: family, mean, SD
- Structural features of the data that are known (e.g. "counts show a weekly cycle"; "ascertainment varies over time")

Estimated by the LLM:
- $R_t$ trajectory (target)
- Ascertainment $\alpha_t$ (both structure and values)
- Dispersion $\phi$
- Smoothness hyperparameters (AR/RW/GP/spline choice and hyperpriors)

Prompts do not provide true parameter values for quantities the LLM is expected to estimate, and do not disclose the simulation DGP.

**Phase 1 prompts are obsolete.** The existing prompts in `prompts/scenario_*/` describe UK COVID-19 case counts with no parameters given; they will be rewritten for the simulation phase to match the specification above.

### Execution: agentic approach

Each LLM is given the prompt and asked to write code, execute it, and fix errors iteratively. This reflects realistic use of coding assistants (Claude Code, Cursor, etc.).

**Protocol:**
1. LLM writes code, executes, and fixes errors
2. Maximum 10 iterations per run
3. 10-minute timeout per execution attempt; 60-minute total session timeout
4. All iterations, error messages, and fixes logged
5. Isolated working directory — model cannot see reference solutions, other runs, or study design

**Tools:**
- Claude Code for Claude models
- Aider with Ollama for open-source models

**Recorded per run:**
- Final code and outputs
- Number of iterations required
- Error types encountered
- Whether the run succeeded within iteration limit

### Expert review protocol

- Two independent infectious-disease modellers review the stratified subsample and semantic departures
- Reviewers blinded to LLM and condition (via stripping preprocessor)
- Each reviewer independently assesses each code sample
- Inter-rater reliability assessed (Cohen's kappa)
- Disagreements resolved by discussion; third reviewer consulted if needed
- **No LLM assistance permitted**

## Pre-specified Predictions

Stated here before any model is queried under this revised design. Each prediction names a quantitative effect size we will treat as confirming the prediction. "95% bootstrap CI" refers to a non-parametric bootstrap over the unit-of-replication implied by the comparison (e.g. paraphrase × replicate × run cells), with 1000 resamples.

1. **no-spec defaults to packages.** In ≥70% of (no-spec, scenario 1a) and (no-spec, scenario 1b) submissions, the produced code uses R + EpiEstim or R + EpiNow2 or Python + PyMC, regardless of model. *Confirmation:* the lower 95%-CI bound of the proportion is ≥ 0.70.
2. **Recovery is comparable across conditions on scenarios 1a/1b.** The median Rt RMSE difference between any pair of conditions on (scenario 1a) and (scenario 1b) is ≤ 0.02. *Confirmation:* all three pairwise condition contrasts on each scenario have a 95%-CI that includes 0 and an absolute-median-difference ≤ 0.02.
3. **EpiAware shows lower Rt RMSE than Julia-bare on scenarios 2–3.** Median Rt RMSE in (epiaware, scenario 2) is at least 0.02 lower than in (julia, scenario 2); the gap on scenario 3 is at least 0.04 lower. *Confirmation:* both differences have 95% bootstrap CIs that exclude zero in the predicted direction.
4. **no-spec fails on scenario 3 more often than EpiAware.** Run-level failure rate (run did not produce a valid `outputs/rt_estimates.csv` within 10 iterations) in (no-spec, scenario 3) is at least 20 percentage points higher than in (epiaware, scenario 3). *Confirmation:* the difference has a 95%-CI excluding zero in the predicted direction and lower bound ≥ 0.20.
5. **Adversarial DGP performance correlates with automated detector flags.**
   - Submissions flagged `no_delay`: median Rt RMSE on `long_delay` is at least 0.05 higher than on `canonical`. 95%-CI excludes zero.
   - Submissions flagged `flag_poisson_only` (Poisson-only observation likelihood): median 90% coverage on `extreme_dispersion` is at least 15 percentage points lower than on `low_dispersion`. 95%-CI excludes zero.
6. **Hallucination rate is higher in `epiaware` than in `julia` or `no-spec`.** Median fraction of agent iterations failing with "function does not exist" / "no method matching" / undefined-symbol errors in `epiaware` is at least 10 percentage points higher than in either `julia` or `no-spec`. 95%-CIs exclude zero.

Predictions 3–5 are the load-bearing composition claims. If they do not hold, the study reports that validated composable tooling does not provide a composition benefit over forced-composition baselines, which is itself informative. Prediction 6 is orthogonal: a finding about in-context docs use rather than composition per se.

## Pre-specified Tables and Figures

### Tables

**Table 1: Recovery by condition × scenario.** Rows: condition (3) × scenario (4) = 12 rows. Columns: median Rt RMSE, IQR of RMSE, median coverage, IQR of coverage, across all paraphrases × replicates × runs on the canonical DGP.

**Table 2: Recovery on adversarial DGPs.** Rows: condition × scenario × DGP variant. Columns as Table 1. Shows scenario-specific bias patterns.

**Table 3: Language and package selection in no-spec.** Rows: LLM × scenario. Columns: distribution of language choice (R / Python / Julia / other), distribution of package choice.

**Table 4: Method selection in scenario 1a.** Rows: condition × LLM. Columns: renewal / Wallinga-Teunis / Bettencourt-Ribeiro / naive / other. From expert review of this subsample.

**Table 5: Automated detector rates.** Rows: departure category. Columns: rate by condition × scenario.

**Table 6: Inter-rater reliability and detector validation.** Cohen's kappa between reviewers; agreement between detectors and reviewer classification on the stratified subsample.

### Figures

**Figure 1: Primary result — recovery distributions.** Violin plots of Rt RMSE by condition, faceted by scenario. Shows full distribution over paraphrases × replicates × runs on canonical DGP.

**Figure 2: Adversarial DGP fingerprint.** Heatmap: rows = DGP variant, columns = condition × scenario. Cell colour = median RMSE relative to canonical. Reveals which conditions fail on which stress tests.

**Figure 3: Rt trajectories.** Representative trajectories per condition × scenario, overlaid on true Rt. Ribbon for uncertainty.

**Figure 4: Correctness ↔ recovery linkage.** Scatter: each point a submission. X = number of automated detector flags; Y = Rt RMSE. Annotated with predicted failure modes for each flag.

**Figure 5: Hallucination and iteration behaviour.** Per condition: hallucination rate (errors from non-existent functions / method matching), iterations to first successful run, error-type distribution. Tests whether bundled API docs equalise working knowledge across conditions.

**Figure 6: Sensitivity to prompt paraphrase.** For a representative (scenario, condition) cell, full distribution of outcomes across paraphrases and runs. Demonstrates the within-cell variability Omar et al. highlight.

## Limitations Acknowledged in Advance

- **Author-designed prompts.** The *base* prompts are designed by the project authors, and the second human-paraphrase wave is by another project author who is *not* blinded to the predictions (the project does not have an author who has not seen the plan). Three paraphrases are LLM-generated, deliberately spread across three frontier model families (OpenAI, Google, Anthropic) so that paraphraser-side wording bias does not coincide with the evaluator's family. The paraphrase randomisation mitigates wording-level effects but not framing-level effects of the base prompt or of an author-internal house style. A hypothesis-blinded human paraphraser (project author or external) and a fully external base-prompt-design exercise are out of scope here; we identify both as the strongest improvements for replication studies.
- **Training-data contamination at DGP level.** The renewal equation is canonical in training data. Simulation with ground truth addresses *data* contamination but cannot address *structural* contamination. Adversarial DGPs that stress specific modelling decisions partially mitigate by separating "recalled machinery" from "correctly implemented machinery".
- **Model coverage.** Two frontier model families (Anthropic, OpenAI) plus one small open-source model (Meta Llama 3.1 8B). Findings may not generalise to Gemini, Qwen, Mistral, DeepSeek, or other frontier families.
- **No independent replication.** We publish the full harness and invite replication with a pre-specified concordance criterion (primary recovery claim replicated if point estimates within 10pp and same qualitative ordering of conditions).
- **Simulation realism.** Real-data secondary check may reveal issues not captured in simulation.
- **No temperature randomisation.** Claude Code CLI does not expose `temperature`. We do not include temperature as a randomisation axis in this study (see Randomisation → "Temperature is not a randomisation axis"). Within-cell variability across runs reflects only the model's intrinsic stochasticity at the API's default temperature.
- **Detectors are heuristics, not graders.** Regex- and AST-based pattern detectors have known false negatives. For example, `flag_no_smoothing_term` does not match a custom multivariate-Normal prior with smoothing covariance even though that constitutes smoothing. Detectors are calibrated against expert review on the stratified subsample (Cohen's kappa) and reported as instruments for analysis, not ground truth.
- **Composition test concentrates in scenario 3.** Scenarios 1a/1b/2 test whether a Bayesian PPL adds value over a default-package shortcut and whether estimator-side choices affect recovery. Composition under genuine no-shortcut conditions is most directly tested in scenario 3, where multi-stream estimation has no canonical package. Predictions 3 and 4 (scenarios 2–3) are the load-bearing composition claims.
- **Rt-definition ambiguity.** Under any stochastic generator, "$R_t$" admits multiple legitimate definitions — the parameter, the realised ratio, and (in some models) a per-step random multiplier (Funk, Abbott & Bracher 2022). Recovery is scored against the parameter $R(d)$. Methods that target the realised ratio (e.g. Wallinga–Teunis) recover a noisier quantity that converges to the parameter at scale; observed disagreement with truth in their case partly reflects target choice rather than implementation error, and is flagged in the scenario 1a method-identification subsample.
- **Single mechanism for overdispersion.** All overdispersion in the GT arises from infection-level offspring heterogeneity (Lloyd-Smith et al. 2005). Other plausible sources (random reporting effort, batched-report processing artefacts, day-of-day administrative noise) are not modelled. Estimators that absorb such effects via NegBin observation likelihoods may handle real data better than they handle our GT, where the same parameter is doing different mechanistic work.
- **Ascertainment is purely temporal.** The GT models ascertainment as a deterministic time-varying multiplier; in real surveillance, *which* individuals get reported depends on severity, age, healthcare access, and other individual covariates. Estimators correctly modelling individual-level ascertainment heterogeneity would not be advantaged on this GT.
- **Multi-stream observation noise is independent.** The three streams (cases, hospitalisations, deaths) share the same latent infection process but their observation noise is independent across streams. Real multi-stream surveillance has correlated observational error (a hospital-system disruption affects both hospitalisations and same-day reports). Multi-stream estimators that exploit cross-stream noise correlation would have nothing to gain on this GT.
- **Scenarios 1a and 1b may be functionally equivalent.** 1a says "open method"; 1b says "use the renewal equation". Most submissions in 1a will use a method that internally implements the renewal equation (e.g. EpiNow2). The 1a/1b distinction is testing what the LLM verbalises about its method, not what it computes — relevant for method-identification analyses but possibly not for recovery.
- **Reviewer blinding.** Package imports are strippable; structural features (multi-stream handling, Julia vs R syntax patterns) may leak condition information. Blinding failure rate is tested and reported.

## Pre-registration

This protocol is time-stamped by commit to the public git repository before any model is queried under the revised design. The commit hash fixing the plan will be cited in the paper. Phase 1 runs are not combined with confirmatory analysis; the confirmatory analysis uses simulated data generated under this plan.

## Discussion Points

### Composition vs retrieval

The primary framing is that LLMs retrieve solutions where task-specific packages exist and must compose where they do not. This study is designed to separate these modes. The no-spec condition measures default behaviour; the Julia condition forces composition (no package shortcut available); the EpiAware condition scaffolds composition with validated primitives.

### Minimum working knowledge and the docs question

Docs are provided to level API-level knowledge across conditions, not as a treatment (see MWK principle under Conditions). A cleaner decomposition of "DSL primitives" vs "worked examples" would require an epiaware condition with Rt-specific tutorials, which leaks solutions and defeats the composition test. The compromise is that we cannot fully separate "primitives help" from "API knowledge helps". We partially address this through the hallucination-rate measurement: if hallucination is the dominant failure mode in `epiaware`, the composition result is confounded with in-context docs use; if hallucination is low, the comparison with `julia` isolates the composition benefit.

### Complexity gradient

Scenarios 1a/1b/2/3 form a gradient of composition-forcing. A DSL benefit that is flat across scenarios suggests a general effect (docs help); a benefit that grows with complexity suggests the composition advantage the DSL is designed to provide.

## Ethical Considerations

- No human subjects
- All data publicly available or simulated
- LLM outputs reviewed before any public release

---

*Document created: 2024-12-07*
*Revised: 2026-04-23 (recovery-based evaluation, no-spec/Julia/EpiAware axis, adversarial DGPs, automated detectors, prompt/temperature randomisation, minimum working knowledge docs principle, hallucination rate as secondary outcome)*
*Status: Revised draft, pending pre-registration*
