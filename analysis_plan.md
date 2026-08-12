# Analysis Plan: AI-Assisted Epidemic Model Code Generation

## Framing

This study varies *observable software interventions* — task instruction, required
language or framework, and provided API documentation — and measures *observable
outcomes* of the code the agent produces. It makes no claim about whether the agent
retrieved a memorised solution or composed one from primitives; those are latent
modes that cannot be identified from LLM outputs. Package use is not evidence of
retrieval, and package-free code is not evidence of composition. Differences between
conditions are reported as effects of the intervention on the produced code, not as
inferences about internal generation strategy.

An earlier framing of this study (2026-04 through 2026-07) described the three
conditions as a retrieval-versus-composition gradient. That framing was retired in
August 2026 for the identifiability reason above. The simulation DGP, harness, pilot
data, and measurements developed under that framing remain valid because they are all
observable; only the causal narrative around them has been rewritten.

## Research Question

How do software instructions and available scaffolding — from unconstrained free
choice, through Julia with a general-purpose probabilistic programming framework, to
Julia with a domain-specific composable framework — affect the correctness,
component fidelity, reliability, interpretability, reviewability, and cost of
LLM-generated epidemiological model code?

Secondary questions:
- What tools do LLMs choose when unconstrained? Which packages are actually reached
  for on which scenarios?
- Which modelling components (delay convolution, censoring, truncation, day-of-week
  effect, ascertainment structure, dispersion) are correctly identified as needed and
  correctly implemented under each intervention?
- Does the magnitude of the scaffolding effect vary with model capability
  (Haiku → Sonnet → Opus)?

## Study History

This study was designed and run in two phases:

1. **Phase 1 (2024-12 to 2025):** Original design under the plan as committed on 2024-12-07 (see git history). Conditions were R / Python / Julia / EpiAware. Some scenario 1a runs were executed; scenario 1b, 2, and 3 runs were not completed. The phase was abandoned before the pre-specified analysis was carried out, and no results from phase 1 are used in the confirmatory analysis below.
2. **Phase 2 (2026-04 onwards):** Design revised in response to methodological concerns raised by Omar et al. (*Nat Med* 2026) and observations during phase 1 (notably that "from-scratch" code generation collapses to task-specific packages when they exist). Phase 2 uses simulation with ground truth (new data, not previously queried) as the primary evaluation, plus additional methodological controls. The revised plan is time-stamped by commit to the public repository before any model is queried under it.

Phase 1 artefacts have been removed from the working tree. The original analysis plan, phase 1 prompts, and tracked expert-review scaffolding are preserved in git history and can be reconstructed from earlier commits. Phase 1 run outputs (agent conversation logs, generated code, in-progress expert-review drafts) were gitignored and have not been retained.

## Motivation for Current Design

Following methodological concerns raised by Omar et al. (*Nat Med* 2026) on the
evaluation of LLMs, the design uses:

1. **Ground-truth recovery on simulated data** as the primary correctness signal,
   removing framing dependence from the scoring.
2. **Adversarial DGPs** that stress specific modelling decisions, so that submissions
   missing a component (delay, censoring, day-of-week, overdispersion) are
   distinguishable from those that include it.
3. **Automated structural-pattern detectors** for mechanically detectable code
   properties, calibrated against expert review on a stratified subsample.
4. **Multiple observable outcome axes** (correctness, component fidelity,
   interpretability, reviewability, reliability, robustness, maintainability,
   epistemic quality) rather than a single primary outcome.
5. **Prompt-paraphrase randomisation**, with results reported as distributions.
6. **Blinded expert review** on a stratified subsample plus semantic-departure cases.
7. **Pre-specified confirmatory outcomes** with quantitative thresholds; the rest are
   reported descriptively without pre-registered claims.

## Study Design

### Conditions

Three conditions, each described as an *intervention* on the agent's task
environment:

| Condition | Intervention | Docs provided | Available shortcuts |
|---|---|---|---|
| **no-spec** | Unconstrained: agent chooses language, package, and approach | None (implicit in training weights) | High-level packages such as EpiEstim, EpiNow2, PyMC are freely available |
| **julia** | Julia required | Turing.jl API reference | General-purpose PPL only; no domain-specific Rt package |
| **epiaware** | EpiAware.jl required | EpiAware API reference (~900 lines) | Domain-specific composable primitives |

The intervention varies along two dimensions in parallel: the language/framework
constraint tightens from left to right, and the available domain-specific vocabulary
shifts from "whatever the agent knows from training" to "explicit primitives supplied
in-context". Differences between conditions therefore reflect a bundle of changes,
not a single manipulation. This is reported plainly rather than attributed to any
one factor.

#### Minimum working knowledge (MWK) principle

Documentation is provided to each condition to ensure API-level knowledge is roughly symmetric across conditions, not as a treatment. The intervention is about which vocabulary and constraints are placed on the code, given roughly-equalised API knowledge — not about recall of specific APIs. Without this levelling, EpiAware submissions would fail predominantly due to hallucinating non-existent functions, which is a docs-availability finding rather than a scaffolding one.

**Operational rule** for what may be included in a docs bundle:

- **May include**: function/type signatures, arguments, return types, brief primitive-level usage snippets (single-function examples).
- **May not include**: end-to-end Rt estimation examples, tutorials solving any of scenarios 1a/1b/2/3, vignettes walking through full model construction.

Applied per condition:

- **no-spec**: MWK is implicit in training weights. R/Python + EpiEstim/EpiNow2/PyMC are well-represented.
- **julia**: Turing.jl is in training but thin enough to hallucinate; a Turing.jl API reference is bundled. Nothing Rt-specific.
- **epiaware**: EpiAware is not well-represented in training; the EpiAware API reference is bundled. The current `prompts/epiaware_api_docs.md` is used after checking it complies with the operational rule above (strip any end-to-end Rt examples if present).

Hallucination rate is recorded as a secondary outcome (see Evaluation). If hallucination persists in the epiaware condition despite the bundled reference, that is itself a finding about how LLMs use in-context API docs.

### Observation, not arm

Within **no-spec**, the language and package the agent selects is recorded per
submission. The distribution of choices — R+EpiEstim, R+EpiNow2, Python+PyMC,
Python+numpyro, Julia+Turing, or something else — is reported per (scenario, model)
as descriptive evidence of what unconstrained LLM agents actually reach for.

### Models Under Evaluation

| Model | Type | Rationale |
|---|---|---|
| Claude Haiku 4.5 (Anthropic) | Commercial frontier (small-tier) | Low end of the Claude size spectrum; tests whether scaffolding effects depend on model capability |
| Claude Sonnet 4.6 (Anthropic) | Commercial frontier (mid-tier) | Default frontier-class deployment for most coding-agent applications |
| Claude Opus 4.7 (Anthropic) | Commercial frontier (top-tier) | High end of the Claude size spectrum; tests whether reasoning depth changes scaffolding effects |
| Qwen3-Coder-30B-A3B-Instruct (Alibaba) | Open-source, coding-tuned | Tertiary; open-weight comparison. Run on the LSHTM HPC via the local vLLM + Qwen Code CLI stack (`~/code/dotfiles/lshtm-local-llm-stack.md`). Included to bound expected open-source performance and demonstrate that the study protocol is reproducible without a commercial API. |

The primary panel is three Claude models at three scales (Haiku → Sonnet → Opus), enabling a within-family scaling test. Cross-family generalisation is not claimed on the basis of this study; a single-family design is chosen because the primary Anthropic research credits fund only Claude runs. Qwen3-Coder-30B is a tertiary open-weight comparison run with reduced sample size (see Sample size and crossing).

Findings may not generalise to other model families (Gemini, GPT, Mistral, DeepSeek). This is a documented limitation.

### Scenarios

| Scenario | Description | Package-shortcut availability |
|---|---|---|
| **1a** | Rt from cases, open method | Multiple canonical packages (EpiEstim, EpiNow2, PyMC) cover this task |
| **1b** | Rt with renewal equation | Same packages cover this task |
| **2** | + DoW, time-varying ascertainment, NegBin | EpiNow2 covers most components; some extension required |
| **3** | Multi-stream (cases, hospitalisations, deaths) | No canonical R/Python package supports shared-latent multi-stream Rt estimation |

Scenario 3 is the most methodologically informative: no dominant package covers the
task, so the agent's tool choice and code structure become directly observable
regardless of condition. The 1a versus 1b comparison also records which method the
agent chooses when unconstrained.

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

where $I_d$ is the realised count of branching-process infections born in day $d$ (the count is integer-valued by construction). The delay PMF $f_e$ is daily-discretised by double interval censoring of the continuous delay distribution. The recovery target is $R_t$, the parameter $R(d)$ of the branching process: definition (c) in Funk, Abbott & Bracher (2022, *J R Stat Soc A*). See "Definition of $R_t$" below.

Real epidemic case counts are overdispersed relative to a homogeneous Poisson model. The DGP captures this through individual-level offspring heterogeneity (Lloyd-Smith et al. 2005): different infectious individuals produce widely different numbers of secondary cases. The population-level marginal $I_t \mid I_{<t}$ under this branching process is a compound (non-NegBin) distribution: a sum of NegBins with shared shape parameter but cohort-dependent means. NegBin renewal estimators (e.g. EpiNow2 with `cluster_factor`) fit a NegBinL approximation to this marginal; the approximation is good but not exact. Observations are Poisson (consistent with binomial-thinning of branching-process infections plus measurement noise), without an additional free observation-level dispersion knob.

All standard population-level renewal estimators (Poisson or NegBin) fit a moment-closed marginal to the data, while the GT is the realised individual-level branching process. The mismatch is fundamental: the exact population marginal of a Lloyd-Smith BP is non-NegBin, so even a correctly-specified NegBin renewal estimator is mildly misspecified relative to truth. This mismatch is a fixed mathematical property that applies equally to all submissions, so it does not bias relative comparisons across conditions, scenarios, or LLMs. The pre-registered predictions are framed as relative comparisons.

**Parameters (canonical):**
- $T = 150$ days, nominal start date 2023-01-01
- $R_t$ trajectory: piecewise-linear, $R_t(1)=0.8$ → $R_t(50)=1.5$ → $R_t(100)=0.8$ → $R_t(150)=0.8$ (rise, fall, plateau)
- Generation interval: Gamma, mean 5.5 days, SD 2 days; truncation at $\tau_{\max} = 20$ days
- Delay (infection → report): log-normal, mean 5 days, SD 2 days; truncation at $D_{\max} = 30$ days
- Ascertainment $\alpha_t$: $0.4 + 0.2\sin(2\pi t / T)$ (incomplete observation rate)
- Day-of-week multiplier $w_{\mathrm{dow}(t)}$: $\{1, 1, 1, 1, 1, 0.5, 0.5\}$ for Mon–Sun
- Offspring dispersion $k = 1.0$ (within Lloyd-Smith respiratory-pathogen range)
- Initialisation: seed individuals placed in the pre-observation window $t \in [-\tau_{\max}, 0]$ with continuous timestamps. The seed-count *expected rate* per day follows the Euler–Lotka equilibrium profile $\lambda(d) = c \cdot 100 \cdot \exp(r_0 \cdot d)$ (with $r_0$ the equilibrium growth rate matching $R_t(1) = 0.8$ under the daily-discretised GI; $c = 0.3$ chosen so that the BP's realised rate at $t = 1$ matches the moment-closed expectation). The realised seed *counts* per day are Poisson draws around that expected rate, and the seed individuals' sub-day timestamps are uniform within their day. Each seed individual then produces offspring per the Lloyd-Smith mechanism, and those offspring produce their own offspring recursively through the pre-observation window before the observation window begins. Seed individuals themselves do not count as new obs-window infections; their realised descendants do. The BP is thus stochastic from seed onwards, but the expected seed rate is a deterministic anchor (replacing it with a longer pure-BP burn-in starting from a single ancestor far in the past is feasible but adds inter-replicate variance not relevant to the intervention-effect analysis).

**Multi-stream parameters (scenario 3):**

Shared $R_t$ and shared latent $I_t$; each stream has its own delay distribution, ascertainment trajectory, and dispersion.

| Stream | Delay (log-normal) | Ascertainment $\alpha_{\text{stream}}(t)$ |
|---|---|---|
| Cases | mean 5d, SD 2d | $0.40 + 0.20\sin(2\pi t / T)$ |
| Hospitalisations | mean 10d, SD 3d | $0.040 + 0.020\sin(2\pi t / T + \pi/3)$ |
| Deaths | mean 20d, SD 5d | $0.008 + 0.004\cos(2\pi t / (1.5 T))$ |

Ascertainment trajectories are phase-shifted or use a different period across streams so that the three series carry partially independent ascertainment signals rather than a single sinusoidal structure scaled three ways. Values are plausible for a moderately severe respiratory pathogen; the DGP is not labelled as a specific disease. Per-stream observations share the same Poisson measurement model and inherit infection-level overdispersion through $k$.

The simulation is not labelled as COVID-19 or any other specific disease. Data files, prompts, and metadata describe "an infectious disease outbreak" with no country or pathogen named. This prevents the LLM from leaning on disease-specific priors or memorised parameter values from training data.

#### Definition of $R_t$

In a stochastic data-generating process, "$R_t$" admits multiple legitimate definitions (Funk, Abbott & Bracher 2022). For a daily Poisson renewal model these include:

(a) the realised ratio $I_t / \Lambda_t$ where $\Lambda_t = \sum_s g_s I_{t-s}$, a noisy quantity that fluctuates around the rate due to Poisson sampling;
(b) a per-step random multiplier (does not arise in the model used here, which has no random R per time step);
(c) the parameter $R(d)$, the deterministic function plugged into the renewal equation as the rate's multiplier.

The recovery target is (c), the parameter $R(d)$. This is the quantity targeted by EpiEstim, EpiNow2, and renewal-equation Bayesian methods, the dominant approaches in the literature and the predicted default for LLM submissions. Methods that target (a), notably Wallinga–Teunis, recover a noisy version of $R(d)$ that converges to (c) at large counts but disagrees at finite counts. The disagreement reflects target choice rather than implementation error, and is flagged in expert review on the scenario 1a method-identification subsample.

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

Before running any LLM condition, the reference EpiAware implementations are applied to the canonical DGP across all 20 replicates and must satisfy:

- Median Rt RMSE on the evaluation window (days 25–125) < 0.10.
- Median 90% coverage on the same window ≥ 0.80 (i.e. not under-covering by more than 10pp).
- Median calibration |median(coverage) − 0.90| ≤ 0.10.

The criterion is **asymmetric**: under-coverage invalidates the reference (intervals too tight to contain the truth); over-coverage means the reference is conservatively calibrated (intervals wider than nominal). Conservative coverage is acceptable because the study's condition-level comparisons are *relative* coverage across conditions, not absolute coverage against the 90% nominal.

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

Each variant stresses one component that appears in the canonical DGP and in the reference specification (GI, delay, observation model with DoW, ascertainment, dispersion / likelihood, smoothness prior). The variants enumerate the modelling decisions that a correctly-specified renewal-equation model must handle; they are not hand-picked to match failures we expect to find. Any component without adversarial stress would be a gap in the evaluation.

The low-dispersion (Poisson-like) variant is included deliberately as a null condition: Poisson submissions should perform comparably to NegBin submissions here, but diverge specifically on `extreme_dispersion`. This controls for the possibility that "bad" submissions just fail everywhere (making the adversarial panel uninformative) versus failing in component-specific ways (making it diagnostic).

The short-GI perturbation changes both discretisation sensitivity (the intended stress) and epidemic dynamics (shorter GI gives a sharper rise and peak under the same $R_t$). The two effects cannot be cleanly separated within a single variant without departing from the "perturb one parameter" principle. Recovery on short_gi is therefore interpreted as a combined stress on discretisation handling and estimator robustness to faster dynamics. We note this when reporting the adversarial-panel results.

The `low_dispersion` and `extreme_dispersion` adversarial variants perturb the offspring dispersion $k$ (Lloyd-Smith parameter) at the infection level, varying the population-level NegBin mass-shape implied by individual-level offspring heterogeneity. Estimators that model NegBin observations (e.g. EpiNow2's `cluster_factor`) absorb this overdispersion through their observation likelihood. Estimators that assume Poisson observations (e.g. EpiEstim) will show undercoverage on `extreme_dispersion`. The `extreme_dispersion` $k = 0.1$ value is *beyond* the range observed in nature; it is a deliberate stress test rather than a realistic-pathogen scenario (Lloyd-Smith reported $k \approx 0.16$ for SARS).

Simulation-based evaluation addresses contamination because the DGP is canonical (LLMs have seen renewal-equation structure in training data) but the specific data does not match any training example. Grading on recovery against truth detects cases where the model recalls textbook machinery but implements it with missing components, because missing components cause bias in scenario-specific ways.

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

Reference implementations serve as a sanity check on the adversarial DGPs (they should recover truth) and as a benchmark for visualisation. They are not used as a grading target; grading is on recovery against simulation truth.

| Scenario | Reference implementation file |
|---|---|
| 1a, 1b | `reference_solutions/scenario_1b_epiaware.jl` (Renewal + AR(1) latent + NegBinomialError observation with reporting-delay convolution; used for both 1a and 1b since 1a's "open method" reduces to the renewal implementation) |
| 2 | `reference_solutions/scenario_2_epiaware.jl` (as above plus DoW effect and time-varying Ascertainment) |
| 3 | `reference_solutions/scenario_3_epiaware.jl` (as above plus `StackObservationModels` for the three streams sharing a single latent) |

An EpiNow2 reference (`reference_solutions/epinow2_baseline.R`) is applied to scenarios 1a, 1b, and 2 for comparison. EpiNow2 does not support multi-stream shared-latent estimation and is not applied to scenario 3.

## Evaluation

Outcomes are grouped into eight axes. Four are **confirmatory** — pre-registered
estimands with quantitative thresholds. Four are **descriptive** — reported per
condition without pre-registered thresholds, to give the reader the fuller picture
needed to evaluate scaffolding choices for their own use.

**Confirmatory axes:**

1. **Statistical correctness** — Rt RMSE against truth, credible-interval coverage,
   uncertainty calibration.
2. **Component correctness** — per-submission presence and correctness of specific
   modelling components (delay convolution, censoring, truncation, day-of-week,
   ascertainment, overdispersion).
3. **Interpretability** — non-comment, non-blank lines of code (automated) and expert
   readability rating on a stratified subsample.
4. **Reviewability** — time for a blinded expert to reconstruct and verify the model,
   plus reviewer confidence in the verification, on a stratified subsample.

**Descriptive axes:**

5. **Execution and engineering reliability** — retries, waits, wall time, token
   cost, convergence rate.
6. **Robustness** — variability across paraphrases and reruns, sensitivity to
   adversarial DGP variants.
7. **Maintainability** — static-analysis metrics: cyclomatic complexity, code
   duplication, hard-coded assumptions.
8. **Epistemic quality** — presence of diagnostic reporting (posterior predictive
   checks, R-hat, ESS), acknowledgement of assumptions, distinction between
   assumptions and estimands.

### Statistical correctness (confirmatory axis 1)

- **Rt RMSE** against true trajectory, averaged over the **evaluation window days 25–125**.
- **90% coverage** of credible/confidence intervals over the same window.
- **Calibration** of uncertainty (width of interval vs error magnitude) over the same window.

The evaluation window excludes the first 24 days (where the lookback into the seed window dominates and any reasonable estimator has insufficient information) and the last 25 days (where the longest reporting delay, 20 days for deaths plus the truncation buffer, means the latest observed cases reflect infections from outside the simulation horizon). $T = 150$ days; window = days 25–125 inclusive (101 day points).

Each metric computed per submission per DGP variant. Reported as the distribution of RMSE and coverage within each (scenario × condition) cell, across paraphrases × replicates × runs (canonical-DGP only for the primary distribution; full variant panel for adversarial fingerprint).

### Component correctness (confirmatory axis 2)

Per-submission binary flags for the presence/absence of the components each scenario
requires:

- Delay convolution present (all scenarios).
- Reporting-delay distribution handled with proper censoring / truncation (all
  scenarios).
- Day-of-week reporting effect modelled (scenarios 2 and 3).
- Time-varying ascertainment structure (scenarios 2 and 3).
- Overdispersed observation likelihood, not Poisson-only (scenarios 2 and 3).
- Multi-stream shared latent (scenario 3).

Automated detectors implement each flag (see Diagnostic: Automated structural-pattern
detectors below). Detectors are calibrated against blinded expert review on a
stratified subsample; per-detector Cohen's kappa is reported alongside the flag
rates.

### Interpretability (confirmatory axis 3)

- **LOC**: non-comment, non-blank lines of the final scripts that produce
  `outputs/rt_estimates.csv`, per submission. Automated; reported as distribution per
  condition.
- **Readability rating**: expert reviewers rate each stratified-subsample submission
  on a 1–5 readability scale (1 = incomprehensible, 5 = crisp) as part of the
  standard review pass. Inter-rater Cohen's kappa is reported.

### Reviewability (confirmatory axis 4)

Blinded expert reviewers, on the stratified subsample, record:

- **Time to verify**: minutes from starting the review to reaching a confident
  assessment of what the model does.
- **Reviewer confidence**: 1–5 scale of how confident the reviewer is in their
  reconstruction of the model.
- **Correct component identification**: for each modelling component (as listed
  under Component correctness), did the reviewer correctly identify whether the
  submission includes it?

Reviewability differs from interpretability: interpretability asks whether the code
reads as legible; reviewability asks whether an expert can efficiently verify that
the code does what it claims.

### Descriptive axes

Recorded per run and reported per condition without pre-registered thresholds:

- **Reliability**: `retry_count`, `post_agent_waits`, wall-clock duration, token
  cost, whether inference converged (from log inspection).
- **Robustness**: variance of the correctness metrics across the four paraphrases
  and five runs per cell; sensitivity of RMSE and coverage to adversarial variants.
- **Maintainability**: cyclomatic complexity, duplicated lines, count of magic
  numbers, count of hard-coded assumptions. Computed via a static-analysis pass on
  the final submitted scripts.
- **Epistemic quality**: automated flags for presence of posterior-predictive
  checks, R-hat / ESS diagnostics, sensitivity analyses; plus reviewer notes on
  whether the submission distinguishes assumptions from estimated quantities.

### Secondary: Hallucination and iteration behaviour

Recorded per run:

- **Hallucination rate**: number of iterations that failed due to "function X does not exist" / "no method matching" / undefined-symbol errors, as a fraction of total iterations.
- **Iterations to success**: number of iterations required to reach runnable code (or NA if not reached).
- **Error type distribution**: syntax, runtime, fitting failure, convergence failure.

Reported per condition. If hallucination rate is materially higher in `epiaware` despite the bundled API reference, or in `julia` despite the Turing.jl reference, that is a finding about in-context docs use, reported alongside but not conflated with the scaffolding effects on correctness.

### Diagnostic: Automated structural-pattern detectors

Static-analysis detectors flag mechanically detectable structural patterns in submitted code and outputs. Each flag identifies a pattern, not a "correctness departure"; the appropriate model for any given task is a judgment call that depends on the data-generating mechanism. Detectors are diagnostic instruments for analysis, not graders. Implemented in `evaluation/detectors.py`.

**Calibration protocol.** After all runs complete, the detector validator (see Roles & Responsibilities) reads a stratified sample of 60 submissions (5 per (scenario, condition) cell), applies each detector heuristically as an independent human read, and produces a per-detector confusion matrix against `evaluation/detectors.py` output. Cohen's kappa is computed per detector. Detectors with kappa < 0.6 are flagged as unreliable and their downstream analyses are marked "diagnostic only" in Tables 5 and 6. The confusion matrices and kappa values are included in the paper.

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

1. **Stratified subsample** (~20% of submissions, balanced across scenarios × conditions): used for inter-rater reliability and calibration of automated detectors against human judgment.
2. **Semantic departures** across all submissions: `si_not_gi`, `confused_rt_r`, `wrong_likelihood` (beyond Poisson/NegBin).
3. **Scenario 1a method identification**: a judgment call between renewal equation, Wallinga-Teunis, Bettencourt-Ribeiro, naive ratio, or other.

Expert review uses no LLM assistance, since the framing sensitivity of LLMs (including when used as judges) undermines the objectivity expert review is supposed to provide.

#### Departure classification (for the subsample and semantic departures)

- **A** Equivalent alternative: different but equally valid
- **B** Minor error: unlikely to substantially affect results
- **C** Major error: would bias results
- **D** Fundamental misunderstanding: lack of grasp of underlying epi/stats

Classification is cross-referenced with recovery: a "C" classification with good recovery (or vice versa) is a case worth discussion, not a contradiction.

## Randomisation

### Prompt paraphrases

For each (scenario, condition), k=4 paraphrases.

- **Slot 01 — Original.** The base prompt as drafted by the project authors with LLM coding-assistant support (the assistant produced the initial draft of each prompt's prose; the project authors reviewed, edited, and accepted).
- **Slot 02 — OpenAI** (GPT-5).
- **Slot 03 — Google** (Gemini 2.5 Flash).
- **Slot 04 — Anthropic** (Claude Sonnet 4.5; same family as the primary evaluator, included as a within-family sanity check).

The three LLM-paraphrase slots are deliberately allocated across model families so that paraphraser-side wording bias does not coincide with the evaluator's family. LLM paraphrases are generated by `evaluation/generate_paraphrases.py`, which calls each provider's API directly with the paraphrasing instruction and `temperature = 0.7` (where the API supports an explicit temperature). The paraphrasing call is API-mode (not Claude Code), so it does set temperature where supported; this is independent of the evaluator runs, which still use Claude Code at its default temperature (see "Temperature is not a randomisation axis" above).

All four paraphrase sources are LLM-mediated (one LLM-edited human draft plus three LLM-generated paraphrases). Human-register prose is not sampled; see Limitations.

Four paraphrases × 4 scenarios × 3 conditions = 48 prompt variants.

The choice of three LLM frontier families (OpenAI, Google, Anthropic) is not exhaustive: Mistral, Qwen, DeepSeek, xAI, Cohere, and others are omitted on cost grounds. We document the choice as a limitation.

The bundled API docs (`prompts/turing_api_docs.md`, `prompts/epiaware_api_docs.md`) are checked for content that constitutes a worked Rt example or otherwise violates the MWK operational rule. Two LLM instances (different families, no study context) read each section of both files against the rule and flag any violating content. The check is implemented in `evaluation/validate_api_docs.py` and the report is committed at `prompts/mwk_validation_report.md`. Any flagged content is removed before the pre-registration commit and the check is re-run.

### Runs

n=5 runs per (scenario, condition, paraphrase) for the primary recovery analysis on the canonical variant. n=3 runs per (scenario, condition, variant) for the adversarial-DGP fingerprint analysis (paraphrase 01 only). See "Sample size and crossing" below.

### Temperature is not a randomisation axis in this study

The Anthropic Claude Code CLI used to drive agent runs does not expose a `temperature` parameter at invocation time. Implementing a separate API-mode runner (re-implementing the agentic loop on top of the bare Messages API) would substantially expand scope. We therefore do *not* include temperature as a randomisation axis. Within-cell variability across runs of the same (scenario, condition, paraphrase) reflects only the model's intrinsic stochasticity at the API's default temperature plus any internal nondeterminism. Documented explicitly in Limitations.

### Sample size and crossing

These dimensions are crossed multiplicatively unless stated otherwise. The design is deliberately compact: the multiplicative crossing of paraphrase × replicate × run in earlier drafts overspecified the sampling and made the study prohibitively expensive. The reduced design below retains statistical power for every pre-specified prediction while keeping wall-clock and cost tractable.

**Primary recovery analysis** (Tables 1, 5, 6; Figures 1, 5, 6):
- Variant: canonical only.
- Replicate: 1 (seed 101) of the 20 generated replicates per cell.
- Paraphrase: 4 (slots 01, 02, 03, 04).
- Runs: 5 per (scenario, condition, paraphrase) cell.
- Total per (scenario, condition): 4 × 1 × 5 = 20 runs.
- Total per model across all (scenario, condition) cells: 12 × 20 = 240 runs.

**Adversarial-DGP fingerprint analysis** (Table 2, Figure 2):
- Variant: 5 (canonical + short_gi + long_delay + extreme_dispersion + abrupt_change), a subset of the nine defined variants below. Chosen to stress the four modelling dimensions most likely to differentiate submissions: discretisation (short_gi), delay handling (long_delay), likelihood / overdispersion (extreme_dispersion), and smoothing prior responsiveness (abrupt_change). The other four variants (strong_dow, high_asc_var, low_dispersion, sinusoidal_rt) remain defined for possible follow-up but are not part of the confirmatory analysis.
- Replicate: 1 (seed 101) per variant.
- Paraphrase: paraphrase 01 only.
- Runs: 3 per (scenario, condition, variant).
- Total per (scenario, condition): 5 × 1 × 3 = 15 runs.
- Total per model: 12 × 15 = 180 runs.

**Across three Claude models (Haiku 4.5, Sonnet 4.6, Opus 4.7):** 3 × (240 + 180) = **1260 runs** for the primary panel. Cost estimate at current API rates: ~$150 for Haiku, ~$650 for Sonnet, ~$1900 for Opus; total ≈ **$2700** for the primary panel, well inside the Anthropic research-credit budget.

The primary and adversarial analyses are independent: the runs do not overlap.

For Qwen3-Coder-30B (tertiary open-weight model), the primary panel is replicated with the same 240 runs; the adversarial panel is reduced to 3 variants (canonical, long_delay, extreme_dispersion) to keep total wall-clock under one week of shared single-GPU compute. Reduced sample size is reported explicitly with the tertiary results.

### Reporting

Every result reported as a distribution across paraphrases × replicates × runs, not a single pass rate. Primary figures show distributions (violin/box plots); point estimates as summaries only.

## Blinding

- **Expert review blinding.** Before review, submissions pass through a deterministic preprocessor (`evaluation/blind_submission.py`) that (a) strips `using`/`import`/`library()`/`from ... import` statements, (b) rewrites known package-namespaced calls (`EpiNow2::`, `EpiAware.`, `PyMC.`, `EpiEstim::` etc.) to a neutral placeholder, and (c) removes filename headers or comments naming a package or condition. Where full stripping is infeasible (e.g. scenario 3 multi-stream structure that only certain packages naturally express), the residual blinding is tested: reviewers guess the condition on a calibration subset of 24 submissions balanced across scenarios × conditions; the blinding-failure rate is reported.
- **Internal role separation.** Detector validation is performed by someone other than the detector implementer (see Roles & Responsibilities below). Other operational roles are concentrated in the project lead; this concentration is acknowledged in Limitations.

## Roles & Responsibilities

This is a small project. Most operational work falls to one person (project lead). Two roles require external (or at least independent) contributors: the expert review, and the detector validator.

| Role | Responsibility | Person |
|---|---|---|
| Project lead | Slot 01 prompts (drafted with LLM assistance), simulation generator, evaluation harness, detector implementation, review coordination, pre-registration commit | Sebastian Funk |
| Expert reviewer A | Reviews stratified subsample + semantic departures + scenario 1a method identification; blinded to LLM and condition | TBD (external preferred; project member acceptable) |
| Expert reviewer B | Independent second review for inter-rater reliability (Cohen's κ) and disagreement resolution | TBD (external preferred) |
| Detector validator | Reads a random sample of 50–100 submissions across cells and confirms that each detector flag (`flag_poisson_only`, `flag_no_smoothing_term`, `flag_no_delay_handling`, etc.) matches an independent human read of the submitted code. Output: per-detector confusion matrix and a list of misfires. | TBD (modeller or statistician who can read Julia/Python; not the project lead) |

If Cohen's κ between reviewers A and B shows substantial disagreement, a third reviewer is consulted. If recruitment of two reviewers fails, expert review is reduced to a single reviewer (no inter-rater stats) and this is reported as a substantive limitation rather than as a minor methodological note.

The "internal role separation" claim in the Blinding section reduces to: detector validation is independent of the project lead (who implemented the detectors). Other roles are concentrated in the project lead, which is honest for a small project and is acknowledged in Limitations.

### Conflict of interest declaration

One or more project authors contributed to EpiAware.jl, EpiNow2, and/or EpiEstim development. The full COI statement is in the paper.

Predictions 3, 5, 7, and 8 (missing-component rates, adversarial RMSE, LOC, and review time favouring `epiaware`) could be read as the authors validating their own work. The mitigations: pre-registering each prediction with a quantitative threshold, publishing the full harness for replication, committing to report non-confirmation honestly, and framing the study around observable scaffolding effects rather than package advocacy. Prediction 2 ("no-spec tool distribution") is affected too; we treat "what package do LLMs default to" as observational.

## Protocol

### Prompt construction

Standardised prompts per (scenario, condition) contain:
- Clear problem statement (epidemiological task)
- Data description and format (no disease or country label)
- Language/framework specification per condition table
- Epidemiological parameters that would plausibly come from external studies (see below)
- For `julia`: Turing.jl API reference in working directory
- For `epiaware`: EpiAware API reference in working directory (conforming to the MWK operational rule: no end-to-end Rt examples)

To isolate the modelling task from parameter-guessing, prompts give the LLM values that would realistically come from external epidemiological studies, and require the LLM to estimate everything else.

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

Base prompts (slot 01) live at `prompts/{scenario}/paraphrases/{condition}/01.md`. Each is a self-contained markdown file. LLM-paraphrased slots 02–04 (see Randomisation → Prompt paraphrases) live alongside as `02.md`, `03.md`, `04.md` in the same directory. The base prompts and paraphrases are pinned in the pre-registration commit and are not modified during the confirmatory analysis.

### Execution: agentic approach

Each LLM is given the prompt and asked to write code, execute it, and fix errors iteratively. This reflects realistic use of coding assistants.

**Protocol:**
1. The LLM writes code, executes it, and fixes errors iteratively.
2. Maximum 200 assistant turns per session (`--max-turns 200` for Claude Code; equivalent limit for the Qwen Code CLI).
3. Bash subprocess timeout is capped by the CLI at 10 minutes per call; longer-running inference is expected to be backgrounded by the agent and polled.
4. If the session ends without producing `outputs/rt_estimates.csv`, the harness checks whether any process is still running inside the sandbox and waits up to `POST_AGENT_WAIT_MIN=60` minutes for it to finish before consuming a retry. Up to `MAX_RETRIES=5` retries are made by resuming the same session with a continuation prompt. All retry sessions and their post-agent waits are logged in `metadata.json`.
5. All conversation logs, generated code, and outputs are preserved. The isolated `mktemp` working directory is deleted at session end; results are copied to `runs/{scenario}/{condition}/par_{p}/{variant}/rep_{r}/{model}/run_{n}/`.
6. Model cannot see reference solutions, other runs, truth files, or study design.

**Tools:**
- Claude Code (`claude --print`) for Claude models (Haiku 4.5, Sonnet 4.6, Opus 4.7). Implemented in `evaluation/run_agentic.sh`.
- Qwen Code CLI + vLLM on the LSHTM HPC (per the shared stack at `~/code/dotfiles/lshtm-local-llm-stack.md`) for Qwen3-Coder-30B. A parallel harness `evaluation/run_agentic_qwen.sh` wraps the same protocol.

**Recorded per run** (in `metadata.json` and the conversation logs):
- Final code and outputs (`outputs/rt_estimates.csv` if produced).
- Number of retries used and the number of post-agent waits.
- Duration (start/end timestamps).
- Whether the run succeeded (`output_present`).
- Iteration-level bash and error events are recoverable from the conversation JSONL.

### Expert review protocol

- Two independent infectious-disease modellers review the stratified subsample and semantic departures
- Reviewers blinded to LLM and condition (via stripping preprocessor)
- Each reviewer independently assesses each code sample
- Inter-rater reliability assessed (Cohen's kappa)
- Disagreements resolved by discussion; third reviewer consulted if needed
- **No LLM assistance permitted**

## Pre-specified Predictions

Stated here before any model is queried under this revised design. Each prediction
names a quantitative effect size we will treat as confirming it. "95% bootstrap CI"
refers to a non-parametric bootstrap over the unit-of-replication implied by the
comparison (paraphrase × run cells for the primary analysis, variant × run cells for
the adversarial fingerprint), with 1000 resamples. Implemented in R with `boot::boot`
(percentile method). The exact analysis code is committed at `evaluation/analyse.R`
and run once, after all agent runs are complete.

Predictions are grouped by confirmatory axis. Non-confirmation is informative and
reported as such.

### Instruction adherence and tool choice

1. **Instruction adherence.** In ≥95% of `julia` and `epiaware` submissions, the
   required language/framework is used. *Confirmation:* lower 95%-CI bounds ≥ 0.95.

2. **no-spec tool distribution.** In ≥70% of `no-spec` submissions on scenarios 1a
   and 1b, the produced code uses R + EpiEstim, R + EpiNow2, Python + PyMC, or
   Python + numpyro. The rate is not preregistered for scenarios 2 and 3 (where
   canonical multi-component packages don't fully cover the task) but the
   distribution of chosen tools is reported. *Confirmation for 1a/1b:* lower 95%-CI
   bound of the proportion ≥ 0.70.

### Component correctness (confirmatory axis 2)

3. **Missing-component rates differ across conditions.** For each scenario × required
   component pairing (day-of-week in scenarios 2/3; ascertainment in scenarios 2/3;
   overdispersion in scenarios 2/3; multi-stream latent in scenario 3), the rate of
   missing components is higher in `julia` than in `epiaware` by at least 15
   percentage points. *Confirmation:* 95%-CI on each difference excludes zero in the
   predicted direction. Reported jointly across the required components as a summary
   plus per-component.

### Statistical correctness (confirmatory axis 1)

4. **Recovery on canonical DGP.** For each scenario, the median Rt RMSE is reported
   per condition. No pre-registered threshold on the pairwise gap: pilots show that
   on canonical data the correctness metric does not discriminate cleanly among
   conditions at the Sonnet capability level, so the confirmatory claim is the
   *description* of the distributions rather than a specific inequality.

5. **Recovery on adversarial variants.** Median Rt RMSE on the four adversarial
   variants (short_gi, long_delay, extreme_dispersion, abrupt_change) is at least
   0.03 lower in `epiaware` than in `julia`, averaged across scenarios 2 and 3.
   *Confirmation:* 95%-CI on the difference excludes zero in the predicted
   direction.

6. **Detector flags predict adversarial degradation.**
   - Submissions flagged `flag_no_delay_handling`: median Rt RMSE on `long_delay` is
     at least 0.05 higher than on `canonical`. 95%-CI excludes zero.
   - Submissions flagged `flag_poisson_only`: median 90% coverage on
     `extreme_dispersion` is at least 15 percentage points lower than on `canonical`.
     95%-CI excludes zero.

### Interpretability (confirmatory axis 3)

7. **LOC.** Median LOC in `epiaware` submissions is at least 50 lower than in
   `julia` submissions, and at least 100 lower than in `no-spec` submissions, per
   scenario averaged. *Confirmation:* both 95%-CIs exclude zero in the predicted
   direction.

### Reviewability (confirmatory axis 4)

8. **Review time.** Blinded expert reviewers verify `epiaware` submissions at least
   25% faster than `julia` submissions on scenarios 2 and 3 (median minutes to
   confident assessment). *Confirmation:* 95%-CI on the ratio of medians excludes 1.

### Capability-conditional gap

9. **Scaffolding effect scales with model capability.** The `julia`-vs-`epiaware`
   RMSE gap on scenarios 2 and 3 is larger for smaller models: |Haiku 4.5| > |Sonnet
   4.6| ≥ |Opus 4.7|. Reported as a difference-in-differences with 95%-CI. Pilots
   show a strong signal in this direction on Haiku; the confirmatory test is
   whether Sonnet and Opus follow the pattern.

### Hallucination behaviour (orthogonal)

10. **Hallucination rate under bundled docs.** Median fraction of agent iterations
    failing with "function does not exist" / "no method matching" /
    undefined-symbol errors in `epiaware` is at least 10 percentage points higher
    than in either `julia` or `no-spec`. 95%-CIs exclude zero. This is a finding
    about in-context docs use, independent of the scaffolding effect on correctness.

## Pre-specified Tables and Figures

Organised by outcome axis. Confirmatory axes lead; descriptive axes follow.

### Tables

**Table 1: Statistical correctness by condition × scenario.** Rows: condition (3) × scenario (4) = 12 rows. Columns: median Rt RMSE, IQR of RMSE, median coverage, IQR of coverage, across all paraphrases × runs on the canonical DGP.

**Table 2: Statistical correctness on adversarial variants.** Rows: condition × scenario × DGP variant. Columns as Table 1. Shows condition-specific degradation under component stress.

**Table 3: Component correctness rates.** Rows: modelling component (delay, censoring, truncation, DoW, ascertainment, overdispersion, multi-stream). Columns: rate of correct presence by condition × scenario, from automated detectors calibrated against expert review.

**Table 4: Interpretability metrics.** Rows: condition × scenario. Columns: median LOC, IQR of LOC, mean readability rating (1–5), inter-rater kappa on readability.

**Table 5: Reviewability metrics.** Rows: condition × scenario. Columns: median minutes to confident assessment, mean reviewer confidence (1–5), rate of correct component identification.

**Table 6: Instruction adherence and tool distribution in no-spec.** For julia and epiaware: adherence rate. For no-spec, per model × scenario: distribution of chosen language (R / Python / Julia / other) and package.

**Table 7: Scenario 1a method identification.** Rows: condition × LLM. Columns: renewal / Wallinga-Teunis / Bettencourt-Ribeiro / naive / other. From expert review of this subsample.

**Table 8: Inter-rater reliability and detector validation.** Cohen's kappa between expert reviewers on each rating axis; agreement between detectors and reviewer classification on the stratified subsample.

**Table 9: Descriptive outcomes — reliability, robustness, maintainability, epistemic quality.** Rows: condition × scenario. Columns: median retries, median waits, median wall time, median token cost, RMSE variance across paraphrases, mean cyclomatic complexity, PPC-present rate, R-hat-report rate. Descriptive; no pre-registered thresholds.

### Figures

**Figure 1: Statistical correctness — recovery distributions.** Violin plots of Rt RMSE by condition, faceted by scenario. Full distribution over paraphrases × runs on canonical DGP.

**Figure 2: Adversarial DGP fingerprint.** Heatmap: rows = DGP variant, columns = condition × scenario. Cell colour = median RMSE relative to canonical.

**Figure 3: Rt trajectories.** Representative trajectories per condition × scenario, overlaid on true Rt. Ribbon for uncertainty.

**Figure 4: Component correctness heatmap.** Rows = component, columns = condition × scenario. Cell colour = correctness rate. Shows which components are missed in which conditions.

**Figure 5: LOC distribution.** Violin plots of LOC by condition, faceted by scenario.

**Figure 6: Capability-conditional gap.** For each model (Haiku, Sonnet, Opus), the julia-vs-epiaware median RMSE gap on scenarios 2 and 3. Tests whether scaffolding effects shrink with model capability.

**Figure 7: Hallucination and iteration behaviour.** Per condition: hallucination rate, iterations to first successful run, error-type distribution.

**Figure 8: Sensitivity to prompt paraphrase.** For a representative (scenario, condition) cell, full distribution of outcomes across paraphrases and runs.

## Limitations Acknowledged in Advance

The base prompts (slot 01) were drafted by the project authors with LLM coding-assistant support: the prose was generated by an LLM and reviewed/edited by the authors. The three paraphrase slots (02–04) are LLM-generated from three different frontier families (OpenAI, Google, Anthropic). All paraphrasing is therefore LLM-mediated; no human-register paraphrase is sampled. A hypothesis-blinded human paraphraser, fully external base-prompt design, and base prompts written without LLM assistance are all out of scope here. We identify them as the strongest improvements for replication studies.

The renewal equation is canonical in training data, so the DGP is also canonical at the structural level. Simulation with ground truth addresses *data* contamination but cannot address *structural* contamination. Adversarial DGPs that stress specific modelling decisions partially mitigate by separating "recalled machinery" from "correctly implemented machinery".

Model coverage is limited to one commercial family (Anthropic: Haiku 4.5, Sonnet 4.6, Opus 4.7) plus one open-weight coding-tuned model (Qwen3-Coder-30B) as a tertiary comparison. The commercial-frontier arm is single-family because the Anthropic research credits fund only Claude runs. Findings may not generalise to GPT-5, Gemini, Mistral, DeepSeek, or other frontier families; any effect specific to Claude's training or post-training would be indistinguishable from an effect of frontier LLMs in general.

The design uses 1 replicate per (scenario, condition) cell rather than crossing agent runs with data replicates. Run-level and data-level stochasticity are therefore pooled in the reported distributions rather than separated. This trades a lower total run count against a weaker claim about across-dataset generalisation. Replicates 102–120 are generated and available for follow-up analysis if the confirmatory result is ambiguous.

There is no independent replication. We publish the full harness and invite replication with a pre-specified concordance criterion (primary recovery claim replicated if point estimates within 10pp and same qualitative ordering of conditions).

Real-data secondary checks may reveal issues not captured in simulation.

No temperature randomisation: Claude Code CLI does not expose `temperature`, so it is not a randomisation axis in this study (see Randomisation, "Temperature is not a randomisation axis"). Within-cell variability across runs reflects only the model's intrinsic stochasticity at the API's default temperature.

Detectors are heuristics, not graders. Regex- and AST-based pattern detectors have known false negatives: `flag_no_smoothing_term` does not match a custom multivariate-Normal prior with smoothing covariance even though that constitutes smoothing. Detectors are calibrated against expert review on the stratified subsample (Cohen's kappa) and reported as instruments for analysis, not ground truth.

Scaffolding effects concentrate in scenarios 2 and 3. Scenarios 1a/1b are covered by canonical packages regardless of condition, so per-condition differences in produced code are expected to be smaller there. The confirmatory correctness and interpretability claims accordingly weight scenarios 2 and 3 more heavily.

Attribution of differences between conditions to a single mechanism is not possible. The intervention bundles language constraint, docs bundle, and available vocabulary; disentangling them would require a fully-crossed design that is out of scope here. This is stated plainly in the paper's methods.

Rt has multiple legitimate definitions under any stochastic generator: the parameter, the realised ratio, and in some models a per-step random multiplier (Funk, Abbott & Bracher 2022). Recovery is scored against the parameter $R(d)$. Methods that target the realised ratio (e.g. Wallinga–Teunis) recover a noisier quantity that converges to the parameter at scale. Observed disagreement with truth in their case partly reflects target choice rather than implementation error, and is flagged in the scenario 1a method-identification subsample.

The GT has a single mechanism for overdispersion: all of it arises from infection-level offspring heterogeneity (Lloyd-Smith et al. 2005). Other plausible sources (random reporting effort, batched-report processing artefacts, day-of-day administrative noise) are not modelled. Estimators that absorb such effects via NegBin observation likelihoods may handle real data better than they handle our GT, where the same parameter is doing different mechanistic work.

Ascertainment is purely temporal in the GT, modelled as a deterministic time-varying multiplier. In real surveillance, *which* individuals get reported depends on severity, age, healthcare access, and other individual covariates. Estimators correctly modelling individual-level ascertainment heterogeneity would have nothing extra to gain on this GT.

Multi-stream observation noise is independent across streams. The three streams (cases, hospitalisations, deaths) share the same latent infection process but their observation noise is independent. Real multi-stream surveillance has correlated observational error: a hospital-system disruption affects both hospitalisations and same-day reports. Multi-stream estimators that exploit cross-stream noise correlation would have nothing extra to gain.

Scenarios 1a and 1b may be functionally equivalent. 1a says "open method"; 1b says "use the renewal equation". Most submissions in 1a will use a method that internally implements the renewal equation (e.g. EpiNow2). The 1a/1b distinction tests what the LLM verbalises about its method, not what it computes. Useful for method-identification analyses but possibly not for recovery.

Reviewer blinding is imperfect. Package imports are strippable; structural features (multi-stream handling, Julia vs R syntax patterns) may leak condition information. Blinding failure rate is tested and reported.

## Pre-registration

This protocol is time-stamped by commit to the public git repository before any model is queried under the revised design. The commit hash fixing the plan will be cited in the paper. Phase 1 runs are not combined with confirmatory analysis; the confirmatory analysis uses simulated data generated under this plan.

## Discussion Points

### Why not framed as retrieval vs composition

An earlier version of this study framed the three conditions as a
retrieval-versus-composition gradient. That framing was retired because neither
mode is identifiable from LLM outputs: package use is not evidence of retrieval
(the agent may have configured and extended the package heavily), and
package-free code is not evidence of composition (it may reproduce a memorised
implementation). The current study restricts its claims to observable
interventions and observable outcomes.

### Minimum working knowledge and the docs question

Docs are provided to level API-level knowledge across conditions, not as a
treatment (see MWK principle under Conditions). A cleaner decomposition of "DSL
primitives" versus "worked examples" would require an epiaware condition with
Rt-specific tutorials, which leaks solutions and would make the API-familiarity
side of the intervention indistinguishable from the primitives side. The
compromise is that primitives-availability and API-knowledge move together
across the condition ladder. The hallucination-rate measurement partly addresses
this: if hallucination dominates the failure profile in `epiaware`, the
scaffolding results are confounded with in-context docs use; if hallucination is
low, the comparison with `julia` isolates the primitives contribution.

### Task-complexity gradient

Scenarios 1a/1b/2/3 form a gradient of task complexity. A scaffolding effect that
is flat across scenarios suggests a general docs / API-vocabulary effect; an
effect that grows with complexity suggests the composable primitives are earning
their keep on tasks that assemble more components.

### Capability-conditional effects

Pilots show that Sonnet 4.6 composes multi-stream renewal models from raw
Turing / numpyro when no scaffold is provided, while Haiku 4.5 does not. The
paper reports scaffolding effects both averaged across models and conditioned on
model capability. If scaffolding matters most for smaller models, the practical
implication is that packages like EpiAware raise the floor of LLM-assisted
modelling for less capable models more than they raise the ceiling for the most
capable ones.

## Ethical Considerations

- No human subjects
- All data publicly available or simulated
- LLM outputs reviewed before any public release

---

*Document created: 2024-12-07*
*Revised: 2026-04-23 (recovery-based evaluation, no-spec/Julia/EpiAware axis, adversarial DGPs, automated detectors, prompt randomisation, minimum working knowledge docs principle, hallucination rate as secondary outcome)*
*Revised: 2026-06 (main reviewer concerns addressed: explicit sample-size and crossing, temperature axis dropped, quantitative effect sizes for all predictions, reference EpiAware AR prior calibrated to HalfNormal(0.05), evaluation window days 25–125, sinusoidal_rt variant added, extreme_dispersion rename, three-family LLM paraphrase scheme, structural-pattern detector framing, slot-01 base prompts acknowledged as LLM-drafted)*
*Revised: 2026-07 (compact design: 5 runs × 4 paraphrases × 1 replicate on canonical + 3 runs on 4 non-canonical variants, ~420 runs per model; model panel switched to Anthropic-only Haiku 4.5 + Sonnet 4.6 + Opus 4.7 with Qwen3-Coder-30B as tertiary open-weight; execution protocol updated to match implemented harness (retries with `--resume`, wait-for-inference, cwd-based process detection); blinding preprocessor, detector calibration protocol, and reference-solution file paths pinned; bootstrap analysis tool specified as R `boot`)*
*Revised: 2026-08 (framing retired from retrieval-versus-composition to observable scaffolding effects; conditions described as interventions; outcomes reorganised into eight axes with four confirmatory (statistical correctness, component correctness, interpretability, reviewability) and four descriptive (execution reliability, robustness, maintainability, epistemic quality); predictions rewritten as intervention-effect claims with per-axis thresholds; capability-conditional gap added as prediction 9)*
*Status: Revised draft, pending pre-registration*
