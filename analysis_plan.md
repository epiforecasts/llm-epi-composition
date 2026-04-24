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
5. **Prompt-paraphrase and temperature randomisation**, with results reported as distributions.
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
| Claude Sonnet 4.6 | Commercial frontier | Current frontier capability |
| Llama 3.1 8B | Open-source | LMIC accessibility, reproducibility |

Llama 3.1 8B (rather than 70B) is chosen to demonstrate local inference on consumer hardware, relevant to LMIC resource constraints. Findings may not generalise to other model families.

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

Renewal equation dynamics:

**Infection process:**
$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

**Observation process:**
$$\mathbb{E}[C_t] = \alpha_t \sum_{d=0}^{D} I_{t-d} \cdot f_d$$

**Observation model:**
$$C_t \sim \text{NegBin}(\mu = \mathbb{E}[C_t], \phi)$$

**Parameters (canonical):**
- $T = 150$ days, nominal start date 2023-01-01
- $R_t$ trajectory: piecewise-linear, $R_t(1)=0.8$ → $R_t(50)=1.5$ → $R_t(100)=0.8$ → $R_t(150)=0.8$ (rise, fall, plateau)
- Generation interval: Gamma, mean 5.5 days, SD 2 days; truncated and discretised to PMF $g_s$ on $s = 1..20$ via double interval censoring
- Delay (infection → report): log-normal, mean 5 days, SD 2 days; discretised to PMF $f_d$ on $d = 0..20$ via double interval censoring
- Ascertainment $\alpha_t$: $0.4 + 0.2\sin(2\pi t / T)$
- Day-of-week multiplier $w_{\mathrm{dow}(t)}$: $\{1, 1, 1, 1, 1, 0.5, 0.5\}$ for Mon–Sun
- Dispersion $\phi = 10$
- Initialisation: $I_{-19}, \ldots, I_0 = 100$

**Multi-stream parameters (scenario 3):**

Shared $R_t$ and shared latent $I_t$; each stream has its own delay distribution, ascertainment trajectory, and dispersion.

| Stream | Delay (log-normal) | Ascertainment $\alpha_{\text{stream}}(t)$ | Dispersion $\phi$ |
|---|---|---|---|
| Cases | mean 5d, SD 2d | $0.40 + 0.20\sin(2\pi t / T)$ | 10 |
| Hospitalisations | mean 10d, SD 3d | $0.040 + 0.020\sin(2\pi t / T)$ | 10 |
| Deaths | mean 20d, SD 5d | $0.008 + 0.004\sin(2\pi t / T)$ | 20 |

Values are plausible for a moderately severe respiratory pathogen; the DGP is not labelled as a specific disease.

**Disease labelling.** The simulation is not labelled as COVID-19 or any other specific disease. Data files, prompts, and metadata describe "an infectious disease outbreak" with no country or pathogen named. This prevents the LLM from leaning on disease-specific priors or memorised parameter values from training data.

#### Generation procedure

For each DGP variant, observed data is generated by the following deterministic-infection, stochastic-observation procedure:

1. Set seed fixed per (variant, replicate).
2. Compute $I_t$ for $t = 1..T$ by the renewal equation $I_t = R_t \sum_{s=1}^{20} I_{t-s} g_s$ (deterministic).
3. Compute $\mathbb{E}[C_t] = \alpha_t \cdot w_{\mathrm{dow}(t)} \cdot \sum_{d=0}^{20} I_{t-d} f_d$.
4. Sample $C_t \sim \mathrm{NegBin}(\mu = \mathbb{E}[C_t], \phi)$.
5. Write `data/cases.csv` (columns: `date`, `cases`); `truth/true_rt.csv`; `truth/params.json`; `truth/sim_script.jl`.

For scenario 3, step 3 is repeated for each stream with stream-specific $\alpha$, delay PMF, and $\phi$; step 4 is applied per stream; `data/` receives `cases.csv`, `hospitalisations.csv`, `deaths.csv`.

**Choices fixed in the plan:**
- Infection dynamics deterministic; observation noise stochastic. (Stochastic infection dynamics would make "true $R_t$" ambiguous; deterministic infections keep the recovery target well-defined.)
- Three independent replicates per variant, seeds $\{101, 102, 103\}$. Recovery metrics computed per (submission, variant, replicate) and averaged.
- Dates are synthetic; no calendar features beyond day-of-week.

**Sanity check.** Before running any LLM condition, the reference implementations (EpiAware) are applied to the canonical DGP and must recover the true $R_t$ within tolerance (e.g. mean RMSE < 0.1, coverage within 10pp of nominal). If not, the simulation or the reference is wrong; fix before proceeding.

#### Adversarial DGPs

Each variant stresses a single modelling decision; a submission missing that component is expected to show scenario-specific bias.

Each variant differs from the canonical DGP in exactly one parameter:

| Variant | Perturbation | Stresses | Predicted failure if missing |
|---|---|---|---|
| Short GI | Gamma mean 2.5d, SD 1d | Discretisation | Bias if continuous density evaluated at integers |
| Long delay | Log-normal mean 10d, SD 3d | Delay handling | Rt estimate lagged/compressed near end |
| Strong DoW | Weekend multiplier 0.25 | Observation model | Oscillating Rt |
| High ascertainment variability | $\alpha_t = 0.4 + 0.35\sin(2\pi t / T)$ | Ascertainment model | Spurious Rt trend |
| Low dispersion | $\phi = 1000$ (near-Poisson) | Likelihood | No effect — null condition |
| High dispersion | $\phi = 2$ | Likelihood | Overconfident intervals if Poisson used |

**Rationale for DGP selection.** Each variant stresses one of the components that appears in the canonical DGP and in the reference specification (GI, delay, observation model with DoW, ascertainment, dispersion / likelihood). The adversarial DGPs are therefore not hand-picked to match failures we expect to find; they enumerate the modelling decisions that a correctly-specified renewal-equation model must handle. Any component *not* adversarially stressed would be a gap in the evaluation.

The low-dispersion (Poisson-like) variant is included deliberately as a null condition: Poisson submissions should perform comparably to NegBin submissions here, but diverge specifically on high dispersion. This controls for the possibility that "bad" submissions just fail everywhere (making the adversarial panel uninformative) versus failing in component-specific ways (making it diagnostic).

**Why simulation-based evaluation addresses contamination.** The DGP is canonical (LLMs have seen renewal-equation structure in training data) but the specific data does not match any training example. Grading on recovery against truth detects cases where the model recalls textbook machinery but implements it with missing components, because missing components cause bias in scenario-specific ways.

#### Isolation of simulation parameters from agent runs

Simulation parameters, the true Rt trajectory, and the data-generation script are stored separately from the observed data files and are never copied into an agentic run's working directory. Repository layout:

```
simulations/
  {variant}/            # canonical, short_gi, long_delay, ...
    truth/              # true_rt.csv, params.json, sim_script.jl — never exposed to agent
    data/               # cases.csv (+ hospitalisations.csv, deaths.csv for scenario 3) — only files copied to the agent
```

`run_agentic.sh` copies `simulations/{variant}/data/` (and the prompt + docs bundle per condition) into a `mktemp` working directory; nothing else from the repository is visible to the agent. Observed files are formatted identically to real-data files so the agent cannot distinguish simulated from real data by file structure. Evaluation (RMSE, coverage) runs outside the agent's sandbox after the run completes, using the truth files the agent never saw.

#### Discretisation and Censoring

The renewal equation operates on discrete time steps, requiring continuous GI and delay distributions to be discretised to PMFs. Proper discretisation should account for **double interval censoring**:

- **Primary censoring**: primary event (infection) occurs at an unknown time within its observation interval
- **Secondary censoring**: secondary event (onset, reporting) also occurs at an unknown time within its interval

Naive discretisation (evaluating PDF at integer points) does not account for this and can introduce bias, particularly for distributions with short means relative to the discretisation interval. The short-GI adversarial DGP specifically tests whether this bias manifests.

Acceptable approaches:
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

- **Rt RMSE** against true trajectory, averaged over time (excluding first/last 7 days to avoid edge effects)
- **Coverage** of 90% credible/confidence intervals (fraction of time points where truth is within interval)
- **Calibration** of uncertainty (width of interval vs error magnitude)

Each metric computed per submission per DGP variant. Primary result is the distribution of RMSE and coverage within each (scenario × condition) cell, across DGP variants × prompt paraphrases × temperatures × runs.

### Secondary: Hallucination and iteration behaviour

Recorded per run:

- **Hallucination rate**: number of iterations that failed due to "function X does not exist" / "no method matching" / undefined-symbol errors, as a fraction of total iterations.
- **Iterations to success**: number of iterations required to reach runnable code (or NA if not reached).
- **Error type distribution**: syntax, runtime, fitting failure, convergence failure.

Reported per condition. If hallucination rate is materially higher in `epiaware` despite the bundled API reference, or in `julia` despite the Turing.jl reference, that is a finding about in-context docs use independent of the composition result.

### Diagnostic: Automated correctness detectors

Static-analysis detectors for mechanically detectable departures. Developed and validated on a training subsample; applied uniformly.

| Departure | Detector approach | Feasibility |
|---|---|---|
| `poisson` vs `negbin` | AST/regex on likelihood specification | Clean |
| `no_smoothing` | Absence of AR/RW/GP/spline terms | Clean |
| `no_delay` | Absence of delay convolution or delay distribution | Mostly clean |
| `no_uncertainty` | Output check — intervals present | Clean |
| `no_discretisation` | Continuous density at integer points, no integration | Partial |
| `negative_rt` | Posterior check — any negative Rt samples | Clean |
| `wrong_likelihood` | Partial — catches some patterns | Partial |
| `confused_rt_r` | Semantic | Hard |

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

For each (scenario, condition), k=5 paraphrases generated by:

1. Manual rewrite by an author blinded to hypothesis direction (not the original prompt author)
2. Paraphrasing via a separate LLM instance, blinded to study design

Five paraphrases × 4 scenarios × 3 conditions = 60 prompt variants.

### Temperature

Four temperatures: {0.0, 0.3, 0.7, 1.0}. Results reported as distributions across temperatures.

### Runs

n=10 per (scenario, condition, paraphrase, temperature) for Sonnet. For Llama 3.1 8B, reduce as needed given compute budget; prioritise paraphrase and temperature coverage over within-cell replication.

### Reporting

Every result reported as a distribution across wordings × temperatures × runs, not a single pass rate. Primary figures show distributions (violin/box plots); point estimates as summaries only.

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

Stated here before any model is queried under this revised design.

1. **no-spec defaults to packages.** In >70% of scenario 1a/1b submissions, no-spec condition produces R + EpiEstim/EpiNow2 or Python + PyMC, regardless of model.
2. **Recovery is comparable across conditions on scenarios 1a/1b** (package shortcut suffices).
3. **EpiAware shows lower Rt RMSE than Julia-bare specifically on scenarios 2–3**, with the gap widening with complexity.
4. **no-spec fails on scenario 3 more often than EpiAware** — EpiNow2 multi-stream limitation forces recovery or switch, and the switch often fails.
5. **Adversarial DGP performance correlates with automated correctness detectors** — submissions flagged `no_delay` show worse recovery on long-delay DGP specifically; submissions flagged `poisson` show worse coverage on high-dispersion DGP specifically.
6. **Hallucination rate is higher in `epiaware` than in `julia` or `no-spec`** — even with the bundled API reference, EpiAware's low training-data representation is not fully compensated by in-context docs.

Predictions 3–5 are the load-bearing composition claims. If they do not hold, the study reports that validated composable tooling does not provide a composition benefit over forced-composition baselines, which is itself informative. Prediction 6 is orthogonal: a finding about in-context docs use rather than composition per se.

## Pre-specified Tables and Figures

### Tables

**Table 1: Recovery by condition × scenario.** Rows: condition (3) × scenario (4) = 12 rows. Columns: median Rt RMSE, IQR of RMSE, median coverage, IQR of coverage, across all paraphrases × temperatures × runs × canonical DGP.

**Table 2: Recovery on adversarial DGPs.** Rows: condition × scenario × DGP variant. Columns as Table 1. Shows scenario-specific bias patterns.

**Table 3: Language and package selection in no-spec.** Rows: LLM × scenario. Columns: distribution of language choice (R / Python / Julia / other), distribution of package choice.

**Table 4: Method selection in scenario 1a.** Rows: condition × LLM. Columns: renewal / Wallinga-Teunis / Bettencourt-Ribeiro / naive / other. From expert review of this subsample.

**Table 5: Automated detector rates.** Rows: departure category. Columns: rate by condition × scenario.

**Table 6: Inter-rater reliability and detector validation.** Cohen's kappa between reviewers; agreement between detectors and reviewer classification on the stratified subsample.

### Figures

**Figure 1: Primary result — recovery distributions.** Violin plots of Rt RMSE by condition, faceted by scenario. Shows full distribution over paraphrases × temperatures × runs on canonical DGP.

**Figure 2: Adversarial DGP fingerprint.** Heatmap: rows = DGP variant, columns = condition × scenario. Cell colour = median RMSE relative to canonical. Reveals which conditions fail on which stress tests.

**Figure 3: Rt trajectories.** Representative trajectories per condition × scenario, overlaid on true Rt. Ribbon for uncertainty.

**Figure 4: Correctness ↔ recovery linkage.** Scatter: each point a submission. X = number of automated detector flags; Y = Rt RMSE. Annotated with predicted failure modes for each flag.

**Figure 5: Hallucination and iteration behaviour.** Per condition: hallucination rate (errors from non-existent functions / method matching), iterations to first successful run, error-type distribution. Tests whether bundled API docs equalise working knowledge across conditions.

**Figure 6: Sensitivity to prompt paraphrase and temperature.** For a representative (scenario, condition) cell, full distribution of outcomes across wordings and temperatures. Demonstrates the within-cell variability Omar et al. highlight.

## Limitations Acknowledged in Advance

- **Author-designed prompts.** External prompt design remains the recommended fix but is out of scope. We partially mitigate via paraphrase randomisation (one wave manual, one wave LLM-generated, both blinded to hypothesis direction) and internal role separation.
- **Training-data contamination at DGP level.** The renewal equation is canonical in training data. Simulation with ground truth addresses *data* contamination but cannot address *structural* contamination. Adversarial DGPs that stress specific modelling decisions partially mitigate by separating "recalled machinery" from "correctly implemented machinery".
- **Model coverage.** Two model families (Claude, Llama). Findings may not generalise.
- **No independent replication.** We publish the full harness and invite replication with a pre-specified concordance criterion (e.g., primary recovery claim replicated if point estimates within 10pp and same qualitative ordering).
- **Simulation realism.** Real-data secondary check may reveal issues not captured in simulation.
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
