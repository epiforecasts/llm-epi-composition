# Analysis plan: LLM-generated epidemic model code

**Status:** revised draft, not yet registered. Development runs have informed the
study design. Confirmatory runs must use a frozen protocol and data that have not
been used to choose prompts, reference priors, outcomes or thresholds.

## Research question

**Primary question.** How do unrestricted tool choice, Julia with Turing, and
Julia with ComposableTuringIDModels.jl compare in statistical performance,
implementation errors and execution success when LLM coding
agents generate epidemic model code?

**Secondary questions.**

- What tools do LLMs choose when unconstrained, and how does this vary across
  models and scenarios?
- Which modelling components (delay handling, censoring, weekly reporting,
  ascertainment structure, observation likelihood, multi-stream shared latent)
  are correctly implemented under each condition?
- How do differences between software conditions vary across the three commercial
  Claude deployments (Haiku 4.5, Sonnet 4.6, Opus 4.7)?

The conditions are observable software interventions: task instruction, required
language or framework, and supplied API documentation. The outcomes are observable
properties of the code and estimates the agent produces. Whether an agent recalled
a solution or assembled one from primitives cannot be identified from its outputs.
Package use is not evidence of recall, and package-free code is not evidence of
assembly from primitives. Differences between conditions are reported as effects
of the intervention on the produced code.

## Study history and registration

Phase 1 (December 2025–April 2026) compared R, Python, Julia and EpiAware. Some
scenario 1a runs were completed; the planned analysis was not. Its tracked
materials remain in git history, but the gitignored run outputs were not retained.
No phase 1 results enter the confirmatory analysis.

Phase 2 development included simulated datasets, reference calibration, prompt
revision and agent/harness pilots. These activities preceded registration and
must be reported as such. Analyses informed by development results are reported
as retrospective. Before registration, publish an
inventory of development datasets and runs, freeze the protocol and analysis
code, and specify the unused evaluation seeds. Record subsequent amendments with
their dates and whether outcomes had been inspected.

## Scientific requirements and implementation status

The current implementation is suitable for development. The following requirements
must be met before recovery results can support a confirmatory comparison:

1. **Align the reproduction-number target.** The development generator assigns
   expected offspring at the parent's infection time, so its saved trajectory is
   a cohort reproduction number. Standard renewal estimators use a multiplier at
   the time of onward transmission. The study target is the instantaneous
   reproduction number; the generator, truth, prompts and scoring must all use it.
2. **Make ascertainment identifiable in the primary recovery task.** With unknown
   time-varying ascertainment, case counts alone cannot generally distinguish
   changing transmission from changing observation. Use known ascertainment or
   external information for primary recovery; retain unknown ascertainment as a
   separate sensitivity task about assumptions and their consequences.
3. **Separate process and observation variation.** The current observation model
   is conditionally Poisson. A negative-binomial observation likelihood is not a
   necessary condition for scientific correctness under this generator.
4. **Freeze an evaluation sample independent of development.** The existing
   one-dataset design estimates variation between agent runs conditional on that
   dataset. Claims about performance across outbreaks require independent data
   replicates, with an analysis that retains their grouping.
5. **Validate automated implementation checks.** Passing static detectors does
   not certify code as correct. Detector flags support per-component error
   estimates only to the extent their accuracy is measured against human review.

The target, ascertainment information and interpretation of variation must be
resolved before recovery can be scored. Independent data replicates are required
only for claims across outbreaks; the retained allocation below makes a narrower,
dataset-conditional claim. These scientific requirements are distinct from recruitment and
implementation tasks, listed under Work required before registration and
execution. The generator, prompts and analysis code have not yet been brought
into line with this specification.

## Conditions and models

| Condition | Requirements | Supplied documentation |
|---|---|---|
| `no-spec` | Agent chooses language, packages and method | No condition-specific API bundle |
| `julia` | Julia; Turing for probabilistic modelling | Turing API reference |
| `composable` | Julia and ComposableTuringIDModels.jl | Shared Turing API reference plus ComposableTuringIDModels API reference |

The domain-specific condition uses
[ComposableTuringIDModels.jl](https://composableturingidmodels.epiaware.org/dev/),
which assembles infection and observation components into Turing models through
`as_turing_model`. Its API differs from EpiAware, which was used during
development; migration requires adapting and validating models. Pin its source
commit, compatible Turing environment and documentation snapshot. The linked `/dev/` site is a documentation source,
not a reproducible version identifier.

Before registration, identify a candidate commit and record its API changes,
release notes and relevant unresolved issues. Confirm the expected timing of
required API changes with the maintainer and record the response when available;
no release date or stability commitment is assumed here. Accept the candidate
only after a clean environment instantiates, all study reference models run,
and inference/output checks pass against that exact
commit. A tagged release is useful but not required. If no candidate passes,
delay registration of the affected comparison. Once frozen, later upstream
releases do not trigger upgrades; a necessary fix requires a documented protocol
amendment and a decision about affected runs before inspecting new outcomes.

Use `composable` as the new condition identifier. Existing `epiaware` directories,
run records and code refer to the earlier development condition and retain their
historical identity. New runs must not be pooled with them by renaming the label.
Both Julia conditions receive the same Turing reference and use the same pinned
Turing version; the composable condition additionally receives the domain API
bundle. Complete tutorials from the website are excluded from agent materials.

The conditions combine several changes at once. Each estimates the effect of the
combination and cannot separate the contribution of language, documentation,
package design or prior API familiarity.

The three conditions differ in the kind of software the agent can use. Under
`no-spec` an agent may reach for a task-specific reproduction-number package
such as EpiEstim or EpiNow2, a general probabilistic programming framework such
as PyMC or numpyro, or write from lower-level primitives. `julia` supplies a
general probabilistic programming framework with no epidemic-specific vocabulary.
`composable` supplies infectious-disease modelling components, including renewal
processes, delays and observation models. Package identity does not certify
correctness: EpiEstim does not correct for reporting delays, EpiNow2 does not
natively support shared-latent multi-stream models, and general probabilistic
programming frameworks require the user to specify the whole model. Whether a
submission implements the task correctly is assessed from its code and estimates.

A Julia condition with optional Turing, or one that excludes Turing, would address
framework choice within a fixed language. Either is a possible follow-up with its
own question and pre-specified analysis. Crossing a fourth condition with the
current allocation would raise the commercial panel from 1260 to 1680 runs. The
`no-spec` observations alone cannot estimate its effect.

Documentation is part of the intervention. Training exposure and usable API
knowledge cannot be equalised by assumption. Record the exact document versions,
lengths and contents. Bundles may contain signatures, argument descriptions and
short examples of individual functions, but no complete solution to a study
scenario. Human review determines compliance; `evaluation/validate_api_docs.py`
can assist this audit.

The authors develop ComposableTuringIDModels, so the domain bundle may be better
written than the Turing bundle for reasons unrelated to package design. Before
freezing, an auditor who did not write either bundle records, for each, its
length, the number of function entries and short examples, and the coverage of
functions needed for each scenario. Report these counts with the results.

At freeze time, a human audit of the shared Turing reference must explicitly
exclude renewal equations, reproduction-number estimation, epidemic observation
models and worked solutions to the study scenarios. Generic probability and
Turing syntax examples are allowed. Check both prose and code, including examples
copied from upstream documentation; a keyword search alone is insufficient.
Record the audited file hash, reviewer and date, and verify that both Julia
conditions receive identical bytes. Repeat the audit after any document change.
API compatibility with the pinned Turing version is a separate validation.

Record the language, packages and method selected in `no-spec`. These are
outcomes of the condition, not randomised subgroups. Comparisons between selected
packages are descriptive.

The planned commercial panel is Claude Haiku 4.5, Sonnet 4.6 and Opus 4.7. These
are three deployments within one family, not a controlled experiment in model
size or reasoning depth. Pin the exact available model identifiers, CLI versions,
settings and run dates at registration. Qwen3-Coder-30B-A3B-Instruct is a secondary
open-weight comparison using Qwen Code and vLLM; differences also include the
agent harness. The study does not estimate performance across model families.

## Tasks

| Scenario | Task |
|---|---|
| 1a | Estimate the reproduction number from cases, with open method choice |
| 1b | Estimate it from cases using a renewal model |
| 2 | Include weekly reporting variation, ascertainment and extra variation |
| 3 | Jointly model cases, hospitalisations and deaths through shared infections |

Scenarios 1a and 1b compare instructions; the chosen methods may be identical.
Scenarios 2 and 3 add modelling requirements. They do not define a calibrated
scale of difficulty. Package coverage, including for multi-stream models, is
recorded from submissions.

All conditions within a scenario receive the same scientific information and
output requirements. GI and reporting-delay distributions are supplied as known
inputs. Recovery is conditional on those inputs being correct; uncertainty in
external epidemiological estimates is outside the primary comparison.

The revised primary tasks must supply ascertainment trajectories or independent
information that constrains them. The existing unknown-ascertainment prompts
remain development materials until that change is made. A separate sensitivity
task may withhold this information to assess how agents describe and handle the
resulting ambiguity. Component presence alone is not evidence that ascertainment
has been estimated from the data.

## Simulation and estimands

### Development infection process

`simulations/generate.jl` implements an age-dependent branching process. For an
individual infected at continuous time $t_i$:

$$\nu_i\sim\mathrm{Gamma}(k, R_c(t_i)/k),\qquad
Z_i\sim\mathrm{Poisson}(\nu_i),\qquad
\tau_{ij}\overset{\mathrm{iid}}\sim g.$$

Here Gamma uses shape and scale. Offspring arrive at $t_i+\tau_{ij}$, so the
expected lifetime number of offspring is $R_c(t_i)$. The Gamma–Poisson mixture
represents individual transmission heterogeneity, following
[Lloyd-Smith et al. (2005)](https://www.nature.com/articles/nature04153).
Offspring occur at separate ages.

Ignoring the initial seed contribution, its mean incidence satisfies

$$m(t)=\int_0^\infty R_c(t-a)m(t-a)g(a)\,da.$$

An instantaneous renewal model instead uses

$$m(t)=R_{\mathrm{inst}}(t)\int_0^\infty m(t-a)g(a)\,da.$$

The location of the time-varying multiplier matters. Instantaneous and cohort
reproduction numbers can differ systematically during changing transmission,
even at high incidence, and Wallinga–Teunis and Cori-type methods target
different quantities. See [Cori et al. (2013)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3816335/)
and [Gostic et al. (2020)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008409).

**Target:** instantaneous $R_{\mathrm{inst}}(d)$. The confirmatory generator is a
daily discrete renewal process. With infectiousness
$\Lambda_d=\sum_{s\ge1}g_sI_{d-s}$ and individual offspring dispersion $k$,

$$I_d\mid I_{<d}\sim\mathrm{NegBin}\bigl(\text{mean }R_{\mathrm{inst}}(d)\Lambda_d,\
\text{size }k\Lambda_d\bigr),$$

which approximates the total offspring of $\Lambda_d$ effective infectors each drawing a
Gamma–Poisson offspring number. $g_s$ is the discretised generation interval.
Validate this generator, including its initialisation, before registration, then
regenerate truth and observed data. The individual-level branching process in
`simulations/generate.jl` remains a development implementation; it may be used as
a sensitivity analysis only with the cohort reproduction number as its target. Do not
score an instantaneous estimate against the existing `true_rt.csv` and label the
difference an implementation error. The ratio of realised infections to realised
infectiousness is a noisy quantity distinct from the simulation parameter.

### Current observation process

For stream $s$, let $I_d$ be the realised number of infections on day $d$:

$$\mu_{s,d}=\alpha_s(d)w_{\mathrm{dow}(d)}
  \sum_{e=0}^{D_{\max}} f_{s,e}I_{d-e},\qquad
C_{s,d}\mid I\sim\mathrm{Poisson}(\mu_{s,d}).$$

This is a Poisson measurement model conditional on realised infections.
It is not binomial ascertainment of a fixed set of infected individuals. Counts
can exceed infections and streams are conditionally independent. Marginal
variation from stochastic infections must be distinguished from additional
measurement variation. A Poisson observation model with a suitable latent
process and a negative-binomial approximation can both be appropriate.

Here ascertainment multiplies reports on the **report date**. An infection-date
probability would instead appear inside the convolution. The prompts and
references must use the intended convention. Hospitalisation and death rates are
stream-specific observation probabilities/rates in this simplified model, not a
mechanistic model of linked clinical outcomes.

The observation mean identifies the product of ascertainment and delayed
infections. Allowing both to vary freely creates confounding. Extra streams with
independent unknown ascertainment curves do not automatically resolve it. For
unknown-ascertainment sensitivity tasks, inspect prior sensitivity, posterior
correlations and infection/ascertainment recovery against truth; do not interpret
a smooth fitted curve as evidence of identification.

### Development parameters

| Quantity | Current value |
|---|---|
| Observation period | 150 days, nominal start 2023-01-01 |
| Cohort reproduction trajectory | Linear between days 1, 50, 100, 150 with values 0.8, 1.5, 0.8, 0.8 |
| Generation interval | Gamma, mean 5.5 days, SD 2 days |
| Individual offspring dispersion | $k=1$ |
| Cases: delay; report-date multiplier | Log-normal, mean 5 days, SD 2 days; $0.4+0.2\sin(2\pi t/150)$ |
| Hospitalisations: delay; multiplier | Log-normal, mean 10 days, SD 3 days; $0.04+0.02\sin(2\pi t/150+\pi/3)$ |
| Deaths: delay; multiplier | Log-normal, mean 20 days, SD 5 days; $0.008+0.004\cos(2\pi t/225)$ |
| Weekly multiplier | 1 Monday–Friday, 0.5 Saturday–Sunday |
| Delay support | Days 0–30, renormalised |
| Available development seeds | 101–120 per variant |

The weekly multipliers average $6/7$, not one. Thus the listed ascertainment
curves are weekday rates, not weekly averages. A normalised parameterisation
requires compensating the ascertainment scale.

The current generator seeds days −19 through 0 with Poisson counts at integrated
rate $0.3\times100\exp(r_0t)$, then propagates their descendants. The factor 0.3 is
an empirical development choice. Continuous GI draws are not truncated at 20 days;
20 days sets the seed window and the GI approximation used for initialisation.

Before freezing a replacement generator, check the initial growth equation and
day indexing, provide enough prehistory for all delay supports, and test a longer
initialisation window. In the current code, seed individuals themselves are
omitted from realised infections used by the observation convolution. This
boundary convention needs explicit validation. Report extinction and low-count
replicates; never silently regenerate them until recovery improves.

### Daily aggregation and delay handling

The current delay PMF integrates over a uniform infection time within a day:

$$P(D=d)=\int_0^1\{F(d+1-p)-F(\max(d-p,0))\}\,dp,$$

then renormalises over the finite support. This assumes uniform timing within
a day, an approximation whose accuracy depends on the epidemic growth rate. Daily
aggregation of continuous transmission also introduces an approximation.

Distinguish integration over observation intervals, finite-support truncation of
a delay distribution, and missing future reports near the observation boundary.
These require different checks. Short-delay sensitivity may reveal discretisation
errors; any correct implementation is accepted.

### Simulation variants

| Variant | Development perturbation | Intended assessment |
|---|---|---|
| `short_gi` | GI mean 2.5 days, SD 1 day | Daily discretisation and faster dynamics |
| `long_delay` | Case delay mean 10 days, SD 3 days | Delay handling |
| `strong_dow` | Weekend multiplier 0.25 | Weekly reporting variation |
| `high_asc_var` | Case multiplier $0.4+0.35\sin(2\pi t/150)$ | Sensitivity to ascertainment information and assumptions |
| `low_dispersion` | $k=1000$ | Weak individual transmission heterogeneity |
| `extreme_dispersion` | $k=0.1$ | Strong individual transmission heterogeneity |
| `abrupt_change` | Reproduction trajectory falls 1.5 to 0.5 over days 74–77 | Response to rapid change |
| `sinusoidal_rt` | $1+0.4\sin(2\pi t/60)$ | Repeated changes; 2.5 cycles over 150 days |

The trajectories in this table and in Development parameters are cohort
reproduction numbers under the current generator. Re-express them as
instantaneous trajectories when the replacement generator is frozen.

These alter more than estimator difficulty: GI changes affect incidence, and
stronger heterogeneity affects extinction and information content. They are
sensitivity analyses and do not isolate why a component failed. The value
$k=0.1$ is a strong-heterogeneity setting. Infection-level $k$ does not identify
an observation-level dispersion parameter.

The compact panel includes canonical, short GI, long delay, extreme dispersion
and abrupt change. It cannot test the proposed low-dispersion comparison without
adding that variant. Record any panel change before registration.

### Reference implementations and validation

Existing development references are
`reference_solutions/scenario_1b_epiaware.jl` (used for 1a and 1b),
`scenario_2_epiaware.jl`, `scenario_3_epiaware.jl`, and
`reference_solutions/epinow2_baseline.R` for scenarios 1a, 1b and 2. The three
EpiAware references must be ported to ComposableTuringIDModels and revalidated;
legacy calibration results do not establish performance of the replacement.

References check feasibility and provide benchmarks. They are not a grading
standard, and using ComposableTuringIDModels references alone risks favouring
assumptions that suit that framework. Validate the corrected target with an independently
implemented simple reference, including a multi-stream check. Apply the same
scientific specification, input information and computational budget where a
comparison is intended.

Development calibration under the cohort target gave median RMSE 0.086 and
median temporal coverage 1.00 across 20 canonical replicates with a
`HalfNormal(0.05)` innovation prior. These results do not transfer to the
instantaneous target. Freeze priors using development data and evaluate them
on independent datasets. Inspect interval width, WIS, convergence and prior
sensitivity alongside recovery. Before evaluating references on the new
generator, record pass/fail thresholds for median WIS and median 90% coverage
on the scoring window; a reference that fails them blocks registration.

## Sampling, randomisation and execution

### Compact allocation

The existing budgeted allocation is retained here so its limits are explicit:

| Panel | Allocation per model | Runs |
|---|---|---|
| Canonical | 4 scenarios × 3 conditions × 4 paraphrases × 1 dataset × 5 reruns | 240 |
| Sensitivity | 4 scenarios × 3 conditions × 5 variants × 1 dataset × 3 reruns; paraphrase 01 | 180 |
| Three commercial models | 3 × (240 + 180) | 1260 |

Canonical runs in the two panels are distinct agent runs on the same dataset;
the panels are not independent replications of epidemic variability. The Qwen
allocation is 240 canonical runs plus 108 runs across canonical, long delay and
extreme dispersion. Runtime and cost estimates must be updated from recorded
usage before launch; development pilots gave a commercial estimate of about US$2700.

One dataset per variant supports a conditional comparison for those datasets.
Agent reruns do not increase the number of simulated outbreaks, and data-level
variation is absent from these run distributions.
The study retains this allocation and makes dataset-conditional claims. Use the
same frozen dataset for every condition, model and paraphrase within a variant.
Select unused evaluation seeds after auditing development use; seed 101 was used
in development and is excluded.

An optional extension would replace the five canonical reruns with five
independent datasets shared across conditions and paraphrases, at the same run
count. This would change the estimand and require a protocol amendment before
execution.
A precision assessment describes expected interval widths and the chance of
resolving the hypotheses under plausible effects. Inconclusive results are
retained if the fixed budget is insufficient.

### Prompts and scheduling

Each prompt contains the epidemiological task, a description of the data files,
the language or framework requirement for its condition, and the supplied inputs:
GI and delay distributions (family, mean and SD) and, in the primary tasks,
ascertainment information. Prompts do not specify an inference method, disclose
the simulation model, or give values of quantities the agent must estimate. Data
files, prompts and metadata describe an infectious disease outbreak without
naming a pathogen or country, and observed files follow the format of the real
surveillance data. Every run must write `outputs/rt_estimates.csv` with one row
per day and columns `date` (YYYY-MM-DD) and `q0.025`, `q0.05`, `q0.1`, `q0.25`,
`q0.5`, `q0.75`, `q0.9`, `q0.95`, `q0.975`, the predictive quantiles of the
reproduction number. Methods without uncertainty quantification report the
point estimate in every quantile column.

There are four prompt versions per scenario and condition, stored at
`prompts/{scenario}/paraphrases/{condition}/NN.md`. Slot 01 was drafted with LLM
assistance and edited by the authors. Slots 02–04 are paraphrases generated by
GPT-5, Gemini 2.5 Flash and Claude Sonnet 4.5 using
`evaluation/generate_paraphrases.py`. These are four selected wordings, not a
random sample of human prompting behaviour. Human checking must confirm that
paraphrases preserve the estimand, information and requirements.

Freeze the 48 canonical prompts, their variant instantiations and the shared
Turing and domain-specific API bundles. Variant prompts differ from canonical
prompts only in the supplied GI, delay and ascertainment values; generate them
from the frozen text by substituting those values, and verify that they match
the variant. Randomise execution order in blocks across
conditions and models to reduce confounding with service changes or machine load.
Record the allocation seed and preserve failed attempts. Claude Code runs use
its default sampling settings; there is no temperature comparison.

### Agent protocol

`evaluation/run_study.sh` allocates runs; `evaluation/run_agentic.sh` runs the
Claude agent. `evaluation/run_agentic_qwen.sh` provides the Qwen implementation.
The agent writes code, executes it and attempts repairs:

1. Maximum 200 assistant turns per session (`claude --print --max-turns 200`;
   the equivalent limit in Qwen Code).
2. Up to five resumed retries (`MAX_RETRIES=5`).
3. If output is absent but inference continues, wait up to 60 minutes
   (`POST_AGENT_WAIT_MIN=60`) before consuming a retry.
4. Save conversations, code, outputs, retry counts, waits, start/end times and
   token usage to
   `runs/{scenario}/{condition}/par_{p}/{variant}/rep_{r}/{model}/run_{n}/`, with
   run metadata in `metadata.json`. Preserve run and model identifiers.

Retain these turn, retry and waiting limits for the main comparison. They define
success under this agent workflow, rather than at equal elapsed time or cost.
Report resources consumed alongside success.
Reconcile prompts requesting foreground-only execution with harness support for
continued background inference before freezing the prompts.

Each run receives only its prompt, applicable API documentation and observed
files in an isolated working directory. Cases alone are supplied for scenarios
1a/1b/2; scenario 3 receives all streams. Truth, references, other runs and the
plan must be inaccessible. A temporary directory alone is not a filesystem
sandbox. The development harness runs agents with unrestricted permissions, so
repository files, including simulation truth, are readable by absolute path.
Enforce and verify read and network restrictions before launch.

Simulation files retain the structure
`simulations/{variant}/rep_{rr}/{data,truth}/`. Evaluation uses truth outside the
agent environment. Pin source and environment hashes and save simulation and
inference seeds where available.

## Outcomes and analysis

| Outcome | Estimand and denominator | Measurement | Role |
|---|---|---|---|
| Recovery | Mean of six scenario–model cell median WIS among valid canonical-panel runs; equal weighting of the four prompts within each condition and cell | Weighted interval score on days 25–125 against the instantaneous target | Primary hypothesis H1 |
| Execution | Mean valid-output proportion across the same six cells and four prompts; all allocated canonical-panel runs | Date and numerical validation, by assigned condition | Primary hypothesis H2; accompanies H1 |
| Uncertainty and point error | Interval availability among all runs; 50% and 90% coverage, width, WIS components and absolute error of the median among valid runs | Numerical validation and scoring | Secondary estimation |
| Implementation | Per-component detector flag proportions among runs with reviewable code, with detector accuracy from 18 human-reviewed submissions | Automated detectors, validated by human review | Secondary estimation |
| Code size and resources | LOC among submitted scripts; time, retries and cost among all runs | Static counts and execution records | Descriptive |
| Method choice and diagnostics | Tool/method frequencies and diagnostic reporting, with applicable denominators | Code and log review | Descriptive |

The six primary cells are scenarios 2 and 3 crossed with the three commercial
models. Scenarios 1a/1b, the open-weight model and sensitivity variants retain the
same outcome definitions and are reported separately. They do not enter the
two-hypothesis testing family below.

### Primary assessment: execution and recovery

Analyse all allocated runs by assigned condition, including instruction violations
and failures. Distinguish infrastructure failures, agent failures, invalid outputs
and statistical fitting failures. Pre-specify any infrastructure-only reruns and
retain the original records.

Report two linked outcomes:

- **Valid-output proportion**, with every allocated run in the denominator.
  Validity requires correct dates, all nine quantile columns with finite,
  non-negative and non-decreasing values, and no duplicate or missing dates in
  the scoring window. File existence alone is insufficient.
- **Weighted interval score (WIS) conditional on valid output**, calculated per
  run as the mean daily WIS against the matched simulation target. Report its
  distribution and the denominator. This comparison is affected by selective
  success and must accompany the failure rate.

For quantile levels forming central intervals $\alpha_k\in\{0.05,0.1,0.2,0.5\}$
with median $m$ and truth $y$,

$$\mathrm{WIS}(y)=\frac{1}{K+1/2}\Bigl(\tfrac12|y-m|+\sum_{k=1}^{K}
\tfrac{\alpha_k}{2}\mathrm{IS}_{\alpha_k}(l_k,u_k;y)\Bigr),\qquad
\mathrm{IS}_\alpha(l,u;y)=(u-l)+\tfrac{2}{\alpha}(l-y)\mathbf 1(y<l)+\tfrac{2}{\alpha}(y-u)\mathbf 1(y>u),$$

with $K=4$. WIS approximates the continuous ranked probability score
([Bracher et al. 2021](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008618)).
A point-only submission reduces to absolute error. Score on the natural scale;
report WIS on the log scale as a secondary analysis.

Do not assign an arbitrary finite score to failed runs or describe conditional
WIS as overall task success. Report instruction adherence separately.

Use days 25–125 inclusive as the proposed retrospective scoring window, subject
to boundary validation under the corrected generator. Early estimates depend on
prehistory; late infections have reports after day 150. Also report early and
late performance separately. Fitting all 150
days is retrospective inference; evaluating operational real-time estimation
would require fitting successive data cut-offs.

### Uncertainty and point error

Report the proportion of runs with non-degenerate intervals. For valid runs,
report 50% and 90% temporal coverage, mean interval widths, the WIS decomposition
into dispersion, overprediction and underprediction, and the absolute error of the
median. Point-only submissions are excluded from coverage and width summaries,
and their number is reported. Temporal coverage on a single trajectory is descriptive;
pointwise repeated-sampling calibration requires independent simulated datasets.
Adjacent days are not independent observations. Excessively wide intervals can
achieve high coverage, so higher coverage alone is not evidence of better performance.

### Implementation assessment

Implementation assessment covers delays, GI discretisation, boundary treatment,
weekly reporting, ascertainment assumptions, process/observation variation and
shared infections. Prompt compliance and scientific adequacy are recorded
separately: an instruction can require a component that is not identifiable or
needed under the DGP. Scientifically equivalent implementations are accepted.
Scenario 1a permits methods other than a renewal estimator, with their estimands
recorded explicitly.

`evaluation/detectors.py` supplies per-component flags for every run with
reviewable code. Imports, names or regular-expression matches cannot establish
semantic correctness, so flags are reported with their measured accuracy.

One infectious-disease modeller who did not implement the detectors reviews 18
submissions: one randomly selected canonical run with reviewable code per
scenario–condition–commercial-model cell in scenarios 2 and 3, using a recorded
seed. For each detector component, the reviewer records **correct**,
**incorrect**, **absent** or **unclear** on a fixed checklist; open-ended review
is not required. Review is capped at 30 minutes per submission, giving a
maximum of 9 hours. Report per-component sensitivity, specificity and raw
agreement with counts and exact binomial intervals. With 18 submissions these
intervals are wide; they indicate gross detector failures, not precise accuracy.
Per-component error estimates remain detector-based.

Purposively selected failed or flagged runs may illustrate failure modes in the
report, within a total human review budget of 10 hours, and do not enter any
estimate. Record the language, packages and method of every run from code and
logs, including scenario 1a estimands, without separate human review.

### Code size and diagnostics

Non-comment, non-blank lines of code measure submitted code size. They do not
measure interpretability or maintainability: libraries move implementation out
of the submitted script. Report LOC descriptively, using consistent file inclusion
rules.

Report retries, wall time, cost, API-related errors and diagnostic reporting.
An undefined function or method error is not necessarily hallucination: classify
unsupported API invention separately from version mismatches, argument errors
and missing imports. Presence of an R-hat or posterior-predictive-check command
does not show that the diagnostics passed. Apply convergence checks appropriate
to each inference method; report failed diagnostics separately from valid CSVs.

### Comparisons and uncertainty estimates

Estimate condition differences within each scenario and model. The principal
comparison is `composable` versus `julia`; `no-spec` comparisons describe the effect
of permitting unrestricted tool choice. Pool scenarios or models only with
explicit equal weights, and show constituent results.

For the one-dataset allocation, uncertainty describes repeated agent runs on the
fixed prompts and dataset. Resample runs within fixed prompt strata, preserving
condition allocation, and label intervals conditional on those strata. For a
multi-dataset allocation, resample datasets as matched clusters across conditions
and retain all nested runs and prompts. Do not resample days as independent units.
Variants are fixed sensitivity settings, not random replicates of a population
of DGPs.

Use a recorded seed for bootstrap intervals, after checking each procedure on
simulated evaluation outcomes. Sparse/all-success proportions
need intervals that do not collapse to certainty; use a suitable binomial method
only where its independence assumptions apply. Freeze and test
`evaluation/analyse.R` against these definitions before registration.

### Pre-specified hypotheses

The predictions concern `composable` versus `julia` under the specified workflow.
They are motivated by the possibility that supplied domain components reduce
implementation work.

| Hypothesis | Contrast, positive in the predicted direction | Prediction | Practical benchmark |
|---|---|---|---|
| H1: improved recovery | Mean over six cells of median WIS in `julia` minus median WIS in `composable`, canonical panel | Contrast > 0 | Awaiting calibration |
| H2: more valid outputs | Mean over six cells of valid-output proportion in `composable` minus that in `julia`, canonical panel | Contrast > 0 | 0.10, or 10 percentage points |

Each cell has equal weight, and each of its four canonical
prompt versions has equal weight before summarising runs. For H1, construct the
cell's WIS distribution as an equal-weight mixture of prompt-specific empirical
distributions among valid runs; take its median using the lower 0.5 quantile.
If any prompt–condition–cell has no valid run, H1 is not estimable as specified.
Report the available conditional distributions without dropping that stratum or
renormalising weights. H2 remains estimable. Each cell has equal weight across the three commercial
models regardless of their capability; model-specific contrasts are reported
alongside.

The practical benchmarks are study design judgements. The H1 benchmark is not
yet set: fix it before registration from reference WIS on development data under
the instantaneous generator, as an absolute difference or a relative reduction.
Report the relative change in either case. The H2 benchmark corresponds to ten more
usable outputs per 100 attempts under the same retry policy. Document investigator agreement on these benchmarks in the
registration; do not choose them from evaluation results.

Use two-sided 97.5% intervals for the two contrasts (Bonferroni allocation of
a familywise 5% error rate). For each hypothesis:

- **Supported direction:** the entire interval is above zero.
- **Contradicted direction:** the entire interval is below zero.
- **Unresolved direction:** the interval includes zero, or the contrast cannot be
  estimated. Failure to support a prediction is not evidence of equality.
- **Practical benchmark exceeded:** the lower limit exceeds its benchmark.
  If the upper limit is below the benchmark, an improvement that large is not
  supported by the interval even when the direction is positive. Otherwise its
  practical magnitude remains unresolved.

Report conventional 95% intervals for secondary estimation, without additional
confirmatory labels. Assess interval behaviour before freezing analysis code.

H1 is conditional on successful execution, so its interval conditions on the
observed number of valid runs in each stratum; H2 covers variation in success.
For each scenario–model–condition–prompt stratum with $n_v\ge1$ valid runs,
resample $n_v$ valid runs with replacement, then recompute the weighted contrast.
Use 10,000 resamples and percentile limits. Strata with no observed valid run
make H1 not estimable, as above. Strata with one valid run contribute no
resampling variation; report how many there are. The precision report checks
coverage of this procedure under sparse success, including strata with one or
two valid runs, and the procedure is amended before registration if coverage
falls below nominal.

For H2, all 24 strata per condition contain five attempts, so the weighted
contrast is also the difference between two proportions with 120 attempts each.
Use the bounded-outcome concentration interval

$$[\widehat\Delta_2-h,\widehat\Delta_2+h]\cap[-1,1],\qquad
h=\sqrt{\frac{\log(2/\delta)}{120}},\quad\delta=0.05/2.$$

This follows from Hoeffding's bound applied to 240 independent signed Bernoulli
terms, each with range width $1/120$. It permits different success probabilities
across fixed strata and remains non-degenerate at all-success/all-failure
outcomes. Its half-width is about 0.19, so exceeding the 0.10 benchmark requires
an estimated difference of at least 0.29; smaller differences leave the benchmark
unresolved. It assumes independent agent attempts conditional on fixed inputs;
blocked scheduling and run logs help assess service or resource dependencies.

Support for H1 alone establishes lower error among successful runs. It does not
establish greater overall usefulness if H2 shows lower execution success. Present
the two contrasts together and describe any trade-off. Recovery on sensitivity
variants, component-specific errors, LOC, selected tools and model-by-condition
interactions remain pre-specified secondary analyses; they have no directional
confirmation thresholds.

Associations between detector flags and recovery are exploratory. Flags are
properties of generated code, and different variants generate different code;
these associations do not identify the causal effect of adding a component.
Model-by-condition interactions are also exploratory, particularly where pilots
motivated their expected direction.

## Work required before registration and execution

The design decisions retained here are the compact run allocation, fixed-input
estimand, two hypotheses, resource limits, scoring rules and review allocation.
The following tasks make those decisions executable:

| Work | Owner | Required evidence and deadline |
|---|---|---|
| Migrate the domain-specific condition | Project lead | ComposableTuringIDModels references, prompt/API bundles, pinned Julia environment, harness routing, detectors and analysis labels validated before registration; legacy EpiAware runs kept separate |
| Align instantaneous target, ascertainment inputs and process/observation specification | Project lead | Validated generator, truth, prompts and references before registration |
| Validate initialisation, scoring window and reference inference | Project lead, with independent implementation check | Versioned development validation report, including coverage, width, WIS and diagnostics, before registration |
| Inventory development use and select evaluation seeds | Project lead | Frozen dataset allocation and record of previous exposure before registration |
| Implement outcome definitions, contrasts and intervals | Project lead | Updated `evaluation/analyse.R`, synthetic-data checks and reproducible precision report before registration |
| Check wording, variant parameters and API bundles | Project lead | Frozen prompt/document versions and human audit before registration |
| Recruit a detector reviewer | Project lead | Name and relevant experience recorded, and the component checklist frozen, before registration |
| Verify run isolation and reproducibility records | Project lead | Harness validation, including enforced filesystem restrictions, and pinned environment/model configuration before agent execution |
| Fix the H1 benchmark | Project lead | Reference WIS under the instantaneous generator on development data, and the agreed absolute or relative benchmark, before registration |
| Update output specification | Project lead | Quantile columns in prompts, validity checks and `analyse.R` before prompts are frozen |
| Audit API bundle comparability | Independent auditor | Length, entry and example counts, and scenario coverage for both bundles before freezing |

The migration audit covers the following code paths:

| File | Required migration check |
|---|---|
| `evaluation/detectors.py` | Validate component patterns on new generated code, including false positives and negatives; retain legacy interpretation for pilot data |
| `evaluation/analyse.R` | Add explicit `composable` handling and implement the specified H1/H2 populations, denominators, weights and intervals; prevent pooling with legacy `epiaware` records |

The analysis entry point is `evaluation/analyse.R`. Existing tool behaviour remains
a development implementation until these checks pass. Preserve protocol/package
identity in records so labels cannot silently combine different interventions. Freeze tool versions with the
validated package and documentation before agent execution.

The precision report uses the fixed allocation and plausible success rates,
WIS distributions, including null effects and the practical benchmarks. It
covers the H1 conditional bootstrap under sparse success. Report interval
widths, empirical interval coverage, and probabilities of support, contradiction
and unresolved results. If an interval
procedure fails validation, amend it before registration and retain the validation
record. Predictions with a low chance of resolution are retained and reported
with that chance.

Optional extensions are additional datasets, a common elapsed-time/cost cap,
extra DGP variants and real-time data cut-offs. None is required for the stated
conditional hypotheses. Any adoption changes the protocol prospectively.

## Reporting, responsibilities and limitations

Publish run allocation and exclusions; execution success; conditional WIS;
interval availability, width and coverage; detector-based implementation
errors with detector validation; selected tools; code size; and costs.
Show results by scenario and model, with variant sensitivity alongside canonical
results. Report variation between the four prompt versions within each cell.
Select example trajectories by a fixed rule, such as the median-WIS valid run
within each cell, and show failures in the aggregate summaries.

UKHSA England data may provide a secondary workflow demonstration. It has no
known true reproduction trajectory and cannot validate recovery. Specify the
data version and period; package agreement alone is not evidence of correctness.

Sebastian Funk coordinates simulation, prompts, execution and analysis. One
infectious-disease modeller who did not implement the detectors reviews the
detector-validation sample, without LLM assistance.

Disclose authors' contributions to ComposableTuringIDModels, EpiAware, EpiNow2
or EpiEstim, funding and research credits. Publish unfavourable and inconclusive
results as well as successful runs. No patient-level data are used.

The principal limits are the selected models, prompts and epidemic processes;
the bundled nature of the conditions; dependence on externally supplied delay
parameters; and detector-based implementation assessment validated on a small
human-reviewed sample. Reviewability of the generated code is not assessed.
Simulation misspecification
can favour one estimator and therefore one condition: exposure to the same DGP
does not ensure an unbiased comparison for other epidemic settings. Conclusions
concern the tested workflows and do not establish clinical usefulness or general
superiority of a framework. Further limits:

- **Structural contamination.** Renewal models are common in training data.
  Simulated data avoid memorised datasets, but not recall of model structure.
- **Prompt wording.** All prompt versions are LLM-drafted or LLM-generated; no
  independently written human prompt is sampled.
- **Model family.** Commercial results come from one provider's models and may
  not generalise to other providers.
- **Sampling settings.** Claude Code does not expose temperature, so run-to-run
  variation reflects default sampling only.
- **Simplified observation process.** Overdispersion arises only from individual
  transmission heterogeneity; ascertainment is a deterministic function of time;
  observation noise is independent across streams. Estimators that model other
  sources of variation, individual-level ascertainment or correlated reporting
  errors gain nothing from those features here.
- **Replication.** There is no independent replication. The harness, prompts and
  analysis code are published to allow it.

*Created: 2025-12-07. Earlier versions and development decisions remain in git history.*
