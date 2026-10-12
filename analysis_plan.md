# Analysis plan: human expertise in LLM-generated epidemic models

**Status:** revised draft, 2026-10-12; not registered. This document specifies the
proposed study. The generator, prompts, harness and analysis still implement parts
of the development design. The registration requirements below must be completed
before confirmatory execution; this revision does not certify their implementation.

## Aim, objectives and hypotheses

**Aim:** assess whether providing validated epidemiological modelling components
helps AI agents recognise and handle the methodological pitfalls of estimating
current transmission from surveillance data, and reduces the tokens required.

The wider research programme asks how AI can help us understand infectious disease
transmission, and where human expertise belongs in that pipeline. This study tests
one way of supplying expertise before an analysis is written: reusable software
components embodying epidemiological and statistical assumptions. It compares
agents using those components with agents addressing the same scientific tasks
using general probabilistic programming tools. Agents choose the analysis, including
which observation issues need attention. Both agents retain knowledge from training.
The comparison concerns the supplied software and documentation.

Both conditions receive the same external epidemiological inputs. The experiment
tests whether agents use those inputs appropriately; it does not test obtaining
or validating generation-interval, delay or ascertainment estimates from scratch.

The primary scientific question is: **does access to expert-informed software help
an agent avoid epidemiological omissions and implementation mistakes when asked to
assess current instantaneous R from surveillance data?** The practical question is
what transmission is now, given reports available at the observation cut-off.
Report an estimate's age alongside accuracy and uncertainty. An omission leaves a
relevant issue unaddressed; an implementation mistake handles it incorrectly.
Assess both from code and explanations, including correct package defaults.

| Objective | Directional hypothesis | Evidence supplied by this design | Limit |
|---|---|---|---|
| Determine whether components help agents avoid epidemiological pitfalls | H_E: fewer substantive omissions and implementation errors under `composable` | Direct human review plus executable input-use probes and targeted checks | Human review remains small; calibrated probe evidence covers specific behaviours, not whole-analysis correctness |
| Assess whether components support informative, appropriately qualified answers about current transmission | Descriptive prediction: better current recovery without unsupported certainty or misleading date claims | All-attempt current/earlier/withheld answer categories, target-date lag, conditional recovery and uncertainty; sampled review of scientific justification | No single accuracy–recency score or additional confirmatory test; earlier and withheld answers leave the current question partly or wholly unanswered |
| Determine whether components reduce agent effort | H_T: lower total tokens under `composable` | Complete token accounting, including documentation, execution feedback and repairs, for all allocated attempts | Tokens measure agent resource use, not human effort or total computation |
| Determine whether components improve statistical recovery | H_R: lower interior held-out WIS under `composable` | Technical check of known instantaneous R recovery on unseen data with different transmission trajectories | Conditional on interior estimates |
| Assess practical usability and the limits of packaged expertise | No additional confirmatory hypothesis | Execution, diagnostics and error types; smaller unrestricted-choice and fixed-package benchmarks | Tool choice is not randomised within unrestricted runs; unsupported package features need a separate interpretation |

H_T and H_R form the two-contrast confirmatory testing family. H_E is a central
scientific prediction assessed descriptively through limited human review and
calibrated probes, with unresolved cases explicit. Assess efficiency jointly with
useful-output availability and accuracy. Current-transmission assessment is central
to interpretation; its recency, availability and conditional accuracy outcomes are
descriptive. Interior H_R alone cannot support a claim of better assessment of
current transmission.

[Gostic et al., Practical considerations for measuring the effective reproductive
number](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008409)
informs the evaluator's checklist of pitfalls, including generation-interval
assumptions, infection-to-observation delays, incomplete observation and right
truncation. Freeze the rubric and accepted alternatives before evaluation. The
paper and checklist are not supplied to core agents. Providing the paper is a
separate guidance intervention specified under secondary extensions.

The study does not estimate the benefit of live expert supervision, the human time
needed to repair generated analyses, or the best position for every kind of human
input. Those are research-programme questions. It also cannot identify whether an
agent recalled a solution or constructed one from primitives from its outputs.

## Study history and registration

Phase 1 (December 2025–April 2026) compared R, Python, Julia and EpiAware. Some
scenario 1a runs were completed; the planned analysis was not. Tracked materials
remain in git history; gitignored run outputs were not retained. Early execution
limits, installation problems and other breakages informed subsequent development.
No phase 1 results enter the confirmatory analysis.

Phase 2 development included simulated data, reference calibration, prompt changes
and agent/harness pilots. Earlier designs and hypothesis changes remain in git
history and belong in the development inventory.

Generator development on 2026-10-12 passed 143 structural assertions and a
300-dataset stream-count/prehistory check. Ten pilot blocks used master seed
2026101201; 100 count-validation blocks used 2026101203. Test seeds and source
hashes are recorded in [the development record](simulations/development_validation.json).
These are development data; reference recovery and interval validation remain pending.

The correspondence in `context/s41591-026-04350-5.pdf` motivates safeguards against
prompt selection, development/evaluation leakage and selective reporting:
independent prompt checking, wording variation, preregistration, unused evaluation
data and complete reporting.

| Concern raised by the correspondence | Safeguard and remaining limit |
|---|---|
| Hypothesis-aware prompt construction | One shared scientific template per task/wording, with only tools and documentation differing; independent neutrality/equivalence audit. Expected direction is not supplied to the auditor and prior awareness is recorded. Author involvement remains; this is not fully external hypothesis-blind prompt construction. |
| Selective configurations and reporting | Freeze both wordings and publish all allocated results and failures. Two wordings give limited robustness evidence. |
| Development-driven hypothesis changes | Inventory prior work and register before confirmatory queries; record prospective amendments. |
| Memorised test cases | Generate new concealed datasets, while acknowledging training familiarity with model structures. |
| Independent replication and reproducibility | Freeze actual agent context, versions and settings and release materials. External replication is enabled but not included. |

Before registration, publish an inventory of development datasets, runs and prior
exposure. Freeze the protocol, prompts, software, analysis and unused evaluation
seeds. Report development-informed analyses as retrospective. Date subsequent
amendments and state whether evaluation outcomes had been inspected. Legacy
`epiaware` runs retain their label and are never pooled with `composable` runs.

## Core comparison: supplied expertise in software

| Condition | Required tools | Documentation |
|---|---|---|
| `julia` | Julia and Turing; implement epidemiological structure using general tools | Shared Turing and generic numerical API reference |
| `composable` | Julia, the same Turing version, and ComposableTuringIDModels.jl | Identical shared reference plus the domain-component API reference |

Both conditions may use the same generic packages for distributions, numerical
integration, optimisation, data handling and plotting. In `julia`, domain packages
for renewal models, reporting-delay correction or reproduction-number estimation
are unavailable and prohibited. Freeze an explicit package allowlist and audit
imports and vendored code. In `composable`, require at least one domain-package
model component to contribute to the fitted model's likelihood or latent process;
custom components remain permitted. Importing the package, constructing an unused
component or using it only for plotting does not satisfy this requirement. Record
the eligible component inventory with the pinned API. Record non-use and uncertain
use separately, with source evidence. Analyse by assigned condition even when the
requirement is violated.

This is the effect of access to, instructions to use, and documentation for a
particular component system. Documentation length, API familiarity and package
assumptions are part of that intervention. Their individual contributions are not
identified. Neither condition removes the agent's existing epidemiological knowledge.

The domain condition uses
[ComposableTuringIDModels.jl](https://composableturingidmodels.epiaware.org/dev/).
Pin a source commit, compatible Turing environment and documentation snapshot;
the `/dev/` URL is not a reproducible version identifier. Migration from EpiAware
requires new references, documentation, harness routing and validation.

Before calling the supplied components validated, record the tests, reference
comparisons, supported assumptions and boundary cases relevant to each task. This
inventory establishes the evidence for the selected version. Package use does not
certify the scientific adequacy of an assembled analysis. Accept a version only
after clean instantiation and all task references and output checks pass. If no
version passes, postpone registration of the affected comparison. After freezing,
upgrades require a prospective amendment and a decision about affected runs.

Both conditions receive identical task information: observed data, external
epidemiological inputs, metadata, requested estimand and output contract. Shared
generic documentation is byte-identical. The additional domain documentation is
part of the intervention and may name scientific issues and offer methodological
cues through component descriptions. Equal task information does not mean equal
total methodological information. Bundles may contain function signatures, argument
explanations and short API examples, but no worked study solutions. Audit the
shared reference to exclude epidemiological model equations and task-specific
recipes; generic probability and Turing examples are allowed. Record hashes,
reviewer and date. Validate API compatibility separately.

An auditor who did not write the bundles checks their scenario coverage, length,
function entries and examples, and verifies that condition differences implement
the intended intervention. Report this audit and authors' package contributions.
`evaluation/validate_api_docs.py` can assist; human checking determines compliance.

## Tasks and agent deployments

| Task identifier | Agent-facing scientific task | Evaluator's observation regime |
|---|---|---|
| `scenario_1b` | Assess current instantaneous R from reported cases, quantify uncertainty, provide the estimated trajectory and explain limitations | Constant report-date ascertainment; no weekly effect; stochastic infections; known GI and case delay |
| `scenario_2` | The same current-R request, with richer observed data and external inputs | Time-varying report-date ascertainment and weekly reporting effects; stochastic infections; known GI and case delay |
| `scenario_3` | Assess current instantaneous R using cases, hospitalisations and deaths, quantify uncertainty, provide one trajectory and explain limitations | Shared latent infections; stream-specific ascertainment and delays; weekly reporting effects |

These tasks represent distinct uses of modelling components. They are not a
calibrated difficulty scale. The simple case task provides a setting accessible
to both fixed and composable packages; the multi-stream task tests composition
beyond a single established workflow. Task differences do not isolate the causal
contribution of an individual component.

All core tasks specify the same reproduction-number estimand, centre the assessment
on current transmission and request a trajectory, uncertainty and limitations.
Current means the observation cut-off date; R is indexed by transmission date.
The target is effective instantaneous R, not the basic reproduction number R0.
Supply data provenance, observation cut-off, date-bin definitions,
and available external GI, delay and ascertainment inputs, with their scientific
meaning. Distinguish these inputs from a recipe: do not prescribe a renewal model,
latent infection construction, delay correction, discretisation algorithm, weekly
effect, observation likelihood or boundary adjustment. Agents decide which issues
require attention and how to address them. Scientifically defensible alternatives
are eligible; presence of a named component is not itself the grading criterion.

Known report-date ascertainment is supplied to make recovery identifiable. Its
baseline scale and date interpretation are metadata, not instructions to fit a
particular model. Weekly effects are not explicitly announced as a task requirement.
Unknown ascertainment remains optional: counts alone generally cannot identify both
infection and ascertainment trajectories, even with several streams. Recognition
of insufficient information would be an appropriate response in that extension.

The legacy identifiers 1b, 2 and 3 are storage labels. Record protocol versions to
keep runs with earlier scientific instructions separate. Broad versus checklist-based
requests are an optional extension.

The core has two Claude deployment slots, one Sonnet and one Haiku. The proposed
models are `claude-sonnet-5-5` and `claude-haiku-5-5`. **Haiku 4.5 is excluded from
the Haiku slot.** As checked on 2026-10-12, the
[official lifecycle table](https://platform.claude.com/docs/en/about-claude/model-deprecations)
gives earliest retirement dates of 2027-09-28 and 2027-10-07 for the proposed models.
Record the checked source snapshot. Any other model choice
requires a prospective protocol amendment and the same lifecycle checks.

Freeze the actual choices after compatible development pilots and an account-level
availability check. Verify lifecycle coverage for the projected launch/completion
window plus two weeks of operational margin; do not rely on being active today.
Pin model identifiers, CLI versions, settings and dates, and record the returned
model identity for every call. If retirement or forced version change interrupts
the study, stop the affected deployment and amend prospectively; do not silently
substitute within blocks or pool versions. These are two deployments in one family,
not a controlled manipulation of model size or reasoning. Opus and Qwen remain
optional secondary comparisons, with cross-harness differences reported.

## Simulation and estimands

### Reproduction-number target and infection process

`simulations/generate.jl` generates development data through the daily renewal
implementation in `simulations/renewal.jl`. The legacy individual-level process
is retained in `simulations/generate_cohort.jl`; its multiplier is indexed by the
parent's infection time and targets cohort R. Existing legacy datasets retain
that interpretation and remain separate from instantaneous-R development data.
Generator implementation does not establish reference recovery or interval coverage.

The confirmatory target is instantaneous $R(d)$. Use a daily renewal generator
with GI masses $g_s$, $s\geq1$, and infectiousness

$$\Lambda_d=\sum_{s\geq1}g_s I_{d-s},\qquad
I_d\mid I_{<d}\sim\mathrm{NegBin}\left(
\text{mean }R(d)\Lambda_d,\ \text{size }k\Lambda_d\right).$$

Set $I_d=0$ when $\Lambda_d=0$. This specification puts the time-varying multiplier
at the time of transmission. It gives conditional variance
$R(d)\Lambda_d+R(d)^2\Lambda_d/k$. Here $k$ controls dispersion relative to daily
effective infectiousness; it does not automatically reproduce a persistent
individual-level infectivity process. Validate its interpretation, Poisson limit,
initialisation and indexing before registration. The realised ratio
$I_d/\Lambda_d$ is not the simulation parameter being scored.

For a cohort process, the mean multiplier instead enters inside the convolution,
$m(t)=\int R_c(t-a)m(t-a)g(a)\,da$, whereas instantaneous renewal uses
$m(t)=R(t)\int m(t-a)g(a)\,da$. Changing transmission can therefore produce
systematic target differences. Do not interpret disagreement with existing cohort
truth as an implementation error. Cohort simulations require separate scoring
with the appropriate target if retained as an extension.

### Observation process and identifiability

For realised infections $I$ and stream $s$:

$$\mu_{s,d}=\alpha_s(d)w_{s,\mathrm{dow}(d)}
\sum_{e=0}^{D_{\max}} f_{s,e}I_{d-e},\qquad
C_{s,d}\mid I\sim\mathrm{Poisson}(\mu_{s,d}).$$

Ascertainment multiplies delayed infections on the **report date**; infection-date
ascertainment would appear inside the convolution. The rates and weekly factors
must use the same convention in truth, prompts and references. Conditional on
infections, streams are independent Poisson measurements, not binomial thinning
or linked clinical outcomes. Counts may exceed infections. Hospitalisation and
death rates are simplified stream observation rates.

Stochastic infections induce marginal variation even with conditionally Poisson
observations. A negative-binomial observation likelihood is not a required marker
of correctness. Assess whether the full model handles the specified variation;
accept scientifically defensible alternatives and approximations. An arbitrary
extra dispersion parameter need not improve recovery.

In 1b, use constant case ascertainment 0.4 and weekly factors one. In 2 and 3,
use the time-varying curves and weekly effects below, with case observations
shared between those tasks within a dataset. Use the same latent infection
trajectory across all three tasks within a dataset, generating the simplified
1b observations separately. The tasks change information and observation
requirements; all conditions within each task receive identical inputs.

### Trajectory family, simulation seeds and boundaries

Generation and held-out trajectories are independently drawn from the same frozen
piecewise-linear family. Knot days are 1, $t_2$, $t_3$, 150, with
$t_2\sim\mathrm{DiscreteUniform}(35,55)$ and
$t_3\sim\mathrm{DiscreteUniform}(85,110)$. Knot values are 0.8,
$U(1.2,1.6)$, $R_3\sim U(0.75,1.05)$ and $R_3+zU(0.1,0.3)$, with independent
draws. Here $z=0$ gives a flat tail, $z=1$ a rising tail and $z=-1$ a falling tail.
Draw $z$ independently with probabilities 0.4, 0.3 and 0.3 respectively; report
realised class counts without redrawing to enforce a balance. Draw all parameters
independently between generation and held-out phases. Thus a new dataset changes
both stochastic realisation and transmission trajectory. No trajectory values
are given to agents.

Show recent-date results by flat, rising and falling tail class descriptively.
A flat tail assesses spurious cut-off effects; changing tails assess response to
ongoing changes in transmission. Development references also use a fixed
0.8–1.5–0.8–0.8 trajectory.

| Quantity | Starting specification, subject to development validation |
|---|---|
| Observation period | 150 days, starting 2023-01-01 |
| Instantaneous R | Independent piecewise-linear trajectories from the family above |
| GI | Gamma, mean 5.5 days, SD 2 days; positive daily lags 1–30, renormalised |
| Process dispersion | $k=1$ under the daily process above |
| Initial infection incidence | 1,000 at day 0; deterministic growth-compatible prehistory on days −59 to 0 |
| Case delay | Log-normal, mean 5 days, SD 2 days |
| Case ascertainment, tasks 2/3 | $0.4+0.2\sin(2\pi d/150)$ |
| Hospitalisation delay; ascertainment | Log-normal, mean 10 days, SD 3 days; $0.04+0.02\sin(2\pi d/150+\pi/3)$ |
| Death delay; ascertainment | Log-normal, mean 20 days, SD 5 days; $0.008+0.004\cos(2\pi d/225)$ |
| Weekly factors, tasks 2/3 | 7/6 Monday–Friday, 7/12 Saturday–Sunday; mean one, common true pattern across streams |
| Delay support | Days 0–30, renormalised |
| Simulation manifest master seed | Proposed 20261012, subject to prior-use audit before registration |

The ascertainment curves give baseline report-date rates on a mean-one
reporting-factor scale. Their variation means a realised weekly average need not
equal a single day's baseline. Input metadata specify this scale without labelling
rates as weekday-specific or announcing a weekend reduction. Assess equivalent
parameterisations through their implemented product.

The existing continuous-delay PMF integrates over uniform infection time within
a day:

$$P(D=e)=\int_0^1\{F(e+1-p)-F(\max(e-p,0))\}\,dp,$$

followed by renormalisation on finite support. The development GI uses the same
daily integration, conditioned on lags 1–30; record this convention with the data
and validate it before registration. Input metadata in
both conditions describe continuous delay distributions, support and daily date
bins; they do not provide a discretisation formula or instruct agents to correct
interval censoring. The rubric accepts defensible approximations and alternative
algorithms, and records incompatibilities with the supplied scientific inputs.

For prehistory, solve $1=R(1)\sum_s g_s\exp(-rs)$ and set
$I_d=\operatorname{round}(1000\exp(rd))$ for days −59 through 0. Include these
infections in the observation convolution, then simulate the stochastic process
from day 1. Validate indexing, rounding, a longer prehistory and scoring-window
stability.

Before registration, inspect expected and realised case/hospital/death counts by
time window across the trajectory family, particularly death counts near the
cut-off; document extinction and near-zero-count frequencies. The starting level
and family are proposed numerical settings, not already validated. Amend them on
development evidence if necessary and freeze before generating evaluation data.
Never regenerate evaluation blocks merely because counts or recovery are poor.

Create a manifest of distinct simulation seeds for every block and phase using
the pinned RNG and master seed; freeze the manifest and RNG version. Audit the
proposed master and all child seeds against prior use, including development seeds
101–120. Any replacement seed must be agreed before evaluation, not chosen after
examining its trajectory. Record allocation, inference, review and bootstrap seeds
separately. Distinguish initial infection incidence from RNG seeding in reports.

The operational target is **R on day 150**, using only reports through day 150.
Report its recovery, uncertainty and availability separately from the **recent
trajectory on days 144–150**. Also retain **interior days 25–125 for H_R**, subject
to development boundary validation before freezing. Current and recent outcomes
are essential scientific evidence with descriptive inference. Earlier or withheld
answers remain in their all-attempt denominators and do not exclude usable
interior estimates from H_R. Early days 1–24 and remaining dates are descriptive.
Recent infections have missing future reports even though all observations through
day 150 are supplied. An earlier estimate is evaluated against truth on its stated
date; it is not scored as an estimate of day-150 R.

This tests a current-transmission assessment at one observation cut-off, alongside
recent and retrospective recovery, in one fit per implementation. It does not
evaluate performance across successive real-time
cut-offs or revisions; those require a separate prospective refit allocation.

### References and optional stress tests

Port and validate the existing EpiAware references for 1b, 2 and 3. Implement an
independent Julia/Turing reference, including the shared-infection task, to check
the scientific specification. References demonstrate feasibility and provide
performance context; matching their code or assumptions is not the grading rule.
Use the same supplied information. Keep references inaccessible to agents.

Choose priors and validate reference inference, interval width, coverage, WIS and
diagnostics on development data under the instantaneous generator. Prespecify
reference pass/fail criteria, then validate acceptance. Freeze practical effect
benchmarks using development results and investigator judgement before evaluation.

The existing short-GI, long-delay, strong-weekly-effect, high-ascertainment-variation,
low/extreme-dispersion, abrupt-change and sinusoidal-R variants remain candidate
secondary stress tests. They are not crossed with every core condition by default.
Specify adopted variants, input configurations, seeds and run counts prospectively.
A refit may change supplied numerical parameters through the frozen interface;
a change requiring new model structure requires a separate code-generation task.
Stress-test refits of one script are correlated evaluations, not new implementations.

## Allocation, prompts and evaluation workflow

### Proposed core allocation

| Dimension | Allocation |
|---|---|
| Software conditions | 2: `julia`, `composable` |
| Scientific tasks | 3: 1b, 2, 3 |
| Agent deployments | 2 |
| Prompt wordings | 2 per task and condition |
| Independent outbreak blocks | 10 |
| Agent code-generation runs | **2 × 3 × 2 × 2 × 10 = 240** |
| Primary unseen-data evaluations without an agent | **240** allocated; missing executable implementations count as failures |
| Diagnostic repeat and input-probe refits | **1,120** additional allocated refits, subject to validation and computing feasibility below |

Each block contains independent generation and primary held-out datasets with
independently drawn transmission trajectories. A third diagnostic dataset supports
the data-use probe, giving 30 outbreak realisations grouped into ten matched blocks;
only the primary held-out dataset enters H_R. Within a block, every condition,
deployment and wording receives identical task-specific generation data and is
evaluated on identical held-out and diagnostic inputs. The diagnostic trajectory
uses the same family with the opposite tail direction to the primary trajectory;
for a flat primary tail, draw a rising or falling diagnostic tail with equal
probability. Draw remaining parameters independently and record all seeds.

There are at most **1,360 machine-only refits**: 240 primary, 240 unchanged repeats,
240 alternative-data probes, 240 delay probes, 240 GI probes and 160 ascertainment
probes (tasks 2/3). This is an explicit computing expansion, with no extra agent
runs or human-review sample. Produce a runtime/cost estimate and validate probe
informativeness before registration. If infeasible, amend the probe allocation
prospectively, with its reduced coverage stated; never choose which checks to run
from evaluation results. Missing executables remain visible in the allocation.

There is one agent attempt per condition–task–model–wording–block combination.
Epidemic and agent variation are sampled together, with ten matched outbreak blocks
as data replication. Within-dataset rerun variance is not separately estimated.
Validate precision and computing feasibility before registration; allocation changes
require prospective amendments.

### Prompt construction

Prompts specify the scientific task, supplied data and inputs, allowed tools and
output/interface contract. They do not disclose true R, latent infections or the
complete simulation equations. Task information and output requirements
match across conditions. Retain a neutral infectious-disease outbreak description
and the surveillance file format used in development.

The common scientific request is: **using the reports available through this date,
assess the current instantaneous reproduction number and its uncertainty. Provide
the estimated trajectory and explain material limitations.**
The multi-stream task requests one trajectory informed by all streams without
prescribing a shared-latent model. Do not provide the evaluation checklist, praise
the assigned software or suggest its expected advantage. Construct each condition
version from the same scientific template by replacing only the tool/documentation
paragraph. Scientific metadata may define an input; it must not add instructions
about the adjustments being evaluated. Require a short `outputs/analysis_notes.md`
describing the chosen method, assumptions and limitations, without listing pitfalls
the agent should discuss. Notes are evidence of what was stated, not internal
reasoning. The refit script supplies a dataset-specific current assessment and
withholding reasons below; these must describe the supplied refit data.

Use two frozen wordings per task and condition: slot 01 is the revised author-edited
base prompt; slot 02 is a separately paraphrased version. Existing paraphrases must
be regenerated after scientific requirements change. Record the writer or model
and editing history. An independent checker verifies scientific equivalence, equal
specificity, input values, neutral tool descriptions and absence of worked solutions
or a supplied scientific checklist. The checker receives the task/input contract
without the predicted direction; record any prior hypothesis awareness. Checking
occurs before evaluation, with no access to outcomes. These are selected wordings,
not a sample of all plausible human prompting. Full external hypothesis-blind
authorship is not a prerequisite; disclose author involvement and its limits.

Freeze 12 core prompt templates (3 tasks × 2 conditions × 2 wordings), documentation
bundles and data/configuration substitutions. Audit equal task information, while
recording the methodological cues in the additional domain documentation as part
of the intervention. Show wording-specific results.
Randomise scheduling across conditions and deployments within blocks; record
worker load, machine and service dates to assess operational dependencies.

### Code generation, execution and infrastructure

Retain the developed agent workflow: at most 200 assistant turns per session, up
to five resumed retries, and a wait of up to 60 minutes for continuing inference
before consuming a retry. These are session and repair limits, not a ten-minute
inference cap. Reconcile foreground/background instructions and freeze the exact
stop and cleanup rules. Use feasibility pilots to ensure legitimate reference
inference fits the available resources. Pin one policy across core conditions.

Preinstall and test both pinned environments, including domain components, before
launch. Agents do not install packages. Use a documented common cache/compilation
policy and balanced worker resources. Installation, missing system libraries and
provider outages are operational failures; unsupported API use, missing imports
and incorrect model code produced by the agent remain study outcomes.

Before launch, freeze a rule for infrastructure-only reruns, with reasons decided
without examining recovery or comparing condition results. Preserve the original
attempt, replacement and linkage. For token accounting, charge all agent calls
and repairs associated with an allocated slot, including any replacement. Report
infrastructure-related resource use separately. Never rerun a scientifically
unfavourable or agent-caused failure as an infrastructure replacement.

Save conversations, code versions, final source hashes, outputs, per-call usage,
retries, waits, diagnostics, start/end times, inference seeds and environment
metadata. Retain existing run identifiers and add explicit block, generation/refit
phase and parent-implementation identifiers. The current drivers need these
allocation and metadata changes before they can implement this design.

Start each allocated run with a fresh session and no inherited conversational or
project memory. Resume history only within that allocated workflow. Record and
freeze the actual system instructions, tool descriptions, injected configuration,
sampling/reasoning settings and memory policy, including defaults the CLI controls.
Audit initial context for the study hypothesis, evaluator checklist and solutions;
these must not enter the agent session. Document inaccessible provider-side context
and settings rather than assuming they are controlled.

Agents receive only their prompt, approved documentation, observed generation
data and scientific inputs. Truth, references, the plan, held-out data and other
runs must be inaccessible. A temporary directory is insufficient: the development
harness permits absolute-path reads. Enforce and test filesystem and network
restrictions, including access to unapproved domain packages, before launch.

### Frozen-code held-out evaluation

Require a documented script entry point accepting observed data, supplied
scientific configuration, inference seed and output directory. It must run ordinary
inference for that task without editing source or requesting agent assistance.
The agent-facing contract states that any supplied configuration value may change
on refit within the published schema and task, including delay location and support.
Read values from the supplied configuration on every refit; only fields explicitly
declared invariant may be fixed in source. Define each field's meaning and allowed
range without disclosing probe values or expected responses.
Validate this interface on development references before freezing prompts.

After generation, freeze source and inference settings and hash the artefacts.
The evaluator supplies the block's unseen dataset and runs that implementation
once for primary recovery in the same pinned environment under a predeclared
computation limit. Each refit starts in a clean directory containing only frozen
source and approved data-free resources, plus evaluator-supplied inputs. Generation
outputs, cached posteriors, fitted states and other refits are inaccessible; package
compilation caches follow the common policy. Audit the source/artefact inventory
for embedded fitted results and record input hashes and file-access provenance.
No LLM repairs, manual code changes or selection of a better refit are allowed. Fixed
parameters that should come from input are an implementation concern. Record
missing executable artefacts as failures; retain partial code for review where
it exists. Further refits, if adopted prospectively, remain nested within source
implementation and outbreak block.

Agents write `outputs/rt_estimates.csv` with one row for each observation date,
`status` (`estimated` or `withheld`), a `reason` field, and nine quantiles:
`q0.025`, `q0.05`, `q0.1`, `q0.25`, `q0.5`, `q0.75`, `q0.9`, `q0.95`, `q0.975`.
These describe uncertainty about instantaneous R, not future observed counts.
Point-only submissions repeat the point estimate across columns and are flagged
as lacking uncertainty despite numerical validity.

An estimated row requires valid quantiles. A withheld row requires empty quantiles
and a non-empty explanation generated by the frozen script. State only that agents
may withhold estimates they judge insufficiently supported; the contract must not
suggest particular dates or name right truncation. Missing rows, unexplained empty
values and invalid numerical estimates are output defects. Declared withholding
is a separate outcome: its scientific justification is assessed in the sampled
review and does not automatically imply an error or successful recovery.

Also require `outputs/current_assessment.json`, regenerated on each refit, with
`observation_cutoff`, `status` (`estimated` or `withheld`), `target_date` and a short
`explanation`. An estimated assessment names the date to which its headline R
estimate applies and references that date's quantiles in `rt_estimates.csv`.
The target date must be on or before the supplied cut-off and have a valid
estimated row. A withheld assessment has a null target date and explains why no
headline estimate is supplied. Require explicit dates and material limitations;
do not suggest a lag, boundary method or list of pitfalls. The summary may identify
an earlier estimate when current R is judged insufficiently supported. This
documents the answer given; it does not establish its scientific justification.
Missing or inconsistent summaries are separate output defects and do not remove
otherwise usable interior estimates from H_R.

### Executable input-use probes

The proposed diagnostic panel applies to every executable core implementation.
Start from its primary held-out data/configuration, alter one input at a time and
refit frozen code in a fresh directory. Use the same inference seed and budget.
Keep the original recovery result; probe outputs never replace it or enter H_R.

| Refit | Change from primary evaluation | Purpose |
|---|---|---|
| Unchanged repeat | Same data, inputs and inference seed | Detect residual numerical variability and non-reproducible execution |
| Alternative data | Replace observed counts by the block's independently generated diagnostic dataset; external inputs unchanged | Assess dependence on observed data; detect cached or input-independent output |
| Delay | Translate every stream's delay distribution by exactly three days, extending daily support from 0–30 to 0–33 | Assess use of delay inputs |
| GI | Multiply supplied GI mean and SD by 1.5; support unchanged | Assess use of generation-interval inputs |
| Ascertainment, tasks 2/3 | Replace each supplied ascertainment trajectory by its time average | Assess use of time-varying observation information |

For each stream's primary daily delay PMF $f_{s,e}$, the delay probe defines
$f^+_{s,e}=f_{s,e-3}$ on days 3–33 and zero elsewhere. It preserves total mass,
variance and shape, increasing the discrete mean by exactly three days. The input
interface represents continuous delay laws with a location parameter, zero for
primary inputs and three days for this probe, with unchanged base log-normal
parameters and extended support. Validate that applying the frozen daily
integration and normalisation reproduces the translated PMF. Both conditions
receive the same continuous-law metadata; no discretisation recipe or supplied
PMF is added to core prompts. Pilot this parameterised interface in both references,
including how the pinned package or custom components implement the location.

Record delay-location/support handling separately as supported, unsupported or
unresolved, using frozen static data-flow or executable provenance checks. Confirmed
non-handling is an **interface failure**, not a no-response result or evidence that
baseline delays were ignored. Unresolved handling makes this probe indeterminate.
A field-name search or evidence that a value was read is insufficient to establish
its use in the delay calculation; trace forwarded configuration through package
calls where needed. Validate these checks on development examples with correct,
ignored and indirectly forwarded location values. Remaining uncertainty stays
unresolved, without increasing the human-review sample or time caps.

Away from boundaries, shifting every stream's delays later is compatible with
infections and instantaneous R moving approximately three days earlier. Use this
as a directional positive control on changing trajectories; priors, initialisation,
stochastic inference and constant-growth periods can limit the fitted response.
Freeze interior comparison windows and tolerances using the independent references.

Compare estimated medians and quantiles over fixed windows and record changes in
estimate availability separately. Freeze distances, numerical tolerances and
eligibility rules using development references, repeated fits and deliberately
ignored-input controls. Validate positive controls on trajectories where the
input change is expected to matter; an unchanged repeat alone does not estimate
all Monte Carlo uncertainty. Do not choose thresholds from agent-condition results.

Classify each check as responsive, no detectable response, indeterminate, interface
failure, execution failure or not applicable. With no comparable estimates,
numerical sensitivity is indeterminate, even if withholding reasons change.
Failure of a diagnostic refit is not failure of the primary fit. Excessive repeat
variability also makes sensitivity indeterminate under the frozen rule. Preserve
denominators for all allocated code, executable code and applicable/assessable
probes separately.

For each input probe $p$ and condition $c$, report the task-standardised response
proportion

$$\widehat P_{p,c}=\frac1{|\mathcal T_p|}\sum_{t\in\mathcal T_p}
\frac{N_{p,c,t}^{\mathrm{responsive}}}{N_{p,c,t}^{\mathrm{assessable}}},$$

where assessable comprises responsive and no-detectable-response checks.
$\mathcal T_p$ contains all three tasks for alternative-data, delay and GI probes,
and tasks 2/3 for ascertainment. Pool deployments and wordings within each task;
report their constituent results and denominator composition. Give each applicable
task equal weight. Report the `composable` minus `julia` difference for each probe
descriptively, with no pooled score across probe types or additional confirmatory
test. A task with no assessable checks makes its standardised summary unavailable.

Alongside each response proportion, report allocated, applicable, executable and
assessable counts, and no-response, indeterminate, interface-failure and
execution-failure proportions among all scientifically applicable allocated checks.
The unchanged repeat has a separate reproducibility/instability summary.
Show block-specific counts and preserve whole
outbreak blocks and all linked refits for any uncertainty estimates, using the
validated clustered procedure below. Report undefined resamples. These response
proportions are conditional on assessability and are not scientific-error rates.

No detectable response is evidence to investigate, not universal proof of an
omission. Delay changes may leave R unchanged during constant growth; GI changes
may have little effect near R=1; constant ascertainment scaling can cancel from
growth-based estimates. Multi-stream information, priors and smoothing can also
limit sensitivity. Conversely, stochastic changes do not prove the supplied input
was used. Combine calibrated responses with source/provenance evidence and human
ratings; report flags separately from confirmed omissions. Component-specific
impact requires appropriate reference or code evidence.

These are repeated measurements of the same source implementation, grouped within
outbreak block, not additional independent implementations. Initial human ratings
remain blinded to probe flags; compare them only after ratings are frozen. This
panel makes targeted checks concrete while preserving the limits of a small human
sample. Probe validation uses a small set of known faulty controls; the previous
full mutation-study allocation remains outside the core.

## Outcomes and inference

| Outcome | Population and measure | Role |
|---|---|---|
| Total agent tokens | All 240 allocated generation slots; complete input and output usage across the workflow | H_T, confirmatory |
| Interior held-out recovery | Mean of six task–deployment cell median WIS values on days 25–125 | H_R, confirmatory; conditional on interior estimates, irrespective of recent withholding |
| Current-transmission assessment | All allocated attempts: current/earlier/withheld/invalid or absent assessment; headline and latest-estimate lags; day-150 availability, conditional WIS and uncertainty | Central scientific outcomes, descriptive; assess whether the operational question is answered |
| Recent recovery and withholding | Days 144–150: WIS, uncertainty, estimate availability and withholding reasons; flat/changing-tail results | Essential secondary context for the current assessment, reported together |
| Scientific omissions and implementation errors | Three randomly sampled reviewable implementations per task–condition cell, 18 dual-reviewed in total | H_E, direct secondary estimation |
| Input-use probe results | Task-standardised responsive proportion by probe and condition, condition differences and all-attempt assessability/failure counts; refits grouped by source and block | Secondary mechanistic evidence for H_E; not whole-code error rates |
| Execution, estimate availability and diagnostics | All allocated implementations; generation/refit completion, valid records, estimated/withheld/invalid dates and diagnostics | Essential companion outcomes; recent-date availability always reported |
| Automated implementation flags | All reviewable implementations; calibration against human review and targeted checks | Secondary, qualified by measurement accuracy |
| Further resources and method choice | Runtime, evaluator compute, monetary cost, retries, submitted LOC, components and tools used | Descriptive |

### Agent token use

For each allocated slot, sum actual input and output tokens over every unique
agent API call, including documentation read into context, repeated context,
tool feedback, reasoning usage as recorded by the provider, repairs and resumed
sessions. Use mutually exclusive usage categories: cache creation, cache reads and
other input tokens must not be double-counted. Likewise, do not add reasoning
again if already included in output totals. Deduplicate records after resumed
sessions and save the provider's raw usage fields and reconciliation rule.

Count failed workflows as well as successful ones. Missing telemetry is missing,
not zero; recover it from source records where possible. Unrecoverable totals make
the prespecified full-sample token contrast unavailable. Report completeness and
bounds or sensitivity analyses explicitly, rather than silently restricting it to
successful or fully logged runs. Record genuine zero-call infrastructure failures.

H_T compares arithmetic mean total tokens, giving equal weight to each of the six
task–deployment cells and equal weight to its two wordings and ten blocks. Report
absolute and relative differences, medians and distributions. Count cached tokens
as processed input for this outcome; report billed cost separately. Model/tokeniser,
cache policy and harness differences limit cross-provider comparisons. Human review
time and machine-only refit computation are separate resources, not agent tokens.

Prespecify a secondary token total excluding cache reads: ordinary uncached input
plus cache-creation input plus output, without double counting provider fields.
Report each component and monetary cost. The primary total measures full processed
context, so long cached histories can dominate it; this is part of its interpretation.

Report the proportion reaching a session-turn cap or exhausting the resumed-retry
policy by condition, task and deployment, with stop reasons. Realised totals at a
cap are complete for the fixed-workflow estimand; they are truncated observations
of an unobserved cost-to-success under an unlimited workflow. Do not interpret low
cost due to early failure or a cap as evidence of efficient successful analysis.

### Current-transmission assessment

Classify each primary held-out assessment using the frozen output contract:

- **Current estimate:** a valid headline assessment targets day 150.
- **Explicitly dated earlier estimate:** a valid headline assessment targets a
  date before day 150. Report its lag and recovery against truth on that date.
- **Withheld assessment:** no headline estimate is supplied, with an explanation.
- **Invalid or absent assessment:** required fields or referenced estimates are
  missing or inconsistent, or no assessment is produced. Separate execution
  failure, malformed output and missing output within this category.

Report category proportions for every assigned condition with all allocated slots
in the denominator, by task/deployment/wording and overall with equal cell weights.
Categories describe the submitted answer; scientific adequacy is assessed separately.
Appropriate uncertainty and justified withholding are assessed in the sampled
review. Clearly dated earlier estimates and justified withholding can be
scientifically defensible; neither is automatically an implementation error.
Presenting an earlier estimate as current is a substantive scientific error.

Report day-150 estimate availability independently of the headline category: a
trajectory may contain a current estimate while its summary chooses an earlier
one. For valid day-150 estimates, report WIS, median absolute error, interval width,
50%/90% coverage and non-degenerate uncertainty availability, with denominators.
Show all-attempt availability beside conditional accuracy. Use the same rules for
point-only submissions as below. Where development establishes a meaningful
day-150 recovery threshold, freeze it and report the all-attempt proportion with
both a valid current estimate
and that threshold as a secondary joint outcome, without calling it proof of
scientific correctness.

Record headline lag as cut-off minus declared target date, and trajectory lag as
cut-off minus the latest date with valid estimates. Report both distributions and
the number with no estimable lag; do not assign an arbitrary lag to missing or
withheld answers. Score each headline or latest trajectory estimate against truth
on its stated date. Earlier-answer recovery is conditional on agent-selected dates;
scores at different lags do not rank the ability to estimate current R. Show
current/recent outcomes for flat, rising and falling tails,
retaining realised category counts. Use sampled review to assess date claims,
uncertainty handling and the justification of withholding within the existing
rubric and time caps. Automated validity does not certify those judgements.

### Execution, recovery and uncertainty

Analyse by assigned condition, including violations and failures. Report generation
and refit completion, executable-artefact availability, valid-output records and
recovery-eligible outputs with all allocated slots in each denominator. A valid
record can contain declared withholding. H_R requires parseable, unique dates and
valid estimates throughout the interior window only. Withholding, invalid estimates
or contract violations outside that window are reported separately and do not
remove usable interior estimates. A missing or unparseable file has no recovery
score. Classify infrastructure failures,
agent failures, invalid outputs and statistical fitting failures separately.

Validate all required dates, statuses and withholding reasons. Estimated rows require
all nine finite, non-negative and non-decreasing quantiles. A valid record with all
estimates withheld has zero estimate coverage and no recovery score. It is not a
numerical failure or evidence of usefulness. A valid CSV does not establish
convergence or scientific correctness; report diagnostics separately.

Compute per-implementation WIS separately in the interior and recent windows against
true instantaneous R. For central intervals with
$\alpha_k\in\{0.05,0.1,0.2,0.5\}$, median $m$ and truth $y$:

$$\mathrm{WIS}(y)=\frac{\frac12|y-m|+
\sum_{k=1}^4\frac{\alpha_k}{2}\mathrm{IS}_{\alpha_k}(l_k,u_k;y)}{4+1/2},$$

$$\mathrm{IS}_\alpha(l,u;y)=(u-l)+\frac2\alpha(l-y)\mathbf1(y<l)
+\frac2\alpha(y-u)\mathbf1(y>u).$$

WIS approximates CRPS using the submitted quantiles; a point-only submission reduces
to absolute error. For implementations estimating every interior date, define

$$S_i=\operatorname{mean}_{d=25}^{125}\mathrm{WIS}_{i,d}.$$

Recent WIS is a separate secondary outcome, accompanied by withholding and estimate
availability. No reference-distribution score is assigned to withheld dates in this
design. Score on the natural R scale. Report median absolute error,
50% and 90% coverage and width, non-degenerate interval availability, WIS components,
and generation-data recovery as secondary outcomes. Exclude point-only submissions
from uncertainty-coverage summaries and report their number. Adjacent days are
correlated; temporal coverage is not a sample of independent calibration trials.

H_R is the equal-weight mean of six cell medians of $S_i$, with each cell pooling
its recovery-eligible outputs across the two wordings and ten blocks. Use the lower
empirical 0.5 quantile. Eligible outputs receive equal weight within a cell; the mix
of wordings and blocks among them may differ between conditions. Report these
counts and wording-specific distributions. If an entire task–deployment–condition
cell has no eligible output, H_R is not estimable as specified; do not drop the cell.

For every window, report the all-attempt proportion providing a complete set of
valid estimates, date-level estimate coverage, declared withholding and invalid or
missing outputs. Recent-window WIS among complete-window estimates is secondary;
partial-window scores are descriptive and labelled with the dates included. Report
uncertainty, errors and method explanations near the cut-off. Do not compare a
selectively estimated subset of dates as though it covered the complete window.

Conditional WIS describes recovery among outputs supplying the required estimates,
whose composition can differ between conditions through failure or withholding.
Never assign arbitrary finite WIS to either. Show execution, estimate availability,
withholding and recovery together. Where a meaningful recovery threshold has
been fixed from development, also report the all-attempt proportion achieving
both valid output and that threshold, as a secondary joint outcome.

### Direct implementation review and targeted checks

Two infectious-disease modellers independently assess the same 18 source
implementations, with at most 10 hours each. Review is direct scientific evidence,
and also calibrates automated flags. Reviewers did not implement the detectors;
record language/framework experience and relevant package contributions.

Randomly sample three reviewable implementations from each task–condition cell,
across both deployments, wordings and all blocks, after generation is complete.
Reviewability means source sufficient to attempt the rubric, including failed
implementations. Preserve sampling probabilities and seed. Review all if fewer
than three are available in a cell; report shortfalls and unreviewable-source counts.

Use a frozen Gostic-informed rubric for the R estimand, GI handling, reporting
delays, daily aggregation/support, initialisation, right truncation near the cut-off,
weekly variation, supplied report-date ascertainment, process/observation variation
and use of multiple streams. Assess both omission and incorrect handling. Include
the method notes, current-assessment dates/explanation and any withholding reasons.
The instantaneous estimand and transmission-date indexing are explicitly supplied;
that rubric item assesses compliance and implementation, not spontaneous recognition
of the distinction from cohort R or report-date R.
Assess date claims and handling of incomplete recent observations. Accept
scientifically defensible alternatives and justified withholding, evaluating the
implemented model for its target and inputs. Separate scientific defects,
instruction violations and uncertain adequacy; additional substantive defects can
be recorded outside the rubric.

For defects, record unaddressed, addressed incorrectly or unclassifiable, with source
and note evidence. These describe observable behaviour, not internal understanding.
Prose acknowledgements and implemented handling are assessed separately; correct
package defaults count even without an explanation.

Use task-specific caps of 20 minutes for 1b, 30 for 2 and 40 for 3. Six submissions
per task take at most 120+180+240=540 minutes, leaving one hour per reviewer for
briefing and familiarisation within ten hours. Time-limited or unclear assessments
with no confirmed defect remain unresolved. Mask agent/run identity, truth, scores,
detector/probe flags and other ratings; package identity may be apparent. Preserve
independent ratings and project-lead adjudication, with adjudication time recorded
outside this budget.
The [review protocol](evaluation/review_protocol.md) specifies operational rules.

Report confirmed substantive-error counts, no-error-found and unresolved counts,
per component and overall, by condition and task. Standardise to equal task weights
for a pooled descriptive contrast, accounting for the stratified sample. Show
sampling fractions and unresolved-case bounds for at most nine implementations
per condition; components and reviewers are correlated observations. No-error-found
is limited to assessed components.

Freeze targeted numerical and static checks using independent references and known
mistakes, including omitted delays, incorrect daily probabilities and boundary
indexing. Keep checks hidden from agents and calibrate adapted detectors. Report
flag rates separately from confirmed errors. Agreement, sensitivity and specificity
use raw counts, with unclear cases excluded and reported; calibration is descriptive.
The core excludes the full mutation catalogue.

Freeze materiality rules by rubric item, separating wrong scientific meaning from
negligible numerical approximations. For discretisation, tail truncation and weekly
smoothing, specify quantitative tolerances or accepted reference comparisons on
development data before reviewing evaluation code. The rubric must include a
scientifically acceptable weekly-aggregation/smoothing example and an example
where unmodelled weekly variation produces a material artefact. A 30-versus-40-day
delay support or different integration scheme is not automatically substantive.
Probe responsiveness alone is not an impact threshold. Where impact cannot be
resolved within the assessment cap, retain unclear rather than assert a defect.
The numerical materiality table remains a registration prerequisite.

### Contrasts, uncertainty and interpretation

| Hypothesis | Positive contrast in predicted direction | Population |
|---|---|---|
| H_T | Mean tokens in `julia` minus mean tokens in `composable` | All allocated generation slots |
| H_R | Mean cell median interior WIS in `julia` minus that in `composable` | Held-out estimates complete on days 25–125, six equally weighted cells |
| H_E | Task-standardised substantive-error proportion in `julia` minus that in `composable` | Random human-reviewed sample of reviewable code; descriptive |

For H_T and H_R, use two-sided 97.5% confidence intervals, allocating a familywise
5% error rate across two contrasts **provided the registered interval procedures
achieve their nominal coverage**. An interval entirely above zero supports the
predicted direction; one entirely below zero contradicts it; otherwise the result
is unresolved. Failure to support a prediction is not evidence of equivalence.
Report absolute and relative effect sizes. Practical benchmarks remain to be
agreed using development calibration and intended resource savings before
registration; they must not be selected from evaluation results.

For H_T, define an explicit paired difference per block:

$$D_{T,b}=\frac1{12}\sum_{t,m,p}(T_{j,btmp}-T_{c,btmp}),\qquad
\widehat\Delta_T=\frac1{10}\sum_bD_{T,b}.$$

There are twelve task–deployment–wording pairs per block, and every allocated slot
contributes. Report all ten block differences. Paired-block interval candidates
must pass the precision/coverage checks below.

H_R is the stated six-cell conditional recovery estimand, with selection through
execution and availability of interior estimates. Matched complete-output
comparisons are secondary and report their selected population.

Preserve matching when estimating uncertainty. A candidate procedure resamples
the ten outbreak blocks with replacement, keeping all datasets, all conditions,
deployments, wordings and source/refit links together, for 10,000 replicates with
a recorded seed. Recompute availability and recovery-eligible medians in each draw;
do not condition on fixed success counts or resample days as independent units.
An empty recovery cell in a bootstrap draw makes that draw undefined. Record this
frequency; do not silently discard such draws or renormalise cells.

Ten blocks, sparse successes and heavy-tailed tokens may make this bootstrap
inadequate. Before registration, validate interval coverage and precision under
nulls, plausible effects, dependence and sparse-success settings, including
undefined draws. Freeze the final interval construction and its failure rules
only after these checks. If no procedure supports the advertised coverage at this
allocation, amend the allocation or label the affected contrast estimation-only
before evaluation starts. Interval validation remains outstanding.

Report constituent task/deployment/wording results and resource distributions.
A resource saving accompanied by worse recovery or execution is a trade-off.
Interpret current-transmission performance using availability, accuracy, uncertainty,
lag and sampled scientific review together.
Lower WIS with unchanged direct error evidence supports a recovery benefit;
it does not establish the proposed error-prevention mechanism. H_E remains
uncertain when limited review cannot resolve it, even if H_T and H_R are supported.
Associations of component flags with WIS and model-by-condition interactions are
exploratory; they do not identify the effect of inserting a particular component.

LOC describes submitted code size, not maintainability: packages move code out of
the script. Report API errors by type, distinguishing invented APIs from version
mismatches, missing imports and argument errors. Diagnostic commands count as
reported checks, not evidence that fitting passed them.

## Secondary benchmarks and extensions

Before any extension runs, freeze its task subset, agent deployment, prompt,
seeds, sample size, resources and analysis. No secondary run count is committed
by the 240-run core allocation.

- **Gostic guidance:** on a prospectively allocated subset, supply a frozen copy
  of the paper alongside the broad scientific request and generic API reference.
  Compare with the same tools and request without the paper, accounting for paper
  input tokens and observed use. This tests the effect of providing methodological
  guidance, not whether the agent previously encountered the paper in training.
  It is distinct from the API documentation needed to operate software.
  To estimate whether guidance and components complement each other, cross paper
  provision with software condition; a paper-versus-package comparison alone cannot
  separate those effects or their interaction. Preserve omission/error, availability
  and recent-recovery outcomes. No extra core runs or human reviews are assumed.
- **Unrestricted tool choice (`no-spec`):** run a smaller matched subset on the
  same tasks/data to describe tools selected, execution, tokens, recovery and
  coverage of scientific requirements. Selected packages are outcomes, not
  randomised subgroups. No additional 18-submission human review is assumed.
- **EpiNow2:** audit the pinned package against the exact task specification and
  supplied ascertainment/delay information. Compare expert-written references
  and, where feasible, package-constrained agents on compatible single-stream
  tasks with the same inputs and outcomes. If an exact task is unsupported,
  report the coverage limitation; do not assign an agent failure or silently
  simplify the specification. A modified common task must be labelled separately.
  Implementing a new multi-stream capability is an extension-development study
  with its own expert and agent effort accounting.
- **Further questions:** unknown ascertainment, additional deployments, broad
  versus checklist-based scientific instructions, stress tests, common elapsed-time
  or cost limits and real-time cut-offs require their own prospective specifications.

The fixed-package benchmark explores how a more prescribed expert-informed
workflow compares with composable components. The core alone does not establish
an optimal level of software flexibility or the best use of expert time.

## Work required before registration and execution

| Requirement | Evidence required before confirmatory launch |
|---|---|
| Confirm scope and feasibility | Cost/runtime estimate for 240 generation runs, up to 1,360 refits and reference validation; explicit probe-allocation decision; model availability and lifecycle coverage through completion plus margin |
| Migrate domain condition | Pinned ComposableTuringIDModels environment, API bundle, component validation inventory and references; legacy labels preserved |
| Validate scientific specification | Independent trajectory family draws; instantaneous generator; proposed 1,000/day initial level and prehistory; stream-count/extinction checks; mean-one weekly factors and supplied ascertainment scale; scoring boundaries |
| Validate independent references | Julia/Turing and composable models with the same inputs, calibration criteria, inference diagnostics and development report |
| Separate development and evaluation | Prior-use audit of proposed master seed and distinct generation/held-out/diagnostic manifests; hashes and block allocation |
| Update prompts and code contract | Twelve broad current-R requests without methodological checklists; independent neutrality/equivalence audit; quantiles, withholding statuses/reasons, dated current-assessment summaries and method notes; parameterised refit pilot including delay location/support |
| Implement allocation and isolation | Drivers for ten blocks and both phases; tested filesystem/network restrictions; allowed-package enforcement; no truth or reference access |
| Stabilise execution | Preinstalled environments, resource/cache policy, adequate computation, fixed session/retry limits and blinded infrastructure-rerun rule |
| Reconcile usage records | Complete per-call token telemetry, cache/reasoning accounting, deduplication, failed-run totals and monetary/compute accounting |
| Implement and validate analysis | `evaluation/analyse.R` with interior H_R, all-attempt current/earlier/withheld categories, both lags, day-150 and recent recovery/uncertainty/availability, task-standardised probe response proportions, explicit block-level H_T, both token totals and cap-hit rates; clustered intervals and precision report under selection/failure |
| Freeze error assessment | Two reviewers with 20/30/40-minute task caps; Gostic-informed omission/error rubric, materiality table, accepted weekly-smoothing example and sampling; hidden checks and calibrated input probes with false-positive controls; validated delay-interface checks and separate failure classification |
| Agree practical effect benchmarks | Development-based recovery threshold and token-saving benchmark with rationale, before evaluation data or results are inspected |

The project lead coordinates these tasks; the documentation auditor and independent
reference checker provide recorded checks outside their own authorship where
possible. Recruit reviewers before freezing the rubric. Registration records
names, roles, package contributions and outstanding conflicts of interest.

This document remains authoritative. Existing tools, including
`evaluation/analyse.R`, detectors and drivers, remain development implementations
until they pass these checks. Record prospective amendments here; git history
preserves earlier versions.

## Reporting and limits

Publish allocation, development history, amendments, all original attempts and
infrastructure replacements, missing artefacts/telemetry, generation and held-out
success, current/earlier/withheld assessments and their dates/lags, conditional
current/interior/recent recovery and uncertainty, estimate availability and
withholding, token distributions, direct omission/error review results and
unresolved cases, detector calibration, diagnostics, runtime and costs. Preserve
unfavourable and inconclusive outcomes. Show all task/deployment/wording results;
choose example trajectories by a frozen rule rather than selecting persuasive
successes. Release prompts, environments, generator, harness, code and analysis
for independent replication.

Disclose contributions to ComposableTuringIDModels, EpiAware, EpiNow2 and other
benchmarked packages, and funding/research credits. Human review assesses code,
not its reviewability or the time an expert would need to make it safe to use.
There are no patient-level data. Real England surveillance data may illustrate a
workflow but have no known true R and cannot validate recovery.

Conclusions concern these components, documentation, agent deployments, tasks
and simulated processes. Ten matched blocks give limited precision; the small
human sample gives especially weak resolution of condition differences. No
independent external replication is included. Simulated data do not remove
training exposure to renewal-model structure. Selected wordings and a single
commercial model family limit generalisation. The simplified conditionally
independent observation streams and known delays/ascertainment do not establish
performance under realistic misspecification or uncertain surveillance inputs.
The renewal-based generator may favour software built around renewal models;
using the same generator across conditions does not remove that limitation.
A model can recover R well while containing errors that these tests miss, and
can implement the intended model correctly while recovering poorly. Neither
clinical usefulness nor an optimal human–AI research pipeline follows from this
experiment alone.

*Created: 2025-12-07. Earlier designs and development decisions remain in git history.*
