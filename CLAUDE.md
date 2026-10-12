# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative design document (unregistered draft). Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Framing

The aim is to test whether validated epidemiological modelling components help
agents recognise and handle pitfalls in assessing current transmission, and reduce token
use. Assess omissions as well as incorrect implementation. This tests one location
of human expertise in an AI analysis pipeline: reusable software supplied upstream.
The intervention includes tools, instructions and documentation. Outputs do not
identify retrieval versus composition, or the best use of live human supervision.

## Current design (revision 2026-10-12, unregistered draft)

- Core conditions: `julia` (Julia/Turing, generic packages only) and `composable`
  (same environment plus ComposableTuringIDModels.jl and its documentation).
  Task inputs and shared documentation are identical; domain documentation may
  supply methodological cues as part of the intervention. At least one domain
  component must contribute to the fitted model; importing it alone is non-use.
  Legacy `epiaware` runs remain separate.
- Tasks: 1b (simple cases), 2 (richer case observations) and 3 (multiple streams).
  All ask broadly for current instantaneous R, its trajectory, uncertainty and
  limitations, with external GI, delay and report-date ascertainment inputs.
  Do not prescribe a renewal model,
  delay adjustment, likelihood or component checklist. Weekly variation applies
  to the 2/3 generator; agents choose its handling. Legacy task IDs do not imply
  the old scientific instructions. Track protocol versions explicitly.
- Proposed allocation: 2 conditions × 3 tasks × 2 deployments (Sonnet/Haiku,
  candidate 5.5 versions, exact choices and lifecycle coverage to freeze) × 2
  wordings × 10 outbreak blocks = 240 agent runs. Generation and held-out data
  have independently drawn R trajectories. A third dataset supports data-use checks.
  There are 240 primary refits and up to 1,120 additional repeat/probe refits, whose
  cost and calibration require validation before registration. Blocks remain
  matched across conditions, deployments and wordings; refits are not independent
  implementations. Clean refit directories exclude generation outputs/posteriors.
- H_T: lower total agent tokens, including documentation, repairs and failures.
  H_T has explicit paired block means and a secondary total excluding cache reads;
  report cap-hit rates. H_R: lower interior WIS (days 25–125), averaged over six
  task–deployment cell medians. The operational target is day-150 R. All-attempt
  current/earlier/withheld assessment categories, headline/latest-estimate lags and
  current/recent recovery, uncertainty and availability are central scientific
  outcomes with descriptive inference. An earlier estimate presented as current
  is substantive error; a clearly dated earlier answer can be defensible but limited.
  Interior H_R alone cannot establish better current assessment. Recent days
  144–150 and withholding do not determine H_R eligibility. Execution failures and
  interior withholding still cause selection. These are the two confirmatory
  contrasts. Clustered interval coverage requires validation; sign-flip exactness
  is not implied by a zero mean contrast or by random scheduling. Do not reuse the
  previous fixed-dataset bootstrap or Hoeffding interval.
- H_E: fewer substantive omissions and implementation errors, assessed directly
  by two modellers reviewing the same 18 randomly sampled implementations (three per task–condition
  cell). Budget: 10 hours each, using 20/30/40-minute caps for tasks 1b/2/3 plus
  familiarisation. This is coarse secondary estimation, with unresolved cases
  explicit; detector calibration is a second use.
  Gostic informs the hidden rubric; correct package defaults and defensible methods
  need no named adjustment. Review method notes and justification of withholding.
  Input-use probes support selected omission checks but unchanged estimates are
  not universally proof of non-use. Freeze materiality rules and examples first.
- `no-spec`, EpiNow2, Gostic guidance, explicit scientific checklists, other
  deployments and stress panels are secondary extensions with prospective
  allocation. Unsupported package features are coverage limits; do not silently
  modify tasks or score them as agent failures.
- Preinstalled, tested environments and adequate inference time are prerequisites.
  Retain the developed session/retry/wait policy, preserve failures, and distinguish
  infrastructure errors from agent code errors. Enforce actual agent isolation.
- **Development only:** the current generator produces cohort R, existing prompts
  use older output/ascertainment requirements, and legacy EpiAware references and
  harness settings need migration. The plan specifies nine R quantiles and an
  instantaneous daily renewal generator; these changes are not implemented yet.
  The new output contract also includes estimated/withheld statuses, reasons and
  a short method note, plus a dated `current_assessment.json` regenerated on each
  refit and referencing a trajectory row or declaring withholding. Declared
  withholding is distinct from an invalid output.

## Required before registration

See the plan's work table for the full requirements. In particular:

- Validate the instantaneous generator, boundaries, task-specific observations,
  known ascertainment inputs on a mean-one weekly scale, trajectory family,
  proposed incidence of 1,000/day at day 0, prehistory, stream counts, independent
  references and distinct generation/held-out/diagnostic evaluation seeds.
- Pin and validate ComposableTuringIDModels, migrate references/API bundles and
  record component validation evidence; preserve legacy identities.
- Revise and independently audit 12 broad current-R request templates and shared/domain
  bundles for neutrality and equal task information; domain-documentation cues
  are part of the intervention. Preserve the hidden
  evaluator rubric. Freeze system/tool context and prevent inherited run memory.
- Implement paired-block allocation, source freezing, parameterised refit interface,
  isolated execution and infrastructure-rerun rules in the drivers.
- Reconcile per-call token records, including caching, reasoning and resumed
  sessions; implement the token and held-out recovery analyses in `analyse.R`,
  including current-assessment categories, recency and conditional day-150 scores.
- Validate clustered interval coverage and precision at ten blocks, handle sparse
  successes explicitly, and agree practical effect benchmarks on development data.
- Recruit two scientific reviewers and freeze the omission/error rubric, sampling and targeted
  checks. Automated flags do not replace direct human evidence about errors.
- Update runtime/cost estimates for 240 agent runs, up to 1,360 refits and reference
  validation; verify provider capacity and lifecycle coverage before launch.

## Key files

- `evaluation/run_agentic.sh` — per-run harness (Claude Code CLI). Retries with `--resume` (`MAX_RETRIES=5`), waits up to `POST_AGENT_WAIT_MIN=60` minutes for backgrounded inference before retrying, uses `/proc/*/cwd` walk to detect and clean up escaped Julia subprocesses.
- `evaluation/run_agentic_qwen.sh` — Qwen harness.
- `evaluation/run_study.sh` — study driver.
- `evaluation/analyse.R` — analysis entry point.
- `evaluation/detectors.py` — automated component detectors.
- `evaluation/mutate.py`, `evaluation/mutation_catalogue.md` — injected defects; not used in the current design.
- `evaluation/review_protocol.md` — direct scientific error review and detector calibration (18 implementations, ≤10 h per reviewer).
- `evaluation/generate_paraphrases.py` — development paraphrase generator; core uses slots 01/02.
- `evaluation/calibrate_reference.jl` — reference calibration.
- `evaluation/validate_api_docs.py` — LLM check that bundled API docs contain no worked solutions.

## Harness state (development pilots, legacy `epiaware` condition)

- Cold-run harness passed all 12 cells in the reliability validation (60/60 Sonnet 4.6). scenario_3/epiaware took ~3 hours but succeeded under the retry + wait-for-inference logic.
- Warm-daemon variant (`USE_DAEMON=1`) is kept but not used for confirmatory runs; cold is the default.

## Project gotchas

(EpiAware notes below apply to legacy development code until the migration lands.)

- **EpiAware API (currently pinned in `evaluation/julia_env/`).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **EpiAware sampling needs `using ReverseDiff` and `using LogDensityProblemsAD`.** `apply_method` defaults to `AutoReverseDiff(compile=true)`, but the `LogDensityProblemsAD` reverse-diff extension only activates when both are loaded. Without them, sampling fails with `MethodError: no method matching ADgradient(::Val{:ReverseDiff}, ...)`. The reference solutions and the EpiAware API docs (`prompts/epiaware_api_docs.md`) include this.
- **Development data routing.** `run_agentic.sh` copies prompts, observed files and condition-specific documentation into a temporary working directory, filtering streams by scenario (1a/1b/2 → cases only; 3 → all three). Copy only `simulations/{variant}/rep_{rr}/data/`, never `truth/`. Temporary-directory routing does not enforce isolation: filesystem and network restrictions must also prevent access to truth, references, held-out data, other runs and unapproved domain packages before confirmatory execution.

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
