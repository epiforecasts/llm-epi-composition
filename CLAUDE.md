# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative design document (unregistered draft). Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Framing

The study reports observable effects of software scaffolding on the produced code. It does not claim to test retrieval versus composition — those are latent LLM modes that cannot be identified from outputs.

## Current design (plan revision 2026-09, unregistered draft)

- Three conditions: `no-spec` (unrestricted tool choice), `julia` (Turing API docs bundled), `composable` (Turing docs plus ComposableTuringIDModels.jl docs). `composable` replaces the earlier `epiaware` condition; legacy `epiaware` runs, prompts and references keep their label and are never pooled with `composable`.
- Four scenarios (1a, 1b, 2, 3). Commercial models: Haiku 4.5, Sonnet 4.6, Opus 4.7. Qwen3-Coder-30B is a secondary open-weight comparison (Qwen Code + vLLM on the LSHTM HPC stack, `~/code/dotfiles/lshtm-local-llm-stack.md`).
- Four paraphrases per (scenario, condition): slot 01 = LLM-drafted-and-edited by project authors; slots 02–04 = GPT-5, Gemini 2.5 Flash, Sonnet 4.5.
- Compact allocation per model: 240 canonical runs (4 paraphrases × 5 reruns on one dataset) + 180 sensitivity runs (5 variants incl. canonical × 3 reruns, paraphrase 01). 1260 commercial runs total. Claims are conditional on the frozen datasets.
- Three hypotheses, `composable` vs `julia`, averaged over six cells (scenarios 2/3 × three commercial models): H1 median RMSE (benchmark 0.03), H2 valid-output proportion (0.10), H3 injected-defect detection sensitivity (0.15). Other outcomes are descriptive.
- Simulation: individual-level branching process with Gamma–Poisson offspring (Lloyd-Smith). Sensitivity panel variants: short_gi, long_delay, extreme_dispersion, abrupt_change; strong_dow, high_asc_var, low_dispersion, sinusoidal_rt remain defined.

## Required before registration (see plan §Work required before registration)

- Migrate to ComposableTuringIDModels: API docs bundle, references, pinned `evaluation/julia_env/`, harness routing, detectors, `analyse.R` labels, `mutate.py` patterns.
- Replace the cohort-Rt generator with an instantaneous-Rt target; supply ascertainment information in the primary tasks; regenerate truth/data and select unused evaluation seeds.
- Implement H1–H3 estimands and intervals in `evaluation/analyse.R`; precision report.
- Recruit two expert reviewers and an independent detector validator.
- Regenerate stale paraphrases (slots 02–04) once prompts are revised; needs `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`.
- Raise the Anthropic account spend cap (hit ~$500 during the July adversarial matrix) before launch.

## Key files

- `evaluation/run_agentic.sh` — per-run harness (Claude Code CLI). Retries with `--resume` (`MAX_RETRIES=5`), waits up to `POST_AGENT_WAIT_MIN=60` minutes for backgrounded inference before retrying, uses `/proc/*/cwd` walk to detect and clean up escaped Julia subprocesses.
- `evaluation/run_agentic_qwen.sh` — Qwen harness.
- `evaluation/run_study.sh` — study driver.
- `evaluation/analyse.R` — analysis entry point.
- `evaluation/detectors.py` — automated component detectors.
- `evaluation/mutate.py`, `evaluation/mutation_catalogue.md` — injected defects for H3.
- `evaluation/review_protocol.md` — expert review protocol.
- `evaluation/generate_paraphrases.py` — paraphrase generator for slots 02–04.
- `evaluation/calibrate_reference.jl` — reference calibration.
- `evaluation/validate_api_docs.py` — LLM check that bundled API docs contain no worked solutions.

## Harness state (development pilots, legacy `epiaware` condition)

- Cold-run harness passed all 12 cells in the reliability validation (60/60 Sonnet 4.6). scenario_3/epiaware took ~3 hours but succeeded under the retry + wait-for-inference logic.
- Warm-daemon variant (`USE_DAEMON=1`) is kept but not used for confirmatory runs; cold is the default.

## Project gotchas

(EpiAware notes below apply to legacy development code until the migration lands.)

- **EpiAware API (currently pinned in `evaluation/julia_env/`).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **EpiAware sampling needs `using ReverseDiff` and `using LogDensityProblemsAD`.** `apply_method` defaults to `AutoReverseDiff(compile=true)`, but the `LogDensityProblemsAD` reverse-diff extension only activates when both are loaded. Without them, sampling fails with `MethodError: no method matching ADgradient(::Val{:ReverseDiff}, ...)`. The reference solutions and the EpiAware API docs (`prompts/epiaware_api_docs.md`) include this.
- **Experiment isolation.** Agentic runs receive only the paraphrase prompt + observed data copied into a `mktemp` working directory. For simulation runs, copy only `simulations/{variant}/rep_{rr}/data/`; the `truth/` subdirectory must never enter the agent's sandbox. `run_agentic.sh` filters streams by scenario (1a/1b/2 → cases only; 3 → all three) and copies the condition-specific docs bundle (`turing_api_docs.md` for julia, `epiaware_api_docs.md` for epiaware).

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
