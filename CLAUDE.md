# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative pre-registered design document. Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Framing (2026-08)

The study reports observable effects of software scaffolding on the produced code. It does not claim to test retrieval versus composition — those are latent LLM modes that cannot be identified from outputs. Earlier drafts framed the conditions as a composition gradient; that framing is retired. See `analysis_plan.md` §Framing.

## Current design

- Three conditions on a scaffolding gradient: `no-spec` (unconstrained), `julia` (Turing.jl docs bundled), `epiaware` (EpiAware.jl composable primitives docs bundled).
- Four scenarios (1a, 1b, 2, 3) of increasing modelling complexity.
- Three Claude models: Haiku 4.5, Sonnet 4.6, Opus 4.7. Qwen3-Coder-30B as a tertiary open-weight comparison via the LSHTM HPC stack (`~/code/dotfiles/lshtm-local-llm-stack.md`).
- Four paraphrases per (scenario, condition): slot 01 = LLM-drafted-and-edited by project authors; slots 02–04 = GPT-5, Gemini 2.5 Flash, Sonnet 4.5.
- Compact sample sizes: 4 paraphrases × 1 replicate × 5 runs = 20 primary runs per cell, plus 3 runs on 4 adversarial variants = 15 adversarial runs per cell. 420 runs per model, ~1260 total across three Claude models. ~$2700 API spend on Anthropic credits.
- Simulation DGP: individual-level Lloyd-Smith Bellman-Harris branching process, canonical + 4 confirmatory adversarial variants (short_gi, long_delay, extreme_dispersion, abrupt_change); four others (strong_dow, high_asc_var, low_dispersion, sinusoidal_rt) remain defined for follow-up.
- Eight outcome axes: four confirmatory (statistical correctness, component correctness, interpretability, reviewability) and four descriptive (execution reliability, robustness, maintainability, epistemic quality).

## Key files

- `evaluation/run_agentic.sh` — per-run harness (Claude Code CLI). Retries with `--resume` (`MAX_RETRIES=5`), waits up to `POST_AGENT_WAIT_MIN=60` minutes for backgrounded inference before retrying, uses `/proc/*/cwd` walk (not command-line matching) to detect and clean up escaped Julia subprocesses.
- `evaluation/run_study.sh` — study driver over (scenario, condition, paraphrase, variant, replicate, model, run).
- `evaluation/detectors.py` — twelve automated component-correctness detectors.
- `evaluation/mutation_catalogue.md` — catalogue of ~15 injected defects for the reviewability outcome.
- `evaluation/review_protocol.md` — expert review protocol.
- `evaluation/generate_paraphrases.py` — LLM paraphrase generator for slots 02–04.
- `evaluation/calibrate_reference.jl` — reference-solution calibration script.
- `evaluation/validate_api_docs.py` — LLM check that the bundled API docs comply with the MWK operational rule.

## Harness state (verified via pilots)

- Cold-run harness passes all 12 cells in the reliability validation (60/60 Sonnet 4.6). scenario_3/epiaware takes longest (~3 hours) but succeeds reliably under the current retry + wait-for-inference logic.
- 3/epiaware run_04 was the pilot that confirmed the wait-for-inference fix: 167 min wall, 2 retries, 2 post-agent waits, valid CSV.
- Warm-daemon variant (opt-in via `USE_DAEMON=1`) is kept in the harness but not used for confirmatory runs. Pilots showed it helps some cells but regresses others; cold is the study default.

## Outstanding

- **Regenerate 33 stale paraphrase files** (slots 02–04 across 11 cells; scenario_1a/no-spec is current). Requires `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY` in the environment, then `python evaluation/generate_paraphrases.py`.
- **Missing scripts referenced by the plan**: `evaluation/analyse.R` (bootstrap analysis), `evaluation/mutate.py` (injects defects from the catalogue), `evaluation/run_agentic_qwen.sh` (Qwen tertiary harness). Need writing before pre-registration.
- **Recruit** two expert reviewers and one detector validator.
- **Pin pre-registration commit hash** in README and external registry (OSF or equivalent).
- **Anthropic account monthly spend limit** hit at ~$500 during the July adversarial matrix. Raise the account cap to at least $3000 (ideally $5000) before launching the main study.

## Project gotchas

- **EpiAware API (currently pinned in `evaluation/julia_env/`).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **EpiAware sampling needs `using ReverseDiff` and `using LogDensityProblemsAD`.** `apply_method` defaults to `AutoReverseDiff(compile=true)`, but the `LogDensityProblemsAD` reverse-diff extension only activates when both are loaded. Without them, sampling fails with `MethodError: no method matching ADgradient(::Val{:ReverseDiff}, ...)`. The reference solutions and the EpiAware API docs (`prompts/epiaware_api_docs.md`) include this.
- **Experiment isolation.** Agentic runs receive only the paraphrase prompt + observed data copied into a `mktemp` working directory. For simulation runs, copy only `simulations/{variant}/rep_{rr}/data/`; the `truth/` subdirectory must never enter the agent's sandbox. `run_agentic.sh` filters streams by scenario (1a/1b/2 → cases only; 3 → all three) and copies the condition-specific docs bundle (`turing_api_docs.md` for julia, `epiaware_api_docs.md` for epiaware).

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
