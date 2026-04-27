# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative pre-registered design document. Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Current status

Phase 2 design committed; phase 1 artefacts removed. Simulation generator produces data + truth for 8 variants × 20 replicates from an individual-level Lloyd-Smith Bellman-Harris BP under `simulations/`. Reference EpiAware sanity check on canonical passed (RMSE 0.09, coverage ≈ 1.0). Next items from the plan:

- Phase 2 prompts under the information-provision rules in Protocol → Prompt construction (old prompts under `prompts/scenario_*/` are obsolete — no-spec/julia/epiaware axis, no disease label).
- `evaluation/run_agentic.sh` revision for simulation runs: copy from `simulations/{variant}/rep_{rr}/data/`, filter streams by scenario (1a/1b/2 → cases only; 3 → all three).
- Reference solutions in `reference_solutions/*.jl` are phase 1 artefacts still reading from `data/cases.csv`; adapt them to read `simulations/canonical/rep_01/data/cases.csv` for any phase-2 use beyond the canonical sanity check we already ran.
- Automated correctness detectors per Evaluation → Diagnostic: Automated correctness detectors.
- Prompt paraphrase and temperature randomisation harness.

## Project gotchas

- **EpiAware API (v0.2.0).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **EpiAware sampling needs `using ReverseDiff`.** `apply_method` defaults to `AutoReverseDiff(compile=true)`, but the `LogDensityProblemsADReverseDiffExt` package extension only activates when both `ReverseDiff` and `LogDensityProblemsAD` are loaded. Without these `using` statements, sampling fails with `MethodError: no method matching ADgradient(::Val{:ReverseDiff}, ...)`. The reference solutions and the EpiAware API docs (`prompts/epiaware_api_docs.md`) include the requirement; LLM submissions in the EpiAware condition must follow it.
- **Experiment isolation.** Agentic runs receive only prompt + observed data copied into a `mktemp` working directory. For simulation runs, copy only `simulations/{variant}/data/`; the `truth/` subdirectory must not enter the agent's sandbox. See `evaluation/run_agentic.sh` for the existing pattern.

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
