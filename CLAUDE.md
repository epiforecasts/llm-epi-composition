# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative pre-registered design document. Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Current status

Phase 2 design committed; phase 1 artefacts removed. Simulation generator produces data + truth for 8 variants x 3 replicates under `simulations/`. Next items from the plan:

- Reference-solution sanity check on canonical DGP (EpiAware must recover true Rt within tolerance before any LLM is queried). Reference solutions in `reference_solutions/*.jl` are phase 1 artefacts still reading from `data/cases.csv`; adapt them to read `simulations/canonical/rep_01/data/cases.csv` and validate recovery.
- Phase 2 prompts under the information-provision rules in Protocol → Prompt construction (old prompts under `prompts/scenario_*/` are obsolete — no-spec/julia/epiaware axis, no disease label).
- `evaluation/run_agentic.sh` revision for simulation runs: copy from `simulations/{variant}/rep_{rr}/data/`, filter streams by scenario (1a/1b/2 → cases only; 3 → all three).
- Automated correctness detectors per Evaluation → Diagnostic: Automated correctness detectors.
- Prompt paraphrase and temperature randomisation harness.

## Project gotchas

- **EpiAware API (v0.2.0).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **Experiment isolation.** Agentic runs receive only prompt + observed data copied into a `mktemp` working directory. For simulation runs, copy only `simulations/{variant}/data/`; the `truth/` subdirectory must not enter the agent's sandbox. See `evaluation/run_agentic.sh` for the existing pattern.

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
