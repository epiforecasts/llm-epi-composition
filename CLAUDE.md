# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative pre-registered design document. Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Current status

Phase 2 design committed; phase 1 artefacts removed. No phase 2 code yet. Next items from the plan:

- Simulation generator producing `simulations/{variant}/{truth,data}/` per the Data generation procedure section
- Reference-solution sanity check on canonical DGP (EpiAware must recover true Rt within tolerance before any LLM is queried)
- Phase 2 prompts under the information-provision rules in Protocol → Prompt construction
- `evaluation/run_agentic.sh` revision for simulation runs (copy from `simulations/{variant}/data/`)
- Automated correctness detectors per Evaluation → Diagnostic: Automated correctness detectors
- Prompt paraphrase and temperature randomisation harness

## Project gotchas

- **EpiAware API (v0.2.0).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **Experiment isolation.** Agentic runs receive only prompt + observed data copied into a `mktemp` working directory. For simulation runs, copy only `simulations/{variant}/data/`; the `truth/` subdirectory must not enter the agent's sandbox. See `evaluation/run_agentic.sh` for the existing pattern.

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
