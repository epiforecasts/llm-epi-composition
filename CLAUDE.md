# llm-epi-composition

## Primary reference

`analysis_plan.md` is the authoritative pre-registered design document. Read it before making substantive changes. Revise it in place (git history preserves versions); do not create parallel protocol files.

## Current status

Phase 2 design committed; phase 1 artefacts removed. In place:

- Simulation generator: 8 variants × 20 replicates from an individual-level Lloyd-Smith Bellman-Harris BP (`simulations/generate.jl`).
- Reference sanity check passed on canonical (RMSE 0.09, coverage ≈ 1.0).
- Twelve base prompts under `prompts/{scenario}/paraphrases/{condition}/01.md`.
- Turing.jl + EpiAware API docs under `prompts/{turing,epiaware}_api_docs.md`.
- Per-run isolation in `evaluation/run_agentic.sh`; study driver in `evaluation/run_study.sh`.
- Automated correctness detectors in `evaluation/detectors.py`.
- LLM paraphrase generator skeleton in `evaluation/generate_paraphrases.py`.

Outstanding:

- LLM-paraphrase wave: run `evaluation/generate_paraphrases.py` after `pip install anthropic openai google-genai` and setting `OPENAI_API_KEY`, `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`. Slot 03 → GPT-5, slot 04 → Gemini 2.5 Flash, slot 05 → Claude Sonnet 4.5. Slot 02 is the human-written paraphrase wave by a project member; brief at `prompts/slot02_brief.md`.
- **Wait-for-output pilot finding**: pilots showed the agent (Sonnet 4.6) writes the Julia script and starts inference but exits before sampling completes, leaving `outputs/rt_estimates.csv` empty. Base prompts have been amended with an explicit "run synchronously and verify output exists before ending response" instruction. If a future agent still backgrounds the job, consider adding a `wait` step to `run_agentic.sh` that re-prompts the agent if the output file is missing.
- **Paraphrase state**: only `scenario_1a/no-spec/{03,04,05}.md` have been regenerated against the current base prompts (the ones with explicit uncertainty-scoring language). The other 11 cells × 3 LLM slots = 33 files are stale and still match the older "optional but encouraged" wording. Re-run the paraphrase generator once API keys are restored to bring them back in sync.
- Temperature randomisation: Claude Code CLI does not expose `temperature`; the temperature axis of the randomisation requires invoking the Anthropic API directly. Currently absent from `evaluation/run_agentic.sh` and `evaluation/run_study.sh`; will need a separate API-mode runner before the temperature axis can be exercised.

## Project gotchas

- **EpiAware API (v0.2.0).** `EpiData(gen_int_pmf, exp)` positional or `EpiData(gen_distribution=Gamma(...))` keyword. Transformation must be `exp`; `identity` fails Pathfinder on negative Rt values. Reference solutions already use `exp`.
- **EpiAware sampling needs `using ReverseDiff`.** `apply_method` defaults to `AutoReverseDiff(compile=true)`, but the `LogDensityProblemsADReverseDiffExt` package extension only activates when both `ReverseDiff` and `LogDensityProblemsAD` are loaded. Without these `using` statements, sampling fails with `MethodError: no method matching ADgradient(::Val{:ReverseDiff}, ...)`. The reference solutions and the EpiAware API docs (`prompts/epiaware_api_docs.md`) include the requirement; LLM submissions in the EpiAware condition must follow it.
- **Experiment isolation.** Agentic runs receive only prompt + observed data copied into a `mktemp` working directory. For simulation runs, copy only `simulations/{variant}/rep_{rr}/data/`; the `truth/` subdirectory must never enter the agent's sandbox. `evaluation/run_agentic.sh` filters streams by scenario (1a/1b/2 → cases only; 3 → all three) and copies the condition-specific docs bundle (`turing_api_docs.md` for julia, `epiaware_api_docs.md` for epiaware).

## Writing style for plan/protocol documents

Plan documents are factual references. Avoid "X, but stronger than Y" or "this approach, though imperfect" framings. State what the design is; do not argue for its strength or justify it by comparison to weaker alternatives.
