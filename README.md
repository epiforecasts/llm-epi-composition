# LLM Epidemiological Code Composition

Can large language models compose correct models for estimating the time-varying reproduction number (Rt)?

## Status

Phase 2 design in preparation. See `analysis_plan.md` for the full pre-registered protocol, including conditions, scenarios, simulation-based evaluation with ground truth, randomisation, blinding, and pre-specified predictions.

Phase 1 ran under an earlier design (see `analysis_plan.md` → Study History) and was not carried through to analysis. Its artefacts have been removed; see git history for the original plan and tracked materials.

## Repository layout

```
analysis_plan.md        # Pre-registered protocol (revised 2026-04)
data/                   # Real data (retained as secondary realism check)
evaluation/             # Agentic run harness (run_agentic.sh)
prompts/                # API docs bundles; scenario prompts to be written for phase 2
reference_solutions/    # EpiAware and EpiNow2 references for sanity checks
scripts/                # Data download and utilities
setup/                  # Language environment setup (R, Python, Julia)
simulations/            # To be populated: DGP variants with truth/ and data/ subdirs
```

## Licence

MIT
