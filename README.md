# LLM Epidemiological Code Composition

How do software instructions and scaffolding — from unconstrained free choice, through Julia with a general-purpose probabilistic programming framework, to Julia with a domain-specific composable framework — affect the correctness, component fidelity, reliability, interpretability, reviewability, and cost of LLM-generated epidemic model code?

## Status

Phase 2 design in preparation. See `analysis_plan.md` for the full pre-registered protocol: three conditions on a scaffolding gradient (no-spec / julia / epiaware), four scenarios of increasing complexity, three Claude models (Haiku 4.5 / Sonnet 4.6 / Opus 4.7) with Qwen3-Coder-30B as a tertiary open-weight comparison, simulation-based evaluation with ground truth, eight outcome axes (four confirmatory, four descriptive), and pre-specified predictions with quantitative thresholds.

An earlier framing (2026-04 through 2026-07) described the conditions as a retrieval-versus-composition gradient. That framing was retired in August 2026 because neither mode is identifiable from model outputs. The current plan reports observable effects of scaffolding on observable code properties; see `analysis_plan.md` §Framing.

Reviewability is measured via injected-defect detection: the review coordinator plants known defects from `evaluation/mutation_catalogue.md` into a subset of samples, and expert reviewers try to detect them. See `evaluation/review_protocol.md`. This gives the reviewability outcome a ground-truth signal that does not depend on blinding (which is not achievable when language and package structure identify the condition on inspection).

Phase 1 ran under an earlier design (see `analysis_plan.md` → Study History) and was not carried through to analysis. Its artefacts have been removed; see git history for the original plan and tracked materials.

## Repository layout

```
analysis_plan.md                 # Pre-registered protocol (current draft)
data/                            # Real data (retained as secondary realism check)
evaluation/
  run_agentic.sh                 # Agentic run harness (Claude Code CLI)
  detectors.py                   # Automated component-correctness detectors
  mutation_catalogue.md          # Injected-defect catalogue for reviewability outcome
  review_protocol.md             # Expert review protocol
  generate_paraphrases.py        # LLM paraphrase generation for slots 02–04
prompts/                         # Base prompts (slot 01) and paraphrases; API docs bundles
reference_solutions/             # EpiAware and EpiNow2 references for sanity checks
runs/                            # Per-run output directories (agent code, conversation, metadata)
scripts/                         # Data download and utilities
setup/                           # Language environment setup (R, Python, Julia)
simulations/                     # DGP variants with truth/ and data/ subdirs
```

## Licence

MIT
