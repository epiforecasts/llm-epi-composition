# LLM Epidemiological Code Composition

How do unrestricted tool choice, Julia with Turing, and Julia with ComposableTuringIDModels.jl compare in statistical performance, implementation errors, execution success and reviewability when LLM agents generate epidemic model code?

## Status

Phase 2 is in development. The [analysis plan](analysis_plan.md) is a revised draft, not a registered protocol. It compares unrestricted tool choice, Julia with Turing, and Julia with [ComposableTuringIDModels.jl](https://composableturingidmodels.epiaware.org/dev/) across four epidemiological tasks. The new `composable` condition replaces EpiAware in the planned study; existing `epiaware` prompts, references and harness settings still need migration and validation. Legacy development runs retain their original identity.

The scientific review identified changes required before confirmatory runs: align the simulated reproduction number with the estimation target, constrain ascertainment in primary recovery tasks, distinguish process from observation variation, and freeze independent evaluation data and analysis rules. Existing code and prompts still implement the development design. The plan retains the compact allocation and specifies three hypotheses on recovery, valid outputs and defect detection, with explicit effect benchmarks and inference rules. Additional datasets and a common time cap are optional extensions.

Reviewability is assessed through injected-defect detection. The [review protocol](evaluation/review_protocol.md) now specifies the generated-code pool, independent clean-code checks and rules for unresolved or invalid H3 inference. Package-specific operations in the [mutation catalogue](evaluation/mutation_catalogue.md) still require validation after migration.

Phase 1 was not carried through to analysis. Tracked materials remain in git history; its gitignored run outputs were not retained. Phase 2 pilots and reference calibration must be documented as development work preceding registration.

## Repository layout

```
analysis_plan.md                 # Analysis protocol (unregistered draft)
data/                            # Real data (retained as secondary realism check)
evaluation/
  run_agentic.sh                 # Agentic run harness (Claude Code CLI)
  detectors.py                   # Automated component-correctness detectors
  mutation_catalogue.md          # Injected-defect catalogue for reviewability outcome
  review_protocol.md             # Expert review protocol
  generate_paraphrases.py        # LLM paraphrase generation for slots 02–04
prompts/                         # Base prompts (slot 01) and paraphrases; API docs bundles
reference_solutions/             # Development references; EpiAware migration pending
runs/                            # Per-run output directories (agent code, conversation, metadata)
scripts/                         # Data download and utilities
setup/                           # Language environment setup (R, Python, Julia)
simulations/                     # DGP variants with truth/ and data/ subdirs
```

## Licence

MIT
