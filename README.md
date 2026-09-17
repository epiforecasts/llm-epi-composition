# LLM Epidemiological Code Composition

How do unrestricted tool choice, Julia with Turing, and Julia with ComposableTuringIDModels.jl compare in statistical performance, implementation errors and execution success when LLM agents generate epidemic model code?

## Status

Phase 2 is in development. The [analysis plan](analysis_plan.md) is an unregistered draft. It compares unrestricted tool choice, Julia with Turing, and Julia with [ComposableTuringIDModels.jl](https://composableturingidmodels.epiaware.org/dev/) across four epidemiological tasks. The new `composable` condition replaces EpiAware in the planned study; existing `epiaware` prompts, references and harness settings still need migration and validation. Legacy development runs retain their original identity.

Before confirmatory runs, the plan requires aligning the simulated reproduction number with the estimation target, constraining ascertainment in primary recovery tasks, distinguishing process from observation variation, and freezing independent evaluation data and analysis rules. Existing code and prompts still implement the development design. The plan retains the compact allocation and specifies two hypotheses on probabilistic recovery and valid outputs, with explicit effect benchmarks and inference rules. Additional datasets and a common time cap are optional extensions.

Implementation errors are measured by automated detectors, validated against a single human reviewer on 18 submissions within a 10-hour budget; see the [review protocol](evaluation/review_protocol.md).

Phase 1 was not carried through to analysis. Tracked materials remain in git history; its gitignored run outputs were not retained. Phase 2 pilots and reference calibration must be documented as development work preceding registration.

## Repository layout

```
analysis_plan.md                 # Analysis protocol (unregistered draft)
data/                            # Real data (retained as secondary realism check)
evaluation/
  run_agentic.sh                 # Agentic run harness (Claude Code CLI)
  detectors.py                   # Automated component-correctness detectors
  mutation_catalogue.md          # Injected-defect catalogue (not used in current design)
  review_protocol.md             # Detector validation review protocol
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
