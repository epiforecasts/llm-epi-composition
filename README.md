# LLM Epidemiological Code Composition

Do validated epidemiological modelling components help AI agents avoid the
methodological pitfalls of estimating transmission, while using fewer tokens?

## Status

The [analysis plan](analysis_plan.md) is an unregistered draft, revised on
2026-10-12. It tests one way of supplying human expertise to AI: reusable modelling
components. The controlled core compares Julia/Turing with Julia/Turing plus
[ComposableTuringIDModels.jl](https://composableturingidmodels.epiaware.org/dev/),
using three tasks spanning simple cases, richer case observations and multiple
surveillance streams. Agents receive broad requests to assess current instantaneous
R and provide its trajectory, uncertainty and limitations, with data and external
inputs, and choose the scientific analysis. Gostic's
practical considerations inform the hidden evaluation rubric; core prompts do
not prescribe delay adjustments, likelihoods or a renewal model.

The proposed allocation is 240 code-generation runs: two software conditions,
three tasks, two agent deployments, two prompt wordings and ten independent
outbreak blocks. Each block has separate generation and unseen evaluation data.
Frozen code is refitted on unseen data with independently drawn transmission
trajectories, in clean directories without cached fitted results. Total agent
tokens and held-out weighted interval score are the two confirmatory outcomes.
H_R scores interior days 25–125 as a technical recovery check. The central practical
question is what R is now, at the observation cut-off. Report current, explicitly
dated earlier and withheld assessments across all attempts, alongside current/recent
recovery, uncertainty, availability and estimate age. These scientific outcomes
are descriptive; interior recovery alone cannot establish better current assessment.
Execution and fitting diagnostics accompany scores.

The proposed input-use checks bring machine-only refits to at most 1,360, including
unchanged repeats and data/delay/GI/ascertainment probes. Their computing feasibility
and calibration must be established before registration. No-response flags are
distinct from confirmed scientific errors.

Two independent modellers directly assess scientific implementation errors in
18 sampled implementations for omissions and incorrect handling, within ten hours
per reviewer, using task caps of 20/30/40 minutes. The same reviews also calibrate
automated checks; the small sample permits a coarse error comparison.
See the [review protocol](evaluation/review_protocol.md). Smaller unrestricted
choice, EpiNow2 and supplied-Gostic-guidance benchmarks are secondary and require
prospective allocation and compatibility checks.

Existing code and prompts still implement parts of the development design.
Registration requires a validated instantaneous-R generator, known ascertainment
inputs, pinned and preinstalled environments, isolated agent execution, the
frozen-code refit interface, complete token accounting and validated analysis.
Migration from EpiAware remains pending. Legacy `epiaware` runs keep their identity.

Phase 1 was not carried through to analysis. Tracked materials remain in git
history; its gitignored outputs were not retained. Phase 2 pilots and reference
calibration must be documented as development work preceding registration.

## Repository layout

```
analysis_plan.md                 # Authoritative analysis plan (unregistered draft)
data/                            # Real data for a possible secondary demonstration
evaluation/
  run_agentic.sh                 # Development agent harness (Claude Code CLI)
  detectors.py                   # Automated component flags; adaptation pending
  mutation_catalogue.md          # Injected-defect catalogue; outside the core
  review_protocol.md             # Direct scientific error review and calibration
  generate_paraphrases.py        # Development paraphrase generator
prompts/                         # Prompts and API bundles; revision pending
reference_solutions/             # Development references; migration pending
runs/                            # Code, conversations, outputs and metadata
scripts/                         # Data download and utilities
setup/                           # Language environment setup
simulations/                     # Development generator and data/truth directories
```

## Licence

MIT
