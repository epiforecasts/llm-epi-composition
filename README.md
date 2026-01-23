# LLM Epidemiological Code Composition

Can large language models write epidemiologically correct code for estimating the time-varying reproduction number (Rt)?

## Study Design

This study evaluates LLM-generated code for Rt estimation across:
- **2 models**: Claude Sonnet 4, Llama 3.1 8B
- **4 scenarios**: From basic Rt estimation to complex multi-stream models
- **5 framework conditions**: Stan, PyMC, Turing, EpiAware, plain R
- **3 runs each**: 120 total submissions

## Important Limitations

This study tests **zero-context, single-shot prompting** - a deliberately harsh baseline:

- No documentation or examples provided
- No iterative refinement
- No tool use or agentic behavior
- No access to framework codebases

This is **not** how practitioners would realistically use LLMs. Real-world use involves iteration, documentation in context, and error feedback. Results should be interpreted as a lower bound on capability.

See [Issue #1](https://github.com/epiforecasts/llm-epi-composition/issues/1) and [Issue #2](https://github.com/epiforecasts/llm-epi-composition/issues/2) for discussion.

## Repository Structure

```
├── prompts/                 # Scenario prompts sent to LLMs
│   ├── scenario_1a/         # Open method choice
│   ├── scenario_1b/         # Renewal equation specified
│   ├── scenario_2/          # Complex model (day-of-week, ascertainment)
│   └── scenario_3/          # Multiple data streams
├── experiments/             # LLM responses (120 submissions)
├── evaluation/              # Execution evaluation
│   ├── run_evaluation.R     # Evaluation script
│   └── results/             # Execution results
├── expert_review/           # Expert assessment materials
│   ├── README.md            # Reviewer instructions
│   ├── all_code.md          # All submissions (blinded)
│   └── scoresheet.csv       # Scoring spreadsheet
├── reference_solutions/     # Ground truth implementations
├── data/                    # Synthetic COVID-19 case data
└── analysis_plan.md         # Pre-registered analysis plan
```

## Scenarios

| Scenario | Description | Method |
|----------|-------------|--------|
| 1a | Estimate Rt from daily cases | Open choice |
| 1b | Estimate Rt using renewal equation | Specified |
| 2 | Rt with day-of-week effects, time-varying ascertainment, NegBin noise | Specified |
| 3 | Joint model of cases, hospitalisations, deaths with shared Rt | Specified |

## Expert Review

Expert reviewers assess each submission for epidemiological correctness:

1. **Method identification** (Scenario 1a): Renewal equation, Wallinga-Teunis, etc.
2. **Departures from reference**: List differences from gold standard
3. **Departure classification**:
   - A: Equivalent alternative
   - B: Minor error
   - C: Major error
   - D: Fundamental misunderstanding
4. **Overall assessment**: Acceptable / Minor issues / Major issues / Incorrect

See [`expert_review/README.md`](expert_review/README.md) for full instructions.

## Reproducing

### Run experiments
```bash
Rscript experiments/run_all.R
```

### Evaluate execution
```bash
Rscript evaluation/run_evaluation.R
```

### Generate review materials
```bash
Rscript expert_review/generate_review_materials.R
```

## License

MIT

## Citation

*Paper forthcoming*
