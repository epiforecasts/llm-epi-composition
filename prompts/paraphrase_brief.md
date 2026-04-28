# Brief for the third-party paraphraser

You have been asked to write paraphrases of task descriptions used in a study evaluating code-writing AI assistants. The information here is the entirety of what you have access to about the project. This is deliberate: we want your paraphrases to reflect a reader who is *not* aware of the study's hypotheses or design.

## What you do

For each task description provided to you, write **two alternative versions**. Each alternative must:

1. **Preserve every factual statement.** This includes:
   - Data file paths and column names exactly as given.
   - Numerical parameters exactly as given (e.g. "Gamma distribution, mean 5.5 days, standard deviation 2 days").
   - Distributional families exactly as given (e.g. "log-normal", "negative binomial").
   - Any framework or language constraint exactly as given (e.g. "Use Julia").
   - The output file path and its required columns exactly as given.
   - Every structural feature mentioned about the data (reporting delay, weekly cycle, time-varying ascertainment, multi-stream coupling, overdispersion, etc.).
   - Any reference to docs files exactly as given.
2. **Vary the wording, ordering, headings, sentence structure, and tone.** Do not write a near-copy.
3. **Add nothing.** No examples, hints, advice, methodological suggestions, strategies, definitions of terms not in the original, or worked solutions.
4. **Reinterpret nothing.** Do not paraphrase a fact in a way that changes its meaning or scope.

The two alternatives you write should differ from each other meaningfully (not just be slight rewordings of the same wording).

## How the output is delivered

Each base prompt is in a file like `01.md`. You write your two alternatives as `03.md` and `04.md` in the same directory. The repository structure is:

```
prompts/
  scenario_1a/paraphrases/no-spec/01.md   ← base
  scenario_1a/paraphrases/no-spec/03.md   ← your alt 1
  scenario_1a/paraphrases/no-spec/04.md   ← your alt 2
  scenario_1a/paraphrases/julia/...
  scenario_1a/paraphrases/epiaware/...
  scenario_1b/paraphrases/...
  scenario_2/paraphrases/...
  scenario_3/paraphrases/...
```

12 base files (4 scenarios × 3 conditions) → 24 paraphrases total from you.

## API docs validation (separate task)

We provide two API reference files to the agents:

- `prompts/turing_api_docs.md` — Turing.jl API
- `prompts/epiaware_api_docs.md` — EpiAware.jl API

These are intended to be **API-level only**: function signatures, types, arguments, return types, and brief primitive-level usage snippets. They must **not** include end-to-end examples that solve any of the task descriptions in `prompts/scenario_*/paraphrases/*/01.md`, nor tutorials walking through full Rt-estimation model construction.

Please read both files and flag any content that, in your judgment as an infectious-disease modeller, constitutes a worked example or violates the API-level constraint. Note the file and the section. We will edit the docs based on your findings.

## What you should not have access to

You should not have, and should not request, the analysis plan, the predictions document, the simulation code, the simulation truth, the reference solutions, or any other study artefact beyond what is listed above. If you are unsure whether a piece of information is in scope, ask for clarification on the rule rather than reading the artefact.

## Acknowledgement

Your contribution will be acknowledged in any resulting publication. We thank you for the time.
