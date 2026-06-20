# Brief for the slot-02 paraphraser

This is a ~90-minute task. You are not the project lead.

## Background

The base prompts at slot 01 were drafted with LLM assistance and edited; slots 03, 04, and 05 are LLM-generated paraphrases by GPT-5, Gemini, and Claude. Without a human paraphrase wave the wording-sensitivity arm of the study never samples outside the LLM-prose register that the evaluator (Claude) is itself trained on. Slot 02 (yours) is that wave. You will be acknowledged as the slot-02 paraphraser in the paper.

Please do not read `analysis_plan.md` or any document describing the study's hypotheses or analyses. The base prompts and this brief are enough. We would like the paraphrase to come from someone who has not been primed on what the analysis is looking for.

## What to do

Twelve base prompts are at:

```
prompts/scenario_{1a,1b,2,3}/paraphrases/{no-spec,julia,epiaware}/01.md
```

For each, write a paraphrased version at the same path with filename `02.md`.

Rules:

- Preserve every factual statement. File paths and column names verbatim. Numerical values verbatim (for instance, "Gamma distribution, mean 5.5 days, standard deviation 2 days"). Distributional families verbatim. Framework or language constraint verbatim. Output file path and required columns verbatim. Every stated structural feature of the data. The error-handling expectation.
- Vary wording, sentence structure, ordering, headings, and tone. Don't write a near-copy.
- Don't add anything that wasn't in the original. No examples, no hints, no methodological suggestions, no strategies, no definitions of terms not already in the original.
- Don't reinterpret. A paraphrase should not change meaning or scope.

Don't read slots 03, 04, or 05; write your version independently.

## Sanity check before committing

The set of numerical tokens in your `02.md` should match the set in `01.md`:

```bash
diff <(grep -oE '[0-9]+(\.[0-9]+)?' prompts/scenario_1a/paraphrases/no-spec/01.md | sort) \
     <(grep -oE '[0-9]+(\.[0-9]+)?' prompts/scenario_1a/paraphrases/no-spec/02.md | sort)
```

Any difference means a number has moved or been altered; nothing should.

## Worked example

Original (excerpt from `prompts/scenario_1a/paraphrases/no-spec/01.md`):

> # Estimate the time-varying reproduction number
>
> You are analysing daily counts from an infectious disease outbreak. Your task is to estimate the time-varying effective reproduction number $R_t$ (the average number of secondary infections caused by an individual infected at time $t$) from the observed data.
>
> ## Data
> The observation period is 150 days, dated 2023-01-01 to 2023-05-30. The data are in:
> - `data/cases.csv` with columns `date` (YYYY-MM-DD) and `cases` (integer counts).

A plausible paraphrase:

> Reconstruct $R_t$ from outbreak case data
>
> Daily case counts from an unfolding outbreak are given for 150 days (2023-01-01 through 2023-05-30) in `data/cases.csv` (columns `date` in YYYY-MM-DD form and `cases` as integers). From these, estimate the time-varying effective reproduction number, that is, the mean number of secondary infections from somebody infected at time $t$.

Dates, column names, file path, and definition of $R_t$ are preserved; structure and prose change.

## Questions

If anything is unclear, please ask Sebastian. If a base prompt contains an apparent typo, inconsistency, or ambiguity, please don't fix it in your paraphrase. Message Sebastian, we'll correct `01.md`, and you'll redo that file.
