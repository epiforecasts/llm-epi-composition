# Brief for the slot-02 paraphraser

Hi the paraphraser — short methodological task. ~90 minutes.

## Why

Slots 01, 03, 04, 05 of the prompt-paraphrase set are all LLM-generated (slot 01 was drafted by Claude Code and edited by Sebastian; slots 03/04/05 are produced by GPT-5, Gemini Flash, and Claude Sonnet). The wording-sensitivity test in this study therefore never escapes the LLM-prose register the evaluator (Claude) is itself trained on. A human-written paraphrase wave (you) is the only piece of the set that breaks that monoculture.

You have read the analysis plan and the predictions; this is not a hypothesis-blinded paraphrase. The methodological role is narrower: introduce non-LLM-register prose into the paraphrase set, decorrelate wording from Claude's house style. You will be acknowledged as the slot-02 paraphraser in the paper.

## What

Twelve base prompts at:

```
prompts/scenario_{1a,1b,2,3}/paraphrases/{no-spec,julia,epiaware}/01.md
```

Write a paraphrased version of each at the same path with filename `02.md`.

The rule:

- **Preserve every factual statement.** Data file paths and column names exactly as given. Numerical parameters exactly as given (e.g. "Gamma distribution, mean 5.5 days, standard deviation 2 days"). Distributional families exactly as given. Framework or language constraint exactly as given. Output file path and required columns exactly as given. Every structural feature about the data. Error-handling expectation.
- **Vary the wording, ordering, headings, sentence structure, and tone.** Don't write a near-copy.
- **Add nothing.** No examples, hints, advice, methodological suggestions, strategies, or definitions of terms not already in the original.
- **Reinterpret nothing.** Don't paraphrase a fact in a way that changes meaning or scope.

Don't read slots 03/04/05 — write your version independently of theirs.

## Sanity check before committing

A diff of all numbers in `01.md` vs your `02.md` should be empty:

```bash
diff <(grep -oE '[0-9]+(\.[0-9]+)?' prompts/scenario_1a/paraphrases/no-spec/01.md | sort) \
     <(grep -oE '[0-9]+(\.[0-9]+)?' prompts/scenario_1a/paraphrases/no-spec/02.md | sort)
```

Any difference is a number you've changed; nothing should change.

## Worked example

Original `prompts/scenario_1a/paraphrases/no-spec/01.md` (excerpt):

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
> Daily case counts from an unfolding outbreak are given for 150 days (2023-01-01 through 2023-05-30) in `data/cases.csv` (columns `date` in YYYY-MM-DD form and `cases` as integers). From these, estimate the time-varying effective reproduction number — the mean number of secondary infections from somebody infected at time $t$.

(Note: dates, column names, file path, definition of Rt all preserved; structure and prose changed.)

## Questions

If anything is unclear, ask Sebastian. If you flag something in a base prompt that looks wrong (a typo, an inconsistency), don't fix it in your paraphrase — message Sebastian and we'll fix it in 01.md and you'll re-do that one.
