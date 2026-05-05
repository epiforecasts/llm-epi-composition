# Brief for the slot-02 paraphraser and MWK reviewer

Two tasks before the project can be pre-registered. Allow ~90 minutes of focused work (12 paraphrases at 5–8 min each, plus ~15 min for the API-docs check).

**Who you are.** You are a human reader who is *not* the original prompt-drafter and *not* an LLM. You may have full access to the analysis plan; hypothesis blinding is not required. Eligible: any second project member, any colleague, any PhD or MSc student under the authors' supervision, or any literate technical reader recruited for the task. The slot-02 paraphrase is the only piece of the study's prompt set that is human-written prose (slots 01, 03–05 are all LLM-generated), so the task's methodological function is to introduce non-LLM-register wording into the paraphrase set.

You will be acknowledged in the paper.

---

## Task A — Manual paraphrase wave (slot 02)

Twelve base prompts live at:

```
prompts/scenario_{1a,1b,2,3}/paraphrases/{no-spec,julia,epiaware}/01.md
```

For each one, write a paraphrased version at the same path with filename `02.md` (same directory, just `02.md` instead of `01.md`).

**The rule** (full version in `prompts/paraphrase_brief.md`):

- Preserve every factual statement: data file paths and column names exactly as given; numerical parameters exactly as given; distributional families exactly as given; the framework or language constraint exactly as given; the output file path and its required columns exactly as given; every structural feature mentioned about the data; the expected error-handling behaviour.
- Vary the wording, ordering, headings, sentence structure, and tone.
- Do not add, remove, or reinterpret any factual content. No examples, hints, advice, methodological suggestions, or strategies.

**What you don't need to do.** The other paraphrase slots (`03.md`, `04.md`, `05.md`) are already filled by GPT-5, Gemini 2.5 Flash, and Claude Sonnet 4.5. Don't read them — write your `02.md` independently of theirs to maximise wording decorrelation.

**Useful sanity check before committing.** Diff `02.md` against `01.md` and confirm no fact has changed:

```bash
diff <(grep -oE '[0-9]+(\.[0-9]+)?' prompts/scenario_1a/paraphrases/no-spec/01.md | sort) \
     <(grep -oE '[0-9]+(\.[0-9]+)?' prompts/scenario_1a/paraphrases/no-spec/02.md | sort)
```

Any difference here is a number that you've changed; it shouldn't be present unless you've corrected a typo (and you should not be correcting any).

---

## Task B — MWK API-docs validation

Two API reference files are bundled with each `julia` and `epiaware` agent run:

```
prompts/turing_api_docs.md       (228 lines)
prompts/epiaware_api_docs.md     (~890 lines)
```

These should be **API-level only**: function/type signatures, arguments, return types, brief primitive-level usage snippets. They should **not** contain end-to-end Rt-estimation examples or tutorials walking through full Rt-model construction. The full rule is in `prompts/paraphrase_brief.md` under "API docs validation".

**What's already been done.** Two LLM instances (GPT-5 and Gemini 2.5 Flash) have read both files. Their flags are at `prompts/mwk_validation_report.md`. GPT-5 flagged one block in `epiaware_api_docs.md` ("Extracting results from `generated`", lines 105–129) as a worked Rt-extraction example; that block has been rewritten as a structural description without code. Gemini hit its free-tier rate limit before reading the EpiAware docs. Turing.jl docs were flagged as clean.

**What you do.** Read both files yourself. For each, decide whether you agree the docs are now API-level only. If you flag any further sections that the LLMs missed:

1. Note them in `prompts/mwk_validation_report.md` under a new `### internal author` subsection per file.
2. Edit the API doc to remove the violation.
3. Commit with a message like `epiaware docs: remove worked example from §X per internal MWK review`.

If you find no further violations, add a single line under each file: `internal author: no further violations found.`

You don't need to re-run the LLM check; it's been done.

---

## Status of the rest of the pre-registration checklist

These are not your tasks — listed here so you know what's already in place:

- 8 DGP variants × 20 replicates from a Lloyd-Smith Bellman-Harris simulator: done.
- 20-rep canonical sanity check on the reference EpiAware: passed (med RMSE 0.086, med coverage 1.00 — conservatively calibrated, asymmetric criterion satisfied).
- LLM paraphrases (slots 03–05) for all 12 cells: done.
- Automated detectors, study runner, paraphrase generator: done.
- Pre-specified quantitative effect sizes for all 6 predictions: done.

After your two tasks above, the remaining step is pinning the pre-registration commit hash in `README.md` and any external registry (OSF, etc.) — that's a few-minute task by the original prompt-author.

---

## Questions

If anything is unclear, ask Sebastian. The whole point of this brief is to make the tasks executable without further context.
