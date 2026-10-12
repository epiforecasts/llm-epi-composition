# Scientific implementation review protocol

This protocol implements the direct error assessment in
[the analysis plan](../analysis_plan.md). Human review supplies evidence about
scientific implementation errors (H_E), and also calibrates automated checks.
The review covers omissions as well as incorrectly implemented adjustments. The
Gostic-informed rubric is evaluator-only; core agents receive broad estimation
requests centred on current instantaneous R rather than this checklist. Assess
whether an answer applies to now or is clearly dated earlier, as well as its
scientific handling of recent observations. The budget is at most 10 hours per reviewer;
no additional sample is implied for secondary benchmarks.

## Reviewers and masking

Two infectious-disease modellers review independently without LLM assistance.
They did not implement the detectors. Record their modelling, language and
framework experience, package contributions and conflicts of interest.

Conceal agent model, wording/run identity, truth, recovery scores, detector/probe
outputs and the other reviewer's ratings. Provide the assigned scientific task
and tool requirements. Condition may be apparent from code and package use;
claim no masking of package identity. Reviewers may consult a frozen package
reference to check what an imported component does. Importing a component is
not by itself proof that it is configured correctly.

Freeze task-specific applicability, materiality rules and accepted alternatives
before drawing the sample. Include numerical tolerances where relevant, with
development evidence for acceptable discretisation/support approximations and
weekly aggregation or smoothing. Minor numerical differences are not automatically
substantive. Probe responsiveness alone is not a materiality threshold.
Familiarisation uses development code outside the evaluation sample. Allow one
hour per reviewer for briefing and familiarisation.

## Sample and population

After core generation is complete, randomly draw three reviewable implementations
from each task–condition cell: tasks 1b, 2 and 3 × `julia` and `composable`, giving
18 implementations. Sample uniformly without replacement within each cell across
both agent deployments, wordings and all blocks. Record the seed, eligible list,
sampling fractions and source hashes. Both reviewers assess the same sample in
independently randomised orders.

Reviewable means that sufficient source exists to attempt the rubric. Define this
eligibility check before sampling and apply it without recovery or detector scores.
Generation failure, invalid output and failed held-out execution do not exclude
available source. If fewer than three submissions are eligible in a cell, review
all and leave the shortfall visible; do not fill it from another cell. Report
missing/unreviewable source among all allocated implementations separately.

The human error comparison concerns reviewable implementations. Missing source
must not be counted as error-free or silently removed from workflow-success
reporting. Review the frozen code used for the held-out evaluation; subsequent
manual fixes are not part of this assessment.

## Materials and timing

Each folder contains anonymised source, the scientific prompt and input
specification, the fixed data/regime summary, frozen API references, the short
method/limitations note and a blank sheet. Include date/status/reason fields from
held-out output and the dated current-assessment summary to assess claims about
recency and withholding. Conceal numerical R estimates and recovery scores,
including values quoted in explanations; preserve dates and qualitative
limitations. No reference solution or realised latent truth is included. Preserve the
original source hash even if filenames are anonymised.

Stop at the task-specific cap: 20 minutes for 1b, 30 for 2 and 40 for 3. Six sampled
implementations per task give at most 6×20+6×30+6×40=540 minutes, or nine hours,
within the ten-hour total including familiarisation. Retain ratings made and mark
unfinished assessments incomplete. A fixed checklist focuses effort; a short
additional note can identify a substantive issue outside it. Shortfalls in the
sample do not authorise extra submissions or extension reviews.

The project lead adjudicates disagreements after both independent sheets are
preserved, recording reasons and remaining uncertainty. Adjudication does not
change original ratings. Record adjudication time separately: it is additional
project-lead effort, not included in the two reviewers' ten-hour budgets. Do not
request unbudgeted additional reviews to resolve difficult submissions.

## Review sheet and task applicability

```text
Sample ID:
Reviewer ID:
Elapsed minutes:
Assessment complete: yes / no

Component | correct / incorrect / required but absent / unclear / not applicable | reason and source location
Instantaneous R estimand, date claims and method suitability |  |
Generation-interval handling                         |  |
Infection-to-observation delay handling               |  |
Daily interval discretisation and finite support     |  |
Initialisation and early-boundary treatment           |  |
Recent-date right truncation or justified withholding |  |
Weekly reporting variation (tasks 2/3)                |  |
Supplied report-date ascertainment                   |  |
Process and observation variation                    |  |
Coherent use of multiple streams (task 3)             |  |

Additional substantive scientific error, if found:
For each defect: unaddressed issue / addressed incorrectly / unclear; source/note evidence:
Instruction violation, if found (separate from scientific error):
Overall: confirmed substantive error / no substantive error found / unresolved
Reason and component(s) supporting overall rating:
```

A substantive error implements an incompatible estimand, leaves a scientifically
necessary issue unaddressed, or computes an adjustment incorrectly. Necessity is
judged against the target, input data and task-specific rubric, even when the broad
prompt does not name the issue. Freeze concrete examples and accepted alternatives
by task. The rubric is not restricted to components recognisable by detectors.

- Accept mathematically equivalent models, discretisations and documented
  scientifically defensible approximations. Record uncertain adequacy as unclear.
- Required but absent means scientifically necessary handling is missing. A named
  renewal model, convolution or shared-latent architecture is not mandatory if an
  alternative handles the relevant issue adequately. Optional structures are not
  applicable when omitted.
- A smoothing prior or negative-binomial observation likelihood is not a
  universal requirement. Assess variation in the full latent/observation model.
- Check ascertainment on the report date and the implemented weekly scale;
  equivalent factor normalisations are acceptable.
- Distinguish finite delay support, daily interval integration and unobserved
  future reports. They are separate scientific issues.
- Judge recent-date estimates and withholding against information available at
  the observation cut-off. Justified withholding can be scientifically appropriate;
  an unexplained omission cannot. A caveat does not correct estimates presented
  with unsupported confidence. Withheld dates contribute no recovery score.
- An explicitly dated earlier estimate can be scientifically defensible while
  answering the current-transmission question only partially. Withholding leaves
  that question unanswered. Neither is automatically an implementation error.
  Presenting an earlier estimate as current is substantive. Assess date claims and
  limitations within the existing estimand and recent-date rubric items; no extra
  review or time allocation is introduced.
- Evaluate custom and imported components against the same scientific rules.
  Package choice, bad recovery or a failed diagnostic alone is not a code defect.

For defects, record the observable distinction between an unaddressed issue and an
incorrect adjustment, using source and notes. Absence of explanation does not imply
absence of handling: a package default may address the issue correctly. Likewise,
acknowledging a pitfall in prose does not establish that the implementation handles
it. Do not infer the agent's internal understanding. These annotations remain
within the task-specific assessment cap; unresolved detail stays unresolved.

Overall **confirmed substantive error** requires at least one identified
substantive scientific defect, including an additional issue outside the rubric.
Other unclear components do not negate that finding. **No substantive error
found** requires a complete assessment with all applicable components rated
correct and no additional substantive defect. All other cases are **unresolved**.
No-error-found is limited to this review; it does not certify the entire analysis.

## Direct error results

Preserve each reviewer's component and overall ratings and the adjudicated result.
Report counts by task and condition: confirmed error, no error found, unresolved,
incomplete and not applicable, with eligibility and sampling denominators. Report
initial raw agreement; Cohen's kappa is supplementary where estimable, with the
small sample and repeated component assessments made explicit.

For a pooled descriptive error contrast, give each task one third of the weight
and use its sampled implementation proportions; show the constituent counts and
sampling fractions. There are at most nine reviewed implementations per condition,
not eighteen independent reviewer observations per condition. Component ratings
are also correlated within implementation. Do not use a significance test that
counts components or reviewers as independent replications.

For each task–condition cell, the sampled error fraction lies between
`confirmed / reviewed` and `(confirmed + unresolved) / reviewed` when unresolved
assessments are assigned either status. Apply the same task weights to these
bounds and propagate them to the condition difference. These are unresolved-case
bounds for the reviewed sample, not confidence intervals or bounds on the entire
unreviewed population. If a cell has no eligible code, the pooled contrast is
unavailable. Small-sample uncertainty and missing-source limits remain explicit.

## Automated-check calibration

Freeze a mapping between each detector/check and the corresponding applicable
rubric item before examining ratings. Some rubric items may have no detector;
some targeted checks assess only part of an item. Report these gaps.

The input-use panel includes an unchanged repeat, independent alternative data,
delay/GI perturbations and time-varying ascertainment perturbations where applicable.
Its responsive/no-response/indeterminate flags are not defect labels. Delays can
change without moving R under constant growth, and some ascertainment changes
cancel from growth ratios. Calibrate on informative references and known ignored
inputs. Keep initial human ratings blinded to probe results; only compare after
ratings are frozen. Never count the several refits as independent implementations.

Compare automated flags against adjudicated component ratings. Incorrect and
required-but-absent are defects; correct is no detected defect. Exclude unclear,
not applicable and unassessed rows and report their counts. Show true/false
positive and negative counts, sensitivity, specificity and agreement where
estimable. With 18 implementations and potentially very few defects, these are
coarse calibration summaries. Do not assume independent observations across
components or present the sample as precise detector validation.

Report human-confirmed errors and full-panel automated flag rates separately.
Weak calibration does not justify a claim that flag-rate differences demonstrate
fewer scientific errors. Illustrative failures outside this random sample, if
reported, have no role in its estimated error proportions and require separately
recorded review time.
