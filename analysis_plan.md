# Analysis Plan: AI-Assisted Epidemic Model Composition

## Research Question

Can large language models correctly compose epidemic models from first principles, and does access to validated modular components affect their performance?

## Objectives

1. Assess the ability of LLMs to generate correct, runnable epidemic models when prompted with epidemiological scenarios
2. Characterise the types of errors that occur when models are generated without domain-specific tooling
3. Evaluate whether providing validated composable components improves model correctness
4. Compare performance across different LLMs, including open-source models relevant to low- and middle-income country (LMIC) settings, in order to test the robustness of findings.

## Study Design

### Conditions

| Condition | Language | Framework | Description |
|-----------|----------|-----------|-------------|
| **A: Stan** | Stan + R | From scratch | Generate Stan code with R interface |
| **B: PyMC** | Python | From scratch | Generate PyMC model code |
| **C: Turing.jl** | Julia | From scratch | Generate Turing.jl code |
| **D: EpiAware** | Julia | Validated components | Generate model using EpiAware, documentation provided |

**Primary comparison:**
- A/B/C vs D: Effect of validated components on model correctness

**Secondary comparison:**
- A vs B vs C: Consistency of performance across probabilistic programming languages

Note: The primary comparison is partially confounded by language (D uses Julia), but C vs D provides a language-controlled comparison.

### Models Under Evaluation

| Model | Type | Rationale |
|-------|------|-----------|
| Claude Sonnet | Commercial | Current frontier capability, cost-effective |
| GPT-4o | Commercial | Generalisability across providers |
| Llama 3.1 70B | Open-source | LMIC accessibility, reproducibility |

### Scenarios

| Scenario | Prompt type | Description | Tests |
|----------|-------------|-------------|-------|
| **1a** | Epidemiological question | "Estimate Rt from case counts" | Method selection + implementation |
| **1b** | Method specified | "Use the renewal equation to estimate Rt" | Implementation correctness |
| **2** | Method specified | Structured Rt with observation processes | Complex implementation |
| **3** | Method specified | Multiple data streams | Multi-component implementation |

**Scenario 1a - Epidemiological question (open method)**
Estimate the time-varying reproduction number (Rt) from daily case counts.

**Scenario 1b - Method specified (renewal equation)**
Estimate the time-varying reproduction number (Rt) from daily case counts using the renewal equation framework.

**Scenario 2 - Structured Rt with observation processes (method specified)**
Estimate Rt using the renewal equation, accounting for:
- Day-of-week effects in reporting
- Time-varying ascertainment
- Negative binomial observation noise

**Scenario 3 - Multiple data streams (method specified)**
Joint model of cases, hospitalisations, and deaths using the renewal equation, with:
- Stream-specific delays
- Stream-specific ascertainment processes
- Shared underlying Rt that varies smoothly over time

The comparison of 1a vs 1b tests whether specifying the method improves correctness and documents what methods LLMs choose when unconstrained.

### Data

UK COVID-19 data from the UKHSA dashboard:
- Daily case counts
- Hospital admissions
- Deaths

Real data is used to test whether generated models can handle actual epidemic dynamics.

### Reference Solutions

Reference solutions serve two purposes: (1) providing a gold standard for expert review of departures, and (2) enabling visual/quantitative comparison of Rt estimates.

#### Mathematical Specification

The core renewal equation model is:

**Infection process (renewal equation):**
$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where:
- $I_t$ is the number of infections at time $t$
- $R_t$ is the instantaneous reproduction number at time $t$
- $g_s$ is the generation interval PMF (probability that generation interval = $s$ days)

**Observation process:**
$$\mathbb{E}[C_t] = \alpha_t \sum_{d=0}^{D} I_{t-d} \cdot f_d$$

where:
- $C_t$ is observed cases at time $t$
- $\alpha_t$ is the ascertainment rate (proportion of infections observed)
- $f_d$ is the delay PMF (probability of delay = $d$ days from infection to report)

**Observation model:**
$$C_t \sim \text{NegBin}(\mu = \mathbb{E}[C_t], \phi)$$

where $\phi$ is the overdispersion parameter.

**Temporal smoothness on $R_t$:**

Some form of smoothness constraint on $R_t$ is necessary to avoid overfitting. Acceptable approaches include:
- **AR(1):** $\log R_t = \rho \log R_{t-1} + \epsilon_t$ (used in EpiAware reference)
- **Random walk:** $\log R_t = \log R_{t-1} + \epsilon_t$
- **Gaussian process:** $\log R_t \sim \text{GP}(0, k)$ (EpiNow2 default)
- **Splines:** $\log R_t = \sum_j \beta_j B_j(t)$

All are considered equivalent alternatives (departure category A) provided they enforce reasonable smoothness.

#### Discretisation and Censoring

The renewal equation operates on discrete time steps (typically days), requiring continuous distributions for generation intervals and delays to be discretised to probability mass functions (PMFs). Proper discretisation should account for **double interval censoring**:

- **Primary censoring**: The primary event (e.g., infection) occurs at an unknown time within its observation interval
- **Secondary censoring**: The secondary event (e.g., symptom onset, reporting) also occurs at an unknown time within its interval

Naive discretisation (e.g., evaluating the PDF at integer points, or simple CDF differences) does not account for this and can introduce bias, particularly for distributions with short means relative to the discretisation interval.

**Acceptable approaches:**
- Double interval censoring (as implemented in EpiAware's `censored_pmf()`)
- Midpoint discretisation with appropriate justification
- Any method that explicitly acknowledges and addresses the censoring problem

**Potential error source:** LLMs generating code from scratch may use naive discretisation without considering censoring. This is recorded as a departure if it occurs.

#### Inference Approach

The prompts do not specify an inference approach. LLM-generated solutions may use:
- **Bayesian MCMC**
- **Variational inference**
- **Maximum likelihood / optimisation**
- **Other approaches**

Non-Bayesian approaches are acceptable if they provide point estimates of $R_t$ over time.

**Note:** Prompts do not explicitly request uncertainty quantification. Whether models provide uncertainty is recorded as an automated evaluation criterion. This tests whether LLMs recognise that uncertainty is essential for epidemiological inference, and whether validated components naturally produce it.

#### Package Baseline: EpiNow2

EpiNow2 (R package) will be run on the same data to provide a "package baseline" estimate of Rt for Scenarios 1a, 1b, and 2. This allows:
- Visual comparison of LLM-generated model outputs against an established tool
- Quantification of how errors in model specification affect Rt estimates
- A practical benchmark that reviewers and readers can relate to

Note: EpiNow2 uses a Gaussian process prior on $R_t$ by default, which differs from the AR(1) used in EpiAware reference solutions. Both are valid smoothness choices. EpiNow2 does not natively support multiple observation streams, so no package baseline is provided for Scenario 3.

#### Custom Reference Solutions

For each scenario, a reference solution is provided implementing the renewal equation approach:

| Scenario | Reference Implementation |
|----------|-------------------------|
| 1a/1b | EpiAware: Renewal + AR(1) latent + NegBin obs with delay |
| 2 | EpiAware: As above + day-of-week effects + time-varying ascertainment |
| 3 | EpiAware: As above + StackObservationModels for multiple streams |

Reference solution code is in `reference_solutions/`.

## Evaluation Criteria

### Objective Criteria (Automated)

| Criterion | Measurement |
|-----------|-------------|
| **Syntactic validity** | Does the code parse without errors? (0/1) |
| **Execution** | Does the model run on the test data? (0/1) |
| **Plausibility** | Are Rt estimates plausible? (0/1) - bounded (e.g., 0.1-10), smooth over time (no implausible jumps), consistent with epidemic dynamics |
| **Uncertainty quantification** | Does the model provide uncertainty estimates? (0/1) - credible/confidence intervals, posterior samples, or similar |
| **Asked clarifying questions** | Did the LLM ask about epidemiological parameters before producing code? (0/1) |
| **Appropriate parameters** | If not asked, did the model use reasonable epidemiological parameters? (0/1) - generation interval ~3-7 days, delay ~2-7 days for COVID-19 |

### Expert Review (Departure-Based Assessment)

Two independent infectious disease modellers, blinded to condition, assess each submission against the reference solutions defined above. Reviewers may use LLMs to streamline the review process.

#### Assessment Process

For each submission, the reviewer:

1. **Identifies the method used** (for Scenario 1a only)

   Based on Gostic et al. (2020) "Practical considerations for measuring the effective reproductive number, Rt":

   - Renewal equation / Cori / EpiEstim-style instantaneous Rt - **Recommended** (accurate, uses only past data)
   - Wallinga-Teunis / case reproductive number - **Acceptable for retrospective analysis** (uses future data, underestimates at end of time series without adjustment)
   - Bettencourt-Ribeiro / SIR-based - **Not recommended** (assumes exponential generation interval, biased when Rt >> 1)
   - Naive ratio-based (e.g., $R_t = C_t / C_{t-1}$) - **Not acceptable** (ignores generation interval, fundamentally biased)
   - Other (describe and assess appropriateness)

2. **Lists departures from reference solution**
   - Each departure documented with description

3. **Classifies each departure**
   - **(A) Equivalent alternative** - Different but equally valid approach
   - **(B) Minor error** - Small mistake unlikely to substantially affect results
   - **(C) Major error** - Significant mistake that would bias results
   - **(D) Fundamental misunderstanding** - Indicates lack of understanding of the underlying epidemiology or statistics

4. **Summary scores**
   - Count of departures by category (A/B/C/D)
   - Overall assessment: Acceptable / Minor issues / Major issues / Incorrect

**Note:** Code that fails automated checks (syntax, execution) is still reviewed to assess the underlying approach. This distinguishes "correct approach with implementation bugs" from "fundamentally flawed approach".

## Protocol

### Prompt Construction

Standardised prompts will be constructed for each scenario containing:
- Clear problem statement (epidemiological question or method specification)
- Data description and format
- Language/framework constraint (e.g., "use Stan", "use PyMC", "use EpiAware components")

**Prompts do not provide epidemiological parameters** (generation interval, delay distributions). This tests whether LLMs:
- Recognise these parameters are needed
- Ask appropriate clarifying questions
- Make reasonable assumptions if not asking

For Condition D (with EpiAware), the prompt will additionally include:
- Package overview and component descriptions
- Type hierarchy and interfaces
- 2-3 worked examples from documentation

### Handling Clarifying Questions

If an LLM asks clarifying questions rather than producing code:

1. **Standard responses are provided** for common questions (see below)
2. **Maximum 3 rounds** of back-and-forth before final code expected
3. **Only final code is evaluated** - intermediate outputs are not assessed
4. **Record whether** the LLM asked clarifying questions (evaluation criterion)
5. **Record what** questions were asked (qualitative analysis)

#### Standard Responses to Clarifying Questions

| Question | Response |
|----------|----------|
| Generation interval? | "Use a gamma distribution with mean 5 days and SD 1.5 days, or an equivalent discrete PMF" |
| Reporting delay? | "Use a gamma distribution with mean 4 days and SD 2 days, or an equivalent discrete PMF" |
| What time period? | "The data covers [start date] to [end date]" |
| What priors? | "Use your best judgement for reasonable priors" |
| Bayesian or frequentist? | "Use whatever approach you think is most appropriate" |
| Other questions | "Make reasonable assumptions based on your knowledge of COVID-19 epidemiology" |

### Execution

1. Each LLM will be prompted 3 times per scenario per condition (to account for stochasticity)
2. Temperature settings will be recorded and held constant where possible
3. All prompts and responses will be logged verbatim
4. Code outputs will be executed in isolated environments with standardised package versions

### Expert Review Protocol

- **Two independent reviewers** (infectious disease modellers) assess each code sample against the reference solution
- Reviewers are blinded to: which LLM generated the code, which condition it belongs to
- Each reviewer independently assesses each code sample
- Departures are documented and classified
- Inter-rater reliability will be assessed (Cohen's kappa or similar)
- Disagreements resolved by discussion between reviewers; a third reviewer may be consulted if resources permit
- Summary scores computed after reconciliation

### Analysis

1. Tabulate pass rates for each automated criterion by condition and LLM
2. Tabulate departure counts by category (A/B/C/D) by condition and LLM
3. Compare success rates across conditions descriptively
4. For Scenario 1a: document method choices by LLM and condition
5. Compare 1a vs 1b to assess effect of method specification
6. Develop taxonomy of error types observed
7. Qualitative analysis of failure modes and their epidemiological implications

## Pre-Specified Results

### Tables

**Table 1: Automated evaluation results by condition and LLM**
- Rows: LLM × Condition (12 rows)
- Columns: Syntactic validity, Execution, Plausibility, Uncertainty quantification, Asked clarifying questions, Appropriate parameters
- Cells: Pass rate (n/3 runs)
- Aggregated rows for "From scratch" (A+B+C) vs "EpiAware" (D)

**Table 2: Expert review summary by condition and LLM**
- Rows: LLM × Condition (12 rows)
- Columns: Departures by category (A/B/C/D counts), Overall assessment distribution
- Aggregated rows for "From scratch" vs "EpiAware"

**Table 3: Method selection in Scenario 1a**
- Rows: LLM × Condition (12 rows)
- Columns: Renewal/Cori, Wallinga-Teunis, Bettencourt-Ribeiro, Naive ratio, Other
- Cells: Count of runs using each method

**Table 4: Common error types (error taxonomy)**
- Rows: Error type (e.g., "naive discretisation", "missing delay convolution", "incorrect likelihood")
- Columns: Description, Category (B/C/D), Frequency by condition, Example

### Figures

**Figure 1: Primary comparison - automated pass rates**
- Bar chart comparing "From scratch" (A+B+C pooled) vs "EpiAware" (D)
- Grouped by automated criterion
- Error bars showing 95% CI (Wilson score interval)

**Figure 2: Rt estimates comparison**
- Panel per scenario (1a, 1b, 2, 3)
- Each panel shows: EpiNow2 baseline (where available), reference solution, and LLM-generated estimates (colour by condition)
- Ribbon for uncertainty where available
- Subset to representative examples (1 success, 1 failure per condition)

**Figure 3: Departure category distribution**
- Stacked bar chart
- X-axis: Condition (A, B, C, D)
- Y-axis: Proportion of departures
- Colours: Category A (green), B (yellow), C (orange), D (red)
- Faceted by scenario

**Figure 4: Effect of method specification (1a vs 1b)**
- Paired comparison showing pass rates for same LLM × Condition across 1a and 1b
- Highlights whether specifying "renewal equation" improves correctness

### Key Results Sentences

**Primary finding (A/B/C vs D):**
"Of [n] code samples generated without validated components (conditions A-C), [x]% passed all automated checks compared to [y]% with EpiAware components (condition D). Expert review identified a mean of [mean] category C/D departures per sample in conditions A-C versus [mean] in condition D."

**Language comparison (A vs B vs C):**
"Performance was [similar/varied] across probabilistic programming languages: Stan [x]%, PyMC [y]%, Turing.jl [z]% passed all automated checks. The language-controlled comparison (C vs D, both Julia) showed [description]."

**Method selection (Scenario 1a):**
"When not constrained to the renewal equation, LLMs selected [most common method] in [x]% of cases. [y]% chose methods rated 'not recommended' or 'not acceptable' per Gostic et al. (2020)."

**Effect of method specification (1a vs 1b):**
"Specifying the renewal equation [improved/did not improve] correctness: [x]% of 1b samples passed automated checks versus [y]% for 1a."

**Uncertainty quantification:**
"[x]% of samples provided uncertainty estimates. This was [higher/similar/lower] for EpiAware ([y]%) compared to from-scratch conditions ([z]%), suggesting validated components [do/do not] naturally encourage uncertainty quantification."

**Common errors:**
"The most frequent errors were [top 3 from taxonomy], occurring in [x]%, [y]%, [z]% of from-scratch samples respectively. These errors were [absent/rare] in EpiAware samples."

## Ethical Considerations

- No human subjects involved
- All data publicly available
- LLM outputs will be reviewed before any public release

---

*Document created: 2024-12-07*
*Status: Draft*
