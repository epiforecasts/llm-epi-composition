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
| **R** | R | Open choice | Generate R code (LLM chooses approach: packages, custom code, or combination) |
| **Python** | Python | Open choice | Generate Python code (LLM chooses libraries: PyMC, NumPyro, JAX, etc.) |
| **EpiAware** | Julia | Validated components | Generate model using EpiAware, documentation provided |

**Primary comparison:**
- R/Python vs EpiAware: Effect of validated components on model correctness

**Secondary comparisons:**
- R vs Python: Consistency of performance across general-purpose languages
- EpiAware: Whether domain-specific tooling with documentation improves correctness

Note: R and Python conditions are "open choice" - the LLM decides which packages or methods to use. This tests what approaches LLMs naturally choose when given freedom. The EpiAware condition provides documentation for a specific domain tool, testing whether guided tooling improves results.

### Models Under Evaluation

| Model | Type | Rationale |
|-------|------|-----------|
| Claude Sonnet 4 | Commercial | Current frontier capability, cost-effective |
| Llama 3.1 8B | Open-source | LMIC accessibility (runs locally), reproducibility |

Note: GPT-4o was originally planned but excluded due to API rate limiting constraints. Llama 3.1 8B (rather than 70B) was chosen to demonstrate local inference on consumer hardware, more relevant to LMIC resource constraints.

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

UK COVID-19 data from the UKHSA dashboard (England):
- Daily case counts (by specimen date)
- Hospital admissions
- Deaths

Real data is used to test whether generated models can handle actual epidemic dynamics. Note that cases are indexed by specimen date, meaning there is a delay between infection and the recorded date.

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
| **Success** | Did the LLM produce working code within the iteration limit? (0/1) |
| **Iterations** | Number of iterations required to produce working code (1-10, or NA if failed) |
| **Produces plot** | Did the code generate a plot of Rt over time? (0/1) |
| **Current estimate** | Did the code report a current (most recent) Rt estimate? (0/1) |
| **Uncertainty** | Did the code provide uncertainty quantification? (0/1) |
| **Plausibility** | Are Rt estimates plausible? (checklist below) |
| **Error types** | Categories of errors encountered during iteration (import errors, syntax errors, runtime errors, etc.) |

#### Plausibility Criteria

Each criterion is scored 0/1. Total plausibility score = sum of criteria met.

| Criterion | Pass condition |
|-----------|----------------|
| **Bounded** | All Rt estimates between 0.1 and 10 |
| **No negative values** | No Rt estimates ≤ 0 |
| **Smooth trajectory** | No day-to-day changes > 0.5 in point estimate (allowing for genuine epidemic dynamics) |
| **Reasonable range** | Rt values span a plausible range for the epidemic phase (not all identical, not wildly variable) |
| **Uncertainty width** | 95% intervals neither too narrow (<0.1) nor too wide (>5) |
| **Uncertainty increases at edges** | Wider uncertainty for recent dates (where data is incomplete) |
| **Consistent direction** | Trajectory broadly consistent with case trend (Rt>1 when cases rising, Rt<1 when falling) |

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
- Language/framework constraint (e.g., "use R", "use Python", "use EpiAware components")

**Prompts do not provide epidemiological parameters** (generation interval, delay distributions). This tests whether LLMs:
- Recognise these parameters are needed
- Ask appropriate clarifying questions
- Make reasonable assumptions if not asking

For the EpiAware condition, the prompt will additionally include:
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

### Execution: Agentic Approach

We use an **agentic approach** where the LLM can iteratively write, execute, and refine code until it works. This reflects realistic use of coding assistants like Claude Code, Cursor, or GitHub Copilot, where users expect the tool to fix its own errors.

**Protocol:**
1. Each LLM is given the prompt and asked to write code, execute it, and fix any errors
2. The LLM can iterate until the code runs successfully or a maximum of 10 iterations is reached
3. Each run is repeated 3 times per scenario per condition (to account for stochasticity)
4. All iterations, error messages, and fixes are logged

**Recorded metrics:**
- Number of iterations required to produce working code
- Types of errors encountered and how they were fixed
- Whether the LLM succeeded within the iteration limit
- Final code and outputs

**Tools:**
- Claude Code (Claude models) - CLI tool with code execution capability
- Aider with Ollama (open-source models) - provides similar agentic capability for local models

**Rationale**: Single-shot code generation often fails due to trivial errors (missing imports, typos) that obscure assessment of methodological correctness. The agentic approach separates "can the LLM eventually produce working code?" from "is the methodology correct?", while reflecting how these tools are actually used in practice.

**Timeout**: Each execution attempt within an iteration is limited to 10 minutes. Total session timeout is 60 minutes.

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
- Rows: LLM × Condition (6 rows: 2 LLMs × 3 conditions)
- Columns: Syntactic validity, Execution, Plausibility, Uncertainty quantification, Asked clarifying questions, Appropriate parameters
- Cells: Pass rate (n/3 runs)
- Aggregated rows for "Open choice" (R + Python) vs "EpiAware"

**Table 2: Expert review summary by condition and LLM**
- Rows: LLM × Condition (6 rows)
- Columns: Departures by category (A/B/C/D counts), Overall assessment distribution
- Aggregated rows for "Open choice" vs "EpiAware"

**Table 3: Method selection in Scenario 1a**
- Rows: LLM × Condition (6 rows)
- Columns: Renewal/Cori, Wallinga-Teunis, Bettencourt-Ribeiro, Naive ratio, Other
- Cells: Count of runs using each method

**Table 4: Common error types (error taxonomy)**
- Rows: Error type (e.g., "naive discretisation", "missing delay convolution", "incorrect likelihood")
- Columns: Description, Category (B/C/D), Frequency by condition, Example

### Figures

**Figure 1: Primary comparison - automated pass rates**
- Bar chart comparing "Open choice" (R + Python) vs "EpiAware"
- Grouped by automated criterion
- Error bars showing 95% CI (Wilson score interval)

**Figure 2: Rt estimates comparison**
- Panel per scenario (1a, 1b, 2, 3)
- Each panel shows: EpiNow2 baseline (where available), reference solution, and LLM-generated estimates (colour by condition)
- Ribbon for uncertainty where available
- Subset to representative examples (1 success, 1 failure per condition)

**Figure 3: Departure category distribution**
- Stacked bar chart
- X-axis: Condition (R, Python, EpiAware)
- Y-axis: Proportion of departures
- Colours: Category A (green), B (yellow), C (orange), D (red)
- Faceted by scenario

**Figure 4: Effect of method specification (1a vs 1b)**
- Paired comparison showing pass rates for same LLM × Condition across 1a and 1b
- Highlights whether specifying "renewal equation" improves correctness

### Key Results

**Primary finding (R/Python vs EpiAware):**
"Of [n] code samples generated with open choice conditions (R, Python), [x]% passed all automated checks compared to [y]% with EpiAware components. Expert review identified a mean of [mean] category C/D departures per sample in open choice conditions versus [mean] with EpiAware."

**Language comparison (R vs Python):**
"Performance was [similar/varied] across general-purpose languages: R [x]%, Python [y]% passed all automated checks."

**Method selection (Scenario 1a):**
"When not constrained to the renewal equation, LLMs selected [most common method] in [x]% of cases. [y]% chose methods rated 'not recommended' or 'not acceptable' per Gostic et al. (2020)."

**Effect of method specification (1a vs 1b):**
"Specifying the renewal equation [improved/did not improve] correctness: [x]% of 1b samples passed automated checks versus [y]% for 1a."

**Uncertainty quantification:**
"[x]% of samples provided uncertainty estimates. This was [higher/similar/lower] for EpiAware ([y]%) compared to open choice conditions ([z]%), suggesting validated components [do/do not] naturally encourage uncertainty quantification."

**Common errors:**
"The most frequent errors were [top 3 from taxonomy], occurring in [x]%, [y]%, [z]% of open choice samples respectively. These errors were [absent/rare] in EpiAware samples."

## Discussion Points

### Open Choice vs Guided Tooling

This study compares "open choice" conditions (R, Python) where LLMs freely select their approach, with EpiAware where documentation for a domain-specific tool is provided. This tests two questions:

1. **What do LLMs naturally choose?** - Recording which packages/methods LLMs select when unconstrained
2. **Does guided tooling help?** - Whether providing documentation for validated components improves correctness

The R and Python conditions test what LLMs naturally do with general-purpose languages that have rich ecosystems. LLMs may choose task-specific packages (EpiEstim, EpiNow2 in R; PyMC, NumPyro in Python), write custom code, or combine approaches.

### DSL vs Task-Specific Packages

The R condition enables comparison between what LLMs naturally produce (which may include task-specific packages like EpiEstim, EpiNow2) and a composable DSL (EpiAware). Task-specific packages may be easier for LLMs to use correctly (fewer choices to make) but offer less flexibility for complex scenarios.

For simpler scenarios (1a, 1b), R packages like EpiEstim provide direct solutions, but for complex scenarios (2, 3) with day-of-week effects, time-varying ascertainment, or multiple data streams, LLMs using R packages may need to compose multiple packages or write custom code, reducing the abstraction benefit. The EpiAware DSL is designed for such composition.

### Functions vs DSL

This study does not cleanly separate the benefit of validated functions from the benefit of a composable DSL structure. Future work could explore this distinction using delay distribution estimation as a test case. The `primarycensoreddist` ecosystem provides validated functions across multiple languages (R, Python, Stan, Julia) without a compositional DSL, enabling a cleaner comparison.

## Ethical Considerations

- No human subjects involved
- All data publicly available
- LLM outputs will be reviewed before any public release

---

*Document created: 2024-12-07*
*Status: Draft*
