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
Estimate the time-varying reproduction number (Rt) from daily case counts. You are provided with a generation interval distribution and a reporting delay distribution.

**Scenario 1b - Method specified (renewal equation)**
Estimate the time-varying reproduction number (Rt) from daily case counts using the renewal equation framework. You are provided with a generation interval distribution and a reporting delay distribution.

**Scenario 2 - Structured Rt with observation processes (method specified)**
Estimate Rt using the renewal equation, accounting for:
- Day-of-week effects in reporting
- Time-varying ascertainment
- Negative binomial observation noise

**Scenario 3 - Multiple data streams (method specified)**
Joint model of cases, hospitalisations, and deaths using the renewal equation, with:
- Stream-specific delays
- Stream-specific ascertainment processes
- Shared underlying Rt with autoregressive dynamics

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

#### Inference Approach

The prompts request Bayesian inference with posterior samples, but LLM-generated solutions may use:
- **Bayesian MCMC** (as requested)
- **Variational inference**
- **Maximum likelihood / optimisation**
- **Other approaches**

Non-Bayesian approaches are acceptable if they:
- Provide point estimates of $R_t$ over time
- Include some measure of uncertainty (confidence intervals, bootstrap, etc.)

Lack of uncertainty quantification is a departure (category B or C depending on context).

#### Package Baseline: EpiNow2

EpiNow2 (R package) will be run on the same data to provide a "package baseline" estimate of Rt. This allows:
- Visual comparison of LLM-generated model outputs against an established tool
- Quantification of how errors in model specification affect Rt estimates
- A practical benchmark that reviewers and readers can relate to

Note: EpiNow2 uses a Gaussian process prior on $R_t$ by default, which differs from the AR(1) used in EpiAware reference solutions. Both are valid smoothness choices.

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
| **Convergence** | Does MCMC sampling converge? (R-hat < 1.05, ESS > 400) (0/1) |
| **Plausibility** | Are Rt estimates plausible? (0/1) - bounded (e.g., 0.1-10), smooth over time (no implausible jumps), consistent with epidemic dynamics |

### Expert Review (Departure-Based Assessment)

Two independent infectious disease modellers, blinded to condition, assess each submission against the reference solutions defined above.

#### Assessment Process

For each submission, the reviewer:

1. **Identifies the method used** (for Scenario 1a only)
   - Renewal equation
   - Bettencourt-Ribeiro / SIR-based
   - Naive ratio-based
   - Other (describe)

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

## Protocol

### Prompt Construction

Standardised prompts will be constructed for each scenario containing:
- Clear problem statement (epidemiological question or method specification)
- Data description and format
- Required outputs (posterior samples for Rt, predictions)
- Language/framework constraint (e.g., "use Stan", "use PyMC", "use EpiAware components")

Prompts will be identical across LLMs within each condition.

For Condition D (with EpiAware), the prompt will additionally include:
- Package overview and component descriptions
- Type hierarchy and interfaces
- 2-3 worked examples from documentation

### Execution

1. Each LLM will be prompted 3 times per scenario per condition (to account for stochasticity)
2. Temperature settings will be recorded and held constant where possible
3. All prompts and responses will be logged verbatim
4. Code outputs will be executed in isolated environments with standardised package versions

### Expert Review Protocol

- **Two independent reviewers**, both infectious disease modellers not involved in prompt construction
- Reviewers are blinded to: which LLM generated the code, which condition it belongs to
- Each reviewer independently assesses each code sample against the reference solution
- Departures are documented and classified
- Inter-rater reliability will be assessed (Cohen's kappa or similar)
- Disagreements resolved by discussion or third reviewer if necessary
- Summary scores computed after reconciliation

### Analysis

1. Tabulate pass rates for each automated criterion by condition and LLM
2. Tabulate departure counts by category (A/B/C/D) by condition and LLM
3. Compare success rates across conditions (Fisher's exact test or similar)
4. For Scenario 1a: document method choices by LLM and condition
5. Compare 1a vs 1b to assess effect of method specification
6. Develop taxonomy of error types observed
7. Qualitative analysis of failure modes and their epidemiological implications

## Outputs

1. Summary statistics and visualisations of performance across conditions
2. Categorised examples of model outputs (successes and failures)
3. Error taxonomy with representative examples
4. Method choice analysis (Scenario 1a)
5. Reproducible analysis code and logged prompts/responses

## Limitations

- EpiAware comparison is confounded with language (Julia only), though C vs D controls for this
- Scenarios assume known epidemiological parameters (e.g., generation interval); joint inference of these parameters is out of scope
- Number of runs per condition (3) may be insufficient to characterise variability
- Departure classification involves judgement despite structured categories
- Results may not generalise to future model versions

## Ethical Considerations

- No human subjects involved
- All data publicly available
- LLM outputs will be reviewed before any public release

## Timeline

- Week 1: Finalise scenarios, construct prompts, obtain data, write reference solutions
- Week 2: Run experiments across models and conditions
- Week 3: Evaluation (automated + expert review)
- Week 4: Analysis and write-up

---

*Document created: 2024-12-07*
*Status: Draft*
