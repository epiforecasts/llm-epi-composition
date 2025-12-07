# Analysis Plan: AI-Assisted Epidemic Model Composition

## Research Question

Can large language models correctly compose epidemic models from first principles, and does access to validated modular components affect their performance?

## Objectives

1. Assess the ability of LLMs to generate correct, runnable epidemic models when prompted with epidemiological scenarios
2. Characterise the types of errors that occur when models are generated without domain-specific tooling
3. Evaluate whether providing validated composable components (EpiAware) improves model correctness
4. Compare performance across different LLMs, including open-source models relevant to low- and middle-income country (LMIC) settings

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

Real data is used to test whether generated models can handle actual epidemic dynamics. No numerical benchmark is required; evaluation focuses on model correctness and plausibility of outputs.

## Evaluation Criteria

### Objective Criteria (Automated)

| Criterion | Measurement |
|-----------|-------------|
| **Syntactic validity** | Does the code parse without errors? (0/1) |
| **Execution** | Does the model run on the test data? (0/1) |
| **Convergence** | Does MCMC sampling converge? (R-hat < 1.05, ESS > 400) (0/1) |
| **Plausibility** | Are posterior Rt estimates within [0.1, 10]? (0/1) |

### Expert Review (Departure-Based Assessment)

Independent infectious disease modeller, blinded to condition, assesses each submission against a reference solution.

#### Reference Solutions

A reference solution will be provided for each scenario, implementing the renewal equation approach with:
- Correct generation interval convolution
- Appropriate delay distribution handling
- Suitable observation model
- Reasonable prior specifications
- Proper initial condition handling

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

- Reviewer is an infectious disease modeller not involved in prompt construction
- Reviewer is blinded to: which LLM generated the code, which condition it belongs to
- Reviewer assesses each code sample against the reference solution
- Departures are documented and classified
- Summary scores computed

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
