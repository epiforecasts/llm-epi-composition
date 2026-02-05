# Scenario 3: Multiple Data Streams

## Task

Estimate the time-varying reproduction number (Rt) jointly from three data streams: cases, hospitalisations, and deaths.

## Background

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function.

Different data streams provide complementary information about the underlying infection dynamics:
- **Cases**: Most timely but subject to testing behaviour and ascertainment changes
- **Hospitalisations**: More reliable denominator but delayed relative to infection
- **Deaths**: Most complete ascertainment but longest delay

Each stream has its own delay from infection to observation, ascertainment rate, and observation noise characteristics.

## Data

Daily COVID-19 observations from England (`data/observations.csv`):
- `date`: Specimen/event date, in YYYY-MM-DD format
- `cases`: Number of confirmed cases (by specimen date)
- `hospitalisations`: Number of hospital admissions
- `deaths`: Number of deaths

Note: Each data stream has different delays from infection to observation. Recent data may be incomplete due to reporting delays.

## Requirements

1. Write complete, runnable code that estimates a single shared Rt over time using the renewal equation
2. The model should include:
   - The renewal equation for infection dynamics with a **shared Rt**
   - **Stream-specific delays**: Each observation type has its own delay from infection
   - **Stream-specific ascertainment**: Each stream has a proportion of infections observed
   - **Overdispersion**: Account for greater variance than Poisson in observations
   - **Smoothness constraint on Rt**: Rt should vary smoothly over time
3. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates and stream-specific parameters
4. Handle the initial infection seeding period appropriately

## Output

Your code should produce:
- Rt estimates for each time point (the full trajectory)
- The current (most recent) Rt estimate with uncertainty
- Stream-specific ascertainment rate estimates
- A summary or plot showing results

Save all results and any other output that may be useful to the user.

## Language

Use EpiAware.jl components in Julia (requires Julia 1.11 or later). The EpiAware documentation is provided below.

---

# EpiAware.jl Documentation Context

## Overview

EpiAware is a modular, composable toolkit for infectious disease modelling built on Julia's Turing.jl probabilistic programming language. It provides independently composable components that can be mixed and matched to build epidemic models.

## Package Structure

EpiAware.jl consists of several submodules:

- **EpiAwareBase**: Core abstract types and interfaces
- **EpiLatentModels**: Latent process components (random walks, AR, etc.)
- **EpiInfModels**: Infection/transmission dynamics (renewal equation, etc.)
- **EpiObsModels**: Observation and measurement models
- **EpiInference**: Inference methods (MCMC sampling, etc.)
- **EpiAwareUtils**: Utility functions

## Key Components

### 1. Latent Models (`AbstractLatentModel`)

Latent models generate time-varying parameters like log(Rt). Available types:

```julia
# AR process for log(Rt)
ar = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],  # AR coefficients
    init_priors = [Normal(0.0, 0.5)],                    # Initial values
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Noise
)

# Random walk
rw = RandomWalk(
    init_prior = Normal(0.0, 0.5),
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Fixed intercept (constant value)
intercept = FixedIntercept(0.0)
```

Generate samples with:
```julia
latent_mdl = generate_latent(ar, n_timepoints)
Z_t = latent_mdl()  # Returns latent process values
```

### 2. Infection Models (`AbstractEpiModel`)

Infection models generate latent infections from Rt. The `Renewal` model implements the renewal equation:

```julia
# Define generation interval distribution
gen_distribution = Gamma(6.5, 0.62)

# Create EpiData with generation interval
model_data = EpiData(gen_distribution = gen_distribution)

# Create renewal model
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))
```

Generate infections with:
```julia
inf_mdl = generate_latent_infs(epi, log_Rt)  # log_Rt is the latent process
infections = inf_mdl()
```

### 3. Observation Models (`AbstractObservationModel`)

Observation models link latent infections to observed data:

```julia
# Negative binomial errors (for overdispersed count data)
obs = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))

# Poisson errors
obs = PoissonError()

# Add delay distribution
delay_distribution = Gamma(5.0, 1.0)
obs_with_delay = LatentDelay(obs, delay_distribution)
```

Generate observations with:
```julia
obs_mdl = generate_observations(obs, y_t, expected_cases)
# y_t = missing for prior predictive, or actual data for inference
```

### 4. Composing Models with `EpiProblem`

The `EpiProblem` constructor combines components into a full model:

```julia
epi_prob = EpiProblem(
    epi_model = epi,           # Infection model (Renewal)
    latent_model = ar,         # Latent process for log(Rt)
    observation_model = obs,   # Observation model
    tspan = (1, 100)           # Time span
)
```

### 5. Inference

Generate a Turing model and run inference:

```julia
# Generate Turing model
mdl = generate_epiaware(epi_prob, (y_t = observed_cases,))

# Define inference method
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 2000,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

# Run inference
results = apply_method(mdl, inference_method, (y_t = observed_cases,))
```

## Complete Example: Basic Renewal Model

```julia
using EpiAware
using Distributions
using Turing

# 1. Define generation interval
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# 2. Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# 3. Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# 4. Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# 5. Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(cases))
)

# 6. Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))

# 7. Extract Rt estimates
# The latent process Z_t = log(Rt), so Rt = exp(Z_t)
```

## Additional Components

### Combining Latent Models

```julia
# Add day-of-week effects
dayofweek_effect = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Combine with AR process
combined = CombineLatentModels([ar, dayofweek_effect])
```

### Ascertainment

```julia
# Time-varying ascertainment
ascertainment_model = Ascertainment(
    obs,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)
```

### Multiple Observation Streams

```julia
# Stack multiple observation models
stacked_obs = StackObservationModels((
    cases = obs_cases,
    deaths = obs_deaths
))
```
