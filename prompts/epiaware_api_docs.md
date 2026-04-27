# EpiAware API Reference

API documentation for the EpiAware Julia package (CDCgov/Rt-without-renewal).

EpiAware is a compositional framework for epidemiological modelling built on top of the Turing probabilistic programming language. Models are composed from three main components: an **infection model** (how infections arise), a **latent model** (the underlying time-varying process), and an **observation model** (how infections become observed data). These are combined into an `EpiProblem` and solved with `apply_method`.

**Required imports for sampling.** `apply_method` (and the underlying `Turing.NUTS`) uses `AutoReverseDiff(compile=true)` by default. The corresponding `LogDensityProblemsAD` package extension only activates when `ReverseDiff` and `LogDensityProblemsAD` are loaded explicitly. A working sampling script therefore needs:

```julia
using EpiAware
using Turing
using Distributions
using ReverseDiff
using LogDensityProblemsAD
```

If `ReverseDiff` is not loaded, sampling fails with `MethodError: no method matching ADgradient(::Val{:ReverseDiff}, ...)`.

---

## EpiAwareBase

Core types, the problem definition, and the top-level inference entry points.

### Abstract type hierarchy

```
AbstractModel
  AbstractEpiModel
    AbstractTuringEpiModel
      AbstractTuringRenewal
  AbstractLatentModel
    AbstractTuringLatentModel
      AbstractTuringIntercept
  AbstractObservationModel
    AbstractTuringObservationModel
      AbstractTuringObservationErrorModel

AbstractEpiProblem
AbstractEpiMethod
  AbstractEpiOptMethod
  AbstractEpiSamplingMethod

AbstractBroadcastRule
AbstractAccumulationStep
```

### `EpiProblem`

Defines the full inference/generative modelling problem.

```julia
@kwdef struct EpiProblem{E, L, O} <: AbstractEpiProblem
    epi_model::E       # Epidemiological model for unobserved infections
    latent_model::L    # Latent model for the underlying latent process
    observation_model::O  # Observation model for observed cases
    tspan::Tuple{Int, Int}  # Time span (start, end) for the case time series
end
```

**Fields:**
- `epi_model`: An infection-generating model (e.g. `Renewal`, `DirectInfections`, `ExpGrowthRate`).
- `latent_model`: A latent process model (e.g. `RandomWalk`, `AR`).
- `observation_model`: An observation model (e.g. `NegativeBinomialError`, `LatentDelay`).
- `tspan`: Tuple `(start_time, end_time)` defining the modelling window. The number of time steps is `tspan[2] - tspan[1] + 1`.

### `EpiMethod`

Combines pre-sampler optimisation steps with a sampler method.

```julia
@kwdef struct EpiMethod{O <: AbstractEpiOptMethod, S <: AbstractEpiSamplingMethod}
    pre_sampler_steps::Vector{O}  # Optimisation steps run before sampling
    sampler::S                     # The sampling method
end
```

The pre-sampler steps (e.g. `ManyPathfinder`) are run sequentially and their results are passed as initialisation to the sampler (e.g. `NUTSampler`).

### `apply_method`

Top-level function to run inference.

```julia
apply_method(epiproblem::EpiProblem, method::AbstractEpiMethod, data;
    fix_parameters::NamedTuple = NamedTuple(),
    condition_parameters::NamedTuple = NamedTuple(),
    kwargs...)
```

**Arguments:**
- `epiproblem`: The `EpiProblem` to solve.
- `method`: An `EpiMethod` (or other `AbstractEpiMethod`).
- `data`: Observed data. Must have a `y_t` field (the observations vector, or `missing` for generative mode).
- `fix_parameters`: Parameters to fix to specific values (uses `DynamicPPL.fix`).
- `condition_parameters`: Parameters to condition on (uses `DynamicPPL.condition`).

**Returns:** An `EpiAwareObservables` with fields:
- `model`: The Turing model.
- `data`: The input data.
- `samples`: The inference result (e.g. `MCMCChains.Chains`).
- `generated`: Generated quantities — an iterable of `NamedTuple`s (indexed `[iteration, chain]` when from MCMC). Each element has fields `generated_y_t`, `I_t`, and `Z_t`.

**Extracting results from `generated`:**

```julia
result = apply_method(problem, method, (y_t = data,))

# Extract infection trajectories (Matrix: time_steps × n_samples)
I_t_samples = mapreduce(hcat, result.generated) do gen
    gen.I_t
end

# Extract latent process Z_t (Matrix: time_steps × n_samples)
Z_t_samples = mapreduce(hcat, result.generated) do gen
    gen.Z_t
end

# For Renewal models: Rt = transformation(Z_t), e.g. Rt = exp.(Z_t)
Rt_samples = exp.(Z_t_samples)   # if transformation = exp

# Compute posterior median and credible intervals
using Statistics
Rt_median = mapslices(median, Rt_samples, dims=2)
Rt_lower = mapslices(x -> quantile(x, 0.025), Rt_samples, dims=2)
Rt_upper = mapslices(x -> quantile(x, 0.975), Rt_samples, dims=2)
```

### `generate_epiaware`

The core Turing model that composes the three sub-models.

```julia
@model function generate_epiaware(y_t, time_steps, epi_model;
    latent_model, observation_model)
```

This model:
1. Generates the latent process `Z_t` from `latent_model` (prefixed `latent.`).
2. Transforms `Z_t` into infections `I_t` via `epi_model`.
3. Generates observations from `observation_model` given `I_t` (prefixed `obs.`).
4. Returns `(; generated_y_t, I_t, Z_t)` as generated quantities.

### Key dispatch functions

- `generate_latent(latent_model, n)` -- generates a latent process of length `n`.
- `generate_latent_infs(epi_model, Z_t)` -- generates infections given latent process `Z_t`.
- `generate_observations(obs_model, y_t, Y_t)` -- generates observations given expected values `Y_t`.

### `condition_model`

```julia
condition_model(model::Model, fix_parameters::NamedTuple, condition_parameters::NamedTuple)
```

Applies `DynamicPPL.fix` and `DynamicPPL.condition` to a Turing model. Use `fix` for parameters you want to set to exact values (they will not be sampled). Use `condition` for observed data.

---

## EpiInfModels

Models for generating unobserved/latent infections from a latent process.

### `EpiData`

Stores the discrete generation interval and a transformation function.

```julia
struct EpiData{T <: Real, F <: Function}
    gen_int::Vector{T}         # Discrete generation interval PMF (must sum to 1, non-negative)
    len_gen_int::Integer       # Length of the generation interval
    transformation::F          # Transformation function (e.g. exp)
end
```

**Constructors:**

```julia
# From a discrete PMF and transformation function
EpiData(gen_int::Vector, transformation::Function)

# From a continuous distribution (double-interval censoring discretisation)
EpiData(; gen_distribution::ContinuousDistribution,
    D_gen = nothing,    # Right truncation point (default: nothing)
    Δd = 1.0,           # Interval width
    transformation::Function = exp)
```

The `transformation` field defines the mapping from unconstrained to constrained space. For the `Renewal` model, `transformation` is applied to the latent process to produce Rt, and also to the initial incidence parameter. **Use `exp` (the default) for standard Renewal models.** Using `identity` will produce unconstrained (potentially negative) Rt values.

When constructing from a continuous distribution, the generation interval is discretised using double-interval censoring (via `censored_pmf`), and the first element (representing delay 0) is removed and the remainder renormalised.

### `Renewal`

Renewal model where infections depend on recent infections weighted by the generation interval, scaled by a time-varying reproduction number Rt.

```julia
struct Renewal{E, S <: Sampleable, A}
    data::E                    # EpiData object
    initialisation_prior::S    # Prior for initial (unconstrained) incidence
    recurrent_step::A          # Step function for the renewal equation
end
```

**Mathematical specification:**

```
Rt = transformation(Z_t)
I_t = Rt * sum(I_{t-i} * g_i, i=1..n)    for t >= 1
I_t = transformation(I_0) * exp(r(R_1) * t)    for t <= 0  (seeding)
```

where `r(R_1)` is the exponential growth rate implied by `R_1`.

**Constructors:**

```julia
Renewal(data::EpiData; initialisation_prior = Normal())
Renewal(; data::EpiData, initialisation_prior = Normal())
Renewal(data, initialisation_prior, recurrent_step)  # Full constructor
```

**Sampled parameters** (in `generate_latent_infs`):
- `init_incidence ~ initialisation_prior` -- initial incidence on unconstrained scale.

### `DirectInfections`

Infections are a direct transformation of the latent process (no renewal equation).

```julia
@kwdef struct DirectInfections{S <: Sampleable}
    data::EpiData
    initialisation_prior::S = Normal()
end
```

**Mathematical specification:** `I_t = transformation(I_0 + Z_t)`

**Sampled parameters:**
- `init_incidence ~ initialisation_prior`

### `ExpGrowthRate`

Infections follow an exponential growth model.

```julia
@kwdef struct ExpGrowthRate{S <: Sampleable}
    data::EpiData
    initialisation_prior::S = Normal()
end
```

**Mathematical specification:** `I_t = transformation(I_0) * exp(cumsum(Z_t))`

**Sampled parameters:**
- `init_incidence ~ initialisation_prior`

### `ODEProcess`

Infections generated from an ODE model (e.g. SIR, SEIR).

```julia
@kwdef struct ODEProcess{P, S, F, D}
    params::P          # AbstractTuringLatentModel defining ODE params and problem
    solver::S = AutoVern7(Rodas5())   # ODE solver
    sol2infs::F        # Function mapping ODESolution to infection counts
    solver_options::D = Dict(:verbose => false, :saveat => 1.0)
end
```

Requires `params` to have a `prob` field containing an `ODEProblem`, and a `generate_latent` method that returns `(u0, p)`. Predefined parameter types `SIRParams` and `SEIRParams` are available.

### Utility functions

```julia
# Convert reproduction number to exponential growth rate
R_to_r(R_0, w::Vector; newton_steps=2, Δd=1.0)
R_to_r(R_0, epi_model; newton_steps=2, Δd=1.0)

# Convert exponential growth rate to reproduction number
r_to_R(r, w::Vector)

# Calculate expected Rt from EpiData and infections
expected_Rt(data::EpiData, infections::Vector)
```

---

## EpiLatentModels

Models for the underlying latent process `Z_t`. All implement `generate_latent(model, n)` returning a vector of length `n`.

### `RandomWalk`

```julia
@kwdef struct RandomWalk{D <: Sampleable, E <: AbstractTuringLatentModel}
    init_prior::D = Normal()
    ϵ_t::E = HierarchicalNormal()
end
```

**Mathematical specification:** `Z_t = Z_0 + sigma * sum(epsilon_1..t)`

The noise model `ϵ_t` defaults to `HierarchicalNormal()` which samples a standard deviation and scales IID standard normal draws.

**Sampled parameters:**
- `rw_init ~ init_prior`
- Parameters from `ϵ_t` sub-model

### `AR`

Autoregressive model of order `p`.

```julia
struct AR{D, I, P <: Int, E}
    damp_prior::D      # Prior for damping coefficients (length p)
    init_prior::I       # Prior for initial conditions (length p)
    p::P                # Order of the AR model
    ϵ_t::E             # Error term model
end
```

**Constructors:**

```julia
# Scalar priors replicated p times
AR(damp_prior, init_prior; p=1, ϵ_t=HierarchicalNormal())

# Vector priors (p inferred from length)
AR(; damp_priors=[truncated(Normal(0.0, 0.05), 0, 1)],
     init_priors=[Normal()],
     ϵ_t=HierarchicalNormal())
```

**Default priors:**
- `damp_priors`: `[truncated(Normal(0.0, 0.05), 0, 1)]` (AR(1) with small positive damping)
- `init_priors`: `[Normal()]`
- `ϵ_t`: `HierarchicalNormal()`

**Sampled parameters:**
- `ar_init ~ init_prior`
- `damp_AR ~ damp_prior`
- Parameters from `ϵ_t` sub-model

### `MA`

Moving average model of order `q`.

```julia
struct MA{C, Q <: Int, E}
    θ::C        # Prior for MA coefficients (length q)
    q::Q        # Order of the MA model
    ϵ_t::E      # Error term model
end
```

**Constructors:**

```julia
MA(θ_dist; q=1, ϵ_t=HierarchicalNormal())
MA(; θ_priors=[truncated(Normal(0.0, 0.05), -1, 1)],
     ϵ_t=HierarchicalNormal())
```

**Sampled parameters:**
- `θ ~ θ` (MA coefficients)
- Parameters from `ϵ_t` sub-model

### `HierarchicalNormal`

Non-centred hierarchical normal. Generates `n` values as `mean + std * epsilon_t` where `std` is sampled from a prior and `epsilon_t` are IID standard normal.

```julia
@kwdef struct HierarchicalNormal{R, D, M}
    mean::R = 0.0
    std_prior::D = truncated(Normal(0, 0.1), 0, Inf)
    add_mean::M = (mean != 0)   # Automatically set
end
```

**Constructors:**

```julia
HierarchicalNormal()                           # mean=0, std_prior=truncated(Normal(0,0.1), 0, Inf)
HierarchicalNormal(std_prior::Distribution)    # mean=0, custom std_prior
HierarchicalNormal(mean, std_prior)            # Custom mean and std_prior
```

**Sampled parameters:**
- `std ~ std_prior`
- IID standard normal `ϵ_t` (length `n`)

### `IID`

Independent and identically distributed draws.

```julia
@kwdef struct IID{D <: Sampleable}
    ϵ_t::D = Normal(0, 1)
end
```

**Sampled parameters:**
- `ϵ_t ~ filldist(ϵ_t, n)` (n IID draws)

### `Intercept`

A constant (sampled) intercept broadcast to length `n`.

```julia
@kwdef struct Intercept{D <: Sampleable}
    intercept_prior::D
end
```

**Sampled parameters:**
- `intercept ~ intercept_prior`

Returns `fill(intercept, n)`.

### `FixedIntercept`

A fixed (non-random) intercept.

```julia
@kwdef struct FixedIntercept{F <: Real}
    intercept::F
end
```

Returns `fill(intercept, n)` with no sampled parameters.

### `Null`

Generates `nothing` as the latent process (no latent variables).

```julia
struct Null <: AbstractTuringLatentModel end
```

### Modifiers

#### `DiffLatentModel`

Wraps a latent model and applies `d`-fold differencing: the inner model generates the differenced process, which is then integrated (cumulative sum) `d` times.

```julia
struct DiffLatentModel{M, P}
    model::M        # Underlying latent model for the differenced process
    init_prior::P   # Prior for the d initial values
    d::Int          # Number of times differenced
end
```

**Constructors:**

```julia
# Common prior for all d initial values
DiffLatentModel(model, init_prior_distribution; d=1)

# Vector of priors (d inferred from length)
DiffLatentModel(; model, init_priors=[Normal()])
```

**Sampled parameters:**
- `latent_init ~ init_prior` (d initial values)
- Parameters from `model` sub-model

The output is of length `n`: the inner model generates `n - d` values, which are concatenated with the `d` initial values and cumulatively summed `d` times.

#### `TransformLatentModel`

Applies a deterministic transformation to the output of another latent model.

```julia
@kwdef struct TransformLatentModel{M, F}
    model::M        # The latent model to transform
    transform::F    # The transformation function
end
```

The `transform` function receives the full vector output of the inner model and should return a vector.

#### `PrefixLatentModel`

Wraps a latent model and prefixes all its parameter names.

```julia
@kwdef struct PrefixLatentModel{M, P <: String}
    model::M
    prefix::P
end
```

This is used internally by `CombineLatentModels` and `ConcatLatentModels` to avoid parameter name collisions.

#### `RecordExpectedLatent`

Wraps a latent model and records its output using Turing's `:=` syntax as `exp_latent`.

```julia
struct RecordExpectedLatent{M}
    model::M
end
```

### Manipulators

#### `CombineLatentModels`

Adds the outputs of multiple latent models element-wise.

```julia
struct CombineLatentModels{M, P}
    models::M       # Vector of latent models
    prefixes::P     # Vector of prefix strings
end
```

**Constructors:**

```julia
# Auto-generated prefixes: "Combine.1", "Combine.2", ...
CombineLatentModels(models::Vector{<:AbstractTuringLatentModel})

# Custom prefixes
CombineLatentModels(models, prefixes)
```

Each model generates a vector of length `n` and the results are summed element-wise.

#### `ConcatLatentModels`

Concatenates the outputs of multiple latent models into a single vector.

```julia
struct ConcatLatentModels{M, N, F, P}
    models::M              # Vector of latent models
    no_models::N           # Number of models
    dimension_adaptor::F   # Function(n, m) -> Vector{Int} specifying how to split n among m models
    prefixes::P            # Vector of prefix strings
end
```

**Constructors:**

```julia
# Default equal_dimensions adaptor, auto prefixes "Concat.1", "Concat.2", ...
ConcatLatentModels(models)

# Custom dimension adaptor
ConcatLatentModels(models, dimension_adaptor; prefixes=nothing)
```

The `dimension_adaptor` function takes `(n, m)` and returns a vector of `m` integers summing to `n`. The default `equal_dimensions` divides `n` as evenly as possible among `m` models.

#### `BroadcastLatentModel`

Generates a short latent process and broadcasts it to length `n` using a rule.

```julia
struct BroadcastLatentModel{M, P <: Integer, B <: AbstractBroadcastRule}
    model::M             # The underlying latent model
    period::P            # The broadcast period
    broadcast_rule::B    # RepeatEach or RepeatBlock
end
```

**Constructors:**

```julia
BroadcastLatentModel(model, period, broadcast_rule)
BroadcastLatentModel(model; period, broadcast_rule)
```

**Broadcast rules:**

- `RepeatEach()`: The inner model generates `period` values, then they are repeated cyclically to length `n`. Useful for day-of-week effects (period=7).
- `RepeatBlock()`: The inner model generates `ceil(n/period)` values, each repeated `period` times. Useful for piecewise-constant (e.g. weekly) processes.

**Helper functions:**

```julia
# Day-of-week broadcast with softmax link (effects sum to 7)
broadcast_dayofweek(model; link = x -> 7 * softmax(x))

# Piecewise-constant weekly broadcast
broadcast_weekly(model)
```

### Combination constructors

```julia
# ARMA(p, q) model -- AR with MA as its error term
arma(; init=[Normal()], damp=[truncated(Normal(0.0, 0.05), 0, 1)],
       θ=[truncated(Normal(0.0, 0.05), -1, 1)], ϵ_t=HierarchicalNormal())

# ARIMA(p, d, q) model -- ARMA wrapped in DiffLatentModel
arima(; ar_init=[Normal()], diff_init=[Normal()],
        damp=[truncated(Normal(0.0, 0.05), 0, 1)],
        θ=[truncated(Normal(0.0, 0.05), -1, 1)], ϵ_t=HierarchicalNormal())
```

---

## EpiObsModels

Models for linking expected infections to observed data.

### Observation Error Models

These are leaf-level models that define the likelihood. They implement `generate_observations(model, y_t, Y_t)` where `y_t` is observed data (or `missing`) and `Y_t` is the expected observation vector.

The base `generate_observations` for all `AbstractTuringObservationErrorModel` subtypes:
- Pads `Y_t` with `1e-6` to avoid numerical issues.
- Supports `y_t` being `missing` (generative mode) or a vector of observations.
- Supports `Y_t` being shorter than `y_t` (aligns to the end of `y_t`).

#### `NegativeBinomialError`

```julia
@kwdef struct NegativeBinomialError{S <: Sampleable}
    cluster_factor_prior::S = HalfNormal(0.01)
end
```

Uses a mean-cluster parameterisation of the negative binomial where variance = `mu + alpha^2 * mu^2`. The `cluster_factor` (alpha) is sampled from `cluster_factor_prior`.

**Sampled parameters:**
- `cluster_factor ~ cluster_factor_prior`

Each observation: `y_t[i] ~ NegativeBinomialMeanClust(Y_t[i] + 1e-6, cluster_factor^2)`

where `NegativeBinomialMeanClust(mu, alpha)` parameterises: `r = 1/alpha`, `p = mu / (mu + alpha * mu^2)`.

#### `PoissonError`

```julia
struct PoissonError <: AbstractTuringObservationErrorModel end
```

No sampled parameters. Each observation: `y_t[i] ~ SafePoisson(Y_t[i] + 1e-6)`.

### Observation Model Modifiers

#### `LatentDelay`

Convolves expected infections with a delay distribution before passing to the inner observation model. The output is shortened by the length of the delay PMF.

```julia
struct LatentDelay{M, T}
    model::M       # Underlying observation model
    rev_pmf::T     # Reversed delay PMF (stored internally)
end
```

**Constructors:**

```julia
# From a discrete PMF
LatentDelay(model, pmf::Vector)

# From a continuous distribution (double-interval censoring)
LatentDelay(model, distribution::ContinuousDistribution; D=nothing, Δd=1.0)
```

When `D = nothing`, the distribution is truncated at its 99th percentile (rounded to nearest `Δd`).

The delay convolution shortens the expected observation vector, so observations before the delay window are not fitted.

#### `Ascertainment`

Modifies expected observations using a latent model (e.g. for time-varying ascertainment).

```julia
struct Ascertainment{M, T, F, P}
    model::M             # Underlying observation model
    latent_model::T      # Latent model for ascertainment
    transform::F         # Function(Y_t, latent_output) -> modified Y_t
    latent_prefix::P     # Prefix for latent model parameters (default: "Ascertainment")
end
```

**Constructors:**

```julia
Ascertainment(model, latent_model;
    transform = (x, y) -> xexpy.(x, y),   # Default: Y_t * exp(latent)
    latent_prefix = "Ascertainment")

Ascertainment(; model, latent_model,
    transform = (x, y) -> xexpy.(x, y),
    latent_prefix = "Ascertainment")
```

The default `transform` multiplies the expected observations by `exp(latent_output)`. The `xexpy(x, y)` function computes `x * exp(y)` in a numerically stable way.

**Helper function:**

```julia
# Day-of-week ascertainment with softmax link
ascertainment_dayofweek(model;
    latent_model = HierarchicalNormal(),
    transform = (x, y) -> x .* y,
    latent_prefix = "DayofWeek")
```

This wraps the latent model in `broadcast_dayofweek` (period 7, `RepeatEach`, softmax constraint so effects sum to 7) and sets the transform to multiplicative.

#### `Aggregate`

Aggregates observations over specified time periods before passing to the inner model.

```julia
struct Aggregate{M, I, J}
    model::M                    # Underlying observation model
    aggregation::I              # Vector of integers (0 = skip, N = aggregate N days)
    present::J                  # Boolean vector (automatically computed: aggregation .!= 0)
end
```

**Constructor:**

```julia
Aggregate(model, aggregation)
Aggregate(; model, aggregation)
```

The `aggregation` vector is cyclically repeated to match the observation length. For weekly aggregation reported on day 5: `aggregation = [0, 0, 0, 0, 7, 0, 0]`.

#### `TransformObservationModel`

Applies a transformation to expected observations before passing to the inner model.

```julia
@kwdef struct TransformObservationModel{M, F}
    model::M
    transform::F = x -> log1pexp.(x)   # Default: softplus
end
```

The default `log1pexp` (softplus) transformation maps real-valued expected observations to positive values.

#### `PrefixObservationModel`

Wraps an observation model and prefixes all parameter names.

```julia
@kwdef struct PrefixObservationModel{M, P <: String}
    model::M
    prefix::P
end
```

#### `RecordExpectedObs`

Records the expected observations `Y_t` as `exp_y_t` using Turing's `:=` syntax.

```julia
struct RecordExpectedObs{M}
    model::M
end
```

### `StackObservationModels`

Stacks multiple observation models to handle multiple data streams from the same infection process.

```julia
struct StackObservationModels{M, N}
    models::M          # Vector of observation models (wrapped in PrefixObservationModel)
    model_names::N     # Vector of model name strings
end
```

**Constructors:**

```julia
# From a NamedTuple (names become prefixes)
StackObservationModels((cases = PoissonError(), deaths = NegativeBinomialError()))

# From vectors
StackObservationModels(models_vector, names_vector)
```

When calling `generate_observations`, `y_t` must be a `NamedTuple` with keys matching `model_names`. `Y_t` can be either:
- A `NamedTuple` with matching keys (1:1 mapping).
- A single `AbstractVector` (broadcast to all models).

---

## EpiInference

Inference methods for fitting models.

### `ManyPathfinder`

Variational inference using multiple Pathfinder runs, selecting the one with the best ELBO.

```julia
@kwdef struct ManyPathfinder <: AbstractEpiOptMethod
    ndraws::Int = 10       # Number of draws per pathfinder run
    nruns::Int = 4         # Number of parallel pathfinder runs
    maxiters::Int = 100    # Maximum optimisation iterations per run
    max_tries::Int = 100   # Maximum retries if all runs fail
end
```

Runs `nruns` pathfinder instances (in parallel via threads), selects the one with the highest ELBO estimate. If all initial runs fail, retries up to `max_tries` times. Returns a `PathfinderResult`.

### `NUTSampler`

No-U-Turn Sampler (NUTS) for posterior sampling.

```julia
@kwdef struct NUTSampler{A, E, M} <: AbstractEpiSamplingMethod
    target_acceptance::Float64 = 0.8
    adtype::A = AutoForwardDiff()
    mcmc_parallel::E = MCMCSerial()
    nchains::Int = 1
    max_depth::Int = 10
    Δ_max::Float64 = 1000.0
    init_ϵ::Float64 = 0.0
    ndraws::Int                    # REQUIRED: total number of draws
    metricT::M = DiagEuclideanMetric
    nadapts::Int = -1              # -1 uses Turing default (half of ndraws)
end
```

**Fields:**
- `target_acceptance`: Target acceptance probability for NUTS.
- `adtype`: Automatic differentiation backend. Default is `AutoForwardDiff()`.
- `mcmc_parallel`: Parallelisation strategy (`MCMCSerial()`, `MCMCThreads()`, `MCMCDistributed()`).
- `nchains`: Number of MCMC chains.
- `ndraws`: Total number of posterior draws (split across chains: each chain draws `ndraws / nchains`).
- `metricT`: Mass matrix type (`DiagEuclideanMetric` or `DenseEuclideanMetric`).
- `nadapts`: Number of adaptation steps. `-1` means use Turing's default (half of draws per chain).

When `prev_result` is a `PathfinderResult`, initial parameters for each chain are drawn from the Pathfinder approximation.

### `DirectSample`

Samples directly from the prior (no inference).

```julia
@kwdef struct DirectSample <: AbstractEpiSamplingMethod
    n_samples::Union{Int, Nothing} = nothing
end
```

- `n_samples::Int`: Draws `n_samples` from the prior using `Turing.Prior()`, returns `Chains`.
- `n_samples = nothing`: Draws a single sample using `rand(model)`, returns a `NamedTuple`.

---

## EpiAwareUtils

Utility functions used across the package.

### `censored_pmf`

Discretises a continuous distribution into a PMF accounting for interval censoring.

```julia
# Double-interval censoring (default)
censored_pmf(dist::Distribution; Δd=1.0, D=nothing, upper=0.99)

# Single-interval censoring
censored_pmf(dist, Val(:single_censored);
    primary_approximation_point=0.5, Δd=1.0, D)
```

**Arguments:**
- `dist`: A non-negative continuous distribution.
- `Δd`: Width of each censoring interval (default 1.0).
- `D`: Right truncation point. If `nothing`, uses the `upper`th quantile rounded to nearest `Δd`.
- `upper`: Quantile for automatic truncation (default 0.99).

Returns a normalised PMF vector. The double-censored version uses numerical quadrature to convolve the distribution CDF with a uniform on `[0, Δd)`.

### `NegativeBinomialMeanClust`

```julia
NegativeBinomialMeanClust(μ, α)
```

Constructs a `SafeNegativeBinomial` with mean `μ` and cluster factor `α`. The variance-mean relationship is:

```
σ^2 = μ + α * μ^2
```

Parameters: `r = 1/α`, `p = μ / σ^2`.

### `SafePoisson` / `SafeNegativeBinomial`

Numerically safe versions of `Poisson` and `NegativeBinomial` that avoid `InexactError` for very large means. Used internally by the observation error models.

### `accumulate_scan`

```julia
accumulate_scan(acc_step::AbstractAccumulationStep, initial_state, ϵ_t)
```

Efficiently applies a scan/fold operation. Used internally by `Renewal`, `AR`, `RandomWalk`, `MA`, and `LatentDelay` to implement their recurrence relations.
