# Turing.jl API Reference

API documentation for the [Turing.jl](https://turing.ml) probabilistic programming language for Julia.

Turing.jl provides a domain-specific language for defining Bayesian models and a suite of MCMC and variational inference algorithms for them. Models are written with the `@model` macro from `using Turing`. Inference is done by calling `sample`. Standard utilities for posterior diagnostics come from `MCMCChains`.

This reference is API-level only; end-to-end models for any specific application are not provided.

---

## Model definition

### `@model`

```julia
using Turing

@model function name(args...; kwargs...)
    # body
end
```

Defines a probabilistic model. The body uses two special operators on random variables:

- `x ~ Distribution(...)` — declare `x` as a random variable distributed as `Distribution`. If `x` was passed in as data (an actual numeric value, not `missing`), it is treated as observed; otherwise it is sampled.
- `x := expression` — declare `x` as a deterministic transformation of other variables. Recorded in the resulting `Chains` for output but does not contribute to the joint density.

Calling the model function returns a `DynamicPPL.Model` value, not the body's return value. The model object is then passed to `sample`, `predict`, etc.

### Distributions

Distributions come from the `Distributions.jl` package. Re-exported by Turing. Standard distributions on the real line (`Normal`, `Cauchy`, `LogNormal`, `Beta`, `Gamma`, `Exponential`, `Uniform`, `Truncated`), positive distributions (`HalfNormal`, `HalfCauchy`), discrete distributions (`Bernoulli`, `Binomial`, `Poisson`, `NegativeBinomial`, `Geometric`, `Categorical`), and multivariate distributions (`MvNormal`, `Dirichlet`, `Multinomial`) are all available. Truncate any continuous distribution with `truncated(d, lo, hi)`.

### `@submodel`

```julia
@submodel x = sub_model(args...)
@submodel prefix="name" x = sub_model(args...)
```

Embed one Turing model inside another. The submodel's random variables become part of the outer model. The optional `prefix` namespaces the submodel's variables in the resulting `Chains` so multiple submodels do not collide.

### `arraydist` and `filldist`

```julia
arraydist([d1, d2, ...])     # vector of independent draws from heterogeneous distributions
filldist(d, n)               # n iid draws from a single distribution
filldist(d, n1, n2)          # n1×n2 array of iid draws
```

Useful for vectorised sampling. Returns a `Distribution` object that can be used with `~`.

---

## Inference

### `sample`

```julia
sample(model, sampler, n; kwargs...)
sample(model, sampler, parallel, n_per_chain, n_chains; kwargs...)
```

Run MCMC inference. Returns an `MCMCChains.Chains` object.

Arguments:

- `model::DynamicPPL.Model`: the model returned by calling the `@model` function.
- `sampler`: a sampler instance (see below).
- `n`: number of samples per chain.
- `parallel`: parallelisation strategy: `MCMCSerial()`, `MCMCThreads()`, or `MCMCDistributed()`.
- `n_per_chain`, `n_chains`: when running multiple chains.

Common keyword arguments:

- `progress::Bool = true`: show a progress bar.
- `discard_initial::Int = 0`: drop this many warm-up samples (separate from sampler-internal adaptation).
- `nadapts::Int`: number of adaptation steps for NUTS/HMC. Defaults to half of `n`.
- `initial_params`: initial values for the parameters.

### Samplers

```julia
NUTS(target_accept_rate::Float64 = 0.65; adtype = AutoForwardDiff(), max_depth = 10)
HMC(ϵ::Float64, n_leapfrog::Int; adtype = AutoForwardDiff())
HMCDA(target_accept::Float64, λ::Float64; adtype = AutoForwardDiff())
MH()
MH(:variable_name => proposal_distribution, ...)
IS()                          # importance sampling
SMC()                         # sequential Monte Carlo
PG(n_particles::Int)          # particle Gibbs
Gibbs((:x => NUTS(), :y => MH()))   # composite sampler
```

NUTS is the default for continuous-parameter models.

### Automatic differentiation

The `adtype` keyword on gradient-based samplers selects the AD backend:

- `AutoForwardDiff()` — uses `ForwardDiff.jl`. Robust default; scales O(n) with parameter dimension.
- `AutoReverseDiff()` — uses `ReverseDiff.jl`. `AutoReverseDiff(compile = true)` compiles the gradient tape for repeated evaluation; faster on high-dimensional models. **`using ReverseDiff` and `using LogDensityProblemsAD` must be in the script for this to activate; otherwise sampling fails with `MethodError: ADgradient(::Val{:ReverseDiff}, ...)`.**
- `AutoZygote()` — uses `Zygote.jl`. Reverse-mode; supports more code patterns than `ReverseDiff`.

### Variational inference

```julia
using Turing.Variational
q = vi(model, ADVI(n_samples_elbo, n_steps))
```

Returns a posterior approximation (a sampleable distribution).

---

## Working with results

### `Chains`

`sample` returns an `MCMCChains.Chains` object. Common operations (with `using MCMCChains` re-exported by Turing):

```julia
chain[:variable_name]     # array of samples for one variable
chain[start:stop]         # restrict to a range of iterations
chain[1:2:end]            # thin
summarystats(chain)       # mean, std, MCSE, ESS, R-hat per parameter
quantile(chain)           # default 0.025, 0.25, 0.5, 0.75, 0.975 quantiles
DataFrame(chain)          # convert to DataFrame (requires DataFrames)
```

For posterior intervals, extract the parameter samples and call `quantile.(eachcol(...), p)` from `Statistics`, or use `quantile(chain; q = [0.05, 0.95])`.

### `generated_quantities`

```julia
gen = generated_quantities(model, chain)
```

Re-runs the model body for each posterior sample, returning the model's *return value* per draw. Useful for post-hoc computation of derived quantities. The result is a vector (or array) where each element corresponds to one posterior draw.

If the original model's `return` value depends on observed data, you can replace observations with `missing` to get prior-predictive or posterior-predictive draws:

```julia
predictive_model = original_model_function(args..., y_t = fill(missing, n))
gen = generated_quantities(predictive_model, chain)
```

### `predict`

```julia
predicted = predict(model_with_missing_data, chain)
```

Convenience wrapper for posterior-predictive sampling. Returns a `Chains` object.

### `condition`, `fix`, `decondition`

```julia
conditioned = condition(model, x = 1.0, y = 2.0)
fixed       = fix(model, x = 1.0)
free        = decondition(conditioned)
```

`condition` and `fix` both pin variables to specified values, but they treat the joint density differently: `condition` includes the variable's log-density contribution, `fix` does not. For inference, both produce equivalent posterior samples for the remaining unfixed variables.

---

## Logging and diagnostics

```julia
using Statistics
using MCMCChains
ess(chain)                # effective sample size per parameter
rhat(chain)               # potential scale reduction factor
DataFrame(summarize(chain, mean, std, MCSE, ESS, rhat))
```

Convergence is typically assessed via R-hat ≈ 1 and ESS at least a few hundred. Divergent transitions in NUTS appear in the chain's internal info as the `:numerical_error` column; persistent divergences indicate a poorly-conditioned posterior and may benefit from reparameterisation.

---

## Common patterns

### Vectorised observations

```julia
@model function obs_model(y)
    σ ~ HalfNormal(1.0)
    μ ~ Normal(0, 1)
    y .~ Normal(μ, σ)
end
```

Broadcasting `~` is supported when the right-hand side is a single distribution. Equivalent to a loop with `y[i] ~ Normal(μ, σ)`.

### Hierarchical priors

```julia
@model function hier(x)
    μ_top ~ Normal(0, 1)
    σ_top ~ HalfNormal(1.0)
    n = length(x)
    μ ~ filldist(Normal(μ_top, σ_top), n)
    x .~ Normal.(μ, 0.1)
end
```

### Multiple observations of different distributions

```julia
@model function multi(y1, y2)
    p ~ Beta(1, 1)
    λ ~ HalfNormal(1.0)
    y1 .~ Bernoulli(p)
    y2 .~ Poisson(λ)
end
```

### Reparameterisation

For variables on a constrained support, sample on an unconstrained scale and transform. Turing handles this automatically for standard constrained distributions, but for custom transforms:

```julia
log_x ~ Normal(0, 1)
x = exp(log_x)
```

`~` does the constrained → unconstrained transform internally for sampling efficiency.
