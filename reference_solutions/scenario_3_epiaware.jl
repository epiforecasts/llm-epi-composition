# Reference Solution: Scenario 3 - Multiple Data Streams
# Using EpiAware.jl
#
# This scenario models cases, hospitalisations, and deaths jointly with:
# - Stream-specific delays
# - Stream-specific ascertainment processes
# - Shared underlying Rt with AR(1) dynamics

using EpiAware
using Distributions
using Turing
using CSV
using DataFrames
using Random

Random.seed!(42)

# ---- Load Data ----
data = CSV.read("data/observations.csv", DataFrame)
cases = data.cases
hospitalisations = data.hospitalisations
deaths = data.deaths
n_days = length(cases)

# ---- Define Epidemiological Parameters ----

# Generation interval (provided in prompt)
gen_int_pmf = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]

# Stream-specific delay distributions (provided in prompt)
delay_cases = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]
delay_hosp = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]
delay_deaths = [0.0, 0.0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.12, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.02]

# ---- Build Model Components ----

# 1. Infection model: Renewal equation (shared across all streams)
model_data = EpiData(gen_int = gen_int_pmf)
epi_model = Renewal(model_data; initialisation_prior = Normal(log(100.0), 2.0))

# 2. Latent model for log(Rt): AR(1) process
#    Shared Rt drives all observation streams
latent_model = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# 3. Observation models for each stream
#    Each stream has:
#    - Its own delay distribution
#    - Its own (fixed) ascertainment rate
#    - Negative binomial observation noise with stream-specific overdispersion

# Cases observation model
obs_cases = Ascertainment(
    LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_cases
    ),
    FixedIntercept(-0.5),  # ~38% ascertainment (logistic(-0.5))
    (Y, asc) -> Y .* logistic.(asc)
)

# Hospitalisations observation model
obs_hosp = Ascertainment(
    LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_hosp
    ),
    FixedIntercept(-3.0),  # ~5% IHR (logistic(-3.0))
    (Y, asc) -> Y .* logistic.(asc)
)

# Deaths observation model
obs_deaths = Ascertainment(
    LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_deaths
    ),
    FixedIntercept(-4.5),  # ~1% IFR (logistic(-4.5))
    (Y, asc) -> Y .* logistic.(asc)
)

# 4. Stack observation models for joint inference
#    StackObservationModels combines multiple streams
obs_model = StackObservationModels((
    cases = obs_cases,
    hospitalisations = obs_hosp,
    deaths = obs_deaths
))

# ---- Compose into EpiProblem ----
epi_prob = EpiProblem(
    epi_model = epi_model,
    latent_model = latent_model,
    observation_model = obs_model,
    tspan = (1, n_days)
)

# ---- Run Inference ----
# Data must be provided as named tuple matching observation model names
obs_data = (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    ),
)

mdl = generate_epiaware(epi_prob, obs_data)

inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 2000,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

@info "Running inference for Scenario 3 (multiple data streams)..."
results = apply_method(mdl, inference_method, obs_data)

# ---- Extract Results ----
missing_data = (
    y_t = (
        cases = fill(missing, n_days),
        hospitalisations = fill(missing, n_days),
        deaths = fill(missing, n_days)
    ),
)

gen_quants = generated_quantities(
    generate_epiaware(epi_prob, missing_data),
    results.samples
)

# Compute Rt quantiles
function compute_quantiles(gen_quants, field; transform = identity, probs = [0.05, 0.5, 0.95])
    samples = mapreduce(hcat, gen_quants) do g
        getfield(g, field) |> transform
    end

    quantiles = mapreduce(hcat, probs) do p
        mapslices(x -> quantile(x, p), samples, dims = 2)
    end

    return quantiles
end

Rt_quantiles = compute_quantiles(gen_quants, :Z_t; transform = exp)

# ---- Output Summary ----
@info "Scenario 3 estimation complete"

summary_df = DataFrame(
    day = 1:n_days,
    Rt_lower = Rt_quantiles[:, 1],
    Rt_median = Rt_quantiles[:, 2],
    Rt_upper = Rt_quantiles[:, 3]
)

println("\nRt estimates from joint model (first 10 days):")
println(first(summary_df, 10))

CSV.write("outputs/scenario_3_rt_estimates.csv", summary_df)
@info "Results saved to outputs/scenario_3_rt_estimates.csv"

# ---- Additional: Compare information from each stream ----
@info "Joint inference uses information from all three streams to estimate shared Rt"
@info "Cases: $(sum(.!ismissing.(cases))) observations"
@info "Hospitalisations: $(sum(.!ismissing.(hospitalisations))) observations"
@info "Deaths: $(sum(.!ismissing.(deaths))) observations"
