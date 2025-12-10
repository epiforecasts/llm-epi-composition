# Reference Solution: Scenario 2 - Structured Rt with Observation Processes
# Using EpiAware.jl
#
# This scenario adds:
# - Day-of-week effects in reporting
# - Time-varying ascertainment
# - Negative binomial observation noise

using EpiAware
using Distributions
using Turing
using CSV
using DataFrames
using Random

Random.seed!(42)

# ---- Load Data ----
data = CSV.read("data/cases.csv", DataFrame)
cases = data.cases
n_days = length(cases)

# ---- Define Epidemiological Parameters ----

# Generation interval (provided in prompt)
gen_int_pmf = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]

# Reporting delay (provided in prompt)
delay_pmf = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]

# ---- Build Model Components ----

# 1. Infection model: Renewal equation
model_data = EpiData(gen_int = gen_int_pmf)
epi_model = Renewal(model_data; initialisation_prior = Normal(log(100.0), 2.0))

# 2. Latent model for log(Rt): AR(1) process
#    This is the "base" Rt process before day-of-week effects
rt_latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# 3. Day-of-week effects on reporting
#    Multiplicative effects for each day of the week
#    Uses BroadcastLatentModel to repeat the 7-day pattern
dayofweek_latent = BroadcastLatentModel(
    HierarchicalNormal(std_prior = HalfNormal(0.3)),  # Day effects
    7,                                                  # 7 days
    RepeatEach()                                        # Repeat pattern
)

# 4. Time-varying ascertainment
#    Random walk on logit scale for proportion of infections reported
ascertainment_latent = RandomWalk(
    init_prior = Normal(-1.0, 0.5),  # Start around ~27% ascertainment
    ϵ_t = IID(Normal(0, 0.05))       # Slow changes in ascertainment
)

# 5. Observation model with delay, day-of-week, and ascertainment
#    Build up the observation model by composing components

# Base: negative binomial with delay
base_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_pmf
)

# Add day-of-week effects
# The day-of-week effect multiplies expected observations
obs_with_dow = Ascertainment(
    base_obs,
    dayofweek_latent,
    (Y, dow) -> Y .* exp.(dow)  # Multiplicative effect on log scale
)

# Add time-varying ascertainment
obs_model = Ascertainment(
    obs_with_dow,
    ascertainment_latent,
    (Y, asc) -> Y .* logistic.(asc)  # Proportion on logistic scale
)

# ---- Compose into EpiProblem ----
epi_prob = EpiProblem(
    epi_model = epi_model,
    latent_model = rt_latent,
    observation_model = obs_model,
    tspan = (1, n_days)
)

# ---- Run Inference ----
mdl = generate_epiaware(epi_prob, (y_t = cases,))

inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 2000,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

@info "Running inference for Scenario 2..."
results = apply_method(mdl, inference_method, (y_t = cases,))

# ---- Extract Results ----
gen_quants = generated_quantities(
    generate_epiaware(epi_prob, (y_t = fill(missing, n_days),)),
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
@info "Scenario 2 estimation complete"

summary_df = DataFrame(
    day = 1:n_days,
    Rt_lower = Rt_quantiles[:, 1],
    Rt_median = Rt_quantiles[:, 2],
    Rt_upper = Rt_quantiles[:, 3]
)

println("\nRt estimates (first 10 days):")
println(first(summary_df, 10))

CSV.write("outputs/scenario_2_rt_estimates.csv", summary_df)
@info "Results saved to outputs/scenario_2_rt_estimates.csv"
