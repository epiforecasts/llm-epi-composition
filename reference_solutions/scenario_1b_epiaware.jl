# Reference Solution: Scenario 1b - Renewal Equation Rt Estimation
# Using EpiAware.jl
#
# This is the reference implementation against which LLM-generated models
# will be evaluated.

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

# ---- Define Epidemiological Parameters ----

# Generation interval (provided in prompt)
gen_int_pmf = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]

# Reporting delay (provided in prompt)
delay_pmf = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]

# ---- Build Model Components ----

# 1. Infection model: Renewal equation
#    Uses the generation interval to model how infections propagate
model_data = EpiData(gen_int = gen_int_pmf)
epi_model = Renewal(model_data; initialisation_prior = Normal(log(100.0), 2.0))

# 2. Latent model: AR(1) process for log(Rt)
#    This provides smoothness in Rt estimates while allowing temporal variation
latent_model = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR(1) coefficient
    init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# 3. Observation model: Negative binomial with delay
#    - Convolves infections with delay distribution to get expected cases
#    - Negative binomial accounts for overdispersion
obs_model = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_pmf
)

# ---- Compose into EpiProblem ----
epi_prob = EpiProblem(
    epi_model = epi_model,
    latent_model = latent_model,
    observation_model = obs_model,
    tspan = (1, length(cases))
)

# ---- Run Inference ----
# Generate Turing model conditioned on observed cases
mdl = generate_epiaware(epi_prob, (y_t = cases,))

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
@info "Running inference..."
results = apply_method(mdl, inference_method, (y_t = cases,))

# ---- Extract Results ----
# Get posterior samples of Rt
# The latent process Z_t = log(Rt), so Rt = exp(Z_t)

# Extract generated quantities
gen_quants = generated_quantities(
    generate_epiaware(epi_prob, (y_t = fill(missing, length(cases)),)),
    results.samples
)

# Compute Rt quantiles
function compute_rt_quantiles(gen_quants; probs = [0.05, 0.5, 0.95])
    Z_t_samples = mapreduce(hcat, gen_quants) do g
        g.Z_t
    end
    Rt_samples = exp.(Z_t_samples)

    quantiles = mapreduce(hcat, probs) do p
        mapslices(x -> quantile(x, p), Rt_samples, dims = 2)
    end

    return quantiles
end

Rt_quantiles = compute_rt_quantiles(gen_quants)

# ---- Output Summary ----
@info "Rt estimation complete"
@info "Posterior Rt summary (median and 90% CI):"

# Create summary DataFrame
n_days = length(cases)
summary_df = DataFrame(
    day = 1:n_days,
    Rt_lower = Rt_quantiles[:, 1],
    Rt_median = Rt_quantiles[:, 2],
    Rt_upper = Rt_quantiles[:, 3]
)

# Print first and last few rows
println("\nFirst 10 days:")
println(first(summary_df, 10))
println("\nLast 10 days:")
println(last(summary_df, 10))

# Save results
CSV.write("outputs/scenario_1b_rt_estimates.csv", summary_df)
@info "Results saved to outputs/scenario_1b_rt_estimates.csv"
