# Import necessary libraries
using EpiAware
using Distributions
using Turing
using DataFrames
using CSV

# Load data
df = DataFrame(CSV.File("observations.csv"))

# Define data structures for each observation type
cases_data = EpiData(
    name="Cases",
    y=df[:cases],
    x=collect(1:length(df[:cases])),
    Δ=collect(0:1:length(df[:cases])-1)
)

hospitalisations_data = EpiData(
    name="Hospitalisations",
    y=df[:hospitalisations],
    x=collect(1:length(df[:hospitalisations])),
    Δ=collect(0:1:length(df[:hospitalisations])-1)
)

deaths_data = EpiData(
    name="Deaths",
    y=df[:deaths],
    x=collect(1:length(df[:deaths])),
    Δ=collect(0:1:length(df[:deaths])-1)
)

# Define generation interval distribution
gen_distribution = Gamma(6.5, 0.62)

# Create EpiData with generation interval
model_data_cases = EpiData(gen_distribution = gen_distribution)
model_data_hospitalisations = EpiData(gen_distribution = gen_distribution)
model_data_deaths = EpiData(gen_distribution = gen_distribution)

# Define infection models (Renewal)
epi_cases = Renewal(model_data_cases; initialisation_prior = Normal(log(100.0), 1.0))
epi_hospitalisations = Renewal(model_data_hospitalisations; initialisation_prior = Normal(log(100.0), 1.0))
epi_deaths = Renewal(model_data_deaths; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

obs_hospitalisations = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

obs_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob_cases = EpiProblem(
    epi_model = epi_cases,
    latent_model = latent,
    observation_model = obs_cases,
    tspan = (1, length(df[:cases]))
)

epi_prob_hospitalisations = EpiProblem(
    epi_model = epi_hospitalisations,
    latent_model = latent,
    observation_model = obs_hospitalisations,
    tspan = (1, length(df[:hospitalisations]))
)

epi_prob_deaths = EpiProblem(
    epi_model = epi_deaths,
    latent_model = latent,
    observation_model = obs_deaths,
    tspan = (1, length(df[:deaths]))
)

# Run inference
mdl_cases = generate_epiaware(epi_prob_cases, (y_t = df[:cases],))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)

results_cases = apply_method(mdl_cases, inference_method, (y_t = df[:cases],))

mdl_hospitalisations = generate_epiaware(epi_prob_hospitalisations, (y_t = df[:hospitalisations],))
results_hospitalisations = apply_method(mdl_hospitalisations, inference_method, (y_t = df[:hospitalisations],))

mdl_deaths = generate_epiaware(epi_prob_deaths, (y_t = df[:deaths],))
results_deaths = apply_method(mdl_deaths, inference_method, (y_t = df[:deaths],))

# Extract Rt estimates
rt_cases = Array(results_cases[1].Z_t)[2:end]
rt_hospitalisations = Array(results_hospitalisations[1].Z_t)[2:end]
rt_deaths = Array(results_deaths[1].Z_t)[2:end]

# Plot results
using Plots

plot(rt_cases, label="Cases")
plot!(rt_hospitalisations, label="Hospitalisations")
plot!(rt_deaths, label="Deaths")

xlabel!("Time")
ylabel!("log(Rt)")
title!("Estimated log(Rt) over time")
legend()

