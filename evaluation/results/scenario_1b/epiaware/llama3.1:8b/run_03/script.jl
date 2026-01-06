
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/epiaware/llama3.1:8b/run_03")

# Import necessary packages
using EpiAware
using Distributions
using Turing

# Define data
data = CSV.read("cases.csv")

# Create generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(data.cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = data.cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = data.cases,))

# Extract Rt estimates
Rt_estimates = map(x -> exp.(x), results.latent_samples)

# Plot Rt over time
using Plots
plot(Rt_estimates[1,:])
xlabel!("Time")
ylabel!("Rt")
title!("Estimated Reproduction Number (Rt)")

