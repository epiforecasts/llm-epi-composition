
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/epiaware/llama3.1:8b/run_01")

using EpiAware
using Distributions
using Turing

# Load data
cases = CSV.read("cases.csv", DataFrame)

# Define generation interval distribution
gen_distribution = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_distribution)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Define observation model with delay and day-of-week effects
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)
dayofweek_effect = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Compose observation model with day-of-week effects and delay
obs_with_day_of_week = StackObservationModels((obs, dayofweek_effect))

# Time-varying ascertainment
ascertainment_model = Ascertainment(
    obs_with_day_of_week,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)

# Create EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = ascertainment_model,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases.cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases.cases,))

# Extract Rt estimates
rt_estimates = exp.(results.latent)

# Plot results
using Plots
plot(rt_estimates[:, 1], label="Rt Estimates")
xlabel("Time Point")
ylabel("Estimated Reproduction Number")

