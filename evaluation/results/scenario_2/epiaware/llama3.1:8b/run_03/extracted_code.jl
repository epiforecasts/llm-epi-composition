using EpiAware
using Distributions
using Turing
using Plots

# Load data
cases = load("cases.csv", header=true)

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Define day-of-week effect model
dayofweek_effect_model = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Combine latent models
combined_latent = CombineLatentModels([latent, dayofweek_effect_model])

# Create observation model with ascertainment
ascertainment_model = Ascertainment(
    obs,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = combined_latent,
    observation_model = ascertainment_model,
    tspan = (1, length(cases))
)

# Generate Turing model and run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt_estimates = exp.(results[1][:Z_t])

# Plot Rt estimates over time
plot(Rt_estimates)

