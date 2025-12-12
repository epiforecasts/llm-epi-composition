Here's the code to estimate time-varying reproduction numbers (Rt) from daily reported COVID-19 case counts, accounting for complex observation processes. This example demonstrates how to combine the components provided in the documentation.

```julia
using EpiAware
using Distributions
using Turing
using Plots

# Load data
cases = readdlm("cases.csv", ',')
dates = cases[:, 1]
day_of_week = cases[:, 2]

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent models for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Add day-of-week effects
dayofweek_effect = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)
combined_latent = CombineLatentModels([latent, dayofweek_effect])

# Create observation model with delay and ascertainment
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

ascertainment_model = Ascertainment(
    obs,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)
stacked_obs = StackObservationModels((cases = obs))

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = combined_latent,
    observation_model = ascertainment_model,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt_estimates = exp.(results.latent_samples)

# Plot results
plot(Rt_estimates[:, 1], label="Day of week effect")
plot!(Rt_estimates[:, 2], label="Ascertainment estimate")

xlabel("Time")
ylabel("Estimate")
legend()
```

This code estimates the time-varying reproduction number (Rt) using an AR(1) process with day-of-week effects, and a time-varying ascertainment model. The results are plotted as separate lines for each component.

Remember to adjust the hyperparameters in the models according to your data characteristics and research question. This example provides a starting point for exploring complex observation processes in infectious disease modeling.