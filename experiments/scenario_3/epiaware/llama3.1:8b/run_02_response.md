To estimate the time-varying reproduction number (Rt) jointly from three data streams: cases, hospitalisations, and deaths, we'll use the EpiAware.jl components in Julia. Here's a complete example of how to write the code that estimates a single shared Rt over time using the renewal equation.

```julia
using EpiAware
using Distributions
using Turing

# Load data
df = readtable("observations.csv")
cases = df[:, :cases]
hospitalisations = df[:, :hospitalisations]
deaths = df[:, :deaths]

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

# Create observation models with delays
delay_dist_cases = Gamma(5.0, 1.0)
obs_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist_cases
)

delay_dist_hospitalisations = Gamma(5.0, 1.0)
obs_hospitalisations = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist_hospitalisations
)

delay_dist_deaths = Gamma(5.0, 1.0)
obs_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist_deaths
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_models = (
        cases = (obs = obs_cases,),
        hospitalisations = (obs = obs_hospitalisations,),
        deaths = (obs = obs_deaths,)
    ),
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (cases, hospitalisations, deaths))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (cases, hospitalisations, deaths))

# Extract Rt estimates
Rt_estimates = results.latent_samples.latent_model.Z_t[1]

# Plot Rt estimates
using Plots
plot(Rt_estimates)
xlabel("Time")
ylabel("log(Rt)")
```

This code defines the generation interval distribution, infection model (Renewal), latent models for log(Rt) as an AR(1) process, and observation models with delays for each data stream. It then composes these components into an EpiProblem instance and runs inference using the `apply_method` function.

The resulting Rt estimates are stored in the `Rt_estimates` array, which is a time series of log(Rt) values estimated by the model. The code also includes a simple plotting command to visualize the Rt estimates over time.

Note that you may need to adjust the hyperparameters (e.g., prior distributions, sampling parameters) and model specifications (e.g., number of latent processes, observation models) to suit your specific data and modeling needs.