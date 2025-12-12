Here is the complete code that meets all the requirements:
```julia
using EpiAware
using Distributions
using Turing

# Load data
data = CSV.read("observations.csv")

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)

# Create infection model (Renewal)
epi_data = EpiData(gen_distribution = gen_int)
epi_model = Renewal(epi_data; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent_model = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Define observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs_model_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)
obs_model_hospitalisations = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)
obs_model_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Create EpiProblem with multiple observation streams
epi_prob = EpiProblem(
    epi_model = epi_model,
    latent_model = latent_model,
    observation_models = [
        obs_model_cases => "cases",
        obs_model_hospitalisations => "hospitalisations",
        obs_model_deaths => "deaths"
    ],
    tspan = (1, length(data.date))
)

# Run inference with multiple chains
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(epi_prob, inference_method)

# Extract Rt estimates and stream-specific parameters
Rt_estimates = results.latent
stream_params = results.obs_params

# Plot results
using Plots
plot(Rt_estimates)
xlabel!("Time")
ylabel!("log(Rt)")
title!("Estimated log(Rt) over time")

# Print summary of model fit
println("Model Fit Summary:")
println("R-squared: $(results.r2)")
println("MAE: $(results.mae)")
println("RMSE: $(results.rmse)")
```
This code assumes that you have already installed the `EpiAware` package and have a CSV file named "observations.csv" with columns for date, cases, hospitalisations, and deaths. The model combines an AR(1) process for log(Rt) with three separate observation models, one for each data stream. The inference method uses multiple chains to estimate the posterior distribution of the parameters.

Note that this is a basic example, and you may need to modify the code to suit your specific needs. Additionally, this code assumes that the data streams are independent and identically distributed (i.i.d.) over time, which may not be the case in practice.