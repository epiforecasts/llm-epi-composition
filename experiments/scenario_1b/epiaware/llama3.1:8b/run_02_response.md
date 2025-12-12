Here is the complete, runnable code for estimating the time-varying reproduction number (Rt) from daily reported COVID-19 case counts using the renewal equation framework:

```julia
using EpiAware
using Distributions
using Turing
using CSV
using Plots

# Load data
data = CSV.File("cases.csv") |> DataFrame

# 1. Define generation interval
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# 2. Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# 3. Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# 4. Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# 5. Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(data.cases))
)

# 6. Run inference
mdl = generate_epiaware(epi_prob, (y_t = data.cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = data.cases,))

# 7. Extract Rt estimates
Rt_estimates = map(x -> exp.(x[1]), results.samples)

# Plot Rt over time
plot(Rt_estimates)
xlabel!("Time")
ylabel!("Rt")
title!("Estimated Reproduction Number Over Time")
```

This code will produce the estimated reproduction number (Rt) for each day of the reported case counts, and plot these estimates over time. The `EpiAware` package provides a framework for constructing and fitting epidemic models using a modular, composable design.