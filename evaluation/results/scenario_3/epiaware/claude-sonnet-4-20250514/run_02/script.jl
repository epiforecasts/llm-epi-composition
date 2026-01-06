
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/epiaware/claude-sonnet-4-20250514/run_02")

using EpiAware
using Distributions
using DataFrames
using CSV
using Plots
using Statistics
using Turing

# Load the data
data = CSV.read("observations.csv", DataFrame)
n_timepoints = nrow(data)

# Extract observation vectors
cases = data.cases
hospitalisations = data.hospitalisations  
deaths = data.deaths

println("Loaded data: $(n_timepoints) time points")
println("Cases range: $(minimum(cases)) - $(maximum(cases))")
println("Hospitalizations range: $(minimum(hospitalisations)) - $(maximum(hospitalisations))")
println("Deaths range: $(minimum(deaths)) - $(maximum(deaths))")

# 1. Define generation interval (infection-to-infection)
gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, SD ~1.6 days
model_data = EpiData(gen_distribution = gen_int)

# 2. Create shared infection model using renewal equation
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# 3. Create shared latent model for log(Rt) - AR(1) with smoothness
latent = AR(
    damp_priors = [truncated(Normal(0.9, 0.05), 0.0, 1.0)],  # High persistence for smoothness
    init_priors = [Normal(0.0, 0.5)],                         # Prior for initial log(Rt)
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))     # Innovation noise
)

# 4. Create stream-specific observation models with different delays
# Cases: Short delay (2-3 days from infection to reporting)
cases_delay = Gamma(2.5, 1.2)  # Mean ~3 days
cases_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.2)),
    cases_delay
)

# Hospitalizations: Medium delay (7-8 days from infection to admission)  
hosp_delay = Gamma(6.5, 1.2)   # Mean ~8 days
hosp_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.3)),
    hosp_delay
)

# Deaths: Long delay (14-16 days from infection to death)
death_delay = Gamma(13.0, 1.2) # Mean ~16 days
death_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.4)),
    death_delay
)

# 5. Add stream-specific ascertainment rates using Ascertainment wrapper
# Cases ascertainment (highest, but variable)
cases_ascert = Ascertainment(
    cases_obs,
    FixedIntercept(-1.5),  # logit scale, ~18% base ascertainment
    (infections, logit_p) -> infections .* logistic.(logit_p)
)

# Hospitalizations ascertainment (lower but more stable)
hosp_ascert = Ascertainment(
    hosp_obs, 
    FixedIntercept(-3.0),  # logit scale, ~5% ascertainment
    (infections, logit_p) -> infections .* logistic.(logit_p)
)

# Deaths ascertainment (lowest but most complete)
death_ascert = Ascertainment(
    death_obs,
    FixedIntercept(-4.5),  # logit scale, ~1% ascertainment  
    (infections, logit_p) -> infections .* logistic.(logit_p)
)

# 6. Stack the observation models for joint inference
stacked_obs = StackObservationModels((
    cases = cases_ascert,
    hospitalisations = hosp_ascert, 
    deaths = death_ascert
))

# 7. Create the complete EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = stacked_obs,
    tspan = (1, n_timepoints)
)

# 8. Generate the Turing model with observed data
println("Generating Turing model...")
mdl = generate_epiaware(epi_prob, (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    ),
))

# 9. Set up inference method
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 1500,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

# 10. Run inference
println("Running MCMC inference...")
results = apply_method(mdl, inference_method, (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations, 
        deaths = deaths
    ),
))

# 11. Extract and process results
println("Processing results...")

# Extract Rt estimates (latent process Z_t = log(Rt))
log_rt_samples = results[:, r"Z_t", :]
rt_samples = exp.(log_rt_samples)

# Compute Rt summary statistics
rt_mean = vec(mean(rt_samples, dims = 1))
rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims = 1))
rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims = 1))

# Extract ascertainment parameters (on logit scale)
cases_ascert_logit = results[:, "y_t.cases.ascert_param.Z_t[1]", :]
hosp_ascert_logit = results[:, "y_t.hospitalisations.ascert_param.Z_t[1]", :]  
death_ascert_logit = results[:, "y_t.deaths.ascert_param.Z_t[1]", :]

# Transform to probability scale
cases_ascert_prob = logistic.(cases_ascert_logit)
hosp_ascert_prob = logistic.(hosp_ascert_logit)
death_ascert_prob = logistic.(death_ascert_logit)

# Compute ascertainment summary statistics  
cases_ascert_mean = mean(cases_ascert_prob)
hosp_ascert_mean = mean(hosp_ascert_prob)
death_ascert_mean = mean(death_ascert_prob)

# 12. Display results summary
println("\n" * "="^60)
println("JOINT Rt ESTIMATION RESULTS")
println("="^60)

println("\nEstimated Parameters:")
println("├─ Cases ascertainment rate: $(round(cases_ascert_mean*100, digits=1))%")
println("├─ Hospitalizations ascertainment rate: $(round(hosp_ascert_mean*100, digits=1))%") 
println("└─ Deaths ascertainment rate: $(round(death_ascert_mean*100, digits=1))%")

println("\nRt Summary:")
println("├─ Mean Rt: $(round(mean(rt_mean), digits=2))")
println("├─ Min Rt: $(round(minimum(rt_mean), digits=2))")
println("├─ Max Rt: $(round(maximum(rt_mean), digits=2))")
println("└─ Final Rt: $(round(rt_mean[end], digits=2)) [$(round(rt_lower[end], digits=2)), $(round(rt_upper[end], digits=2))]")

# 13. Create visualization
println("\nGenerating plots...")

# Plot 1: Rt estimates over time
p1 = plot(1:n_timepoints, rt_mean, 
         ribbon = (rt_mean .- rt_lower, rt_upper .- rt_mean),
         fillalpha = 0.3,
         color = :red,
         linewidth = 2,
         label = "Rt (95% CI)",
         title = "Time-varying Reproduction Number (Rt)",
         xlabel = "Days",
         ylabel = "Rt",
         legend = :topright)

hline!([1.0], linestyle = :dash, color = :black, alpha = 0.5, label = "Rt = 1")

# Plot 2: Observed data streams
p2 = plot(1:n_timepoints, cases, 
         label = "Cases", 
         color = :blue, 
         linewidth = 2,
         title = "Observed Data Streams",
         xlabel = "Days", 
         ylabel = "Count")

plot!(1:n_timepoints, hospitalisations, 
      label = "Hospitalizations", 
      color = :orange, 
      linewidth = 2)

plot!(1:n_timepoints, deaths, 
      label = "Deaths", 
      color = :red, 
      linewidth = 2)

# Combine plots
combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 600))
display(combined_plot)

# Save results to CSV
results_df = DataFrame(
    day = 1:n_timepoints,
    date = data.date,
    rt_mean = rt_mean,
    rt_lower = rt_lower, 
    rt_upper = rt_upper,
    cases_obs = cases,
    hosp_obs = hospitalisations,
    deaths_obs = deaths
)

CSV.write("rt_estimates_joint.csv", results_df)
println("\nResults saved to 'rt_estimates_joint.csv'")

println("\n" * "="^60)
println("ANALYSIS COMPLETE")
println("="^60)

