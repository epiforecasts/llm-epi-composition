
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/epiaware/claude-sonnet-4-20250514/run_01")

using EpiAware
using CSV
using DataFrames
using Distributions
using Plots
using StatsPlots
using Turing
using MCMCChains

# Load and prepare the data
function load_case_data(filename::String)
    df = CSV.read(filename, DataFrame)
    # Ensure data is sorted by date
    sort!(df, :date)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_renewal(cases::Vector{Int})
    println("Setting up renewal equation model for Rt estimation...")
    
    # 1. Define epidemiological parameters
    # Generation interval - using COVID-19 typical values
    gen_int_mean = 6.5
    gen_int_var = 4.03  # variance
    # Convert to shape/scale parameterization for Gamma distribution
    gen_int_scale = gen_int_var / gen_int_mean
    gen_int_shape = gen_int_mean / gen_int_scale
    gen_distribution = Gamma(gen_int_shape, gen_int_scale)
    
    println("Generation interval: Gamma($(round(gen_int_shape, digits=2)), $(round(gen_int_scale, digits=2)))")
    
    # 2. Create EpiData with generation interval
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 3. Create infection model using Renewal equation
    # Prior for initial infections (seeding)
    initial_inf_prior = Normal(log(mean(cases[1:7])), 1.0)  # Use first week average as guide
    epi = Renewal(model_data; initialisation_prior = initial_inf_prior)
    
    # 4. Create latent model for log(Rt) - AR(1) process for smooth evolution
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Innovation noise
    )
    
    # 5. Create observation model with reporting delay
    # Delay from infection to case reporting (incubation + reporting delay)
    delay_mean = 8.0
    delay_var = 16.0
    delay_scale = delay_var / delay_mean
    delay_shape = delay_mean / delay_scale
    delay_dist = Gamma(delay_shape, delay_scale)
    
    # Negative binomial observation model to handle overdispersion
    obs_base = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs = LatentDelay(obs_base, delay_dist)
    
    println("Reporting delay: Gamma($(round(delay_shape, digits=2)), $(round(delay_scale, digits=2)))")
    
    # 6. Compose into EpiProblem
    n_timepoints = length(cases)
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    println("Model setup complete. Running inference...")
    
    # 7. Generate Turing model and run inference
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # Configure inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 1000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    println("Inference complete!")
    
    return results, epi_prob
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(results)
    # Extract the latent process Z_t which represents log(Rt)
    chain = results
    
    # Get parameter names that contain "Z_t"
    param_names = names(chain)
    z_params = filter(name -> occursin("Z_t", string(name)), param_names)
    
    # Sort parameters by time index
    z_params_sorted = sort(z_params, by = x -> parse(Int, match(r"\[(\d+)\]", string(x)).captures[1]))
    
    # Extract log(Rt) values
    log_rt_samples = Array(chain[z_params_sorted])
    
    # Convert to Rt (exponentiate)
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(median(rt_samples, dims=1))
    rt_q025 = vec(quantile.(eachcol(rt_samples), 0.025))
    rt_q975 = vec(quantile.(eachcol(rt_samples), 0.975))
    rt_q10 = vec(quantile.(eachcol(rt_samples), 0.10))
    rt_q90 = vec(quantile.(eachcol(rt_samples), 0.90))
    
    return DataFrame(
        time = 1:length(rt_mean),
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_q025 = rt_q025,
        rt_q975 = rt_q975,
        rt_q10 = rt_q10,
        rt_q90 = rt_q90
    )
end

# Function to create plots
function plot_rt_estimates(rt_estimates, cases)
    # Create subplot layout
    p1 = plot(rt_estimates.time, rt_estimates.rt_median, 
             ribbon=(rt_estimates.rt_median .- rt_estimates.rt_q025,
                    rt_estimates.rt_q975 .- rt_estimates.rt_median),
             label="Rt (95% CI)", 
             color=:blue, 
             alpha=0.3,
             linewidth=2,
             title="Estimated Reproduction Number (Rt)",
             xlabel="Time (days)",
             ylabel="Rt")
    
    # Add 50% credible interval
    plot!(p1, rt_estimates.time, rt_estimates.rt_median, 
          ribbon=(rt_estimates.rt_median .- rt_estimates.rt_q10,
                 rt_estimates.rt_q90 .- rt_estimates.rt_median),
          label="Rt (80% CI)", 
          color=:blue, 
          alpha=0.5,
          linewidth=2)
    
    # Add reference line at Rt = 1
    hline!(p1, [1.0], line=:dash, color=:red, label="Rt = 1", linewidth=2)
    
    # Plot observed cases
    p2 = bar(1:length(cases), cases, 
             label="Observed cases",
             color=:gray,
             alpha=0.7,
             title="Observed Case Counts",
             xlabel="Time (days)",
             ylabel="Cases")
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main execution function
function main()
    println("=== Rt Estimation using Renewal Equation ===\n")
    
    # Load data
    println("Loading case data...")
    cases = load_case_data("cases.csv")
    println("Loaded $(length(cases)) days of case data")
    println("Total cases: $(sum(cases))")
    println("Mean daily cases: $(round(mean(cases), digits=1))\n")
    
    # Estimate Rt
    results, epi_prob = estimate_rt_renewal(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results)
    
    # Display summary
    println("\n=== Rt Estimation Summary ===")
    println("Time period: $(length(cases)) days")
    println("Mean Rt: $(round(mean(rt_estimates.rt_mean), digits=2))")
    println("Rt range: $(round(minimum(rt_estimates.rt_mean), digits=2)) - $(round(maximum(rt_estimates.rt_mean), digits=2))")
    
    # Count periods where Rt > 1
    days_above_1 = sum(rt_estimates.rt_median .> 1.0)
    println("Days with Rt > 1: $days_above_1 / $(length(cases)) ($(round(100*days_above_1/length(cases), digits=1))%)")
    
    # Show recent Rt values
    println("\nRecent Rt estimates (last 7 days):")
    recent_idx = max(1, length(cases)-6):length(cases)
    for i in recent_idx
        println("Day $i: $(round(rt_estimates.rt_median[i], digits=2)) [$(round(rt_estimates.rt_q025[i], digits=2)) - $(round(rt_estimates.rt_q975[i], digits=2))]")
    end
    
    # Create and save plots
    println("\nCreating plots...")
    p = plot_rt_estimates(rt_estimates, cases)
    
    # Save results
    println("Saving results...")
    CSV.write("rt_estimates.csv", rt_estimates)
    savefig(p, "rt_estimates_plot.png")
    
    println("\n=== Analysis Complete ===")
    println("Results saved to:")
    println("- rt_estimates.csv: Rt estimates with credible intervals")
    println("- rt_estimates_plot.png: Visualization of results")
    
    return rt_estimates, results
end

# Helper function to create synthetic data if needed (for testing)
function create_synthetic_cases(n_days::Int = 100)
    # Simple synthetic epidemic with changing Rt
    true_rt = vcat(
        fill(2.0, 20),    # Initial growth
        1.5 .- 0.025*(1:20),  # Declining
        fill(0.8, 20),    # Below 1
        0.8 .+ 0.01*(1:20),   # Rising again
        fill(1.2, 20)     # Above 1 again
    )
    
    gen_int = Gamma(6.5, 0.62)
    
    infections = zeros(n_days)
    infections[1:7] .= 50  # Seed infections
    
    for t in 8:n_days
        # Renewal equation
        inf_pressure = 0.0
        for s in 1:min(t-1, 20)  # Truncate generation interval
            gen_prob = pdf(gen_int, s) / sum([pdf(gen_int, i) for i in 1:20])
            inf_pressure += infections[t-s] * gen_prob
        end
        infections[t] = true_rt[t] * inf_pressure
    end
    
    # Add reporting delay and noise
    delay_dist = Gamma(2.0, 3.0)
    cases = zeros(Int, n_days)
    
    for t in 1:n_days
        for delay in 1:20
            delay_prob = pdf(delay_dist, delay) / sum([pdf(delay_dist, i) for i in 1:20])
            if t + delay <= n_days
                cases[t + delay] += round(Int, infections[t] * delay_prob * rand(LogNormal(0, 0.1)))
            end
        end
    end
    
    # Create DataFrame and save
    dates = [Date(2023, 1, 1) + Day(i-1) for i in 1:n_days]
    df = DataFrame(date = dates, cases = cases)
    CSV.write("cases.csv", df)
    
    return df
end

# Run the analysis
# Uncomment the next line to create synthetic data for testing
# create_synthetic_cases(100)

# Run main analysis
rt_estimates, results = main()

