
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/epiaware/claude-sonnet-4-20250514/run_03")

using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using StatsPlots

# Load and prepare the data
function load_case_data(filepath::String)
    df = CSV.read(filepath, DataFrame)
    
    # Ensure proper date parsing and sorting
    df.date = Date.(df.date)
    sort!(df, :date)
    
    # Handle any missing or negative values
    df.cases = max.(df.cases, 0)  # Replace negative with 0
    
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_from_cases(case_data::Vector{Int})
    n_timepoints = length(case_data)
    
    # 1. Define generation interval (COVID-19 typical values)
    # Mean ~6.5 days, std ~4 days
    gen_int = Gamma(6.5, 0.62)  # shape = 6.5, rate = 1/0.62
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)  # Prior for initial infections
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smooth evolution
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Innovation noise
    )
    
    # 4. Create observation model with reporting delay
    # COVID-19 typically has ~5 day delay from infection to case reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean 5 days delay
    obs = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_dist
    )
    
    # 5. Compose into EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    # 6. Generate Turing model
    mdl = generate_epiaware(epi_prob, (y_t = case_data,))
    
    # 7. Define inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 8. Run inference
    println("Running MCMC inference...")
    results = apply_method(mdl, inference_method, (y_t = case_data,))
    
    return results, epi_prob
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(results)
    # Extract the latent process samples (Z_t = log(Rt))
    samples = results.samples
    
    # Get parameter names containing "Z_t"
    param_names = String.(keys(samples))
    z_params = filter(x -> occursin("Z_t", x), param_names)
    
    # Sort parameters by time index
    z_indices = [parse(Int, match(r"Z_t\[(\d+)\]", p).captures[1]) for p in z_params]
    sorted_idx = sortperm(z_indices)
    z_params_sorted = z_params[sorted_idx]
    
    # Extract samples and compute Rt = exp(Z_t)
    rt_samples = Matrix{Float64}(undef, length(samples[:, z_params_sorted[1], 1]), length(z_params_sorted))
    
    for (i, param) in enumerate(z_params_sorted)
        log_rt_samples = vec(samples[:, param, :])
        rt_samples[:, i] = exp.(log_rt_samples)
    end
    
    # Compute summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_q025 = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_q975 = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    rt_q25 = vec(mapslices(x -> quantile(x, 0.25), rt_samples, dims=1))
    rt_q75 = vec(mapslices(x -> quantile(x, 0.75), rt_samples, dims=1))
    
    return (
        samples = rt_samples,
        mean = rt_mean,
        median = rt_median,
        q025 = rt_q025,
        q975 = rt_q975,
        q25 = rt_q25,
        q75 = rt_q75,
        timepoints = 1:length(rt_mean)
    )
end

# Function to plot Rt estimates
function plot_rt_estimates(rt_estimates, dates=nothing)
    timepoints = rt_estimates.timepoints
    
    # Create x-axis (use dates if provided, otherwise time indices)
    x_axis = dates !== nothing ? dates : timepoints
    
    p = plot(
        x_axis, rt_estimates.median,
        ribbon = (rt_estimates.median .- rt_estimates.q025, 
                 rt_estimates.q975 .- rt_estimates.median),
        fillalpha = 0.3,
        label = "Rt (95% CI)",
        linewidth = 2,
        color = :blue,
        title = "Time-varying Reproduction Number (Rt)",
        xlabel = dates !== nothing ? "Date" : "Time",
        ylabel = "Rt",
        legend = :topright
    )
    
    # Add 50% credible interval
    plot!(p, x_axis, rt_estimates.median,
          ribbon = (rt_estimates.median .- rt_estimates.q25,
                   rt_estimates.q75 .- rt_estimates.median),
          fillalpha = 0.5,
          color = :blue,
          label = "Rt (50% CI)")
    
    # Add horizontal line at Rt = 1
    hline!(p, [1.0], linestyle = :dash, color = :red, label = "Rt = 1", linewidth = 2)
    
    return p
end

# Main execution function
function main()
    # Load data
    println("Loading case data...")
    cases = load_case_data("cases.csv")
    
    # Load dates for plotting (optional)
    df = CSV.read("cases.csv", DataFrame)
    dates = Date.(df.date)
    
    println("Loaded $(length(cases)) days of case data")
    println("Date range: $(minimum(dates)) to $(maximum(dates))")
    println("Case range: $(minimum(cases)) to $(maximum(cases))")
    
    # Estimate Rt
    results, epi_prob = estimate_rt_from_cases(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results)
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("======================")
    println("Time points: $(length(rt_estimates.timepoints))")
    println("Mean Rt: $(round(mean(rt_estimates.mean), digits=3))")
    println("Median Rt: $(round(mean(rt_estimates.median), digits=3))")
    println("Rt range: $(round(minimum(rt_estimates.q025), digits=3)) - $(round(maximum(rt_estimates.q975), digits=3))")
    
    # Create summary DataFrame
    rt_summary = DataFrame(
        date = dates,
        rt_median = rt_estimates.median,
        rt_mean = rt_estimates.mean,
        rt_q025 = rt_estimates.q025,
        rt_q975 = rt_estimates.q975,
        rt_q25 = rt_estimates.q25,
        rt_q75 = rt_estimates.q75
    )
    
    # Save results
    CSV.write("rt_estimates.csv", rt_summary)
    println("Rt estimates saved to 'rt_estimates.csv'")
    
    # Create and save plot
    p = plot_rt_estimates(rt_estimates, dates)
    savefig(p, "rt_plot.png")
    println("Rt plot saved to 'rt_plot.png'")
    
    # Display plot
    display(p)
    
    return rt_estimates, results, rt_summary
end

# Run the analysis
rt_estimates, mcmc_results, rt_summary = main()

# Print first few Rt estimates
println("\nFirst 10 Rt estimates:")
for i in 1:min(10, length(rt_estimates.median))
    println("Day $i: Rt = $(round(rt_estimates.median[i], digits=3)) (95% CI: $(round(rt_estimates.q025[i], digits=3)) - $(round(rt_estimates.q975[i], digits=3)))")
end

