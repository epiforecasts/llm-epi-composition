using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics

# Load the data
function load_case_data(filename::String)
    df = CSV.read(filename, DataFrame)
    # Ensure dates are parsed correctly and data is sorted by date
    df.date = Date.(df.date)
    sort!(df, :date)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_from_cases(cases_file::String)
    
    # 1. Load case data
    println("Loading case data...")
    cases = load_case_data(cases_file)
    n_timepoints = length(cases)
    println("Loaded $n_timepoints days of case data")
    
    # 2. Define generation interval distribution
    # Using COVID-19 typical values: mean ~6.5 days, shape parameter
    gen_int = Gamma(6.5, 0.62)  # mean = 6.5 * 0.62 ≈ 4 days, reasonable for COVID-19
    model_data = EpiData(gen_distribution = gen_int)
    
    # 3. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)  # Prior for initial infections
    )
    
    # 4. Create latent model for log(Rt) - AR(1) process for smoothness
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Process noise
    )
    
    # 5. Create observation model with reporting delay
    # COVID-19 typically has ~5 day delay from infection to case reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean delay of 5 days
    obs = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_dist
    )
    
    # 6. Compose the full epidemiological model
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    # 7. Generate Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 8. Set up inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 9. Run inference
    println("Running MCMC inference...")
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    # 10. Extract Rt estimates
    println("Extracting Rt estimates...")
    
    # Get posterior samples of the latent process (log Rt)
    log_rt_samples = results[:Z_t]  # Shape: (n_samples, n_timepoints)
    
    # Transform to Rt scale
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))  # 2.5th percentile
    rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))  # 97.5th percentile
    
    # Create results DataFrame
    dates = Date("2020-01-01") .+ Day.(0:(n_timepoints-1))  # Placeholder dates
    rt_results = DataFrame(
        date = dates,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper,
        cases = cases
    )
    
    return rt_results, results
end

# Function to plot results
function plot_rt_estimates(rt_results::DataFrame)
    
    # Create Rt plot
    p1 = plot(rt_results.date, rt_results.rt_median, 
              ribbon = (rt_results.rt_median .- rt_results.rt_lower,
                       rt_results.rt_upper .- rt_results.rt_median),
              label = "Rt (95% CI)",
              color = :blue,
              alpha = 0.3,
              linewidth = 2,
              title = "Time-varying Reproduction Number (Rt)",
              ylabel = "Rt",
              xlabel = "Date")
    
    # Add horizontal line at Rt = 1
    hline!([1.0], color = :red, linestyle = :dash, linewidth = 2, label = "Rt = 1")
    
    # Create cases plot
    p2 = plot(rt_results.date, rt_results.cases,
              color = :black,
              linewidth = 1,
              title = "Daily Case Counts",
              ylabel = "Cases",
              xlabel = "Date",
              label = "Observed cases")
    
    # Combine plots
    combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 600))
    
    return combined_plot
end

# Function to summarize results
function summarize_rt_estimates(rt_results::DataFrame)
    println("\n=== Rt Estimation Summary ===")
    println("Time period: $(rt_results.date[1]) to $(rt_results.date[end])")
    println("Number of time points: $(nrow(rt_results))")
    
    # Overall statistics
    overall_median = median(rt_results.rt_median)
    println("\nOverall median Rt: $(round(overall_median, digits=3))")
    
    # Periods above/below 1
    above_one = sum(rt_results.rt_lower .> 1.0)
    below_one = sum(rt_results.rt_upper .< 1.0)
    uncertain = nrow(rt_results) - above_one - below_one
    
    println("\nTime points where:")
    println("  - Rt significantly > 1: $above_one ($(round(100*above_one/nrow(rt_results), digits=1))%)")
    println("  - Rt significantly < 1: $below_one ($(round(100*below_one/nrow(rt_results), digits=1))%)")
    println("  - Rt uncertain: $uncertain ($(round(100*uncertain/nrow(rt_results), digits=1))%)")
    
    # Recent Rt
    recent_rt = rt_results.rt_median[end-min(6, nrow(rt_results)-1):end]
    println("\nRecent Rt trend (last 7 days): $(round(mean(recent_rt), digits=3))")
    
    return nothing
end

# Main execution function
function main()
    cases_file = "cases.csv"
    
    try
        # Estimate Rt
        rt_results, mcmc_results = estimate_rt_from_cases(cases_file)
        
        # Print summary
        summarize_rt_estimates(rt_results)
        
        # Create and display plot
        rt_plot = plot_rt_estimates(rt_results)
        display(rt_plot)
        
        # Save results
        CSV.write("rt_estimates.csv", rt_results)
        println("\nResults saved to rt_estimates.csv")
        
        return rt_results, mcmc_results
        
    catch e
        println("Error: $e")
        println("\nMake sure 'cases.csv' exists with columns 'date' and 'cases'")
        rethrow(e)
    end
end

# Run the analysis
rt_results, mcmc_results = main()

