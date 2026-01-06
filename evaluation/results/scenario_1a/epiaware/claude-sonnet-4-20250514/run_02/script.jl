
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/epiaware/claude-sonnet-4-20250514/run_02")

using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics

# Load the data
function load_case_data(filepath::String)
    df = CSV.read(filepath, DataFrame)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_from_cases(cases_file::String = "cases.csv")
    
    # 1. Load the case data
    println("Loading case data...")
    cases = load_case_data(cases_file)
    n_timepoints = length(cases)
    println("Loaded $n_timepoints days of case data")
    
    # 2. Define the generation interval distribution
    # Using typical COVID-19 parameters: mean ~6.5 days, shape parameter
    gen_int = Gamma(6.5, 0.62)  # Mean ≈ 4.0 days
    model_data = EpiData(gen_distribution = gen_int)
    
    # 3. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(mean(cases[1:7])), 1.0)  # Initialize based on early cases
    )
    
    # 4. Create latent model for log(Rt) - AR(1) process
    # This allows Rt to evolve smoothly over time
    latent = AR(
        damp_priors = [truncated(Normal(0.9, 0.05), 0, 1)],  # High persistence
        init_priors = [Normal(log(1.0), 0.5)],               # Start around R=1
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Moderate noise
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
    
    # 7. Generate the Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 8. Set up inference method
    println("Setting up inference...")
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
    
    # 10. Extract and process results
    println("Processing results...")
    
    # Extract the latent process Z_t (which is log(Rt))
    # Get posterior samples
    posterior_samples = results.result.value
    
    # Find Z_t columns (latent log(Rt) values)
    param_names = string.(keys(posterior_samples))
    z_indices = findall(name -> startswith(name, "Z_t"), param_names)
    
    if isempty(z_indices)
        error("Could not find Z_t parameters in results")
    end
    
    # Extract log(Rt) samples and convert to Rt
    log_rt_samples = hcat([posterior_samples[param_names[i]] for i in z_indices]...)
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_q025 = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_q975 = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    
    # Create results dataframe
    rt_estimates = DataFrame(
        day = 1:length(rt_mean),
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_q025 = rt_q025,
        rt_q975 = rt_q975
    )
    
    println("Rt estimation complete!")
    
    return rt_estimates, rt_samples, results
end

# Function to plot results
function plot_rt_estimates(rt_estimates, cases)
    # Create a two-panel plot
    p1 = plot(rt_estimates.day, rt_estimates.rt_median, 
              ribbon=(rt_estimates.rt_median - rt_estimates.rt_q025, 
                     rt_estimates.rt_q975 - rt_estimates.rt_median),
              fillalpha=0.3,
              label="Rt (95% CI)",
              xlabel="Days",
              ylabel="Rt",
              title="Estimated Reproduction Number Over Time",
              linewidth=2)
    
    # Add horizontal line at Rt = 1
    hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1", linewidth=2)
    
    # Plot case counts
    p2 = plot(1:length(cases), cases,
              label="Observed Cases",
              xlabel="Days", 
              ylabel="Daily Cases",
              title="Daily Case Counts",
              linewidth=2,
              color=:blue)
    
    # Combine plots
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    return combined_plot
end

# Function to summarize results
function summarize_rt_estimates(rt_estimates)
    println("\n=== Rt Estimation Summary ===")
    println("Time period: $(length(rt_estimates.day)) days")
    println("Mean Rt: $(round(mean(rt_estimates.rt_mean), digits=3))")
    println("Median Rt: $(round(median(rt_estimates.rt_median), digits=3))")
    
    # Find periods where Rt > 1 vs Rt < 1
    above_one = sum(rt_estimates.rt_median .> 1)
    below_one = sum(rt_estimates.rt_median .< 1)
    
    println("Days with Rt > 1: $above_one ($(round(100*above_one/length(rt_estimates.day), digits=1))%)")
    println("Days with Rt < 1: $below_one ($(round(100*below_one/length(rt_estimates.day), digits=1))%)")
    
    # Recent Rt trend (last 7 days)
    if length(rt_estimates.day) >= 7
        recent_rt = rt_estimates.rt_median[end-6:end]
        println("Recent Rt (last 7 days): $(round(mean(recent_rt), digits=3))")
    end
    
    return nothing
end

# Main execution function
function main(cases_file::String = "cases.csv")
    try
        # Estimate Rt
        rt_estimates, rt_samples, mcmc_results = estimate_rt_from_cases(cases_file)
        
        # Load cases for plotting
        cases = load_case_data(cases_file)
        
        # Display summary
        summarize_rt_estimates(rt_estimates)
        
        # Create and display plot
        rt_plot = plot_rt_estimates(rt_estimates, cases)
        display(rt_plot)
        
        # Save results
        CSV.write("rt_estimates.csv", rt_estimates)
        println("\nResults saved to 'rt_estimates.csv'")
        
        return rt_estimates, rt_samples, mcmc_results
        
    catch e
        println("Error during Rt estimation: $e")
        rethrow(e)
    end
end

# Run the analysis
# Uncomment the line below to execute:
# rt_estimates, rt_samples, results = main("cases.csv")

