
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/epiaware/claude-sonnet-4-20250514/run_02")

using EpiAware
using Distributions
using Turing
using CSV
using DataFrames
using Plots
using StatsPlots
using Random

# Set random seed for reproducibility
Random.seed!(123)

# Load the data
function load_data(filename)
    df = CSV.read(filename, DataFrame)
    return df
end

# Main analysis function
function estimate_rt_with_observation_processes(data_file::String)
    
    # Load data
    df = load_data(data_file)
    cases = df.cases
    day_of_week = df.day_of_week
    n_timepoints = length(cases)
    
    println("Data loaded: $(n_timepoints) time points")
    println("Total cases: $(sum(cases))")
    
    # 1. Define generation interval (COVID-19 typical values)
    gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, shape consistent with COVID-19
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(max(10.0, cases[1])), 1.0)
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smooth evolution
    rt_latent = AR(
        damp_priors = [truncated(Normal(0.9, 0.05), 0.0, 0.99)],  # Strong persistence
        init_priors = [Normal(0.0, 0.5)],  # log(Rt) starts around 1
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))  # Moderate variation
    )
    
    # 4. Create day-of-week effects model
    # This creates 7 random effects, one for each day of the week
    dow_effect = BroadcastLatentModel(
        RepeatBlock(),  # Repeat the 7-day pattern
        7,              # 7 days in a week
        IID(Normal(0.0, 0.2))  # Day-of-week effects with moderate variation
    )
    
    # 5. Create time-varying ascertainment model
    # This captures changes in testing/reporting over time
    ascertainment_latent = RandomWalk(
        init_prior = Normal(logit(0.3), 0.5),  # Start at ~30% ascertainment
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Slow changes
    )
    
    # 6. Combine all latent processes
    # The combined model will have: [log_Rt, dow_effects..., ascertainment_logit]
    combined_latent = CombineLatentModels([rt_latent, dow_effect, ascertainment_latent])
    
    # 7. Create observation model with reporting delay and overdispersion
    delay_dist = Gamma(5.0, 1.2)  # Mean delay ~6 days from infection to reporting
    
    # Base observation model with overdispersion (negative binomial)
    base_obs = NegativeBinomialError(
        cluster_factor_prior = HalfNormal(0.2)  # Allows for overdispersion
    )
    
    # Add reporting delay
    obs_with_delay = LatentDelay(base_obs, delay_dist)
    
    # 8. Create custom observation model that incorporates day-of-week and ascertainment effects
    # We need to define a custom transformation function
    function observation_transform(infections, latent_vars, day_of_week_vec)
        n_time = length(infections)
        
        # Extract components from latent variables
        # latent_vars structure: [log_Rt (n_time), dow_effects (7), ascertainment_logit (n_time)]
        rt_start = 1
        rt_end = n_time
        dow_start = rt_end + 1
        dow_end = dow_start + 6  # 7 effects (indexed 0-6, so 6 additional)
        asc_start = dow_end + 1
        asc_end = asc_start + n_time - 1
        
        log_rt = latent_vars[rt_start:rt_end]
        dow_effects = latent_vars[dow_start:dow_end]
        ascertainment_logits = latent_vars[asc_start:asc_end]
        
        # Convert to probabilities
        ascertainment_probs = logistic.(ascertainment_logits)
        
        # Apply day-of-week effects
        dow_multipliers = exp.(dow_effects[day_of_week_vec])
        
        # Calculate expected reported cases
        expected_cases = infections .* ascertainment_probs .* dow_multipliers
        
        return expected_cases
    end
    
    # For EpiAware, we need to create a custom observation model
    # We'll use a simpler approach by modifying the expected cases within the model
    
    # 9. Compose the full EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = combined_latent,
        observation_model = obs_with_delay,
        tspan = (1, n_timepoints)
    )
    
    # 10. Generate Turing model with custom modifications for our observation process
    @model function custom_epiaware_model(y_t, day_of_week_indices)
        # Generate latent variables
        latent_vars ~ generate_latent(combined_latent, n_timepoints)
        
        # Extract Rt values (first n_timepoints elements)
        log_rt = latent_vars[1:n_timepoints]
        rt = exp.(log_rt)
        
        # Generate infections using renewal equation
        infections ~ generate_latent_infs(epi, log_rt)
        
        # Extract day-of-week effects (next 7 elements)
        dow_effects = latent_vars[(n_timepoints+1):(n_timepoints+7)]
        
        # Extract ascertainment (last n_timepoints elements)  
        ascertainment_logits = latent_vars[(n_timepoints+8):(2*n_timepoints+7)]
        ascertainment = logistic.(ascertainment_logits)
        
        # Apply observation process transformations
        dow_multipliers = exp.(dow_effects[day_of_week_indices])
        expected_reported = infections .* ascertainment .* dow_multipliers
        
        # Apply delay and generate observations
        delayed_expected ~ generate_observations(obs_with_delay, y_t, expected_reported)
        
        return (
            rt = rt,
            infections = infections,
            ascertainment = ascertainment,
            dow_effects = dow_effects,
            expected_reported = expected_reported
        )
    end
    
    # 11. Prepare data for inference
    mdl = custom_epiaware_model(cases, day_of_week)
    
    # 12. Set up inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 1000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    println("Starting MCMC sampling...")
    
    # 13. Run inference
    results = sample(mdl, NUTS(0.8), MCMCThreads(), 1000, 4)
    
    println("MCMC sampling completed!")
    
    return results, df
end

# Function to extract and summarize results
function extract_results(results, df)
    # Extract posterior samples
    rt_samples = Array(group(results, :rt))
    ascertainment_samples = Array(group(results, :ascertainment))
    dow_effects_samples = Array(group(results, :dow_effects))
    
    # Calculate summaries
    rt_mean = mean(rt_samples, dims=1)[1, :]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:size(rt_samples, 2)]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:size(rt_samples, 2)]
    
    ascertainment_mean = mean(ascertainment_samples, dims=1)[1, :]
    ascertainment_lower = [quantile(ascertainment_samples[:, i], 0.025) for i in 1:size(ascertainment_samples, 2)]
    ascertainment_upper = [quantile(ascertainment_samples[:, i], 0.975) for i in 1:size(ascertainment_samples, 2)]
    
    dow_mean = mean(dow_effects_samples, dims=1)[1, :]
    dow_lower = [quantile(dow_effects_samples[:, i], 0.025) for i in 1:size(dow_effects_samples, 2)]
    dow_upper = [quantile(dow_effects_samples[:, i], 0.975) for i in 1:size(dow_effects_samples, 2)]
    
    return (
        rt = (mean = rt_mean, lower = rt_lower, upper = rt_upper),
        ascertainment = (mean = ascertainment_mean, lower = ascertainment_lower, upper = ascertainment_upper),
        dow_effects = (mean = dow_mean, lower = dow_lower, upper = dow_upper)
    )
end

# Function to create plots
function plot_results(results_summary, df)
    dates = df.date
    cases = df.cases
    
    # Plot 1: Rt over time
    p1 = plot(dates, results_summary.rt.mean, 
             ribbon = (results_summary.rt.mean - results_summary.rt.lower,
                      results_summary.rt.upper - results_summary.rt.mean),
             label = "Rt", title = "Reproduction Number (Rt) Over Time",
             xlabel = "Date", ylabel = "Rt", linewidth = 2)
    hline!([1.0], linestyle = :dash, color = :red, label = "Rt = 1")
    
    # Plot 2: Ascertainment over time
    p2 = plot(dates, results_summary.ascertainment.mean * 100,
             ribbon = ((results_summary.ascertainment.mean - results_summary.ascertainment.lower) * 100,
                      (results_summary.ascertainment.upper - results_summary.ascertainment.mean) * 100),
             label = "Ascertainment Rate", title = "Time-varying Ascertainment Rate",
             xlabel = "Date", ylabel = "Ascertainment (%)", linewidth = 2)
    
    # Plot 3: Day-of-week effects
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    p3 = bar(dow_names, exp.(results_summary.dow_effects.mean),
            yerror = (exp.(results_summary.dow_effects.mean) - exp.(results_summary.dow_effects.lower),
                     exp.(results_summary.dow_effects.upper) - exp.(results_summary.dow_effects.mean)),
            title = "Day-of-Week Effects (Multiplicative)",
            xlabel = "Day of Week", ylabel = "Reporting Multiplier", legend = false)
    hline!([1.0], linestyle = :dash, color = :red)
    
    # Plot 4: Observed vs expected cases (sample from posterior predictive)
    p4 = plot(dates, cases, label = "Observed Cases", title = "Observed vs Expected Cases",
             xlabel = "Date", ylabel = "Daily Cases", linewidth = 2, alpha = 0.7)
    
    return plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 800))
end

# Main execution function
function main(data_file = "cases.csv")
    println("=== Estimating Rt with Complex Observation Processes ===")
    
    try
        # Run the analysis
        results, df = estimate_rt_with_observation_processes(data_file)
        
        # Extract results
        println("Extracting posterior summaries...")
        results_summary = extract_results(results, df)
        
        # Print summary statistics
        println("\n=== RESULTS SUMMARY ===")
        println("Rt estimates:")
        println("  Mean Rt: $(round(mean(results_summary.rt.mean), digits=2))")
        println("  Min Rt: $(round(minimum(results_summary.rt.mean), digits=2))")
        println("  Max Rt: $(round(maximum(results_summary.rt.mean), digits=2))")
        
        println("\nAscertainment rates:")
        println("  Mean ascertainment: $(round(mean(results_summary.ascertainment.mean) * 100, digits=1))%")
        println("  Min ascertainment: $(round(minimum(results_summary.ascertainment.mean) * 100, digits=1))%")
        println("  Max ascertainment: $(round(maximum(results_summary.ascertainment.mean) * 100, digits=1))%")
        
        println("\nDay-of-week effects (multiplicative):")
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for (i, name) in enumerate(dow_names)
            effect = exp(results_summary.dow_effects.mean[i])
            println("  $name: $(round(effect, digits=2))")
        end
        
        # Create plots
        println("\nGenerating plots...")
        final_plot = plot_results(results_summary, df)
        display(final_plot)
        
        # Save results
        savefig(final_plot, "rt_estimation_results.png")
        println("Plot saved as 'rt_estimation_results.png'")
        
        return results_summary, df, results
        
    catch e
        println("Error occurred: $e")
        rethrow(e)
    end
end

# Run the analysis (uncomment to execute)
# results_summary, df, mcmc_results = main("cases.csv")

