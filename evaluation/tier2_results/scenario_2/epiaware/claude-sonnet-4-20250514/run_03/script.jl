
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/epiaware/claude-sonnet-4-20250514/run_03")

using Dates

using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using StatsPlots
using Statistics

# Load the data
function load_data(filename::String)
    df = CSV.read(filename, DataFrame)
    return df.cases, df.day_of_week
end

# Main function to estimate Rt with observation processes
function estimate_rt_with_observations(cases::Vector{Int}, day_of_week::Vector{Int})
    n_timepoints = length(cases)
    
    println("Fitting Rt model with $(n_timepoints) time points...")
    
    # 1. Define generation interval (COVID-19 typical values)
    gen_distribution = Gamma(6.5, 0.62)  # Mean ~4 days
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create infection model using renewal equation
    epi_model = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smoothness
    rt_latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0.0, 1.0)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Create day-of-week effects model
    # This creates a latent effect for each day of the week
    dayofweek_latent = BroadcastLatentModel(
        RepeatEach(7),  # Repeat 7-day pattern
        n_timepoints ÷ 7 + 1,  # Number of weeks needed
        HierarchicalNormal(std_prior = HalfNormal(0.2))  # Day effects
    )
    
    # 5. Create time-varying ascertainment model
    # This models the changing proportion of infections that are reported
    ascertainment_latent = RandomWalk(
        init_prior = Normal(logit(0.3), 0.5),  # Start around 30% ascertainment
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Slow changes
    )
    
    # 6. Combine all latent processes
    combined_latent = CombineLatentModels([
        rt_latent,           # log(Rt)
        dayofweek_latent,    # Day-of-week effects  
        ascertainment_latent # Ascertainment rate (logit scale)
    ])
    
    # 7. Create observation model with delay and overdispersion
    delay_distribution = Gamma(5.0, 1.0)  # Mean delay ~5 days
    base_obs = NegativeBinomialError(
        cluster_factor_prior = HalfNormal(0.2)  # Allows overdispersion
    )
    obs_with_delay = LatentDelay(base_obs, delay_distribution)
    
    # 8. Create custom observation model that incorporates day-of-week effects
    # and time-varying ascertainment
    function custom_observation_model()
        @submodel log_rt, dayofweek_effects, logit_ascertainment = combined_latent()
        
        # Extract day-of-week effects for actual dates
        dow_effects = dayofweek_effects[day_of_week]
        
        # Convert ascertainment from logit to probability scale
        ascertainment_prob = logistic.(logit_ascertainment)
        
        # Generate latent infections from Rt
        @submodel I_t = generate_latent_infs(epi_model, log_rt)()
        
        # Apply delay to get expected reported infections
        @submodel delayed_I = generate_latent_infs(
            LatentDelay(I_t, delay_distribution)
        )()
        
        # Apply ascertainment and day-of-week effects
        expected_cases = delayed_I .* ascertainment_prob .* exp.(dow_effects)
        
        # Generate observations with overdispersion
        @submodel y_t = generate_observations(
            base_obs, 
            missing,  # Will be filled with actual data
            expected_cases
        )()
        
        return (
            y_t = y_t,
            rt = exp.(log_rt),
            dayofweek_effects = dow_effects,
            ascertainment = ascertainment_prob,
            expected_cases = expected_cases,
            latent_infections = I_t
        )
    end
    
    return custom_observation_model
end

# Simplified approach using EpiProblem structure
function estimate_rt_simplified(cases::Vector{Int}, day_of_week::Vector{Int})
    n_timepoints = length(cases)
    
    println("Fitting simplified Rt model with $(n_timepoints) time points...")
    
    # 1. Define generation interval
    gen_distribution = Gamma(6.5, 0.62)
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create infection model
    epi_model = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)
    )
    
    # 3. Create latent model for log(Rt)
    latent_model = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0.0, 1.0)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Create observation model with delay and overdispersion
    delay_distribution = Gamma(5.0, 1.0)
    obs_model = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.2)),
        delay_distribution
    )
    
    # 5. Create EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi_model,
        latent_model = latent_model,
        observation_model = obs_model,
        tspan = (1, n_timepoints)
    )
    
    # 6. Generate Turing model
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 7. Define inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 1000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    println("Running MCMC sampling...")
    
    # 8. Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract and summarize results
function extract_results(results, n_timepoints::Int)
    # Extract posterior samples
    posterior_samples = results.chain
    
    # Get parameter names
    param_names = names(posterior_samples)
    
    println("Available parameters:")
    for name in param_names
        println("  ", name)
    end
    
    # Extract Rt estimates (exp of latent process Z_t)
    rt_samples = []
    for t in 1:n_timepoints
        z_param = "generated_quantities.generated_quantities[$t]"
        if z_param in string.(param_names)
            z_samples = posterior_samples[z_param]
            rt_samples_t = exp.(z_samples)
            push!(rt_samples, rt_samples_t)
        end
    end
    
    if !isempty(rt_samples)
        # Calculate summary statistics
        rt_mean = [mean(rt_t) for rt_t in rt_samples]
        rt_lower = [quantile(rt_t, 0.025) for rt_t in rt_samples]
        rt_upper = [quantile(rt_t, 0.975) for rt_t in rt_samples]
        
        return (
            rt_mean = rt_mean,
            rt_lower = rt_lower,
            rt_upper = rt_upper,
            samples = rt_samples
        )
    else
        println("Warning: Could not find Rt parameters in chain")
        return nothing
    end
end

# Function to create plots
function plot_results(rt_results, cases::Vector{Int})
    if rt_results === nothing
        println("No results to plot")
        return nothing
    end
    
    n_timepoints = length(cases)
    time_points = 1:n_timepoints
    
    # Plot Rt over time
    p1 = plot(time_points, rt_results.rt_mean, 
             ribbon = (rt_results.rt_mean .- rt_results.rt_lower,
                      rt_results.rt_upper .- rt_results.rt_mean),
             label = "Rt estimate", 
             title = "Reproduction Number Over Time",
             xlabel = "Days", 
             ylabel = "Rt",
             linewidth = 2)
    hline!([1.0], linestyle = :dash, color = :red, label = "Rt = 1")
    
    # Plot observed cases
    p2 = bar(time_points, cases, 
            label = "Observed cases",
            title = "Daily Case Counts",
            xlabel = "Days", 
            ylabel = "Cases",
            alpha = 0.7)
    
    # Combine plots
    combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 600))
    
    return combined_plot
end

# Main execution function
function main()
    # Load data
    println("Loading data...")
    cases, day_of_week = load_data("cases.csv")
    
    println("Data summary:")
    println("  Total days: ", length(cases))
    println("  Total cases: ", sum(cases))
    println("  Mean daily cases: ", round(mean(cases), digits=1))
    
    # Run analysis with simplified model
    results, epi_prob = estimate_rt_simplified(cases, day_of_week)
    
    # Extract results
    rt_results = extract_results(results, length(cases))
    
    # Create plots
    plt = plot_results(rt_results, cases)
    
    if plt !== nothing
        display(plt)
        savefig(plt, "rt_estimates.png")
        println("Plot saved as rt_estimates.png")
    end
    
    # Print summary
    if rt_results !== nothing
        println("\nRt Summary:")
        println("  Current Rt estimate: ", 
               round(rt_results.rt_mean[end], digits=2),
               " (95% CI: ", 
               round(rt_results.rt_lower[end], digits=2), "-",
               round(rt_results.rt_upper[end], digits=2), ")")
        
        println("  Mean Rt over period: ", 
               round(mean(rt_results.rt_mean), digits=2))
        
        # Check convergence
        println("\nModel diagnostics:")
        println("  Number of samples: ", size(results.chain, 1))
        println("  Number of chains: ", size(results.chain, 3))
        
        return (
            rt_estimates = rt_results,
            mcmc_results = results,
            model = epi_prob
        )
    else
        println("Analysis completed but could not extract Rt estimates")
        return results
    end
end

# For testing with synthetic data if cases.csv doesn't exist
function create_synthetic_data()
    n_days = 100
    dates = [Date(2023, 1, 1) + Day(i-1) for i in 1:n_days]
    
    # Generate synthetic Rt that varies over time
    true_rt = 1.5 .- 0.8 .* (1:n_days) ./ n_days .+ 0.2 .* sin.((1:n_days) .* 2π ./ 14)
    
    # Generate synthetic cases with renewal process (simplified)
    cases = zeros(Int, n_days)
    cases[1:7] .= [10, 12, 8, 15, 18, 14, 9]  # Initial cases
    
    for t in 8:n_days
        # Simplified renewal equation
        new_infections = true_rt[t] * sum(cases[max(1, t-7):t-1] .* [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05][1:min(7, t-1)])
        # Add noise and delay
        cases[t] = max(0, round(Int, new_infections * (0.8 + 0.4*rand()) * 
                               (t % 7 ∈ [6, 0] ? 0.7 : 1.0)))  # Weekend effect
    end
    
    day_of_week = [dayofweek(d) for d in dates]
    
    df = DataFrame(
        date = dates,
        cases = cases,
        day_of_week = day_of_week
    )
    
    CSV.write("cases.csv", df)
    println("Created synthetic data in cases.csv")
    
    return cases, day_of_week
end

# Run the analysis
if !isfile("cases.csv")
    println("cases.csv not found, creating synthetic data...")
    create_synthetic_data()
end

# Execute main analysis
results = main()

