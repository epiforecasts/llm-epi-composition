
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/epiaware/claude-sonnet-4-20250514/run_01")

using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics
using StatsPlots

# Load and prepare the data
function load_data(filename="cases.csv")
    df = CSV.read(filename, DataFrame)
    return df.cases, df.day_of_week
end

# Main function to estimate Rt with observation processes
function estimate_rt_with_observation_processes(cases, day_of_week)
    n_timepoints = length(cases)
    
    println("Setting up model components...")
    
    # 1. Define generation interval (COVID-19 typical values)
    gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, appropriate for COVID-19
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(mean(cases[1:7])), 1.0)
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smoothness
    rt_process = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # Strong persistence
        init_priors = [Normal(0.0, 0.5)],  # Initial log(Rt) ~ log(1) = 0
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Create day-of-week effects model
    # This creates a repeating pattern for 7 days of the week
    dow_effects = BroadcastLatentModel(
        CyclicRepeat(7),  # Repeat every 7 days
        [Normal(0.0, 0.2) for _ in 1:7]  # Prior for each day effect
    )
    
    # 5. Create time-varying ascertainment model
    # This allows the reporting rate to change smoothly over time
    ascertainment_process = RandomWalk(
        init_prior = Normal(0.0, 0.5),  # Initial ascertainment on log scale
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Slow changes
    )
    
    # 6. Combine latent processes
    # We need Rt process, day-of-week effects, and ascertainment
    combined_latent = CombineLatentModels([
        rt_process,           # log(Rt) over time
        dow_effects,          # Day-of-week multiplicative effects  
        ascertainment_process # Time-varying ascertainment
    ])
    
    # 7. Create observation model with delays and overdispersion
    # Delay from infection to reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean delay ~5 days
    
    # Base observation model with overdispersion
    base_obs = NegativeBinomialError(
        cluster_factor_prior = HalfNormal(0.1)
    )
    
    # Add delay to observation model
    obs_with_delay = LatentDelay(base_obs, delay_dist)
    
    # 8. Create custom observation model that incorporates all effects
    # We need to modify the observation model to handle multiple latent processes
    obs_model = CustomObservationModel(obs_with_delay, n_timepoints, day_of_week)
    
    # 9. Compose into EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = combined_latent,
        observation_model = obs_model,
        tspan = (1, n_timepoints)
    )
    
    println("Generating Turing model...")
    
    # 10. Generate the full model
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    println("Starting inference...")
    
    # 11. Set up inference
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 200)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 12. Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Custom observation model to handle multiple latent processes
struct CustomObservationModel{T,S} <: AbstractObservationModel
    base_model::T
    n_timepoints::Int
    day_of_week::S
end

function EpiAware.generate_observations(obs::CustomObservationModel, y_t, expected_infections)
    @model function custom_obs_model(y_t, expected_infections, latent_processes)
        # Extract the different latent processes
        # latent_processes should contain: [log_Rt, dow_effects, log_ascertainment]
        n_rt = obs.n_timepoints
        
        log_Rt = latent_processes[1:n_rt]
        dow_effects = latent_processes[(n_rt+1):(n_rt+7)]
        log_ascertainment = latent_processes[(n_rt+8):end]
        
        # Apply day-of-week effects
        dow_multiplier = [dow_effects[obs.day_of_week[t]] for t in 1:obs.n_timepoints]
        
        # Apply ascertainment
        ascertainment = exp.(log_ascertainment)
        
        # Modify expected observations
        adjusted_expected = expected_infections .* ascertainment .* exp.(dow_multiplier)
        
        # Use base observation model
        base_model = obs.base_model
        return generate_observations(base_model, y_t, adjusted_expected)
    end
    
    return custom_obs_model
end

# Simplified version using available EpiAware components
function estimate_rt_simplified(cases, day_of_week)
    n_timepoints = length(cases)
    
    println("Setting up simplified model...")
    
    # 1. Generation interval
    gen_int = Gamma(6.5, 0.62)
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Infection model
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(mean(cases[1:7])), 1.0)
    )
    
    # 3. Latent model for log(Rt) with day-of-week effects
    rt_process = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Observation model with delay and overdispersion
    delay_dist = Gamma(5.0, 1.0)
    base_obs = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs_model = LatentDelay(base_obs, delay_dist)
    
    # 5. Add time-varying ascertainment
    ascertainment_rw = RandomWalk(
        init_prior = Normal(0.0, 0.5),
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))
    )
    
    obs_with_ascertainment = Ascertainment(
        obs_model,
        ascertainment_rw,
        (Y, x) -> Y .* exp.(x)  # Multiplicative ascertainment
    )
    
    # 6. Compose model
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = rt_process,
        observation_model = obs_with_ascertainment,
        tspan = (1, n_timepoints)
    )
    
    println("Running inference...")
    
    # 7. Generate and fit model
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 150)],
        sampler = NUTSampler(
            ndraws = 1500,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract and analyze results
function analyze_results(results, cases, day_of_week)
    println("Analyzing results...")
    
    # Extract parameter chains
    chains = results.chains
    
    # Extract Rt estimates (exp of latent process)
    n_timepoints = length(cases)
    
    # Get latent process samples (these are log(Rt))
    log_rt_samples = []
    for i in 1:n_timepoints
        param_name = "Z_t[$i]"
        if param_name in string.(keys(chains))
            push!(log_rt_samples, vec(chains[param_name]))
        end
    end
    
    # Convert to Rt
    rt_samples = [exp.(samples) for samples in log_rt_samples]
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_lower = [quantile(samples, 0.025) for samples in rt_samples]
    rt_upper = [quantile(samples, 0.975) for samples in rt_samples]
    
    # Extract other parameters if available
    ascertainment_samples = []
    dow_effects_samples = []
    
    # Try to extract ascertainment parameters
    for i in 1:n_timepoints
        param_name = "ascertainment[$i]"
        if param_name in string.(keys(chains))
            push!(ascertainment_samples, vec(chains[param_name]))
        end
    end
    
    # Try to extract day-of-week effects
    for i in 1:7
        param_name = "dow_effect[$i]"
        if param_name in string.(keys(chains))
            push!(dow_effects_samples, vec(chains[param_name]))
        end
    end
    
    return (
        rt_mean = rt_mean,
        rt_lower = rt_lower, 
        rt_upper = rt_upper,
        rt_samples = rt_samples,
        ascertainment_samples = ascertainment_samples,
        dow_effects_samples = dow_effects_samples,
        chains = chains
    )
end

# Plotting function
function plot_results(results_analysis, cases, day_of_week)
    n_timepoints = length(cases)
    dates = 1:n_timepoints
    
    # Plot Rt over time
    p1 = plot(dates, results_analysis.rt_mean, 
              ribbon=(results_analysis.rt_mean .- results_analysis.rt_lower,
                     results_analysis.rt_upper .- results_analysis.rt_mean),
              label="Rt estimate", linewidth=2, color=:blue, alpha=0.7,
              title="Time-varying Reproduction Number (Rt)",
              xlabel="Time", ylabel="Rt")
    hline!([1.0], label="Rt = 1", linestyle=:dash, color=:red)
    
    # Plot observed cases
    p2 = plot(dates, cases, label="Observed cases", 
              title="Observed Cases", xlabel="Time", ylabel="Cases",
              color=:black, linewidth=1)
    
    # Combine plots
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    return combined_plot
end

# Main execution function
function main()
    println("Loading data...")
    
    # Load data (replace with actual file path)
    cases, day_of_week = load_data("cases.csv")
    
    println("Data loaded: $(length(cases)) time points")
    println("Running Rt estimation with observation processes...")
    
    # Run the simplified model (more likely to work with current EpiAware)
    results, epi_prob = estimate_rt_simplified(cases, day_of_week)
    
    println("Analyzing results...")
    
    # Analyze results
    analysis = analyze_results(results, cases, day_of_week)
    
    # Print summary
    println("\n=== RESULTS SUMMARY ===")
    println("Mean Rt over time:")
    for (i, rt) in enumerate(analysis.rt_mean)
        if i <= 10 || i > length(analysis.rt_mean) - 5  # Show first 10 and last 5
            println("  Time $i: Rt = $(round(rt, digits=3)) [$(round(analysis.rt_lower[i], digits=3)), $(round(analysis.rt_upper[i], digits=3))]")
        elseif i == 11
            println("  ...")
        end
    end
    
    # Create plots
    println("\nCreating plots...")
    result_plot = plot_results(analysis, cases, day_of_week)
    display(result_plot)
    
    # Save results
    println("Saving results...")
    
    # Create results DataFrame
    results_df = DataFrame(
        time = 1:length(cases),
        rt_mean = analysis.rt_mean,
        rt_lower = analysis.rt_lower,
        rt_upper = analysis.rt_upper,
        observed_cases = cases,
        day_of_week = day_of_week
    )
    
    CSV.write("rt_estimates.csv", results_df)
    savefig(result_plot, "rt_estimates_plot.png")
    
    println("Analysis complete!")
    println("Results saved to: rt_estimates.csv")
    println("Plot saved to: rt_estimates_plot.png")
    
    return results, analysis, epi_prob
end

# Example of how to create synthetic data for testing
function create_synthetic_data(n_days=100)
    # Create synthetic case data with day-of-week effects
    dates = 1:n_days
    day_of_week = mod1.(dates, 7)
    
    # True Rt that varies over time
    true_rt = 1.5 * exp.(-0.02 * dates) .+ 0.5  # Declining epidemic
    
    # Day-of-week effects (multiplicative)
    dow_effects = [1.0, 0.9, 0.8, 0.85, 0.95, 1.1, 0.7]  # Lower on weekends
    
    # Generate synthetic cases (simplified)
    base_cases = 100 .+ 50 * sin.(0.1 * dates)  # Base pattern
    cases = Int.(round.(base_cases .* [dow_effects[d] for d in day_of_week] .* 
                       (0.8 .+ 0.4 * rand(n_days))))  # Add noise
    
    # Create DataFrame and save
    df = DataFrame(
        date = [Date(2023, 1, 1) + Day(i-1) for i in 1:n_days],
        cases = cases,
        day_of_week = day_of_week
    )
    
    CSV.write("cases.csv", df)
    
    println("Synthetic data created and saved to cases.csv")
    return df
end

# Uncomment the following lines to run:
# create_synthetic_data(80)  # Create synthetic data
# results, analysis, model = main()  # Run the analysis

