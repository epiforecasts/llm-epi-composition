using EpiAware
using DataFrames
using CSV
using Distributions
using Turing
using Plots
using Statistics

# Load the data
function load_case_data(filename)
    df = CSV.read(filename, DataFrame)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_renewal(cases; 
                           gen_mean = 6.5, 
                           gen_std = 0.62,
                           delay_mean = 5.0,
                           delay_std = 1.0)
    
    println("Setting up Rt estimation using renewal equation...")
    println("Number of time points: $(length(cases))")
    
    # 1. Define generation interval distribution
    # Convert mean/std to shape/scale parameters for Gamma distribution
    gen_scale = gen_std^2 / gen_mean
    gen_shape = gen_mean / gen_scale
    gen_distribution = Gamma(gen_shape, gen_scale)
    
    println("Generation interval: Gamma(shape=$(round(gen_shape, digits=2)), scale=$(round(gen_scale, digits=2)))")
    
    # Create EpiData with generation interval
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create renewal infection model
    # Prior for initial seeding infections
    epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))
    
    # 3. Create latent model for log(Rt) - AR(1) process
    # This allows Rt to vary smoothly over time
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Innovation noise
    )
    
    # 4. Create observation model with reporting delay
    # Convert delay mean/std to Gamma parameters
    delay_scale = delay_std^2 / delay_mean
    delay_shape = delay_mean / delay_scale
    delay_distribution = Gamma(delay_shape, delay_scale)
    
    println("Reporting delay: Gamma(shape=$(round(delay_shape, digits=2)), scale=$(round(delay_scale, digits=2)))")
    
    # Negative binomial observation model to handle overdispersion
    obs_base = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs = LatentDelay(obs_base, delay_distribution)
    
    # 5. Compose the full epidemiological problem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, length(cases))
    )
    
    # 6. Generate the Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 7. Set up inference method
    println("Setting up MCMC inference...")
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
    println("Running MCMC sampling...")
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(results)
    # Extract the latent process Z_t (log Rt)
    log_rt_samples = results["latent_process[1]"]
    
    # Convert to Rt (exp transform)
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    n_timepoints = size(rt_samples, 2)
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    rt_lower_50 = vec(mapslices(x -> quantile(x, 0.25), rt_samples, dims=1))
    rt_upper_50 = vec(mapslices(x -> quantile(x, 0.75), rt_samples, dims=1))
    
    return DataFrame(
        time = 1:n_timepoints,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower_95 = rt_lower,
        rt_upper_95 = rt_upper,
        rt_lower_50 = rt_lower_50,
        rt_upper_50 = rt_upper_50
    )
end

# Function to plot results
function plot_rt_estimates(rt_estimates, cases)
    # Create subplot layout
    p1 = plot(rt_estimates.time, rt_estimates.rt_median, 
              ribbon = (rt_estimates.rt_median .- rt_estimates.rt_lower_95,
                       rt_estimates.rt_upper_95 .- rt_estimates.rt_median),
              fillalpha = 0.2, color = :blue, linewidth = 2,
              title = "Estimated Rt over Time", 
              xlabel = "Time (days)", 
              ylabel = "Reproduction Number (Rt)",
              label = "Rt (95% CI)")
    
    # Add 50% credible interval
    plot!(p1, rt_estimates.time, rt_estimates.rt_median,
          ribbon = (rt_estimates.rt_median .- rt_estimates.rt_lower_50,
                   rt_estimates.rt_upper_50 .- rt_estimates.rt_median),
          fillalpha = 0.4, color = :blue, label = "Rt (50% CI)")
    
    # Add horizontal line at Rt = 1
    hline!(p1, [1.0], linestyle = :dash, color = :red, linewidth = 2, label = "Rt = 1")
    
    # Plot observed cases
    p2 = bar(1:length(cases), cases, 
             title = "Observed Cases", 
             xlabel = "Time (days)", 
             ylabel = "Daily Cases",
             color = :orange, alpha = 0.7, label = "Cases")
    
    # Combine plots
    plot(p1, p2, layout = (2, 1), size = (800, 600))
end

# Function to extract posterior predictive checks
function extract_posterior_predictions(results)
    # Extract generated quantities if available
    if haskey(results, "generated_y_t")
        pred_samples = results["generated_y_t"]
        
        n_timepoints = size(pred_samples, 2)
        pred_mean = vec(mean(pred_samples, dims=1))
        pred_lower = vec(mapslices(x -> quantile(x, 0.025), pred_samples, dims=1))
        pred_upper = vec(mapslices(x -> quantile(x, 0.975), pred_samples, dims=1))
        
        return DataFrame(
            time = 1:n_timepoints,
            pred_mean = pred_mean,
            pred_lower = pred_lower,
            pred_upper = pred_upper
        )
    else
        println("Warning: Posterior predictive samples not found in results")
        return nothing
    end
end

# Main execution function
function main()
    println("=== Rt Estimation Using Renewal Equation ===")
    
    # Load data
    println("Loading case data...")
    cases = load_case_data("cases.csv")
    
    # Estimate Rt
    results, epi_prob = estimate_rt_renewal(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results)
    
    # Display summary
    println("\nRt Estimation Summary:")
    println("Time period: $(rt_estimates.time[1]) to $(rt_estimates.time[end])")
    println("Final Rt estimate: $(round(rt_estimates.rt_median[end], digits=3)) (95% CI: $(round(rt_estimates.rt_lower_95[end], digits=3))-$(round(rt_estimates.rt_upper_95[end], digits=3)))")
    
    # Count time points where Rt > 1
    prob_rt_greater_1 = mean(rt_estimates.rt_median .> 1.0)
    println("Proportion of time with Rt > 1: $(round(prob_rt_greater_1, digits=3))")
    
    # Create plots
    println("Creating plots...")
    p = plot_rt_estimates(rt_estimates, cases)
    display(p)
    
    # Save results
    CSV.write("rt_estimates.csv", rt_estimates)
    println("Rt estimates saved to rt_estimates.csv")
    
    # Extract posterior predictions for model checking
    pred_data = extract_posterior_predictions(results)
    if pred_data !== nothing
        CSV.write("posterior_predictions.csv", pred_data)
        println("Posterior predictions saved to posterior_predictions.csv")
    end
    
    return rt_estimates, results
end

# Alternative simplified function for quick estimation
function quick_rt_estimate(cases_file = "cases.csv")
    cases = load_case_data(cases_file)
    results, _ = estimate_rt_renewal(cases)
    rt_estimates = extract_rt_estimates(results)
    return rt_estimates
end

# Run the main analysis
if abspath(PROGRAM_FILE) == @__FILE__
    rt_estimates, results = main()
end


# Quick estimation
rt_estimates = quick_rt_estimate("cases.csv")

# Full analysis with custom parameters
results, epi_prob = estimate_rt_renewal(cases, 
                                      gen_mean=7.0, 
                                      delay_mean=6.0)

