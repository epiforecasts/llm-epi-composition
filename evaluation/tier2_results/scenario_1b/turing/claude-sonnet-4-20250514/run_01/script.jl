
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/turing/claude-sonnet-4-20250514/run_01")

using Dates

using Turing, CSV, DataFrames, Distributions, Plots, StatsPlots, LinearAlgebra
using Random
Random.seed!(42)

# Load the data
function load_data(filename="cases.csv")
    df = CSV.read(filename, DataFrame)
    return df.cases
end

# Define generation interval PMF (COVID-19 parameters)
function generation_interval_pmf(max_days=20)
    # Using a discretized Gamma distribution with mean ~5.2 days, std ~1.8 days
    # Based on COVID-19 literature
    α = 8.5  # shape parameter
    β = 1.6  # rate parameter
    
    pmf = zeros(max_days)
    for s in 1:max_days
        # Probability mass for day s (discretized continuous distribution)
        pmf[s] = cdf(Gamma(α, 1/β), s) - cdf(Gamma(α, 1/β), s-1)
    end
    
    # Normalize to ensure sum = 1
    return pmf ./ sum(pmf)
end

# Define delay from infection to case reporting PMF
function reporting_delay_pmf(max_days=15)
    # Incubation period + reporting delay
    # Using discretized Gamma with mean ~7 days
    α = 4.0
    β = 0.57
    
    pmf = zeros(max_days)
    for d in 1:max_days
        pmf[d] = cdf(Gamma(α, 1/β), d) - cdf(Gamma(α, 1/β), d-1)
    end
    
    return pmf ./ sum(pmf)
end

@model function renewal_model(cases, generation_pmf, delay_pmf)
    n_days = length(cases)
    max_gen = length(generation_pmf)
    max_delay = length(delay_pmf)
    
    # Priors
    # Initial infections (seeding period)
    I₀ ~ LogNormal(log(10), 1)  # Initial seed infections
    seed_growth ~ Normal(0.1, 0.1)  # Growth rate during seeding
    
    # Rt parameters - using random walk on log scale
    log_R₁ ~ Normal(log(2), 0.5)  # Initial Rt
    σ_R ~ Exponential(0.2)  # Innovation variance for Rt random walk
    
    # Observation model parameters
    ψ ~ Beta(2, 8)  # Reporting rate
    φ ~ Exponential(0.1)  # Overdispersion parameter for negative binomial
    
    # Initialize vectors
    infections = Vector{Real}(undef, n_days)
    expected_cases = Vector{Real}(undef, n_days)
    log_Rt = Vector{Real}(undef, n_days)
    
    # Seeding period (first max_gen days)
    for t in 1:min(max_gen, n_days)
        infections[t] = I₀ * exp(seed_growth * (t-1))
        log_Rt[t] = log_R₁
    end
    
    # Random walk for log(Rt)
    for t in 2:n_days
        if t > max_gen
            log_Rt[t] ~ Normal(log_Rt[t-1], σ_R)
        end
    end
    
    # Renewal equation for infections
    for t in (max_gen+1):n_days
        Rt = exp(log_Rt[t])
        
        # Convolution with generation interval
        infectiousness = 0.0
        for s in 1:min(max_gen, t-1)
            infectiousness += infections[t-s] * generation_pmf[s]
        end
        
        infections[t] = Rt * infectiousness
    end
    
    # Delay from infections to case reports
    for t in 1:n_days
        expected_cases[t] = 0.0
        for d in 1:min(max_delay, t)
            expected_cases[t] += infections[t-d+1] * delay_pmf[d] * ψ
        end
        expected_cases[t] = max(expected_cases[t], 1e-6)  # Numerical stability
    end
    
    # Observation model - Negative Binomial
    for t in 1:n_days
        # Parameterization: NB(r, p) where mean = r*p/(1-p), var = r*p/(1-p)²
        r = 1/φ
        p = expected_cases[t] / (expected_cases[t] + r)
        cases[t] ~ NegativeBinomial(r, 1-p)
    end
    
    return (infections=infections, Rt=exp.(log_Rt), expected_cases=expected_cases)
end

# Function to fit the model and extract results
function estimate_rt(cases_data; n_samples=2000, n_warmup=1000, n_chains=4)
    println("Setting up model...")
    
    # Get PMFs
    gen_pmf = generation_interval_pmf()
    delay_pmf = reporting_delay_pmf()
    
    println("Generation interval PMF (first 10 days): ", round.(gen_pmf[1:10], digits=3))
    println("Delay PMF (first 10 days): ", round.(delay_pmf[1:10], digits=3))
    
    # Create model
    model = renewal_model(cases_data, gen_pmf, delay_pmf)
    
    println("Fitting model with NUTS sampler...")
    println("Number of observations: ", length(cases_data))
    
    # Sample using NUTS
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    return chain, gen_pmf, delay_pmf
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(chain, n_days)
    # Extract Rt samples
    rt_samples = []
    
    for t in 1:n_days
        param_name = "infections[$(t)]"
        if param_name in names(chain)
            push!(rt_samples, vec(chain["Rt[$(t)]"]))
        end
    end
    
    # If the above doesn't work, try alternative extraction
    if isempty(rt_samples)
        # Extract all Rt parameters
        rt_params = [name for name in names(chain) if startswith(string(name), "log_Rt")]
        rt_samples = [exp.(vec(chain[param])) for param in rt_params]
    end
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_lower = [quantile(samples, 0.025) for samples in rt_samples]
    rt_upper = [quantile(samples, 0.975) for samples in rt_samples]
    rt_median = [median(samples) for samples in rt_samples]
    
    return DataFrame(
        day = 1:length(rt_mean),
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper
    )
end

# Plotting function
function plot_rt_estimates(rt_df, cases_data)
    n_days = length(cases_data)
    days = 1:n_days
    
    # Create subplot layout
    p1 = plot(days, cases_data, 
             seriestype=:line, 
             linewidth=2, 
             title="Observed Cases", 
             xlabel="Day", 
             ylabel="Cases",
             legend=false)
    
    p2 = plot(rt_df.day, rt_df.rt_median,
             ribbon=(rt_df.rt_median .- rt_df.rt_lower, rt_df.rt_upper .- rt_df.rt_median),
             linewidth=2,
             fillalpha=0.3,
             title="Estimated Rt Over Time",
             xlabel="Day",
             ylabel="Rt",
             label="Rt (95% CI)",
             legend=:topright)
    
    # Add horizontal line at Rt = 1
    hline!(p2, [1.0], linestyle=:dash, linecolor=:red, linewidth=1, label="Rt = 1")
    
    return plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main execution function
function main()
    println("Loading data...")
    
    # Create sample data if file doesn't exist
    if !isfile("cases.csv")
        println("Creating sample data...")
        dates = [Date(2020, 3, 1) + Day(i-1) for i in 1:100]
        # Simulate some realistic case data
        true_rt = [2.5 * exp(-0.05*i) + 0.8 + 0.3*sin(i/10) for i in 1:100]
        cases_sim = max.(1, round.(Int, 50 * cumprod(true_rt .^ (1/5)) .+ 10*randn(100)))
        
        sample_df = DataFrame(date=dates, cases=cases_sim)
        CSV.write("cases.csv", sample_df)
        println("Sample data created and saved to cases.csv")
    end
    
    cases_data = load_data("cases.csv")
    
    println("Estimating Rt...")
    chain, gen_pmf, delay_pmf = estimate_rt(cases_data, n_samples=1000, n_warmup=500, n_chains=2)
    
    println("Extracting results...")
    rt_estimates = extract_rt_estimates(chain, length(cases_data))
    
    println("Creating plots...")
    plt = plot_rt_estimates(rt_estimates, cases_data)
    display(plt)
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("======================")
    println("Number of days: ", length(cases_data))
    println("Mean Rt over period: ", round(mean(rt_estimates.rt_mean), digits=2))
    println("Final Rt estimate: ", round(rt_estimates.rt_mean[end], digits=2), 
            " (95% CI: ", round(rt_estimates.rt_lower[end], digits=2), 
            " - ", round(rt_estimates.rt_upper[end], digits=2), ")")
    
    # Show model diagnostics
    println("\nModel Diagnostics:")
    println("==================")
    println(chain)
    
    return rt_estimates, chain, plt
end

# Run the analysis
rt_results, fitted_chain, result_plot = main()

# Additional utility functions for analysis
function get_model_summary(chain)
    """Get summary statistics for key model parameters"""
    key_params = ["σ_R", "ψ", "φ", "I₀", "seed_growth"]
    
    summary_df = DataFrame(
        parameter = String[],
        mean = Float64[],
        std = Float64[],
        q025 = Float64[],
        q975 = Float64[]
    )
    
    for param in key_params
        if param in names(chain)
            samples = vec(chain[param])
            push!(summary_df, (
                parameter = param,
                mean = mean(samples),
                std = std(samples),
                q025 = quantile(samples, 0.025),
                q975 = quantile(samples, 0.975)
            ))
        end
    end
    
    return summary_df
end

# Print parameter summary
param_summary = get_model_summary(fitted_chain)
println("\nParameter Estimates:")
println("====================")
println(param_summary)


