using Pkg
# Pkg.add(["Turing", "CSV", "DataFrames", "Distributions", "Plots", "StatsPlots", "Dates"])

using Turing
using CSV
using DataFrames
using Distributions
using Plots
using StatsPlots
using Dates
using LinearAlgebra

# Load and prepare the data
function load_data(filename::String)
    df = CSV.read(filename, DataFrame)
    df.date = Date.(df.date)
    sort!(df, :date)
    return df
end

# Define the generation interval (time from infection to onward transmission)
# Using a discretized gamma distribution with mean 5.2 days, std 2.8 days
function generation_interval(max_days::Int = 20)
    # Parameters for gamma distribution (mean=5.2, std=2.8)
    mean_gi = 5.2
    std_gi = 2.8
    shape = (mean_gi / std_gi)^2
    rate = mean_gi / std_gi^2
    
    # Discretize the generation interval
    gi = zeros(max_days)
    for i in 1:max_days
        gi[i] = cdf(Gamma(shape, 1/rate), i) - cdf(Gamma(shape, 1/rate), i-1)
    end
    
    # Normalize to ensure sum = 1
    gi = gi ./ sum(gi)
    return gi
end

# Turing model for estimating time-varying Rt
@model function rt_model(cases, generation_interval)
    n_days = length(cases)
    gi_length = length(generation_interval)
    
    # Priors
    # Initial reproduction number
    R0 ~ LogNormal(log(2.0), 0.5)
    
    # Random walk standard deviation for Rt
    σ_rw ~ Exponential(0.1)
    
    # Reporting probability
    ρ ~ Beta(2, 2)
    
    # Over-dispersion parameter for negative binomial
    φ ~ Exponential(10.0)
    
    # Random walk for log(Rt)
    log_Rt = Vector{Real}(undef, n_days)
    log_Rt[1] = log(R0)
    
    for t in 2:n_days
        log_Rt[t] ~ Normal(log_Rt[t-1], σ_rw)
    end
    
    Rt = exp.(log_Rt)
    
    # Calculate expected cases based on renewal equation
    λ = Vector{Real}(undef, n_days)
    
    for t in 1:n_days
        if t <= gi_length
            # For early days, use a simple exponential growth model
            λ[t] = cases[1] * exp((t-1) * (log_Rt[t] - 1) / 5.2)
        else
            # Renewal equation: λ(t) = Rt * Σ(λ(t-s) * g(s))
            infectivity = 0.0
            for s in 1:min(gi_length, t-1)
                if t-s >= 1
                    infectivity += λ[t-s] * generation_interval[s]
                end
            end
            λ[t] = Rt[t] * infectivity
        end
        
        # Ensure λ is positive
        λ[t] = max(λ[t], 1e-10)
    end
    
    # Likelihood: reported cases ~ NegativeBinomial(expected_cases, φ)
    for t in 1:n_days
        expected_cases = ρ * λ[t]
        # Parameterization: NegativeBinomial2(μ, φ) where var = μ + μ²/φ
        p = φ / (φ + expected_cases)
        r = φ
        cases[t] ~ NegativeBinomial(r, p)
    end
    
    return Rt
end

# Function to run the estimation
function estimate_rt(cases_data; n_samples=2000, n_chains=4, n_adapt=1000)
    println("Setting up model...")
    
    # Remove initial zeros and very small values
    start_idx = findfirst(x -> x >= 5, cases_data.cases)
    if start_idx === nothing
        start_idx = 1
    end
    
    cases = cases_data.cases[start_idx:end]
    dates = cases_data.date[start_idx:end]
    
    # Get generation interval
    gi = generation_interval(20)
    
    println("Running MCMC with $(length(cases)) data points...")
    
    # Sample from the model
    model = rt_model(cases, gi)
    
    # Use NUTS sampler
    sampler = NUTS(n_adapt, 0.8)
    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
    
    return chain, dates, cases
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(chain, dates)
    # Get Rt parameter names
    rt_params = [Symbol("Rt[$i]") for i in 1:length(dates)]
    
    # Extract samples
    rt_samples = Array(group(chain, :Rt))
    
    # Calculate summary statistics
    rt_mean = mean(rt_samples, dims=1)[:]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:size(rt_samples, 2)]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:size(rt_samples, 2)]
    rt_median = [quantile(rt_samples[:, i], 0.5) for i in 1:size(rt_samples, 2)]
    
    # Create results DataFrame
    results = DataFrame(
        date = dates,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper
    )
    
    return results
end

# Function to plot results
function plot_rt_estimates(results, cases_data)
    # Create subplot layout
    p1 = plot(cases_data.date, cases_data.cases, 
              line=:stem, marker=:circle, markersize=2,
              title="Daily Cases", ylabel="Cases", 
              legend=false, color=:blue)
    
    p2 = plot(results.date, results.rt_median,
              ribbon=(results.rt_median - results.rt_lower, 
                     results.rt_upper - results.rt_median),
              fillalpha=0.3, color=:red,
              title="Time-varying Reproduction Number (Rt)",
              ylabel="Rt", xlabel="Date",
              label="Rt (95% CI)", linewidth=2)
    
    # Add horizontal line at Rt = 1
    hline!(p2, [1.0], linestyle=:dash, color=:black, 
           label="Rt = 1", linewidth=1)
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main execution function
function main()
    println("Loading data...")
    
    # Load the data
    cases_data = load_data("cases.csv")
    
    println("Data loaded: $(nrow(cases_data)) days")
    println("Date range: $(minimum(cases_data.date)) to $(maximum(cases_data.date))")
    println("Total cases: $(sum(cases_data.cases))")
    
    # Estimate Rt
    chain, dates, cases = estimate_rt(cases_data, n_samples=1000, n_chains=2)
    
    println("MCMC completed. Extracting results...")
    
    # Extract results
    results = extract_rt_estimates(chain, dates)
    
    # Display summary
    println("\nRt Estimates Summary:")
    println("Mean Rt: $(round(mean(results.rt_mean), digits=2))")
    println("Latest Rt: $(round(results.rt_median[end], digits=2)) ",
            "(95% CI: $(round(results.rt_lower[end], digits=2)) - ",
            "$(round(results.rt_upper[end], digits=2)))")
    
    # Create and display plot
    p = plot_rt_estimates(results, cases_data)
    display(p)
    
    # Save results
    CSV.write("rt_estimates.csv", results)
    savefig(p, "rt_plot.png")
    
    println("\nResults saved to 'rt_estimates.csv' and 'rt_plot.png'")
    
    return results, chain
end

# Example of creating sample data if needed
function create_sample_data()
    dates = Date(2020, 3, 1):Day(1):Date(2020, 6, 30)
    n_days = length(dates)
    
    # Simulate cases with changing Rt
    cases = Int[]
    current_cases = 10
    
    for i in 1:n_days
        # Simulate different phases
        if i < 30  # Initial growth
            rt = 2.5
        elseif i < 60  # Intervention effect
            rt = 1.5 - (i-30) * 0.02
        elseif i < 90  # Controlled phase
            rt = 0.8
        else  # Relaxation
            rt = 1.2
        end
        
        # Simple exponential growth with noise
        expected = current_cases * rt^(1/5.2)  # 5.2 day generation interval
        current_cases = max(1, round(Int, expected + randn() * sqrt(expected)))
        push!(cases, current_cases)
    end
    
    df = DataFrame(date=dates, cases=cases)
    CSV.write("cases.csv", df)
    return df
end

# Uncomment the following lines to create sample data and run the analysis
# create_sample_data()
# results, chain = main()

