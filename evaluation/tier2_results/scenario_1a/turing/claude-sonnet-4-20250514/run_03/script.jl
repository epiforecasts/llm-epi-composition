
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/turing/claude-sonnet-4-20250514/run_03")

using Dates

using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots
using LinearAlgebra, Random, MCMCChains

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare the data
function load_data(filename="cases.csv")
    df = CSV.read(filename, DataFrame)
    # Ensure dates are properly parsed
    df.date = Date.(df.date)
    # Sort by date to ensure chronological order
    sort!(df, :date)
    return df
end

# Define the serial interval distribution (time from infection to transmission)
# Using a discretized gamma distribution with mean ~5 days, std ~3 days
function serial_interval_pmf(max_days=20)
    # Gamma distribution parameters for COVID-19 serial interval
    shape, rate = 2.8, 0.56  # Mean ≈ 5 days, std ≈ 3 days
    gamma_dist = Gamma(shape, 1/rate)
    
    # Discretize the distribution
    pmf = zeros(max_days)
    for i in 1:max_days
        pmf[i] = cdf(gamma_dist, i) - cdf(gamma_dist, i-1)
    end
    
    # Normalize to ensure sum = 1
    pmf = pmf ./ sum(pmf)
    return pmf
end

# Calculate the infectiousness profile
function calculate_infectiousness(cases, serial_pmf)
    n = length(cases)
    infectiousness = zeros(n)
    
    for t in 1:n
        for s in 1:min(t-1, length(serial_pmf))
            if t-s > 0
                infectiousness[t] += cases[t-s] * serial_pmf[s]
            end
        end
    end
    
    return infectiousness
end

# Turing model for estimating time-varying Rt
@model function rt_model(cases, infectiousness, n_days)
    # Priors
    R0 ~ LogNormal(log(2.0), 0.5)  # Initial reproduction number
    σ_R ~ truncated(Normal(0, 0.1), 0, Inf)  # Random walk standard deviation
    
    # Time-varying reproduction number as a random walk
    log_Rt = Vector{Real}(undef, n_days)
    log_Rt[1] = log(R0)
    
    for t in 2:n_days
        log_Rt[t] ~ Normal(log_Rt[t-1], σ_R)
    end
    
    # Convert to Rt scale
    Rt = exp.(log_Rt)
    
    # Likelihood: cases follow Poisson distribution
    for t in 1:n_days
        if infectiousness[t] > 0
            λ = Rt[t] * infectiousness[t]
            cases[t] ~ Poisson(max(λ, 1e-8))  # Avoid λ = 0
        else
            # For early days with no infectiousness, use a simple prior
            cases[t] ~ Poisson(max(cases[t], 1))
        end
    end
    
    return Rt
end

# Function to estimate Rt
function estimate_rt(cases_data; 
                    n_samples=2000, 
                    n_chains=4, 
                    max_serial_days=20)
    
    println("Preparing data...")
    cases = cases_data.cases
    n_days = length(cases)
    
    # Calculate serial interval and infectiousness
    serial_pmf = serial_interval_pmf(max_serial_days)
    infectiousness = calculate_infectiousness(cases, serial_pmf)
    
    println("Setting up model...")
    model = rt_model(cases, infectiousness, n_days)
    
    println("Running MCMC sampling...")
    # Use NUTS sampler
    sampler = NUTS(0.65)
    
    # Sample from the posterior
    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
    
    println("Extracting results...")
    
    # Extract Rt estimates
    rt_samples = Array(group(chain, :Rt))
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_lower = vec([quantile(rt_samples[:, i], 0.025) for i in 1:n_days])
    rt_upper = vec([quantile(rt_samples[:, i], 0.975) for i in 1:n_days])
    rt_median = vec([quantile(rt_samples[:, i], 0.5) for i in 1:n_days])
    
    # Create results DataFrame
    results = DataFrame(
        date = cases_data.date,
        cases = cases,
        infectiousness = infectiousness,
        Rt_mean = rt_mean,
        Rt_median = rt_median,
        Rt_lower = rt_lower,
        Rt_upper = rt_upper
    )
    
    return results, chain, serial_pmf
end

# Plotting function
function plot_results(results, title_prefix="COVID-19")
    # Create subplot layout
    p1 = plot(results.date, results.cases, 
             label="Reported Cases", 
             title="$title_prefix Daily Cases",
             xlabel="Date", 
             ylabel="Cases",
             linewidth=2,
             color=:blue)
    
    p2 = plot(results.date, results.Rt_median,
             ribbon=(results.Rt_median .- results.Rt_lower, 
                    results.Rt_upper .- results.Rt_median),
             label="Rt (95% CI)",
             title="$title_prefix Reproduction Number (Rt)",
             xlabel="Date",
             ylabel="Rt",
             linewidth=2,
             fillalpha=0.3,
             color=:red)
    
    # Add horizontal line at Rt = 1
    hline!(p2, [1.0], linestyle=:dash, color=:black, 
           label="Rt = 1", linewidth=1)
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Summary statistics function
function summarize_rt(results)
    println("\n=== Rt Estimation Summary ===")
    println("Date Range: $(first(results.date)) to $(last(results.date))")
    println("Total Days: $(nrow(results))")
    println("Total Cases: $(sum(results.cases))")
    
    println("\nRt Statistics:")
    println("Mean Rt: $(round(mean(results.Rt_mean), digits=3))")
    println("Median Rt: $(round(median(results.Rt_median), digits=3))")
    println("Min Rt: $(round(minimum(results.Rt_median), digits=3))")
    println("Max Rt: $(round(maximum(results.Rt_median), digits=3))")
    
    # Days above/below Rt = 1
    days_above_1 = sum(results.Rt_median .> 1.0)
    days_below_1 = sum(results.Rt_median .< 1.0)
    
    println("\nDays with Rt > 1: $days_above_1 ($(round(100*days_above_1/nrow(results), digits=1))%)")
    println("Days with Rt < 1: $days_below_1 ($(round(100*days_below_1/nrow(results), digits=1))%)")
end

# Main execution function
function main()
    try
        println("Loading case data...")
        cases_data = load_data("cases.csv")
        
        println("Estimating Rt...")
        results, chain, serial_pmf = estimate_rt(cases_data)
        
        println("Creating visualizations...")
        p = plot_results(results)
        display(p)
        
        # Save results
        CSV.write("rt_estimates.csv", results)
        println("Results saved to rt_estimates.csv")
        
        # Print summary
        summarize_rt(results)
        
        # Display convergence diagnostics
        println("\n=== MCMC Diagnostics ===")
        println("R-hat statistics (should be close to 1.0):")
        rhat_vals = rhat(chain)
        println("Mean R-hat: $(round(mean(values(rhat_vals)), digits=4))")
        
        return results, chain, serial_pmf
        
    catch e
        if isa(e, SystemError) && occursin("cases.csv", string(e))
            println("Error: cases.csv file not found!")
            println("Creating example data file...")
            create_example_data()
            println("Example data created. Please run main() again.")
        else
            rethrow(e)
        end
    end
end

# Function to create example data if file doesn't exist
function create_example_data()
    # Generate synthetic COVID-19-like case data
    dates = Date(2023, 1, 1):Day(1):Date(2023, 3, 31)
    n_days = length(dates)
    
    # Simulate cases with changing Rt
    cases = zeros(Int, n_days)
    cases[1:7] .= [5, 8, 12, 15, 22, 18, 25]  # Initial cases
    
    # Simulate epidemic curve
    for i in 8:n_days
        # Add some realistic variation
        trend = 20 * exp(0.05 * (i - 40)) * exp(-0.001 * (i - 40)^2)
        noise = randn() * 5
        cases[i] = max(1, round(Int, trend + noise + 0.7 * cases[i-1]))
    end
    
    example_data = DataFrame(date=dates, cases=cases)
    CSV.write("cases.csv", example_data)
    
    println("Example data written to cases.csv")
    println("First few rows:")
    println(first(example_data, 10))
end

# Run the analysis
println("COVID-19 Rt Estimation using Turing.jl")
println("=" ^ 40)

results, chain, serial_pmf = main()

