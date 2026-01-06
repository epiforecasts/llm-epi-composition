
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/turing/claude-sonnet-4-20250514/run_02")

using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots
using LinearAlgebra, Random, MCMCChains

# Set random seed for reproducibility
Random.seed!(42)

# Load and prepare the data
function load_data(filepath="cases.csv")
    df = CSV.read(filepath, DataFrame)
    # Ensure dates are properly parsed
    df.date = Date.(df.date)
    # Sort by date to ensure proper ordering
    sort!(df, :date)
    return df
end

# Define generation interval distribution
function get_generation_interval(max_days=20)
    # Using a discretized gamma distribution for generation interval
    # Mean ~5.2 days, std ~1.72 days (typical COVID-19 values)
    shape = 9.0
    scale = 0.58
    
    # Discretize the continuous distribution
    g = zeros(max_days)
    for i in 1:max_days
        g[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure sum = 1
    g = g ./ sum(g)
    return g
end

# Define incubation and reporting delay distribution
function get_reporting_delay(max_days=15)
    # Combined incubation + reporting delay
    # Approximated as discretized gamma with mean ~7 days
    shape = 4.0
    scale = 1.75
    
    d = zeros(max_days)
    for i in 1:max_days
        d[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize
    d = d ./ sum(d)
    return d
end

# Turing model for Rt estimation
@model function renewal_model(cases, g, d, n_days)
    # Priors
    # Log Rt follows a random walk
    log_R0 ~ Normal(0.0, 1.0)  # Initial log(Rt)
    σ_R ~ truncated(Normal(0, 0.2), 0, Inf)  # Innovation variance for Rt
    
    # Overdispersion parameter for negative binomial
    ϕ ~ truncated(Normal(0, 10), 0.1, Inf)
    
    # Initial infections (seeding period)
    n_seed = length(g)
    I_seed ~ MvNormal(zeros(n_seed), I(n_seed) * 10.0)
    
    # Initialize arrays
    log_Rt = Vector{Real}(undef, n_days)
    I_t = Vector{Real}(undef, n_days)
    λ_t = Vector{Real}(undef, n_days)
    
    # Set initial log_Rt
    log_Rt[1] = log_R0
    
    # Random walk for log(Rt)
    for t in 2:n_days
        log_Rt[t] ~ Normal(log_Rt[t-1], σ_R)
    end
    
    # Convert to Rt
    Rt = exp.(log_Rt)
    
    # Calculate infections using renewal equation
    for t in 1:n_days
        if t <= n_seed
            # Use seeded infections for initial period
            I_t[t] = exp(I_seed[t])  # Ensure positive
        else
            # Renewal equation: I_t = Rt * sum(I_{t-s} * g_s)
            infectivity = 0.0
            for s in 1:min(t-1, length(g))
                infectivity += I_t[t-s] * g[s]
            end
            I_t[t] = Rt[t] * infectivity
        end
    end
    
    # Calculate expected reported cases (accounting for reporting delay)
    for t in 1:n_days
        λ_t[t] = 0.0
        for s in 1:min(t, length(d))
            if t-s+1 >= 1
                λ_t[t] += I_t[t-s+1] * d[s]
            end
        end
        λ_t[t] = max(λ_t[t], 1e-10)  # Ensure positive
    end
    
    # Likelihood: observed cases
    for t in 1:n_days
        # Use negative binomial to account for overdispersion
        p = ϕ / (λ_t[t] + ϕ)
        cases[t] ~ NegativeBinomial(ϕ, p)
    end
    
    return (Rt=Rt, I_t=I_t, λ_t=λ_t)
end

# Function to estimate Rt
function estimate_rt(cases_data; n_iter=2000, n_chains=4)
    println("Preparing data...")
    
    # Get case counts
    cases = cases_data.cases
    n_days = length(cases)
    
    # Get generation interval and reporting delay
    g = get_generation_interval()
    d = get_reporting_delay()
    
    println("Setting up model...")
    
    # Create model
    model = renewal_model(cases, g, d, n_days)
    
    println("Running MCMC sampling...")
    
    # Sample from posterior
    chain = sample(model, NUTS(0.8), MCMCThreads(), n_iter, n_chains, 
                  progress=true, drop_warmup=true)
    
    return chain, g, d
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(chain, dates)
    # Get Rt parameter names
    rt_params = [Symbol("Rt[$i]") for i in 1:length(dates)]
    
    # Extract Rt estimates
    rt_samples = chain[rt_params]
    
    # Calculate summary statistics
    rt_mean = mean(rt_samples).nt.mean
    rt_lower = [quantile(rt_samples[Symbol("Rt[$i]")], 0.025) for i in 1:length(dates)]
    rt_upper = [quantile(rt_samples[Symbol("Rt[$i]")], 0.975) for i in 1:length(dates)]
    rt_median = [quantile(rt_samples[Symbol("Rt[$i]")], 0.5) for i in 1:length(dates)]
    
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

# Function to create plots
function plot_results(cases_data, rt_estimates, chain)
    dates = cases_data.date
    
    # Plot 1: Cases over time
    p1 = plot(dates, cases_data.cases, 
             title="Observed Cases", 
             xlabel="Date", ylabel="Daily Cases",
             linewidth=2, color=:blue, legend=false)
    
    # Plot 2: Rt estimates over time
    p2 = plot(dates, rt_estimates.rt_median,
             ribbon=(rt_estimates.rt_median - rt_estimates.rt_lower,
                    rt_estimates.rt_upper - rt_estimates.rt_median),
             title="Rt Estimates Over Time",
             xlabel="Date", ylabel="Rt",
             linewidth=2, color=:red, fillalpha=0.3,
             label="Rt (95% CI)")
    
    # Add horizontal line at Rt = 1
    hline!([1.0], linestyle=:dash, color=:black, label="Rt = 1", linewidth=2)
    
    # Plot 3: MCMC diagnostics
    p3 = plot(chain[[:σ_R, :ϕ]], title="MCMC Traces for Key Parameters")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    
    return combined_plot
end

# Main execution function
function main()
    println("Loading data...")
    
    # Create sample data if file doesn't exist
    if !isfile("cases.csv")
        println("Creating sample data...")
        dates = Date(2023, 1, 1):Day(1):Date(2023, 3, 31)
        # Simulate some realistic case data
        n_days = length(dates)
        true_rt = 1.2 * exp.(-0.02 * (1:n_days)) .+ 0.1 * sin.(2π * (1:n_days) / 14)
        cases = max.(1, round.(Int, 100 * true_rt .* (1 .+ 0.3 * randn(n_days))))
        
        sample_data = DataFrame(date=dates, cases=cases)
        CSV.write("cases.csv", sample_data)
        println("Sample data created and saved to cases.csv")
    end
    
    # Load the data
    cases_data = load_data("cases.csv")
    println("Data loaded: $(nrow(cases_data)) days of data")
    
    # Estimate Rt
    chain, g, d = estimate_rt(cases_data, n_iter=1000, n_chains=2)
    
    # Extract Rt estimates
    rt_estimates = extract_rt_estimates(chain, cases_data.date)
    
    # Print summary
    println("\nRt Estimation Results:")
    println("======================")
    for i in [1, nrow(rt_estimates)÷2, nrow(rt_estimates)]
        row = rt_estimates[i, :]
        println("$(row.date): Rt = $(round(row.rt_median, digits=2)) " *
               "(95% CI: $(round(row.rt_lower, digits=2))-$(round(row.rt_upper, digits=2)))")
    end
    
    # Create and display plots
    p = plot_results(cases_data, rt_estimates, chain)
    display(p)
    
    # Save results
    CSV.write("rt_estimates.csv", rt_estimates)
    println("\nResults saved to rt_estimates.csv")
    
    return rt_estimates, chain
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    rt_estimates, chain = main()
end

