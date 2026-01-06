using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots, Random, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare the data
function load_data(filename="cases.csv")
    # If the file doesn't exist, create sample data for demonstration
    if !isfile(filename)
        println("Creating sample data for demonstration...")
        dates = Date(2020, 3, 1):Day(1):Date(2020, 6, 30)
        
        # Simulate realistic COVID-19 case trajectory
        n_days = length(dates)
        true_rt = vcat(
            fill(2.5, 20),  # Initial exponential growth
            2.5 .- 0.05 * (1:30),  # Gradual decline
            fill(1.0, 30),  # Around critical threshold
            1.0 .+ 0.02 * (1:20),  # Slight increase
            fill(1.2, n_days - 100)  # Sustained transmission
        )
        
        # Simple simulation for realistic case counts
        cases = zeros(Int, n_days)
        cases[1:7] .= [5, 8, 12, 18, 25, 35, 45]  # Seed cases
        
        for t in 8:n_days
            lambda = sum(cases[max(1, t-14):t-1] .* 
                        exp.(-((1:min(14, t-1)) .- 5.5).^2 / (2 * 2.5^2)))
            expected_cases = max(1.0, true_rt[t] * lambda * 0.1)
            cases[t] = rand(Poisson(expected_cases))
        end
        
        df = DataFrame(date=dates, cases=cases)
        CSV.write(filename, df)
    end
    
    df = CSV.read(filename, DataFrame)
    return df
end

# Define generation interval distribution
function generation_interval_pmf(max_gen=20)
    # Discretized gamma distribution for generation interval
    # Mean ≈ 5.5 days, SD ≈ 2.5 days (typical for COVID-19)
    shape, scale = 4.8, 1.15
    
    pmf = zeros(max_gen)
    for i in 1:max_gen
        pmf[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure sum = 1
    pmf = pmf / sum(pmf)
    return pmf
end

# Define reporting delay distribution
function reporting_delay_pmf(max_delay=15)
    # Log-normal distribution for reporting delay
    # Mean delay ≈ 7 days
    μ, σ = 1.8, 0.5
    
    pmf = zeros(max_delay)
    for i in 1:max_delay
        pmf[i] = cdf(LogNormal(μ, σ), i) - cdf(LogNormal(μ, σ), i-1)
    end
    
    pmf = pmf / sum(pmf)
    return pmf
end

# Turing model for Rt estimation
@model function rt_model(cases, gen_pmf, delay_pmf)
    n_days = length(cases)
    max_gen = length(gen_pmf)
    max_delay = length(delay_pmf)
    
    # Priors for initial infections (seeding period)
    seed_days = 7
    I_seed ~ filldist(Exponential(10.0), seed_days)
    
    # Prior for Rt - using a random walk on log scale for smoothness
    log_rt_init ~ Normal(0.5, 0.5)  # Initial Rt around exp(0.5) ≈ 1.65
    σ_rt ~ Exponential(0.1)  # Innovation standard deviation
    
    log_rt_innovations ~ filldist(Normal(0, 1), n_days - seed_days - 1)
    
    # Construct log_rt time series
    log_rt = Vector{eltype(log_rt_init)}(undef, n_days)
    log_rt[seed_days + 1] = log_rt_init
    
    for t in (seed_days + 2):n_days
        log_rt[t] = log_rt[t-1] + σ_rt * log_rt_innovations[t - seed_days - 1]
    end
    
    # Convert to Rt
    rt = exp.(log_rt[(seed_days + 1):end])
    
    # Initialize infections
    infections = Vector{eltype(I_seed[1])}(undef, n_days)
    infections[1:seed_days] = I_seed
    
    # Renewal equation: compute infections for t > seed_days
    for t in (seed_days + 1):n_days
        lambda = 0.0
        for s in 1:min(max_gen, t-1)
            if t - s >= 1
                lambda += infections[t - s] * gen_pmf[s]
            end
        end
        infections[t] = rt[t - seed_days] * lambda
    end
    
    # Convolve infections with reporting delay to get expected cases
    expected_cases = Vector{eltype(infections[1])}(undef, n_days)
    
    for t in 1:n_days
        expected_cases[t] = 0.0
        for d in 1:min(max_delay, t)
            if t - d + 1 >= 1
                expected_cases[t] += infections[t - d + 1] * delay_pmf[d]
            end
        end
        expected_cases[t] = max(expected_cases[t], 1e-6)  # Avoid zero
    end
    
    # Observation model - Negative Binomial for overdispersion
    φ ~ Exponential(0.1)  # Overdispersion parameter
    
    for t in 1:n_days
        if cases[t] >= 0  # Only observe non-missing cases
            # Negative binomial parameterized by mean and overdispersion
            p = φ / (φ + expected_cases[t])
            r = φ
            cases[t] ~ NegativeBinomial(r, 1 - p)
        end
    end
end

# Function to run the analysis
function estimate_rt(data_file="cases.csv")
    println("Loading data...")
    df = load_data(data_file)
    cases = df.cases
    dates = df.date
    n_days = length(cases)
    
    println("Setting up model...")
    gen_pmf = generation_interval_pmf()
    delay_pmf = reporting_delay_pmf()
    
    println("Generation interval (first 10 days): ", round.(gen_pmf[1:10], digits=3))
    println("Reporting delay (first 10 days): ", round.(delay_pmf[1:10], digits=3))
    
    # Create and sample from the model
    println("Fitting model...")
    model = rt_model(cases, gen_pmf, delay_pmf)
    
    # Use NUTS sampler
    n_samples = 1000
    n_chains = 2
    
    println("Running MCMC sampling...")
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    println("Processing results...")
    
    # Extract Rt estimates
    seed_days = 7
    rt_samples = []
    
    for i in 1:(n_days - seed_days)
        rt_col = "rt[$i]"
        if rt_col in names(chain)
            push!(rt_samples, chain[rt_col])
        end
    end
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_q025 = [quantile(vec(Array(samples)), 0.025) for samples in rt_samples]
    rt_q975 = [quantile(vec(Array(samples)), 0.975) for samples in rt_samples]
    
    # Create results dataframe
    rt_dates = dates[(seed_days + 1):end]
    results_df = DataFrame(
        date = rt_dates,
        rt_mean = rt_mean,
        rt_q025 = rt_q025,
        rt_q975 = rt_q975
    )
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("=" ^ 50)
    println("Data period: $(dates[1]) to $(dates[end])")
    println("Total days: $n_days")
    println("Rt estimation period: $(rt_dates[1]) to $(rt_dates[end])")
    println("Mean Rt over period: $(round(mean(rt_mean), digits=2))")
    println("Days with Rt > 1: $(sum(rt_mean .> 1)) / $(length(rt_mean))")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot 1: Case counts over time
    p1 = plot(dates, cases, 
             title="Daily Case Counts", 
             xlabel="Date", 
             ylabel="Cases",
             linewidth=2,
             color=:blue,
             legend=false)
    
    # Plot 2: Rt over time
    p2 = plot(rt_dates, rt_mean,
             ribbon=(rt_mean .- rt_q025, rt_q975 .- rt_mean),
             title="Estimated Rt Over Time",
             xlabel="Date",
             ylabel="Rt",
             linewidth=2,
             color=:red,
             fillalpha=0.3,
             label="Rt (95% CI)")
    
    hline!([1.0], linestyle=:dash, color=:black, linewidth=2, label="Rt = 1")
    
    # Combined plot
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    display(combined_plot)
    
    # Save results
    CSV.write("rt_estimates.csv", results_df)
    savefig(combined_plot, "rt_estimation_results.png")
    
    println("\nResults saved to:")
    println("- rt_estimates.csv (Rt estimates)")
    println("- rt_estimation_results.png (plots)")
    
    return results_df, chain, combined_plot
end

# Run the analysis
println("Starting Rt estimation using renewal equation...")
results_df, chain, plot_result = estimate_rt()

# Display final results
println("\nFinal Rt estimates (last 10 days):")
println(last(results_df, 10))

# Model diagnostics
println("\nModel diagnostics:")
println("MCMC Summary:")
display(chain)

