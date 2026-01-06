
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/turing/claude-sonnet-4-20250514/run_02")

using Turing, Distributions, DataFrames, CSV, Plots, StatsPlots
using LinearAlgebra, Random, Dates
using MCMCChains

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare data
function load_data(filepath="observations.csv")
    df = CSV.read(filepath, DataFrame)
    df.date = Date.(df.date)
    return df
end

# Generation interval (gamma distribution discretized)
function get_generation_interval(max_days=21)
    # Gamma distribution parameters for generation interval
    # Mean ~5 days, std ~2.5 days
    shape, scale = 4.0, 1.25
    
    # Discretize and normalize
    gen_interval = pdf.(Gamma(shape, scale), 1:max_days)
    gen_interval = gen_interval ./ sum(gen_interval)
    
    return gen_interval
end

# Stream-specific delay distributions
function get_delay_distributions(max_delay=28)
    delays = Dict()
    
    # Cases: Short delay (mean ~3 days)
    delays["cases"] = pdf.(Gamma(3.0, 1.0), 1:max_delay)
    delays["cases"] = delays["cases"] ./ sum(delays["cases"])
    
    # Hospitalizations: Medium delay (mean ~7 days) 
    delays["hospitalisations"] = pdf.(Gamma(7.0, 1.0), 1:max_delay)
    delays["hospitalisations"] = delays["hospitalisations"] ./ sum(delays["hospitalisations"])
    
    # Deaths: Long delay (mean ~14 days)
    delays["deaths"] = pdf.(Gamma(14.0, 1.0), 1:max_delay)
    delays["deaths"] = delays["deaths"] ./ sum(delays["deaths"])
    
    return delays
end

# Convolve infections with delay distribution
function convolve_delay(infections, delay_dist)
    n = length(infections)
    m = length(delay_dist)
    observed = zeros(n)
    
    for t in 1:n
        for d in 1:min(m, t)
            observed[t] += infections[t-d+1] * delay_dist[d]
        end
    end
    
    return observed
end

# Apply renewal equation
function apply_renewal(Rt, gen_interval, initial_infections)
    n = length(Rt)
    g = length(gen_interval)
    infections = zeros(n)
    
    # Set initial infections
    infections[1:min(g, n)] = initial_infections[1:min(g, n)]
    
    # Apply renewal equation
    for t in (g+1):n
        infections[t] = Rt[t] * sum(infections[(t-g):(t-1)] .* reverse(gen_interval))
    end
    
    return infections
end

@model function multi_stream_renewal_model(cases, hospitalisations, deaths, 
                                          gen_interval, delay_dists)
    n = length(cases)
    g = length(gen_interval)
    
    # Priors for initial infections (seeding period)
    initial_infections ~ MvNormal(zeros(g), 100.0 * I(g))
    
    # Prior for log(R0)
    log_R0 ~ Normal(log(1.2), 0.2)
    
    # Random walk on log(Rt) with smoothness constraint
    σ_rw ~ truncated(Normal(0, 0.1), 0, Inf)  # Standard deviation of random walk
    log_Rt_raw ~ MvNormal(zeros(n-1), I(n-1))
    
    # Construct log(Rt) as cumulative sum (random walk)
    log_Rt = Vector{eltype(log_Rt_raw)}(undef, n)
    log_Rt[1] = log_R0
    for t in 2:n
        log_Rt[t] = log_Rt[t-1] + σ_rw * log_Rt_raw[t-1]
    end
    
    Rt = exp.(log_Rt)
    
    # Apply renewal equation to get infections
    infections = apply_renewal(Rt, gen_interval, max.(0.1, initial_infections))
    
    # Stream-specific parameters
    # Ascertainment rates (proportion of infections observed in each stream)
    asc_cases ~ Beta(2, 8)        # Prior: mean ~0.2
    asc_hosp ~ Beta(1, 9)         # Prior: mean ~0.1  
    asc_deaths ~ Beta(1, 99)      # Prior: mean ~0.01
    
    # Overdispersion parameters (inverse overdispersion, higher = less overdispersed)
    ϕ_cases ~ Gamma(5, 1)
    ϕ_hosp ~ Gamma(5, 1)
    ϕ_deaths ~ Gamma(5, 1)
    
    # Convolve infections with delays to get expected observations
    expected_cases_raw = convolve_delay(infections, delay_dists["cases"])
    expected_hosp_raw = convolve_delay(infections, delay_dists["hospitalisations"])
    expected_deaths_raw = convolve_delay(infections, delay_dists["deaths"])
    
    # Apply ascertainment
    expected_cases = asc_cases .* expected_cases_raw
    expected_hosp = asc_hosp .* expected_hosp_raw
    expected_deaths = asc_deaths .* expected_deaths_raw
    
    # Likelihood with overdispersion (negative binomial)
    for t in 1:n
        # Ensure positive expected values
        μ_cases = max(0.1, expected_cases[t])
        μ_hosp = max(0.1, expected_hosp[t])
        μ_deaths = max(0.1, expected_deaths[t])
        
        # Negative binomial parameterization: NegativeBinomial(r, p)
        # where mean = r(1-p)/p and var = r(1-p)/p²
        # We use: r = ϕ, p = ϕ/(ϕ + μ)
        cases[t] ~ NegativeBinomial(ϕ_cases, ϕ_cases/(ϕ_cases + μ_cases))
        hospitalisations[t] ~ NegativeBinomial(ϕ_hosp, ϕ_hosp/(ϕ_hosp + μ_hosp))
        deaths[t] ~ NegativeBinomial(ϕ_deaths, ϕ_deaths/(ϕ_deaths + μ_deaths))
    end
end

# Main analysis function
function estimate_rt_multi_stream(data_file="observations.csv")
    println("Loading data...")
    df = load_data(data_file)
    
    # Extract observations
    cases = df.cases
    hospitalisations = df.hospitalisations  
    deaths = df.deaths
    dates = df.date
    
    println("Data loaded: $(length(cases)) time points from $(dates[1]) to $(dates[end])")
    
    # Get generation interval and delay distributions
    gen_interval = get_generation_interval()
    delay_dists = get_delay_distributions()
    
    println("Setting up model...")
    
    # Create model
    model = multi_stream_renewal_model(cases, hospitalisations, deaths, 
                                     gen_interval, delay_dists)
    
    println("Starting MCMC sampling...")
    
    # Sample from posterior
    n_samples = 1000
    n_chains = 4
    
    chain = sample(model, NUTS(0.8), MCMCThreads(), n_samples, n_chains, 
                  progress=true, drop_warmup=true)
    
    println("MCMC sampling completed!")
    
    # Extract results
    return extract_results(chain, dates, cases, hospitalisations, deaths)
end

function extract_results(chain, dates, cases, hospitalisations, deaths)
    n = length(dates)
    
    # Extract Rt estimates
    Rt_samples = Array(group(chain, :Rt))
    Rt_mean = [mean(Rt_samples[:, t]) for t in 1:n]
    Rt_lower = [quantile(Rt_samples[:, t], 0.025) for t in 1:n]
    Rt_upper = [quantile(Rt_samples[:, t], 0.975) for t in 1:n]
    
    # Extract ascertainment rates
    asc_cases_samples = Array(group(chain, :asc_cases))
    asc_hosp_samples = Array(group(chain, :asc_hosp))
    asc_deaths_samples = Array(group(chain, :asc_deaths))
    
    # Create results summary
    results = DataFrame(
        date = dates,
        Rt_mean = Rt_mean,
        Rt_lower = Rt_lower,
        Rt_upper = Rt_upper,
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    # Print summary statistics
    println("\n=== RESULTS SUMMARY ===")
    println("Rt estimates:")
    println("  Mean Rt: $(round(mean(Rt_mean), digits=3))")
    println("  Min Rt:  $(round(minimum(Rt_mean), digits=3))")
    println("  Max Rt:  $(round(maximum(Rt_mean), digits=3))")
    
    println("\nStream-specific ascertainment rates:")
    println("  Cases:           $(round(mean(asc_cases_samples), digits=4)) (95% CI: $(round(quantile(asc_cases_samples, 0.025), digits=4))-$(round(quantile(asc_cases_samples, 0.975), digits=4)))")
    println("  Hospitalizations: $(round(mean(asc_hosp_samples), digits=4)) (95% CI: $(round(quantile(asc_hosp_samples, 0.025), digits=4))-$(round(quantile(asc_hosp_samples, 0.975), digits=4)))")
    println("  Deaths:          $(round(mean(asc_deaths_samples), digits=4)) (95% CI: $(round(quantile(asc_deaths_samples, 0.025), digits=4))-$(round(quantile(asc_deaths_samples, 0.975), digits=4)))")
    
    # Create plots
    create_plots(results, chain)
    
    return results, chain
end

function create_plots(results, chain)
    # Plot 1: Rt over time
    p1 = plot(results.date, results.Rt_mean, 
             ribbon=(results.Rt_mean .- results.Rt_lower, 
                    results.Rt_upper .- results.Rt_mean),
             label="Rt estimate", color=:blue, alpha=0.7,
             title="Reproduction Number (Rt) Over Time",
             xlabel="Date", ylabel="Rt",
             legend=:topright)
    hline!([1.0], color=:red, linestyle=:dash, label="Rt = 1", alpha=0.7)
    
    # Plot 2: Data streams
    p2 = plot(results.date, results.cases, 
             label="Cases", color=:orange, alpha=0.7,
             title="Observed Data Streams",
             xlabel="Date", ylabel="Count")
    plot!(results.date, results.hospitalisations, 
          label="Hospitalizations", color=:red, alpha=0.7)
    plot!(results.date, results.deaths, 
          label="Deaths", color=:black, alpha=0.7)
    
    # Plot 3: Posterior distributions of ascertainment rates
    asc_cases_samples = Array(group(chain, :asc_cases))
    asc_hosp_samples = Array(group(chain, :asc_hosp))
    asc_deaths_samples = Array(group(chain, :asc_deaths))
    
    p3 = density(asc_cases_samples[:], label="Cases", alpha=0.7, color=:orange,
                title="Posterior Ascertainment Rates",
                xlabel="Ascertainment Rate", ylabel="Density")
    density!(asc_hosp_samples[:], label="Hospitalizations", alpha=0.7, color=:red)
    density!(asc_deaths_samples[:], label="Deaths", alpha=0.7, color=:black)
    
    # Combine plots
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    display(final_plot)
    
    # Save plot
    savefig(final_plot, "rt_multi_stream_results.png")
    println("\nPlots saved as 'rt_multi_stream_results.png'")
    
    return final_plot
end

# Generate synthetic data for testing
function generate_synthetic_data(n_days=100)
    dates = Date(2023, 1, 1):Day(1):(Date(2023, 1, 1) + Day(n_days-1))
    
    # True Rt trajectory (starts high, decreases, then increases)
    true_Rt = 1.5 .* exp.(-0.02 .* (1:n_days)) .+ 0.5 .+ 0.3 .* sin.(0.1 .* (1:n_days))
    
    # Generate true infections using renewal equation
    gen_interval = get_generation_interval()
    initial_infections = fill(100.0, length(gen_interval))
    true_infections = apply_renewal(true_Rt, gen_interval, initial_infections)
    
    # Generate observations with delays and ascertainment
    delay_dists = get_delay_distributions()
    
    # True ascertainment rates
    true_asc_cases = 0.15
    true_asc_hosp = 0.08
    true_asc_deaths = 0.012
    
    # Expected observations
    exp_cases = true_asc_cases .* convolve_delay(true_infections, delay_dists["cases"])
    exp_hosp = true_asc_hosp .* convolve_delay(true_infections, delay_dists["hospitalisations"])
    exp_deaths = true_asc_deaths .* convolve_delay(true_infections, delay_dists["deaths"])
    
    # Add overdispersion (negative binomial observations)
    ϕ = 10.0  # Overdispersion parameter
    
    cases = [rand(NegativeBinomial(ϕ, ϕ/(ϕ + max(0.1, μ)))) for μ in exp_cases]
    hospitalisations = [rand(NegativeBinomial(ϕ, ϕ/(ϕ + max(0.1, μ)))) for μ in exp_hosp]
    deaths = [rand(NegativeBinomial(ϕ, ϕ/(ϕ + max(0.1, μ)))) for μ in exp_deaths]
    
    # Create DataFrame
    df = DataFrame(
        date = dates,
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    # Save synthetic data
    CSV.write("observations.csv", df)
    println("Synthetic data generated and saved to 'observations.csv'")
    
    return df, true_Rt
end

# Main execution
function main()
    println("=== Multi-Stream Rt Estimation ===\n")
    
    # Generate synthetic data if file doesn't exist
    if !isfile("observations.csv")
        println("Generating synthetic data...")
        synthetic_data, true_Rt = generate_synthetic_data()
    end
    
    # Run analysis
    results, chain = estimate_rt_multi_stream()
    
    # Print convergence diagnostics
    println("\n=== MCMC DIAGNOSTICS ===")
    println("ESS summary:")
    display(summarize(chain, ess))
    
    return results, chain
end

# Run the analysis
results, chain = main()

