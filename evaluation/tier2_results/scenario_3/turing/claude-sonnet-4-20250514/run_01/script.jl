
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/turing/claude-sonnet-4-20250514/run_01")

using Dates

using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots
using LinearAlgebra, Random
using MCMCChains

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare data
function load_data(filepath="observations.csv")
    df = CSV.read(filepath, DataFrame)
    
    # Convert to arrays
    cases = df.cases
    hospitalisations = df.hospitalisations  
    deaths = df.deaths
    dates = df.date
    
    T = length(cases)
    
    return cases, hospitalisations, deaths, dates, T
end

# Generate interval probability mass function
function generate_interval_pmf(mean_interval, sd_interval, max_interval=30)
    # Use gamma distribution parameterization
    shape = (mean_interval / sd_interval)^2
    scale = sd_interval^2 / mean_interval
    
    # Discretize gamma distribution
    pmf = zeros(max_interval)
    for i in 1:max_interval
        pmf[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure sum = 1
    pmf = pmf ./ sum(pmf)
    return pmf
end

# Define delay distributions for each stream
function get_delay_distributions()
    # Generation interval (infection to infection)
    gen_interval = generate_interval_pmf(6.5, 4.0, 20)
    
    # Delay from infection to observation for each stream
    case_delay = generate_interval_pmf(5.0, 3.0, 20)      # Cases: ~5 days
    hosp_delay = generate_interval_pmf(8.0, 4.0, 25)      # Hospitalisation: ~8 days  
    death_delay = generate_interval_pmf(18.0, 8.0, 40)    # Deaths: ~18 days
    
    return gen_interval, case_delay, hosp_delay, death_delay
end

@model function renewal_model(cases, hospitalisations, deaths, gen_interval, 
                             case_delay, hosp_delay, death_delay)
    
    T = length(cases)
    S_gen = length(gen_interval)
    S_case = length(case_delay)
    S_hosp = length(hosp_delay)
    S_death = length(death_delay)
    
    # Prior for initial infections (seed infections)
    seed_days = max(S_gen, S_case, S_hosp, S_death)
    I_seed ~ filldist(Exponential(100.0), seed_days)
    
    # Prior for log(Rt) - smooth time-varying reproduction number
    log_R0 ~ Normal(0.0, 0.5)  # Initial log(Rt)
    σ_R ~ Exponential(0.1)     # Smoothness parameter for Rt random walk
    
    # Random walk for log(Rt)
    log_R_raw ~ filldist(Normal(0.0, 1.0), T-1)
    
    # Construct smooth log(Rt) series
    log_R = Vector{eltype(log_R0)}(undef, T)
    log_R[1] = log_R0
    for t in 2:T
        log_R[t] = log_R[t-1] + σ_R * log_R_raw[t-1]
    end
    
    # Convert to Rt
    R = exp.(log_R)
    
    # Stream-specific ascertainment rates
    p_case ~ Beta(2, 5)      # Ascertainment rate for cases
    p_hosp ~ Beta(1, 10)     # Ascertainment rate for hospitalizations  
    p_death ~ Beta(1, 5)     # Ascertainment rate for deaths
    
    # Overdispersion parameters (inverse overdispersion)
    ϕ_case ~ Exponential(0.1)
    ϕ_hosp ~ Exponential(0.1)  
    ϕ_death ~ Exponential(0.1)
    
    # Generate infections using renewal equation
    I = Vector{eltype(log_R0)}(undef, T + seed_days)
    
    # Set seed infections
    for i in 1:seed_days
        I[i] = I_seed[i]
    end
    
    # Renewal equation for infections
    for t in 1:T
        t_idx = t + seed_days
        renewal_sum = zero(eltype(log_R0))
        
        for s in 1:min(S_gen, t_idx-1)
            renewal_sum += I[t_idx - s] * gen_interval[s]
        end
        
        I[t_idx] = R[t] * renewal_sum
    end
    
    # Extract infections corresponding to observation times
    I_obs = I[(seed_days+1):end]
    
    # Generate expected observations for each stream with delays
    λ_case = Vector{eltype(log_R0)}(undef, T)
    λ_hosp = Vector{eltype(log_R0)}(undef, T)
    λ_death = Vector{eltype(log_R0)}(undef, T)
    
    for t in 1:T
        # Cases
        case_sum = zero(eltype(log_R0))
        for s in 1:min(S_case, t + seed_days - 1)
            if t + seed_days - s >= 1
                case_sum += I[t + seed_days - s] * case_delay[s]
            end
        end
        λ_case[t] = p_case * case_sum
        
        # Hospitalizations
        hosp_sum = zero(eltype(log_R0))
        for s in 1:min(S_hosp, t + seed_days - 1)
            if t + seed_days - s >= 1
                hosp_sum += I[t + seed_days - s] * hosp_delay[s]
            end
        end
        λ_hosp[t] = p_hosp * hosp_sum
        
        # Deaths  
        death_sum = zero(eltype(log_R0))
        for s in 1:min(S_death, t + seed_days - 1)
            if t + seed_days - s >= 1
                death_sum += I[t + seed_days - s] * death_delay[s]
            end
        end
        λ_death[t] = p_death * death_sum
    end
    
    # Likelihood - negative binomial for overdispersion
    for t in 1:T
        # Add small constant to prevent numerical issues
        λ_case_safe = max(λ_case[t], 1e-6)
        λ_hosp_safe = max(λ_hosp[t], 1e-6)
        λ_death_safe = max(λ_death[t], 1e-6)
        
        # Negative binomial: NB(r, p) where r is shape, p is prob of success
        # Mean = r(1-p)/p, Var = r(1-p)/p^2
        # Reparameterize using mean μ and overdispersion ϕ
        cases[t] ~ NegativeBinomial2(λ_case_safe, ϕ_case)
        hospitalisations[t] ~ NegativeBinomial2(λ_hosp_safe, ϕ_hosp) 
        deaths[t] ~ NegativeBinomial2(λ_death_safe, ϕ_death)
    end
    
    return R, I_obs, λ_case, λ_hosp, λ_death
end

# Main analysis function
function analyze_reproduction_number(filepath="observations.csv")
    println("Loading data...")
    cases, hospitalisations, deaths, dates, T = load_data(filepath)
    
    println("Setting up delay distributions...")
    gen_interval, case_delay, hosp_delay, death_delay = get_delay_distributions()
    
    println("Fitting model...")
    model = renewal_model(cases, hospitalisations, deaths, 
                         gen_interval, case_delay, hosp_delay, death_delay)
    
    # Sample from posterior
    n_samples = 2000
    n_chains = 4
    
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains, 
                  progress=true)
    
    println("\nModel Summary:")
    display(summarystats(chain))
    
    # Extract Rt estimates
    R_samples = Array(group(chain, :R))
    R_mean = mean(R_samples, dims=1)[1, :]
    R_lower = [quantile(R_samples[:, t], 0.025) for t in 1:T]
    R_upper = [quantile(R_samples[:, t], 0.975) for t in 1:T]
    
    # Extract stream-specific parameters
    p_case_est = mean(chain[:p_case])
    p_hosp_est = mean(chain[:p_hosp])  
    p_death_est = mean(chain[:p_death])
    
    println("\nStream-specific ascertainment rates:")
    println("Cases: $(round(p_case_est[1], digits=3))")
    println("Hospitalizations: $(round(p_hosp_est[1], digits=3))")
    println("Deaths: $(round(p_death_est[1], digits=3))")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot Rt over time
    p1 = plot(1:T, R_mean, ribbon=(R_mean .- R_lower, R_upper .- R_mean),
              label="Rt estimate", xlabel="Time", ylabel="Rt", 
              title="Time-varying Reproduction Number",
              fillalpha=0.3, linewidth=2)
    hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1")
    
    # Plot observations and fitted values
    λ_case_samples = Array(group(chain, :λ_case))
    λ_hosp_samples = Array(group(chain, :λ_hosp))
    λ_death_samples = Array(group(chain, :λ_death))
    
    λ_case_mean = mean(λ_case_samples, dims=1)[1, :]
    λ_hosp_mean = mean(λ_hosp_samples, dims=1)[1, :]
    λ_death_mean = mean(λ_death_samples, dims=1)[1, :]
    
    p2 = plot(1:T, cases, label="Observed cases", alpha=0.7)
    plot!(1:T, λ_case_mean, label="Expected cases", linewidth=2)
    plot!(xlabel="Time", ylabel="Count", title="Cases: Observed vs Expected")
    
    p3 = plot(1:T, hospitalisations, label="Observed hospitalizations", alpha=0.7)
    plot!(1:T, λ_hosp_mean, label="Expected hospitalizations", linewidth=2)
    plot!(xlabel="Time", ylabel="Count", title="Hospitalizations: Observed vs Expected")
    
    p4 = plot(1:T, deaths, label="Observed deaths", alpha=0.7) 
    plot!(1:T, λ_death_mean, label="Expected deaths", linewidth=2)
    plot!(xlabel="Time", ylabel="Count", title="Deaths: Observed vs Expected")
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    display(final_plot)
    
    # Return results
    results = Dict(
        "Rt_mean" => R_mean,
        "Rt_lower" => R_lower, 
        "Rt_upper" => R_upper,
        "ascertainment_rates" => Dict(
            "cases" => p_case_est[1],
            "hospitalizations" => p_hosp_est[1], 
            "deaths" => p_death_est[1]
        ),
        "chain" => chain,
        "dates" => dates
    )
    
    return results
end

# Create synthetic data for testing if no file exists
function create_synthetic_data(filename="observations.csv")
    T = 100
    dates = [Date(2023, 1, 1) + Day(i-1) for i in 1:T]
    
    # Simulate true Rt that changes over time
    true_Rt = 1.5 * exp.(-0.02 * (1:T)) .+ 0.5 .+ 0.2 * sin.(0.1 * (1:T))
    
    # Generate synthetic infections
    gen_interval, case_delay, hosp_delay, death_delay = get_delay_distributions()
    
    I = zeros(T + 20)
    I[1:20] .= 50.0  # Seed infections
    
    for t in 1:T
        renewal_sum = sum(I[t+20-s] * gen_interval[s] for s in 1:min(20, t+19))
        I[t+20] = true_Rt[t] * renewal_sum + rand(Normal(0, 5))
        I[t+20] = max(I[t+20], 1.0)  # Ensure positive
    end
    
    # Generate observations with delays and noise
    cases = zeros(Int, T)
    hospitalisations = zeros(Int, T) 
    deaths = zeros(Int, T)
    
    for t in 1:T
        # Cases
        case_exp = 0.15 * sum(I[t+20-s+1] * case_delay[s] for s in 1:min(20, t+20))
        cases[t] = max(0, rand(Poisson(case_exp + 1e-6)))
        
        # Hospitalizations
        hosp_exp = 0.03 * sum(I[t+20-s+1] * hosp_delay[s] for s in 1:min(25, t+20))
        hospitalisations[t] = max(0, rand(Poisson(hosp_exp + 1e-6)))
        
        # Deaths
        death_exp = 0.008 * sum(I[t+20-s+1] * death_delay[s] for s in 1:min(40, t+20))
        deaths[t] = max(0, rand(Poisson(death_exp + 1e-6)))
    end
    
    # Save synthetic data
    df = DataFrame(
        date = dates,
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    CSV.write(filename, df)
    println("Synthetic data saved to $filename")
    
    return df
end

# Main execution
function main()
    # Create synthetic data if file doesn't exist
    if !isfile("observations.csv")
        println("Creating synthetic data...")
        create_synthetic_data("observations.csv")
    end
    
    # Run analysis
    results = analyze_reproduction_number("observations.csv")
    
    println("\nAnalysis complete!")
    println("Final Rt estimate: $(round(results["Rt_mean"][end], digits=2))")
    
    return results
end

# Run the analysis
results = main()

