
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/turing/claude-sonnet-4-20250514/run_03")

using Dates

using Turing, CSV, DataFrames, Distributions, Random, Plots, StatsPlots
using LinearAlgebra, Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Load and prepare the data
function load_data(filename="observations.csv")
    df = CSV.read(filename, DataFrame)
    
    # Convert date column if needed and sort by date
    sort!(df, :date)
    
    # Extract observation vectors
    cases = df.cases
    hospitalisations = df.hospitalisations  
    deaths = df.deaths
    
    return cases, hospitalisations, deaths, nrow(df)
end

# Define delay distributions (using gamma distributions)
function setup_delay_distributions()
    # Cases: shorter delay from infection (mean ~5 days)
    cases_delay = [pdf(Gamma(2, 2.5), x) for x in 1:20]
    cases_delay = cases_delay ./ sum(cases_delay)
    
    # Hospitalisations: medium delay (mean ~10 days)  
    hosp_delay = [pdf(Gamma(3, 3.33), x) for x in 1:25]
    hosp_delay = hosp_delay ./ sum(hosp_delay)
    
    # Deaths: longer delay (mean ~18 days)
    death_delay = [pdf(Gamma(4, 4.5), x) for x in 1:35]
    death_delay = death_delay ./ sum(death_delay)
    
    return cases_delay, hosp_delay, death_delay
end

# Define generation interval
function setup_generation_interval()
    # Generation interval with mean ~6 days
    gen_interval = [pdf(Gamma(2.5, 2.4), x) for x in 1:15]
    gen_interval = gen_interval ./ sum(gen_interval)
    return gen_interval
end

@model function joint_rt_model(cases, hospitalisations, deaths, T, 
                              cases_delay, hosp_delay, death_delay, gen_interval)
    
    # Dimensions
    max_delay = max(length(cases_delay), length(hosp_delay), length(death_delay))
    gen_len = length(gen_interval)
    seed_period = max(max_delay, gen_len, 7)  # Initial seeding period
    
    # Priors for ascertainment rates (proportion of infections observed)
    p_cases ~ Beta(2, 8)        # Cases have lower ascertainment
    p_hosp ~ Beta(1, 19)        # Hospitalisations much lower
    p_deaths ~ Beta(1, 99)      # Deaths lowest but most complete
    
    # Overdispersion parameters (smaller = more overdispersed)
    φ_cases ~ Gamma(2, 5)
    φ_hosp ~ Gamma(2, 5) 
    φ_deaths ~ Gamma(2, 5)
    
    # Initial infections (seeding period)
    log_I0 ~ Normal(5, 1)
    I_seed ~ arraydist([Normal(exp(log_I0), exp(log_I0)/3) for _ in 1:seed_period])
    
    # Time-varying Rt with smoothness constraint
    log_Rt_raw ~ arraydist([Normal(0, 0.1) for _ in 1:(T-seed_period)])
    
    # Initial Rt
    log_Rt_init ~ Normal(0, 0.5)
    
    # Construct smoothed log Rt
    log_Rt = Vector{eltype(log_Rt_raw)}(undef, T-seed_period)
    log_Rt[1] = log_Rt_init + log_Rt_raw[1]
    
    for t in 2:(T-seed_period)
        log_Rt[t] = log_Rt[t-1] + log_Rt_raw[t]  # Random walk
    end
    
    Rt = exp.(log_Rt)
    
    # Infections via renewal equation
    infections = Vector{eltype(I_seed)}(undef, T)
    
    # Seeding period
    for t in 1:seed_period
        infections[t] = I_seed[t]
    end
    
    # Renewal equation period
    for t in (seed_period+1):T
        renewal_sum = zero(eltype(infections))
        for s in 1:min(gen_len, t-1)
            renewal_sum += infections[t-s] * gen_interval[s]
        end
        infections[t] = Rt[t-seed_period] * renewal_sum
    end
    
    # Expected observations for each stream
    expected_cases = Vector{eltype(infections)}(undef, T)
    expected_hosp = Vector{eltype(infections)}(undef, T)  
    expected_deaths = Vector{eltype(infections)}(undef, T)
    
    for t in 1:T
        # Cases
        cases_sum = zero(eltype(infections))
        for d in 1:min(length(cases_delay), t)
            cases_sum += infections[t-d+1] * cases_delay[d]
        end
        expected_cases[t] = p_cases * cases_sum
        
        # Hospitalisations
        hosp_sum = zero(eltype(infections))
        for d in 1:min(length(hosp_delay), t)
            hosp_sum += infections[t-d+1] * hosp_delay[d]
        end
        expected_hosp[t] = p_hosp * hosp_sum
        
        # Deaths  
        death_sum = zero(eltype(infections))
        for d in 1:min(length(death_delay), t)
            death_sum += infections[t-d+1] * death_delay[d]
        end
        expected_deaths[t] = p_deaths * death_sum
    end
    
    # Likelihood with overdispersion (negative binomial)
    for t in 1:T
        # Negative binomial parameterization: NegativeBinomial(r, p) where mean = r(1-p)/p
        # We want mean = μ and variance = μ + μ²/φ
        # This gives us: r = φ, p = φ/(φ + μ)
        
        if expected_cases[t] > 0
            p_nb_cases = φ_cases / (φ_cases + expected_cases[t])
            cases[t] ~ NegativeBinomial(φ_cases, p_nb_cases)
        end
        
        if expected_hosp[t] > 0  
            p_nb_hosp = φ_hosp / (φ_hosp + expected_hosp[t])
            hospitalisations[t] ~ NegativeBinomial(φ_hosp, p_nb_hosp)
        end
        
        if expected_deaths[t] > 0
            p_nb_deaths = φ_deaths / (φ_deaths + expected_deaths[t])  
            deaths[t] ~ NegativeBinomial(φ_deaths, p_nb_deaths)
        end
    end
    
    return infections, Rt, expected_cases, expected_hosp, expected_deaths
end

# Main analysis function
function run_joint_rt_estimation()
    println("Loading data...")
    cases, hospitalisations, deaths, T = load_data()
    
    println("Setting up delay distributions...")
    cases_delay, hosp_delay, death_delay = setup_delay_distributions()
    gen_interval = setup_generation_interval()
    
    println("Fitting model...")
    model = joint_rt_model(cases, hospitalisations, deaths, T,
                          cases_delay, hosp_delay, death_delay, gen_interval)
    
    # Sample from posterior
    sampler = NUTS(0.65)
    n_samples = 1000
    n_chains = 4
    
    chains = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
    
    println("Extracting results...")
    
    # Extract Rt estimates
    rt_samples = Array(group(chains, :Rt))
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    
    # Extract ascertainment rates
    p_cases_est = mean(Array(group(chains, :p_cases)))
    p_hosp_est = mean(Array(group(chains, :p_hosp))) 
    p_deaths_est = mean(Array(group(chains, :p_deaths)))
    
    # Extract overdispersion parameters
    phi_cases_est = mean(Array(group(chains, :φ_cases)))
    phi_hosp_est = mean(Array(group(chains, :φ_hosp)))
    phi_deaths_est = mean(Array(group(chains, :φ_deaths)))
    
    println("\n=== RESULTS ===")
    println("Ascertainment Rates:")
    println("  Cases: $(round(p_cases_est, digits=3))")
    println("  Hospitalisations: $(round(p_hosp_est, digits=3))")  
    println("  Deaths: $(round(p_deaths_est, digits=3))")
    
    println("\nOverdispersion Parameters:")
    println("  Cases φ: $(round(phi_cases_est, digits=2))")
    println("  Hospitalisations φ: $(round(phi_hosp_est, digits=2))")
    println("  Deaths φ: $(round(phi_deaths_est, digits=2))")
    
    println("\nRt Summary:")
    println("  Mean Rt: $(round(mean(rt_mean), digits=2))")
    println("  Min Rt: $(round(minimum(rt_mean), digits=2))")
    println("  Max Rt: $(round(maximum(rt_mean), digits=2))")
    
    # Create plots
    println("\nGenerating plots...")
    
    # Determine seeding period for plotting
    max_delay = max(length(cases_delay), length(hosp_delay), length(death_delay))
    gen_len = length(gen_interval)
    seed_period = max(max_delay, gen_len, 7)
    
    # Time vector for Rt (excluding seeding period)
    time_rt = (seed_period+1):T
    
    # Plot 1: Rt over time
    p1 = plot(time_rt, rt_mean, ribbon=(rt_mean .- rt_lower, rt_upper .- rt_mean),
              fillalpha=0.3, linewidth=2, label="Rt estimate",
              xlabel="Time", ylabel="Reproduction Number (Rt)", 
              title="Time-varying Reproduction Number")
    hline!([1.0], linestyle=:dash, color=:red, alpha=0.7, label="Rt = 1")
    
    # Plot 2: Observed data streams
    p2 = plot(1:T, cases, label="Cases", linewidth=2, alpha=0.8)
    plot!(1:T, hospitalisations, label="Hospitalisations", linewidth=2, alpha=0.8)
    plot!(1:T, deaths, label="Deaths", linewidth=2, alpha=0.8)
    xlabel!("Time")
    ylabel!("Daily Count") 
    title!("Observed Data Streams")
    
    # Combined plot
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    display(combined_plot)
    
    return (
        rt_estimates = (mean=rt_mean, lower=rt_lower, upper=rt_upper),
        ascertainment = (cases=p_cases_est, hosp=p_hosp_est, deaths=p_deaths_est),
        overdispersion = (cases=phi_cases_est, hosp=phi_hosp_est, deaths=phi_deaths_est),
        chains = chains,
        time_rt = time_rt
    )
end

# Generate sample data if observations.csv doesn't exist
function generate_sample_data()
    println("Generating sample data...")
    
    T = 100
    true_rt = vcat(fill(1.5, 20), 
                   1.5 .- 0.8 * (1:30)/30,  # Declining phase
                   fill(0.7, 25),            # Low phase  
                   0.7 .+ 0.6 * (1:25)/25)  # Recovery phase
    
    # Generate infections using renewal equation
    gen_interval = setup_generation_interval()
    infections = Vector{Float64}(undef, T)
    infections[1:7] .= 100.0  # Initial seeding
    
    for t in 8:T
        renewal_sum = 0.0
        for s in 1:min(length(gen_interval), t-1)
            renewal_sum += infections[t-s] * gen_interval[s]
        end
        infections[t] = true_rt[t-7] * renewal_sum
    end
    
    # Generate observations with delays
    cases_delay, hosp_delay, death_delay = setup_delay_distributions()
    
    true_p_cases = 0.15
    true_p_hosp = 0.03  
    true_p_deaths = 0.005
    
    cases = Vector{Int}(undef, T)
    hospitalisations = Vector{Int}(undef, T)  
    deaths = Vector{Int}(undef, T)
    
    for t in 1:T
        # Expected values with delays
        exp_cases = true_p_cases * sum(infections[max(1,t-d+1)] * cases_delay[d] 
                                     for d in 1:min(length(cases_delay), t))
        exp_hosp = true_p_hosp * sum(infections[max(1,t-d+1)] * hosp_delay[d]
                                   for d in 1:min(length(hosp_delay), t))
        exp_deaths = true_p_deaths * sum(infections[max(1,t-d+1)] * death_delay[d]
                                       for d in 1:min(length(death_delay), t))
        
        # Add overdispersion
        cases[t] = rand(NegativeBinomial(10, 10/(10 + exp_cases)))
        hospitalisations[t] = rand(NegativeBinomial(5, 5/(5 + exp_hosp)))
        deaths[t] = rand(NegativeBinomial(3, 3/(3 + exp_deaths)))
    end
    
    # Save to CSV
    df = DataFrame(
        date = [Date(2023, 1, 1) + Day(i-1) for i in 1:T],
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    CSV.write("observations.csv", df)
    println("Sample data saved to observations.csv")
    
    return df
end

# Main execution
function main()
    # Check if data file exists, if not generate sample data
    if !isfile("observations.csv")
        generate_sample_data()
    end
    
    # Run the joint Rt estimation
    results = run_joint_rt_estimation()
    
    return results
end

# Run the analysis
results = main()

