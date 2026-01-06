
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/turing/claude-sonnet-4-20250514/run_03")

using Turing, StatsPlots, CSV, DataFrames, Dates, LinearAlgebra, Random
using Distributions, MCMCChains, StatsBase

# Set random seed for reproducibility
Random.seed!(42)

# First, let's create some synthetic data for demonstration
function create_synthetic_data()
    n_days = 150
    dates = Date(2020, 3, 1):Day(1):(Date(2020, 3, 1) + Day(n_days - 1))
    
    # True Rt that varies over time (simulating epidemic waves)
    true_rt = [
        i <= 30 ? 2.5 : 
        i <= 60 ? 2.5 - 1.8 * (i - 30) / 30 :  # Decline due to interventions
        i <= 90 ? 0.7 + 0.6 * (i - 60) / 30 :   # Rise due to relaxation
        1.3 - 0.8 * (i - 90) / 60               # Final decline
        for i in 1:n_days
    ]
    
    # Generation interval (gamma distribution with mean 5.2, sd 5.1)
    g = [pdf(Gamma(1.04, 5.0), x) for x in 1:20]
    g = g ./ sum(g)
    
    # Reporting delay (gamma distribution with mean 7, sd 4)
    delay_dist = [pdf(Gamma(3.0, 2.3), x) for x in 1:25]
    delay_dist = delay_dist ./ sum(delay_dist)
    
    # Simulate true infections
    infections = zeros(n_days)
    infections[1:10] .= 50.0  # Initial seeding
    
    for t in 11:n_days
        infections[t] = true_rt[t] * sum(infections[max(1,t-length(g)+1):t-1] .* reverse(g[1:min(length(g)-1, t-1)]))
    end
    
    # Apply reporting delays and day-of-week effects
    day_effects = [1.2, 1.1, 1.0, 0.95, 0.9, 0.7, 0.6]  # Mon-Sun multipliers
    ascertainment = 0.3 * (1 .+ 0.3 * sin.(2π * (1:n_days) / 50))  # Time-varying ascertainment
    
    expected_reports = zeros(n_days)
    for t in 1:n_days
        for d in 1:min(length(delay_dist), t)
            dow = dayofweek(dates[t])
            expected_reports[t] += infections[t-d+1] * delay_dist[d] * ascertainment[t-d+1] * day_effects[dow]
        end
    end
    
    # Add overdispersion (negative binomial)
    cases = [rand(NegativeBinomial(max(1, r), 0.3)) for r in expected_reports]
    
    df = DataFrame(
        date = dates,
        cases = cases,
        day_of_week = dayofweek.(dates)
    )
    
    CSV.write("cases.csv", df)
    return df, true_rt, infections
end

# Load or create data
if !isfile("cases.csv")
    println("Creating synthetic data...")
    data, true_rt, true_infections = create_synthetic_data()
else
    data = CSV.read("cases.csv", DataFrame)
end

# Prepare generation interval and reporting delay distributions
function get_generation_interval(max_gen = 20)
    # Generation interval based on COVID-19 literature (mean ≈ 5.2 days)
    g = [pdf(Gamma(1.04, 5.0), x) for x in 1:max_gen]
    return g ./ sum(g)
end

function get_reporting_delay(max_delay = 25)
    # Reporting delay (mean ≈ 7 days)
    d = [pdf(Gamma(3.0, 2.3), x) for x in 1:max_delay]
    return d ./ sum(d)
end

# Turing model
@model function rt_model(cases, day_of_week, n_days, gen_interval, reporting_delay)
    
    # Priors for day-of-week effects (multiplicative, sum to 7)
    dow_raw ~ MvNormal(zeros(7), I(7))
    dow_effects = exp.(dow_raw .- mean(dow_raw))
    
    # Prior for Rt (random walk on log scale)
    rt_log_init ~ Normal(0, 0.5)  # Initial Rt around 1
    rt_innovations ~ MvNormal(zeros(n_days-1), 0.1^2 * I(n_days-1))
    rt_log = cumsum([rt_log_init; rt_innovations])
    rt = exp.(rt_log)
    
    # Time-varying ascertainment (smooth via random walk)
    ascertainment_logit_init ~ Normal(-1, 0.5)  # Initial ascertainment around 0.3
    ascertainment_innovations ~ MvNormal(zeros(n_days-1), 0.05^2 * I(n_days-1))
    ascertainment_logit = cumsum([ascertainment_logit_init; ascertainment_innovations])
    ascertainment = 1 ./ (1 .+ exp.(-ascertainment_logit))
    
    # Initial infections (seeding period)
    init_infections ~ MvNormal(log.(50) * ones(10), 0.2^2 * I(10))
    infections = zeros(n_days)
    infections[1:10] = exp.(init_infections)
    
    # Overdispersion parameter
    φ ~ Exponential(10)  # For negative binomial
    
    # Generate infections via renewal equation
    for t in 11:n_days
        # Convolution with generation interval
        renewal_sum = 0.0
        for s in 1:min(length(gen_interval), t-1)
            if t-s >= 1
                renewal_sum += infections[t-s] * gen_interval[s]
            end
        end
        infections[t] = rt[t] * renewal_sum
    end
    
    # Generate expected reported cases with delays and effects
    for t in 1:n_days
        expected_reports = 0.0
        
        # Apply reporting delay
        for d in 1:min(length(reporting_delay), t)
            if t-d+1 >= 1
                expected_reports += infections[t-d+1] * reporting_delay[d] * ascertainment[t-d+1]
            end
        end
        
        # Apply day-of-week effect
        expected_reports *= dow_effects[day_of_week[t]]
        expected_reports = max(expected_reports, 1e-6)
        
        # Negative binomial observation model
        p_nb = φ / (φ + expected_reports)
        cases[t] ~ NegativeBinomial(φ, p_nb)
    end
    
    return (rt = rt, dow_effects = dow_effects, ascertainment = ascertainment, 
            infections = infections)
end

# Prepare data
cases_data = data.cases
dow_data = data.day_of_week
n_days = length(cases_data)

# Get distributions
gen_interval = get_generation_interval()
reporting_delay = get_reporting_delay()

println("Setting up model...")
model = rt_model(cases_data, dow_data, n_days, gen_interval, reporting_delay)

# Sample from the model
println("Sampling from model (this may take several minutes)...")
n_samples = 1000
n_chains = 4

# Use NUTS sampler with adaptation
sampler = NUTS(0.65)
chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains, 
               progress=true, verbose=true)

println("Sampling complete!")

# Extract results
function extract_results(chain, n_days)
    # Extract Rt estimates
    rt_samples = Array(group(chain, :rt))
    rt_mean = mean(rt_samples, dims=1)[1, :]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:n_days]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:n_days]
    
    # Extract day-of-week effects
    dow_samples = Array(group(chain, :dow_effects))
    dow_mean = mean(dow_samples, dims=1)[1, :]
    dow_lower = [quantile(dow_samples[:, i], 0.025) for i in 1:7]
    dow_upper = [quantile(dow_samples[:, i], 0.975) for i in 1:7]
    
    # Extract ascertainment
    asc_samples = Array(group(chain, :ascertainment))
    asc_mean = mean(asc_samples, dims=1)[1, :]
    asc_lower = [quantile(asc_samples[:, i], 0.025) for i in 1:n_days]
    asc_upper = [quantile(asc_samples[:, i], 0.975) for i in 1:n_days]
    
    return (
        rt = (mean = rt_mean, lower = rt_lower, upper = rt_upper),
        dow_effects = (mean = dow_mean, lower = dow_lower, upper = dow_upper),
        ascertainment = (mean = asc_mean, lower = asc_lower, upper = asc_upper)
    )
end

results = extract_results(chain, n_days)

# Create summary
function print_summary(results, data)
    println("\n" * "="^60)
    println("RESULTS SUMMARY")
    println("="^60)
    
    println("\nDay-of-week effects (multiplicative):")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for (i, day) in enumerate(days)
        println("$day: $(round(results.dow_effects.mean[i], digits=3)) " *
                "[$(round(results.dow_effects.lower[i], digits=3)), " *
                "$(round(results.dow_effects.upper[i], digits=3))]")
    end
    
    println("\nRt statistics:")
    rt_mean = results.rt.mean
    println("Mean Rt: $(round(mean(rt_mean), digits=3))")
    println("Rt range: $(round(minimum(rt_mean), digits=3)) - $(round(maximum(rt_mean), digits=3))")
    println("Days with Rt > 1: $(sum(rt_mean .> 1)) / $(length(rt_mean))")
    
    println("\nAscertainment statistics:")
    asc_mean = results.ascertainment.mean
    println("Mean ascertainment: $(round(mean(asc_mean), digits=3))")
    println("Ascertainment range: $(round(minimum(asc_mean), digits=3)) - $(round(maximum(asc_mean), digits=3))")
    
    println("\nData summary:")
    println("Total cases: $(sum(data.cases))")
    println("Date range: $(data.date[1]) to $(data.date[end])")
    println("Days analyzed: $(length(data.cases))")
end

print_summary(results, data)

# Create plots
function create_plots(results, data)
    dates = data.date
    
    # Plot 1: Rt over time
    p1 = plot(dates, results.rt.mean, 
              ribbon=(results.rt.mean .- results.rt.lower, results.rt.upper .- results.rt.mean),
              label="Rt estimate", color=:blue, fillalpha=0.3,
              title="Time-varying Reproduction Number (Rt)",
              xlabel="Date", ylabel="Rt", 
              linewidth=2)
    hline!([1.0], color=:red, linestyle=:dash, label="Rt = 1", linewidth=2)
    
    # Plot 2: Day-of-week effects
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    p2 = scatter(1:7, results.dow_effects.mean,
                 yerror=(results.dow_effects.mean .- results.dow_effects.lower,
                        results.dow_effects.upper .- results.dow_effects.mean),
                 title="Day-of-week Effects", xlabel="Day", ylabel="Multiplicative Effect",
                 xticks=(1:7, days), color=:green, markersize=6, linewidth=2)
    hline!([1.0], color=:red, linestyle=:dash, label="No effect", linewidth=1)
    
    # Plot 3: Ascertainment over time
    p3 = plot(dates, results.ascertainment.mean,
              ribbon=(results.ascertainment.mean .- results.ascertainment.lower,
                     results.ascertainment.upper .- results.ascertainment.mean),
              label="Ascertainment", color=:purple, fillalpha=0.3,
              title="Time-varying Ascertainment Rate",
              xlabel="Date", ylabel="Ascertainment Rate",
              linewidth=2)
    
    # Plot 4: Observed cases
    p4 = plot(dates, data.cases, title="Observed Cases", xlabel="Date", ylabel="Cases",
              color=:orange, linewidth=1, label="Cases")
    
    # Combine plots
    plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
end

# Generate and display plots
println("\nGenerating plots...")
final_plot = create_plots(results, data)
display(final_plot)

# Save results to CSV
function save_results(results, data)
    results_df = DataFrame(
        date = data.date,
        rt_mean = results.rt.mean,
        rt_lower = results.rt.lower,
        rt_upper = results.rt.upper,
        ascertainment_mean = results.ascertainment.mean,
        ascertainment_lower = results.ascertainment.lower,
        ascertainment_upper = results.ascertainment.upper,
        observed_cases = data.cases
    )
    
    CSV.write("rt_estimates.csv", results_df)
    
    dow_df = DataFrame(
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        effect_mean = results.dow_effects.mean,
        effect_lower = results.dow_effects.lower,
        effect_upper = results.dow_effects.upper
    )
    
    CSV.write("day_of_week_effects.csv", dow_df)
    
    println("Results saved to rt_estimates.csv and day_of_week_effects.csv")
end

save_results(results, data)

# Print diagnostics
println("\nModel diagnostics:")
println("Chain summary:")
display(chain)

println("\n" * "="^60)
println("Analysis complete!")
println("="^60)

