
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/turing/claude-sonnet-4-20250514/run_01")

using Turing, CSV, DataFrames, Dates, Plots, StatsPlots, LinearAlgebra
using Distributions, Random, MCMCChains
using Statistics, StatsBase

# Set random seed for reproducibility
Random.seed!(42)

# Function to create synthetic data if needed
function create_synthetic_data()
    # Create synthetic data for demonstration
    dates = Date(2023, 1, 1):Day(1):Date(2023, 6, 30)
    n_days = length(dates)
    
    # True Rt trajectory (decreasing then increasing)
    true_rt = [2.0 * exp(-0.02 * t) + 0.8 + 0.3 * sin(0.03 * t) for t in 1:n_days]
    
    # Generation interval (gamma distribution, mean ~5 days)
    gen_interval = [pdf(Gamma(2, 2.5), s) for s in 1:14]
    gen_interval = gen_interval ./ sum(gen_interval)
    
    # Simulate infections using renewal equation
    infections = zeros(n_days)
    infections[1:7] .= 100  # Initial seeding
    
    for t in 8:n_days
        infections[t] = true_rt[t] * sum(infections[max(1,t-13):t-1] .* 
                                       gen_interval[1:min(13, t-1)])
    end
    
    # Reporting delay (log-normal, mean ~7 days)
    delay_pmf = [pdf(LogNormal(log(7), 0.5), d) for d in 1:21]
    delay_pmf = delay_pmf ./ sum(delay_pmf)
    
    # Day-of-week effects (lower on weekends)
    true_dow_effects = [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5]
    
    # Time-varying ascertainment (starts high, drops, then increases)
    true_ascertainment = [0.8 * exp(-0.01 * t) + 0.3 + 0.2 * sin(0.02 * t) 
                         for t in 1:n_days]
    true_ascertainment = clamp.(true_ascertainment, 0.1, 1.0)
    
    # Generate reported cases with all observation processes
    cases = zeros(Int, n_days)
    
    for t in 1:n_days
        expected_reports = 0.0
        
        # Apply reporting delay
        for d in 1:min(21, t)
            inf_day = t - d + 1
            if inf_day > 0
                expected_reports += infections[inf_day] * delay_pmf[d] * 
                                  true_ascertainment[inf_day]
            end
        end
        
        # Apply day-of-week effect
        dow = dayofweek(dates[t])
        expected_reports *= true_dow_effects[dow]
        
        # Generate overdispersed cases (negative binomial)
        if expected_reports > 0
            cases[t] = rand(NegativeBinomial(expected_reports / (1 + expected_reports/20), 
                                           20 / (20 + expected_reports)))
        end
    end
    
    # Create DataFrame
    df = DataFrame(
        date = dates,
        cases = cases,
        day_of_week = dayofweek.(dates)
    )
    
    # Save to CSV
    CSV.write("cases.csv", df)
    
    return df, true_rt, true_dow_effects, true_ascertainment
end

# Load or create data
function load_data()
    if !isfile("cases.csv")
        println("Creating synthetic data...")
        return create_synthetic_data()
    else
        df = CSV.read("cases.csv", DataFrame)
        df.date = Date.(df.date)
        return df, nothing, nothing, nothing
    end
end

# Define the Bayesian model
@model function rt_model(cases, day_of_week, n_days)
    
    # Priors
    # Initial Rt
    rt_init ~ LogNormal(log(1.0), 0.5)
    
    # Rt random walk innovation standard deviation
    σ_rt ~ Exponential(0.1)
    
    # Day-of-week effects (Monday is reference)
    dow_raw ~ MvNormal(zeros(6), 0.5)
    dow_effects = vcat([1.0], exp.(dow_raw))
    
    # Ascertainment process
    asc_init ~ Beta(2, 2)  # Initial ascertainment rate
    σ_asc ~ Exponential(0.1)  # Innovation SD for ascertainment
    
    # Overdispersion parameter
    ϕ ~ Exponential(10.0)
    
    # Generation interval (fixed gamma distribution)
    gen_mean = 5.0
    gen_sd = 2.5
    gen_shape = (gen_mean / gen_sd)^2
    gen_scale = gen_sd^2 / gen_mean
    max_gen = 14
    gen_interval = [pdf(Gamma(gen_shape, gen_scale), s) for s in 1:max_gen]
    gen_interval = gen_interval ./ sum(gen_interval)
    
    # Reporting delay (fixed log-normal)
    delay_mean = 7.0
    delay_sd = 0.5
    max_delay = 21
    delay_pmf = [pdf(LogNormal(log(delay_mean), delay_sd), d) for d in 1:max_delay]
    delay_pmf = delay_pmf ./ sum(delay_pmf)
    
    # Time-varying parameters
    rt = Vector{Float64}(undef, n_days)
    ascertainment = Vector{Float64}(undef, n_days)
    infections = Vector{Float64}(undef, n_days)
    
    # Initial seeding period (first 7 days)
    for t in 1:7
        infections[t] ~ LogNormal(log(50.0), 0.5)
        rt[t] = rt_init
        ascertainment[t] = asc_init
    end
    
    # Subsequent time points
    rt[8] = rt_init
    ascertainment[8] = asc_init
    
    for t in 8:n_days
        # Rt random walk
        if t > 8
            rt[t] ~ LogNormal(log(rt[t-1]), σ_rt)
        end
        
        # Ascertainment random walk (on logit scale)
        if t > 8
            asc_logit_prev = log(ascertainment[t-1] / (1 - ascertainment[t-1]))
            asc_logit ~ Normal(asc_logit_prev, σ_asc)
            ascertainment[t] = 1 / (1 + exp(-asc_logit))
        end
        
        # Renewal equation for infections
        renewal_sum = 0.0
        for s in 1:min(max_gen, t-1)
            if t-s >= 1
                renewal_sum += infections[t-s] * gen_interval[s]
            end
        end
        infections[t] = rt[t] * renewal_sum
    end
    
    # Observation model
    for t in 1:n_days
        expected_cases = 0.0
        
        # Apply reporting delay and ascertainment
        for d in 1:min(max_delay, t)
            inf_day = t - d + 1
            if inf_day >= 1
                expected_cases += infections[inf_day] * delay_pmf[d] * 
                                ascertainment[inf_day]
            end
        end
        
        # Apply day-of-week effect
        expected_cases *= dow_effects[day_of_week[t]]
        
        # Overdispersed observation
        if expected_cases > 0
            # Negative binomial parameterization
            p = ϕ / (ϕ + expected_cases)
            cases[t] ~ NegativeBinomial(ϕ, p)
        else
            cases[t] ~ Poisson(0.001)
        end
    end
    
    return rt, dow_effects, ascertainment, infections
end

# Main analysis function
function analyze_rt()
    println("Loading data...")
    data, true_rt, true_dow_effects, true_ascertainment = load_data()
    
    n_days = nrow(data)
    println("Analyzing $n_days days of data")
    
    # Fit the model
    println("Fitting Bayesian model...")
    model = rt_model(data.cases, data.day_of_week, n_days)
    
    # Sample from posterior
    n_samples = 1000
    n_chains = 4
    
    println("Running MCMC sampling...")
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    println("Extracting results...")
    
    # Extract Rt estimates
    rt_samples = group(chain, :rt)
    rt_mean = [mean(rt_samples["rt[$i]"]) for i in 1:n_days]
    rt_lower = [quantile(rt_samples["rt[$i]"], 0.025) for i in 1:n_days]
    rt_upper = [quantile(rt_samples["rt[$i]"], 0.975) for i in 1:n_days]
    
    # Extract day-of-week effects
    dow_samples = group(chain, :dow_effects)
    dow_mean = [mean(dow_samples["dow_effects[$i]"]) for i in 1:7]
    dow_lower = [quantile(dow_samples["dow_effects[$i]"], 0.025) for i in 1:7]
    dow_upper = [quantile(dow_samples["dow_effects[$i]"], 0.975) for i in 1:7]
    
    # Extract ascertainment estimates
    asc_samples = group(chain, :ascertainment)
    asc_mean = [mean(asc_samples["ascertainment[$i]"]) for i in 1:n_days]
    asc_lower = [quantile(asc_samples["ascertainment[$i]"], 0.025) for i in 1:n_days]
    asc_upper = [quantile(asc_samples["ascertainment[$i]"], 0.975) for i in 1:n_days]
    
    # Create results summary
    results = DataFrame(
        date = data.date,
        rt_mean = rt_mean,
        rt_lower = rt_lower,
        rt_upper = rt_upper,
        ascertainment_mean = asc_mean,
        ascertainment_lower = asc_lower,
        ascertainment_upper = asc_upper,
        cases = data.cases
    )
    
    dow_results = DataFrame(
        day_of_week = 1:7,
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                   "Friday", "Saturday", "Sunday"],
        effect_mean = dow_mean,
        effect_lower = dow_lower,
        effect_upper = dow_upper
    )
    
    # Print summary
    println("\n=== Rt ESTIMATION RESULTS ===")
    println("Final Rt estimate: $(round(rt_mean[end], digits=3)) ",
            "[$(round(rt_lower[end], digits=3)), $(round(rt_upper[end], digits=3))]")
    
    println("\n=== DAY-OF-WEEK EFFECTS ===")
    for i in 1:7
        println("$(dow_results.day_name[i]): $(round(dow_results.effect_mean[i], digits=3)) ",
                "[$(round(dow_results.effect_lower[i], digits=3)), $(round(dow_results.effect_upper[i], digits=3))]")
    end
    
    println("\n=== ASCERTAINMENT SUMMARY ===")
    println("Initial ascertainment: $(round(asc_mean[1], digits=3)) ",
            "[$(round(asc_lower[1], digits=3)), $(round(asc_upper[1], digits=3))]")
    println("Final ascertainment: $(round(asc_mean[end], digits=3)) ",
            "[$(round(asc_lower[end], digits=3)), $(round(asc_upper[end], digits=3))]")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot 1: Rt over time
    p1 = plot(data.date, rt_mean, ribbon=(rt_mean .- rt_lower, rt_upper .- rt_mean),
             label="Rt estimate", linewidth=2, fillalpha=0.3,
             title="Time-varying Reproduction Number (Rt)",
             xlabel="Date", ylabel="Rt")
    hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1", linewidth=2)
    
    if true_rt !== nothing
        plot!(data.date, true_rt, label="True Rt", linestyle=:dash, linewidth=2)
    end
    
    # Plot 2: Cases and fitted values
    p2 = plot(data.date, data.cases, label="Observed cases", 
             title="Observed Cases Over Time", xlabel="Date", ylabel="Cases",
             seriestype=:scatter, alpha=0.6)
    
    # Plot 3: Day-of-week effects
    p3 = bar(dow_results.day_name, dow_results.effect_mean, 
            yerr=(dow_results.effect_mean .- dow_results.effect_lower,
                  dow_results.effect_upper .- dow_results.effect_mean),
            title="Day-of-Week Effects", xlabel="Day", ylabel="Relative Effect",
            xrotation=45)
    hline!([1.0], linestyle=:dash, color=:red, linewidth=2)
    
    # Plot 4: Ascertainment over time
    p4 = plot(data.date, asc_mean, ribbon=(asc_mean .- asc_lower, asc_upper .- asc_mean),
             label="Ascertainment rate", linewidth=2, fillalpha=0.3,
             title="Time-varying Ascertainment Rate",
             xlabel="Date", ylabel="Ascertainment Rate")
    
    if true_ascertainment !== nothing
        plot!(data.date, true_ascertainment, label="True ascertainment", 
              linestyle=:dash, linewidth=2)
    end
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
    display(final_plot)
    savefig(final_plot, "rt_analysis_results.png")
    
    # Save results
    CSV.write("rt_estimates.csv", results)
    CSV.write("dow_effects.csv", dow_results)
    
    println("\nResults saved to:")
    println("- rt_estimates.csv: Time-varying Rt and ascertainment estimates")
    println("- dow_effects.csv: Day-of-week effect estimates")
    println("- rt_analysis_results.png: Summary plots")
    
    return results, dow_results, chain
end

# Run the analysis
results, dow_results, chain = analyze_rt()

