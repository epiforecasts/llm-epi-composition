using Turing
using DataFrames
using Statistics
using Plots

# Load data
df = DataFrame(CSV.File("cases.csv"))

# Model definition
@model function covid_model(df)
    # Infection process
    R_t ~ LogNormal(0, 1) # Time-varying reproduction number
    I_s ~ Normal.(zeros(length(df.cases)), R_t .* df.cases ./ 100)

    # Observation process
    delay ~ Gamma(1, 3) # Delay from infection to reporting (mean=3 days)
    day_effect = @fun() begin
        for i in 1:7
            if df.day_of_week .== i
                return 1.5 + rand(Normal(0, 0.2)) # Day-of-week effect
            end
        end
        return 1
    end

    ascertainment ~ Beta(1, 1) # Time-varying proportion of infections reported
    overdispersion ~ Gamma(1, 10) # Overdispersion parameter (alpha=10)

    for i in 1:length(df.cases)
        y_i = df.cases[i]
        λ_i = R_t * sum(I_s[j] .* day_effect[j] .* exp(-delay * j) for j in 1:S) * ascertainment
        y_i_obs ~ Poisson(λ_i .* overdispersion)
    end

    # Prior on initial infection seeding period (e.g., assume it's negligible)
    I_s[1] ~ Normal(0, 10)

    # Handle the initial infection seeding period by ignoring the first few days of data
    for i in 1:5 # arbitrary choice; adjust as needed
        y_i = df.cases[i]
        λ_i = R_t * sum(I_s[j] .* day_effect[j] .* exp(-delay * j) for j in 1:S) * ascertainment
        y_i_obs ~ Poisson(λ_i .* overdispersion)
    end
end

# Define model parameters and data
S = 10 # Number of past infections to consider (days)
N = length(df.cases) # Total number of days

# Initialize model and data
m = covid_model(df)

# Sample from the posterior distribution
chain = sample(m, NUTS(1000), MCMCThreads(), 4)

# Extract estimates
Rt_estimates = mean.(map(x -> x.R_t, chain))
day_effect_estimates = mean.(map(x -> x.day_effect, chain))
ascertainment_estimates = mean.(map(x -> x.ascertainment, chain))

# Plot results
plot(Rt_estimates)
xlabel("Time (days)")
ylabel("Rt estimate")
title("Estimated Time-Varying Reproduction Number")

plot(day_effect_estimates)
xlabel("Day of Week")
ylabel("Day-of-Week Effect Estimate")
title("Estimated Day-of-Week Effects")

plot(ascertainment_estimates)
xlabel("Time (days)")
ylabel("Ascertainment Estimate")
title("Estimated Time-Varying Ascertainment")

# Summarize results
println("Rt estimates:")
for i in 1:length(Rt_estimates)
    println("Day $i: $(round(Rt_estimates[i], digits=2))")
end

println("Day-of-Week Effect Estimates:")
for i in 1:7
    println("Day of Week $i: $(round(day_effect_estimates[i], digits=2))")
end

println("Time-Varying Ascertainment Estimates:")
for i in 1:length(ascertainment_estimates)
    println("Day $i: $(round(ascertainment_estimates[i], digits=2))")
end

