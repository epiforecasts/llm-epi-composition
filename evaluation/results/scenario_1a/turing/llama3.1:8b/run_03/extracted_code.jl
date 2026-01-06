using Turing
using DataFrames
using CSV


# Load data
df = CSV.read("cases.csv")


# Convert date column to datetime
df[!, :date] = DateTime.(df.date)


# Define model parameters
N = 100000.0  # Population size
I0 = 10.0     # Initial number of infected

# Define the SIR model
@model function sir_model(β, gamma, t, y)
    σ ~ Normal(0, 1)  # Prior for Rt
    β ~ TruncatedNormal(0, Inf, lower = 0)
    gamma ~ TruncatedNormal(0, Inf, lower = 0)

    Rt = σ .* exp.(t)

    for (i, t_i) in enumerate(t)
        y[i] ~ Poisson(exp.((β + γ * Rt[t_i]) * I0))
    end
end

# Fit the model to the data
model = sir_model(df.cases[1], df.date, df.cases[2:end])
chain = sample(model, NUTS(100), MCMCThreads(), 4)

# Extract Rt estimates for each time point
Rt_estimates = map(t -> var(chains[:σ][t]), [i for i in 1:length(df.cases)])


# Extract Rt estimates as an array
Rt_array = Array{Float64, 2}(undef, length(df.cases), size(chains[:σ])[2])
for i in 1:size(chains[:σ])[2]
    for j in 1:length(df.cases)
        Rt_array[j, i] = var(chains[:σ][j, i])
    end
end

# Plot the estimated Rt over time
using Plots
plot(Rt_array')
xlabel!("Time")
ylabel!("Estimated Rt")
title!("Time-Varying Reproduction Number (Rt) Estimates")

