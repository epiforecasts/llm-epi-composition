
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/turing/llama3.1:8b/run_02")

# Load necessary libraries
using Turing
using DataFrames
using CSV
import StatsBase

# Load data
df = DataFrame(CSV.File("cases.csv"))

# Convert date to DateTime for time series analysis
df[:date] = parse.(Date, df[:date])

# Initialize model parameters and state variables
@model function seir_model(cases)
    # Model parameters
    R0 ~ Normal(2.5, 1)  # Initial reproduction number
    gamma ~ Truncated(Normal(1/7, 0.5), (1/30, 1))  # Recovery rate

    # State variables
    I[1] = 1.0  # Number of infected at day 1
    E[1] = 0.0  # Number of exposed at day 1

    for i in 2:length(cases)
        # Transition equations for SEIR compartments
        E[i] ~ Truncated(Normal(I[i-1] * R0 / (gamma + 1), I[i-1] * R0 / sqrt(gamma)), (0, I[i-1] * R0 / gamma))
        I[i] ~ Truncated(Normal(E[i-1] - E[i], sqrt(E[i-1])), (0, I[i-1]))
    end

    # Observed cases
    for i in 2:length(cases)
        cases[i] ~ Poisson(I[i])
    end
end

# Define the model with data and initialize parameters
model = seir_model(df.cases)

# Define a function to calculate Rt for each day
function calc_rt(model, cdf::DataFrame)
    rt_estimates = []
    for i in 2:length(cdf.cases)
        R0 = Turing.extract_parameters(model)[1]
        E_last_i = getfield.(Turing.extract_states(model), :E)[i-1]
        I_last_i = getfield.(Turing.extract_states(model), :I)[i-1]

        # Estimate Rt
        rt_estimate = I_last_i * R0 / (gamma + 1)
        push!(rt_estimates, rt_estimate)

        # Update model parameters and state variables for next iteration
        Turing.update_params!(model, NamedTuple(R0=rt_estimate))
    end

    return rt_estimates
end

# Sample the model using MCMC
chn = sample(model, MCMCChain(10000))

# Calculate Rt estimates for each time point
rt_estimates = calc_rt(model, df)

# Summarize or plot Rt over time
using Plots
plot(df.date[2:end], rt_estimates)
xlabel!("Day")
ylabel!("Rt Estimate")
title!("Estimated Reproduction Number (Rt) Over Time")

