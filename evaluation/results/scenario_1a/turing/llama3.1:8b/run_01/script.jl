
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/turing/llama3.1:8b/run_01")

using CSV
using DataFrames
using Statistics
using Turing
import Plots

# Load data
df = CSV.read("cases.csv", DataFrame)

# Function to estimate Rt using SEIR model
function estimate_Rt(df; seed=123, iter=1000)
    # Define the model
    @model function SEIR_model(cases, t)
        β ~ InverseGamma(2, 1)  # transmission rate
        γ ~ InverseGamma(2, 1)  # recovery rate

        # SEIR dynamics
        S[t] = maximum(cases[:cases][1:t]) - sum(cases[:cases][1:t-1])
        E[t] = (β * S[t-1] * cases[:cases][t-1]) / (S[t-1] + γ)
        I[t] = (γ * E[t-1]) / (E[t-1] + γ)
        R[t] = maximum(cases[:cases][1:t])

        Rt ~ Normal(0, 1)   # reproduction number

        [Rt]
    end

    # Fit the model
    chain = sample(SEIR_model(df.cases, 1:size(df, 1)), NUTS(100), MCMCThreads(), 4, progress=false)

    # Extract Rt estimates for each time point
    rt_estimates = Array{Turing.AbstractRVar}(undef, size(df, 1))
    for i in 1:size(df, 1)
        rt_estimates[i] = chain[i, :Rt]
    end

    return rt_estimates
end

# Estimate Rt over time
rt_estimates = estimate_Rt(df)

# Plot Rt over time
plot(rt_estimates, label="Estimated Rt")
xlabel!("Time (Days)")
ylabel!("Rt")
title!("Reproduction Number Estimates Over Time")
display(plot())

