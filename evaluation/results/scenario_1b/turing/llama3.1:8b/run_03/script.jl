
cd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/turing/llama3.1:8b/run_03")

# Import necessary libraries
using Turing
using DataFrames
using Statistics
using Plots

# Load data
df = DataFrame(CSV.File("cases.csv"))

# Define parameters and priors
@model function model(df)
    # Number of days to consider in the past
    S ~ DiscreteUniform(1, 14)

    # Prior for Rt
    R_t ~ TruncatedNormal(0.5, 10, -Inf, 100)

    # Generate daily case counts using the renewal equation
    for t in 1:size(df, 1)
        I_t = @trace(R_t[t] * sum(@map(s -> df.cases[t-s], 1:S)) ~ Normal(0, 10))
    end

    return R_t
end

# Define the observation model (in this case, Poisson for case counts)
function obs_model(df, Rt)
    # Number of days to consider in the past
    S = 14

    # Generate daily case counts using the renewal equation and observation model
    for t in 1:size(df, 1)
        I_t ~ Poisson(Rt[t] * sum(@map(s -> df.cases[t-s], 1:S)))
    end
end

# Define the model to estimate Rt over time
function rt_model(df)
    # Initialize parameters and data
    params = Turing.initialize_parameters(model(df))
    data = obs_model(df, [params[R_t] for R_t in eachindex(params)])

    return data
end

# Fit the model using maximum a posteriori (MAP) estimation
ch = Turing.mapredict(rt_model(df))

# Extract Rt estimates for each time point
Rt_estimates = Array{Float64}(undef, size(ch, 1))
for t in eachindex(ch)
    Rt_estimates[t] = ch[t].R_t.value[1]
end

# Plot the estimated Rt over time
plot(Rt_estimates)
xlabel!("Time")
ylabel!("Estimated Rt")

println("Estimated Rt values:")
println.(eachindex(Rt_estimates), " ", Rt_estimates)

