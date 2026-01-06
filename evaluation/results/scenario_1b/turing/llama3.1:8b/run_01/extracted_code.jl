# Load required packages
using CSV
using DataFrames
using Turing
using Plots

# Load data
df = CSV.read("cases.csv", DataFrame)

# Prepare time series data (convert date to Date type)
dates = map(date -> Date(date), df.date);
df.date = dates;

# Define parameters
n_steps = 100; # number of steps to estimate Rt
S = 14;         # generation interval

# Define model
@model function model(df, S)
    R_t[1] ~ Uniform(0.5, 2) # initial value for Rt (can be improved)
    
    for t in 2:n_steps
        I_t[t] = R_t[t-1] * sum(@smootheddf(I[t-s], 1, g_s)) 
        R_t[t] ~ Uniform(0.5, 2)
    end
    
    return nothing
end

# Define smoothed function for generation interval
g_s(x) = [0.25; 0.5; 0.25][x]

# Define observation model (Poisson distribution)
function obs_model(I_t, cases_t)
    return Poisson(cases_t .* I_t)
end

# Define the data structure to hold our observations
observations = Array{Any}(undef, n_steps)

# Fit the model
model_data = DataFrame()
for t in 1:n_steps
    if df.date[t] < Dates.value(now(Dates.Second(0))) - S # ignore too old cases
    
        # Create an observation array for this time step
        obs_t = obs_model(df.cases[t], ones(1))
        
        push!(model_data, (date=Dates.value(df.date[t]), cases=df.cases[t]))
        
    else
        
        observations[t] = obs_model(df.cases[t], ones(1))
        
    end
end

# Define model and fit it
model = model(model_data, S)
chain = sample(model, NUTS(), 10000);

# Extract Rt estimates for each time point
Rt_estimates = zeros(n_steps)
for i in 1:n_steps
    Rt_estimates[i] = mean(Chain[:R_t][i])
end

# Plot Rt over time
plot(Rt_estimates, label="Estimated Rt")
xlabel("Time (days)")
ylabel("Reproduction Number (Rt)")
title("Estimated Reproduction Number Over Time")

display(plot)

