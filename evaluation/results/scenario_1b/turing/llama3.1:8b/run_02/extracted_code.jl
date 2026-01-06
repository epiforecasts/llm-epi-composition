# Required packages
using Pkg; Pkg.activate(".."); Pkg.add(["Turing", "DataFrames", "CSV"])

# Load required packages
using Turing, DataFrames, CSV

# Define constants and hyperparameters
g_s = [0.2 0.8] # Generation interval probability mass function (default)
S = length(g_s) # Number of time steps for the generation interval
T = 90 # Maximum number of days to model

# Load data from CSV file
data_path = "cases.csv"
df = CSV.read(data_path, DataFrame)

# Convert date column to Date type and sort by date
df.date = map(Date, df.date)
sort!(df, :date)

# Resample case counts to ensure daily data (in case of missing values)
case_counts = Array{Int}(undef, T+1) # Initialize array with zeros
for i in 1:T+1
    if i <= length(df)
        case_counts[i] = df.cases[df.date .== Date("2020-02-$(i)-01")]
    end
end

# Define the model using Turing.jl
@model function renewal_eq(Rt, g_s, S, T, data)
    # Priors for Rt and generation interval parameters
    rt_prior = Beta(1, 2) # Prior distribution for Rt (informative prior)
    
    # Generate the latent process I_t as a sum of past infections weighted by the generation intervals
    I = zeros(T+1)
    for t in 1:T+1
        if t <= S
            I[t] ~ InverseGamma(10, 2) * (Rt + 0.5) # Prior distribution for initial values
        else
            I[t] = sum(g_s[j].*I[t-j] for j=1:S)
        end
        
    end
    
    # Observation model: Poisson likelihood for case counts
    for t in 1:T+1
        data[t] ~ Poisson(I[t]*case_counts[t])
        
    end

end

# Define the parameters and variables of interest
param = [:Rt, :g_s]

# Initialize variables with default values
n_chains = 4 # Number of chains for parallel sampling
n_samples = 2000 # Number of samples to generate
thin = 2 # Sampling thinning factor
burnin = 100 # Burn-in period

# Fit the model using MCMC (Markov Chain Monte Carlo)
model_data = [case_counts; g_s]
chain = sample(renewal_eq(model_data, param..., S=S, T=T), 
              NUTS(0.65),
              MCMCThreads(n_chains, n_samples),
              thin=thin,
              burnin=burnin)

# Extract Rt estimates
Rt_estimates = chain[:Rt]

# Plot the time-varying reproduction number (Rt) over time
using Plots
plot(Rt_estimates[1:end-1], label="Rt")
xlabel!("Time point")
ylabel!("Rt estimate")


