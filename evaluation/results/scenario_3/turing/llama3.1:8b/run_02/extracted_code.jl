using Turing
using DataFrames
using Statistics
using Plots
include("observations.csv")  # load data from observations.csv

# Model parameters
S = 7  # number of days in generation interval
delay_cases = 1  # delay from infection to case observation (days)
delay_hospitalisations = 5  # delay from infection to hospitalisation observation (days)
delay_deaths = 10  # delay from infection to death observation (days)

# Model

@model function rt_model(obs::DataFrame, stream_delays, stream_ascertainty_rates)
    n = length(unique(obs.date))
    Rt ~ InverseGamma(2, 1)  # prior on shared reproduction number
    β_stream_cases ~ Normal(0, 5)  # prior on ascertainment rate for cases
    β_stream_hospitalisations ~ Normal(0, 5)  # prior on ascertainment rate for hospitalisations
    β_stream_deaths ~ Normal(0, 5)  # prior on ascertainment rate for deaths

    for t in 1:n
        I_t = sum([obs[i, j] * stream_ascertainty_rates[j] *
                    stream_delays[j] > delay obs.date[t]
                   for (j, i) in enumerate(findall(obs.date .== obs.date[t])))
        rt_t ~ Beta(2, 2)  # prior on Rt at time t
        I_t ~ Poisson(Rt * sum([I_t_i * exp(-rt_t / S) *
                              exp(-(i - stream_delays[j]) / S)
                   for (j, i) in enumerate(obs[1:n, "cases", t])]))
    end

    return Rt, β_stream_cases, β_stream_hospitalisations, β_stream_deaths
end

# Load data and create observation matrix
data = readcsv("observations.csv")
dates = unique(data.date)
n_dates = length(dates)

# Create streams-specific delays and ascertainment rates
delay_cases = fill(delay_cases, n_dates);
delay_hospitalisations = fill(delay_hospitalisations, n_dates);
delay_deaths = fill(delay_deaths, n_dates);

β_stream_cases = ones(n_dates)  # assume constant ascertainment rate for cases
β_stream_hospitalisations = ones(n_dates)  # assume constant ascertainment rate for hospitalisations
β_stream_deaths = ones(n_dates)  # assume constant ascertainment rate for deaths

# Define the data structure
obs = DataFrame(
    date=repeat(dates, eachcol(data), outer=true),
    cases=data[!, "cases"],
    hospitalisations=data[!, "hospitalisations"],
    deaths=data[!, "deaths"]
)

# Fit the model
model = rt_model(obs, delay_cases, delay_hospitalisations, delay_deaths)
chain = sample(model, MCMCThreads(), 1000)

# Extract and plot results

# Rt estimates for each time point
Rt_chain = chain[:Rt]
plot(Rt_chain, label="Rt")

# Stream-specific ascertainment rate estimates
stream_ascertainty_rates = chain[[:β_stream_cases, :β_stream_hospitalisations, :β_stream_deaths]]
bar(stream_ascertainty_rates, label=["cases" "hospitalisations" "deaths"])

# Summary plot
plot!(legend=:topright)
xlabel!("Time")
ylabel!("Value")

# Display results
println("Rt estimates for each time point: ", Rt_chain)
println("Stream-specific ascertainment rate estimates: ")
println(stream_ascertainty_rates)

display(plot())

