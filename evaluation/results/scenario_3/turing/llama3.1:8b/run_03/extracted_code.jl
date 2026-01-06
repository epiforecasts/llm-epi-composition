using Turing, DataFrames, StatsPlots

# Load data
data = readcsv("observations.csv")
df = DataFrame(data)

# Define model
@model function Rt_model(observations)
    R_t ~ TruncatedNormal(0.5, 1.0, lower=0) # Shared reproduction number

    # Model each stream separately to account for different delays and ascertainment rates
    for (i, row) in enumerate(eachrow(observations))
        date = Dates.parse(row.date)

        # Define observation-specific parameters
        cases_delay = Dates.value(Dates.day(Date("2022-02-01"), date))  # Assume hospitalisations and deaths are reported with a fixed delay of approximately 14 days after infection onset for most data sets. Adjust according to the actual delays in your dataset.
        hospitalisations_delay = cases_delay + 14
        deaths_delay = cases_delay + 28

        cases_observed = row.cases
        hospitalisations_observed = row.hospitalisations
        deaths_observed = row.deaths

        # Model the number of infections given the past infections and reproduction number
        logit_cases = R_t * sum(@. observations[i-s, :cases] / (1 + cases_delay))  # Assume ascertainment rate is 1 for simplicity in this example; replace with actual rates if available.
        logit_hospitalisations = R_t * sum(@. observations[i-s, :hospitalisations] / (1 + hospitalisations_delay))
        logit_deaths = R_t * sum(@. observations[i-s, :deaths] / (1 + deaths_delay))

        # Fit overdispersed Poisson distribution for each stream
        cases ~ InverseGamma(α=1, β=1) .* (exp.(logit_cases)) ./ (1 .+ exp.(logit_cases))
        hospitalisations ~ InverseGamma(α=1, β=1) .* (exp.(logit_hospitalisations)) ./ (1 .+ exp.(logit_hospitalisations))
        deaths ~ InverseGamma(α=1, β=1) .* (exp.(logit_deaths)) ./ (1 .+ exp.(logit_deaths))

    end
end

# Define a function to fit the model and extract Rt estimates
function fit_model(data)
    observations = [nrow(df)-i for i in eachindex(eachcol(df))]
    chain = sample(Rt_model(observations), MCMCThreads(), 1000, burnin=500, thinning=5)

    # Extract Rt estimates and stream-specific parameters
    rt_estimates = mean(chain[:R_t])
    return rt_estimates, chain
end

# Fit the model
rt_estimate, _ = fit_model(df)
println("Estimated reproduction number (Rt): ", rt_estimate)

# Plot results
plot([Dates.parse.(df.date) for i in eachindex(eachrow(df))], rt_estimate,
     label="Estimated Rt", xlabel="Time", ylabel="Reproduction Number",
     title="Estimated Time-Varying Reproduction Number")

