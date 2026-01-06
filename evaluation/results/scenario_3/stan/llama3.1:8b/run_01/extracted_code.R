# Load required libraries
library(cmdstanr)
library(tidyverse)

# Load the data
data <- read_csv("observations.csv")

# Convert date to datetime format
data$date <- as_datetime(data$date, format = "%Y-%m-%d")

# Calculate delays and ascertainment rates for each stream
delays <- c(cases = 0, hospitalisations = 5, deaths = 10)
ascertains <- c(cases = 1, hospitalisations = 0.8, deaths = 0.95)

# Define the Stan model
model_code <- "
data {
  int<lower=1> S; // Number of streams (3: cases, hospitalisations, deaths)
  int<lower=1> T; // Number of time points
  vector[T] date; // Date for each time point
  array[S, T] y; // Observations for each stream and time point
  
  real<lower=0> rt_smoothness;
}

parameters {
  real<lower=0> rt[T]; // Time-varying reproduction number
  real<lower=0> mu_g; // Mean generation interval
  array[S] alpha; // Stream-specific ascertainment rates
  real<lower=0> sigma_y[S]; // Observation noise for each stream
  
  vector[3] beta; // Shared coefficients for exponential family
}

model {
  // Prior on rt_smoothness (e.g., Gaussian with mean and sd)
  rt_smoothness ~ normal(1, 2);
  
  // Priors on mu_g (e.g., Uniform between 0.5 and 10 days)
  mu_g ~ uniform(0.5, 10);
  
  // Prior on alpha (e.g., Dirichlet with concentration parameter 2)
  for (s in 1:S) {
    alpha[s] ~ dirichlet(rep(1, S));
  }
  
  // Prior on sigma_y (e.g., Half-Cauchy with scale parameter 2.5)
  for (s in 1:S) {
    sigma_y[s] ~ cauchy(0, 2.5);
  }
  
  // Priors on beta (e.g., Normal with mean and sd)
  beta ~ normal(0, 2);
  
  // Likelihood
  for (t in 1:T) {
    // Calculate observed infections using the renewal equation
    real observed_infections = 0;
    for (s in 1:S) {
      real y_observed = y[s, t] * alpha[s];
      observed_infections += y_observed / sigma_y[s]^2;
    }
    
    // Use exponential family likelihood with shared coefficients beta
    target += log_sum_exp(beta[1] + rt[t], observed_infections);
  }
  
  // Smoothness penalty on Rt (e.g., squared differences between adjacent values)
  for (t in 2:T) {
    target += pow(rt[t] - rt[t-1], 2) * rt_smoothness;
  }
}
"

# Fit the model
fit <- cmdstanr::cmd_stan_model(model_code = model_code, iter = 2000, chains = 4,
                                data = list(S = ncol(delays), T = length(unique(data$date)),
                                            date = unique(data$date),
                                            y = rbind(cases = data$cases[match(unique(data$date), data$date)],
                                                      hospitalisations = data$hospitalisations[match(unique(data$date), data$date)],
                                                      deaths = data$deaths[match(unique(data$date), data$date)]),
                                            delays = as.matrix(delays),
                                            ascertains = c(ascertains, 1 - sum(ascertains)) // include a dummy stream with ascertainment rate 1
                                ))

# Extract Rt estimates and stream-specific parameters
posterior <- extract(fit)

# Calculate Rt estimates by time point (e.g., average of posterior samples)
rt_estimates <- posterior$rt[, , ] %>%
  apply(2, mean)

# Calculate stream-specific ascertainment rates (e.g., average of posterior samples)
ascertainment_rates <- posterior$alpha[, ]

# Plot the results
plot(unique(data$date), rt_estimates,
     type = "l", main = "Time-varying Reproduction Number (Rt)",
     xlab = "Date", ylab = "Rt",
     col = "blue")
legend("topright", legend = c("Rt Estimates"),
       col = "blue",
       lty = 1, pch = 19,
       bty = "n")

# Plot the stream-specific ascertainment rates
plot(unique(data$date), ascertains[, 1], type = "l",
     main = "Stream-Specific Ascertainment Rates",
     xlab = "Date", ylab = "Ascertainment Rate")
legend("topright", legend = c("Cases", "Hospitalisations", "Deaths"),
       col = c(1, 2, 3),
       lty = 1,
       bty = "n")

# Handle the initial infection seeding period
initial_period <- data$date < as_date("2020-03-01") // assume start of pandemic

# Plot Rt estimates with initial period highlighted
plot(unique(data$date), rt_estimates, type = "l",
     main = "Time-varying Reproduction Number (Rt)",
     xlab = "Date", ylab = "Rt")
abline(v = unique(data$date)[initial_period], col = "red")

# Show the summary of the model
summary(fit)

