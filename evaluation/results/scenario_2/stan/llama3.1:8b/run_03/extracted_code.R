# Install necessary packages
install.packages("cmdstanr")
install.packages("dplyr")

# Load necessary libraries
library(cmdstanr)
library(dplyr)

# Read data
cases <- read.csv("cases.csv", colClasses = c("Date", "integer"))

# Prepare data for Stan
cases$day_of_week <- factor(cases$day_of_week, levels = 1:7, ordered = TRUE)
cases$date <- as.Date(cases$date)

# Define the model in Stan

model_code <- "
data {
  int<lower=0> T; // number of time points
  vector[T] cases; // reported cases at each time point
  vector[T] day_of_week; // day-of-week at each time point (1-7)
  real<lower=0, upper=1> ascertainment[T]; // proportion of infections that are reported at each time point
}

parameters {
  real<lower=0> R; // average reproduction number
  vector[T] gamma; // delay between infection and reporting at each time point (days)
  real<lower=0, upper=1> phi_day_of_week[7]; // day-of-week effect on reported cases (7 levels)
  real<lower=0, upper=1> alpha; // overdispersion parameter
}

model {
  // Model for day of week effects
  for (t in 1:T) {
    gamma[t] ~ normal(3, 2); // prior for delay between infection and reporting
    phi_day_of_week[day_of_week[t]] ~ beta(1, 1); // prior for day-of-week effect on reported cases
  }

  // Model for observed cases (renewal equation)
  for (t in 1:T) {
    R ~ normal(log(mean(gamma)) + 0.5 * (log(sum(cases[gamma < mean(gamma)])) - log(T)), sqrt(2^2));
    target += poisson_log_lik(cases[t], 
                              R * sum(phi_day_of_week[day_of_week[t]] .* cases[(gamma < T) & (gamma >= t)]) .* ascertainment[t]);
  }
  
  // Model for overdispersion
  alpha ~ beta(1, 1);
}
"

# Fit the model

fit <- cmdstanr::sample(
  file = "reproduction_number.stan", 
  data = list(T = nrow(cases), cases = cases$cases, day_of_week = as.numeric(factor(cases$day_of_week)), ascertainment = rep(1, nrow(cases))),
  chains = 4, iter_warmup = 1000, iter_sampling = 2000
)

# Extract results

Rt_estimates <- fit$summary("R", probs = c(0.025, 0.975))
day_of_week_effects <- fit$summary("phi_day_of_week", probs = c(0.025, 0.975))
ascertainment_estimates <- fit$summary("ascertainment", probs = c(0.025, 0.975))

# Plot the results

library(ggplot2)

ggplot(cases, aes(x = date)) +
  geom_line(aes(y = Rt_estimates[, 1]), color = "red") +
  geom_point(data = cases, aes(y = Rt_estimates[, 2]), color = "blue") +
  labs(title = "Estimated Time-Varying Reproduction Number (Rt)", x = "", y = "") +
  theme_classic()

ggplot(cases, aes(x = day_of_week)) +
  geom_bar(aes(fill = factor(day_of_week)), position = "dodge", color = "black") +
  labs(title = "Day-of-Week Effects on Reported Cases", fill = "Day of Week")

ggplot(cases, aes(x = date)) +
  geom_line(aes(y = ascertainment_estimates[, 1]), color = "red") +
  geom_point(data = cases, aes(y = ascertainment_estimates[, 2]), color = "blue") +
  labs(title = "Estimated Time-Varying Ascertainment", x = "", y = "") +
  theme_classic()

