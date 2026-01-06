# Load necessary libraries
library(cmdstanr)
library(tidyverse)

# Load the data from cases.csv
cases_df <- read_csv("cases.csv")

# Define the model in Stan
model_code <- "
data {
  int<lower=1> S; // Number of generations to consider
  int<lower=1> T; // Total number of time points
  vector[T] cases; // Reported cases at each time point
  int<lower=1,upper=7> day_of_week[T]; // Day of week (Monday = 1, Sunday = 7)
}

parameters {
  real<lower=0> rt[T]; // Time-varying reproduction number
  real<lower=0, upper=1> delay; // Delay from infection to reporting
  vector[7] day_of_week_effects; // Day-of-week effects (multiplicative)
  real<lower=0, upper=1> ascertained_proportion[T]; // Time-varying ascertainment
}

model {
  for (t in 1:T) {
    rt[t] ~ normal(2, 1); // Prior on Rt with mean 2 and SD 1
    delay ~ normal(7, 3); // Prior on delay with mean 7 and SD 3
    day_of_week_effects ~ normal(0, 0.5); // Prior on day-of-week effects (centered at 0)
    ascertained_proportion[t] ~ beta(1, 2); // Prior on ascertainment proportion (beta distribution with mean 0.5 and SD 0.25)
  }
  
  for (t in 1:T) {
    target += normal_lpdf(cases[t] | rt[t] * delay * day_of_week_effects[day_of_week[t]] * ascertained_proportion[t], 
                          cases[t] / delay / day_of_week_effects[day_of_week[t]] / ascertained_proportion[t]);
  }
}
"

# Fit the model using cmdstanr
fit <- cmdstanr::cmdstan_model(model_code = model_code) %>%
  cmdstanr::fit(data = list(
    S = 30, # Number of generations to consider
    T = nrow(cases_df), # Total number of time points
    cases = cases_df$cases,
    day_of_week = cases_df$day_of_week
  ))

# Extract estimates of Rt, day-of-week effects, and ascertainment over time
rt_estimates <- fit$draws(samples = "posterior_summary") %>%
  as_data_frame() %>%
  dplyr::select(rt)

day_of_week_effects <- fit$draws(samples = "posterior_summary") %>%
  as_data_frame() %>%
  dplyr::select(day_of_week_effects)

ascertainment_proportion <- fit$draws(samples = "posterior_summary") %>%
  as_data_frame() %>%
  dplyr::select(ascertained_proportion)

# Handle the initial infection seeding period appropriately
initial_period <- 14 # Number of days to exclude from estimates (e.g., early pandemic)

rt_estimates <- rt_estimates %>%
  filter(t > initial_period) %>%
  group_by(t = t - initial_period) %>%
  summarise(rt_mean = mean(rt), rt_sd = sd(rt))

day_of_week_effects <- day_of_week_effects %>%
  filter(t > initial_period) %>%
  group_by(day_of_week) %>%
  summarise(day_of_week_effect = mean(day_of_week_effects)) %>%
  mutate(day_of_week_name = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")[day_of_week])

ascertainment_proportion <- ascertainment_proportion %>%
  filter(t > initial_period) %>%
  group_by(t = t - initial_period) %>%
  summarise(ascertainment_mean = mean(ascertained_proportion), ascertainment_sd = sd(ascertained_proportion))

# Plot the results
library(ggplot2)

ggplot(rt_estimates, aes(x = t, y = rt_mean)) +
  geom_line() +
  geom_errorbar(aes(ymin = rt_mean - rt_sd, ymax = rt_mean + rt_sd), width = 0.1) +
  labs(title = "Time-Varying Reproduction Number (Rt)", x = "Days", y = "Rt")

ggplot(day_of_week_effects, aes(x = day_of_week_name, y = day_of_week_effect)) +
  geom_col() +
  labs(title = "Day-of-Week Effects on Reported Cases")

ggplot(ascertainment_proportion, aes(x = t, y = ascertainment_mean)) +
  geom_line() +
  geom_errorbar(aes(ymin = ascertainment_mean - ascertainment_sd, ymax = ascertainment_mean + ascertainment_sd), width = 0.1) +
  labs(title = "Time-Varying Ascertainment Proportion", x = "Days", y = "Ascertainment Proportion")

