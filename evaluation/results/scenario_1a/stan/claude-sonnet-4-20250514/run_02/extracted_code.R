# Load required libraries
library(cmdstanr)
library(dplyr)
library(ggplot2)
library(readr)
library(posterior)
library(bayesplot)

# Load the data
cases_data <- read_csv("cases.csv")

# Data preprocessing
cases_data <- cases_data %>%
  arrange(date) %>%
  mutate(
    day = as.numeric(date - min(date)) + 1,
    cases = pmax(cases, 0)  # Ensure non-negative cases
  )

# Serial interval distribution (assuming COVID-19)
# Mean serial interval ~ 5.2 days, std ~ 2.8 days (discretized)
max_serial_interval <- 20
serial_interval <- dgamma(1:max_serial_interval, shape = 3.5, rate = 0.67)
serial_interval <- serial_interval / sum(serial_interval)

# Prepare data for Stan
stan_data <- list(
  n_days = nrow(cases_data),
  cases = cases_data$cases,
  serial_interval = serial_interval,
  n_serial = length(serial_interval),
  # Priors
  rt_prior_mean = 1.0,
  rt_prior_sd = 0.5,
  rt_walk_sd = 0.1
)

# Compile and fit the Stan model
model <- cmdstan_model("rt_estimation.stan")

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  refresh = 100,
  seed = 12345
)

# Extract Rt estimates
rt_draws <- fit$draws("Rt", format = "matrix")
rt_summary <- summarise_draws(rt_draws)

# Create results dataframe
results <- cases_data %>%
  bind_cols(
    rt_mean = rt_summary$mean,
    rt_lower = rt_summary$q5,
    rt_upper = rt_summary$q95,
    rt_median = rt_summary$median
  )

# Print summary
cat("Rt Estimation Results Summary:\n")
cat("=============================\n")
print(results %>% 
  select(date, cases, rt_mean, rt_lower, rt_upper) %>%
  slice(c(1:5, (n()-4):n())))

# Plot results
p1 <- ggplot(results, aes(x = date)) +
  geom_col(aes(y = cases), alpha = 0.6, fill = "steelblue") +
  labs(title = "Daily COVID-19 Cases", 
       x = "Date", y = "Cases") +
  theme_minimal()

p2 <- ggplot(results, aes(x = date)) +
  geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), 
              alpha = 0.3, fill = "red") +
  geom_line(aes(y = rt_median), color = "red", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       subtitle = "Red line: median estimate, shaded area: 90% credible interval",
       x = "Date", y = "Rt") +
  theme_minimal() +
  ylim(0, max(results$rt_upper) * 1.1)

# Display plots
print(p1)
print(p2)

# Additional diagnostics
cat("\nModel Diagnostics:\n")
cat("==================\n")
fit$cmdstan_diagnose()

# Save results
write_csv(results, "rt_estimates.csv")
cat("\nResults saved to 'rt_estimates.csv'\n")


# Generate sample COVID-19 case data for testing
library(dplyr)
library(readr)

set.seed(123)
n_days <- 100
dates <- seq(as.Date("2023-01-01"), by = "day", length.out = n_days)

# Simulate time-varying Rt
true_rt <- c(
  rep(2.5, 20),           # Initial growth
  seq(2.5, 0.8, length.out = 30),  # Interventions
  rep(0.8, 25),           # Controlled period
  seq(0.8, 1.3, length.out = 25)   # Relaxation
)

# Simple simulation of cases
cases <- numeric(n_days)
cases[1] <- 10

for (t in 2:n_days) {
  # Simple renewal equation approximation
  expected_cases <- max(1, cases[max(1, t-7):max(1, t-1)] %>% 
                       sum() * true_rt[t] / 7)
  cases[t] <- rpois(1, expected_cases)
}

# Create and save sample data
sample_data <- data.frame(
  date = dates,
  cases = cases
)

write_csv(sample_data, "cases.csv")
cat("Sample data saved to 'cases.csv'\n")

