# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Set up cmdstanr (uncomment if needed)
# install_cmdstan()

# Load and prepare data
data <- read_csv("cases.csv")
data <- data %>%
  mutate(date = as.Date(date),
         t = row_number()) %>%
  arrange(date)

# Generation interval (discretized gamma distribution)
# Mean ~5.2 days, SD ~1.72 days (typical COVID-19 values)
gen_mean <- 5.2
gen_sd <- 1.72
gen_shape <- (gen_mean / gen_sd)^2
gen_rate <- gen_mean / gen_sd^2

# Discretize generation interval (truncate at 15 days)
max_gen <- 15
gen_pmf <- diff(pgamma(0:(max_gen), shape = gen_shape, rate = gen_rate))
gen_pmf <- gen_pmf / sum(gen_pmf)  # Normalize

# Delay distribution (infection to reporting)
# Mean ~7 days, representing incubation + reporting delay
delay_mean <- 7
delay_sd <- 3
delay_shape <- (delay_mean / delay_sd)^2
delay_rate <- delay_mean / delay_sd^2

# Discretize delay distribution (truncate at 21 days)
max_delay <- 21
delay_pmf <- diff(pgamma(0:(max_delay), shape = delay_shape, rate = delay_rate))
delay_pmf <- delay_pmf / sum(delay_pmf)  # Normalize

# Prepare data for Stan
T <- nrow(data)
stan_data <- list(
  T = T,
  cases = data$cases,
  day_of_week = data$day_of_week,
  
  # Generation interval
  G = length(gen_pmf),
  gen_pmf = gen_pmf,
  
  # Delay distribution  
  D = length(delay_pmf),
  delay_pmf = delay_pmf,
  
  # Priors
  rt_prior_mean = 1.0,
  rt_prior_sd = 0.5,
  rt_rw_sd = 0.1,
  ascertainment_rw_sd = 0.1
)

print(paste("Data loaded:", T, "days"))
print(paste("Date range:", min(data$date), "to", max(data$date)))


# Compile and fit the model
model <- cmdstan_model("rt_model.stan")

# Fit the model
fit <- model$sample(
  data = stan_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100,
  adapt_delta = 0.95,
  max_treedepth = 12
)

# Check diagnostics
fit$diagnostic_summary()
print(fit$summary(c("phi", "log_rt_init", "logit_ascertainment_init")))

# Extract results
draws <- fit$draws()

# Extract Rt estimates
rt_summary <- fit$summary("rt", ~quantile(.x, c(0.025, 0.25, 0.5, 0.75, 0.975)))
rt_results <- data %>%
  select(date, cases) %>%
  bind_cols(rt_summary %>% select(-variable))

# Extract ascertainment estimates  
ascertainment_summary <- fit$summary("ascertainment", 
                                   ~quantile(.x, c(0.025, 0.25, 0.5, 0.75, 0.975)))
ascertainment_results <- data %>%
  select(date) %>%
  bind_cols(ascertainment_summary %>% select(-variable))

# Extract day-of-week effects
dow_summary <- fit$summary("dow_effect", 
                          ~quantile(.x, c(0.025, 0.5, 0.975)))
dow_results <- data.frame(
  day = c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"),
  dow_summary %>% select(-variable)
)

# Print results
cat("\n=== Day-of-Week Effects ===\n")
print(dow_results)

cat("\n=== Summary Statistics ===\n")
cat("Median Rt:", median(rt_results$`50%`), "\n")
cat("Median Ascertainment:", median(ascertainment_results$`50%`), "\n")


# Plot Rt over time
p1 <- ggplot(rt_results, aes(x = date)) +
  geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "blue") +
  geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "blue") +
  geom_line(aes(y = `50%`), color = "darkblue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       x = "Date", y = "Rt",
       subtitle = "Shaded areas: 50% and 95% credible intervals") +
  theme_minimal()

# Plot ascertainment over time
p2 <- ggplot(ascertainment_results, aes(x = date)) +
  geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "green") +
  geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "green") +
  geom_line(aes(y = `50%`), color = "darkgreen", size = 1) +
  labs(title = "Time-varying Ascertainment Rate",
       x = "Date", y = "Ascertainment Rate",
       subtitle = "Proportion of infections that become reported cases") +
  scale_y_continuous(labels = scales::percent) +
  theme_minimal()

# Plot observed vs expected cases
expected_cases_summary <- fit$summary("expected_cases", ~quantile(.x, c(0.025, 0.5, 0.975)))
comparison_data <- data %>%
  select(date, cases) %>%
  bind_cols(expected_cases_summary %>% select(-variable) %>% rename(exp_median = `50%`, exp_lower = `2.5%`, exp_upper = `97.5%`))

p3 <- ggplot(comparison_data, aes(x = date)) +
  geom_ribbon(aes(ymin = exp_lower, ymax = exp_upper), alpha = 0.3, fill = "orange") +
  geom_line(aes(y = exp_median), color = "darkorange", size = 1) +
  geom_point(aes(y = cases), color = "black", size = 0.5, alpha = 0.7) +
  labs(title = "Observed vs Expected Cases",
       x = "Date", y = "Cases",
       subtitle = "Points: observed cases; Line: expected cases (median + 95% CI)") +
  theme_minimal()

# Day-of-week effects plot
p4 <- ggplot(dow_results, aes(x = factor(day, levels = day), y = `50%`)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = 0.3) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Day-of-Week Effects on Case Reporting",
       x = "Day of Week", y = "Multiplicative Effect",
       subtitle = "Relative to average reporting (dashed line = no effect)") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display plots
print(p1)
print(p2)  
print(p3)
print(p4)

# Save results
write_csv(rt_results, "rt_estimates.csv")
write_csv(ascertainment_results, "ascertainment_estimates.csv") 
write_csv(dow_results, "day_of_week_effects.csv")

cat("Results saved to CSV files.\n")

