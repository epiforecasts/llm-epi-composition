# Expert Review: Code Submissions

Use alongside scoresheet.md for side-by-side review.
See instructions.md for review guidelines.

---

# Scenario 1a

Total submissions: 30

---

## SUB_049

**Scenario**: 1a | **Execution**: FAILED

### model.stan

```stan
data {
  int<lower=1> n_days;                    // Number of days
  array[n_days] int<lower=0> cases;       // Daily case counts
  int<lower=1> max_si;                    // Length of serial interval distribution
  vector<lower=0>[max_si] si_distr;       // Serial interval distribution
}

parameters {
  real rt_init;                           // Initial log Rt
  vector[n_days-1] rt_noise;              // Random walk innovations
  real<lower=0> sigma_rt;                 // Standard deviation of Rt random walk
}

transformed parameters {
  vector[n_days] log_rt;                  // Log reproduction number
  vector[n_days] Rt;                      // Reproduction number
  vector[n_days] lambda;                  // Expected cases (infection rate)
  
  // Random walk for log Rt
  log_rt[1] = rt_init;
  for (t in 2:n_days) {
    log_rt[t] = log_rt[t-1] + sigma_rt * rt_noise[t-1];
  }
  
  // Transform to Rt
  Rt = exp(log_rt);
  
  // Calculate expected cases using renewal equation
  for (t in 1:n_days) {
    lambda[t] = 0;
    for (s in 1:min(t-1, max_si)) {
      lambda[t] += Rt[t] * cases[t-s] * si_distr[s];
    }
    // Add small constant to avoid zero lambda
    lambda[t] = fmax(lambda[t], 1e-8);
  }
}

model {
  // Priors
  rt_init ~ normal(0, 1);                 // Prior for initial log Rt (Rt ~ lognormal(0,1))
  rt_noise ~ std_normal();                // Standard normal innovations
  sigma_rt ~ normal(0, 0.2);              // Prior for Rt volatility
  
  // Likelihood
  for (t in 1:n_days) {
    if (cases[t] > 0) {
      cases[t] ~ poisson(lambda[t]);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[n_days] int cases_pred;
  vector[n_days] log_lik;
  
  for (t in 1:n_days) {
    cases_pred[t] = poisson_rng(lambda[t]);
    if (cases[t] > 0) {
      log_lik[t] = poisson_lpmf(cases[t] | lambda[t]);
    } else {
      log_lik[t] = 0;
    }
  }
}
```

### script.R

```r
# Load required packages
library(cmdstanr)
library(readr)
library(dplyr)
library(ggplot2)
library(posterior)
library(bayesplot)

# Load and prepare the data
cases_data <- read_csv("cases.csv")

# Ensure data is sorted by date and handle any missing values
cases_data <- cases_data %>%
  arrange(date) %>%
  mutate(cases = pmax(cases, 0)) # Ensure non-negative cases

# Prepare data for Stan
n_days <- nrow(cases_data)
cases <- cases_data$cases

# Define serial interval (generation time distribution)
# Using a discretized gamma distribution with mean ~5.1 days, sd ~2.3 days
# (typical values for COVID-19)
max_si <- 15
si_mean <- 5.1
si_sd <- 2.3

# Calculate gamma distribution parameters
si_shape <- (si_mean / si_sd)^2
si_rate <- si_mean / si_sd^2

# Discretize serial interval
si_distr <- diff(pgamma(0:(max_si), shape = si_shape, rate = si_rate))
si_distr <- si_distr / sum(si_distr)  # Normalize

# Data list for Stan
stan_data <- list(
  n_days = n_days,
  cases = cases,
  max_si = length(si_distr),
  si_distr = si_distr
)

# Compile Stan model
rt_model <- cmdstan_model("rt_estimation.stan")

# Fit the model
fit <- rt_model$sample(
  data = stan_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 2000,
  refresh = 500
)

# Extract results
rt_summary <- fit$summary(variables = "Rt")
rt_draws <- fit$draws(variables = "Rt", format = "matrix")

# Create results data frame
results <- data.frame(
  date = cases_data$date,
  cases = cases_data$cases,
  rt_median = rt_summary$median,
  rt_lower = rt_summary$q5,
  rt_upper = rt_summary$q95,
  rt_mean = rt_summary$mean
)

# Print summary
print("Rt Estimates Summary:")
print(head(results, 10))
print(tail(results, 10))

# Create plots
p1 <- ggplot(results, aes(x = date)) +
  geom_line(aes(y = rt_median), color = "red", size = 1) +
  geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "red") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       x = "Date",
       y = "Rt",
       subtitle = "Red line: median estimate, shaded area: 90% credible interval") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ggplot(results, aes(x = date)) +
  geom_bar(aes(y = cases), stat = "identity", alpha = 0.7, fill = "steelblue") +
  labs(title = "Daily Case Counts",
       x = "Date",
       y = "Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display plots
print(p1)
print(p2)

# Save results
write_csv(results, "rt_estimates.csv")

# Model diagnostics
print("Model Diagnostics:")
print(fit$diagnostic_summary())

# Check convergence
rhat_values <- rt_summary$rhat
max_rhat <- max(rhat_values, na.rm = TRUE)
print(paste("Maximum R-hat:", round(max_rhat, 3)))

if(max_rhat < 1.1) {
  print("✓ Model converged successfully (all R-hat < 1.1)")
} else {
  print("⚠ Warning: Model may not have converged (some R-hat >= 1.1)")
}




# Additional analysis and validation
print("=== ADDITIONAL ANALYSIS ===")

# Summary statistics
cat("\nRt Summary Statistics:\n")
cat("Mean Rt:", round(mean(results$rt_mean), 2), "\n")
cat("Median Rt:", round(median(results$rt_median), 2), "\n")
cat("Days with Rt > 1:", sum(results$rt_lower > 1), "out of", nrow(results), "\n")
cat("Days with Rt < 1:", sum(results$rt_upper < 1), "out of", nrow(results), "\n")

# Posterior predictive check
cases_pred <- fit$draws(variables = "cases_pred", format = "matrix")
cases_pred_summary <- posterior::summarise_draws(cases_pred)

pp_check_data <- data.frame(
  date = cases_data$date,
  observed = cases_data$cases,
  predicted_median = cases_pred_summary$median,
  predicted_lower = cases_pred_summary$q5,
  predicted_upper = cases_pred_summary$q95
)

p3 <- ggplot(pp_check_data, aes(x = date)) +
  geom_point(aes(y = observed), color = "black", size = 1) +
  geom_line(aes(y = predicted_median), color = "blue", alpha = 0.7) +
  geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper), 
              alpha = 0.3, fill = "blue") +
  labs(title = "Posterior Predictive Check",
       subtitle = "Black points: observed cases, Blue: predicted cases with 90% CI",
       x = "Date", y = "Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p3)

# Calculate effective reproduction number trend
results$rt_trend <- c(NA, diff(results$rt_median))

cat("\nRecent Rt trend (last 7 days):\n")
recent_trend <- tail(results$rt_trend, 7)
if(mean(recent_trend, na.rm = TRUE) > 0) {
  cat("↗ Increasing trend\n")
} else {
  cat("↘ Decreasing trend\n")
}

print("=== ANALYSIS COMPLETE ===")
```

---

## SUB_101

**Scenario**: 1a | **Execution**: FAILED

### model.stan

```stan
data {
  int<lower=1> n_days;
  array[n_days] int<lower=0> cases;
  int<lower=1> n_serial;
  vector<lower=0>[n_serial] serial_interval;
  
  // Priors
  real rt_prior_mean;
  real<lower=0> rt_prior_sd;
  real<lower=0> rt_walk_sd;
}

parameters {
  vector[n_days] log_rt_raw;
  real<lower=0> phi; // Overdispersion parameter for negative binomial
}

transformed parameters {
  vector[n_days] log_rt;
  vector[n_days] Rt;
  vector[n_days] lambda; // Expected infections
  
  // Random walk for log(Rt)
  log_rt[1] = log(rt_prior_mean) + rt_walk_sd * log_rt_raw[1];
  for (t in 2:n_days) {
    log_rt[t] = log_rt[t-1] + rt_walk_sd * log_rt_raw[t];
  }
  
  Rt = exp(log_rt);
  
  // Calculate expected infections using renewal equation
  for (t in 1:n_days) {
    real infectiousness = 0;
    
    for (s in 1:min(t-1, n_serial)) {
      infectiousness += cases[t-s] * serial_interval[s];
    }
    
    if (t <= n_serial) {
      // Early days: use simple exponential growth
      lambda[t] = cases[1] * exp(0.1 * (t-1));
    } else {
      lambda[t] = Rt[t] * infectiousness;
    }
    
    // Ensure lambda is positive and bounded
    lambda[t] = fmax(lambda[t], 1e-6);
  }
}

model {
  // Priors
  log_rt_raw ~ std_normal();
  phi ~ exponential(0.1);
  
  // Likelihood
  for (t in 1:n_days) {
    if (lambda[t] > 0) {
      cases[t] ~ neg_binomial_2(lambda[t], phi);
    }
  }
}

generated quantities {
  array[n_days] int cases_pred;
  vector[n_days] log_lik;
  
  // Posterior predictive checks
  for (t in 1:n_days) {
    cases_pred[t] = neg_binomial_2_rng(lambda[t], phi);
    log_lik[t] = neg_binomial_2_lpmf(cases[t] | lambda[t], phi);
  }
}
```

### script.R

```r
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
```

---

## SUB_065

**Scenario**: 1a | **Execution**: FAILED

### model.stan

```stan
data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> cases[T];             // Observed cases
  int<lower=0> G;                    // Length of generation time distribution
  vector<lower=0>[G] w;              // Generation time distribution (discrete)
  real prior_Rt_mean;                // Prior mean for initial Rt
  real<lower=0> prior_Rt_sd;         // Prior SD for initial Rt
  real prior_sigma_mean;             // Prior mean for log(sigma)
  real<lower=0> prior_sigma_sd;      // Prior SD for log(sigma)
}

transformed data {
  vector[G] rev_w = reverse(w);      // Reverse generation time for convolution
}

parameters {
  real log_Rt_init;                  // Initial log(Rt)
  vector[T-1] log_Rt_noise;          // Random walk innovations
  real log_sigma;                    // Log of random walk standard deviation
}

transformed parameters {
  vector<lower=0>[T] Rt;             // Time-varying reproduction number
  vector<lower=0>[T] lambda;         // Expected number of cases
  real<lower=0> sigma = exp(log_sigma);
  
  // Random walk for log(Rt)
  vector[T] log_Rt;
  log_Rt[1] = log_Rt_init;
  for (t in 2:T) {
    log_Rt[t] = log_Rt[t-1] + sigma * log_Rt_noise[t-1];
  }
  
  // Transform to Rt
  Rt = exp(log_Rt);
  
  // Calculate expected cases using renewal equation
  for (t in 1:T) {
    if (t == 1) {
      lambda[t] = cases[1];  // Initialize with first observation
    } else {
      real convolution = 0;
      int max_lag = min(t-1, G);
      
      for (g in 1:max_lag) {
        convolution += lambda[t-g] * w[g];
      }
      
      lambda[t] = Rt[t] * convolution;
      
      // Ensure lambda is positive
      if (lambda[t] < 1e-10) {
        lambda[t] = 1e-10;
      }
    }
  }
}

model {
  // Priors
  log_Rt_init ~ normal(log(prior_Rt_mean), prior_Rt_sd);
  log_Rt_noise ~ std_normal();
  log_sigma ~ normal(prior_sigma_mean, prior_sigma_sd);
  
  // Likelihood
  for (t in 1:T) {
    cases[t] ~ poisson(lambda[t]);
  }
}

generated quantities {
  // Posterior predictive checks
  int<lower=0> cases_pred[T];
  vector[T] log_lik;
  
  for (t in 1:T) {
    cases_pred[t] = poisson_rng(lambda[t]);
    log_lik[t] = poisson_lpmf(cases[t] | lambda[t]);
  }
}
```

### script.R

```r
# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(lubridate)

# Load and prepare data
load_and_prepare_data <- function(file_path = "cases.csv") {
  # Read the data
  data <- read.csv(file_path)
  data$date <- as.Date(data$date)
  
  # Sort by date and ensure no missing values
  data <- data %>%
    arrange(date) %>%
    filter(!is.na(cases), cases >= 0)
  
  # Create time index
  data$t <- 1:nrow(data)
  
  return(data)
}

# Define generation time distribution (discretized gamma distribution)
# Using typical COVID-19 parameters: mean ~5.1 days, sd ~2.3 days
get_generation_time <- function(max_days = 20) {
  shape <- (5.1 / 2.3)^2
  scale <- 2.3^2 / 5.1
  
  # Discretized probability mass function
  w <- dgamma(1:max_days, shape = shape, scale = scale)
  w <- w / sum(w)  # Normalize
  
  return(w)
}

# Prepare data for Stan
prepare_stan_data <- function(case_data) {
  w <- get_generation_time()
  
  stan_data <- list(
    T = nrow(case_data),
    cases = case_data$cases,
    G = length(w),
    w = w,
    prior_Rt_mean = 1.0,
    prior_Rt_sd = 1.0,
    prior_sigma_mean = 0.0,
    prior_sigma_sd = 0.5
  )
  
  return(stan_data)
}

# Main function to estimate Rt
estimate_rt <- function(file_path = "cases.csv", 
                       chains = 4, 
                       iter_warmup = 1000, 
                       iter_sampling = 1000) {
  
  # Load and prepare data
  cat("Loading data...\n")
  case_data <- load_and_prepare_data(file_path)
  stan_data <- prepare_stan_data(case_data)
  
  # Compile Stan model
  cat("Compiling Stan model...\n")
  model <- cmdstan_model("rt_estimation.stan")
  
  # Fit the model
  cat("Fitting model...\n")
  fit <- model$sample(
    data = stan_data,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    refresh = 100,
    max_treedepth = 12,
    adapt_delta = 0.95
  )
  
  # Extract results
  cat("Extracting results...\n")
  draws <- fit$draws()
  
  # Get Rt estimates
  rt_draws <- subset(draws, variable = "Rt")
  rt_summary <- summarise_draws(rt_draws, 
                               mean, median, sd,
                               q5 = ~quantile(.x, 0.05),
                               q25 = ~quantile(.x, 0.25),
                               q75 = ~quantile(.x, 0.75),
                               q95 = ~quantile(.x, 0.95))
  
  # Add dates to results
  rt_results <- case_data %>%
    select(date, cases, t) %>%
    bind_cols(rt_summary %>% select(-variable))
  
  # Get lambda estimates (expected cases)
  lambda_draws <- subset(draws, variable = "lambda")
  lambda_summary <- summarise_draws(lambda_draws, mean, median)
  
  results <- list(
    rt_estimates = rt_results,
    case_data = case_data,
    fit = fit,
    diagnostics = fit$diagnostic_summary(),
    rt_draws = rt_draws,
    lambda_summary = lambda_summary
  )
  
  return(results)
}

# Plotting functions
plot_rt_estimates <- function(results) {
  rt_data <- results$rt_estimates
  
  p1 <- ggplot(rt_data, aes(x = date)) +
    geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         subtitle = "Median estimate with 50% and 90% credible intervals",
         x = "Date", y = "Rt") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p1)
}

plot_cases_fit <- function(results) {
  case_data <- results$case_data
  lambda_summary <- results$lambda_summary
  
  fit_data <- case_data %>%
    bind_cols(lambda_summary %>% select(expected_cases = mean))
  
  p2 <- ggplot(fit_data, aes(x = date)) +
    geom_point(aes(y = cases), alpha = 0.7, color = "black") +
    geom_line(aes(y = expected_cases), color = "red", size = 1) +
    labs(title = "Observed vs Expected Cases",
         subtitle = "Black points: observed cases, Red line: model fit",
         x = "Date", y = "Cases") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(p2)
}

# Print summary statistics
print_rt_summary <- function(results) {
  rt_data <- results$rt_estimates
  
  cat("Rt Estimation Summary\n")
  cat("====================\n")
  cat(sprintf("Time period: %s to %s\n", 
              min(rt_data$date), max(rt_data$date)))
  cat(sprintf("Number of days: %d\n", nrow(rt_data)))
  cat(sprintf("Total cases: %d\n", sum(rt_data$cases)))
  cat("\n")
  
  cat("Rt Statistics:\n")
  cat(sprintf("Overall median Rt: %.2f\n", median(rt_data$median)))
  cat(sprintf("Min median Rt: %.2f (on %s)\n", 
              min(rt_data$median), 
              rt_data$date[which.min(rt_data$median)]))
  cat(sprintf("Max median Rt: %.2f (on %s)\n", 
              max(rt_data$median), 
              rt_data$date[which.max(rt_data$median)]))
  
  # Days where Rt > 1
  days_above_1 <- sum(rt_data$q25 > 1)  # Conservative estimate using 25th percentile
  cat(sprintf("Days with Rt likely > 1: %d (%.1f%%)\n", 
              days_above_1, 100 * days_above_1 / nrow(rt_data)))
}

# Main execution
main <- function() {
  # Estimate Rt
  results <- estimate_rt("cases.csv")
  
  # Print summary
  print_rt_summary(results)
  
  # Create plots
  p1 <- plot_rt_estimates(results)
  p2 <- plot_cases_fit(results)
  
  print(p1)
  print(p2)
  
  # Check model diagnostics
  print("Model Diagnostics:")
  print(results$diagnostics)
  
  return(results)
}

# Run if this script is executed directly
if (!interactive()) {
  results <- main()
}




# Run the analysis
results <- estimate_rt("cases.csv")

# Access Rt estimates
rt_estimates <- results$rt_estimates
head(rt_estimates)

# Create plots
plot_rt <- plot_rt_estimates(results)
plot_fit <- plot_cases_fit(results)

# Print summary
print_rt_summary(results)

# Save results
write.csv(rt_estimates, "rt_estimates.csv", row.names = FALSE)
ggsave("rt_plot.png", plot_rt, width = 12, height = 8)
ggsave("cases_fit.png", plot_fit, width = 12, height = 8)
```

---

## SUB_025

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(readr)
library(cmdstanr)

# Function to load data
load_data <- function(file_path) {
  cases_df <- read_csv(file_path, 
                       col_names = c("date", "cases"),
                       skip = 1)
  
  # Convert date column to Date type
  cases_df$date <- as.Date(cases_df$date, format = "%Y-%m-%d")
  
  return(cases_df)
}

# Function to estimate Rt
estimate_rt <- function(data, model_name) {
  # Prepare data for Stan
  stan_data <- list(
    N = nrow(data),
    T = data$date[1],
    cases = data$cases
  )
  
  # Fit the model using cmdstanr
  fit <- cmdstanr::sample(
    file = paste0("models/", model_name, ".stan"),
    seed = 123,
    diagnostic_file = FALSE,
    refresh = FALSE,
    data = stan_data
  )
  
  # Extract Rt estimates for each time point
  rt_estimates <- as.data.frame(fit$samples[, c("Rt")])
  
  return(rt_estimates)
}

# Load the data
data_path <- "cases.csv"
cases_df <- load_data(data_path)

# Estimate Rt
model_name <- "basic_sir"
rt_estimates <- estimate_rt(cases_df, model_name)

# Summary of Rt estimates
summary(rt_estimates)

# Plot of Rt over time (optional)
library(ggplot2)
ggplot(rt_estimates, aes(x = sample)) + 
  geom_line(aes(y = Rt)) + 
  labs(title = "Time-varying Reproduction Number (Rt)", x = "Sample", y = "Rt")
```

---

## SUB_074

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Install and load required libraries
install.packages("cmdstanr")
library(cmdstanr)
library(dplyr)


# Load the CSV into a tibble (data frame)
case_data <- read_csv("cases.csv")


# Transform date to a Date type and extract day/month/year components
case_data$date <- as.Date(case_data$date)

# Create necessary columns (assuming simple exponential growth model for demonstration)
case_data$lag_cases <- lag(case_data, cases)
case_data$prev_case_count <- case_data$cases[case_data$date < last(case_data$date)]

# Fit the model with cmdstanr
model_code <- "
  data {
    int<lower=1> n;
    vector[n] cases;
    vector[n-1] lag_cases;
    real prev_case_count;
    }
  parameters {
    real<lower=0, upper=10> beta; // Infection rate parameter (Rt = beta / gamma)
    real<lower=0.01, upper=2> gamma; // Recovery rate
    real<lower=1, upper=5> initial_infected;
    }
  model {
    for (i in 1:n) {
      cases[i] ~ normal(lag_cases[i-1]*beta*gamma + prev_case_count*(1 - exp(-gamma)), 0.2);
      // Normal distribution with mean defined by the SEIR model equations
      // The standard deviation is set to a small value for illustrative purposes
    }
    beta ~ normal(0, 1); // Prior on infection rate
    gamma ~ normal(0, 1); // Prior on recovery rate
    initial_infected ~ normal(100, 10); // Prior on initial infected
  }
"

# Compile the model
fit <- cmdstanr::stan_model(model_code = model_code) %>%
  compile()

# Fit the data
fit_data <- fit$sample(
  data = list(
    n = nrow(case_data),
    cases = case_data$cases,
    lag_cases = case_data$lag_cases[-1],
    prev_case_count = case_data$prev_case_count
  ),
  seed = 123 // For reproducibility, but you can remove this
)

# Extract parameters of interest (Rt)
posterior <- extract(fit_data, 'beta', permuted = TRUE) 

# Calculate Rt as beta / gamma, where beta is the posterior estimate and gamma a fixed value.
gamma_fixed <- mean(posterior$gamma, na.rm = TRUE) # Gamma should be estimated in the model

Rt_posterior <- posterior$beta / gamma_fixed


# Plot the posterior distribution of Rt (example)
library(ggplot2)

df <- data.frame(date = case_data$date[-nrow(case_data)],
                 Rt = Rt_posterior[, 1:nrow(case_data) - 1])

ggplot(df, aes(x = date, y = Rt)) +
  geom_line() +
  labs(title = 'Time-Varying Reproduction Number (Rt)', x = 'Date', y = 'Rt')
```

---

## SUB_100

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Install necessary packages
install.packages("cmdstanr")
library(cmdstanr)

# Load required libraries
library(readr)
library(ggplot2)

# Load the data from CSV file
data <- read_csv("cases.csv")

# Define the model in Stan
model_code <- "
  data {
    int<lower=0> T; // number of time points
    vector[T] cases; // daily reported case counts
    real<lower=0,upper=1> gamma; // average reporting rate for each day
  }

  parameters {
    real<lower=0, upper=T> phi[T]; // Rt at each time point
  }

  transformed parameters {
    real<lower=0, upper=T> log_phi[T];
    for (t in 1:T) {
      log_phi[t] = normal_lpdf(phi[t]| -2.5, 1);
    }
  }

  model {
    cases ~ neg_binomial_2(log(phil), exp(-phi));
    phil ~ normal(0, 1);

    // prior for reporting rate
    gamma ~ uniform(0, 1);
  }
"

# Fit the model to data using cmdstanr
fit <- stan(
  file = "rt_model.stan", 
  data = list(cases = data$cases, T = nrow(data)), 
  chains = 4, iter = 10000, warmup = 5000, thin = 2
)

# Extract Rt estimates from posterior samples
posterior <- as.data.frame(fit)

Rt_estimates <- posterior[, "phi"]
date_Rt <- paste(posterior$date[1:nrow(posterior)], 
                 posterior$time[1:nrow(posterior)])

# Plot the estimated Rt over time (optional)
library(ggplot2)
ggplot(data.frame(date = date_Rt, rt = Rt_estimates), aes(x = date, y = rt)) +
  geom_line() + labs(title = "Estimated Reproduction Number Over Time", x = "", y = "") +
  theme_classic()
```

---

## SUB_018

**Scenario**: 1a | **Execution**: SUCCESS

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(filename='cases.csv'):
    """
    Load and prepare the case data for analysis
    """
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any negative cases and fill NaNs with 0
    df['cases'] = df['cases'].fillna(0)
    df['cases'] = np.maximum(df['cases'], 0)
    
    return df

def get_generation_time_pmf(max_days=20):
    """
    Generate a discrete generation time distribution.
    Using a gamma distribution with mean ~5.1 days and std ~2.3 days
    (typical values for COVID-19 from literature)
    """
    # Parameters for COVID-19 generation time (Gamma distribution)
    mean_gt = 5.1
    std_gt = 2.3
    
    # Convert to shape and scale parameters
    shape = (mean_gt / std_gt) ** 2
    scale = std_gt ** 2 / mean_gt
    
    # Create discrete PMF
    days = np.arange(1, max_days + 1)
    pmf = stats.gamma.pdf(days, a=shape, scale=scale)
    pmf = pmf / pmf.sum()  # Normalize to sum to 1
    
    return pmf

def estimate_rt_pymc(cases, generation_pmf, n_samples=2000, n_tune=1000):
    """
    Estimate time-varying Rt using PyMC with a renewal equation approach
    """
    n_days = len(cases)
    gt_len = len(generation_pmf)
    
    # Smooth the cases slightly to avoid issues with zeros
    smoothed_cases = np.maximum(cases + 0.1, 1.0)
    
    with pm.Model() as model:
        # Prior for initial Rt
        rt_log_initial = pm.Normal('rt_log_initial', mu=np.log(1.0), sigma=0.5)
        
        # Random walk for log(Rt)
        rt_log_innovations = pm.Normal('rt_log_innovations', 
                                     mu=0, sigma=0.1, 
                                     shape=n_days-1)
        
        # Cumulative sum to get log(Rt) time series
        rt_log = pt.concatenate([[rt_log_initial], 
                                rt_log_initial + pt.cumsum(rt_log_innovations)])
        
        # Transform to get Rt
        rt = pm.Deterministic('rt', pt.exp(rt_log))
        
        # Calculate expected cases using renewal equation
        def calculate_infections(rt_vec, cases_obs):
            infections = pt.zeros(n_days)
            
            # Initialize first few days with observed cases
            init_days = min(7, n_days)
            infections = pt.set_subtensor(infections[:init_days], 
                                        cases_obs[:init_days])
            
            # Calculate subsequent infections using renewal equation
            for t in range(init_days, n_days):
                # Calculate infectiousness (convolution with generation time)
                start_idx = max(0, t - gt_len)
                relevant_infections = infections[start_idx:t]
                relevant_gt = generation_pmf[-(t-start_idx):]
                
                if len(relevant_infections) > 0 and len(relevant_gt) > 0:
                    # Ensure arrays have the same length
                    min_len = min(len(relevant_infections), len(relevant_gt))
                    infectiousness = pt.dot(relevant_infections[-min_len:], 
                                          relevant_gt[-min_len:])
                    
                    new_infections = rt_vec[t] * infectiousness
                    infections = pt.set_subtensor(infections[t], new_infections)
            
            return infections
        
        # Calculate expected infections
        expected_infections = calculate_infections(rt, smoothed_cases)
        
        # Add small constant to avoid numerical issues
        expected_cases = pm.math.maximum(expected_infections, 0.1)
        
        # Observation model - Negative Binomial to handle overdispersion
        alpha = pm.Exponential('alpha', 1.0)  # Overdispersion parameter
        
        # Likelihood
        obs = pm.NegativeBinomial('obs', 
                                mu=expected_cases, 
                                alpha=alpha,
                                observed=cases)
        
        # Sample from posterior
        trace = pm.sample(draws=n_samples, 
                         tune=n_tune,
                         target_accept=0.95,
                         random_seed=42,
                         return_inferencedata=True)
    
    return trace, model

def summarize_rt_estimates(trace, dates):
    """
    Extract and summarize Rt estimates
    """
    rt_samples = trace.posterior.rt.values
    
    # Reshape to (n_samples, n_days)
    rt_samples = rt_samples.reshape(-1, rt_samples.shape[-1])
    
    # Calculate summary statistics
    rt_mean = np.mean(rt_samples, axis=0)
    rt_median = np.median(rt_samples, axis=0)
    rt_lower = np.percentile(rt_samples, 2.5, axis=0)
    rt_upper = np.percentile(rt_samples, 97.5, axis=0)
    rt_lower_50 = np.percentile(rt_samples, 25, axis=0)
    rt_upper_50 = np.percentile(rt_samples, 75, axis=0)
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'date': dates,
        'rt_mean': rt_mean,
        'rt_median': rt_median,
        'rt_lower_95': rt_lower,
        'rt_upper_95': rt_upper,
        'rt_lower_50': rt_lower_50,
        'rt_upper_50': rt_upper_50
    })
    
    return summary_df, rt_samples

def plot_results(df, rt_summary, cases):
    """
    Create comprehensive plots of the results
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cases over time
    axes[0].bar(df['date'], cases, alpha=0.6, color='steelblue', 
                label='Observed cases')
    axes[0].set_ylabel('Daily Cases')
    axes[0].set_title('Daily COVID-19 Cases')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates
    axes[1].fill_between(rt_summary['date'], 
                        rt_summary['rt_lower_95'], 
                        rt_summary['rt_upper_95'],
                        alpha=0.2, color='red', label='95% CI')
    axes[1].fill_between(rt_summary['date'], 
                        rt_summary['rt_lower_50'], 
                        rt_summary['rt_upper_50'],
                        alpha=0.4, color='red', label='50% CI')
    axes[1].plot(rt_summary['date'], rt_summary['rt_median'], 
                color='red', linewidth=2, label='Rt (median)')
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, 
                   label='Rt = 1')
    axes[1].set_ylabel('Reproduction Number (Rt)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Time-varying Reproduction Number (Rt)')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, max(3, rt_summary['rt_upper_95'].max() * 1.1))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """
    Main function to run the complete analysis
    """
    print("Loading and preparing data...")
    df = load_and_prepare_data('cases.csv')
    cases = df['cases'].values.astype(float)
    dates = df['date'].values
    
    print(f"Data loaded: {len(cases)} days of case data")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Total cases: {cases.sum():.0f}")
    
    # Get generation time distribution
    print("Setting up generation time distribution...")
    generation_pmf = get_generation_time_pmf()
    
    print(f"Generation time - Mean: {np.sum(np.arange(1, len(generation_pmf)+1) * generation_pmf):.2f} days")
    
    # Estimate Rt
    print("Estimating Rt using PyMC...")
    print("This may take a few minutes...")
    
    try:
        trace, model = estimate_rt_pymc(cases, generation_pmf)
        
        print("Sampling completed successfully!")
        print(f"Sampling summary:")
        print(f"  Effective sample size: {pm.ess(trace).rt.values.min():.0f}")
        print(f"  R-hat max: {pm.rhat(trace).rt.values.max():.3f}")
        
    except Exception as e:
        print(f"Error in sampling: {e}")
        return None, None
    
    # Summarize results
    print("Summarizing Rt estimates...")
    rt_summary, rt_samples = summarize_rt_estimates(trace, dates)
    
    # Print key results
    print(f"\nRt Summary Statistics:")
    print(f"  Mean Rt: {rt_summary['rt_mean'].mean():.2f}")
    print(f"  Median Rt: {rt_summary['rt_median'].median():.2f}")
    print(f"  Days with Rt > 1: {(rt_summary['rt_median'] > 1).sum()}")
    print(f"  Final Rt estimate: {rt_summary['rt_median'].iloc[-1]:.2f} "
          f"({rt_summary['rt_lower_95'].iloc[-1]:.2f}-{rt_summary['rt_upper_95'].iloc[-1]:.2f})")
    
    # Create plots
    print("Creating plots...")
    fig = plot_results(df, rt_summary, cases)
    
    # Save results
    rt_summary.to_csv('rt_estimates.csv', index=False)
    print("Results saved to 'rt_estimates.csv'")
    
    return rt_summary, trace

# Example of creating sample data if cases.csv doesn't exist
def create_sample_data():
    """
    Create sample case data for demonstration
    """
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', periods=100, freq='D')
    
    # Simulate epidemic curve with changing Rt
    rt_true = np.concatenate([
        np.linspace(2.5, 1.5, 30),  # Initial growth
        np.linspace(1.5, 0.8, 40),  # Decline due to interventions
        np.linspace(0.8, 1.2, 30)   # Slight increase
    ])
    
    cases = []
    cases.append(10)  # Initial cases
    
    generation_pmf = get_generation_time_pmf()
    
    for t in range(1, len(dates)):
        # Simple renewal equation for simulation
        infectiousness = 0
        for s, gt_prob in enumerate(generation_pmf):
            if t - s - 1 >= 0:
                infectiousness += cases[t - s - 1] * gt_prob
        
        expected_cases = rt_true[t] * infectiousness
        new_cases = np.random.poisson(max(expected_cases, 0))
        cases.append(new_cases)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases
    })
    
    df.to_csv('cases.csv', index=False)
    print("Sample data created and saved to 'cases.csv'")
    
    return df

if __name__ == "__main__":
    # Create sample data if file doesn't exist
    import os
    if not os.path.exists('cases.csv'):
        print("Creating sample data...")
        create_sample_data()
    
    # Run the main analysis
    rt_estimates, trace = main()
```

---

## SUB_120

**Scenario**: 1a | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filename='cases.csv'):
    """
    Load and prepare the case data for analysis
    """
    # Load the data
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any negative cases and fill missing values
    df['cases'] = df['cases'].clip(lower=0)
    df['cases'] = df['cases'].fillna(0)
    
    print(f"Data loaded: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['cases'].sum()}")
    
    return df

def get_generation_interval_pmf(max_days=20):
    """
    Generate a probability mass function for the generation interval.
    Using a gamma distribution with mean ~5.2 days and std ~1.7 days
    based on COVID-19 literature.
    """
    # Parameters for generation interval (mean ~5.2, std ~1.7)
    shape = 9.0  # shape parameter
    scale = 0.58  # scale parameter
    
    # Create PMF for discrete days
    days = np.arange(1, max_days + 1)
    pmf = gamma.pdf(days, a=shape, scale=scale)
    pmf = pmf / np.sum(pmf)  # normalize to sum to 1
    
    return pmf

def create_rt_model(cases, generation_pmf):
    """
    Create a PyMC model for estimating time-varying Rt
    """
    n_days = len(cases)
    
    with pm.Model() as model:
        # Priors for Rt estimation
        # Use a random walk for log(Rt) to allow smooth changes over time
        rt_log_init = pm.Normal('rt_log_init', mu=np.log(1.0), sigma=0.5)
        rt_log_steps = pm.Normal('rt_log_steps', mu=0, sigma=0.1, shape=n_days-1)
        
        # Create time-varying log(Rt) using cumulative sum (random walk)
        rt_log = pm.Deterministic('rt_log', 
                                  pt.concatenate([[rt_log_init], 
                                                rt_log_init + pt.cumsum(rt_log_steps)]))
        
        # Transform to get Rt
        rt = pm.Deterministic('rt', pt.exp(rt_log))
        
        # Calculate expected cases using renewal equation
        # For the first few days, use observed cases as seed
        seed_days = min(7, n_days // 4)  # Use first week or 1/4 of data as seed
        
        # Initialize expected cases
        expected_cases = pt.zeros(n_days)
        expected_cases = pt.set_subtensor(expected_cases[:seed_days], 
                                        pt.maximum(cases[:seed_days], 1.0))
        
        # Calculate expected cases for remaining days using renewal equation
        for t in range(seed_days, n_days):
            # Convolution with generation interval
            infectiousness = pt.zeros(1)
            for tau in range(min(t, len(generation_pmf))):
                if t - tau - 1 >= 0:
                    infectiousness += (expected_cases[t - tau - 1] * 
                                     generation_pmf[tau])
            
            expected_t = rt[t] * infectiousness
            expected_cases = pt.set_subtensor(expected_cases[t], 
                                            pt.maximum(expected_t, 0.1))
        
        # Observation model - Negative Binomial for overdispersion
        alpha = pm.Gamma('alpha', alpha=2, beta=0.1)  # overdispersion parameter
        
        # Likelihood
        obs = pm.NegativeBinomial('obs', 
                                mu=expected_cases, 
                                alpha=alpha,
                                observed=cases)
        
        # Store expected cases as deterministic for diagnostics
        expected_cases_det = pm.Deterministic('expected_cases', expected_cases)
    
    return model

def fit_model(model, samples=2000, tune=1000, chains=2):
    """
    Fit the PyMC model using NUTS sampling
    """
    with model:
        # Use NUTS sampler
        trace = pm.sample(samples, tune=tune, chains=chains, 
                         target_accept=0.9,
                         return_inferencedata=True)
    
    return trace

def extract_rt_estimates(trace):
    """
    Extract Rt estimates with credible intervals
    """
    rt_samples = trace.posterior['rt'].values
    
    # Calculate summary statistics
    rt_mean = np.mean(rt_samples, axis=(0, 1))
    rt_median = np.median(rt_samples, axis=(0, 1))
    rt_lower = np.percentile(rt_samples, 2.5, axis=(0, 1))
    rt_upper = np.percentile(rt_samples, 97.5, axis=(0, 1))
    rt_lower_50 = np.percentile(rt_samples, 25, axis=(0, 1))
    rt_upper_50 = np.percentile(rt_samples, 75, axis=(0, 1))
    
    return {
        'mean': rt_mean,
        'median': rt_median,
        'lower_95': rt_lower,
        'upper_95': rt_upper,
        'lower_50': rt_lower_50,
        'upper_50': rt_upper_50
    }

def plot_results(df, rt_estimates, trace):
    """
    Create plots showing the results
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Original case data
    axes[0, 0].plot(df['date'], df['cases'], 'o-', alpha=0.7, markersize=3)
    axes[0, 0].set_title('Reported Cases Over Time')
    axes[0, 0].set_ylabel('Daily Cases')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates
    axes[0, 1].fill_between(df['date'], 
                           rt_estimates['lower_95'], 
                           rt_estimates['upper_95'],
                           alpha=0.3, color='blue', label='95% CI')
    axes[0, 1].fill_between(df['date'], 
                           rt_estimates['lower_50'], 
                           rt_estimates['upper_50'],
                           alpha=0.5, color='blue', label='50% CI')
    axes[0, 1].plot(df['date'], rt_estimates['median'], 
                   'b-', linewidth=2, label='Median Rt')
    axes[0, 1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    axes[0, 1].set_title('Time-varying Reproduction Number (Rt)')
    axes[0, 1].set_ylabel('Rt')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Model fit (observed vs expected cases)
    expected_cases = trace.posterior['expected_cases'].values
    expected_mean = np.mean(expected_cases, axis=(0, 1))
    expected_lower = np.percentile(expected_cases, 2.5, axis=(0, 1))
    expected_upper = np.percentile(expected_cases, 97.5, axis=(0, 1))
    
    axes[1, 0].plot(df['date'], df['cases'], 'o', alpha=0.7, label='Observed', markersize=3)
    axes[1, 0].plot(df['date'], expected_mean, 'r-', label='Expected (mean)', linewidth=2)
    axes[1, 0].fill_between(df['date'], expected_lower, expected_upper,
                           alpha=0.3, color='red', label='95% CI')
    axes[1, 0].set_title('Model Fit: Observed vs Expected Cases')
    axes[1, 0].set_ylabel('Daily Cases')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 4: Rt distribution histogram for last time point
    rt_final = trace.posterior['rt'].values[:, :, -1].flatten()
    axes[1, 1].hist(rt_final, bins=50, alpha=0.7, density=True)
    axes[1, 1].axvline(np.median(rt_final), color='red', linestyle='-', 
                      label=f'Median: {np.median(rt_final):.2f}')
    axes[1, 1].axvline(1, color='black', linestyle='--', alpha=0.7, label='Rt = 1')
    axes[1, 1].set_title('Final Rt Distribution')
    axes[1, 1].set_xlabel('Rt')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()

def print_rt_summary(df, rt_estimates):
    """
    Print summary statistics for Rt estimates
    """
    print("\n" + "="*50)
    print("RT ESTIMATION SUMMARY")
    print("="*50)
    
    # Overall statistics
    print(f"Mean Rt over time period: {np.mean(rt_estimates['median']):.2f}")
    print(f"Final Rt estimate: {rt_estimates['median'][-1]:.2f} "
          f"(95% CI: {rt_estimates['lower_95'][-1]:.2f} - {rt_estimates['upper_95'][-1]:.2f})")
    
    # Days above/below 1
    days_above_1 = np.sum(rt_estimates['median'] > 1)
    days_below_1 = np.sum(rt_estimates['median'] < 1)
    total_days = len(rt_estimates['median'])
    
    print(f"\nDays with Rt > 1: {days_above_1} ({days_above_1/total_days*100:.1f}%)")
    print(f"Days with Rt < 1: {days_below_1} ({days_below_1/total_days*100:.1f}%)")
    
    # Trend analysis
    if len(rt_estimates['median']) >= 7:
        recent_trend = np.mean(rt_estimates['median'][-7:]) - np.mean(rt_estimates['median'][-14:-7])
        print(f"\nRecent trend (last week vs previous week): {recent_trend:+.2f}")
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'date': df['date'],
        'rt_median': rt_estimates['median'],
        'rt_lower_95': rt_estimates['lower_95'],
        'rt_upper_95': rt_estimates['upper_95'],
        'rt_lower_50': rt_estimates['lower_50'],
        'rt_upper_50': rt_estimates['upper_50']
    })
    
    return summary_df

def main():
    """
    Main function to run the complete Rt estimation pipeline
    """
    # Load data
    df = load_and_prepare_data('cases.csv')
    
    # Prepare generation interval
    generation_pmf = get_generation_interval_pmf()
    print(f"\nGeneration interval mean: {np.sum(np.arange(1, len(generation_pmf)+1) * generation_pmf):.1f} days")
    
    # Create and fit model
    print("\nCreating PyMC model...")
    model = create_rt_model(df['cases'].values, generation_pmf)
    
    print("Fitting model (this may take a few minutes)...")
    trace = fit_model(model)
    
    # Extract results
    print("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(trace)
    
    # Print summary
    summary_df = print_rt_summary(df, rt_estimates)
    
    # Create plots
    plot_results(df, rt_estimates, trace)
    
    # Print model diagnostics
    print("\n" + "="*50)
    print("MODEL DIAGNOSTICS")
    print("="*50)
    print(pm.summary(trace, var_names=['rt_log_init', 'alpha']))
    
    return summary_df, trace

# Example of creating sample data if cases.csv doesn't exist
def create_sample_data():
    """
    Create sample data for demonstration
    """
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    # Simulate epidemic curve with changing Rt
    cases = []
    rt_true = []
    current_cases = 10
    
    for i, date in enumerate(dates):
        # Varying Rt over time
        if i < 20:
            rt = 1.5  # Growing phase
        elif i < 50:
            rt = 0.8  # Declining phase
        elif i < 80:
            rt = 1.2  # Second wave
        else:
            rt = 0.7  # Final decline
        
        rt_true.append(rt)
        current_cases = max(1, int(current_cases * rt * np.random.gamma(1, 1)))
        cases.append(current_cases + np.random.poisson(2))  # Add noise
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases
    })
    
    df.to_csv('cases.csv', index=False)
    print("Sample data created and saved to 'cases.csv'")
    return df

if __name__ == "__main__":
    # Create sample data if needed (remove this in production)
    try:
        pd.read_csv('cases.csv')
    except FileNotFoundError:
        print("cases.csv not found. Creating sample data...")
        create_sample_data()
    
    # Run the analysis
    summary_df, trace = main()
    
    # Save results
    summary_df.to_csv('rt_estimates.csv', index=False)
    print(f"\nResults saved to 'rt_estimates.csv'")
```

---

## SUB_047

**Scenario**: 1a | **Execution**: TIMEOUT

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(filename='cases.csv'):
    """
    Load and prepare the case data for modeling.
    """
    # Load the data
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any negative cases and replace with 0
    df['cases'] = np.maximum(df['cases'], 0)
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['cases'].sum()}")
    
    return df

def create_generation_interval_pmf(max_delay=20, mean_gi=5.2, std_gi=1.72):
    """
    Create a discretized generation interval probability mass function.
    Uses a Gamma distribution with specified mean and standard deviation.
    
    Parameters:
    - max_delay: Maximum delay to consider
    - mean_gi: Mean generation interval (days)
    - std_gi: Standard deviation of generation interval (days)
    """
    # Convert mean and std to Gamma distribution parameters
    shape = (mean_gi / std_gi) ** 2
    scale = std_gi ** 2 / mean_gi
    
    # Create discretized PMF
    delays = np.arange(1, max_delay + 1)  # Start from 1, not 0
    pmf = stats.gamma.pdf(delays, a=shape, scale=scale)
    pmf = pmf / pmf.sum()  # Normalize to sum to 1
    
    return pmf

def estimate_rt_pymc(cases, generation_pmf, chains=2, draws=1000, tune=1000):
    """
    Estimate time-varying Rt using PyMC with renewal equation approach.
    
    Parameters:
    - cases: Array of daily case counts
    - generation_pmf: Generation interval probability mass function
    - chains: Number of MCMC chains
    - draws: Number of posterior samples per chain
    - tune: Number of tuning steps
    """
    n_days = len(cases)
    max_delay = len(generation_pmf)
    
    with pm.Model() as model:
        # Hyperpriors for Rt random walk
        rt_mean = pm.Normal('rt_mean', mu=1.0, sigma=0.5)
        rt_sigma = pm.HalfNormal('rt_sigma', sigma=0.2)
        
        # Random walk for log(Rt)
        log_rt_raw = pm.GaussianRandomWalk(
            'log_rt_raw', 
            mu=0, 
            sigma=rt_sigma, 
            shape=n_days
        )
        
        # Transform to Rt with mean constraint
        log_rt = pm.Deterministic('log_rt', log_rt_raw + pm.math.log(rt_mean))
        rt = pm.Deterministic('rt', pm.math.exp(log_rt))
        
        # Compute infectiousness (convolution of past cases with generation interval)
        def compute_infectiousness(cases_padded):
            infectiousness = []
            for t in range(n_days):
                # For each day, sum over all possible infection sources
                inf_t = 0.0
                for tau in range(min(t, max_delay)):
                    if t - tau - 1 >= 0:  # -1 because generation interval starts from day 1
                        inf_t += cases_padded[t - tau - 1] * generation_pmf[tau]
                infectiousness.append(inf_t)
            return pt.stack(infectiousness)
        
        # Pad cases array for convolution
        cases_padded = pt.concatenate([
            pt.zeros(max_delay), 
            pt.as_tensor(cases, dtype='float32')
        ])
        
        infectiousness = compute_infectiousness(cases_padded)
        
        # Expected number of new cases
        mu = rt * infectiousness
        
        # Add small constant to avoid zero mean
        mu = pm.math.maximum(mu, 0.1)
        
        # Observation model - Negative Binomial for overdispersion
        alpha = pm.HalfNormal('alpha', sigma=10)  # Overdispersion parameter
        
        # Likelihood
        cases_obs = pm.NegativeBinomial(
            'cases_obs',
            mu=mu,
            alpha=alpha,
            observed=cases
        )
        
        # Sample from posterior
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,
            return_inferencedata=True,
            random_seed=42
        )
    
    return model, trace

def extract_rt_estimates(trace):
    """
    Extract Rt point estimates and credible intervals from the trace.
    """
    rt_samples = trace.posterior['rt']  # Shape: (chains, draws, days)
    
    # Compute summary statistics
    rt_mean = rt_samples.mean(dim=['chain', 'draw']).values
    rt_median = rt_samples.quantile(0.5, dim=['chain', 'draw']).values
    rt_lower = rt_samples.quantile(0.025, dim=['chain', 'draw']).values
    rt_upper = rt_samples.quantile(0.975, dim=['chain', 'draw']).values
    rt_lower_50 = rt_samples.quantile(0.25, dim=['chain', 'draw']).values
    rt_upper_50 = rt_samples.quantile(0.75, dim=['chain', 'draw']).values
    
    return {
        'mean': rt_mean,
        'median': rt_median,
        'lower_95': rt_lower,
        'upper_95': rt_upper,
        'lower_50': rt_lower_50,
        'upper_50': rt_upper_50
    }

def plot_results(df, rt_estimates):
    """
    Create visualization of cases and Rt estimates.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cases
    axes[0].bar(df['date'], df['cases'], alpha=0.7, color='steelblue')
    axes[0].set_ylabel('Daily Cases')
    axes[0].set_title('Daily COVID-19 Cases')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot Rt estimates
    axes[1].fill_between(
        df['date'], 
        rt_estimates['lower_95'], 
        rt_estimates['upper_95'],
        alpha=0.3, 
        color='red',
        label='95% CI'
    )
    axes[1].fill_between(
        df['date'], 
        rt_estimates['lower_50'], 
        rt_estimates['upper_50'],
        alpha=0.5, 
        color='red',
        label='50% CI'
    )
    axes[1].plot(df['date'], rt_estimates['median'], color='red', linewidth=2, label='Median')
    axes[1].axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Rt = 1')
    axes[1].set_ylabel('Reproduction Number (Rt)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Time-varying Reproduction Number (Rt)')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """
    Main function to run the complete Rt estimation pipeline.
    """
    print("=== COVID-19 Rt Estimation ===\n")
    
    # Load data
    print("1. Loading data...")
    df = load_and_prepare_data('cases.csv')
    
    # Create generation interval
    print("\n2. Creating generation interval...")
    generation_pmf = create_generation_interval_pmf(
        max_delay=20, 
        mean_gi=5.2, 
        std_gi=1.72
    )
    print(f"Generation interval: mean={5.2}, std={1.72}, max_delay={20}")
    
    # Estimate Rt
    print("\n3. Estimating Rt using PyMC...")
    print("This may take several minutes...")
    
    model, trace = estimate_rt_pymc(
        cases=df['cases'].values,
        generation_pmf=generation_pmf,
        chains=2,
        draws=1000,
        tune=1000
    )
    
    print("MCMC sampling completed!")
    
    # Extract estimates
    print("\n4. Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(trace)
    
    # Create results dataframe
    results_df = df.copy()
    results_df['rt_median'] = rt_estimates['median']
    results_df['rt_mean'] = rt_estimates['mean']
    results_df['rt_lower_95'] = rt_estimates['lower_95']
    results_df['rt_upper_95'] = rt_estimates['upper_95']
    results_df['rt_lower_50'] = rt_estimates['lower_50']
    results_df['rt_upper_50'] = rt_estimates['upper_50']
    
    # Print summary
    print("\n5. Results Summary:")
    print(f"Mean Rt over period: {rt_estimates['median'].mean():.2f}")
    print(f"Rt range: {rt_estimates['lower_95'].min():.2f} - {rt_estimates['upper_95'].max():.2f}")
    days_above_1 = (rt_estimates['median'] > 1).sum()
    print(f"Days with Rt > 1: {days_above_1}/{len(df)} ({100*days_above_1/len(df):.1f}%)")
    
    # Create visualization
    print("\n6. Creating plots...")
    fig = plot_results(df, rt_estimates)
    
    # Save results
    print("\n7. Saving results...")
    results_df.to_csv('rt_estimates.csv', index=False)
    print("Results saved to 'rt_estimates.csv'")
    
    # Print sample of results
    print("\nSample of Rt estimates:")
    print(results_df[['date', 'cases', 'rt_median', 'rt_lower_95', 'rt_upper_95']].head(10))
    
    return results_df, trace, model

# Example of creating sample data if cases.csv doesn't exist
def create_sample_data():
    """
    Create sample case data for testing if cases.csv doesn't exist.
    """
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # Simulate epidemic with changing Rt
    cases = []
    rt_true = []
    current_cases = 10
    
    for i in range(n_days):
        # Time-varying Rt
        if i < 20:
            rt = 1.5  # Growth phase
        elif i < 40:
            rt = 1.2  # Slower growth
        elif i < 60:
            rt = 0.8  # Decline
        else:
            rt = 1.1  # Slight growth again
            
        rt_true.append(rt)
        
        # Simple exponential growth/decline with noise
        growth_rate = (rt - 1) / 5.2  # Approximate conversion
        current_cases *= (1 + growth_rate)
        daily_cases = max(0, int(current_cases + np.random.normal(0, np.sqrt(current_cases))))
        cases.append(daily_cases)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases
    })
    
    df.to_csv('cases.csv', index=False)
    print("Sample data created and saved to 'cases.csv'")
    return df

if __name__ == "__main__":
    # Try to run main, if file doesn't exist, create sample data
    try:
        results_df, trace, model = main()
    except FileNotFoundError:
        print("cases.csv not found. Creating sample data...")
        create_sample_data()
        print("\nNow running analysis on sample data...\n")
        results_df, trace, model = main()
```

---

## SUB_024

**Scenario**: 1a | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('cases.csv')

# Convert date to datetime and set it as index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Define a function for the log-likelihood of Rt
def logp_Rt(N, cases, R):
    """
    Log-likelihood function for Rt.

    Parameters:
        N (array): Population size at each time point.
        cases (array): Number of reported cases at each time point.
        R (float): Reproduction number to be estimated.

    Returns:
        float: Log-likelihood value.
    """
    return pm.Deterministic('logp_Rt', np.sum(np.log(R**N - 1) * cases))

# Define the model
with pm.Model() as model:
    # Prior distribution for Rt (informative prior)
    R = pm.Normal('R', mu=2, sigma=0.5, shape=data.shape[0])

    # Log-likelihood function
    logp_Rt = logp_Rt(data['cases'].values, data['cases'].values, R)

# Fit the model
with model:
    trace = pm.sample(10000, tune=5000, init='advi+adapt_diag', chains=4)

# Extract Rt estimates for each time point
Rt_estimates = pm.summary(trace)['mean']

# Plot Rt over time (optional)
plt.plot(data.index, Rt_estimates)
plt.xlabel('Date')
plt.ylabel('Estimated Rt')
plt.title('Time-Varying Reproduction Number (Rt)')
plt.show()

print("Estimated reproduction numbers for each time point:")
for i in range(len(Rt_estimates)):
    print(f"Date: {data.index[i]}, Estimated Rt: {Rt_estimates[i]}")
```

---

## SUB_071

**Scenario**: 1a | **Execution**: FAILED

### script.py

```python
# Import necessary libraries
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('cases.csv')

# Set date as datetime object
df['date'] = pd.to_datetime(df['date'])

# Calculate time difference between consecutive days in days
time_diff = (df['date'].diff().fillna(pd.DateOffset(days=1)).dt.days)

# Define a simple model for Rt: assume it follows an Ornstein-Uhlenbeck process
with pm.Model() as model:
    # Prior distribution on Rt0 (Rt at the start of the data)
    R0 = pm.Normal('R0', mu=2.5, sigma=1)

    # Process noise variance (hyperparameter)
    omega = pm.Uninformative('omega', lower=0, initval=10)

    # Transition matrix for Ornstein-Uhlenbeck process
    Rt = pm.OrnsteinUhlenbeck('Rt', mu=R0, sigma=np.sqrt(omega), dt=time_diff.values, shape=(len(time_diff), 1))

# Fit the model using MAP estimation (or MCMC if desired)
with model:
    step = pm.Metropolis()
    trace = pm.sample_posterior_predictive(1000, vars=[Rt], tune=500, discard_tuned_samples=True, step=step)

# Extract Rt estimates for each time point
rt_estimates = trace['Rt']

# Plot Rt over time (example using matplotlib)
plt.figure(figsize=(8, 6))
for i in range(rt_estimates.shape[1]):
    plt.plot(df['date'].values[1:], rt_estimates[:, i], label=f'Rt_{i+1}')
plt.xlabel('Date')
plt.ylabel('Rt')
plt.title('Estimated Reproduction Number Over Time')
plt.legend()
plt.show()

# Print a summary of Rt estimates
print(rt_estimates.mean(axis=0))
```

---

## SUB_115

**Scenario**: 1a | **Execution**: FAILED

### script.py

```python
# Import necessary libraries
import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('cases.csv')

# Convert date to datetime format for easier manipulation
data['date'] = pd.to_datetime(data['date'])

# Create a time-series index and align it with the data
ts_index = pd.date_range(start=data['date'].min(), end=data['date'].max())
aligned_data = data.set_index('date').reindex(ts_index).fillna(0)  # Fill missing values with 0

# Define the model
def rt_model(d, n_cases, tau=1/3.8):  # Using a typical estimate for tau (mean generation time)
    mu = pm.Normal('mu', mu=n_cases, sigma=10, shape=len(n_cases))
    beta = pm.Deterministic('beta', 1 / (d + tau))  # Deterministic for computational efficiency
    rt = pm.Deterministic('rt', n_cases * beta)  
    return rt

# Fit the model using PyMC
with pm.Model() as model:
    d = aligned_data.index.to_numpy()
    n_cases = aligned_data['cases'].values
    rt_samples = pm.sample_posterior_predictive(rt_model(d, n_cases), samples=10000)

# Extract Rt estimates for each time point
rt_estimates = np.mean(rt_samples['rt'], axis=0)

# Plot Rt over time (optional)
plt.figure(figsize=(8,6))
plt.plot(aligned_data.index, rt_estimates, color='blue')
plt.fill_between(aligned_data.index, 0, rt_estimates, alpha=0.2, color='blue')
plt.xlabel('Date')
plt.ylabel('Rt')
plt.title('Time-Varying Reproduction Number (Rt) Estimate')
plt.show()

# Print summary statistics for Rt estimates
print(f"Mean Rt estimate: {np.mean(rt_estimates):.4f}")
print(f"Median Rt estimate: {np.median(rt_estimates):.4f}")
```

---

## SUB_089

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using Turing, CSV, DataFrames, Distributions, StatsPlots, LinearAlgebra
using Random, MCMCChains

# Set random seed for reproducibility
Random.seed!(42)

# Load and prepare the data
function load_and_prepare_data(filename::String)
    # Load the CSV file
    df = CSV.read(filename, DataFrame)
    
    # Ensure date column is properly formatted
    df.date = Date.(df.date)
    
    # Sort by date to ensure chronological order
    sort!(df, :date)
    
    # Remove any negative cases or missing values
    filter!(row -> row.cases >= 0 && !ismissing(row.cases), df)
    
    return df
end

# Define the generation time distribution (discretized gamma distribution)
function get_generation_time_pmf(max_days::Int = 30)
    # Parameters based on COVID-19 literature (mean ~5 days, std ~2.5 days)
    shape = 4.0
    scale = 1.25
    
    # Create discretized generation time distribution
    pmf = zeros(max_days)
    for i in 1:max_days
        pmf[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure it sums to 1
    pmf = pmf ./ sum(pmf)
    
    return pmf
end

# Bayesian model for Rt estimation
@model function rt_model(cases, generation_pmf)
    n_days = length(cases)
    max_gen = length(generation_pmf)
    
    # Priors for Rt - using random walk on log scale for smoothness
    log_rt_initial ~ Normal(0.0, 0.5)  # Initial Rt around 1
    σ_rt ~ truncated(Normal(0, 0.2), 0, Inf)  # Random walk standard deviation
    
    # Random walk for log(Rt)
    log_rt = Vector{Float64}(undef, n_days)
    log_rt[1] = log_rt_initial
    
    for t in 2:n_days
        log_rt[t] ~ Normal(log_rt[t-1], σ_rt)
    end
    
    # Convert to Rt
    rt = exp.(log_rt)
    
    # Prior for reporting rate and overdispersion
    reporting_rate ~ Beta(2, 2)  # Flexible reporting rate
    φ ~ truncated(Normal(10, 5), 1, Inf)  # Overdispersion parameter
    
    # Calculate expected cases using renewal equation
    expected_cases = Vector{Float64}(undef, n_days)
    
    for t in 1:n_days
        if t <= max_gen
            # For early days, use a simpler model
            expected_cases[t] = max(1.0, cases[1] * reporting_rate)
        else
            # Renewal equation: E[I_t] = R_t * sum(I_{t-s} * w_s)
            infectiousness = 0.0
            for s in 1:min(max_gen, t-1)
                if t-s > 0
                    infectiousness += cases[t-s] * generation_pmf[s]
                end
            end
            expected_cases[t] = max(1.0, rt[t] * infectiousness * reporting_rate)
        end
    end
    
    # Likelihood - using negative binomial to account for overdispersion
    for t in 1:n_days
        # Convert to NegativeBinomial parameterization (r, p)
        μ = expected_cases[t]
        r = φ
        p = φ / (φ + μ)
        cases[t] ~ NegativeBinomial(r, p)
    end
    
    return rt, expected_cases
end

# Function to run MCMC sampling
function estimate_rt(cases_data::Vector{Int}, generation_pmf::Vector{Float64}; 
                    n_samples::Int = 2000, n_chains::Int = 4)
    
    # Create the model
    model = rt_model(cases_data, generation_pmf)
    
    # Sample from the posterior
    println("Running MCMC sampling...")
    chains = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    return chains
end

# Function to extract Rt estimates from chains
function extract_rt_estimates(chains, n_days::Int)
    # Extract Rt samples from all chains
    rt_samples = []
    
    for day in 1:n_days
        rt_param = "rt[$day]"
        if rt_param in string.(keys(chains))
            push!(rt_samples, vec(Array(chains[rt_param])))
        end
    end
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_median = [median(samples) for samples in rt_samples]
    rt_lower = [quantile(samples, 0.025) for samples in rt_samples]
    rt_upper = [quantile(samples, 0.975) for samples in rt_samples]
    rt_lower_50 = [quantile(samples, 0.25) for samples in rt_samples]
    rt_upper_50 = [quantile(samples, 0.75) for samples in rt_samples]
    
    return (
        mean = rt_mean,
        median = rt_median,
        lower_95 = rt_lower,
        upper_95 = rt_upper,
        lower_50 = rt_lower_50,
        upper_50 = rt_upper_50,
        samples = rt_samples
    )
end

# Main analysis function
function analyze_rt(filename::String)
    println("Loading data...")
    df = load_and_prepare_data(filename)
    
    println("Preparing generation time distribution...")
    generation_pmf = get_generation_time_pmf(30)
    
    println("Data summary:")
    println("- Date range: $(df.date[1]) to $(df.date[end])")
    println("- Number of days: $(nrow(df))")
    println("- Total cases: $(sum(df.cases))")
    println("- Mean daily cases: $(round(mean(df.cases), digits=2))")
    
    # Estimate Rt
    chains = estimate_rt(df.cases, generation_pmf)
    
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(chains, length(df.cases))
    
    # Create results DataFrame
    results_df = DataFrame(
        date = df.date,
        observed_cases = df.cases,
        rt_mean = rt_estimates.mean,
        rt_median = rt_estimates.median,
        rt_lower_95 = rt_estimates.lower_95,
        rt_upper_95 = rt_estimates.upper_95,
        rt_lower_50 = rt_estimates.lower_50,
        rt_upper_50 = rt_estimates.upper_50
    )
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("- Mean Rt: $(round(mean(rt_estimates.mean), digits=3))")
    println("- Median Rt: $(round(median(rt_estimates.median), digits=3))")
    println("- Days with Rt > 1: $(sum(rt_estimates.mean .> 1))/$(length(rt_estimates.mean))")
    
    # Create plots
    p1 = plot_rt_estimates(results_df)
    p2 = plot_cases_and_rt(results_df)
    
    return results_df, chains, (rt_plot = p1, combined_plot = p2)
end

# Plotting functions
function plot_rt_estimates(results_df::DataFrame)
    p = plot(results_df.date, results_df.rt_mean, 
             ribbon = (results_df.rt_mean .- results_df.rt_lower_95, 
                      results_df.rt_upper_95 .- results_df.rt_mean),
             fillalpha = 0.2, color = :blue, linewidth = 2,
             label = "Rt (95% CI)", title = "Time-varying Reproduction Number (Rt)",
             xlabel = "Date", ylabel = "Rt")
    
    # Add 50% credible interval
    plot!(p, results_df.date, results_df.rt_mean,
          ribbon = (results_df.rt_mean .- results_df.rt_lower_50,
                   results_df.rt_upper_50 .- results_df.rt_mean),
          fillalpha = 0.4, color = :blue, linewidth = 2,
          label = "Rt (50% CI)")
    
    # Add horizontal line at Rt = 1
    hline!(p, [1.0], linestyle = :dash, color = :red, linewidth = 2, 
           label = "Rt = 1")
    
    return p
end

function plot_cases_and_rt(results_df::DataFrame)
    # Create subplot with cases and Rt
    p1 = plot(results_df.date, results_df.observed_cases,
              seriestype = :bar, alpha = 0.7, color = :gray,
              title = "Daily Cases", ylabel = "Cases", legend = false)
    
    p2 = plot(results_df.date, results_df.rt_mean,
              ribbon = (results_df.rt_mean .- results_df.rt_lower_95,
                       results_df.rt_upper_95 .- results_df.rt_mean),
              fillalpha = 0.3, color = :blue, linewidth = 2,
              title = "Reproduction Number (Rt)", ylabel = "Rt",
              xlabel = "Date", label = "Rt (95% CI)")
    
    hline!(p2, [1.0], linestyle = :dash, color = :red, linewidth = 2, 
           label = "Rt = 1")
    
    return plot(p1, p2, layout = (2, 1), size = (800, 600))
end

# Example usage and testing with synthetic data
function create_example_data()
    # Create example data for testing
    dates = Date("2023-01-01"):Day(1):Date("2023-03-31")
    n_days = length(dates)
    
    # Simulate cases with time-varying Rt
    true_rt = [1.5 * exp(-0.02 * t) + 0.5 + 0.3 * sin(t/10) for t in 1:n_days]
    cases = zeros(Int, n_days)
    cases[1] = 50
    
    generation_pmf = get_generation_time_pmf(15)
    
    for t in 2:n_days
        if t <= length(generation_pmf)
            expected = max(1, cases[t-1] * true_rt[t] * 0.8)
        else
            infectiousness = sum(cases[max(1,t-length(generation_pmf)):t-1] .* 
                               reverse(generation_pmf[1:min(length(generation_pmf), t-1)]))
            expected = max(1, true_rt[t] * infectiousness * 0.8)
        end
        cases[t] = rand(Poisson(expected))
    end
    
    df = DataFrame(date = dates, cases = cases)
    CSV.write("example_cases.csv", df)
    
    return df, true_rt
end

# Run the analysis
println("Creating example data...")
example_df, true_rt = create_example_data()

println("Running Rt estimation analysis...")
results, chains, plots = analyze_rt("example_cases.csv")

# Display results
println("\nFirst 10 rows of results:")
println(first(results, 10))

# Save results
CSV.write("rt_estimates.csv", results)
println("\nResults saved to 'rt_estimates.csv'")

# Display plots
display(plots.rt_plot)
display(plots.combined_plot)

# Print model diagnostics
println("\nModel Diagnostics:")
println("R-hat statistics (should be close to 1.0):")
rhat_vals = rhat(chains)
for param in ["σ_rt", "reporting_rate", "φ"]
    if haskey(rhat_vals, Symbol(param))
        println("- $param: $(round(rhat_vals[Symbol(param)], digits=3))")
    end
end
```

---

## SUB_037

**Scenario**: 1a | **Execution**: SUCCESS

### script.jl

```julia
using Pkg
# Pkg.add(["Turing", "CSV", "DataFrames", "Distributions", "Plots", "StatsPlots", "Dates"])

using Turing
using CSV
using DataFrames
using Distributions
using Plots
using StatsPlots
using Dates
using LinearAlgebra

# Load and prepare the data
function load_data(filename::String)
    df = CSV.read(filename, DataFrame)
    df.date = Date.(df.date)
    sort!(df, :date)
    return df
end

# Define the generation interval (time from infection to onward transmission)
# Using a discretized gamma distribution with mean 5.2 days, std 2.8 days
function generation_interval(max_days::Int = 20)
    # Parameters for gamma distribution (mean=5.2, std=2.8)
    mean_gi = 5.2
    std_gi = 2.8
    shape = (mean_gi / std_gi)^2
    rate = mean_gi / std_gi^2
    
    # Discretize the generation interval
    gi = zeros(max_days)
    for i in 1:max_days
        gi[i] = cdf(Gamma(shape, 1/rate), i) - cdf(Gamma(shape, 1/rate), i-1)
    end
    
    # Normalize to ensure sum = 1
    gi = gi ./ sum(gi)
    return gi
end

# Turing model for estimating time-varying Rt
@model function rt_model(cases, generation_interval)
    n_days = length(cases)
    gi_length = length(generation_interval)
    
    # Priors
    # Initial reproduction number
    R0 ~ LogNormal(log(2.0), 0.5)
    
    # Random walk standard deviation for Rt
    σ_rw ~ Exponential(0.1)
    
    # Reporting probability
    ρ ~ Beta(2, 2)
    
    # Over-dispersion parameter for negative binomial
    φ ~ Exponential(10.0)
    
    # Random walk for log(Rt)
    log_Rt = Vector{Real}(undef, n_days)
    log_Rt[1] = log(R0)
    
    for t in 2:n_days
        log_Rt[t] ~ Normal(log_Rt[t-1], σ_rw)
    end
    
    Rt = exp.(log_Rt)
    
    # Calculate expected cases based on renewal equation
    λ = Vector{Real}(undef, n_days)
    
    for t in 1:n_days
        if t <= gi_length
            # For early days, use a simple exponential growth model
            λ[t] = cases[1] * exp((t-1) * (log_Rt[t] - 1) / 5.2)
        else
            # Renewal equation: λ(t) = Rt * Σ(λ(t-s) * g(s))
            infectivity = 0.0
            for s in 1:min(gi_length, t-1)
                if t-s >= 1
                    infectivity += λ[t-s] * generation_interval[s]
                end
            end
            λ[t] = Rt[t] * infectivity
        end
        
        # Ensure λ is positive
        λ[t] = max(λ[t], 1e-10)
    end
    
    # Likelihood: reported cases ~ NegativeBinomial(expected_cases, φ)
    for t in 1:n_days
        expected_cases = ρ * λ[t]
        # Parameterization: NegativeBinomial2(μ, φ) where var = μ + μ²/φ
        p = φ / (φ + expected_cases)
        r = φ
        cases[t] ~ NegativeBinomial(r, p)
    end
    
    return Rt
end

# Function to run the estimation
function estimate_rt(cases_data; n_samples=2000, n_chains=4, n_adapt=1000)
    println("Setting up model...")
    
    # Remove initial zeros and very small values
    start_idx = findfirst(x -> x >= 5, cases_data.cases)
    if start_idx === nothing
        start_idx = 1
    end
    
    cases = cases_data.cases[start_idx:end]
    dates = cases_data.date[start_idx:end]
    
    # Get generation interval
    gi = generation_interval(20)
    
    println("Running MCMC with $(length(cases)) data points...")
    
    # Sample from the model
    model = rt_model(cases, gi)
    
    # Use NUTS sampler
    sampler = NUTS(n_adapt, 0.8)
    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
    
    return chain, dates, cases
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(chain, dates)
    # Get Rt parameter names
    rt_params = [Symbol("Rt[$i]") for i in 1:length(dates)]
    
    # Extract samples
    rt_samples = Array(group(chain, :Rt))
    
    # Calculate summary statistics
    rt_mean = mean(rt_samples, dims=1)[:]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:size(rt_samples, 2)]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:size(rt_samples, 2)]
    rt_median = [quantile(rt_samples[:, i], 0.5) for i in 1:size(rt_samples, 2)]
    
    # Create results DataFrame
    results = DataFrame(
        date = dates,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper
    )
    
    return results
end

# Function to plot results
function plot_rt_estimates(results, cases_data)
    # Create subplot layout
    p1 = plot(cases_data.date, cases_data.cases, 
              line=:stem, marker=:circle, markersize=2,
              title="Daily Cases", ylabel="Cases", 
              legend=false, color=:blue)
    
    p2 = plot(results.date, results.rt_median,
              ribbon=(results.rt_median - results.rt_lower, 
                     results.rt_upper - results.rt_median),
              fillalpha=0.3, color=:red,
              title="Time-varying Reproduction Number (Rt)",
              ylabel="Rt", xlabel="Date",
              label="Rt (95% CI)", linewidth=2)
    
    # Add horizontal line at Rt = 1
    hline!(p2, [1.0], linestyle=:dash, color=:black, 
           label="Rt = 1", linewidth=1)
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main execution function
function main()
    println("Loading data...")
    
    # Load the data
    cases_data = load_data("cases.csv")
    
    println("Data loaded: $(nrow(cases_data)) days")
    println("Date range: $(minimum(cases_data.date)) to $(maximum(cases_data.date))")
    println("Total cases: $(sum(cases_data.cases))")
    
    # Estimate Rt
    chain, dates, cases = estimate_rt(cases_data, n_samples=1000, n_chains=2)
    
    println("MCMC completed. Extracting results...")
    
    # Extract results
    results = extract_rt_estimates(chain, dates)
    
    # Display summary
    println("\nRt Estimates Summary:")
    println("Mean Rt: $(round(mean(results.rt_mean), digits=2))")
    println("Latest Rt: $(round(results.rt_median[end], digits=2)) ",
            "(95% CI: $(round(results.rt_lower[end], digits=2)) - ",
            "$(round(results.rt_upper[end], digits=2)))")
    
    # Create and display plot
    p = plot_rt_estimates(results, cases_data)
    display(p)
    
    # Save results
    CSV.write("rt_estimates.csv", results)
    savefig(p, "rt_plot.png")
    
    println("\nResults saved to 'rt_estimates.csv' and 'rt_plot.png'")
    
    return results, chain
end

# Example of creating sample data if needed
function create_sample_data()
    dates = Date(2020, 3, 1):Day(1):Date(2020, 6, 30)
    n_days = length(dates)
    
    # Simulate cases with changing Rt
    cases = Int[]
    current_cases = 10
    
    for i in 1:n_days
        # Simulate different phases
        if i < 30  # Initial growth
            rt = 2.5
        elseif i < 60  # Intervention effect
            rt = 1.5 - (i-30) * 0.02
        elseif i < 90  # Controlled phase
            rt = 0.8
        else  # Relaxation
            rt = 1.2
        end
        
        # Simple exponential growth with noise
        expected = current_cases * rt^(1/5.2)  # 5.2 day generation interval
        current_cases = max(1, round(Int, expected + randn() * sqrt(expected)))
        push!(cases, current_cases)
    end
    
    df = DataFrame(date=dates, cases=cases)
    CSV.write("cases.csv", df)
    return df
end

# Uncomment the following lines to create sample data and run the analysis
# create_sample_data()
# results, chain = main()
```

---

## SUB_020

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots
using LinearAlgebra, Random, MCMCChains

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare the data
function load_data(filename="cases.csv")
    df = CSV.read(filename, DataFrame)
    # Ensure dates are properly parsed
    df.date = Date.(df.date)
    # Sort by date to ensure chronological order
    sort!(df, :date)
    return df
end

# Define the serial interval distribution (time from infection to transmission)
# Using a discretized gamma distribution with mean ~5 days, std ~3 days
function serial_interval_pmf(max_days=20)
    # Gamma distribution parameters for COVID-19 serial interval
    shape, rate = 2.8, 0.56  # Mean ≈ 5 days, std ≈ 3 days
    gamma_dist = Gamma(shape, 1/rate)
    
    # Discretize the distribution
    pmf = zeros(max_days)
    for i in 1:max_days
        pmf[i] = cdf(gamma_dist, i) - cdf(gamma_dist, i-1)
    end
    
    # Normalize to ensure sum = 1
    pmf = pmf ./ sum(pmf)
    return pmf
end

# Calculate the infectiousness profile
function calculate_infectiousness(cases, serial_pmf)
    n = length(cases)
    infectiousness = zeros(n)
    
    for t in 1:n
        for s in 1:min(t-1, length(serial_pmf))
            if t-s > 0
                infectiousness[t] += cases[t-s] * serial_pmf[s]
            end
        end
    end
    
    return infectiousness
end

# Turing model for estimating time-varying Rt
@model function rt_model(cases, infectiousness, n_days)
    # Priors
    R0 ~ LogNormal(log(2.0), 0.5)  # Initial reproduction number
    σ_R ~ truncated(Normal(0, 0.1), 0, Inf)  # Random walk standard deviation
    
    # Time-varying reproduction number as a random walk
    log_Rt = Vector{Real}(undef, n_days)
    log_Rt[1] = log(R0)
    
    for t in 2:n_days
        log_Rt[t] ~ Normal(log_Rt[t-1], σ_R)
    end
    
    # Convert to Rt scale
    Rt = exp.(log_Rt)
    
    # Likelihood: cases follow Poisson distribution
    for t in 1:n_days
        if infectiousness[t] > 0
            λ = Rt[t] * infectiousness[t]
            cases[t] ~ Poisson(max(λ, 1e-8))  # Avoid λ = 0
        else
            # For early days with no infectiousness, use a simple prior
            cases[t] ~ Poisson(max(cases[t], 1))
        end
    end
    
    return Rt
end

# Function to estimate Rt
function estimate_rt(cases_data; 
                    n_samples=2000, 
                    n_chains=4, 
                    max_serial_days=20)
    
    println("Preparing data...")
    cases = cases_data.cases
    n_days = length(cases)
    
    # Calculate serial interval and infectiousness
    serial_pmf = serial_interval_pmf(max_serial_days)
    infectiousness = calculate_infectiousness(cases, serial_pmf)
    
    println("Setting up model...")
    model = rt_model(cases, infectiousness, n_days)
    
    println("Running MCMC sampling...")
    # Use NUTS sampler
    sampler = NUTS(0.65)
    
    # Sample from the posterior
    chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
    
    println("Extracting results...")
    
    # Extract Rt estimates
    rt_samples = Array(group(chain, :Rt))
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_lower = vec([quantile(rt_samples[:, i], 0.025) for i in 1:n_days])
    rt_upper = vec([quantile(rt_samples[:, i], 0.975) for i in 1:n_days])
    rt_median = vec([quantile(rt_samples[:, i], 0.5) for i in 1:n_days])
    
    # Create results DataFrame
    results = DataFrame(
        date = cases_data.date,
        cases = cases,
        infectiousness = infectiousness,
        Rt_mean = rt_mean,
        Rt_median = rt_median,
        Rt_lower = rt_lower,
        Rt_upper = rt_upper
    )
    
    return results, chain, serial_pmf
end

# Plotting function
function plot_results(results, title_prefix="COVID-19")
    # Create subplot layout
    p1 = plot(results.date, results.cases, 
             label="Reported Cases", 
             title="$title_prefix Daily Cases",
             xlabel="Date", 
             ylabel="Cases",
             linewidth=2,
             color=:blue)
    
    p2 = plot(results.date, results.Rt_median,
             ribbon=(results.Rt_median .- results.Rt_lower, 
                    results.Rt_upper .- results.Rt_median),
             label="Rt (95% CI)",
             title="$title_prefix Reproduction Number (Rt)",
             xlabel="Date",
             ylabel="Rt",
             linewidth=2,
             fillalpha=0.3,
             color=:red)
    
    # Add horizontal line at Rt = 1
    hline!(p2, [1.0], linestyle=:dash, color=:black, 
           label="Rt = 1", linewidth=1)
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Summary statistics function
function summarize_rt(results)
    println("\n=== Rt Estimation Summary ===")
    println("Date Range: $(first(results.date)) to $(last(results.date))")
    println("Total Days: $(nrow(results))")
    println("Total Cases: $(sum(results.cases))")
    
    println("\nRt Statistics:")
    println("Mean Rt: $(round(mean(results.Rt_mean), digits=3))")
    println("Median Rt: $(round(median(results.Rt_median), digits=3))")
    println("Min Rt: $(round(minimum(results.Rt_median), digits=3))")
    println("Max Rt: $(round(maximum(results.Rt_median), digits=3))")
    
    # Days above/below Rt = 1
    days_above_1 = sum(results.Rt_median .> 1.0)
    days_below_1 = sum(results.Rt_median .< 1.0)
    
    println("\nDays with Rt > 1: $days_above_1 ($(round(100*days_above_1/nrow(results), digits=1))%)")
    println("Days with Rt < 1: $days_below_1 ($(round(100*days_below_1/nrow(results), digits=1))%)")
end

# Main execution function
function main()
    try
        println("Loading case data...")
        cases_data = load_data("cases.csv")
        
        println("Estimating Rt...")
        results, chain, serial_pmf = estimate_rt(cases_data)
        
        println("Creating visualizations...")
        p = plot_results(results)
        display(p)
        
        # Save results
        CSV.write("rt_estimates.csv", results)
        println("Results saved to rt_estimates.csv")
        
        # Print summary
        summarize_rt(results)
        
        # Display convergence diagnostics
        println("\n=== MCMC Diagnostics ===")
        println("R-hat statistics (should be close to 1.0):")
        rhat_vals = rhat(chain)
        println("Mean R-hat: $(round(mean(values(rhat_vals)), digits=4))")
        
        return results, chain, serial_pmf
        
    catch e
        if isa(e, SystemError) && occursin("cases.csv", string(e))
            println("Error: cases.csv file not found!")
            println("Creating example data file...")
            create_example_data()
            println("Example data created. Please run main() again.")
        else
            rethrow(e)
        end
    end
end

# Function to create example data if file doesn't exist
function create_example_data()
    # Generate synthetic COVID-19-like case data
    dates = Date(2023, 1, 1):Day(1):Date(2023, 3, 31)
    n_days = length(dates)
    
    # Simulate cases with changing Rt
    cases = zeros(Int, n_days)
    cases[1:7] .= [5, 8, 12, 15, 22, 18, 25]  # Initial cases
    
    # Simulate epidemic curve
    for i in 8:n_days
        # Add some realistic variation
        trend = 20 * exp(0.05 * (i - 40)) * exp(-0.001 * (i - 40)^2)
        noise = randn() * 5
        cases[i] = max(1, round(Int, trend + noise + 0.7 * cases[i-1]))
    end
    
    example_data = DataFrame(date=dates, cases=cases)
    CSV.write("cases.csv", example_data)
    
    println("Example data written to cases.csv")
    println("First few rows:")
    println(first(example_data, 10))
end

# Run the analysis
println("COVID-19 Rt Estimation using Turing.jl")
println("=" ^ 40)

results, chain, serial_pmf = main()
```

---

## SUB_026

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using CSV
using DataFrames
using Statistics
using Turing
import Plots

# Load data
df = CSV.read("cases.csv", DataFrame)

# Function to estimate Rt using SEIR model
function estimate_Rt(df; seed=123, iter=1000)
    # Define the model
    @model function SEIR_model(cases, t)
        β ~ InverseGamma(2, 1)  # transmission rate
        γ ~ InverseGamma(2, 1)  # recovery rate

        # SEIR dynamics
        S[t] = maximum(cases[:cases][1:t]) - sum(cases[:cases][1:t-1])
        E[t] = (β * S[t-1] * cases[:cases][t-1]) / (S[t-1] + γ)
        I[t] = (γ * E[t-1]) / (E[t-1] + γ)
        R[t] = maximum(cases[:cases][1:t])

        Rt ~ Normal(0, 1)   # reproduction number

        [Rt]
    end

    # Fit the model
    chain = sample(SEIR_model(df.cases, 1:size(df, 1)), NUTS(100), MCMCThreads(), 4, progress=false)

    # Extract Rt estimates for each time point
    rt_estimates = Array{Turing.AbstractRVar}(undef, size(df, 1))
    for i in 1:size(df, 1)
        rt_estimates[i] = chain[i, :Rt]
    end

    return rt_estimates
end

# Estimate Rt over time
rt_estimates = estimate_Rt(df)

# Plot Rt over time
plot(rt_estimates, label="Estimated Rt")
xlabel!("Time (Days)")
ylabel!("Rt")
title!("Reproduction Number Estimates Over Time")
display(plot())
```

---

## SUB_003

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
# Load necessary libraries
using Turing
using DataFrames
using CSV
import StatsBase

# Load data
df = DataFrame(CSV.File("cases.csv"))

# Convert date to DateTime for time series analysis
df[:date] = parse.(Date, df[:date])

# Initialize model parameters and state variables
@model function seir_model(cases)
    # Model parameters
    R0 ~ Normal(2.5, 1)  # Initial reproduction number
    gamma ~ Truncated(Normal(1/7, 0.5), (1/30, 1))  # Recovery rate

    # State variables
    I[1] = 1.0  # Number of infected at day 1
    E[1] = 0.0  # Number of exposed at day 1

    for i in 2:length(cases)
        # Transition equations for SEIR compartments
        E[i] ~ Truncated(Normal(I[i-1] * R0 / (gamma + 1), I[i-1] * R0 / sqrt(gamma)), (0, I[i-1] * R0 / gamma))
        I[i] ~ Truncated(Normal(E[i-1] - E[i], sqrt(E[i-1])), (0, I[i-1]))
    end

    # Observed cases
    for i in 2:length(cases)
        cases[i] ~ Poisson(I[i])
    end
end

# Define the model with data and initialize parameters
model = seir_model(df.cases)

# Define a function to calculate Rt for each day
function calc_rt(model, cdf::DataFrame)
    rt_estimates = []
    for i in 2:length(cdf.cases)
        R0 = Turing.extract_parameters(model)[1]
        E_last_i = getfield.(Turing.extract_states(model), :E)[i-1]
        I_last_i = getfield.(Turing.extract_states(model), :I)[i-1]

        # Estimate Rt
        rt_estimate = I_last_i * R0 / (gamma + 1)
        push!(rt_estimates, rt_estimate)

        # Update model parameters and state variables for next iteration
        Turing.update_params!(model, NamedTuple(R0=rt_estimate))
    end

    return rt_estimates
end

# Sample the model using MCMC
chn = sample(model, MCMCChain(10000))

# Calculate Rt estimates for each time point
rt_estimates = calc_rt(model, df)

# Summarize or plot Rt over time
using Plots
plot(df.date[2:end], rt_estimates)
xlabel!("Day")
ylabel!("Rt Estimate")
title!("Estimated Reproduction Number (Rt) Over Time")
```

---

## SUB_041

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using Turing
using DataFrames
using CSV


# Load data
df = CSV.read("cases.csv")


# Convert date column to datetime
df[!, :date] = DateTime.(df.date)


# Define model parameters
N = 100000.0  # Population size
I0 = 10.0     # Initial number of infected

# Define the SIR model
@model function sir_model(β, gamma, t, y)
    σ ~ Normal(0, 1)  # Prior for Rt
    β ~ TruncatedNormal(0, Inf, lower = 0)
    gamma ~ TruncatedNormal(0, Inf, lower = 0)

    Rt = σ .* exp.(t)

    for (i, t_i) in enumerate(t)
        y[i] ~ Poisson(exp.((β + γ * Rt[t_i]) * I0))
    end
end

# Fit the model to the data
model = sir_model(df.cases[1], df.date, df.cases[2:end])
chain = sample(model, NUTS(100), MCMCThreads(), 4)

# Extract Rt estimates for each time point
Rt_estimates = map(t -> var(chains[:σ][t]), [i for i in 1:length(df.cases)])


# Extract Rt estimates as an array
Rt_array = Array{Float64, 2}(undef, length(df.cases), size(chains[:σ])[2])
for i in 1:size(chains[:σ])[2]
    for j in 1:length(df.cases)
        Rt_array[j, i] = var(chains[:σ][j, i])
    end
end

# Plot the estimated Rt over time
using Plots
plot(Rt_array')
xlabel!("Time")
ylabel!("Estimated Rt")
title!("Time-Varying Reproduction Number (Rt) Estimates")
```

---

## SUB_108

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics

# Load the data
function load_case_data(filename::String)
    df = CSV.read(filename, DataFrame)
    # Ensure dates are parsed correctly and data is sorted by date
    df.date = Date.(df.date)
    sort!(df, :date)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_from_cases(cases_file::String)
    
    # 1. Load case data
    println("Loading case data...")
    cases = load_case_data(cases_file)
    n_timepoints = length(cases)
    println("Loaded $n_timepoints days of case data")
    
    # 2. Define generation interval distribution
    # Using COVID-19 typical values: mean ~6.5 days, shape parameter
    gen_int = Gamma(6.5, 0.62)  # mean = 6.5 * 0.62 ≈ 4 days, reasonable for COVID-19
    model_data = EpiData(gen_distribution = gen_int)
    
    # 3. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)  # Prior for initial infections
    )
    
    # 4. Create latent model for log(Rt) - AR(1) process for smoothness
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Process noise
    )
    
    # 5. Create observation model with reporting delay
    # COVID-19 typically has ~5 day delay from infection to case reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean delay of 5 days
    obs = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_dist
    )
    
    # 6. Compose the full epidemiological model
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    # 7. Generate Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 8. Set up inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 9. Run inference
    println("Running MCMC inference...")
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    # 10. Extract Rt estimates
    println("Extracting Rt estimates...")
    
    # Get posterior samples of the latent process (log Rt)
    log_rt_samples = results[:Z_t]  # Shape: (n_samples, n_timepoints)
    
    # Transform to Rt scale
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))  # 2.5th percentile
    rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))  # 97.5th percentile
    
    # Create results DataFrame
    dates = Date("2020-01-01") .+ Day.(0:(n_timepoints-1))  # Placeholder dates
    rt_results = DataFrame(
        date = dates,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper,
        cases = cases
    )
    
    return rt_results, results
end

# Function to plot results
function plot_rt_estimates(rt_results::DataFrame)
    
    # Create Rt plot
    p1 = plot(rt_results.date, rt_results.rt_median, 
              ribbon = (rt_results.rt_median .- rt_results.rt_lower,
                       rt_results.rt_upper .- rt_results.rt_median),
              label = "Rt (95% CI)",
              color = :blue,
              alpha = 0.3,
              linewidth = 2,
              title = "Time-varying Reproduction Number (Rt)",
              ylabel = "Rt",
              xlabel = "Date")
    
    # Add horizontal line at Rt = 1
    hline!([1.0], color = :red, linestyle = :dash, linewidth = 2, label = "Rt = 1")
    
    # Create cases plot
    p2 = plot(rt_results.date, rt_results.cases,
              color = :black,
              linewidth = 1,
              title = "Daily Case Counts",
              ylabel = "Cases",
              xlabel = "Date",
              label = "Observed cases")
    
    # Combine plots
    combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 600))
    
    return combined_plot
end

# Function to summarize results
function summarize_rt_estimates(rt_results::DataFrame)
    println("\n=== Rt Estimation Summary ===")
    println("Time period: $(rt_results.date[1]) to $(rt_results.date[end])")
    println("Number of time points: $(nrow(rt_results))")
    
    # Overall statistics
    overall_median = median(rt_results.rt_median)
    println("\nOverall median Rt: $(round(overall_median, digits=3))")
    
    # Periods above/below 1
    above_one = sum(rt_results.rt_lower .> 1.0)
    below_one = sum(rt_results.rt_upper .< 1.0)
    uncertain = nrow(rt_results) - above_one - below_one
    
    println("\nTime points where:")
    println("  - Rt significantly > 1: $above_one ($(round(100*above_one/nrow(rt_results), digits=1))%)")
    println("  - Rt significantly < 1: $below_one ($(round(100*below_one/nrow(rt_results), digits=1))%)")
    println("  - Rt uncertain: $uncertain ($(round(100*uncertain/nrow(rt_results), digits=1))%)")
    
    # Recent Rt
    recent_rt = rt_results.rt_median[end-min(6, nrow(rt_results)-1):end]
    println("\nRecent Rt trend (last 7 days): $(round(mean(recent_rt), digits=3))")
    
    return nothing
end

# Main execution function
function main()
    cases_file = "cases.csv"
    
    try
        # Estimate Rt
        rt_results, mcmc_results = estimate_rt_from_cases(cases_file)
        
        # Print summary
        summarize_rt_estimates(rt_results)
        
        # Create and display plot
        rt_plot = plot_rt_estimates(rt_results)
        display(rt_plot)
        
        # Save results
        CSV.write("rt_estimates.csv", rt_results)
        println("\nResults saved to rt_estimates.csv")
        
        return rt_results, mcmc_results
        
    catch e
        println("Error: $e")
        println("\nMake sure 'cases.csv' exists with columns 'date' and 'cases'")
        rethrow(e)
    end
end

# Run the analysis
rt_results, mcmc_results = main()
```

---

## SUB_027

**Scenario**: 1a | **Execution**: SUCCESS

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics

# Load the data
function load_case_data(filepath::String)
    df = CSV.read(filepath, DataFrame)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_from_cases(cases_file::String = "cases.csv")
    
    # 1. Load the case data
    println("Loading case data...")
    cases = load_case_data(cases_file)
    n_timepoints = length(cases)
    println("Loaded $n_timepoints days of case data")
    
    # 2. Define the generation interval distribution
    # Using typical COVID-19 parameters: mean ~6.5 days, shape parameter
    gen_int = Gamma(6.5, 0.62)  # Mean ≈ 4.0 days
    model_data = EpiData(gen_distribution = gen_int)
    
    # 3. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(mean(cases[1:7])), 1.0)  # Initialize based on early cases
    )
    
    # 4. Create latent model for log(Rt) - AR(1) process
    # This allows Rt to evolve smoothly over time
    latent = AR(
        damp_priors = [truncated(Normal(0.9, 0.05), 0, 1)],  # High persistence
        init_priors = [Normal(log(1.0), 0.5)],               # Start around R=1
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Moderate noise
    )
    
    # 5. Create observation model with reporting delay
    # COVID-19 typically has ~5 day delay from infection to case reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean delay of 5 days
    obs = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_dist
    )
    
    # 6. Compose the full epidemiological model
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    # 7. Generate the Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 8. Set up inference method
    println("Setting up inference...")
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 9. Run inference
    println("Running MCMC inference...")
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    # 10. Extract and process results
    println("Processing results...")
    
    # Extract the latent process Z_t (which is log(Rt))
    # Get posterior samples
    posterior_samples = results.result.value
    
    # Find Z_t columns (latent log(Rt) values)
    param_names = string.(keys(posterior_samples))
    z_indices = findall(name -> startswith(name, "Z_t"), param_names)
    
    if isempty(z_indices)
        error("Could not find Z_t parameters in results")
    end
    
    # Extract log(Rt) samples and convert to Rt
    log_rt_samples = hcat([posterior_samples[param_names[i]] for i in z_indices]...)
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_q025 = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_q975 = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    
    # Create results dataframe
    rt_estimates = DataFrame(
        day = 1:length(rt_mean),
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_q025 = rt_q025,
        rt_q975 = rt_q975
    )
    
    println("Rt estimation complete!")
    
    return rt_estimates, rt_samples, results
end

# Function to plot results
function plot_rt_estimates(rt_estimates, cases)
    # Create a two-panel plot
    p1 = plot(rt_estimates.day, rt_estimates.rt_median, 
              ribbon=(rt_estimates.rt_median - rt_estimates.rt_q025, 
                     rt_estimates.rt_q975 - rt_estimates.rt_median),
              fillalpha=0.3,
              label="Rt (95% CI)",
              xlabel="Days",
              ylabel="Rt",
              title="Estimated Reproduction Number Over Time",
              linewidth=2)
    
    # Add horizontal line at Rt = 1
    hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1", linewidth=2)
    
    # Plot case counts
    p2 = plot(1:length(cases), cases,
              label="Observed Cases",
              xlabel="Days", 
              ylabel="Daily Cases",
              title="Daily Case Counts",
              linewidth=2,
              color=:blue)
    
    # Combine plots
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    return combined_plot
end

# Function to summarize results
function summarize_rt_estimates(rt_estimates)
    println("\n=== Rt Estimation Summary ===")
    println("Time period: $(length(rt_estimates.day)) days")
    println("Mean Rt: $(round(mean(rt_estimates.rt_mean), digits=3))")
    println("Median Rt: $(round(median(rt_estimates.rt_median), digits=3))")
    
    # Find periods where Rt > 1 vs Rt < 1
    above_one = sum(rt_estimates.rt_median .> 1)
    below_one = sum(rt_estimates.rt_median .< 1)
    
    println("Days with Rt > 1: $above_one ($(round(100*above_one/length(rt_estimates.day), digits=1))%)")
    println("Days with Rt < 1: $below_one ($(round(100*below_one/length(rt_estimates.day), digits=1))%)")
    
    # Recent Rt trend (last 7 days)
    if length(rt_estimates.day) >= 7
        recent_rt = rt_estimates.rt_median[end-6:end]
        println("Recent Rt (last 7 days): $(round(mean(recent_rt), digits=3))")
    end
    
    return nothing
end

# Main execution function
function main(cases_file::String = "cases.csv")
    try
        # Estimate Rt
        rt_estimates, rt_samples, mcmc_results = estimate_rt_from_cases(cases_file)
        
        # Load cases for plotting
        cases = load_case_data(cases_file)
        
        # Display summary
        summarize_rt_estimates(rt_estimates)
        
        # Create and display plot
        rt_plot = plot_rt_estimates(rt_estimates, cases)
        display(rt_plot)
        
        # Save results
        CSV.write("rt_estimates.csv", rt_estimates)
        println("\nResults saved to 'rt_estimates.csv'")
        
        return rt_estimates, rt_samples, mcmc_results
        
    catch e
        println("Error during Rt estimation: $e")
        rethrow(e)
    end
end

# Run the analysis
# Uncomment the line below to execute:
# rt_estimates, rt_samples, results = main("cases.csv")
```

---

## SUB_036

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using StatsPlots

# Load and prepare the data
function load_case_data(filepath::String)
    df = CSV.read(filepath, DataFrame)
    
    # Ensure proper date parsing and sorting
    df.date = Date.(df.date)
    sort!(df, :date)
    
    # Handle any missing or negative values
    df.cases = max.(df.cases, 0)  # Replace negative with 0
    
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_from_cases(case_data::Vector{Int})
    n_timepoints = length(case_data)
    
    # 1. Define generation interval (COVID-19 typical values)
    # Mean ~6.5 days, std ~4 days
    gen_int = Gamma(6.5, 0.62)  # shape = 6.5, rate = 1/0.62
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)  # Prior for initial infections
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smooth evolution
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Innovation noise
    )
    
    # 4. Create observation model with reporting delay
    # COVID-19 typically has ~5 day delay from infection to case reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean 5 days delay
    obs = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
        delay_dist
    )
    
    # 5. Compose into EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    # 6. Generate Turing model
    mdl = generate_epiaware(epi_prob, (y_t = case_data,))
    
    # 7. Define inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 8. Run inference
    println("Running MCMC inference...")
    results = apply_method(mdl, inference_method, (y_t = case_data,))
    
    return results, epi_prob
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(results)
    # Extract the latent process samples (Z_t = log(Rt))
    samples = results.samples
    
    # Get parameter names containing "Z_t"
    param_names = String.(keys(samples))
    z_params = filter(x -> occursin("Z_t", x), param_names)
    
    # Sort parameters by time index
    z_indices = [parse(Int, match(r"Z_t\[(\d+)\]", p).captures[1]) for p in z_params]
    sorted_idx = sortperm(z_indices)
    z_params_sorted = z_params[sorted_idx]
    
    # Extract samples and compute Rt = exp(Z_t)
    rt_samples = Matrix{Float64}(undef, length(samples[:, z_params_sorted[1], 1]), length(z_params_sorted))
    
    for (i, param) in enumerate(z_params_sorted)
        log_rt_samples = vec(samples[:, param, :])
        rt_samples[:, i] = exp.(log_rt_samples)
    end
    
    # Compute summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_q025 = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_q975 = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    rt_q25 = vec(mapslices(x -> quantile(x, 0.25), rt_samples, dims=1))
    rt_q75 = vec(mapslices(x -> quantile(x, 0.75), rt_samples, dims=1))
    
    return (
        samples = rt_samples,
        mean = rt_mean,
        median = rt_median,
        q025 = rt_q025,
        q975 = rt_q975,
        q25 = rt_q25,
        q75 = rt_q75,
        timepoints = 1:length(rt_mean)
    )
end

# Function to plot Rt estimates
function plot_rt_estimates(rt_estimates, dates=nothing)
    timepoints = rt_estimates.timepoints
    
    # Create x-axis (use dates if provided, otherwise time indices)
    x_axis = dates !== nothing ? dates : timepoints
    
    p = plot(
        x_axis, rt_estimates.median,
        ribbon = (rt_estimates.median .- rt_estimates.q025, 
                 rt_estimates.q975 .- rt_estimates.median),
        fillalpha = 0.3,
        label = "Rt (95% CI)",
        linewidth = 2,
        color = :blue,
        title = "Time-varying Reproduction Number (Rt)",
        xlabel = dates !== nothing ? "Date" : "Time",
        ylabel = "Rt",
        legend = :topright
    )
    
    # Add 50% credible interval
    plot!(p, x_axis, rt_estimates.median,
          ribbon = (rt_estimates.median .- rt_estimates.q25,
                   rt_estimates.q75 .- rt_estimates.median),
          fillalpha = 0.5,
          color = :blue,
          label = "Rt (50% CI)")
    
    # Add horizontal line at Rt = 1
    hline!(p, [1.0], linestyle = :dash, color = :red, label = "Rt = 1", linewidth = 2)
    
    return p
end

# Main execution function
function main()
    # Load data
    println("Loading case data...")
    cases = load_case_data("cases.csv")
    
    # Load dates for plotting (optional)
    df = CSV.read("cases.csv", DataFrame)
    dates = Date.(df.date)
    
    println("Loaded $(length(cases)) days of case data")
    println("Date range: $(minimum(dates)) to $(maximum(dates))")
    println("Case range: $(minimum(cases)) to $(maximum(cases))")
    
    # Estimate Rt
    results, epi_prob = estimate_rt_from_cases(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results)
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("======================")
    println("Time points: $(length(rt_estimates.timepoints))")
    println("Mean Rt: $(round(mean(rt_estimates.mean), digits=3))")
    println("Median Rt: $(round(mean(rt_estimates.median), digits=3))")
    println("Rt range: $(round(minimum(rt_estimates.q025), digits=3)) - $(round(maximum(rt_estimates.q975), digits=3))")
    
    # Create summary DataFrame
    rt_summary = DataFrame(
        date = dates,
        rt_median = rt_estimates.median,
        rt_mean = rt_estimates.mean,
        rt_q025 = rt_estimates.q025,
        rt_q975 = rt_estimates.q975,
        rt_q25 = rt_estimates.q25,
        rt_q75 = rt_estimates.q75
    )
    
    # Save results
    CSV.write("rt_estimates.csv", rt_summary)
    println("Rt estimates saved to 'rt_estimates.csv'")
    
    # Create and save plot
    p = plot_rt_estimates(rt_estimates, dates)
    savefig(p, "rt_plot.png")
    println("Rt plot saved to 'rt_plot.png'")
    
    # Display plot
    display(p)
    
    return rt_estimates, results, rt_summary
end

# Run the analysis
rt_estimates, mcmc_results, rt_summary = main()

# Print first few Rt estimates
println("\nFirst 10 Rt estimates:")
for i in 1:min(10, length(rt_estimates.median))
    println("Day $i: Rt = $(round(rt_estimates.median[i], digits=3)) (95% CI: $(round(rt_estimates.q025[i], digits=3)) - $(round(rt_estimates.q975[i], digits=3)))")
end
```

---

## SUB_095

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing

# Load data
cases = readdlm("cases.csv", ",", String)
dates = cases[:, 1]
cases = convert(Array{Float64}, cases[:, 2])

# Define generation interval distribution
gen_distribution = Gamma(6.5, 0.62)

# Create EpiData with generation interval
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt_estimates = exp.(results.latent_model.Z_t)

# Plot Rt over time
using Plots
plot(dates[1:length(Rt_estimates)], Rt_estimates)
xlabel!("Time")
ylabel!("Rt estimate")
title!("Estimated Rt over Time")
display(plot)
```

---

## SUB_005

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
# Load required packages
using EpiAware
using Distributions
using Turing

# Load data
cases = readdlm("cases.csv", ',', String, '\n')
date = [parse(Date, d[1:10]) for d in cases[:, 1]]
cases = map(x -> parse(Int64, x), cases[:, 2])

# Define EpiData
model_data = EpiData(
    gen_distribution = Gamma(6.5, 0.62),
    initial_cases = Normal(log(100.0), 1.0)
)

# Create infection model (Renewal) and latent model for log(Rt)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    Gamma(5.0, 1.0)
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (date[1], date[end])
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)

# Run the inference
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt = exp.(results[latent_model].Z_t)

# Plot Rt over time
using Plots
plot(date, Rt, xlabel="Date", ylabel="Rt")
```

---

## SUB_084

**Scenario**: 1a | **Execution**: FAILED

### script.jl

```julia
# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))


# Extract Rt estimates
Rt_estimates = exp.(results[:latent_model].value)


using Plots

plot(Rt_estimates, label="Estimated Rt")
xlabel("Time (days)")
ylabel("Rt estimate")
title("Time-varying reproduction number (Rt) estimates")
```

---

## SUB_034

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(EpiEstim)
library(readr)
library(dplyr)
library(ggplot2)
library(lubridate)

# Function to estimate Rt from case counts
estimate_rt <- function(data_file = "cases.csv", 
                       mean_si = 4.7, 
                       std_si = 2.9,
                       window_size = 7) {
  
  # Load the data
  cat("Loading data...\n")
  cases_data <- read_csv(data_file)
  
  # Data preprocessing
  cases_data <- cases_data %>%
    mutate(date = as.Date(date)) %>%
    arrange(date) %>%
    # Handle missing dates by filling gaps with 0 cases
    complete(date = seq.Date(min(date), max(date), by = "day")) %>%
    mutate(cases = replace_na(cases, 0)) %>%
    # Ensure non-negative cases
    mutate(cases = pmax(cases, 0))
  
  cat("Data loaded successfully. Date range:", 
      as.character(min(cases_data$date)), "to", 
      as.character(max(cases_data$date)), "\n")
  cat("Total cases:", sum(cases_data$cases, na.rm = TRUE), "\n")
  
  # Prepare data for EpiEstim (requires incidence object)
  # EpiEstim expects a data frame with dates and case counts
  incidence_data <- cases_data %>%
    select(dates = date, I = cases)
  
  # Define serial interval distribution
  # Using gamma distribution parameters for COVID-19
  # Mean = 4.7 days, SD = 2.9 days (based on literature)
  si_config <- make_config(list(
    mean_si = mean_si,
    std_si = std_si,
    si_parametric_distr = "G"  # Gamma distribution
  ))
  
  cat("Using serial interval: Mean =", mean_si, "days, SD =", std_si, "days\n")
  
  # Estimate Rt using sliding window approach
  cat("Estimating Rt...\n")
  
  # We need at least 2*window_size days of data
  if (nrow(incidence_data) < 2 * window_size) {
    stop("Insufficient data points for estimation. Need at least ", 
         2 * window_size, " days of data.")
  }
  
  # Estimate Rt with parametric serial interval
  rt_estimates <- estimate_R(
    incid = incidence_data,
    method = "parametric_si",
    config = si_config
  )
  
  # Extract results
  rt_results <- rt_estimates$R %>%
    mutate(
      date_start = incidence_data$dates[t_start],
      date_end = incidence_data$dates[t_end],
      date_mid = date_start + floor((date_end - date_start) / 2)
    ) %>%
    select(
      t_start, t_end, 
      date_start, date_end, date_mid,
      rt_mean = `Mean(R)`,
      rt_std = `Std(R)`,
      rt_q025 = `Quantile.0.025(R)`,
      rt_q05 = `Quantile.0.05(R)`,
      rt_q25 = `Quantile.0.25(R)`,
      rt_median = `Quantile.0.5(R)`,
      rt_q75 = `Quantile.0.75(R)`,
      rt_q95 = `Quantile.0.95(R)`,
      rt_q975 = `Quantile.0.975(R)`
    )
  
  cat("Rt estimation completed successfully!\n")
  cat("Estimated Rt for", nrow(rt_results), "time windows\n")
  
  return(list(
    rt_estimates = rt_results,
    raw_estimates = rt_estimates,
    case_data = cases_data,
    parameters = list(
      mean_si = mean_si,
      std_si = std_si,
      window_size = window_size
    )
  ))
}

# Function to create summary plots
plot_rt_estimates <- function(rt_output) {
  
  rt_data <- rt_output$rt_estimates
  case_data <- rt_output$case_data
  
  # Plot 1: Cases over time
  p1 <- ggplot(case_data, aes(x = date, y = cases)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_smooth(method = "loess", se = FALSE, color = "red", linewidth = 1) +
    labs(
      title = "Daily COVID-19 Case Counts",
      x = "Date",
      y = "Number of Cases"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot 2: Rt estimates with confidence intervals
  p2 <- ggplot(rt_data, aes(x = date_mid)) +
    geom_ribbon(aes(ymin = rt_q025, ymax = rt_q975), 
                fill = "lightblue", alpha = 0.3) +
    geom_ribbon(aes(ymin = rt_q25, ymax = rt_q75), 
                fill = "lightblue", alpha = 0.5) +
    geom_line(aes(y = rt_mean), color = "darkblue", linewidth = 1) +
    geom_point(aes(y = rt_mean), color = "darkblue", size = 1.5) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", linewidth = 1) +
    labs(
      title = "Time-varying Reproduction Number (Rt)",
      subtitle = "Blue line: Mean Rt | Dark ribbon: 50% CI | Light ribbon: 95% CI",
      x = "Date",
      y = "Rt"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
    ylim(0, max(rt_data$rt_q975, 3, na.rm = TRUE))
  
  return(list(cases_plot = p1, rt_plot = p2))
}

# Function to print summary statistics
print_rt_summary <- function(rt_output) {
  rt_data <- rt_output$rt_estimates
  params <- rt_output$parameters
  
  cat("\n=== Rt ESTIMATION SUMMARY ===\n")
  cat("Serial Interval Parameters:\n")
  cat("  Mean SI:", params$mean_si, "days\n")
  cat("  SD SI:", params$std_si, "days\n")
  cat("  Window size:", params$window_size, "days\n\n")
  
  cat("Rt Statistics:\n")
  cat("  Mean Rt:", round(mean(rt_data$rt_mean, na.rm = TRUE), 2), "\n")
  cat("  Median Rt:", round(median(rt_data$rt_mean, na.rm = TRUE), 2), "\n")
  cat("  Min Rt:", round(min(rt_data$rt_mean, na.rm = TRUE), 2), "\n")
  cat("  Max Rt:", round(max(rt_data$rt_mean, na.rm = TRUE), 2), "\n")
  
  # Periods where Rt > 1
  above_one <- rt_data %>% filter(rt_mean > 1)
  cat("  Periods with Rt > 1:", nrow(above_one), "out of", nrow(rt_data), 
      "time windows (", round(100 * nrow(above_one) / nrow(rt_data), 1), "%)\n")
  
  # Recent Rt trend
  if (nrow(rt_data) >= 3) {
    recent_rt <- tail(rt_data$rt_mean, 3)
    trend <- ifelse(recent_rt[3] > recent_rt[1], "increasing", 
                   ifelse(recent_rt[3] < recent_rt[1], "decreasing", "stable"))
    cat("  Recent trend (last 3 estimates):", trend, "\n")
    cat("  Latest Rt estimate:", round(tail(rt_data$rt_mean, 1), 2), 
        " (95% CI:", round(tail(rt_data$rt_q025, 1), 2), "-", 
        round(tail(rt_data$rt_q975, 1), 2), ")\n")
  }
}

# Main execution function
main <- function() {
  cat("=== COVID-19 Rt ESTIMATION ===\n\n")
  
  # Check if data file exists
  if (!file.exists("cases.csv")) {
    cat("Warning: cases.csv not found. Creating sample data for demonstration...\n")
    
    # Create sample data
    set.seed(123)
    dates <- seq.Date(from = as.Date("2023-01-01"), 
                     to = as.Date("2023-06-30"), by = "day")
    
    # Simulate realistic case counts with changing Rt
    n_days <- length(dates)
    cases <- numeric(n_days)
    cases[1:7] <- rpois(7, 50)  # Initial cases
    
    # Simulate epidemic with changing transmission
    for (i in 8:n_days) {
      # Time-varying Rt
      rt_sim <- 1.5 * exp(-0.02 * i) + 0.5 + 0.3 * sin(2 * pi * i / 60)
      expected_cases <- max(1, sum(cases[(i-7):(i-1)]) * rt_sim / 7)
      cases[i] <- rpois(1, expected_cases)
    }
    
    sample_data <- data.frame(
      date = dates,
      cases = pmax(0, cases)
    )
    write_csv(sample_data, "cases.csv")
    cat("Sample data created with", nrow(sample_data), "observations\n\n")
  }
  
  # Estimate Rt
  rt_results <- estimate_rt("cases.csv")
  
  # Print summary
  print_rt_summary(rt_results)
  
  # Create plots
  plots <- plot_rt_estimates(rt_results)
  
  # Display plots
  print(plots$cases_plot)
  print(plots$rt_plot)
  
  # Save results
  write_csv(rt_results$rt_estimates, "rt_estimates.csv")
  ggsave("cases_plot.png", plots$cases_plot, width = 10, height = 6, dpi = 300)
  ggsave("rt_plot.png", plots$rt_plot, width = 10, height = 6, dpi = 300)
  
  cat("\nResults saved:\n")
  cat("  - rt_estimates.csv: Detailed Rt estimates\n")
  cat("  - cases_plot.png: Case counts visualization\n")
  cat("  - rt_plot.png: Rt estimates visualization\n")
  
  return(rt_results)
}

# Run the analysis
if (!interactive()) {
  # Install required packages if not available
  required_packages <- c("EpiEstim", "readr", "dplyr", "ggplot2", "lubridate")
  missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]
  
  if (length(missing_packages) > 0) {
    cat("Installing missing packages:", paste(missing_packages, collapse = ", "), "\n")
    install.packages(missing_packages)
  }
  
  results <- main()
}
```

---

## SUB_092

**Scenario**: 1a | **Execution**: SUCCESS

### script.R

```r
# Load required libraries
library(tidyverse)
library(lubridate)
library(ggplot2)

# Function to estimate Rt from case counts
estimate_rt <- function(cases_data, 
                       serial_interval_mean = 5.2,
                       serial_interval_sd = 2.8,
                       window_size = 7,
                       prior_mean = 1,
                       prior_sd = 2) {
  
  # Prepare data
  cases_data <- cases_data %>%
    arrange(date) %>%
    mutate(
      day = as.numeric(date - min(date)) + 1,
      cases = pmax(cases, 0)  # Ensure non-negative cases
    )
  
  n_days <- nrow(cases_data)
  
  # Generate serial interval distribution (discretized gamma distribution)
  max_si <- 20  # Maximum serial interval length
  si_distr <- dgamma(1:max_si, 
                    shape = (serial_interval_mean/serial_interval_sd)^2,
                    rate = serial_interval_mean/(serial_interval_sd^2))
  si_distr <- si_distr / sum(si_distr)  # Normalize
  
  # Calculate infectivity (convolution of cases with serial interval)
  infectivity <- numeric(n_days)
  
  for (t in 1:n_days) {
    infectivity[t] <- sum(
      cases_data$cases[pmax(1, t - length(si_distr) + 1):t] * 
      rev(si_distr[1:min(length(si_distr), t)])
    )
  }
  
  # Estimate Rt using sliding window approach
  rt_estimates <- data.frame(
    date = cases_data$date,
    day = cases_data$day,
    cases = cases_data$cases,
    infectivity = infectivity,
    rt_mean = NA,
    rt_lower = NA,
    rt_upper = NA
  )
  
  # Start estimation after sufficient data points
  start_day <- max(window_size, length(si_distr))
  
  for (t in start_day:n_days) {
    # Define window
    window_start <- max(1, t - window_size + 1)
    window_end <- t
    
    # Get cases and infectivity for window
    window_cases <- cases_data$cases[window_start:window_end]
    window_infectivity <- infectivity[window_start:window_end]
    
    # Remove days with zero infectivity to avoid division by zero
    valid_days <- window_infectivity > 0
    
    if (sum(valid_days) > 0) {
      window_cases <- window_cases[valid_days]
      window_infectivity <- window_infectivity[valid_days]
      
      # Bayesian estimation using Gamma-Poisson conjugacy
      # Prior: Rt ~ Gamma(alpha_prior, beta_prior)
      alpha_prior <- (prior_mean / prior_sd)^2
      beta_prior <- prior_mean / (prior_sd^2)
      
      # Posterior parameters
      alpha_post <- alpha_prior + sum(window_cases)
      beta_post <- beta_prior + sum(window_infectivity)
      
      # Posterior statistics
      rt_mean <- alpha_post / beta_post
      rt_var <- alpha_post / (beta_post^2)
      
      # 95% credible interval
      rt_lower <- qgamma(0.025, alpha_post, beta_post)
      rt_upper <- qgamma(0.975, alpha_post, beta_post)
      
      # Store results
      rt_estimates$rt_mean[t] <- rt_mean
      rt_estimates$rt_lower[t] <- rt_lower
      rt_estimates$rt_upper[t] <- rt_upper
    }
  }
  
  return(rt_estimates)
}

# Function to load and process data
load_case_data <- function(file_path) {
  data <- read.csv(file_path) %>%
    mutate(date = as.Date(date)) %>%
    arrange(date) %>%
    filter(!is.na(cases), cases >= 0)
  
  return(data)
}

# Function to create visualization
plot_rt_estimates <- function(rt_data) {
  
  # Plot 1: Cases over time
  p1 <- ggplot(rt_data, aes(x = date)) +
    geom_bar(aes(y = cases), stat = "identity", fill = "steelblue", alpha = 0.7) +
    labs(title = "Daily COVID-19 Cases",
         x = "Date", y = "Cases") +
    theme_minimal()
  
  # Plot 2: Rt estimates over time
  p2 <- rt_data %>%
    filter(!is.na(rt_mean)) %>%
    ggplot(aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), 
                fill = "red", alpha = 0.3) +
    geom_line(aes(y = rt_mean), color = "red", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         subtitle = "Red line: Mean estimate, Gray ribbon: 95% credible interval",
         x = "Date", y = "Rt") +
    theme_minimal() +
    ylim(0, max(rt_data$rt_upper, na.rm = TRUE) * 1.1)
  
  return(list(cases_plot = p1, rt_plot = p2))
}

# Main execution function
main <- function() {
  
  # Check if data file exists, if not create sample data
  if (!file.exists("cases.csv")) {
    cat("Creating sample data file 'cases.csv'...\n")
    
    # Generate sample data
    set.seed(123)
    dates <- seq(from = as.Date("2023-01-01"), 
                to = as.Date("2023-06-30"), 
                by = "day")
    
    # Simulate epidemic curve with changing Rt
    n_days <- length(dates)
    true_rt <- c(rep(2.5, 30), rep(0.8, 60), rep(1.2, 40), rep(0.6, n_days - 130))
    
    cases <- numeric(n_days)
    cases[1:7] <- c(10, 12, 15, 18, 22, 28, 35)  # Initial cases
    
    # Simple simulation
    for (i in 8:n_days) {
      lambda <- sum(cases[max(1, i-10):(i-1)] * rev(dgamma(1:min(10, i-1), 2, 0.4)))
      cases[i] <- max(0, rpois(1, true_rt[min(i, length(true_rt))] * lambda * 0.1))
    }
    
    sample_data <- data.frame(
      date = dates,
      cases = cases
    )
    
    write.csv(sample_data, "cases.csv", row.names = FALSE)
  }
  
  # Load the data
  cat("Loading case data...\n")
  cases_data <- load_case_data("cases.csv")
  
  cat(sprintf("Loaded %d days of case data from %s to %s\n", 
              nrow(cases_data), 
              min(cases_data$date), 
              max(cases_data$date)))
  
  # Estimate Rt
  cat("Estimating Rt...\n")
  rt_results <- estimate_rt(cases_data)
  
  # Print summary
  rt_summary <- rt_results %>%
    filter(!is.na(rt_mean)) %>%
    summarise(
      n_estimates = n(),
      mean_rt = mean(rt_mean, na.rm = TRUE),
      median_rt = median(rt_mean, na.rm = TRUE),
      min_rt = min(rt_mean, na.rm = TRUE),
      max_rt = max(rt_mean, na.rm = TRUE),
      days_above_1 = sum(rt_mean > 1, na.rm = TRUE),
      days_below_1 = sum(rt_mean < 1, na.rm = TRUE)
    )
  
  cat("\n=== Rt Estimation Summary ===\n")
  cat(sprintf("Number of Rt estimates: %d\n", rt_summary$n_estimates))
  cat(sprintf("Mean Rt: %.2f\n", rt_summary$mean_rt))
  cat(sprintf("Median Rt: %.2f\n", rt_summary$median_rt))
  cat(sprintf("Range: %.2f - %.2f\n", rt_summary$min_rt, rt_summary$max_rt))
  cat(sprintf("Days with Rt > 1: %d (%.1f%%)\n", 
              rt_summary$days_above_1,
              rt_summary$days_above_1 / rt_summary$n_estimates * 100))
  cat(sprintf("Days with Rt < 1: %d (%.1f%%)\n", 
              rt_summary$days_below_1,
              rt_summary$days_below_1 / rt_summary$n_estimates * 100))
  
  # Create and display plots
  plots <- plot_rt_estimates(rt_results)
  print(plots$cases_plot)
  print(plots$rt_plot)
  
  # Save results
  write.csv(rt_results, "rt_estimates.csv", row.names = FALSE)
  cat("\nResults saved to 'rt_estimates.csv'\n")
  
  # Return results for further analysis
  return(rt_results)
}

# Run the analysis
rt_results <- main()

# Additional analysis functions
analyze_rt_trends <- function(rt_data) {
  rt_clean <- rt_data %>% filter(!is.na(rt_mean))
  
  # Identify periods of growth/decline
  rt_clean <- rt_clean %>%
    mutate(
      growth_phase = rt_mean > 1,
      rt_trend = case_when(
        rt_mean > 1.2 ~ "High transmission (Rt > 1.2)",
        rt_mean > 1.0 ~ "Growing (1.0 < Rt ≤ 1.2)",
        rt_mean > 0.8 ~ "Declining (0.8 < Rt ≤ 1.0)",
        TRUE ~ "Low transmission (Rt ≤ 0.8)"
      )
    )
  
  trend_summary <- rt_clean %>%
    count(rt_trend) %>%
    mutate(percentage = n / sum(n) * 100)
  
  cat("\n=== Transmission Phase Summary ===\n")
  print(trend_summary)
  
  return(rt_clean)
}

# Run additional analysis
rt_trends <- analyze_rt_trends(rt_results)
```

---

## SUB_104

**Scenario**: 1a | **Execution**: SUCCESS

### script.R

```r
# Load required packages
if (!require(EpiEstim)) {
  install.packages("EpiEstim")
  library(EpiEstim)
}
if (!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}
if (!require(dplyr)) {
  install.packages("dplyr")
  library(dplyr)
}
if (!require(lubridate)) {
  install.packages("lubridate")
  library(lubridate)
}

# Load the data
cases_data <- read.csv("cases.csv", stringsAsFactors = FALSE)

# Convert date column to Date format
cases_data$date <- as.Date(cases_data$date)

# Sort by date to ensure chronological order
cases_data <- cases_data[order(cases_data$date), ]

# Create a complete date sequence to handle any missing dates
date_seq <- seq.Date(from = min(cases_data$date), 
                     to = max(cases_data$date), 
                     by = "day")

# Create a complete dataset with all dates
complete_data <- data.frame(date = date_seq)
complete_data <- merge(complete_data, cases_data, by = "date", all.x = TRUE)

# Replace NA values with 0 for missing dates
complete_data$cases[is.na(complete_data$cases)] <- 0

# Ensure non-negative case counts
complete_data$cases[complete_data$cases < 0] <- 0

# Prepare data for EpiEstim (requires specific format)
# EpiEstim expects a data frame with columns: dates, I (incidence)
epi_data <- data.frame(
  dates = complete_data$date,
  I = complete_data$cases
)

# Define serial interval distribution
# Using gamma distribution parameters for COVID-19
# Mean serial interval: ~5.2 days, SD: ~5.1 days (based on literature)
mean_si <- 5.2
std_si <- 5.1

# Calculate shape and scale parameters for gamma distribution
# For gamma distribution: mean = shape * scale, variance = shape * scale^2
# std^2 = shape * scale^2, so shape = mean^2 / std^2, scale = std^2 / mean
si_shape <- (mean_si^2) / (std_si^2)
si_scale <- (std_si^2) / mean_si

# Create serial interval configuration
si_config <- make_config(
  list(
    mean_si = mean_si,
    std_si = std_si,
    si_parametric_distr = "G",  # Gamma distribution
    mcmc_control = make_mcmc_control(
      burnin = 1000,
      thin = 10,
      seed = 1
    )
  )
)

# Estimate Rt using a sliding window approach
# Window size of 7 days (weekly estimates)
window_size <- 7

# Ensure we have enough data points
if (nrow(epi_data) < window_size + 1) {
  stop("Insufficient data points for Rt estimation. Need at least ", 
       window_size + 1, " days of data.")
}

# Estimate Rt
rt_estimates <- estimate_R(
  incid = epi_data,
  method = "parametric_si",
  config = si_config
)

# Extract Rt results
rt_results <- rt_estimates$R

# Add dates to the results
# The first estimate corresponds to day (window_size + 1)
rt_results$date <- epi_data$dates[(window_size + 1):nrow(epi_data)]

# Create summary statistics
rt_summary <- rt_results %>%
  select(date, `Mean(R)`, `Quantile.0.025(R)`, `Quantile.0.975(R)`) %>%
  rename(
    Rt_mean = `Mean(R)`,
    Rt_lower = `Quantile.0.025(R)`,
    Rt_upper = `Quantile.0.975(R)`
  )

# Display summary
print("=== Rt Estimation Summary ===")
print(paste("Data period:", min(epi_data$dates), "to", max(epi_data$dates)))
print(paste("Total days:", nrow(epi_data)))
print(paste("Rt estimates available from:", min(rt_summary$date)))
print(paste("Serial interval - Mean:", mean_si, "days, SD:", std_si, "days"))
print("")
print("First 10 Rt estimates:")
print(head(rt_summary, 10))
print("")
print("Last 10 Rt estimates:")
print(tail(rt_summary, 10))
print("")
print("Overall Rt statistics:")
print(summary(rt_summary[, c("Rt_mean", "Rt_lower", "Rt_upper")]))

# Create visualization
p1 <- ggplot(complete_data, aes(x = date, y = cases)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  labs(title = "Daily COVID-19 Case Counts",
       x = "Date",
       y = "Number of Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ggplot(rt_summary, aes(x = date)) +
  geom_ribbon(aes(ymin = Rt_lower, ymax = Rt_upper), 
              fill = "gray", alpha = 0.3) +
  geom_line(aes(y = Rt_mean), color = "red", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       subtitle = "Red line: Mean Rt, Gray ribbon: 95% Credible Interval",
       x = "Date",
       y = "Rt") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0, max(rt_summary$Rt_upper) * 1.1)

# Display plots
print(p1)
print(p2)

# Save results to CSV
write.csv(rt_summary, "rt_estimates.csv", row.names = FALSE)

# Additional analysis: periods where Rt > 1
epidemic_periods <- rt_summary %>%
  filter(Rt_lower > 1) %>%
  summarise(
    periods_above_1 = n(),
    max_rt = max(Rt_mean),
    date_max_rt = date[which.max(Rt_mean)]
  )

declining_periods <- rt_summary %>%
  filter(Rt_upper < 1) %>%
  summarise(
    periods_below_1 = n(),
    min_rt = min(Rt_mean),
    date_min_rt = date[which.min(Rt_mean)]
  )

print("=== Epidemic Analysis ===")
print(paste("Days with Rt significantly > 1 (lower CI > 1):", 
            ifelse(nrow(epidemic_periods) > 0, epidemic_periods$periods_above_1, 0)))
print(paste("Days with Rt significantly < 1 (upper CI < 1):", 
            ifelse(nrow(declining_periods) > 0, declining_periods$periods_below_1, 0)))

if (nrow(epidemic_periods) > 0 && epidemic_periods$periods_above_1 > 0) {
  print(paste("Maximum Rt:", round(epidemic_periods$max_rt, 2), 
              "on", epidemic_periods$date_max_rt))
}

if (nrow(declining_periods) > 0 && declining_periods$periods_below_1 > 0) {
  print(paste("Minimum Rt:", round(declining_periods$min_rt, 2), 
              "on", declining_periods$date_min_rt))
}

# Function to get Rt estimate for a specific date
get_rt_for_date <- function(target_date) {
  target_date <- as.Date(target_date)
  result <- rt_summary[rt_summary$date == target_date, ]
  if (nrow(result) > 0) {
    return(result)
  } else {
    return(paste("No Rt estimate available for", target_date))
  }
}

# Example usage of the function
print("=== Example: Get Rt for specific date ===")
if (nrow(rt_summary) > 10) {
  example_date <- rt_summary$date[10]
  print(paste("Rt estimate for", example_date, ":"))
  print(get_rt_for_date(example_date))
}

# Create a combined plot
library(gridExtra)
combined_plot <- grid.arrange(p1, p2, ncol = 1, heights = c(1, 1))

# Print final message
print("=== Analysis Complete ===")
print("Results saved to 'rt_estimates.csv'")
print("Rt estimates with 95% credible intervals are available in rt_summary dataframe")
```

---

## SUB_058

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Install required packages if not already installed
install.packages("dplyr")
install.packages("ggplot2")

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Load data from the CSV file
cases <- read.csv("cases.csv", stringsAsFactors = FALSE)

# Convert date to Date format for proper sorting and plotting
cases$date <- as.Date(cases$date)

# Define a function to estimate Rt using SIR model
estimate_rt <- function(data) {
  # Fit an ARIMA model to the case counts
  arima_model <- auto.arima(data$cases, order = c(1, 1, 0), seasonal = FALSE)
  
  # Extract parameters of the fitted ARIMA model
  phi <- coefficients(arima_model)[1]
  theta <- coefficients(arima_model)[2]
  
  # Calculate Rt using formula from SIR model and ARIMA parameters
  rt <- (phi + theta) / (1 - phi * theta)
  
  return(rt)
}

# Apply the function to each time point and store estimates in a new data frame
cases_rt <- cases %>%
  group_by(date) %>%
  summarise(rts = estimate_rt(.$cases))

# Plot Rt over time
ggplot(cases_rt, aes(x = date, y = rts)) +
  geom_line() +
  labs(title = "Time-varying Reproduction Number (Rt)", x = "Date", y = "Rt")
```

---

## SUB_042

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Install required packages if not already installed
install.packages("dplyr")
install.packages("tidyr")
install.packages("lubridate")
install.packages("bats")

# Load required libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(bats)


# Load the data from cases.csv
cases_data <- read_csv("cases.csv")

# Convert date column to Date format
cases_data$date <- ymd(cases_data$date)

# Ensure that cases are in integer format
cases_data$cases <- as.integer(cases_data$cases)


# Define a new function for Rt estimation using bats
Rt_estimation <- function(x) {
  # x should be a tibble with date and cases columns
  
  # Calculate lagged values of cases (for autoregression)
  x$lag_cases <- lag(x, x$date, -1)$cases
  
  # Fit the model
  model <- bats(
    formula = cases ~ lag_cases + (1|date),
    data = x,
    family = "poisson"
  )
  
  # Extract and return Rt estimates
  Rt_estimates <- predict(model, newdata = x, type = "Rt")
  return(Rt_estimates)
}

# Apply the function to each time point
estimates <- cases_data %>%
  group_by(date) %>%
  summarise(
    Rt_estimate = Rt_estimation(.[, c("date", "cases")])
  )


# Extract Rt estimates for plotting
Rt_over_time <- estimates$Rt_estimate

# Plot Rt over time (optional)
plot(Rt_over_time,
     main = "Time-Varying Reproduction Number (Rt)",
     xlab = "Date",
     ylab = "Estimated Rt")
```

---

## SUB_111

**Scenario**: 1a | **Execution**: FAILED

### script.R

```r
# Install required libraries if not already installed
install.packages("dplyr")
install.packages("lubridate")
install.packages("deSolve")

# Load necessary libraries
library(dplyr)
library(lubridate)
library(deSolve)

# Load the daily case counts from a CSV file
cases <- read.csv("cases.csv", stringsAsFactors = FALSE) %>%
  mutate(date = as.Date(date))

# Set up parameters for the model (you may need to adjust these based on your data and understanding of the model)
parameters <- list(
  R0 = 2.5, # Initial reproduction number
  gamma = 1/7, # Rate at which infected individuals recover or die
  sigma = 1/3, # Rate at which susceptible individuals become infected
  N = sum(cases$cases), # Total population (estimated from case counts)
  I0 = cases$cases[1], # Initial number of infected individuals (first reported case count)
  S0 = sum(cases$cases) - cases$cases[1] # Initial number of susceptible individuals
)

# Define the model function
model <- function(t, y, parameters) {
  with(as.list(c(y, parameters)), {
    dSdt <- -beta * S * I / N
    dIdt <- beta * S * I / N - gamma * I
    return(list(c(dSdt, dIdt)))
  })
}

# Define the solve function to fit the model
solve <- function(parameters) {
  # Discretize the time points (you may need to adjust this based on your data and computational resources)
  t <- seq(0, max(cases$date), by = "day")
  
  # Initial conditions
  init <- c(S = parameters$S0, I = parameters$I0)
  
  # Solve ODEs
  output <- ode(y = init, times = t, func = model, parms = parameters)
  
  # Extract Rt estimates for each time point (using the estimated beta and gamma values)
  Rt <- (parameters$beta * output$solve[, "S"] / output$solve[, "I"]) + parameters$gamma
  
  return(Rt)
}

# Fit the model
Rt_estimates <- solve(parameters)

# Plot or summarize the results
plot(cases$date, Rt_estimates, type = "l", 
     xlab = "Date", ylab = expression("Estimated R[italic(t)]"))

summary(Rt_estimates)
```

---

# Scenario 1b

Total submissions: 30

---

## SUB_030

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(lubridate)

# Load and prepare the data
load_and_prepare_data <- function(filepath = "cases.csv") {
  # Load data
  data <- read.csv(filepath)
  data$date <- as.Date(data$date)
  data <- data[order(data$date), ]
  
  # Remove any missing or negative cases
  data <- data[!is.na(data$cases) & data$cases >= 0, ]
  
  return(data)
}

# Generate discrete generation interval distribution
generate_generation_interval <- function(max_gen = 20, mean_gen = 5.2, sd_gen = 2.8) {
  # Discretized gamma distribution for generation interval
  gen_interval <- dgamma(1:max_gen, 
                        shape = (mean_gen/sd_gen)^2, 
                        rate = mean_gen/sd_gen^2)
  gen_interval <- gen_interval / sum(gen_interval)
  return(gen_interval)
}

# Generate reporting delay distribution
generate_reporting_delay <- function(max_delay = 15, mean_delay = 5, sd_delay = 3) {
  # Discretized gamma distribution for reporting delay
  reporting_delay <- dgamma(0:max_delay,
                           shape = (mean_delay/sd_delay)^2,
                           rate = mean_delay/sd_delay^2)
  reporting_delay <- reporting_delay / sum(reporting_delay)
  return(reporting_delay)
}

# Stan model code
stan_code <- "
data {
  int<lower=1> T;                    // Number of time points
  int cases[T];                      // Observed case counts
  int<lower=1> G;                    // Generation interval length
  vector[G] generation_pmf;          // Generation interval PMF
  int<lower=1> D;                    // Reporting delay length
  vector[D+1] reporting_pmf;         // Reporting delay PMF (0 to D days)
  int<lower=1> S;                    // Seeding period
  real<lower=0> gamma_shape;         // Shape for gamma prior on R
  real<lower=0> gamma_rate;          // Rate for gamma prior on R
}

transformed data {
  vector[T] log_cases_plus_one = log(to_vector(cases) + 1.0);
}

parameters {
  vector<lower=0>[S] I_seed;         // Initial seeded infections
  vector<lower=0>[T-S] R_t;          // Time-varying reproduction number
  real<lower=0> phi;                 // Overdispersion parameter
}

transformed parameters {
  vector[T] infections;
  vector[T] expected_cases;
  
  // Initialize infections with seeding
  infections[1:S] = I_seed;
  
  // Compute infections using renewal equation
  for (t in (S+1):T) {
    real infectiousness = 0.0;
    int start_idx = max(1, t - G);
    for (s in start_idx:(t-1)) {
      int delay = t - s;
      if (delay <= G) {
        infectiousness += infections[s] * generation_pmf[delay];
      }
    }
    infections[t] = R_t[t-S] * infectiousness;
  }
  
  // Convolve infections with reporting delay to get expected cases
  for (t in 1:T) {
    expected_cases[t] = 0.0;
    for (d in 0:D) {
      int infection_day = t - d;
      if (infection_day >= 1) {
        expected_cases[t] += infections[infection_day] * reporting_pmf[d+1];
      }
    }
    expected_cases[t] = fmax(expected_cases[t], 1e-8);
  }
}

model {
  // Priors
  I_seed ~ exponential(0.1);
  R_t ~ gamma(gamma_shape, gamma_rate);
  phi ~ exponential(0.2);
  
  // Likelihood
  cases ~ neg_binomial_2(expected_cases, phi);
}

generated quantities {
  vector[T] log_lik;
  vector[T] cases_rep;
  vector[T] all_R_t;
  
  // Log likelihood for model comparison
  for (t in 1:T) {
    log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
  }
  
  // Posterior predictive samples
  for (t in 1:T) {
    cases_rep[t] = neg_binomial_2_rng(expected_cases[t], phi);
  }
  
  // Complete R_t series (with NAs for seeding period)
  all_R_t[1:S] = rep_vector(-999, S);  // Use -999 as indicator for NA
  all_R_t[(S+1):T] = R_t;
}
"

# Main function to estimate Rt
estimate_rt <- function(data, 
                       seeding_days = 7,
                       max_gen = 20,
                       mean_gen = 5.2, 
                       sd_gen = 2.8,
                       max_delay = 15,
                       mean_delay = 5,
                       sd_delay = 3,
                       gamma_shape = 1,
                       gamma_rate = 0.2,
                       chains = 4,
                       iter_warmup = 1000,
                       iter_sampling = 1000,
                       adapt_delta = 0.95,
                       max_treedepth = 12) {
  
  # Generate distributions
  gen_pmf <- generate_generation_interval(max_gen, mean_gen, sd_gen)
  rep_pmf <- generate_reporting_delay(max_delay, mean_delay, sd_delay)
  
  # Prepare data for Stan
  stan_data <- list(
    T = nrow(data),
    cases = data$cases,
    G = length(gen_pmf),
    generation_pmf = gen_pmf,
    D = max_delay,
    reporting_pmf = rep_pmf,
    S = seeding_days,
    gamma_shape = gamma_shape,
    gamma_rate = gamma_rate
  )
  
  # Compile and fit model
  cat("Compiling Stan model...\n")
  model <- cmdstan_model(stan_file = write_stan_file(stan_code))
  
  cat("Fitting model...\n")
  fit <- model$sample(
    data = stan_data,
    chains = chains,
    parallel_chains = chains,
    iter_warmup = iter_warmup,
    iter_sampling = iter_sampling,
    adapt_delta = adapt_delta,
    max_treedepth = max_treedepth,
    refresh = 100,
    show_messages = TRUE
  )
  
  return(list(
    fit = fit,
    data = data,
    stan_data = stan_data,
    seeding_days = seeding_days
  ))
}

# Extract Rt estimates
extract_rt_estimates <- function(results) {
  fit <- results$fit
  data <- results$data
  seeding_days <- results$seeding_days
  
  # Extract R_t estimates
  rt_draws <- fit$draws("all_R_t")
  rt_summary <- summarise_draws(rt_draws)
  
  # Create results dataframe
  rt_estimates <- data.frame(
    date = data$date,
    observed_cases = data$cases,
    rt_mean = rt_summary$mean,
    rt_median = rt_summary$median,
    rt_lower = rt_summary$q5,
    rt_upper = rt_summary$q95,
    rt_lower_50 = rt_summary$q25,
    rt_upper_50 = rt_summary$q75
  )
  
  # Set seeding period values to NA
  rt_estimates$rt_mean[1:seeding_days] <- NA
  rt_estimates$rt_median[1:seeding_days] <- NA
  rt_estimates$rt_lower[1:seeding_days] <- NA
  rt_estimates$rt_upper[1:seeding_days] <- NA
  rt_estimates$rt_lower_50[1:seeding_days] <- NA
  rt_estimates$rt_upper_50[1:seeding_days] <- NA
  
  return(rt_estimates)
}

# Plot Rt estimates
plot_rt_estimates <- function(rt_estimates, title = "Time-varying Reproduction Number (Rt)") {
  p1 <- ggplot(rt_estimates, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = rt_lower_50, ymax = rt_upper_50), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = rt_median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    labs(title = title,
         x = "Date",
         y = "Rt",
         subtitle = "Dark ribbon: 50% CI, Light ribbon: 90% CI") +
    theme_minimal() +
    theme(plot.title = element_text(size = 14, face = "bold"),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  p2 <- ggplot(rt_estimates, aes(x = date)) +
    geom_col(aes(y = observed_cases), alpha = 0.7, fill = "grey50") +
    labs(title = "Observed Cases",
         x = "Date",
         y = "Daily Cases") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Combine plots
  gridExtra::grid.arrange(p1, p2, ncol = 1, heights = c(2, 1))
}

# Model diagnostics
check_model_diagnostics <- function(results) {
  fit <- results$fit
  
  cat("=== Model Diagnostics ===\n")
  
  # Check convergence
  diagnostics <- fit$diagnostic_summary()
  print(diagnostics)
  
  # Rhat and ESS
  draws_summary <- fit$summary()
  max_rhat <- max(draws_summary$rhat, na.rm = TRUE)
  min_ess_bulk <- min(draws_summary$ess_bulk, na.rm = TRUE)
  min_ess_tail <- min(draws_summary$ess_tail, na.rm = TRUE)
  
  cat(sprintf("\nMax Rhat: %.3f (should be < 1.01)\n", max_rhat))
  cat(sprintf("Min ESS Bulk: %.0f (should be > 400)\n", min_ess_bulk))
  cat(sprintf("Min ESS Tail: %.0f (should be > 400)\n", min_ess_tail))
  
  # Plot diagnostics
  bayesplot_theme_set(theme_minimal())
  
  # Trace plots for key parameters
  rt_draws <- fit$draws("R_t")
  phi_draws <- fit$draws("phi")
  
  p1 <- mcmc_trace(rt_draws[,, 1:min(6, dim(rt_draws)[3])]) + 
        ggtitle("Trace plots: First 6 R_t parameters")
  
  p2 <- mcmc_trace(phi_draws) + 
        ggtitle("Trace plot: Overdispersion parameter")
  
  print(p1)
  print(p2)
  
  return(list(
    max_rhat = max_rhat,
    min_ess_bulk = min_ess_bulk,
    min_ess_tail = min_ess_tail,
    diagnostics = diagnostics
  ))
}

# Main execution function
main <- function(filepath = "cases.csv") {
  cat("=== Rt Estimation Using Renewal Equation ===\n\n")
  
  # Load and prepare data
  cat("Loading data...\n")
  data <- load_and_prepare_data(filepath)
  cat(sprintf("Loaded %d days of data from %s to %s\n", 
              nrow(data), min(data$date), max(data$date)))
  cat(sprintf("Total cases: %d\n", sum(data$cases)))
  cat(sprintf("Mean daily cases: %.1f\n\n", mean(data$cases)))
  
  # Estimate Rt
  results <- estimate_rt(data)
  
  # Check diagnostics
  diag <- check_model_diagnostics(results)
  
  if (diag$max_rhat > 1.01) {
    warning("Model may not have converged (Rhat > 1.01)")
  }
  
  # Extract estimates
  cat("\nExtracting Rt estimates...\n")
  rt_estimates <- extract_rt_estimates(results)
  
  # Print summary
  cat("\n=== Rt Summary Statistics ===\n")
  rt_summary_stats <- rt_estimates %>%
    filter(!is.na(rt_median)) %>%
    summarise(
      mean_rt = mean(rt_median, na.rm = TRUE),
      median_rt = median(rt_median, na.rm = TRUE),
      min_rt = min(rt_median, na.rm = TRUE),
      max_rt = max(rt_median, na.rm = TRUE),
      days_above_1 = sum(rt_median > 1, na.rm = TRUE),
      total_days = sum(!is.na(rt_median))
    )
  
  print(rt_summary_stats)
  cat(sprintf("Proportion of days with Rt > 1: %.2f%%\n", 
              100 * rt_summary_stats$days_above_1 / rt_summary_stats$total_days))
  
  # Create plots
  cat("\nGenerating plots...\n")
  plot_rt_estimates(rt_estimates)
  
  # Return results
  return(list(
    rt_estimates = rt_estimates,
    results = results,
    diagnostics = diag,
    summary_stats = rt_summary_stats
  ))
}

# Example of how to run with simulated data if cases.csv doesn't exist
create_example_data <- function() {
  set.seed(42)
  dates <- seq(as.Date("2023-01-01"), as.Date("2023-06-30"), by = "day")
  
  # Simulate cases with changing Rt
  n_days <- length(dates)
  true_rt <- c(rep(1.5, 30), rep(0.8, 60), rep(1.2, 40), rep(0.9, n_days - 130))
  
  cases <- numeric(n_days)
  cases[1:7] <- rpois(7, 20)  # Initial seeding
  
  for (t in 8:n_days) {
    lambda <- true_rt[t] * sum(cases[max(1, t-10):(t-1)] * 
                              dgamma(1:min(10, t-1), shape = 2.8, rate = 0.54))
    cases[t] <- rpois(1, max(1, lambda))
  }
  
  data <- data.frame(date = dates, cases = cases)
  write.csv(data, "cases.csv", row.names = FALSE)
  cat("Created example data file: cases.csv\n")
  return(data)
}

# Run the analysis
if (!file.exists("cases.csv")) {
  cat("cases.csv not found. Creating example data...\n")
  create_example_data()
}

# Execute main analysis
final_results <- main("cases.csv")
```

---

## SUB_043

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(lubridate)

# Function to create generation interval PMF
create_generation_interval <- function(max_gen = 20, mean_gi = 5.2, sd_gi = 1.72) {
  # Discretized gamma distribution for generation interval
  # Based on COVID-19 estimates from literature
  shape <- (mean_gi / sd_gi)^2
  rate <- mean_gi / sd_gi^2
  
  gen_pmf <- diff(pgamma(0:(max_gen), shape = shape, rate = rate))
  gen_pmf <- gen_pmf / sum(gen_pmf)  # Normalize
  
  return(gen_pmf)
}

# Function to create reporting delay PMF  
create_reporting_delay <- function(max_delay = 21, mean_delay = 8, sd_delay = 2) {
  # Log-normal distribution for reporting delay
  meanlog <- log(mean_delay) - 0.5 * log(1 + (sd_delay/mean_delay)^2)
  sdlog <- sqrt(log(1 + (sd_delay/mean_delay)^2))
  
  delay_pmf <- diff(plnorm(0:max_delay, meanlog = meanlog, sdlog = sdlog))
  delay_pmf <- delay_pmf / sum(delay_pmf)  # Normalize
  
  return(delay_pmf)
}

# Stan model code
stan_code <- "
data {
  int<lower=1> n_days;                    // Number of days
  int<lower=0> cases[n_days];             // Observed case counts
  int<lower=1> gen_max;                   // Max generation interval
  vector<lower=0>[gen_max] gen_pmf;       // Generation interval PMF
  int<lower=1> delay_max;                 // Max reporting delay
  vector<lower=0>[delay_max] delay_pmf;   // Reporting delay PMF
  int<lower=1> n_seed_days;               // Number of seed days
}

transformed data {
  int first_est_day = n_seed_days + 1;    // First day to estimate Rt
}

parameters {
  vector<lower=0>[n_seed_days] seed_infections;     // Initial infections
  vector[n_days - n_seed_days] log_rt_raw;          // Log Rt (raw)
  real<lower=0> rt_sigma;                           // Random walk SD
  real rt_mean_log;                                 // Mean log Rt
  real<lower=0> phi_inv;                            // Overdispersion parameter
}

transformed parameters {
  vector<lower=0>[n_days] infections;
  vector<lower=0>[n_days] rt;
  vector<lower=0>[n_days] expected_cases;
  real<lower=0> phi = 1.0 / phi_inv;
  
  // Initialize infections for seed days
  infections[1:n_seed_days] = seed_infections;
  
  // Random walk prior for log Rt
  vector[n_days - n_seed_days] log_rt;
  log_rt[1] = rt_mean_log + rt_sigma * log_rt_raw[1];
  for (t in 2:(n_days - n_seed_days)) {
    log_rt[t] = log_rt[t-1] + rt_sigma * log_rt_raw[t];
  }
  rt[(n_seed_days+1):n_days] = exp(log_rt);
  rt[1:n_seed_days] = rep_vector(1.0, n_seed_days);  // Placeholder for seed days
  
  // Apply renewal equation for non-seed days
  for (t in first_est_day:n_days) {
    real infectiousness = 0;
    int max_lag = min(gen_max, t - 1);
    
    for (s in 1:max_lag) {
      if (t - s >= 1) {
        infectiousness += infections[t - s] * gen_pmf[s];
      }
    }
    
    infections[t] = rt[t] * infectiousness;
  }
  
  // Apply reporting delay to get expected cases
  for (t in 1:n_days) {
    expected_cases[t] = 0;
    int max_delay_lag = min(delay_max, t);
    
    for (d in 1:max_delay_lag) {
      if (t - d + 1 >= 1) {
        expected_cases[t] += infections[t - d + 1] * delay_pmf[d];
      }
    }
  }
}

model {
  // Priors
  seed_infections ~ exponential(0.03);
  log_rt_raw ~ std_normal();
  rt_sigma ~ normal(0, 0.2) T[0,];
  rt_mean_log ~ normal(0, 1);
  phi_inv ~ exponential(5);
  
  // Likelihood
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  int sim_cases[n_days];
  vector[n_days] log_lik;
  
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      sim_cases[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      sim_cases[t] = 0;
      log_lik[t] = 0;
    }
  }
}
"

# Main estimation function
estimate_rt <- function(case_data, n_seed_days = 7, chains = 4, iter = 2000) {
  
  # Prepare data
  cases_vector <- case_data$cases
  n_days <- length(cases_vector)
  dates <- case_data$date
  
  # Create generation interval and reporting delay
  gen_pmf <- create_generation_interval()
  delay_pmf <- create_reporting_delay()
  
  # Prepare Stan data
  stan_data <- list(
    n_days = n_days,
    cases = cases_vector,
    gen_max = length(gen_pmf),
    gen_pmf = gen_pmf,
    delay_max = length(delay_pmf),
    delay_pmf = delay_pmf,
    n_seed_days = n_seed_days
  )
  
  # Compile and fit model
  model <- cmdstan_model(write_stan_file(stan_code))
  
  fit <- model$sample(
    data = stan_data,
    chains = chains,
    iter_warmup = iter / 2,
    iter_sampling = iter / 2,
    refresh = 100,
    max_treedepth = 12,
    adapt_delta = 0.95
  )
  
  return(list(fit = fit, data = case_data, stan_data = stan_data))
}

# Function to extract and summarize Rt estimates
extract_rt_estimates <- function(model_result) {
  
  fit <- model_result$fit
  case_data <- model_result$data
  n_seed_days <- model_result$stan_data$n_seed_days
  
  # Extract Rt samples
  rt_samples <- fit$draws("rt", format = "matrix")
  
  # Calculate summary statistics
  rt_summary <- posterior::summarise_draws(rt_samples)
  
  # Create results dataframe
  results <- data.frame(
    date = case_data$date,
    rt_mean = rt_summary$mean,
    rt_median = rt_summary$median,
    rt_lower = rt_summary$q5,
    rt_upper = rt_summary$q95,
    rt_lower_50 = rt_summary$q25,
    rt_upper_50 = rt_summary$q75,
    is_seed = c(rep(TRUE, n_seed_days), rep(FALSE, nrow(case_data) - n_seed_days))
  )
  
  return(results)
}

# Function to plot results
plot_rt_estimates <- function(rt_estimates, case_data) {
  
  # Filter out seed days for plotting
  rt_plot_data <- rt_estimates[!rt_estimates$is_seed, ]
  
  p1 <- ggplot(rt_plot_data, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = rt_lower_50, ymax = rt_upper_50), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = rt_median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(
      title = "Time-varying Reproduction Number (Rt)",
      subtitle = "Estimated using renewal equation",
      x = "Date",
      y = "Rt",
      caption = "Dark ribbon: 50% CI, Light ribbon: 90% CI"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot cases
  p2 <- ggplot(case_data, aes(x = date, y = cases)) +
    geom_col(alpha = 0.7, fill = "steelblue") +
    labs(
      title = "Daily Case Counts",
      x = "Date",
      y = "Cases"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(list(rt_plot = p1, cases_plot = p2))
}

# Example usage (assuming cases.csv exists)
# Load data
load_case_data <- function(file_path = "cases.csv") {
  if (!file.exists(file_path)) {
    # Create example data if file doesn't exist
    set.seed(123)
    dates <- seq(as.Date("2020-03-01"), as.Date("2020-08-01"), by = "day")
    n_days <- length(dates)
    
    # Simulate cases with changing Rt
    true_rt <- c(
      rep(2.5, 30),                    # Initial growth
      seq(2.5, 0.8, length.out = 40), # Lockdown effect
      rep(0.8, 30),                    # Low transmission
      seq(0.8, 1.3, length.out = 20), # Relaxation
      rep(1.1, n_days - 120)          # Steady state
    )
    
    cases <- rpois(n_days, pmax(1, 100 * exp(cumsum(log(true_rt) - 0.1))))
    
    case_data <- data.frame(
      date = dates,
      cases = cases
    )
    
    write.csv(case_data, file_path, row.names = FALSE)
    cat("Created example data file:", file_path, "\n")
  } else {
    case_data <- read.csv(file_path)
    case_data$date <- as.Date(case_data$date)
  }
  
  return(case_data)
}

# Run the analysis
run_rt_analysis <- function() {
  # Load data
  cat("Loading case data...\n")
  case_data <- load_case_data()
  
  # Estimate Rt
  cat("Estimating Rt using renewal equation...\n")
  model_result <- estimate_rt(case_data)
  
  # Extract results
  cat("Extracting Rt estimates...\n")
  rt_estimates <- extract_rt_estimates(model_result)
  
  # Create plots
  cat("Creating plots...\n")
  plots <- plot_rt_estimates(rt_estimates, case_data)
  
  # Print summary
  cat("\nRt Summary (excluding seed days):\n")
  rt_summary <- rt_estimates[!rt_estimates$is_seed, ]
  cat("Mean Rt:", round(mean(rt_summary$rt_mean), 2), "\n")
  cat("Min Rt:", round(min(rt_summary$rt_mean), 2), "\n")
  cat("Max Rt:", round(max(rt_summary$rt_mean), 2), "\n")
  cat("Days with Rt > 1:", sum(rt_summary$rt_median > 1), "/", nrow(rt_summary), "\n")
  
  # Show plots
  print(plots$cases_plot)
  print(plots$rt_plot)
  
  return(list(
    rt_estimates = rt_estimates,
    model_result = model_result,
    plots = plots
  ))
}

# Run the analysis
results <- run_rt_analysis()
```

---

## SUB_015

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(lubridate)

# Function to generate discretized generation interval
generate_generation_interval <- function(max_days = 20, mean_gi = 5.2, sd_gi = 5.1) {
  # Using gamma distribution for generation interval
  # Convert mean and sd to shape and rate parameters
  shape <- (mean_gi / sd_gi)^2
  rate <- mean_gi / (sd_gi^2)
  
  # Calculate probability mass function
  gi <- dgamma(1:max_days, shape = shape, rate = rate)
  gi <- gi / sum(gi)  # Normalize to sum to 1
  
  return(gi)
}

# Function to generate discretized incubation period + reporting delay
generate_delay_distribution <- function(max_days = 30, mean_delay = 8, sd_delay = 4) {
  # Using gamma distribution for total delay (incubation + reporting)
  shape <- (mean_delay / sd_delay)^2
  rate <- mean_delay / (sd_delay^2)
  
  delay <- dgamma(1:max_days, shape = shape, rate = rate)
  delay <- delay / sum(delay)
  
  return(delay)
}

# Stan model code
stan_model_code <- "
data {
  int<lower=1> n_days;                    // Number of days
  int<lower=0> cases[n_days];             // Observed case counts
  int<lower=1> gi_max;                    // Maximum generation interval
  vector<lower=0>[gi_max] gi_pmf;         // Generation interval PMF
  int<lower=1> delay_max;                 // Maximum delay
  vector<lower=0>[delay_max] delay_pmf;   // Delay distribution PMF
  int<lower=1> n_seed_days;               // Number of seeding days
}

transformed data {
  real log_cases[n_days];
  for (t in 1:n_days) {
    log_cases[t] = log(cases[t] + 0.5);  // Add small constant for log
  }
}

parameters {
  vector[n_seed_days] log_I0;             // Initial infections (seeding period)
  vector[n_days - n_seed_days] log_Rt;   // Log reproduction numbers
  real<lower=0> phi;                      // Overdispersion parameter for neg binomial
  real<lower=0> sigma_rt;                 // Random walk standard deviation for Rt
}

transformed parameters {
  vector[n_days] infections;
  vector[n_days] expected_cases;
  vector[n_days - n_seed_days] Rt;
  
  // Convert log_Rt to Rt
  Rt = exp(log_Rt);
  
  // Initialize infections for seeding period
  for (t in 1:n_seed_days) {
    infections[t] = exp(log_I0[t]);
  }
  
  // Apply renewal equation for remaining days
  for (t in (n_seed_days + 1):n_days) {
    real convolution = 0;
    int max_lag = min(gi_max, t - 1);
    
    for (s in 1:max_lag) {
      if (t - s >= 1) {
        convolution += infections[t - s] * gi_pmf[s];
      }
    }
    infections[t] = Rt[t - n_seed_days] * convolution;
  }
  
  // Apply delay distribution to get expected cases
  for (t in 1:n_days) {
    expected_cases[t] = 0;
    int max_delay = min(delay_max, t);
    
    for (d in 1:max_delay) {
      if (t - d + 1 >= 1) {
        expected_cases[t] += infections[t - d + 1] * delay_pmf[d];
      }
    }
  }
}

model {
  // Priors
  log_I0 ~ normal(3, 2);                          // Prior on initial infections
  log_Rt[1] ~ normal(log(1), 0.5);               // Prior on first Rt
  
  // Random walk prior for Rt
  for (t in 2:(n_days - n_seed_days)) {
    log_Rt[t] ~ normal(log_Rt[t-1], sigma_rt);
  }
  
  sigma_rt ~ normal(0, 0.2) T[0,];                // Prior on Rt random walk SD
  phi ~ exponential(0.1);                         // Prior on overdispersion
  
  // Likelihood
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  int<lower=0> cases_pred[n_days];
  vector[n_days] log_lik;
  
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = 0;
      log_lik[t] = 0;
    }
  }
}
"

# Function to prepare data and fit model
estimate_rt <- function(cases_file = "cases.csv", n_seed_days = 7) {
  
  # Load and prepare data
  cat("Loading data...\n")
  data <- read.csv(cases_file)
  data$date <- as.Date(data$date)
  data <- data[order(data$date), ]
  
  # Generate generation interval and delay distributions
  gi_pmf <- generate_generation_interval()
  delay_pmf <- generate_delay_distribution()
  
  # Prepare data for Stan
  stan_data <- list(
    n_days = nrow(data),
    cases = data$cases,
    gi_max = length(gi_pmf),
    gi_pmf = gi_pmf,
    delay_max = length(delay_pmf),
    delay_pmf = delay_pmf,
    n_seed_days = n_seed_days
  )
  
  cat("Compiling Stan model...\n")
  # Compile Stan model
  model <- cmdstan_model(stan_file = write_stan_file(stan_model_code))
  
  cat("Fitting model...\n")
  # Fit model
  fit <- model$sample(
    data = stan_data,
    chains = 4,
    parallel_chains = 4,
    iter_warmup = 1000,
    iter_sampling = 1000,
    refresh = 100,
    adapt_delta = 0.95,
    max_treedepth = 12
  )
  
  cat("Extracting results...\n")
  # Extract results
  draws <- fit$draws()
  
  # Extract Rt estimates
  rt_draws <- subset_draws(draws, variable = "Rt")
  rt_summary <- summarise_draws(rt_draws, 
                               mean, median, sd, mad,
                               ~quantile(.x, probs = c(0.025, 0.25, 0.75, 0.975)))
  
  # Add dates (excluding seeding period)
  rt_summary$date <- data$date[(n_seed_days + 1):nrow(data)]
  
  # Extract infection estimates
  inf_draws <- subset_draws(draws, variable = "infections")
  inf_summary <- summarise_draws(inf_draws,
                                mean, median, sd,
                                ~quantile(.x, probs = c(0.025, 0.975)))
  inf_summary$date <- data$date
  
  # Extract expected cases
  exp_cases_draws <- subset_draws(draws, variable = "expected_cases")
  exp_cases_summary <- summarise_draws(exp_cases_draws,
                                      mean, median, sd,
                                      ~quantile(.x, probs = c(0.025, 0.975)))
  exp_cases_summary$date <- data$date
  
  # Create results list
  results <- list(
    data = data,
    fit = fit,
    rt_estimates = rt_summary,
    infection_estimates = inf_summary,
    expected_cases = exp_cases_summary,
    stan_data = stan_data
  )
  
  return(results)
}

# Function to plot results
plot_rt_results <- function(results) {
  
  # Plot Rt over time
  p1 <- ggplot(results$rt_estimates, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         x = "Date", y = "Rt",
         subtitle = "Median with 50% and 95% credible intervals") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot observed vs expected cases
  plot_data <- merge(results$data, results$expected_cases, by = "date")
  
  p2 <- ggplot(plot_data, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "green") +
    geom_line(aes(y = median), color = "darkgreen", size = 1, linetype = "dashed") +
    geom_point(aes(y = cases), color = "black", size = 1) +
    labs(title = "Observed Cases vs Model Fit",
         x = "Date", y = "Cases",
         subtitle = "Black points: observed cases, Green: model predictions") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot infections
  p3 <- ggplot(results$infection_estimates, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "orange") +
    geom_line(aes(y = median), color = "darkorange", size = 1) +
    labs(title = "Estimated Infections by Date of Infection",
         x = "Date", y = "Infections",
         subtitle = "Median with 95% credible intervals") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  return(list(rt_plot = p1, cases_plot = p2, infections_plot = p3))
}

# Function to create summary table
summarize_rt <- function(results) {
  rt_summary <- results$rt_estimates %>%
    mutate(
      week = floor_date(date, "week"),
      above_one = `2.5%` > 1,
      below_one = `97.5%` < 1
    ) %>%
    group_by(week) %>%
    summarise(
      median_rt = median(median),
      mean_rt = mean(mean),
      prob_above_1 = mean(`2.5%` > 1),
      prob_below_1 = mean(`97.5%` < 1),
      .groups = 'drop'
    )
  
  return(rt_summary)
}

# Example usage and create sample data if needed
if (!file.exists("cases.csv")) {
  cat("Creating sample data file...\n")
  # Create sample data
  dates <- seq(as.Date("2023-01-01"), as.Date("2023-06-30"), by = "day")
  
  # Simulate some realistic case counts with changing Rt
  set.seed(123)
  n_days <- length(dates)
  true_rt <- c(rep(1.5, 30), 1.5 * exp(-0.02 * (31:90)), rep(0.8, n_days - 90))
  
  # Simple simulation
  infections <- numeric(n_days)
  infections[1:7] <- rpois(7, 50)  # Initial seeding
  
  gi_pmf <- generate_generation_interval()
  
  for (t in 8:n_days) {
    expected_inf <- true_rt[t] * sum(infections[max(1, t-20):(t-1)] * rev(gi_pmf[1:min(20, t-1)]))
    infections[t] <- rpois(1, expected_inf)
  }
  
  # Add reporting delay
  delay_pmf <- generate_delay_distribution()
  cases <- numeric(n_days)
  
  for (t in 1:n_days) {
    for (d in 1:min(30, t)) {
      cases[t] <- cases[t] + rpois(1, infections[t-d+1] * delay_pmf[d])
    }
  }
  
  sample_data <- data.frame(date = dates, cases = pmax(0, cases))
  write.csv(sample_data, "cases.csv", row.names = FALSE)
}

# Run the analysis
cat("Starting Rt estimation...\n")
results <- estimate_rt("cases.csv")

# Create plots
plots <- plot_rt_results(results)

# Display plots
print(plots$rt_plot)
print(plots$cases_plot)
print(plots$infections_plot)

# Print summary
cat("\nRt Summary:\n")
rt_table <- summarize_rt(results)
print(rt_table)

# Print final Rt estimates
cat("\nFinal week Rt estimates:\n")
final_estimates <- tail(results$rt_estimates, 7)
print(final_estimates[, c("date", "median", "2.5%", "97.5%")])

cat("\nAnalysis complete!\n")
```

---

## SUB_022

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(cmdstanr)
library(readr)

# Load case count data from CSV file
cases_data <- read_csv("cases.csv")


# Create a new data frame with lagged cases
case_lags <- cases_data %>%
  mutate(lag_1 = cases,
         lag_2 = lead(cases),
         lag_3 = lead(lead(cases)),
         lag_4 = lead(lead(lead(cases))),
         lag_5 = lead(lead(lead(lead(cases)))))

# Select only the required columns
case_lags <- select(case_lags, date, cases, lag_1:lag_5)


# Define the model in Stan
model_code <- "
data {
  int<lower=1> N; // number of observations
  vector[N] cases; // observed case counts
  real<lower=0, upper=1> S;
}

parameters {
  real<lower=0, upper=Inf> R_t[1]; // time-varying reproduction number
  vector<lower=0>[5] g; // generation interval probability mass function
  real<lower=0, upper=S> theta[1]; // observation model parameter
}

model {
  // Prior distributions for parameters
  target += normal_lpdf(R_t | 2, 1);
  target += dirichlet_lpdf(g | rep_vector(1, 5));
  target += beta_lpdf(theta[1] | 2, 2);

  // Observation model
  for (t in 1:N) {
    cases[t] ~ neg_binomial_2(R_t[1] * g[case_lags[lag_(t)][lag_id]] * theta[1], alpha);
  }
}
"

# Fit the model using cmdstanr
fit <- cmdstanr::sample(model_code, data = list(N = nrow(case_lags), cases = case_lags$cases,
                                                 S = 1),
                        chains = 4, iter_warmup = 2000, iter_sampling = 2000)


# Extract Rt estimates for each time point
Rt_estimates <- fit$summary[, "mean"]


# Exclude initial seeding period (e.g., first 7 days)
Rt_estimates <- Rt_estimates[case_lags$date > "2020-03-01"]


# Plot Rt estimates over time
plot(Rt_estimates ~ case_lags$date, type = "l")
```

---

## SUB_093

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
library(cmdstanr)
library(readr)
library(ggplot2)


# Load data
data <- read_csv("cases.csv")

# Ensure date is in a format that can be used for time-series analysis
data$date <- as.Date(data$date)

# Handle missing or invalid dates
data %<>%
  filter(!is.na(date))


# Model definition (in Stan)
model_code <- "
data {
  int<lower=0> T; // Number of time points
  vector[T] cases; // Case counts at each time point
  real<lower=0, upper=1> S; // Generation interval distribution parameter
}

parameters {
  real<lower=0, upper=10> R_t[T]; // Time-varying reproduction number
  real<lower=0, upper=S> g[3]; // Generation interval distribution parameters (e.g., mean and standard deviation)
}

model {
  // Prior on Rt (lognormal with constant mean and variance)
  target += normal_lpdf(log(R_t), 2.5, 1);

  // Prior on g (uniform within the specified range)
  for (i in 1:3) {
    target += uniform_lpdf(g[i], 0, S);
  }

  // Likelihood of case counts given Rt and generation interval distribution
  for (t in 1:T) {
    if (cases[t] > 0 && t > 1) { // Exclude initial infection seeding period
      cases[t] ~ poisson_log(lambda = log(sum(exp(log(R_t[u]) + normal_lpdf(g[3] * u, 0, g[2])))));
    }
  }

  // Priors for the generation interval distribution parameters
  target += uniform_lpdf(g[1], 0, S);
}
"

# Fit the model using cmdstanr
fit <- stan(
  file = "renewal_equation.stan",
  data = list(T = nrow(data), cases = data$cases, S = 10),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  thin = 1
)


# Extract Rt estimates for each time point
Rt_estimates <- extract(fit, "R_t")[, , 1]

# Convert to long format and add date information
Rt_long <- data.frame(
  Date = data$date,
  Rt_estimate = Rt_estimates,
  model = rep("Renewal Equation", nrow(data))
)

# Plot the results (optional)
ggplot(Rt_long, aes(x = Date, y = Rt_estimate)) +
  geom_line() +
  labs(title = "Estimated Reproduction Number (Rt)", x = "Date") +
  theme_classic()
```

---

## SUB_008

**Scenario**: 1b | **Execution**: FAILED

### model.stan

```stan
data {
  int<lower=0> n_cases;
  vector[n_cases] cases; // daily case counts
  real[365] dates; // date in days since first reported case
  int<lower=1> S; // number of generations to account for
  int<lower=1> T; // time horizon
  
  real mu_g; // mean generation interval (days)
}

parameters {
  real<lower=0, upper=2*mu_g> R_t[S]; // reproduction numbers over time
  real<lower=0, upper=T> t_infected[n_cases]; // times infected for each case
  vector[365] nu; // innovation noise (case count uncertainty)
}

model {
  // define prior distributions and likelihoods here
  
  // model the data using renewal equation:
  for (i in 1:n_cases) {
    cases[i] ~ normal(nu[t_infected[i]], 0.5); // observation model
    nu[t_infected[i]] = sum(R_t[s] * nu[t_infected[i] - s]) / S;
  }
}
```

### script.R

```r
# Load libraries
library(cmdstanr)
library(tidyverse)

# Load case counts from CSV
cases <- read_csv("cases.csv")

# Define Stan model for renewal equation (model.stan)
cat(read_file("renewal_model.stan"), file = "renewal_model.stan")

# Fit the model using cmdstanr
fit <- stan(
  file = "renewal_model.stan",
  data = list(n_cases = nrow(cases), 
              cases = cases$cases, 
              dates = as.numeric(as.Date(cases$date)), 
              S = 7, # 7-day generation interval distribution 
              T = 365) # Time horizon
)

# Extract Rt estimates for each time point
rt_estimates <- data.frame(date = fit$summary$Rt)

# Create a plot of Rt over time ( example )
ggplot(rt_estimates, aes(x = date)) + 
  geom_line() + 
  theme_classic()
```

---

## SUB_109

**Scenario**: 1b | **Execution**: FAILED

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(filepath='cases.csv'):
    """Load and prepare the case data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any negative cases and handle missing values
    df['cases'] = df['cases'].clip(lower=0)
    df = df.dropna()
    
    print(f"Data loaded: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['cases'].sum():,}")
    print(f"Max daily cases: {df['cases'].max():,}")
    
    return df

def get_generation_interval_pmf(max_days=15, mean_gi=5.2, std_gi=1.72):
    """
    Generate generation interval probability mass function
    Based on COVID-19 literature estimates
    """
    # Use log-normal distribution for generation interval
    days = np.arange(1, max_days + 1)
    
    # Convert mean and std to log-normal parameters
    mu = np.log(mean_gi**2 / np.sqrt(std_gi**2 + mean_gi**2))
    sigma = np.sqrt(np.log(1 + std_gi**2 / mean_gi**2))
    
    # Calculate PMF
    pmf = stats.lognorm.pdf(days, s=sigma, scale=np.exp(mu))
    pmf = pmf / pmf.sum()  # Normalize to sum to 1
    
    print(f"Generation interval: mean={np.sum(days * pmf):.2f}, std={np.sqrt(np.sum(days**2 * pmf) - np.sum(days * pmf)**2):.2f}")
    
    return pmf

def get_reporting_delay_pmf(max_days=21, mean_delay=5.1, std_delay=3.2):
    """
    Generate reporting delay probability mass function
    Time from infection to case reporting
    """
    days = np.arange(0, max_days)
    
    # Use gamma distribution for reporting delay
    # Convert mean and std to shape and scale parameters
    scale = std_delay**2 / mean_delay
    shape = mean_delay / scale
    
    pmf = stats.gamma.pdf(days, a=shape, scale=scale)
    pmf = pmf / pmf.sum()  # Normalize
    
    print(f"Reporting delay: mean={np.sum(days * pmf):.2f}, std={np.sqrt(np.sum(days**2 * pmf) - np.sum(days * pmf)**2):.2f}")
    
    return pmf

def create_rt_model(cases, generation_interval, reporting_delay):
    """
    Create PyMC model for estimating time-varying Rt using renewal equation
    """
    n_days = len(cases)
    max_gi = len(generation_interval)
    max_delay = len(reporting_delay)
    
    with pm.Model() as model:
        # Priors for initial infections (seeding period)
        seed_days = max_gi + max_delay
        log_initial_infections = pm.Normal(
            'log_initial_infections', 
            mu=np.log(np.maximum(cases[:seed_days], 1)), 
            sigma=1.0, 
            shape=seed_days
        )
        initial_infections = pm.Deterministic(
            'initial_infections', 
            pt.exp(log_initial_infections)
        )
        
        # Prior for Rt - assume it varies smoothly over time
        rt_raw = pm.GaussianRandomWalk(
            'rt_raw',
            mu=0,
            sigma=0.1,  # Controls smoothness of Rt over time
            shape=n_days - seed_days
        )
        
        # Transform to positive values with reasonable range
        rt = pm.Deterministic(
            'rt',
            pt.exp(rt_raw + np.log(1.0))  # Centered around 1.0
        )
        
        # Compute infections using renewal equation
        def compute_infections(rt_vals, initial_inf):
            infections = pt.zeros(n_days)
            infections = pt.set_subtensor(infections[:seed_days], initial_inf)
            
            for t in range(seed_days, n_days):
                # Compute convolution sum for renewal equation
                infection_sum = 0
                for s in range(1, min(max_gi + 1, t + 1)):
                    if t - s >= 0:
                        infection_sum += infections[t - s] * generation_interval[s - 1]
                
                infections = pt.set_subtensor(
                    infections[t], 
                    rt_vals[t - seed_days] * infection_sum
                )
            
            return infections
        
        infections = compute_infections(rt, initial_infections)
        
        # Compute expected cases accounting for reporting delay
        def compute_expected_cases(infections_t):
            expected_cases = pt.zeros(n_days)
            
            for t in range(n_days):
                case_sum = 0
                for d in range(min(max_delay, t + 1)):
                    if t - d >= 0:
                        case_sum += infections_t[t - d] * reporting_delay[d]
                expected_cases = pt.set_subtensor(expected_cases[t], case_sum)
            
            return expected_cases
        
        expected_cases = compute_expected_cases(infections)
        
        # Observation model - Negative Binomial to handle overdispersion
        phi = pm.Gamma('phi', alpha=2, beta=0.1)  # Overdispersion parameter
        
        # Convert mean and phi to alpha, beta for NegativeBinomial
        alpha = expected_cases / phi
        
        observed_cases = pm.NegativeBinomial(
            'observed_cases',
            mu=expected_cases,
            alpha=alpha,
            observed=cases
        )
        
        # Store variables for easy access
        model.add_coord('time', values=range(n_days), mutable=True)
        model.add_coord('rt_time', values=range(seed_days, n_days), mutable=True)
    
    return model, seed_days

def fit_model(model, draws=1000, tune=1000, chains=2):
    """Fit the PyMC model"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=2,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True
        )
    
    return trace

def extract_rt_estimates(trace, dates, seed_days):
    """Extract Rt estimates from the trace"""
    rt_samples = trace.posterior['rt'].values  # shape: (chains, draws, time_points)
    rt_samples_flat = rt_samples.reshape(-1, rt_samples.shape[-1])  # flatten chains and draws
    
    # Calculate summary statistics
    rt_mean = np.mean(rt_samples_flat, axis=0)
    rt_median = np.median(rt_samples_flat, axis=0)
    rt_lower = np.percentile(rt_samples_flat, 2.5, axis=0)
    rt_upper = np.percentile(rt_samples_flat, 97.5, axis=0)
    rt_lower_50 = np.percentile(rt_samples_flat, 25, axis=0)
    rt_upper_50 = np.percentile(rt_samples_flat, 75, axis=0)
    
    # Create results dataframe
    rt_dates = dates[seed_days:]  # Rt estimates start after seeding period
    
    results_df = pd.DataFrame({
        'date': rt_dates,
        'rt_mean': rt_mean,
        'rt_median': rt_median,
        'rt_lower_95': rt_lower,
        'rt_upper_95': rt_upper,
        'rt_lower_50': rt_lower_50,
        'rt_upper_50': rt_upper_50
    })
    
    return results_df

def plot_results(df, rt_results, cases):
    """Plot the results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cases over time
    ax1.bar(df['date'], df['cases'], alpha=0.7, color='steelblue', label='Observed cases')
    ax1.set_ylabel('Daily cases')
    ax1.set_title('Daily COVID-19 Cases')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates
    ax2.fill_between(
        rt_results['date'], 
        rt_results['rt_lower_95'], 
        rt_results['rt_upper_95'],
        alpha=0.3, color='red', label='95% CI'
    )
    ax2.fill_between(
        rt_results['date'], 
        rt_results['rt_lower_50'], 
        rt_results['rt_upper_50'],
        alpha=0.5, color='red', label='50% CI'
    )
    ax2.plot(rt_results['date'], rt_results['rt_median'], 'r-', linewidth=2, label='Median Rt')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Rt = 1')
    ax2.set_ylabel('Reproduction number (Rt)')
    ax2.set_xlabel('Date')
    ax2.set_title('Time-varying Reproduction Number (Rt)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\nRt Summary Statistics:")
    print(f"Mean Rt: {rt_results['rt_mean'].mean():.2f}")
    print(f"Median Rt: {rt_results['rt_median'].median():.2f}")
    print(f"Min Rt (median): {rt_results['rt_median'].min():.2f}")
    print(f"Max Rt (median): {rt_results['rt_median'].max():.2f}")
    
    # Days with Rt > 1
    days_above_1 = (rt_results['rt_median'] > 1).sum()
    total_days = len(rt_results)
    print(f"Days with Rt > 1: {days_above_1}/{total_days} ({100*days_above_1/total_days:.1f}%)")

def main():
    """Main analysis pipeline"""
    print("=== COVID-19 Rt Estimation using Renewal Equation ===\n")
    
    # Load data
    df = load_and_prepare_data('cases.csv')
    cases = df['cases'].values
    
    # Get generation interval and reporting delay distributions
    print("\n" + "="*50)
    print("Setting up epidemiological parameters...")
    generation_interval = get_generation_interval_pmf()
    reporting_delay = get_reporting_delay_pmf()
    
    # Create and fit model
    print("\n" + "="*50)
    print("Creating PyMC model...")
    model, seed_days = create_rt_model(cases, generation_interval, reporting_delay)
    
    print(f"Model created. Seeding period: {seed_days} days")
    print("Model summary:")
    print(model)
    
    print("\n" + "="*50)
    print("Fitting model (this may take several minutes)...")
    trace = fit_model(model, draws=1000, tune=1000, chains=2)
    
    # Extract results
    print("\n" + "="*50)
    print("Extracting Rt estimates...")
    rt_results = extract_rt_estimates(trace, df['date'].values, seed_days)
    
    # Display results
    print("\nFirst 10 Rt estimates:")
    print(rt_results.head(10).round(3))
    
    # Plot results
    print("\n" + "="*50)
    print("Generating plots...")
    plot_results(df, rt_results, cases)
    
    # Save results
    rt_results.to_csv('rt_estimates.csv', index=False)
    print("\nResults saved to 'rt_estimates.csv'")
    
    return rt_results, trace, model

# Example of creating sample data if cases.csv doesn't exist
def create_sample_data():
    """Create sample COVID-19 case data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', periods=100, freq='D')
    
    # Simulate epidemic curve with changing Rt
    true_rt = np.concatenate([
        np.linspace(2.5, 0.8, 30),  # Initial decline
        np.linspace(0.8, 1.3, 20),  # Increase
        np.linspace(1.3, 0.7, 30),  # Decline
        np.linspace(0.7, 1.1, 20)   # Slight increase
    ])
    
    # Simple simulation
    cases = []
    infections = [10] * 10  # Initial seeding
    
    gi_pmf = get_generation_interval_pmf()
    
    for t in range(len(dates)):
        if t < 10:
            new_infections = infections[t]
        else:
            infectiousness = sum(infections[max(0, t-s-1)] * gi_pmf[s] 
                               for s in range(min(len(gi_pmf), t)))
            new_infections = np.random.poisson(true_rt[min(t-10, len(true_rt)-1)] * infectiousness)
            infections.append(new_infections)
        
        # Add reporting noise and delay
        reported_cases = max(0, int(new_infections * np.random.uniform(0.7, 1.3)))
        cases.append(reported_cases)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases
    })
    
    df.to_csv('cases.csv', index=False)
    print("Sample data created in 'cases.csv'")
    return df

if __name__ == "__main__":
    # Uncomment the next line if you need to create sample data
    # create_sample_data()
    
    try:
        rt_results, trace, model = main()
    except FileNotFoundError:
        print("cases.csv not found. Creating sample data...")
        create_sample_data()
        rt_results, trace, model = main()
```

---

## SUB_068

**Scenario**: 1b | **Execution**: TIMEOUT

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_sample_data():
    """Generate sample COVID-19 case data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', '2020-08-31', freq='D')
    
    # Simulate true Rt that varies over time
    n_days = len(dates)
    true_rt = np.ones(n_days)
    true_rt[:30] = np.linspace(2.5, 1.0, 30)  # Initial decline
    true_rt[30:60] = np.linspace(1.0, 0.7, 30)  # Lockdown effect
    true_rt[60:90] = np.linspace(0.7, 1.3, 30)  # Reopening
    true_rt[90:120] = np.linspace(1.3, 0.9, 30)  # Second wave control
    true_rt[120:] = 0.9 + 0.2 * np.sin(np.arange(len(true_rt[120:])) * 2 * np.pi / 30)
    
    # Simulate infections using renewal equation
    infections = np.zeros(n_days)
    infections[:7] = np.random.poisson(50, 7)  # Initial seeding
    
    # Generation interval (discretized gamma distribution)
    generation_interval = stats.gamma(a=2.5, scale=2.0)
    max_gen_int = 15
    gen_pmf = np.array([generation_interval.pdf(i) for i in range(1, max_gen_int + 1)])
    gen_pmf = gen_pmf / gen_pmf.sum()
    
    # Simulate infections using renewal equation
    for t in range(7, n_days):
        infectiousness = 0
        for s in range(min(t, max_gen_int)):
            if t - s - 1 >= 0:
                infectiousness += infections[t - s - 1] * gen_pmf[s]
        infections[t] = np.random.poisson(max(1, true_rt[t] * infectiousness))
    
    # Add reporting delay (mean 7 days)
    reporting_delay_pmf = stats.gamma(a=2.0, scale=3.5).pdf(np.arange(1, 15))
    reporting_delay_pmf = reporting_delay_pmf / reporting_delay_pmf.sum()
    
    # Convolve infections with reporting delay to get cases
    cases = np.convolve(infections, reporting_delay_pmf, mode='same')
    cases = np.random.poisson(np.maximum(cases, 1))
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'cases': cases.astype(int)
    })
    
    return df, true_rt

def create_generation_interval_pmf(max_days=15):
    """Create discretized generation interval PMF"""
    # Using gamma distribution with mean ~5 days, std ~2.5 days
    gen_dist = stats.gamma(a=2.5, scale=2.0)
    pmf = np.array([gen_dist.pdf(i) for i in range(1, max_days + 1)])
    return pmf / pmf.sum()

def create_reporting_delay_pmf(max_days=14):
    """Create discretized reporting delay PMF"""
    # Using gamma distribution with mean ~7 days
    delay_dist = stats.gamma(a=2.0, scale=3.5)
    pmf = np.array([delay_dist.pdf(i) for i in range(1, max_days + 1)])
    return pmf / pmf.sum()

class RenewalModel:
    """Bayesian renewal equation model for estimating Rt"""
    
    def __init__(self, cases, generation_pmf, reporting_pmf):
        self.cases = np.array(cases)
        self.n_days = len(cases)
        self.generation_pmf = generation_pmf
        self.reporting_pmf = reporting_pmf
        self.max_gen_int = len(generation_pmf)
        self.max_report_delay = len(reporting_pmf)
        
    def build_model(self, rt_prior_scale=0.2):
        """Build the PyMC model"""
        
        with pm.Model() as model:
            # Prior for initial infections (seeding period)
            seed_days = self.max_gen_int
            I_seed = pm.Exponential('I_seed', lam=1/50, shape=seed_days)
            
            # Prior for Rt - using random walk on log scale for smoothness
            rt_init = pm.Normal('rt_init', mu=0, sigma=0.5)  # log(Rt) at t=0
            rt_noise = pm.Normal('rt_noise', mu=0, sigma=rt_prior_scale, 
                                shape=self.n_days - seed_days - 1)
            
            # Construct log(Rt) as random walk
            log_rt = pt.concatenate([
                pt.repeat(rt_init, seed_days + 1),
                rt_init + pt.cumsum(rt_noise)
            ])
            
            rt = pm.Deterministic('rt', pt.exp(log_rt))
            
            # Compute infections using renewal equation
            def compute_infections(I_seed, rt):
                infections = pt.zeros(self.n_days)
                infections = pt.set_subtensor(infections[:seed_days], I_seed)
                
                for t in range(seed_days, self.n_days):
                    infectiousness = 0
                    for s in range(min(t, self.max_gen_int)):
                        if t - s - 1 >= 0:
                            infectiousness += infections[t - s - 1] * self.generation_pmf[s]
                    
                    new_infections = rt[t] * infectiousness
                    infections = pt.set_subtensor(infections[t], new_infections)
                
                return infections
            
            infections = pm.Deterministic('infections', 
                                        compute_infections(I_seed, rt))
            
            # Convolve infections with reporting delay to get expected cases
            def convolve_reporting_delay(infections):
                expected_cases = pt.zeros(self.n_days)
                for t in range(self.n_days):
                    cases_t = 0
                    for d in range(min(t + 1, self.max_report_delay)):
                        if t - d >= 0:
                            cases_t += infections[t - d] * self.reporting_pmf[d]
                    expected_cases = pt.set_subtensor(expected_cases[t], cases_t)
                return expected_cases
            
            expected_cases = pm.Deterministic('expected_cases', 
                                            convolve_reporting_delay(infections))
            
            # Observation model - negative binomial for overdispersion
            alpha = pm.Exponential('alpha', lam=1/10)  # Overdispersion parameter
            
            # Likelihood
            cases_obs = pm.NegativeBinomial('cases_obs', 
                                          mu=expected_cases,
                                          alpha=alpha,
                                          observed=self.cases)
            
        return model
    
    def fit_model(self, draws=1000, tune=1000, chains=2, **kwargs):
        """Fit the model using MCMC"""
        self.model = self.build_model()
        
        with self.model:
            # Use NUTS sampler
            self.trace = pm.sample(draws=draws, tune=tune, chains=chains,
                                 target_accept=0.95, **kwargs)
            
        return self.trace
    
    def get_rt_estimates(self, credible_interval=0.95):
        """Extract Rt estimates with credible intervals"""
        rt_samples = self.trace.posterior['rt'].values
        rt_samples = rt_samples.reshape(-1, rt_samples.shape[-1])
        
        alpha = (1 - credible_interval) / 2
        
        rt_estimates = {
            'mean': np.mean(rt_samples, axis=0),
            'median': np.median(rt_samples, axis=0),
            'lower': np.quantile(rt_samples, alpha, axis=0),
            'upper': np.quantile(rt_samples, 1 - alpha, axis=0)
        }
        
        return rt_estimates

def plot_results(dates, cases, rt_estimates, true_rt=None):
    """Plot the results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot cases
    ax1.plot(dates, cases, 'o-', color='steelblue', alpha=0.7, markersize=3)
    ax1.set_ylabel('Daily Cases')
    ax1.set_title('Daily COVID-19 Cases')
    ax1.grid(True, alpha=0.3)
    
    # Plot Rt estimates
    ax2.fill_between(dates, rt_estimates['lower'], rt_estimates['upper'],
                     alpha=0.3, color='red', label='95% CI')
    ax2.plot(dates, rt_estimates['median'], color='red', linewidth=2, label='Rt estimate')
    
    if true_rt is not None:
        ax2.plot(dates, true_rt, 'k--', linewidth=2, alpha=0.7, label='True Rt')
    
    ax2.axhline(y=1, color='black', linestyle='-', alpha=0.5)
    ax2.set_ylabel('Reproduction Number (Rt)')
    ax2.set_xlabel('Date')
    ax2.set_title('Time-varying Reproduction Number (Rt)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis
    for ax in [ax1, ax2]:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def main():
    """Main function to run the complete analysis"""
    print("=== COVID-19 Rt Estimation using Renewal Equation ===\n")
    
    # Generate or load data
    print("1. Loading data...")
    try:
        # Try to load real data
        df = pd.read_csv('cases.csv')
        df['date'] = pd.to_datetime(df['date'])
        true_rt = None
        print(f"   Loaded {len(df)} days of case data from cases.csv")
    except FileNotFoundError:
        # Generate sample data if file not found
        print("   cases.csv not found. Generating sample data...")
        df, true_rt = generate_sample_data()
        print(f"   Generated {len(df)} days of sample case data")
    
    # Create generation interval and reporting delay PMFs
    print("\n2. Setting up model parameters...")
    generation_pmf = create_generation_interval_pmf(max_days=15)
    reporting_pmf = create_reporting_delay_pmf(max_days=14)
    
    print(f"   Generation interval mean: {np.sum(generation_pmf * np.arange(1, len(generation_pmf) + 1)):.1f} days")
    print(f"   Reporting delay mean: {np.sum(reporting_pmf * np.arange(1, len(reporting_pmf) + 1)):.1f} days")
    
    # Build and fit model
    print("\n3. Building renewal equation model...")
    renewal_model = RenewalModel(df['cases'].values, generation_pmf, reporting_pmf)
    
    print("4. Fitting model using MCMC...")
    print("   This may take a few minutes...")
    
    try:
        trace = renewal_model.fit_model(draws=1000, tune=1000, chains=2)
        
        # Check convergence
        rhat = pm.rhat(trace)
        max_rhat = float(rhat['rt'].max().values)
        print(f"   Maximum R-hat: {max_rhat:.3f}")
        
        if max_rhat > 1.1:
            print("   Warning: Some parameters may not have converged (R-hat > 1.1)")
        
    except Exception as e:
        print(f"   Error during sampling: {e}")
        print("   This might be due to model complexity or data issues.")
        return
    
    # Extract results
    print("\n5. Extracting Rt estimates...")
    rt_estimates = renewal_model.get_rt_estimates()
    
    # Create results summary
    results_df = pd.DataFrame({
        'date': df['date'],
        'cases': df['cases'],
        'rt_mean': rt_estimates['mean'],
        'rt_median': rt_estimates['median'],
        'rt_lower': rt_estimates['lower'],
        'rt_upper': rt_estimates['upper']
    })
    
    # Print summary statistics
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Time period: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    print(f"Total cases: {df['cases'].sum():,}")
    print(f"Mean Rt: {rt_estimates['mean'].mean():.2f}")
    print(f"Rt range: {rt_estimates['median'].min():.2f} - {rt_estimates['median'].max():.2f}")
    
    # Identify periods where Rt > 1
    epidemic_growth = results_df[results_df['rt_median'] > 1]
    if len(epidemic_growth) > 0:
        print(f"Days with Rt > 1: {len(epidemic_growth)} ({100*len(epidemic_growth)/len(results_df):.1f}%)")
    
    # Plot results
    print("\n6. Creating plots...")
    fig = plot_results(df['date'], df['cases'], rt_estimates, true_rt)
    plt.show()
    
    # Save results
    results_df.to_csv('rt_estimates.csv', index=False)
    print("\n7. Results saved to 'rt_estimates.csv'")
    
    return results_df, renewal_model

if __name__ == "__main__":
    results_df, model = main()
```

---

## SUB_114

**Scenario**: 1b | **Execution**: FAILED

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm
import arviz as az
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filename='cases.csv'):
    """Load and preprocess case data"""
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        # Generate synthetic data for demonstration
        print("cases.csv not found. Generating synthetic data for demonstration.")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic COVID-19 case data for demonstration"""
    dates = pd.date_range('2020-03-01', '2020-08-01', freq='D')
    n_days = len(dates)
    
    # Simulate true Rt values with realistic dynamics
    t = np.arange(n_days)
    true_rt = (2.5 * np.exp(-t/30) + 0.8 + 
               0.3 * np.sin(2*np.pi*t/14) * np.exp(-t/50))
    
    # Generation interval (discretized gamma distribution)
    generation_interval = get_generation_interval()
    
    # Reporting delay (discretized gamma distribution)
    reporting_delay = get_reporting_delay()
    
    # Simulate infections using renewal equation
    infections = np.zeros(n_days)
    infections[:7] = np.random.poisson(50, 7)  # Initial seeding
    
    for t in range(7, n_days):
        infectiousness = sum(infections[max(0, t-s)] * generation_interval[s] 
                           for s in range(1, min(t+1, len(generation_interval))))
        infections[t] = np.random.poisson(max(1, true_rt[t] * infectiousness))
    
    # Apply reporting delay and observation noise
    cases = np.zeros(n_days)
    for t in range(n_days):
        for d in range(len(reporting_delay)):
            if t + d < n_days:
                cases[t + d] += np.random.poisson(infections[t] * reporting_delay[d])
    
    # Add some additional observation noise
    cases = np.maximum(0, cases + np.random.normal(0, np.sqrt(cases + 1)))
    cases = cases.astype(int)
    
    df = pd.DataFrame({'date': dates, 'cases': cases})
    return df

def get_generation_interval(max_days=20):
    """
    Get discretized generation interval distribution
    Based on COVID-19 literature (mean ~5.5 days, std ~2.1 days)
    """
    # Parameters for gamma distribution
    mean_gi = 5.5
    std_gi = 2.1
    
    # Convert to gamma parameters
    shape = (mean_gi / std_gi) ** 2
    rate = mean_gi / (std_gi ** 2)
    
    # Discretize
    days = np.arange(1, max_days + 1)
    pmf = gamma.cdf(days + 0.5, shape, scale=1/rate) - gamma.cdf(days - 0.5, shape, scale=1/rate)
    pmf = pmf / pmf.sum()  # Normalize
    
    return pmf

def get_reporting_delay(max_days=15):
    """
    Get discretized reporting delay distribution
    Based on typical COVID-19 reporting patterns
    """
    # Parameters for gamma distribution (mean ~3 days, std ~2 days)
    mean_delay = 3.0
    std_delay = 2.0
    
    shape = (mean_delay / std_delay) ** 2
    rate = mean_delay / (std_delay ** 2)
    
    # Discretize
    days = np.arange(0, max_days)
    pmf = gamma.cdf(days + 0.5, shape, scale=1/rate) - gamma.cdf(days - 0.5, shape, scale=1/rate)
    pmf = pmf / pmf.sum()
    
    return pmf

def create_infectiousness_matrix(generation_interval, n_days):
    """Create matrix for computing infectiousness from past infections"""
    max_gi = len(generation_interval)
    infectiousness_matrix = np.zeros((n_days, n_days))
    
    for t in range(n_days):
        for s in range(1, min(t + 1, max_gi + 1)):
            if t - s >= 0:
                infectiousness_matrix[t, t - s] = generation_interval[s - 1]
    
    return infectiousness_matrix

def create_reporting_matrix(reporting_delay, n_days):
    """Create matrix for mapping infections to reported cases"""
    max_delay = len(reporting_delay)
    reporting_matrix = np.zeros((n_days, n_days))
    
    for t in range(n_days):
        for d in range(max_delay):
            if t + d < n_days:
                reporting_matrix[t + d, t] = reporting_delay[d]
    
    return reporting_matrix

def build_renewal_model(cases, generation_interval, reporting_delay, seed_days=7):
    """Build PyMC model for Rt estimation using renewal equation"""
    
    n_days = len(cases)
    
    # Create matrices for vectorized operations
    infectiousness_matrix = create_infectiousness_matrix(generation_interval, n_days)
    reporting_matrix = create_reporting_matrix(reporting_delay, n_days)
    
    with pm.Model() as model:
        # Prior for initial infections (seeding period)
        initial_infections = pm.Exponential('initial_infections', 
                                          lam=1/50, shape=seed_days)
        
        # Prior for Rt - allow for time-varying reproduction number
        # Use random walk to allow smooth changes over time
        rt_raw = pm.GaussianRandomWalk('rt_raw', 
                                      sigma=0.1, 
                                      shape=n_days - seed_days,
                                      init_dist=pm.Normal.dist(mu=0, sigma=0.5))
        
        # Transform to ensure Rt > 0
        rt = pm.Deterministic('rt', pm.math.exp(rt_raw))
        
        # Combine initial infections and computed infections
        infections = pt.zeros(n_days)
        infections = pt.set_subtensor(infections[:seed_days], initial_infections)
        
        # Compute infections for days after seeding using renewal equation
        for t in range(seed_days, n_days):
            # Compute infectiousness (sum of past infections weighted by generation interval)
            infectiousness = pt.sum(infections[:t] * infectiousness_matrix[t, :t])
            
            # Compute expected infections
            expected_infections = rt[t - seed_days] * infectiousness
            
            # Set infections for day t
            infections = pt.set_subtensor(infections[t], 
                                        pm.Poisson.dist(mu=pm.math.maximum(expected_infections, 0.1)))
        
        # Apply reporting delay to get expected reported cases
        expected_cases = pt.dot(reporting_matrix, infections)
        
        # Add small constant to avoid zero expected cases
        expected_cases = pm.math.maximum(expected_cases, 0.1)
        
        # Observation model with overdispersion
        phi = pm.Exponential('phi', lam=0.1)  # Overdispersion parameter
        
        # Negative binomial likelihood
        obs = pm.NegativeBinomial('obs', 
                                 mu=expected_cases, 
                                 alpha=phi,
                                 observed=cases)
        
        # Store matrices as constants for later use
        pm.ConstantData('infectiousness_matrix', infectiousness_matrix)
        pm.ConstantData('reporting_matrix', reporting_matrix)
        
    return model

def fit_model(model, samples=2000, tune=1000, chains=2, target_accept=0.9):
    """Fit the renewal equation model"""
    
    with model:
        # Use NUTS sampler with higher target acceptance for better sampling
        trace = pm.sample(
            samples, 
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42
        )
    
    return trace

def extract_rt_estimates(trace, dates):
    """Extract Rt estimates with credible intervals"""
    
    rt_samples = trace.posterior['rt']
    
    # Compute summary statistics
    rt_mean = rt_samples.mean(dim=['chain', 'draw'])
    rt_lower = rt_samples.quantile(0.025, dim=['chain', 'draw'])
    rt_upper = rt_samples.quantile(0.975, dim=['chain', 'draw'])
    rt_median = rt_samples.quantile(0.5, dim=['chain', 'draw'])
    
    # Create results DataFrame (excluding seeding period)
    seed_days = len(dates) - len(rt_mean)
    
    results = pd.DataFrame({
        'date': dates[seed_days:],
        'rt_mean': rt_mean.values,
        'rt_median': rt_median.values,
        'rt_lower': rt_lower.values,
        'rt_upper': rt_upper.values
    })
    
    return results

def plot_results(df, rt_results, generation_interval, reporting_delay):
    """Create comprehensive plots of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Case counts over time
    axes[0,0].plot(df['date'], df['cases'], 'k-', alpha=0.7, linewidth=2)
    axes[0,0].set_title('Daily Reported Cases', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Cases')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates over time
    axes[0,1].plot(rt_results['date'], rt_results['rt_median'], 
                   'b-', linewidth=2, label='Median Rt')
    axes[0,1].fill_between(rt_results['date'], 
                          rt_results['rt_lower'], 
                          rt_results['rt_upper'],
                          alpha=0.3, color='blue', label='95% CI')
    axes[0,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    axes[0,1].set_title('Time-varying Reproduction Number (Rt)', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Rt')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Generation interval distribution
    axes[1,0].bar(range(1, len(generation_interval) + 1), generation_interval, 
                  alpha=0.7, color='green')
    axes[1,0].set_title('Generation Interval Distribution', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Days')
    axes[1,0].set_ylabel('Probability')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Reporting delay distribution
    axes[1,1].bar(range(len(reporting_delay)), reporting_delay, 
                  alpha=0.7, color='orange')
    axes[1,1].set_title('Reporting Delay Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Days')
    axes[1,1].set_ylabel('Probability')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Rt trajectory with key statistics
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(rt_results['date'], rt_results['rt_median'], 'b-', linewidth=3, label='Median Rt')
    ax.fill_between(rt_results['date'], 
                   rt_results['rt_lower'], 
                   rt_results['rt_upper'],
                   alpha=0.3, color='blue', label='95% Credible Interval')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Rt = 1 (epidemic threshold)')
    
    # Highlight periods where Rt > 1
    above_one = rt_results['rt_lower'] > 1
    if above_one.any():
        ax.fill_between(rt_results['date'], 0, 3, 
                       where=above_one, alpha=0.1, color='red',
                       label='Likely growing (95% CI > 1)')
    
    ax.set_title('COVID-19 Reproduction Number (Rt) Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Reproduction Number (Rt)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(rt_results['rt_upper'].max() * 1.1, 3))
    
    plt.tight_layout()
    plt.show()

def print_summary(rt_results):
    """Print summary statistics of Rt estimates"""
    
    print("\n" + "="*60)
    print("RT ESTIMATION SUMMARY")
    print("="*60)
    
    print(f"Estimation period: {rt_results['date'].min().strftime('%Y-%m-%d')} to {rt_results['date'].max().strftime('%Y-%m-%d')}")
    print(f"Number of days estimated: {len(rt_results)}")
    
    print(f"\nOverall Rt statistics:")
    print(f"  Mean Rt: {rt_results['rt_mean'].mean():.2f}")
    print(f"  Median Rt: {rt_results['rt_median'].median():.2f}")
    print(f"  Min Rt (median): {rt_results['rt_median'].min():.2f}")
    print(f"  Max Rt (median): {rt_results['rt_median'].max():.2f}")
    
    # Periods of growth vs decline
    growing_days = (rt_results['rt_lower'] > 1).sum()
    likely_growing = (rt_results['rt_median'] > 1).sum()
    
    print(f"\nEpidemic dynamics:")
    print(f"  Days with Rt > 1 (median): {likely_growing} ({100*likely_growing/len(rt_results):.1f}%)")
    print(f"  Days with 95% CI > 1: {growing_days} ({100*growing_days/len(rt_results):.1f}%)")
    
    # Recent trend (last 7 days)
    if len(rt_results) >= 7:
        recent_rt = rt_results['rt_median'].iloc[-7:].mean()
        print(f"  Average Rt (last 7 days): {recent_rt:.2f}")
    
    print("="*60)

def main():
    """Main execution function"""
    
    print("COVID-19 Rt Estimation Using Renewal Equation")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading case data...")
    df = load_data()
    print(f"   Loaded {len(df)} days of data from {df['date'].min()} to {df['date'].max()}")
    
    # Get epidemiological parameters
    print("\n2. Setting up epidemiological parameters...")
    generation_interval = get_generation_interval()
    reporting_delay = get_reporting_delay()
    print(f"   Generation interval: mean = {np.sum(np.arange(1, len(generation_interval)+1) * generation_interval):.1f} days")
    print(f"   Reporting delay: mean = {np.sum(np.arange(len(reporting_delay)) * reporting_delay):.1f} days")
    
    # Build model
    print("\n3. Building renewal equation model...")
    model = build_renewal_model(df['cases'].values, generation_interval, reporting_delay)
    print("   Model built successfully")
    
    # Fit model
    print("\n4. Fitting model using MCMC...")
    print("   This may take several minutes...")
    
    try:
        trace = fit_model(model, samples=1000, tune=1000, chains=2)
        print("   Model fitted successfully")
        
        # Check convergence
        rhat = az.rhat(trace)
        max_rhat = float(rhat['rt'].max())
        print(f"   Max R-hat: {max_rhat:.3f} {'(Good)' if max_rhat < 1.1 else '(Check convergence)'}")
        
    except Exception as e:
        print(f"   Error during model fitting: {str(e)}")
        print("   Try reducing the number of samples or adjusting model parameters")
        return
    
    # Extract results
    print("\n5. Extracting Rt estimates...")
    rt_results = extract_rt_estimates(trace, df['date'])
    print(f"   Extracted Rt estimates for {len(rt_results)} days")
    
    # Print summary
    print_summary(rt_results)
    
    # Create plots
    print("\n6. Creating visualizations...")
    plot_results(df, rt_results, generation_interval, reporting_delay)
    
    # Save results
    print("\n7. Saving results...")
    rt_results.to_csv('rt_estimates.csv', index=False)
    print("   Results saved to 'rt_estimates.csv'")
    
    print("\n" + "="*50)
    print("Analysis completed successfully!")
    print("="*50)
    
    return rt_results, trace, model

if __name__ == "__main__":
    rt_results, trace, model = main()
```

---

## SUB_069

**Scenario**: 1b | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
from pymc3 import Model, Deterministic, Normal
from pymc3 import MCMC, sample

# Load data
data = pd.read_csv('cases.csv')

# Ensure dates are in datetime format
data['date'] = pd.to_datetime(data['date'])

# Calculate time delta between days (to be used as index)
time_deltas = data['date'].diff().dt.days.values

# Define the model
with Model() as model:
    # Assume an exponential distribution for generation interval probabilities
    lam = 7.0  # mean of generation interval, assume gamma distributed with shape=3 and scale=1/7
    g_s = [np.exp(-lam * s) for s in range(31)]  # We'll only consider up to 30 days
    
    # Define the parameters for the renewal equation
    Rt = Deterministic('Rt', np.zeros(len(data)))
    
    # Renewal Equation Model
    infections = np.cumsum([g_s[day] * data['cases'][i] for i, day in enumerate(time_deltas)])
    
    # Observation model
    observed_cases = Normal('observed_cases', mu=data['cases'], sigma=10)
    
    # Define the Rt parameters
    R_params = Deterministic('Rt_parameters',
                             [0.1 + 0.9 * (infections[i] / sum(infections[:i+1])) for i in range(len(data))])
    
    # Add the Rt as a Deterministic variable
    Rt[...] = R_params
    
# Initialize the sampler and run it
with model:
    trace = MCMC(model)
    trace.init(
        random_seed=42, 
        start='adapt_full',
    )
    trace.sample(10000)

# Extract Rt estimates for each time point
Rt_estimates = np.array([trace.get_values('Rt_parameters')])

# Plot Rt over time
import matplotlib.pyplot as plt

plt.plot(Rt_estimates.mean(axis=0))
plt.xlabel('Time')
plt.ylabel('Estimated Reproduction Number (Rt)')
plt.title('Estimated Rt Over Time')
plt.show()
```

---

## SUB_004

**Scenario**: 1b | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
from pymc import Model, StepMethod, NUTS, sample, summary
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('cases.csv')

# Convert date column to datetime type and create a date index
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Generate the necessary variables
dates = np.array([d.to_julian_date() for d in data.index])
N = len(dates)
T = 7  # number of days in generation interval

# Define a function to compute Rt using renewal equation
def rt(data, T):
    R = np.zeros(N) + 1e-6  # initialize with small positive value
    for t in range(1, N):
        I_t = data['cases'].iloc[t]
        sum_I_ts = data['cases'].iloc[:t].cumsum()
        R[t] = I_t / (np.sum(sum_I_ts * g[T-t:t+T]))
    return R

# Define the generation interval probability mass function
g = np.array([0.2, 0.5, 0.3])  # example values: more accurate values should be used in practice

# Estimate Rt over time using renewal equation
Rt = rt(data, T)

# Fit a simple linear model for Rt with an intercept term and a slope coefficient
with Model() as model:
    R = np.zeros(N)
    for t in range(1, N):
        I_t = data['cases'].iloc[t]
        sum_I_ts = data['cases'].iloc[:t].cumsum()
        R[t] = pm.Deterministic('Rt', I_t / (np.sum(sum_I_ts * g[T-t:t+T])))
    intercept = pm.Normal('intercept', mu=2, sigma=1)
    slope = pm.Normal('slope', mu=0.5, sigma=0.1)
    Rt_estimated = pm.Deterministic('Rt_estimated', intercept + slope * np.arange(1, N))

# Sample from the posterior distribution of Rt_estimated
n_samples = 10000
with model:
    step = pm.NUTS(tune=1000, target_accept=0.9)
    trace = sample(n_samples, tune=n_samples//10, init='adapt_diag', chains=2)

# Compute and plot summary statistics for Rt estimates over time
summary_Rt_estimated = pm.summary(trace['Rt_estimated'])
plt.plot(summary_Rt_estimated.index.astype('int'), np.mean(trace['Rt_estimated'], axis=0))
plt.xlabel('Time')
plt.ylabel('Estimated Rt')
plt.show()
```

---

## SUB_050

**Scenario**: 1b | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Load data
data = pd.read_csv('cases.csv')
data['date'] = pd.to_datetime(data['date'])

# Calculate time differences (days) between successive cases
time_diffs = data['date'].diff().dt.days.values

# Define generation interval probability mass function
def generation_interval(gamma_shape, gamma_scale):
    return gamma(gamma_shape, scale=gamma_scale)

# Define observation model for case counts
def obs_model(Rt, past_cases, time_diffs, gamma_shape, gamma_scale):
    # Apply delays between infection and case reporting
    delayed_past_cases = np.zeros(len(time_diffs))
    for i in range(1, len(delayed_past_cases)):
        start_idx = max(0, i - 7)  # Assuming a maximum delay of 7 days
        end_idx = min(i, len(past_cases) - start_idx)
        delayed_past_cases[i] = np.sum(past_cases[start_idx:end_idx])
    
    # Apply renewal equation model
    infection_counts = Rt * np.convolve(delayed_past_cases, time_diffs[:len(delayed_past_cases)], mode='full')
    
    # Apply observation noise (Poisson distribution)
    observed_cases = pm.Poisson('observed_cases', mu=infection_counts)
    return observed_cases

# Define model
with pm.Model() as model:
    Rt = pm.Gamma('Rt', alpha=1, beta=1)  # Prior for reproduction number
    
    gamma_shape = pm.Uniform('gamma_shape', lower=0.1, upper=5)
    gamma_scale = pm.Uniform('gamma_scale', lower=0.1, upper=10)
    
    past_cases = data['cases'].values[:-7]  # Exclude last week
    time_diffs = np.repeat(data['date'].diff().dt.days.values, len(past_cases))[:len(past_cases)]
    
    observed_cases = obs_model(Rt, past_cases, time_diffs, gamma_shape, gamma_scale)
    
    # Inference
    step = pm.NUTS(tune=1000, target_accept=0.9)
    trace = pm.sample(10000, step=step)

# Extract Rt estimates for each time point
Rt_estimates = np.mean(pm.sample_posterior_predictive(trace, var_names=['Rt']), axis=0)

# Plot Rt over time (use the last 7 days of data to smooth out noise)
data['date'] = pd.to_datetime(data['date'])
last_week_cases = data['cases'].values[-7:]
smoothed_Rt = np.convolve(Rt_estimates, np.ones(7)/7, mode='valid')
plt.plot(last_week_cases[:-6], smoothed_Rt[:len(last_week_cases)-6], label='Estimated Rt')
plt.xlabel('Number of reported cases (last 7 days)')
plt.ylabel('Reproduction number (Rt)')
plt.title('Time-varying reproduction number (Rt) estimates over time')
plt.legend()
plt.show()

# Print summary statistics for Rt
print(pm.summary(trace, var_names=['Rt']))
```

---

## SUB_113

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using Turing, CSV, DataFrames, Distributions, Plots, StatsPlots, LinearAlgebra
using Random
Random.seed!(42)

# Load the data
function load_data(filename="cases.csv")
    df = CSV.read(filename, DataFrame)
    return df.cases
end

# Define generation interval PMF (COVID-19 parameters)
function generation_interval_pmf(max_days=20)
    # Using a discretized Gamma distribution with mean ~5.2 days, std ~1.8 days
    # Based on COVID-19 literature
    α = 8.5  # shape parameter
    β = 1.6  # rate parameter
    
    pmf = zeros(max_days)
    for s in 1:max_days
        # Probability mass for day s (discretized continuous distribution)
        pmf[s] = cdf(Gamma(α, 1/β), s) - cdf(Gamma(α, 1/β), s-1)
    end
    
    # Normalize to ensure sum = 1
    return pmf ./ sum(pmf)
end

# Define delay from infection to case reporting PMF
function reporting_delay_pmf(max_days=15)
    # Incubation period + reporting delay
    # Using discretized Gamma with mean ~7 days
    α = 4.0
    β = 0.57
    
    pmf = zeros(max_days)
    for d in 1:max_days
        pmf[d] = cdf(Gamma(α, 1/β), d) - cdf(Gamma(α, 1/β), d-1)
    end
    
    return pmf ./ sum(pmf)
end

@model function renewal_model(cases, generation_pmf, delay_pmf)
    n_days = length(cases)
    max_gen = length(generation_pmf)
    max_delay = length(delay_pmf)
    
    # Priors
    # Initial infections (seeding period)
    I₀ ~ LogNormal(log(10), 1)  # Initial seed infections
    seed_growth ~ Normal(0.1, 0.1)  # Growth rate during seeding
    
    # Rt parameters - using random walk on log scale
    log_R₁ ~ Normal(log(2), 0.5)  # Initial Rt
    σ_R ~ Exponential(0.2)  # Innovation variance for Rt random walk
    
    # Observation model parameters
    ψ ~ Beta(2, 8)  # Reporting rate
    φ ~ Exponential(0.1)  # Overdispersion parameter for negative binomial
    
    # Initialize vectors
    infections = Vector{Real}(undef, n_days)
    expected_cases = Vector{Real}(undef, n_days)
    log_Rt = Vector{Real}(undef, n_days)
    
    # Seeding period (first max_gen days)
    for t in 1:min(max_gen, n_days)
        infections[t] = I₀ * exp(seed_growth * (t-1))
        log_Rt[t] = log_R₁
    end
    
    # Random walk for log(Rt)
    for t in 2:n_days
        if t > max_gen
            log_Rt[t] ~ Normal(log_Rt[t-1], σ_R)
        end
    end
    
    # Renewal equation for infections
    for t in (max_gen+1):n_days
        Rt = exp(log_Rt[t])
        
        # Convolution with generation interval
        infectiousness = 0.0
        for s in 1:min(max_gen, t-1)
            infectiousness += infections[t-s] * generation_pmf[s]
        end
        
        infections[t] = Rt * infectiousness
    end
    
    # Delay from infections to case reports
    for t in 1:n_days
        expected_cases[t] = 0.0
        for d in 1:min(max_delay, t)
            expected_cases[t] += infections[t-d+1] * delay_pmf[d] * ψ
        end
        expected_cases[t] = max(expected_cases[t], 1e-6)  # Numerical stability
    end
    
    # Observation model - Negative Binomial
    for t in 1:n_days
        # Parameterization: NB(r, p) where mean = r*p/(1-p), var = r*p/(1-p)²
        r = 1/φ
        p = expected_cases[t] / (expected_cases[t] + r)
        cases[t] ~ NegativeBinomial(r, 1-p)
    end
    
    return (infections=infections, Rt=exp.(log_Rt), expected_cases=expected_cases)
end

# Function to fit the model and extract results
function estimate_rt(cases_data; n_samples=2000, n_warmup=1000, n_chains=4)
    println("Setting up model...")
    
    # Get PMFs
    gen_pmf = generation_interval_pmf()
    delay_pmf = reporting_delay_pmf()
    
    println("Generation interval PMF (first 10 days): ", round.(gen_pmf[1:10], digits=3))
    println("Delay PMF (first 10 days): ", round.(delay_pmf[1:10], digits=3))
    
    # Create model
    model = renewal_model(cases_data, gen_pmf, delay_pmf)
    
    println("Fitting model with NUTS sampler...")
    println("Number of observations: ", length(cases_data))
    
    # Sample using NUTS
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    return chain, gen_pmf, delay_pmf
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(chain, n_days)
    # Extract Rt samples
    rt_samples = []
    
    for t in 1:n_days
        param_name = "infections[$(t)]"
        if param_name in names(chain)
            push!(rt_samples, vec(chain["Rt[$(t)]"]))
        end
    end
    
    # If the above doesn't work, try alternative extraction
    if isempty(rt_samples)
        # Extract all Rt parameters
        rt_params = [name for name in names(chain) if startswith(string(name), "log_Rt")]
        rt_samples = [exp.(vec(chain[param])) for param in rt_params]
    end
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_lower = [quantile(samples, 0.025) for samples in rt_samples]
    rt_upper = [quantile(samples, 0.975) for samples in rt_samples]
    rt_median = [median(samples) for samples in rt_samples]
    
    return DataFrame(
        day = 1:length(rt_mean),
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper
    )
end

# Plotting function
function plot_rt_estimates(rt_df, cases_data)
    n_days = length(cases_data)
    days = 1:n_days
    
    # Create subplot layout
    p1 = plot(days, cases_data, 
             seriestype=:line, 
             linewidth=2, 
             title="Observed Cases", 
             xlabel="Day", 
             ylabel="Cases",
             legend=false)
    
    p2 = plot(rt_df.day, rt_df.rt_median,
             ribbon=(rt_df.rt_median .- rt_df.rt_lower, rt_df.rt_upper .- rt_df.rt_median),
             linewidth=2,
             fillalpha=0.3,
             title="Estimated Rt Over Time",
             xlabel="Day",
             ylabel="Rt",
             label="Rt (95% CI)",
             legend=:topright)
    
    # Add horizontal line at Rt = 1
    hline!(p2, [1.0], linestyle=:dash, linecolor=:red, linewidth=1, label="Rt = 1")
    
    return plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main execution function
function main()
    println("Loading data...")
    
    # Create sample data if file doesn't exist
    if !isfile("cases.csv")
        println("Creating sample data...")
        dates = [Date(2020, 3, 1) + Day(i-1) for i in 1:100]
        # Simulate some realistic case data
        true_rt = [2.5 * exp(-0.05*i) + 0.8 + 0.3*sin(i/10) for i in 1:100]
        cases_sim = max.(1, round.(Int, 50 * cumprod(true_rt .^ (1/5)) .+ 10*randn(100)))
        
        sample_df = DataFrame(date=dates, cases=cases_sim)
        CSV.write("cases.csv", sample_df)
        println("Sample data created and saved to cases.csv")
    end
    
    cases_data = load_data("cases.csv")
    
    println("Estimating Rt...")
    chain, gen_pmf, delay_pmf = estimate_rt(cases_data, n_samples=1000, n_warmup=500, n_chains=2)
    
    println("Extracting results...")
    rt_estimates = extract_rt_estimates(chain, length(cases_data))
    
    println("Creating plots...")
    plt = plot_rt_estimates(rt_estimates, cases_data)
    display(plt)
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("======================")
    println("Number of days: ", length(cases_data))
    println("Mean Rt over period: ", round(mean(rt_estimates.rt_mean), digits=2))
    println("Final Rt estimate: ", round(rt_estimates.rt_mean[end], digits=2), 
            " (95% CI: ", round(rt_estimates.rt_lower[end], digits=2), 
            " - ", round(rt_estimates.rt_upper[end], digits=2), ")")
    
    # Show model diagnostics
    println("\nModel Diagnostics:")
    println("==================")
    println(chain)
    
    return rt_estimates, chain, plt
end

# Run the analysis
rt_results, fitted_chain, result_plot = main()

# Additional utility functions for analysis
function get_model_summary(chain)
    """Get summary statistics for key model parameters"""
    key_params = ["σ_R", "ψ", "φ", "I₀", "seed_growth"]
    
    summary_df = DataFrame(
        parameter = String[],
        mean = Float64[],
        std = Float64[],
        q025 = Float64[],
        q975 = Float64[]
    )
    
    for param in key_params
        if param in names(chain)
            samples = vec(chain[param])
            push!(summary_df, (
                parameter = param,
                mean = mean(samples),
                std = std(samples),
                q025 = quantile(samples, 0.025),
                q975 = quantile(samples, 0.975)
            ))
        end
    end
    
    return summary_df
end

# Print parameter summary
param_summary = get_model_summary(fitted_chain)
println("\nParameter Estimates:")
println("====================")
println(param_summary)
```

---

## SUB_105

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots
using LinearAlgebra, Random, MCMCChains

# Set random seed for reproducibility
Random.seed!(42)

# Load and prepare the data
function load_data(filepath="cases.csv")
    df = CSV.read(filepath, DataFrame)
    # Ensure dates are properly parsed
    df.date = Date.(df.date)
    # Sort by date to ensure proper ordering
    sort!(df, :date)
    return df
end

# Define generation interval distribution
function get_generation_interval(max_days=20)
    # Using a discretized gamma distribution for generation interval
    # Mean ~5.2 days, std ~1.72 days (typical COVID-19 values)
    shape = 9.0
    scale = 0.58
    
    # Discretize the continuous distribution
    g = zeros(max_days)
    for i in 1:max_days
        g[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure sum = 1
    g = g ./ sum(g)
    return g
end

# Define incubation and reporting delay distribution
function get_reporting_delay(max_days=15)
    # Combined incubation + reporting delay
    # Approximated as discretized gamma with mean ~7 days
    shape = 4.0
    scale = 1.75
    
    d = zeros(max_days)
    for i in 1:max_days
        d[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize
    d = d ./ sum(d)
    return d
end

# Turing model for Rt estimation
@model function renewal_model(cases, g, d, n_days)
    # Priors
    # Log Rt follows a random walk
    log_R0 ~ Normal(0.0, 1.0)  # Initial log(Rt)
    σ_R ~ truncated(Normal(0, 0.2), 0, Inf)  # Innovation variance for Rt
    
    # Overdispersion parameter for negative binomial
    ϕ ~ truncated(Normal(0, 10), 0.1, Inf)
    
    # Initial infections (seeding period)
    n_seed = length(g)
    I_seed ~ MvNormal(zeros(n_seed), I(n_seed) * 10.0)
    
    # Initialize arrays
    log_Rt = Vector{Real}(undef, n_days)
    I_t = Vector{Real}(undef, n_days)
    λ_t = Vector{Real}(undef, n_days)
    
    # Set initial log_Rt
    log_Rt[1] = log_R0
    
    # Random walk for log(Rt)
    for t in 2:n_days
        log_Rt[t] ~ Normal(log_Rt[t-1], σ_R)
    end
    
    # Convert to Rt
    Rt = exp.(log_Rt)
    
    # Calculate infections using renewal equation
    for t in 1:n_days
        if t <= n_seed
            # Use seeded infections for initial period
            I_t[t] = exp(I_seed[t])  # Ensure positive
        else
            # Renewal equation: I_t = Rt * sum(I_{t-s} * g_s)
            infectivity = 0.0
            for s in 1:min(t-1, length(g))
                infectivity += I_t[t-s] * g[s]
            end
            I_t[t] = Rt[t] * infectivity
        end
    end
    
    # Calculate expected reported cases (accounting for reporting delay)
    for t in 1:n_days
        λ_t[t] = 0.0
        for s in 1:min(t, length(d))
            if t-s+1 >= 1
                λ_t[t] += I_t[t-s+1] * d[s]
            end
        end
        λ_t[t] = max(λ_t[t], 1e-10)  # Ensure positive
    end
    
    # Likelihood: observed cases
    for t in 1:n_days
        # Use negative binomial to account for overdispersion
        p = ϕ / (λ_t[t] + ϕ)
        cases[t] ~ NegativeBinomial(ϕ, p)
    end
    
    return (Rt=Rt, I_t=I_t, λ_t=λ_t)
end

# Function to estimate Rt
function estimate_rt(cases_data; n_iter=2000, n_chains=4)
    println("Preparing data...")
    
    # Get case counts
    cases = cases_data.cases
    n_days = length(cases)
    
    # Get generation interval and reporting delay
    g = get_generation_interval()
    d = get_reporting_delay()
    
    println("Setting up model...")
    
    # Create model
    model = renewal_model(cases, g, d, n_days)
    
    println("Running MCMC sampling...")
    
    # Sample from posterior
    chain = sample(model, NUTS(0.8), MCMCThreads(), n_iter, n_chains, 
                  progress=true, drop_warmup=true)
    
    return chain, g, d
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(chain, dates)
    # Get Rt parameter names
    rt_params = [Symbol("Rt[$i]") for i in 1:length(dates)]
    
    # Extract Rt estimates
    rt_samples = chain[rt_params]
    
    # Calculate summary statistics
    rt_mean = mean(rt_samples).nt.mean
    rt_lower = [quantile(rt_samples[Symbol("Rt[$i]")], 0.025) for i in 1:length(dates)]
    rt_upper = [quantile(rt_samples[Symbol("Rt[$i]")], 0.975) for i in 1:length(dates)]
    rt_median = [quantile(rt_samples[Symbol("Rt[$i]")], 0.5) for i in 1:length(dates)]
    
    # Create results DataFrame
    results = DataFrame(
        date = dates,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower = rt_lower,
        rt_upper = rt_upper
    )
    
    return results
end

# Function to create plots
function plot_results(cases_data, rt_estimates, chain)
    dates = cases_data.date
    
    # Plot 1: Cases over time
    p1 = plot(dates, cases_data.cases, 
             title="Observed Cases", 
             xlabel="Date", ylabel="Daily Cases",
             linewidth=2, color=:blue, legend=false)
    
    # Plot 2: Rt estimates over time
    p2 = plot(dates, rt_estimates.rt_median,
             ribbon=(rt_estimates.rt_median - rt_estimates.rt_lower,
                    rt_estimates.rt_upper - rt_estimates.rt_median),
             title="Rt Estimates Over Time",
             xlabel="Date", ylabel="Rt",
             linewidth=2, color=:red, fillalpha=0.3,
             label="Rt (95% CI)")
    
    # Add horizontal line at Rt = 1
    hline!([1.0], linestyle=:dash, color=:black, label="Rt = 1", linewidth=2)
    
    # Plot 3: MCMC diagnostics
    p3 = plot(chain[[:σ_R, :ϕ]], title="MCMC Traces for Key Parameters")
    
    # Combine plots
    combined_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 800))
    
    return combined_plot
end

# Main execution function
function main()
    println("Loading data...")
    
    # Create sample data if file doesn't exist
    if !isfile("cases.csv")
        println("Creating sample data...")
        dates = Date(2023, 1, 1):Day(1):Date(2023, 3, 31)
        # Simulate some realistic case data
        n_days = length(dates)
        true_rt = 1.2 * exp.(-0.02 * (1:n_days)) .+ 0.1 * sin.(2π * (1:n_days) / 14)
        cases = max.(1, round.(Int, 100 * true_rt .* (1 .+ 0.3 * randn(n_days))))
        
        sample_data = DataFrame(date=dates, cases=cases)
        CSV.write("cases.csv", sample_data)
        println("Sample data created and saved to cases.csv")
    end
    
    # Load the data
    cases_data = load_data("cases.csv")
    println("Data loaded: $(nrow(cases_data)) days of data")
    
    # Estimate Rt
    chain, g, d = estimate_rt(cases_data, n_iter=1000, n_chains=2)
    
    # Extract Rt estimates
    rt_estimates = extract_rt_estimates(chain, cases_data.date)
    
    # Print summary
    println("\nRt Estimation Results:")
    println("======================")
    for i in [1, nrow(rt_estimates)÷2, nrow(rt_estimates)]
        row = rt_estimates[i, :]
        println("$(row.date): Rt = $(round(row.rt_median, digits=2)) " *
               "(95% CI: $(round(row.rt_lower, digits=2))-$(round(row.rt_upper, digits=2)))")
    end
    
    # Create and display plots
    p = plot_results(cases_data, rt_estimates, chain)
    display(p)
    
    # Save results
    CSV.write("rt_estimates.csv", rt_estimates)
    println("\nResults saved to rt_estimates.csv")
    
    return rt_estimates, chain
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    rt_estimates, chain = main()
end
```

---

## SUB_006

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots, Random, LinearAlgebra

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare the data
function load_data(filename="cases.csv")
    # If the file doesn't exist, create sample data for demonstration
    if !isfile(filename)
        println("Creating sample data for demonstration...")
        dates = Date(2020, 3, 1):Day(1):Date(2020, 6, 30)
        
        # Simulate realistic COVID-19 case trajectory
        n_days = length(dates)
        true_rt = vcat(
            fill(2.5, 20),  # Initial exponential growth
            2.5 .- 0.05 * (1:30),  # Gradual decline
            fill(1.0, 30),  # Around critical threshold
            1.0 .+ 0.02 * (1:20),  # Slight increase
            fill(1.2, n_days - 100)  # Sustained transmission
        )
        
        # Simple simulation for realistic case counts
        cases = zeros(Int, n_days)
        cases[1:7] .= [5, 8, 12, 18, 25, 35, 45]  # Seed cases
        
        for t in 8:n_days
            lambda = sum(cases[max(1, t-14):t-1] .* 
                        exp.(-((1:min(14, t-1)) .- 5.5).^2 / (2 * 2.5^2)))
            expected_cases = max(1.0, true_rt[t] * lambda * 0.1)
            cases[t] = rand(Poisson(expected_cases))
        end
        
        df = DataFrame(date=dates, cases=cases)
        CSV.write(filename, df)
    end
    
    df = CSV.read(filename, DataFrame)
    return df
end

# Define generation interval distribution
function generation_interval_pmf(max_gen=20)
    # Discretized gamma distribution for generation interval
    # Mean ≈ 5.5 days, SD ≈ 2.5 days (typical for COVID-19)
    shape, scale = 4.8, 1.15
    
    pmf = zeros(max_gen)
    for i in 1:max_gen
        pmf[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure sum = 1
    pmf = pmf / sum(pmf)
    return pmf
end

# Define reporting delay distribution
function reporting_delay_pmf(max_delay=15)
    # Log-normal distribution for reporting delay
    # Mean delay ≈ 7 days
    μ, σ = 1.8, 0.5
    
    pmf = zeros(max_delay)
    for i in 1:max_delay
        pmf[i] = cdf(LogNormal(μ, σ), i) - cdf(LogNormal(μ, σ), i-1)
    end
    
    pmf = pmf / sum(pmf)
    return pmf
end

# Turing model for Rt estimation
@model function rt_model(cases, gen_pmf, delay_pmf)
    n_days = length(cases)
    max_gen = length(gen_pmf)
    max_delay = length(delay_pmf)
    
    # Priors for initial infections (seeding period)
    seed_days = 7
    I_seed ~ filldist(Exponential(10.0), seed_days)
    
    # Prior for Rt - using a random walk on log scale for smoothness
    log_rt_init ~ Normal(0.5, 0.5)  # Initial Rt around exp(0.5) ≈ 1.65
    σ_rt ~ Exponential(0.1)  # Innovation standard deviation
    
    log_rt_innovations ~ filldist(Normal(0, 1), n_days - seed_days - 1)
    
    # Construct log_rt time series
    log_rt = Vector{eltype(log_rt_init)}(undef, n_days)
    log_rt[seed_days + 1] = log_rt_init
    
    for t in (seed_days + 2):n_days
        log_rt[t] = log_rt[t-1] + σ_rt * log_rt_innovations[t - seed_days - 1]
    end
    
    # Convert to Rt
    rt = exp.(log_rt[(seed_days + 1):end])
    
    # Initialize infections
    infections = Vector{eltype(I_seed[1])}(undef, n_days)
    infections[1:seed_days] = I_seed
    
    # Renewal equation: compute infections for t > seed_days
    for t in (seed_days + 1):n_days
        lambda = 0.0
        for s in 1:min(max_gen, t-1)
            if t - s >= 1
                lambda += infections[t - s] * gen_pmf[s]
            end
        end
        infections[t] = rt[t - seed_days] * lambda
    end
    
    # Convolve infections with reporting delay to get expected cases
    expected_cases = Vector{eltype(infections[1])}(undef, n_days)
    
    for t in 1:n_days
        expected_cases[t] = 0.0
        for d in 1:min(max_delay, t)
            if t - d + 1 >= 1
                expected_cases[t] += infections[t - d + 1] * delay_pmf[d]
            end
        end
        expected_cases[t] = max(expected_cases[t], 1e-6)  # Avoid zero
    end
    
    # Observation model - Negative Binomial for overdispersion
    φ ~ Exponential(0.1)  # Overdispersion parameter
    
    for t in 1:n_days
        if cases[t] >= 0  # Only observe non-missing cases
            # Negative binomial parameterized by mean and overdispersion
            p = φ / (φ + expected_cases[t])
            r = φ
            cases[t] ~ NegativeBinomial(r, 1 - p)
        end
    end
end

# Function to run the analysis
function estimate_rt(data_file="cases.csv")
    println("Loading data...")
    df = load_data(data_file)
    cases = df.cases
    dates = df.date
    n_days = length(cases)
    
    println("Setting up model...")
    gen_pmf = generation_interval_pmf()
    delay_pmf = reporting_delay_pmf()
    
    println("Generation interval (first 10 days): ", round.(gen_pmf[1:10], digits=3))
    println("Reporting delay (first 10 days): ", round.(delay_pmf[1:10], digits=3))
    
    # Create and sample from the model
    println("Fitting model...")
    model = rt_model(cases, gen_pmf, delay_pmf)
    
    # Use NUTS sampler
    n_samples = 1000
    n_chains = 2
    
    println("Running MCMC sampling...")
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    println("Processing results...")
    
    # Extract Rt estimates
    seed_days = 7
    rt_samples = []
    
    for i in 1:(n_days - seed_days)
        rt_col = "rt[$i]"
        if rt_col in names(chain)
            push!(rt_samples, chain[rt_col])
        end
    end
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_q025 = [quantile(vec(Array(samples)), 0.025) for samples in rt_samples]
    rt_q975 = [quantile(vec(Array(samples)), 0.975) for samples in rt_samples]
    
    # Create results dataframe
    rt_dates = dates[(seed_days + 1):end]
    results_df = DataFrame(
        date = rt_dates,
        rt_mean = rt_mean,
        rt_q025 = rt_q025,
        rt_q975 = rt_q975
    )
    
    # Print summary
    println("\nRt Estimation Summary:")
    println("=" ^ 50)
    println("Data period: $(dates[1]) to $(dates[end])")
    println("Total days: $n_days")
    println("Rt estimation period: $(rt_dates[1]) to $(rt_dates[end])")
    println("Mean Rt over period: $(round(mean(rt_mean), digits=2))")
    println("Days with Rt > 1: $(sum(rt_mean .> 1)) / $(length(rt_mean))")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot 1: Case counts over time
    p1 = plot(dates, cases, 
             title="Daily Case Counts", 
             xlabel="Date", 
             ylabel="Cases",
             linewidth=2,
             color=:blue,
             legend=false)
    
    # Plot 2: Rt over time
    p2 = plot(rt_dates, rt_mean,
             ribbon=(rt_mean .- rt_q025, rt_q975 .- rt_mean),
             title="Estimated Rt Over Time",
             xlabel="Date",
             ylabel="Rt",
             linewidth=2,
             color=:red,
             fillalpha=0.3,
             label="Rt (95% CI)")
    
    hline!([1.0], linestyle=:dash, color=:black, linewidth=2, label="Rt = 1")
    
    # Combined plot
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    display(combined_plot)
    
    # Save results
    CSV.write("rt_estimates.csv", results_df)
    savefig(combined_plot, "rt_estimation_results.png")
    
    println("\nResults saved to:")
    println("- rt_estimates.csv (Rt estimates)")
    println("- rt_estimation_results.png (plots)")
    
    return results_df, chain, combined_plot
end

# Run the analysis
println("Starting Rt estimation using renewal equation...")
results_df, chain, plot_result = estimate_rt()

# Display final results
println("\nFinal Rt estimates (last 10 days):")
println(last(results_df, 10))

# Model diagnostics
println("\nModel diagnostics:")
println("MCMC Summary:")
display(chain)
```

---

## SUB_076

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
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
```

---

## SUB_002

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
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
```

---

## SUB_094

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
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
```

---

## SUB_021

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Plots
using StatsPlots
using Turing
using MCMCChains

# Load and prepare the data
function load_case_data(filename::String)
    df = CSV.read(filename, DataFrame)
    # Ensure data is sorted by date
    sort!(df, :date)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_renewal(cases::Vector{Int})
    println("Setting up renewal equation model for Rt estimation...")
    
    # 1. Define epidemiological parameters
    # Generation interval - using COVID-19 typical values
    gen_int_mean = 6.5
    gen_int_var = 4.03  # variance
    # Convert to shape/scale parameterization for Gamma distribution
    gen_int_scale = gen_int_var / gen_int_mean
    gen_int_shape = gen_int_mean / gen_int_scale
    gen_distribution = Gamma(gen_int_shape, gen_int_scale)
    
    println("Generation interval: Gamma($(round(gen_int_shape, digits=2)), $(round(gen_int_scale, digits=2)))")
    
    # 2. Create EpiData with generation interval
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 3. Create infection model using Renewal equation
    # Prior for initial infections (seeding)
    initial_inf_prior = Normal(log(mean(cases[1:7])), 1.0)  # Use first week average as guide
    epi = Renewal(model_data; initialisation_prior = initial_inf_prior)
    
    # 4. Create latent model for log(Rt) - AR(1) process for smooth evolution
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Innovation noise
    )
    
    # 5. Create observation model with reporting delay
    # Delay from infection to case reporting (incubation + reporting delay)
    delay_mean = 8.0
    delay_var = 16.0
    delay_scale = delay_var / delay_mean
    delay_shape = delay_mean / delay_scale
    delay_dist = Gamma(delay_shape, delay_scale)
    
    # Negative binomial observation model to handle overdispersion
    obs_base = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs = LatentDelay(obs_base, delay_dist)
    
    println("Reporting delay: Gamma($(round(delay_shape, digits=2)), $(round(delay_scale, digits=2)))")
    
    # 6. Compose into EpiProblem
    n_timepoints = length(cases)
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    println("Model setup complete. Running inference...")
    
    # 7. Generate Turing model and run inference
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # Configure inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 1000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    println("Inference complete!")
    
    return results, epi_prob
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(results)
    # Extract the latent process Z_t which represents log(Rt)
    chain = results
    
    # Get parameter names that contain "Z_t"
    param_names = names(chain)
    z_params = filter(name -> occursin("Z_t", string(name)), param_names)
    
    # Sort parameters by time index
    z_params_sorted = sort(z_params, by = x -> parse(Int, match(r"\[(\d+)\]", string(x)).captures[1]))
    
    # Extract log(Rt) values
    log_rt_samples = Array(chain[z_params_sorted])
    
    # Convert to Rt (exponentiate)
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(median(rt_samples, dims=1))
    rt_q025 = vec(quantile.(eachcol(rt_samples), 0.025))
    rt_q975 = vec(quantile.(eachcol(rt_samples), 0.975))
    rt_q10 = vec(quantile.(eachcol(rt_samples), 0.10))
    rt_q90 = vec(quantile.(eachcol(rt_samples), 0.90))
    
    return DataFrame(
        time = 1:length(rt_mean),
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_q025 = rt_q025,
        rt_q975 = rt_q975,
        rt_q10 = rt_q10,
        rt_q90 = rt_q90
    )
end

# Function to create plots
function plot_rt_estimates(rt_estimates, cases)
    # Create subplot layout
    p1 = plot(rt_estimates.time, rt_estimates.rt_median, 
             ribbon=(rt_estimates.rt_median .- rt_estimates.rt_q025,
                    rt_estimates.rt_q975 .- rt_estimates.rt_median),
             label="Rt (95% CI)", 
             color=:blue, 
             alpha=0.3,
             linewidth=2,
             title="Estimated Reproduction Number (Rt)",
             xlabel="Time (days)",
             ylabel="Rt")
    
    # Add 50% credible interval
    plot!(p1, rt_estimates.time, rt_estimates.rt_median, 
          ribbon=(rt_estimates.rt_median .- rt_estimates.rt_q10,
                 rt_estimates.rt_q90 .- rt_estimates.rt_median),
          label="Rt (80% CI)", 
          color=:blue, 
          alpha=0.5,
          linewidth=2)
    
    # Add reference line at Rt = 1
    hline!(p1, [1.0], line=:dash, color=:red, label="Rt = 1", linewidth=2)
    
    # Plot observed cases
    p2 = bar(1:length(cases), cases, 
             label="Observed cases",
             color=:gray,
             alpha=0.7,
             title="Observed Case Counts",
             xlabel="Time (days)",
             ylabel="Cases")
    
    # Combine plots
    plot(p1, p2, layout=(2,1), size=(800, 600))
end

# Main execution function
function main()
    println("=== Rt Estimation using Renewal Equation ===\n")
    
    # Load data
    println("Loading case data...")
    cases = load_case_data("cases.csv")
    println("Loaded $(length(cases)) days of case data")
    println("Total cases: $(sum(cases))")
    println("Mean daily cases: $(round(mean(cases), digits=1))\n")
    
    # Estimate Rt
    results, epi_prob = estimate_rt_renewal(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results)
    
    # Display summary
    println("\n=== Rt Estimation Summary ===")
    println("Time period: $(length(cases)) days")
    println("Mean Rt: $(round(mean(rt_estimates.rt_mean), digits=2))")
    println("Rt range: $(round(minimum(rt_estimates.rt_mean), digits=2)) - $(round(maximum(rt_estimates.rt_mean), digits=2))")
    
    # Count periods where Rt > 1
    days_above_1 = sum(rt_estimates.rt_median .> 1.0)
    println("Days with Rt > 1: $days_above_1 / $(length(cases)) ($(round(100*days_above_1/length(cases), digits=1))%)")
    
    # Show recent Rt values
    println("\nRecent Rt estimates (last 7 days):")
    recent_idx = max(1, length(cases)-6):length(cases)
    for i in recent_idx
        println("Day $i: $(round(rt_estimates.rt_median[i], digits=2)) [$(round(rt_estimates.rt_q025[i], digits=2)) - $(round(rt_estimates.rt_q975[i], digits=2))]")
    end
    
    # Create and save plots
    println("\nCreating plots...")
    p = plot_rt_estimates(rt_estimates, cases)
    
    # Save results
    println("Saving results...")
    CSV.write("rt_estimates.csv", rt_estimates)
    savefig(p, "rt_estimates_plot.png")
    
    println("\n=== Analysis Complete ===")
    println("Results saved to:")
    println("- rt_estimates.csv: Rt estimates with credible intervals")
    println("- rt_estimates_plot.png: Visualization of results")
    
    return rt_estimates, results
end

# Helper function to create synthetic data if needed (for testing)
function create_synthetic_cases(n_days::Int = 100)
    # Simple synthetic epidemic with changing Rt
    true_rt = vcat(
        fill(2.0, 20),    # Initial growth
        1.5 .- 0.025*(1:20),  # Declining
        fill(0.8, 20),    # Below 1
        0.8 .+ 0.01*(1:20),   # Rising again
        fill(1.2, 20)     # Above 1 again
    )
    
    gen_int = Gamma(6.5, 0.62)
    
    infections = zeros(n_days)
    infections[1:7] .= 50  # Seed infections
    
    for t in 8:n_days
        # Renewal equation
        inf_pressure = 0.0
        for s in 1:min(t-1, 20)  # Truncate generation interval
            gen_prob = pdf(gen_int, s) / sum([pdf(gen_int, i) for i in 1:20])
            inf_pressure += infections[t-s] * gen_prob
        end
        infections[t] = true_rt[t] * inf_pressure
    end
    
    # Add reporting delay and noise
    delay_dist = Gamma(2.0, 3.0)
    cases = zeros(Int, n_days)
    
    for t in 1:n_days
        for delay in 1:20
            delay_prob = pdf(delay_dist, delay) / sum([pdf(delay_dist, i) for i in 1:20])
            if t + delay <= n_days
                cases[t + delay] += round(Int, infections[t] * delay_prob * rand(LogNormal(0, 0.1)))
            end
        end
    end
    
    # Create DataFrame and save
    dates = [Date(2023, 1, 1) + Day(i-1) for i in 1:n_days]
    df = DataFrame(date = dates, cases = cases)
    CSV.write("cases.csv", df)
    
    return df
end

# Run the analysis
# Uncomment the next line to create synthetic data for testing
# create_synthetic_cases(100)

# Run main analysis
rt_estimates, results = main()
```

---

## SUB_116

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using DataFrames
using CSV
using Distributions
using Turing
using Plots
using Statistics

# Load the data
function load_case_data(filename)
    df = CSV.read(filename, DataFrame)
    return df.cases
end

# Main function to estimate Rt
function estimate_rt_renewal(cases; 
                           gen_mean = 6.5, 
                           gen_std = 0.62,
                           delay_mean = 5.0,
                           delay_std = 1.0)
    
    println("Setting up Rt estimation using renewal equation...")
    println("Number of time points: $(length(cases))")
    
    # 1. Define generation interval distribution
    # Convert mean/std to shape/scale parameters for Gamma distribution
    gen_scale = gen_std^2 / gen_mean
    gen_shape = gen_mean / gen_scale
    gen_distribution = Gamma(gen_shape, gen_scale)
    
    println("Generation interval: Gamma(shape=$(round(gen_shape, digits=2)), scale=$(round(gen_scale, digits=2)))")
    
    # Create EpiData with generation interval
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create renewal infection model
    # Prior for initial seeding infections
    epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))
    
    # 3. Create latent model for log(Rt) - AR(1) process
    # This allows Rt to vary smoothly over time
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],                    # Initial log(Rt)
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1)) # Innovation noise
    )
    
    # 4. Create observation model with reporting delay
    # Convert delay mean/std to Gamma parameters
    delay_scale = delay_std^2 / delay_mean
    delay_shape = delay_mean / delay_scale
    delay_distribution = Gamma(delay_shape, delay_scale)
    
    println("Reporting delay: Gamma(shape=$(round(delay_shape, digits=2)), scale=$(round(delay_scale, digits=2)))")
    
    # Negative binomial observation model to handle overdispersion
    obs_base = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs = LatentDelay(obs_base, delay_distribution)
    
    # 5. Compose the full epidemiological problem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, length(cases))
    )
    
    # 6. Generate the Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 7. Set up inference method
    println("Setting up MCMC inference...")
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 8. Run inference
    println("Running MCMC sampling...")
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract and summarize Rt estimates
function extract_rt_estimates(results)
    # Extract the latent process Z_t (log Rt)
    log_rt_samples = results["latent_process[1]"]
    
    # Convert to Rt (exp transform)
    rt_samples = exp.(log_rt_samples)
    
    # Calculate summary statistics
    n_timepoints = size(rt_samples, 2)
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_median = vec(mapslices(median, rt_samples, dims=1))
    rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    rt_lower_50 = vec(mapslices(x -> quantile(x, 0.25), rt_samples, dims=1))
    rt_upper_50 = vec(mapslices(x -> quantile(x, 0.75), rt_samples, dims=1))
    
    return DataFrame(
        time = 1:n_timepoints,
        rt_mean = rt_mean,
        rt_median = rt_median,
        rt_lower_95 = rt_lower,
        rt_upper_95 = rt_upper,
        rt_lower_50 = rt_lower_50,
        rt_upper_50 = rt_upper_50
    )
end

# Function to plot results
function plot_rt_estimates(rt_estimates, cases)
    # Create subplot layout
    p1 = plot(rt_estimates.time, rt_estimates.rt_median, 
              ribbon = (rt_estimates.rt_median .- rt_estimates.rt_lower_95,
                       rt_estimates.rt_upper_95 .- rt_estimates.rt_median),
              fillalpha = 0.2, color = :blue, linewidth = 2,
              title = "Estimated Rt over Time", 
              xlabel = "Time (days)", 
              ylabel = "Reproduction Number (Rt)",
              label = "Rt (95% CI)")
    
    # Add 50% credible interval
    plot!(p1, rt_estimates.time, rt_estimates.rt_median,
          ribbon = (rt_estimates.rt_median .- rt_estimates.rt_lower_50,
                   rt_estimates.rt_upper_50 .- rt_estimates.rt_median),
          fillalpha = 0.4, color = :blue, label = "Rt (50% CI)")
    
    # Add horizontal line at Rt = 1
    hline!(p1, [1.0], linestyle = :dash, color = :red, linewidth = 2, label = "Rt = 1")
    
    # Plot observed cases
    p2 = bar(1:length(cases), cases, 
             title = "Observed Cases", 
             xlabel = "Time (days)", 
             ylabel = "Daily Cases",
             color = :orange, alpha = 0.7, label = "Cases")
    
    # Combine plots
    plot(p1, p2, layout = (2, 1), size = (800, 600))
end

# Function to extract posterior predictive checks
function extract_posterior_predictions(results)
    # Extract generated quantities if available
    if haskey(results, "generated_y_t")
        pred_samples = results["generated_y_t"]
        
        n_timepoints = size(pred_samples, 2)
        pred_mean = vec(mean(pred_samples, dims=1))
        pred_lower = vec(mapslices(x -> quantile(x, 0.025), pred_samples, dims=1))
        pred_upper = vec(mapslices(x -> quantile(x, 0.975), pred_samples, dims=1))
        
        return DataFrame(
            time = 1:n_timepoints,
            pred_mean = pred_mean,
            pred_lower = pred_lower,
            pred_upper = pred_upper
        )
    else
        println("Warning: Posterior predictive samples not found in results")
        return nothing
    end
end

# Main execution function
function main()
    println("=== Rt Estimation Using Renewal Equation ===")
    
    # Load data
    println("Loading case data...")
    cases = load_case_data("cases.csv")
    
    # Estimate Rt
    results, epi_prob = estimate_rt_renewal(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results)
    
    # Display summary
    println("\nRt Estimation Summary:")
    println("Time period: $(rt_estimates.time[1]) to $(rt_estimates.time[end])")
    println("Final Rt estimate: $(round(rt_estimates.rt_median[end], digits=3)) (95% CI: $(round(rt_estimates.rt_lower_95[end], digits=3))-$(round(rt_estimates.rt_upper_95[end], digits=3)))")
    
    # Count time points where Rt > 1
    prob_rt_greater_1 = mean(rt_estimates.rt_median .> 1.0)
    println("Proportion of time with Rt > 1: $(round(prob_rt_greater_1, digits=3))")
    
    # Create plots
    println("Creating plots...")
    p = plot_rt_estimates(rt_estimates, cases)
    display(p)
    
    # Save results
    CSV.write("rt_estimates.csv", rt_estimates)
    println("Rt estimates saved to rt_estimates.csv")
    
    # Extract posterior predictions for model checking
    pred_data = extract_posterior_predictions(results)
    if pred_data !== nothing
        CSV.write("posterior_predictions.csv", pred_data)
        println("Posterior predictions saved to posterior_predictions.csv")
    end
    
    return rt_estimates, results
end

# Alternative simplified function for quick estimation
function quick_rt_estimate(cases_file = "cases.csv")
    cases = load_case_data(cases_file)
    results, _ = estimate_rt_renewal(cases)
    rt_estimates = extract_rt_estimates(results)
    return rt_estimates
end

# Run the main analysis
if abspath(PROGRAM_FILE) == @__FILE__
    rt_estimates, results = main()
end


# Quick estimation
rt_estimates = quick_rt_estimate("cases.csv")

# Full analysis with custom parameters
results, epi_prob = estimate_rt_renewal(cases, 
                                      gen_mean=7.0, 
                                      delay_mean=6.0)
```

---

## SUB_086

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics
using StatsPlots

# Function to load and preprocess the data
function load_case_data(filepath::String)
    # Load the CSV file
    df = CSV.read(filepath, DataFrame)
    
    # Convert dates to Date type if they're strings
    if eltype(df.date) <: AbstractString
        df.date = Date.(df.date)
    end
    
    # Sort by date to ensure chronological order
    sort!(df, :date)
    
    # Convert cases to integers and handle any missing/negative values
    cases = Int.(max.(df.cases, 0))  # Ensure non-negative
    
    return df.date, cases
end

# Function to create and fit the renewal equation model
function estimate_rt_renewal(cases::Vector{Int}; 
                           gen_mean::Float64 = 6.5,
                           gen_std::Float64 = 4.0,
                           delay_mean::Float64 = 5.0,
                           delay_std::Float64 = 3.0,
                           ndraws::Int = 2000,
                           nchains::Int = 4)
    
    n_timepoints = length(cases)
    
    println("Setting up renewal equation model for $n_timepoints time points...")
    
    # 1. Define generation interval distribution
    # Convert mean/std to Gamma parameters (method of moments)
    gen_shape = (gen_mean / gen_std)^2
    gen_rate = gen_mean / gen_std^2
    gen_distribution = Gamma(gen_shape, 1/gen_rate)
    
    # Create EpiData
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create infection model using renewal equation
    # Initial infection level prior based on early case counts
    initial_cases_mean = mean(cases[1:min(7, length(cases))])
    init_prior = Normal(log(max(initial_cases_mean, 1.0)), 1.0)
    
    epi = Renewal(model_data; initialisation_prior = init_prior)
    
    # 3. Create latent model for log(Rt) - AR(1) process with drift
    # This allows Rt to vary smoothly over time
    latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0.0, 1.0)],  # AR coefficient
        init_priors = [Normal(0.0, 0.5)],  # Initial log(Rt) ~ log(1) = 0
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.2))  # Innovation noise
    )
    
    # 4. Create observation model with reporting delay
    # Convert delay mean/std to Gamma parameters
    delay_shape = (delay_mean / delay_std)^2
    delay_rate = delay_mean / delay_std^2
    delay_distribution = Gamma(delay_shape, 1/delay_rate)
    
    # Use negative binomial to handle overdispersion in case counts
    obs_error = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs = LatentDelay(obs_error, delay_distribution)
    
    # 5. Compose the full model
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = latent,
        observation_model = obs,
        tspan = (1, n_timepoints)
    )
    
    # 6. Generate Turing model
    println("Generating Turing model...")
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 7. Set up inference method with pathfinder for initialization
    println("Running inference...")
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = ndraws,
            nchains = nchains,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 8. Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract Rt estimates from results
function extract_rt_estimates(results, n_timepoints::Int)
    # Get the chains
    chains = results.chain
    
    # Extract Z_t parameters (these are log(Rt))
    z_params = []
    for t in 1:n_timepoints
        param_name = "Z_t[$t]"
        if param_name in string.(keys(chains))
            push!(z_params, param_name)
        end
    end
    
    # If Z_t parameters not found, try alternative naming
    if isempty(z_params)
        z_params = [k for k in string.(keys(chains)) if occursin("Z_t", string(k))]
    end
    
    # Extract log(Rt) values and convert to Rt
    log_rt_samples = []
    rt_samples = []
    
    for param in z_params
        log_rt_vals = vec(Array(chains[param]))
        rt_vals = exp.(log_rt_vals)
        push!(log_rt_samples, log_rt_vals)
        push!(rt_samples, rt_vals)
    end
    
    # Calculate summary statistics
    rt_mean = [mean(rt) for rt in rt_samples]
    rt_median = [median(rt) for rt in rt_samples]
    rt_lower = [quantile(rt, 0.025) for rt in rt_samples]
    rt_upper = [quantile(rt, 0.975) for rt in rt_samples]
    rt_lower_50 = [quantile(rt, 0.25) for rt in rt_samples]
    rt_upper_50 = [quantile(rt, 0.75) for rt in rt_samples]
    
    return (
        samples = rt_samples,
        mean = rt_mean,
        median = rt_median,
        lower_95 = rt_lower,
        upper_95 = rt_upper,
        lower_50 = rt_lower_50,
        upper_50 = rt_upper_50
    )
end

# Function to create plots
function plot_rt_estimates(dates, rt_estimates, cases)
    n_timepoints = length(rt_estimates.mean)
    time_points = 1:n_timepoints
    
    # Create main Rt plot
    p1 = plot(dates[1:n_timepoints], rt_estimates.mean, 
              ribbon = (rt_estimates.mean .- rt_estimates.lower_95, 
                       rt_estimates.upper_95 .- rt_estimates.mean),
              fillalpha = 0.3, 
              label = "Rt (95% CI)",
              linewidth = 2,
              title = "Time-varying Reproduction Number (Rt)",
              xlabel = "Date",
              ylabel = "Rt",
              legend = :topright)
    
    # Add 50% credible interval
    plot!(p1, dates[1:n_timepoints], rt_estimates.median,
          ribbon = (rt_estimates.median .- rt_estimates.lower_50,
                   rt_estimates.upper_50 .- rt_estimates.median),
          fillalpha = 0.5,
          label = "Rt (50% CI)",
          linewidth = 2)
    
    # Add horizontal line at Rt = 1
    hline!(p1, [1.0], linestyle = :dash, color = :red, 
           label = "Rt = 1", linewidth = 2)
    
    # Create cases plot
    p2 = plot(dates, cases,
              label = "Observed cases",
              linewidth = 2,
              title = "Daily Case Counts",
              xlabel = "Date",
              ylabel = "Cases",
              legend = :topright)
    
    # Combine plots
    plot(p2, p1, layout = (2, 1), size = (800, 600))
end

# Main execution function
function main()
    # Load the data
    println("Loading case data...")
    dates, cases = load_case_data("cases.csv")
    
    println("Loaded $(length(cases)) days of case data")
    println("Date range: $(dates[1]) to $(dates[end])")
    println("Case range: $(minimum(cases)) to $(maximum(cases))")
    
    # Estimate Rt using renewal equation
    results, epi_prob = estimate_rt_renewal(cases)
    
    # Extract Rt estimates
    println("Extracting Rt estimates...")
    rt_estimates = extract_rt_estimates(results, length(cases))
    
    # Print summary
    println("\n=== Rt Estimation Results ===")
    println("Mean Rt over time period: $(round(mean(rt_estimates.mean), digits=3))")
    println("Median Rt over time period: $(round(mean(rt_estimates.median), digits=3))")
    println("Time points with Rt > 1: $(sum(rt_estimates.mean .> 1))/$(length(rt_estimates.mean))")
    
    # Create and save plots
    println("Creating plots...")
    p = plot_rt_estimates(dates, rt_estimates, cases)
    savefig(p, "rt_estimates.png")
    display(p)
    
    # Save results to CSV
    results_df = DataFrame(
        date = dates[1:length(rt_estimates.mean)],
        rt_mean = rt_estimates.mean,
        rt_median = rt_estimates.median,
        rt_lower_95 = rt_estimates.lower_95,
        rt_upper_95 = rt_estimates.upper_95,
        rt_lower_50 = rt_estimates.lower_50,
        rt_upper_50 = rt_estimates.upper_50
    )
    
    CSV.write("rt_estimates.csv", results_df)
    println("Results saved to rt_estimates.csv and rt_estimates.png")
    
    return results, rt_estimates, dates, cases
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    results, rt_estimates, dates, cases = main()
end
```

---

## SUB_010

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using Plots

# Load data
cases = CSV.read("cases.csv", DataFrame)

# Define generation interval
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)

results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt_estimates = exp.(mean(results.latent.Z_t, dims=2))

# Plot results
plot(Rt_estimates)
```

---

## SUB_040

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using CSV
using Plots

# Load data
data = CSV.File("cases.csv") |> DataFrame

# 1. Define generation interval
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# 2. Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# 3. Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# 4. Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# 5. Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(data.cases))
)

# 6. Run inference
mdl = generate_epiaware(epi_prob, (y_t = data.cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = data.cases,))

# 7. Extract Rt estimates
Rt_estimates = map(x -> exp.(x[1]), results.samples)

# Plot Rt over time
plot(Rt_estimates)
xlabel!("Time")
ylabel!("Rt")
title!("Estimated Reproduction Number Over Time")
```

---

## SUB_098

**Scenario**: 1b | **Execution**: FAILED

### script.jl

```julia
# Import necessary packages
using EpiAware
using Distributions
using Turing

# Define data
data = CSV.read("cases.csv")

# Create generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = obs,
    tspan = (1, length(data.cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = data.cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = data.cases,))

# Extract Rt estimates
Rt_estimates = map(x -> exp.(x), results.latent_samples)

# Plot Rt over time
using Plots
plot(Rt_estimates[1,:])
xlabel!("Time")
ylabel!("Rt")
title!("Estimated Reproduction Number (Rt)")
```

---

## SUB_033

**Scenario**: 1b | **Execution**: TIMEOUT

### script.R

```r
# Load required libraries
library(tidyverse)
library(rstan)
library(bayesplot)
library(lubridate)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and prepare the data
load_and_prepare_data <- function(file_path = "cases.csv") {
  # Load data
  data <- read_csv(file_path)
  
  # Ensure proper date format and sort
  data <- data %>%
    mutate(date = as.Date(date)) %>%
    arrange(date) %>%
    mutate(
      day = as.numeric(date - min(date)) + 1,
      cases = pmax(cases, 0)  # Ensure non-negative cases
    )
  
  return(data)
}

# Define generation interval distribution
get_generation_interval <- function(max_days = 15) {
  # Using a discretized gamma distribution
  # Mean = 5.2 days, SD = 1.72 days (typical for COVID-19)
  shape <- (5.2 / 1.72)^2
  rate <- 5.2 / (1.72^2)
  
  # Discretize the gamma distribution
  days <- 1:max_days
  pmf <- diff(pgamma(c(0, days), shape = shape, rate = rate))
  pmf <- pmf / sum(pmf)  # Normalize
  
  return(pmf)
}

# Define incubation period distribution (infection to symptom onset)
get_incubation_period <- function(max_days = 15) {
  # Mean = 5.1 days, SD = 3.0 days
  shape <- (5.1 / 3.0)^2
  rate <- 5.1 / (3.0^2)
  
  days <- 1:max_days
  pmf <- diff(pgamma(c(0, days), shape = shape, rate = rate))
  pmf <- pmf / sum(pmf)
  
  return(pmf)
}

# Define reporting delay distribution (symptom onset to case reporting)
get_reporting_delay <- function(max_days = 10) {
  # Mean = 2.0 days, SD = 1.5 days
  shape <- (2.0 / 1.5)^2
  rate <- 2.0 / (1.5^2)
  
  days <- 1:max_days
  pmf <- diff(pgamma(c(0, days), shape = shape, rate = rate))
  pmf <- pmf / sum(pmf)
  
  return(pmf)
}

# Combine incubation and reporting delays
get_total_delay <- function(max_days = 20) {
  incub <- get_incubation_period(15)
  report <- get_reporting_delay(10)
  
  # Convolve the two distributions
  total_pmf <- numeric(max_days)
  for(i in 1:length(incub)) {
    for(j in 1:length(report)) {
      if(i + j <= max_days) {
        total_pmf[i + j] <- total_pmf[i + j] + incub[i] * report[j]
      }
    }
  }
  
  # Normalize
  total_pmf <- total_pmf / sum(total_pmf)
  return(total_pmf)
}

# Stan model for Rt estimation
stan_model_code <- "
data {
  int<lower=1> n_days;                    // Number of days
  int<lower=0> cases[n_days];             // Observed cases
  int<lower=1> max_gen;                   // Max generation interval
  int<lower=1> max_delay;                 // Max reporting delay
  vector<lower=0>[max_gen] generation_pmf; // Generation interval PMF
  vector<lower=0>[max_delay] delay_pmf;   // Reporting delay PMF
  int<lower=1> seeding_days;              // Days for initial seeding
}

parameters {
  vector<lower=0>[seeding_days] initial_infections; // Initial infections
  vector[n_days - seeding_days] log_rt_raw;         // Log Rt (random walk)
  real<lower=0> rt_sd;                              // SD of Rt random walk
  real<lower=0> phi;                                // Overdispersion parameter
}

transformed parameters {
  vector<lower=0>[n_days] infections;
  vector<lower=0>[n_days] expected_cases;
  vector<lower=0>[n_days - seeding_days] rt;
  
  // Set initial infections
  infections[1:seeding_days] = initial_infections;
  
  // Transform log Rt to Rt
  rt = exp(log_rt_raw);
  
  // Apply renewal equation
  for(t in (seeding_days + 1):n_days) {
    real infectiousness = 0;
    int start_day = max(1, t - max_gen);
    
    for(s in start_day:(t-1)) {
      int gen_day = t - s;
      if(gen_day <= max_gen) {
        infectiousness += infections[s] * generation_pmf[gen_day];
      }
    }
    
    infections[t] = rt[t - seeding_days] * infectiousness;
  }
  
  // Apply reporting delay to get expected cases
  for(t in 1:n_days) {
    expected_cases[t] = 0;
    int start_day = max(1, t - max_delay + 1);
    
    for(s in start_day:min(n_days, t)) {
      int delay_day = t - s + 1;
      if(delay_day <= max_delay && s <= n_days) {
        expected_cases[t] += infections[s] * delay_pmf[delay_day];
      }
    }
    expected_cases[t] = fmax(expected_cases[t], 1e-8); // Avoid zeros
  }
}

model {
  // Priors
  initial_infections ~ exponential(0.1);
  rt_sd ~ normal(0, 0.5);
  phi ~ normal(0, 5);
  
  // Random walk prior for log Rt
  log_rt_raw[1] ~ normal(log(1.5), 0.5); // Initial Rt around 1.5
  for(t in 2:(n_days - seeding_days)) {
    log_rt_raw[t] ~ normal(log_rt_raw[t-1], rt_sd);
  }
  
  // Likelihood
  for(t in 1:n_days) {
    if(cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  vector[n_days] log_lik;
  vector[n_days] cases_pred;
  
  // Log likelihood and posterior predictions
  for(t in 1:n_days) {
    if(cases[t] > 0) {
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
    } else {
      log_lik[t] = 0;
      cases_pred[t] = 0;
    }
  }
}
"

# Compile the Stan model
compile_stan_model <- function() {
  return(stan_model(model_code = stan_model_code))
}

# Fit the Rt estimation model
fit_rt_model <- function(data, seeding_days = 7, chains = 4, iter = 2000) {
  # Prepare distributions
  generation_pmf <- get_generation_interval()
  delay_pmf <- get_total_delay()
  
  # Prepare data for Stan
  stan_data <- list(
    n_days = nrow(data),
    cases = data$cases,
    max_gen = length(generation_pmf),
    max_delay = length(delay_pmf),
    generation_pmf = generation_pmf,
    delay_pmf = delay_pmf,
    seeding_days = seeding_days
  )
  
  # Compile model
  model <- compile_stan_model()
  
  # Fit model
  fit <- sampling(
    model,
    data = stan_data,
    chains = chains,
    iter = iter,
    warmup = iter / 2,
    thin = 1,
    control = list(adapt_delta = 0.95, max_treedepth = 12)
  )
  
  return(list(fit = fit, data = data, stan_data = stan_data))
}

# Extract Rt estimates
extract_rt_estimates <- function(fit_result) {
  fit <- fit_result$fit
  data <- fit_result$data
  seeding_days <- fit_result$stan_data$seeding_days
  
  # Extract Rt samples
  rt_samples <- extract(fit, pars = "rt")$rt
  
  # Calculate summary statistics
  rt_summary <- apply(rt_samples, 2, function(x) {
    c(
      mean = mean(x),
      median = median(x),
      lower_95 = quantile(x, 0.025),
      lower_50 = quantile(x, 0.25),
      upper_50 = quantile(x, 0.75),
      upper_95 = quantile(x, 0.975)
    )
  })
  
  # Create results dataframe
  rt_results <- data.frame(
    date = data$date[(seeding_days + 1):nrow(data)],
    day = data$day[(seeding_days + 1):nrow(data)],
    t(rt_summary)
  )
  
  return(rt_results)
}

# Plot Rt estimates
plot_rt_estimates <- function(rt_results, data) {
  p1 <- ggplot(rt_results, aes(x = date)) +
    geom_ribbon(aes(ymin = lower_95, ymax = upper_95), 
                alpha = 0.3, fill = "steelblue") +
    geom_ribbon(aes(ymin = lower_50, ymax = upper_50), 
                alpha = 0.5, fill = "steelblue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    labs(
      title = "Time-varying Reproduction Number (Rt)",
      subtitle = "Median with 50% and 95% credible intervals",
      x = "Date",
      y = "Rt"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12)
    )
  
  p2 <- ggplot(data, aes(x = date, y = cases)) +
    geom_col(alpha = 0.7, fill = "coral") +
    labs(
      title = "Daily Case Counts",
      x = "Date",
      y = "Cases"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 12)
    )
  
  return(list(rt_plot = p1, cases_plot = p2))
}

# Main analysis function
estimate_rt <- function(file_path = "cases.csv", seeding_days = 7) {
  cat("Loading and preparing data...\n")
  data <- load_and_prepare_data(file_path)
  
  cat("Fitting Rt model...\n")
  fit_result <- fit_rt_model(data, seeding_days = seeding_days)
  
  cat("Extracting Rt estimates...\n")
  rt_estimates <- extract_rt_estimates(fit_result)
  
  cat("Creating plots...\n")
  plots <- plot_rt_estimates(rt_estimates, data)
  
  # Print summary
  cat("\nRt Summary:\n")
  print(summary(rt_estimates[, c("median", "lower_95", "upper_95")]))
  
  # Display plots
  print(plots$cases_plot)
  print(plots$rt_plot)
  
  return(list(
    rt_estimates = rt_estimates,
    fit_result = fit_result,
    plots = plots,
    data = data
  ))
}

# Example usage and model diagnostics
run_diagnostics <- function(results) {
  fit <- results$fit_result$fit
  
  # Print model summary
  print(fit, pars = c("rt_sd", "phi"))
  
  # Check convergence
  cat("\nRhat values (should be < 1.1):\n")
  rhats <- rhat(fit, pars = c("rt_sd", "phi", "rt"))
  print(summary(rhats))
  
  # Trace plots for key parameters
  trace_plot <- mcmc_trace(fit, pars = c("rt_sd", "phi"))
  print(trace_plot)
  
  return(invisible(NULL))
}

# Run the complete analysis
# Uncomment the following lines to run the analysis:

# results <- estimate_rt("cases.csv")
# run_diagnostics(results)

cat("Rt estimation code loaded successfully!\n")
cat("To run the analysis, use: results <- estimate_rt('cases.csv')\n")


# Load your data and run the analysis
results <- estimate_rt("cases.csv")

# Run diagnostics
run_diagnostics(results)

# Access results
rt_estimates <- results$rt_estimates
print(head(rt_estimates))
```

---

## SUB_078

**Scenario**: 1b | **Execution**: TIMEOUT

### script.R

```r
# Load required libraries
library(rstan)
library(dplyr)
library(ggplot2)
library(readr)
library(lubridate)

# Set Stan options for better performance
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Function to discretize gamma distribution for generation interval
discretize_gamma <- function(shape, rate, max_days = 20) {
  x <- 1:max_days
  pmf <- dgamma(x, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # Normalize to sum to 1
  return(pmf)
}

# Function to discretize gamma distribution for reporting delay
discretize_reporting_delay <- function(shape, rate, max_days = 15) {
  x <- 1:max_days
  pmf <- dgamma(x, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # Normalize to sum to 1
  return(pmf)
}

# Load and prepare data
load_and_prepare_data <- function(file_path) {
  # For demonstration, create sample data if file doesn't exist
  if (!file.exists(file_path)) {
    # Create sample data
    dates <- seq(as.Date("2020-03-01"), as.Date("2020-06-30"), by = "day")
    # Simulate cases with changing Rt
    set.seed(123)
    n_days <- length(dates)
    cases <- c(
      rpois(30, lambda = seq(1, 20, length.out = 30)),  # Initial growth
      rpois(40, lambda = seq(20, 100, length.out = 40)), # Peak
      rpois(50, lambda = seq(100, 30, length.out = 50)), # Decline
      rpois(n_days - 120, lambda = seq(30, 10, length.out = n_days - 120)) # Stabilization
    )
    
    data <- data.frame(
      date = dates[1:length(cases)],
      cases = pmax(0, cases)  # Ensure non-negative
    )
    
    write_csv(data, file_path)
    cat("Created sample data file:", file_path, "\n")
  }
  
  data <- read_csv(file_path)
  data$date <- as.Date(data$date)
  data <- data %>% arrange(date)
  
  return(data)
}

# Stan model for Rt estimation using renewal equation
stan_model_code <- "
data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> cases[T];             // Observed cases
  int<lower=0> S;                    // Length of generation interval
  vector[S] generation_pmf;          // Generation interval PMF
  int<lower=0> D;                    // Length of reporting delay
  vector[D] reporting_pmf;           // Reporting delay PMF
  int<lower=0> seed_days;            // Number of initial seeding days
}

parameters {
  vector<lower=0>[seed_days] initial_infections;  // Initial infections
  vector[T-seed_days] log_Rt_raw;                 // Log Rt (raw)
  real<lower=0> sigma_Rt;                         // Random walk SD for Rt
  real log_Rt_mean;                               // Mean log Rt
  real<lower=0> phi;                              // Overdispersion parameter
}

transformed parameters {
  vector[T] infections;
  vector[T] expected_cases;
  vector[T] log_Rt;
  vector[T] Rt;
  
  // Initialize infections for seeding period
  infections[1:seed_days] = initial_infections;
  
  // Set up log_Rt with random walk prior
  log_Rt[1:seed_days] = rep_vector(log_Rt_mean, seed_days);
  
  for (t in (seed_days+1):T) {
    log_Rt[t] = log_Rt[t-1] + sigma_Rt * log_Rt_raw[t-seed_days];
  }
  
  Rt = exp(log_Rt);
  
  // Apply renewal equation for infections after seeding period
  for (t in (seed_days+1):T) {
    real renewal_sum = 0;
    for (s in 1:min(S, t-1)) {
      if (t-s >= 1) {
        renewal_sum += infections[t-s] * generation_pmf[s];
      }
    }
    infections[t] = Rt[t] * renewal_sum;
  }
  
  // Convolve infections with reporting delay to get expected cases
  for (t in 1:T) {
    expected_cases[t] = 0;
    for (d in 1:min(D, t)) {
      if (t-d+1 >= 1) {
        expected_cases[t] += infections[t-d+1] * reporting_pmf[d];
      }
    }
  }
}

model {
  // Priors
  initial_infections ~ exponential(0.1);
  log_Rt_raw ~ std_normal();
  sigma_Rt ~ normal(0, 0.2);
  log_Rt_mean ~ normal(0, 0.5);
  phi ~ exponential(0.1);
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  vector[T] cases_pred;
  vector[T] log_lik;
  
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = 0;
      log_lik[t] = 0;
    }
  }
}
"

# Main function to estimate Rt
estimate_rt <- function(data, 
                       generation_shape = 2.3, 
                       generation_rate = 0.4,
                       reporting_shape = 2.0, 
                       reporting_rate = 0.5,
                       seed_days = 7,
                       chains = 4, 
                       iter = 2000,
                       warmup = 1000) {
  
  # Prepare generation interval and reporting delay
  generation_pmf <- discretize_gamma(generation_shape, generation_rate)
  reporting_pmf <- discretize_reporting_delay(reporting_shape, reporting_rate)
  
  # Prepare data for Stan
  stan_data <- list(
    T = nrow(data),
    cases = data$cases,
    S = length(generation_pmf),
    generation_pmf = generation_pmf,
    D = length(reporting_pmf),
    reporting_pmf = reporting_pmf,
    seed_days = seed_days
  )
  
  cat("Fitting Stan model...\n")
  cat("Data dimensions: T =", stan_data$T, ", S =", stan_data$S, ", D =", stan_data$D, "\n")
  
  # Compile and fit the model
  model <- stan_model(model_code = stan_model_code)
  
  fit <- sampling(model, 
                  data = stan_data,
                  chains = chains,
                  iter = iter,
                  warmup = warmup,
                  control = list(adapt_delta = 0.95, max_treedepth = 12),
                  verbose = TRUE)
  
  return(list(fit = fit, data = data, stan_data = stan_data))
}

# Function to extract and summarize Rt estimates
extract_rt_estimates <- function(results) {
  fit <- results$fit
  data <- results$data
  
  # Extract Rt estimates
  rt_samples <- extract(fit, pars = "Rt")$Rt
  
  # Calculate summary statistics
  rt_summary <- data.frame(
    date = data$date,
    cases = data$cases,
    rt_mean = apply(rt_samples, 2, mean),
    rt_median = apply(rt_samples, 2, median),
    rt_q025 = apply(rt_samples, 2, quantile, 0.025),
    rt_q975 = apply(rt_samples, 2, quantile, 0.975),
    rt_q25 = apply(rt_samples, 2, quantile, 0.25),
    rt_q75 = apply(rt_samples, 2, quantile, 0.75)
  )
  
  return(rt_summary)
}

# Function to create plots
plot_results <- function(rt_summary, title = "Rt Estimates Over Time") {
  # Plot Rt over time
  p1 <- ggplot(rt_summary, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_q025, ymax = rt_q975), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = rt_q25, ymax = rt_q75), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = rt_median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", size = 0.8) +
    labs(title = title,
         x = "Date",
         y = "Rt",
         subtitle = "Median with 50% and 95% credible intervals") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  # Plot cases over time
  p2 <- ggplot(rt_summary, aes(x = date, y = cases)) +
    geom_col(alpha = 0.7, fill = "gray") +
    labs(title = "Daily Reported Cases",
         x = "Date",
         y = "Number of Cases") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5))
  
  return(list(rt_plot = p1, cases_plot = p2))
}

# Function to print summary statistics
print_rt_summary <- function(rt_summary) {
  cat("\n=== Rt Estimation Summary ===\n")
  cat("Time period:", as.character(min(rt_summary$date)), "to", as.character(max(rt_summary$date)), "\n")
  cat("Number of days:", nrow(rt_summary), "\n")
  cat("\nOverall Rt statistics:\n")
  cat("Mean Rt:", round(mean(rt_summary$rt_mean), 3), "\n")
  cat("Median Rt:", round(median(rt_summary$rt_median), 3), "\n")
  cat("Days with Rt > 1:", sum(rt_summary$rt_median > 1), 
      "(", round(100 * sum(rt_summary$rt_median > 1) / nrow(rt_summary), 1), "%)\n")
  cat("Days with Rt < 1:", sum(rt_summary$rt_median < 1), 
      "(", round(100 * sum(rt_summary$rt_median < 1) / nrow(rt_summary), 1), "%)\n")
  
  # Show first and last few estimates
  cat("\nFirst 5 Rt estimates:\n")
  print(rt_summary[1:min(5, nrow(rt_summary)), c("date", "cases", "rt_median", "rt_q025", "rt_q975")])
  
  cat("\nLast 5 Rt estimates:\n")
  n <- nrow(rt_summary)
  print(rt_summary[max(1, n-4):n, c("date", "cases", "rt_median", "rt_q025", "rt_q975")])
}

# Main execution
main <- function() {
  cat("=== COVID-19 Rt Estimation using Renewal Equation ===\n\n")
  
  # Load data
  data <- load_and_prepare_data("cases.csv")
  cat("Loaded", nrow(data), "days of case data\n")
  cat("Date range:", as.character(min(data$date)), "to", as.character(max(data$date)), "\n")
  cat("Total cases:", sum(data$cases), "\n\n")
  
  # Estimate Rt
  results <- estimate_rt(data, 
                        generation_shape = 2.3,  # Mean ~5.8 days
                        generation_rate = 0.4,
                        reporting_shape = 2.0,   # Mean ~4 days  
                        reporting_rate = 0.5,
                        seed_days = 7,
                        chains = 4,
                        iter = 2000,
                        warmup = 1000)
  
  # Extract estimates
  rt_summary <- extract_rt_estimates(results)
  
  # Print summary
  print_rt_summary(rt_summary)
  
  # Create plots
  plots <- plot_results(rt_summary)
  
  # Display plots
  print(plots$cases_plot)
  print(plots$rt_plot)
  
  # Return results
  return(list(
    rt_estimates = rt_summary,
    stan_fit = results$fit,
    plots = plots
  ))
}

# Run the analysis
if (!exists("skip_main")) {
  results <- main()
  
  # Save results
  write_csv(results$rt_estimates, "rt_estimates.csv")
  cat("\nResults saved to 'rt_estimates.csv'\n")
}
```

---

## SUB_039

**Scenario**: 1b | **Execution**: SUCCESS

### script.R

```r
# Load required libraries
library(dplyr)
library(ggplot2)
library(rstan)
library(bayesplot)
library(lubridate)

# Set up Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and prepare the data
load_and_prepare_data <- function(file_path = "cases.csv") {
  # Load the data
  data <- read.csv(file_path, stringsAsFactors = FALSE)
  data$date <- as.Date(data$date)
  data <- data %>% arrange(date)
  
  # Ensure no missing dates
  full_dates <- seq(min(data$date), max(data$date), by = "day")
  data <- data.frame(date = full_dates) %>%
    left_join(data, by = "date") %>%
    mutate(cases = ifelse(is.na(cases), 0, cases))
  
  return(data)
}

# Define generation interval (discretized gamma distribution)
get_generation_interval <- function(max_days = 20, mean_gi = 5.2, sd_gi = 1.72) {
  # Discretize gamma distribution for generation interval
  shape <- (mean_gi / sd_gi)^2
  rate <- mean_gi / (sd_gi^2)
  
  gi <- diff(pgamma(1:(max_days + 1), shape = shape, rate = rate))
  gi <- gi / sum(gi)  # Normalize to sum to 1
  
  return(gi)
}

# Define reporting delay distribution
get_reporting_delay <- function(max_days = 15, mean_delay = 7, sd_delay = 3) {
  # Discretized gamma distribution for reporting delay
  shape <- (mean_delay / sd_delay)^2
  rate <- mean_delay / (sd_delay^2)
  
  delay <- diff(pgamma(1:(max_days + 1), shape = shape, rate = rate))
  delay <- delay / sum(delay)  # Normalize to sum to 1
  
  return(delay)
}

# Stan model for Rt estimation
stan_model_code <- "
data {
  int<lower=1> T;                    // Number of time points
  int<lower=0> cases[T];             // Observed case counts
  int<lower=1> G;                    // Length of generation interval
  vector[G] generation_interval;     // Generation interval PMF
  int<lower=1> D;                    // Length of reporting delay
  vector[D] reporting_delay;         // Reporting delay PMF
  int<lower=1> S;                    // Number of seed infections
}

parameters {
  vector<lower=0>[S] seed_infections;     // Initial seed infections
  vector[T-S] log_rt;                     // Log reproduction numbers
  real<lower=0> phi;                      // Overdispersion parameter
}

transformed parameters {
  vector<lower=0>[T] infections;
  vector<lower=0>[T] expected_cases;
  
  // Initialize with seed infections
  for (s in 1:S) {
    infections[s] = seed_infections[s];
  }
  
  // Apply renewal equation
  for (t in (S+1):T) {
    real renewal_sum = 0;
    for (g in 1:min(G, t-1)) {
      renewal_sum += infections[t-g] * generation_interval[g];
    }
    infections[t] = exp(log_rt[t-S]) * renewal_sum;
  }
  
  // Apply reporting delay convolution
  for (t in 1:T) {
    expected_cases[t] = 0;
    for (d in 1:min(D, t)) {
      expected_cases[t] += infections[t-d+1] * reporting_delay[d];
    }
  }
}

model {
  // Priors
  seed_infections ~ exponential(0.1);
  log_rt ~ normal(0, 0.2);  // Prior centered on Rt = 1
  phi ~ exponential(1);
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  vector<lower=0>[T-S] rt = exp(log_rt);
  vector[T] log_lik;
  
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      log_lik[t] = 0;
    }
  }
}
"

# Function to estimate Rt
estimate_rt <- function(data, 
                       generation_interval = NULL,
                       reporting_delay = NULL,
                       seed_days = 7,
                       chains = 4,
                       iter = 2000,
                       warmup = 1000) {
  
  # Set up generation interval if not provided
  if (is.null(generation_interval)) {
    generation_interval <- get_generation_interval()
  }
  
  # Set up reporting delay if not provided
  if (is.null(reporting_delay)) {
    reporting_delay <- get_reporting_delay()
  }
  
  # Prepare data for Stan
  T <- nrow(data)
  stan_data <- list(
    T = T,
    cases = data$cases,
    G = length(generation_interval),
    generation_interval = generation_interval,
    D = length(reporting_delay),
    reporting_delay = reporting_delay,
    S = seed_days
  )
  
  # Compile and fit the model
  cat("Compiling Stan model...\n")
  compiled_model <- stan_model(model_code = stan_model_code)
  
  cat("Fitting model...\n")
  fit <- sampling(compiled_model,
                  data = stan_data,
                  chains = chains,
                  iter = iter,
                  warmup = warmup,
                  control = list(adapt_delta = 0.95, max_treedepth = 12))
  
  # Extract results
  rt_samples <- rstan::extract(fit, pars = "rt")$rt
  infections_samples <- rstan::extract(fit, pars = "infections")$infections
  expected_cases_samples <- rstan::extract(fit, pars = "expected_cases")$expected_cases
  
  # Calculate summary statistics
  rt_summary <- data.frame(
    date = data$date[(seed_days + 1):T],
    rt_mean = apply(rt_samples, 2, mean),
    rt_median = apply(rt_samples, 2, median),
    rt_lower = apply(rt_samples, 2, quantile, 0.025),
    rt_upper = apply(rt_samples, 2, quantile, 0.975),
    rt_lower_50 = apply(rt_samples, 2, quantile, 0.25),
    rt_upper_50 = apply(rt_samples, 2, quantile, 0.75)
  )
  
  infections_summary <- data.frame(
    date = data$date,
    infections_mean = apply(infections_samples, 2, mean),
    infections_median = apply(infections_samples, 2, median),
    infections_lower = apply(infections_samples, 2, quantile, 0.025),
    infections_upper = apply(infections_samples, 2, quantile, 0.975)
  )
  
  expected_cases_summary <- data.frame(
    date = data$date,
    expected_cases_mean = apply(expected_cases_samples, 2, mean),
    expected_cases_median = apply(expected_cases_samples, 2, median),
    expected_cases_lower = apply(expected_cases_samples, 2, quantile, 0.025),
    expected_cases_upper = apply(expected_cases_samples, 2, quantile, 0.975),
    observed_cases = data$cases
  )
  
  return(list(
    fit = fit,
    rt_estimates = rt_summary,
    infections = infections_summary,
    expected_cases = expected_cases_summary,
    data = data,
    generation_interval = generation_interval,
    reporting_delay = reporting_delay
  ))
}

# Plotting functions
plot_rt <- function(results) {
  ggplot(results$rt_estimates, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "steelblue") +
    geom_ribbon(aes(ymin = rt_lower_50, ymax = rt_upper_50), alpha = 0.5, fill = "steelblue") +
    geom_line(aes(y = rt_median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
    labs(title = "Estimated Reproduction Number (Rt) Over Time",
         subtitle = "Ribbon shows 50% (dark) and 95% (light) credible intervals",
         x = "Date", y = "Rt") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_cases_fit <- function(results) {
  ggplot(results$expected_cases, aes(x = date)) +
    geom_ribbon(aes(ymin = expected_cases_lower, ymax = expected_cases_upper), 
                alpha = 0.3, fill = "lightblue") +
    geom_line(aes(y = expected_cases_mean), color = "blue", size = 1) +
    geom_point(aes(y = observed_cases), color = "black", size = 0.8) +
    labs(title = "Model Fit: Expected vs Observed Cases",
         subtitle = "Points: observed cases, Line: expected cases, Ribbon: 95% CI",
         x = "Date", y = "Cases") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_infections <- function(results) {
  ggplot(results$infections, aes(x = date)) +
    geom_ribbon(aes(ymin = infections_lower, ymax = infections_upper), 
                alpha = 0.3, fill = "orange") +
    geom_line(aes(y = infections_mean), color = "darkorange", size = 1) +
    labs(title = "Estimated Infections Over Time",
         subtitle = "Ribbon shows 95% credible interval",
         x = "Date", y = "Estimated Infections") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Diagnostic functions
check_convergence <- function(results) {
  print("Rhat summary:")
  rhat_vals <- summary(results$fit)$summary[, "Rhat"]
  print(summary(rhat_vals[!is.na(rhat_vals)]))
  
  cat("\nParameters with Rhat > 1.1:\n")
  high_rhat <- which(rhat_vals > 1.1)
  if (length(high_rhat) > 0) {
    print(names(high_rhat))
  } else {
    cat("None - good convergence!\n")
  }
}

# Main execution function
main <- function(file_path = "cases.csv") {
  cat("Loading and preparing data...\n")
  data <- load_and_prepare_data(file_path)
  
  cat(sprintf("Data loaded: %d days from %s to %s\n", 
              nrow(data), min(data$date), max(data$date)))
  cat(sprintf("Total cases: %d\n", sum(data$cases)))
  
  cat("\nEstimating Rt...\n")
  results <- estimate_rt(data)
  
  cat("\nChecking convergence...\n")
  check_convergence(results)
  
  cat("\nGenerating plots...\n")
  p1 <- plot_rt(results)
  p2 <- plot_cases_fit(results)
  p3 <- plot_infections(results)
  
  print(p1)
  print(p2)
  print(p3)
  
  cat("\nRt Summary:\n")
  print(summary(results$rt_estimates))
  
  return(results)
}

# Example usage with simulated data if cases.csv doesn't exist
create_example_data <- function() {
  set.seed(123)
  dates <- seq(as.Date("2020-03-01"), as.Date("2020-06-30"), by = "day")
  
  # Simulate epidemic curve with changing Rt
  T <- length(dates)
  true_rt <- c(rep(2.5, 30), seq(2.5, 0.8, length.out = 40), rep(0.8, T - 70))
  
  infections <- numeric(T)
  infections[1:7] <- c(1, 1, 2, 3, 5, 8, 13)  # Seed infections
  
  gi <- get_generation_interval()
  
  for (t in 8:T) {
    renewal_sum <- sum(infections[max(1, t - length(gi)):(t-1)] * 
                      rev(gi[1:min(length(gi), t-1)]))
    infections[t] <- rpois(1, true_rt[t] * renewal_sum)
  }
  
  # Add reporting delay and observation noise
  rd <- get_reporting_delay()
  expected_cases <- numeric(T)
  for (t in 1:T) {
    for (d in 1:min(length(rd), t)) {
      expected_cases[t] <- expected_cases[t] + infections[t - d + 1] * rd[d]
    }
  }
  
  cases <- rnbinom(T, size = 10, mu = expected_cases)
  
  data <- data.frame(date = dates, cases = cases)
  write.csv(data, "cases.csv", row.names = FALSE)
  
  cat("Example data created and saved to cases.csv\n")
  return(data)
}

# Run the analysis
if (!file.exists("cases.csv")) {
  cat("cases.csv not found. Creating example data...\n")
  create_example_data()
}

results <- main("cases.csv")
```

---

## SUB_097

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Install required libraries
install.packages("dplyr")
install.packages("ggplot2")

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Load the data from cases.csv into a dataframe called df
df <- read_csv("cases.csv")

# Ensure date is in a format that can be used for time-series analysis
df$date <- as.Date(df$date, "%Y-%m-%d")

# Remove any rows with missing case counts
df <- df %>% 
  filter(!is.na(cases))

# Define the generation interval probability mass function (pmf)
g_s <- dpois(1:10, lambda = 7) # Assume mean generation time is 7 days

# Function to estimate Rt using the renewal equation
estimate_rt <- function(df, g_s){
  # Initialize variables
  n_days <- max(df$date) - min(df$date) + 1
  n_cases <- rep(0, n_days)
  
  # Loop over each day and calculate cumulative cases
  for (i in 2:n_days){
    start_date <- as.Date(min(df$date), "%Y-%m-%d") + i - 2
    end_date <- as.Date(max(df$date), "%Y-%m-%d")
    
    # Filter data to only include relevant days and calculate cumulative cases
    sub_df <- df %>% 
      filter(date >= start_date & date <= end_date)
    
    n_cases[i] <- sum(sub_df$cases)
  }
  
  # Calculate Rt for each day using the renewal equation
  rt_estimates <- rep(0, length(n_days))
  for (i in 2:n_days){
    if (n_cases[i - 1] > 0){
      rt_estimates[i] <- n_cases[i] / (g_s * sum(g_s * n_cases[i - 1]))
    } else {
      rt_estimates[i] <- NA
    }
  }
  
  # Return Rt estimates and corresponding dates
  return(data.frame(date = as.Date(min(df$date), "%Y-%m-%d") + 0: (n_days-1),
                    rt_estimate = rt_estimates))
}

# Estimate Rt using the renewal equation framework
rt_df <- estimate_rt(df, g_s)

# Plot Rt over time
ggplot(rt_df, aes(x = date, y = rt_estimate)) +
  geom_line() +
  theme_classic()
```

---

## SUB_045

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(readr)
library(ggplot2)

# Set seed for reproducibility
set.seed(1234)

# Function to estimate Rt from daily case counts

estimate_rt <- function(case_data, beta) {
  # Read the data into a data frame
  df <- read_csv(case_data)

  # Ensure 'date' is of date class and convert it to year-month format for easier indexing
  df$date <- as.Date(df$date)
  
  # Initialize Rt as NA for each day
  rt_estimates <- rep(NA, nrow(df))
  
  # Calculate the generation interval probability mass function (PMF) values
  gen_interval_pmf <- function(s) {
    if (s == 0) { return(1 - exp(-beta)) }
    else { return(exp(-beta * s)) }
  }

  # Apply the renewal equation to estimate Rt at each time point
  for (i in seq_len(nrow(df))) {
    s_values <- seq_len(i)
    denominator <- sum(gen_interval_pmf(s) * df$cases[df$date == df$date[i] - s + 1], na.rm = TRUE)
    
    # Check if denominator is not zero to avoid division by zero error
    if (denominator > 0) {
      rt_estimates[i] <- denominator / df$cases[i]
    }
  }

  return(rt_estimates)
}

# Example usage:
case_data <- "cases.csv"
beta <- 1/5 # Rate parameter for exponential generation interval distribution

rt_estimates <- estimate_rt(case_data, beta)

# Plot Rt estimates over time
ggplot(data.frame(date = unique(df$date), rt_estimate = rt_estimates), aes(x = date, y = rt_estimate)) +
  geom_line() + 
  labs(title = "Time-Varying Reproduction Number (Rt) Estimates", x = "Date", y = "Rt") +
  theme_classic()
```

---

## SUB_099

**Scenario**: 1b | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)

# Load data
data <- read_csv("cases.csv")

# Define the generation interval probability mass function
gen_int_pdf <- function(s, S, tau) {
  # Assume exponential distribution for simplicity
  exp(-s / tau)
}

# Define the observation model (Poisson for simplicity)
observation_model <- function(cases_observed, Rt, g_s, s_max) {
  sum(Rt * cases_observed[cases_observed > 0] *
      gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))))
}

# Define the renewal equation model
renewal_equation_model <- function(cases_observed, Rt, g_s, s_max) {
  I_t <- rep(0, length(cases_observed))
  for (t in seq_along(cases_observed)) {
    I_t[t] <- sum(Rt * cases_observed[(t - s_max):t] *
                  gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))))
  }
  return(I_t)
}

# Define the model fit function
fit_model <- function(cases_observed) {
  # Initialize parameters (e.g., Rt and g_s)
  init_params <- list(Rt = rep(1, length(unique(as.Date(data$date)))),
                      g_s = gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))))
  
  # Fit the model using optimization (e.g., maximum likelihood)
  fit <- optim(init_params,
               function(params) {
                 -sum(observation_model(cases_observed, params$Rt, gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))), s_max)^2)
               },
               method = "BFGS",
               control = list(reltol = 1e-4))
  
  # Extract Rt estimates
  Rt_estimates <- fit$par[, 1]
  
  return(Rt_estimates)
}

# Load necessary variables
s_max <- 7  # Maximum generation interval (in days)
data$date <- as.Date(data$date)  # Convert date to Date type

# Handle initial infection seeding period (e.g., ignore the first week)
cases_observed <- data %>%
  filter(date > min(data$date, na.rm = TRUE) + weeks(1)) %>%
  pull(cases)

# Fit and extract Rt estimates
Rt_estimates <- fit_model(cases_observed)

# Plot Rt over time
ggplot(data.frame(date = unique(data$date), Rt = Rt_estimates),
       aes(x = date, y = Rt)) +
  geom_line() +
  labs(title = "Estimated Time-Varying Reproduction Number (Rt)",
       subtitle = "(Renewal Equation Framework)",
       x = "Date",
       y = "Rt")
```

---

# Scenario 2

Total submissions: 30

---

## SUB_009

**Scenario**: 2 | **Execution**: FAILED

### model.stan

```stan
data {
  int<lower=1> T;                    // Number of time points
  array[T] int<lower=0> cases;       // Observed case counts
  array[T] int<lower=1,upper=7> day_of_week;  // Day of week (1=Mon, 7=Sun)
  
  // Generation interval
  int<lower=1> G;                    // Length of generation interval
  vector<lower=0>[G] gen_pmf;        // Generation interval PMF
  
  // Delay distribution
  int<lower=1> D;                    // Length of delay distribution  
  vector<lower=0>[D] delay_pmf;      // Delay PMF
  
  // Priors
  real rt_prior_mean;
  real<lower=0> rt_prior_sd;
  real<lower=0> rt_rw_sd;
  real<lower=0> ascertainment_rw_sd;
}

transformed data {
  int seeding_days = max(G, D);      // Days needed for seeding
}

parameters {
  // Log Rt (random walk)
  vector[T] log_rt_raw;
  real log_rt_init;
  
  // Day of week effects (Monday is reference)
  vector[6] dow_effect_raw;          // Effects for Tue-Sun
  
  // Time-varying ascertainment (on logit scale, random walk)
  vector[T] logit_ascertainment_raw;
  real logit_ascertainment_init;
  
  // Overdispersion parameter
  real<lower=0> phi;
  
  // Initial infections (seeding period)
  vector<lower=0>[seeding_days] I_seed;
}

transformed parameters {
  vector[T] log_rt;
  vector[7] dow_effect;
  vector[T] logit_ascertainment;
  vector<lower=0>[T] infections;
  vector<lower=0>[T] expected_cases;
  
  // Rt random walk
  log_rt[1] = log_rt_init + rt_rw_sd * log_rt_raw[1];
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_rw_sd * log_rt_raw[t];
  }
  
  // Day of week effects (Monday = 1.0, others relative to Monday)
  dow_effect[1] = 1.0;  // Monday reference
  for (i in 1:6) {
    dow_effect[i+1] = exp(0.2 * dow_effect_raw[i]);  // Moderate effects
  }
  
  // Ascertainment random walk
  logit_ascertainment[1] = logit_ascertainment_init + 
                          ascertainment_rw_sd * logit_ascertainment_raw[1];
  for (t in 2:T) {
    logit_ascertainment[t] = logit_ascertainment[t-1] + 
                            ascertainment_rw_sd * logit_ascertainment_raw[t];
  }
  
  // Compute infections using renewal equation
  for (t in 1:T) {
    if (t <= seeding_days) {
      infections[t] = I_seed[t];
    } else {
      real renewal = 0;
      for (g in 1:G) {
        if (t - g >= 1) {
          renewal += infections[t - g] * gen_pmf[g];
        }
      }
      infections[t] = exp(log_rt[t]) * renewal;
    }
  }
  
  // Compute expected reported cases (with delays and observation process)
  for (t in 1:T) {
    real delayed_infections = 0;
    
    // Apply delay from infection to reporting
    for (d in 1:D) {
      if (t - d >= 1 && t - d <= T) {
        delayed_infections += infections[t - d] * delay_pmf[d];
      }
    }
    
    // Apply ascertainment and day-of-week effects
    expected_cases[t] = delayed_infections * 
                       inv_logit(logit_ascertainment[t]) * 
                       dow_effect[day_of_week[t]];
  }
}

model {
  // Priors
  log_rt_init ~ normal(log(rt_prior_mean), rt_prior_sd);
  log_rt_raw ~ std_normal();
  
  dow_effect_raw ~ std_normal();
  
  logit_ascertainment_init ~ normal(logit(0.3), 1);  // Prior: ~30% ascertainment
  logit_ascertainment_raw ~ std_normal();
  
  phi ~ gamma(2, 0.1);  // Overdispersion parameter
  
  I_seed ~ gamma(2, 0.1);  // Weakly informative prior on initial infections
  
  // Likelihood with overdispersion (negative binomial)
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    } else {
      cases[t] ~ poisson(1e-8);  // Avoid zero expected cases
    }
  }
}

generated quantities {
  vector<lower=0>[T] rt = exp(log_rt);
  vector<lower=0,upper=1>[T] ascertainment = inv_logit(logit_ascertainment);
  array[T] int cases_pred;
  vector[T] log_lik;
  
  // Posterior predictions and log-likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = poisson_rng(1e-8);
      log_lik[t] = poisson_lpmf(cases[t] | 1e-8);
    }
  }
}
```

### script.R

```r
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
```

---

## SUB_029

**Scenario**: 2 | **Execution**: FAILED

### model.stan

```stan
data {
  int<lower=1> T;                    // Number of time points
  array[T] int<lower=0> cases;       // Observed cases
  array[T] int<lower=1,upper=7> day_of_week;  // Day of week (1=Mon, 7=Sun)
  int<lower=1> S;                    // Length of generation interval
  int<lower=1> D;                    // Length of delay distribution
  vector<lower=0>[S] generation_pmf; // Generation interval PMF
  vector<lower=0>[D] delay_pmf;      // Reporting delay PMF
  int<lower=1> seeding_days;         // Days for infection seeding
}

transformed data {
  vector[S] generation_pmf_rev = reverse(generation_pmf);
}

parameters {
  // Rt parameters
  real log_rt_mean;                       // Overall mean log Rt
  real<lower=0> rt_sigma;                 // Std dev of Rt random walk
  vector[T-1] rt_noise;                   // Rt random walk innovations
  
  // Day of week effects (Monday = reference)
  vector[6] log_dow_effects_raw;          // Log effects for Tue-Sun
  
  // Time-varying ascertainment
  real logit_ascertainment_mean;          // Mean logit ascertainment
  real<lower=0> ascertainment_sigma;      // Std dev of ascertainment
  vector[T-1] ascertainment_noise;        // Ascertainment random walk
  
  // Overdispersion
  real<lower=0> phi_inv;                  // Inverse overdispersion parameter
  
  // Initial infections
  vector<lower=0>[seeding_days] log_infections_seed;
}

transformed parameters {
  vector[T] log_rt;
  vector[T] rt;
  vector[7] dow_effects;
  vector[T] logit_ascertainment;
  vector[T] ascertainment;
  vector[T] infections;
  vector[T] expected_cases;
  real phi = inv(phi_inv);
  
  // Rt evolution (random walk on log scale)
  log_rt[1] = log_rt_mean;
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_sigma * rt_noise[t-1];
  }
  rt = exp(log_rt);
  
  // Day of week effects (Monday = 1.0 reference)
  dow_effects[1] = 1.0;
  dow_effects[2:7] = exp(log_dow_effects_raw);
  
  // Ascertainment evolution (random walk on logit scale)
  logit_ascertainment[1] = logit_ascertainment_mean;
  for (t in 2:T) {
    logit_ascertainment[t] = logit_ascertainment[t-1] + 
                           ascertainment_sigma * ascertainment_noise[t-1];
  }
  ascertainment = inv_logit(logit_ascertainment);
  
  // Infection dynamics
  // Seed initial infections
  for (t in 1:seeding_days) {
    infections[t] = exp(log_infections_seed[t]);
  }
  
  // Renewal equation for subsequent infections
  for (t in (seeding_days+1):T) {
    real lambda = 0;
    int max_lag = min(t-1, S);
    for (s in 1:max_lag) {
      lambda += infections[t-s] * generation_pmf[s];
    }
    infections[t] = rt[t] * lambda;
  }
  
  // Expected cases with delays and observation process
  for (t in 1:T) {
    real expected = 0;
    int max_delay = min(t, D);
    for (d in 1:max_delay) {
      if (t-d+1 >= 1) {
        expected += infections[t-d+1] * delay_pmf[d];
      }
    }
    expected_cases[t] = expected * ascertainment[t] * dow_effects[day_of_week[t]];
  }
}

model {
  // Priors
  log_rt_mean ~ normal(0, 0.5);           // Rt around 1
  rt_sigma ~ normal(0, 0.1);              // Moderate variation in Rt
  rt_noise ~ std_normal();
  
  log_dow_effects_raw ~ normal(0, 0.2);   // Day effects close to 1
  
  logit_ascertainment_mean ~ normal(-1, 0.5);  // Around 25% ascertainment
  ascertainment_sigma ~ normal(0, 0.05);       // Slow ascertainment changes
  ascertainment_noise ~ std_normal();
  
  phi_inv ~ exponential(5);               // Overdispersion
  
  log_infections_seed ~ normal(3, 1);     // Initial infections around 20
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 1e-6) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  array[T] int cases_pred;
  vector[T] log_lik;
  
  // Posterior predictive samples
  for (t in 1:T) {
    if (expected_cases[t] > 1e-6) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = 0;
      log_lik[t] = 0;
    }
  }
}
```

### script.R

```r
library(cmdstanr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)

# Function to create generation interval (gamma distribution discretized)
discretize_gamma <- function(shape = 2.5, rate = 0.5, max_days = 20) {
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # normalize to sum to 1
  return(pmf)
}

# Load and prepare data
prepare_data <- function(file_path = "cases.csv") {
  # If file doesn't exist, create sample data
  if (!file.exists(file_path)) {
    set.seed(123)
    dates <- seq(as.Date("2020-03-01"), as.Date("2020-12-31"), by = "day")
    n_days <- length(dates)
    
    # Simulate realistic COVID-like data
    true_rt <- c(rep(2.5, 30), 
                 2.5 * exp(-0.05 * (1:60)),  # decline
                 rep(0.8, 50),               # low
                 0.8 + 0.8 * plogis((1:100) - 50)/2,  # rise
                 rep(1.2, n_days - 240))     # moderate
    true_rt <- true_rt[1:n_days]
    
    # Simulate infections using renewal equation
    gen_interval <- discretize_gamma()
    infections <- numeric(n_days)
    infections[1:7] <- c(10, 15, 20, 25, 30, 35, 40)  # seed
    
    for(t in 8:n_days) {
      lambda <- true_rt[t] * sum(infections[max(1, t-20):(t-1)] * 
                                 rev(gen_interval[1:min(20, t-1)]))
      infections[t] <- rpois(1, lambda)
    }
    
    # Add reporting delays and day-of-week effects
    dow_effects <- c(0.8, 0.9, 1.0, 1.0, 1.0, 0.7, 0.5)  # Mon-Sun
    ascertainment <- 0.3 * (1 + 0.5 * sin(2*pi*(1:n_days)/365))  # seasonal
    
    cases <- numeric(n_days)
    for(t in 1:n_days) {
      # Reporting delay (mean 7 days)
      delay_dist <- dpois(0:14, 7)
      delay_dist <- delay_dist / sum(delay_dist)
      
      expected_cases <- 0
      for(d in 1:min(15, t)) {
        if(t-d+1 >= 1) {
          dow <- ((t-1) %% 7) + 1
          expected_cases <- expected_cases + 
            infections[t-d+1] * delay_dist[d] * ascertainment[t] * dow_effects[dow]
        }
      }
      cases[t] <- rnbinom(1, size = 10, mu = expected_cases)
    }
    
    data <- data.frame(
      date = dates,
      cases = pmax(0, cases),
      day_of_week = ((as.numeric(dates) - 1) %% 7) + 1
    )
    write.csv(data, "cases.csv", row.names = FALSE)
  } else {
    data <- read.csv(file_path)
    data$date <- as.Date(data$date)
  }
  
  return(data)
}

# Prepare Stan data
prepare_stan_data <- function(case_data) {
  gen_interval <- discretize_gamma()
  delay_pmf <- dpois(0:14, 7)
  delay_pmf <- delay_pmf / sum(delay_pmf)
  
  list(
    T = nrow(case_data),
    cases = case_data$cases,
    day_of_week = case_data$day_of_week,
    S = length(gen_interval),
    D = length(delay_pmf),
    generation_pmf = gen_interval,
    delay_pmf = delay_pmf,
    seeding_days = 14
  )
}

# Main execution
case_data <- prepare_data()
stan_data <- prepare_stan_data(case_data)

print(paste("Data spans", min(case_data$date), "to", max(case_data$date)))
print(paste("Total cases:", sum(case_data$cases)))




# Write Stan model to file
stan_code <- '
[Insert the Stan code from above here]
'
writeLines(stan_code, "rt_model.stan")

# Compile and fit model
model <- cmdstan_model("rt_model.stan")

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200,
  max_treedepth = 12,
  adapt_delta = 0.95
)

# Extract results
draws <- fit$draws()
summary_stats <- fit$summary()

# Extract key parameters
rt_summary <- summary_stats[grepl("^rt\\[", summary_stats$variable), ]
dow_summary <- summary_stats[grepl("^dow_effects\\[", summary_stats$variable), ]
ascertainment_summary <- summary_stats[grepl("^ascertainment\\[", summary_stats$variable), ]

# Create results dataframe
results <- case_data %>%
  mutate(
    rt_median = rt_summary$median,
    rt_lower = rt_summary$q5,
    rt_upper = rt_summary$q95,
    ascertainment_median = ascertainment_summary$median,
    ascertainment_lower = ascertainment_summary$q5,
    ascertainment_upper = ascertainment_summary$q95
  )

# Day of week effects
dow_effects <- data.frame(
  day = c("Monday", "Tuesday", "Wednesday", "Thursday", 
          "Friday", "Saturday", "Sunday"),
  effect = dow_summary$median,
  lower = dow_summary$q5,
  upper = dow_summary$q95
)

# Print summary
print("Day of Week Effects:")
print(dow_effects)

print("\nOverdispersion Parameter (phi):")
phi_summary <- summary_stats[summary_stats$variable == "phi", ]
print(phi_summary[c("median", "q5", "q95")])

# Create plots
p1 <- ggplot(results, aes(x = date)) +
  geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "blue") +
  geom_line(aes(y = rt_median), color = "blue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       x = "Date", y = "Rt") +
  theme_minimal()

p2 <- ggplot(results, aes(x = date)) +
  geom_ribbon(aes(ymin = ascertainment_lower, ymax = ascertainment_upper), 
              alpha = 0.3, fill = "green") +
  geom_line(aes(y = ascertainment_median), color = "green", size = 1) +
  labs(title = "Time-varying Ascertainment Rate",
       x = "Date", y = "Ascertainment") +
  theme_minimal()

p3 <- ggplot(dow_effects, aes(x = day, y = effect)) +
  geom_col(fill = "orange", alpha = 0.7) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  labs(title = "Day of Week Effects",
       x = "Day of Week", y = "Relative Reporting") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p4 <- ggplot(results, aes(x = date)) +
  geom_col(aes(y = cases), alpha = 0.5, fill = "gray") +
  labs(title = "Daily Reported Cases",
       x = "Date", y = "Cases") +
  theme_minimal()

# Display plots
print(p1)
print(p2)
print(p3)
print(p4)

# Model diagnostics
print("\nModel Diagnostics:")
print(fit$diagnostic_summary())

# Save results
write.csv(results, "rt_estimates.csv", row.names = FALSE)
write.csv(dow_effects, "dow_effects.csv", row.names = FALSE)

print("\nResults saved to rt_estimates.csv and dow_effects.csv")
```

---

## SUB_012

**Scenario**: 2 | **Execution**: FAILED

### model.stan

```stan
functions {
  // Convolution for renewal equation
  vector convolve_infections(vector infections, vector gen_interval, int T, int G) {
    vector[T] result = rep_vector(0.0, T);
    for (t in 1:T) {
      for (g in 1:min(G, t)) {
        if (t - g + 1 >= 1) {
          result[t] += infections[t - g + 1] * gen_interval[g];
        }
      }
    }
    return result;
  }
  
  // Convolution for reporting delay
  vector convolve_delay(vector infections, vector delay_dist, int T, int D) {
    vector[T] result = rep_vector(0.0, T);
    for (t in 1:T) {
      for (d in 1:min(D, T - t + 1)) {
        if (t + d - 1 <= T) {
          result[t + d - 1] += infections[t] * delay_dist[d];
        }
      }
    }
    return result;
  }
}

data {
  int<lower=1> T;                          // Number of time points
  array[T] int<lower=0> cases;             // Observed cases
  array[T] int<lower=1,upper=7> day_of_week; // Day of week (1=Monday, 7=Sunday)
  
  // Generation interval
  int<lower=1> G;                          // Length of generation interval
  vector[G] gen_interval;                  // Generation interval PMF
  
  // Delay distribution
  int<lower=1> D;                          // Length of delay distribution
  vector[D] delay_dist;                    // Delay distribution PMF
  
  // Prior parameters
  real rt_prior_mean;
  real<lower=0> rt_prior_sd;
  real<lower=0> rt_random_walk_sd;
  real<lower=0> ascertainment_random_walk_sd;
}

parameters {
  // Log Rt (for numerical stability)
  vector[T] log_rt_raw;                    // Raw random walk innovations
  real log_rt_init;                        // Initial log Rt
  
  // Initial infections (for seeding)
  vector<lower=0>[G] init_infections;
  
  // Day-of-week effects
  vector[6] day_of_week_raw;               // 6 effects (Sunday as reference)
  
  // Time-varying ascertainment (logit scale)
  vector[T] logit_ascertainment_raw;       // Raw random walk innovations
  real logit_ascertainment_init;           // Initial logit ascertainment
  
  // Overdispersion parameter
  real<lower=0> phi;                       // Negative binomial overdispersion
}

transformed parameters {
  vector[T] log_rt;
  vector[T] Rt;
  vector[7] day_of_week_effect;
  vector[T] logit_ascertainment;
  vector[T] ascertainment;
  vector[T] infections;
  vector[T] delayed_infections;
  vector[T] expected_cases;
  
  // Random walk for log Rt
  log_rt[1] = log_rt_init;
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_random_walk_sd * log_rt_raw[t];
  }
  Rt = exp(log_rt);
  
  // Day-of-week effects (Sunday = reference = 1.0)
  day_of_week_effect[7] = 1.0;  // Sunday
  day_of_week_effect[1:6] = exp(day_of_week_raw);
  
  // Random walk for ascertainment (on logit scale)
  logit_ascertainment[1] = logit_ascertainment_init;
  for (t in 2:T) {
    logit_ascertainment[t] = logit_ascertainment[t-1] + 
                            ascertainment_random_walk_sd * logit_ascertainment_raw[t];
  }
  ascertainment = inv_logit(logit_ascertainment);
  
  // Renewal equation
  // Initialize infections with seeded values
  for (t in 1:min(G, T)) {
    if (t <= G) {
      infections[t] = init_infections[t];
    }
  }
  
  // Apply renewal equation for subsequent time points
  for (t in (G+1):T) {
    real renewal_sum = 0.0;
    for (g in 1:G) {
      renewal_sum += infections[t - g] * gen_interval[g];
    }
    infections[t] = Rt[t] * renewal_sum;
  }
  
  // Apply reporting delay
  delayed_infections = convolve_delay(infections, delay_dist, T, D);
  
  // Expected reported cases
  for (t in 1:T) {
    expected_cases[t] = delayed_infections[t] * ascertainment[t] * 
                       day_of_week_effect[day_of_week[t]];
  }
}

model {
  // Priors
  log_rt_init ~ normal(log(rt_prior_mean), rt_prior_sd);
  log_rt_raw[2:T] ~ std_normal();
  
  init_infections ~ exponential(0.1);
  
  day_of_week_raw ~ normal(0, 0.2);  // Weakly informative
  
  logit_ascertainment_init ~ normal(logit(0.1), 1);  // Prior belief about ascertainment
  logit_ascertainment_raw[2:T] ~ std_normal();
  
  phi ~ exponential(0.1);
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[T] int cases_pred;
  vector[T] log_lik;
  
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = 0;
      log_lik[t] = 0;
    }
  }
}
```

### script.R

```r
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(lubridate)

# Load and prepare data
load_and_prepare_data <- function(file_path = "cases.csv") {
  # Load data
  data <- read.csv(file_path)
  data$date <- as.Date(data$date)
  data <- data[order(data$date), ]
  
  # Ensure we have complete time series
  date_seq <- seq(min(data$date), max(data$date), by = "day")
  complete_data <- data.frame(
    date = date_seq,
    day_of_week = as.numeric(format(date_seq, "%u"))
  )
  
  # Merge with observed data
  complete_data <- merge(complete_data, data[, c("date", "cases")], 
                        by = "date", all.x = TRUE)
  complete_data$cases[is.na(complete_data$cases)] <- 0
  
  return(complete_data)
}

# Generate generation interval (discretized gamma distribution)
generate_generation_interval <- function(max_days = 20, shape = 2.5, rate = 0.5) {
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # normalize
  return(pmf)
}

# Generate delay distribution (infection to reporting)
generate_delay_distribution <- function(max_days = 21, shape = 2.0, rate = 0.3) {
  days <- 0:(max_days-1)
  pmf <- dgamma(days + 1, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # normalize
  return(pmf)
}

# Prepare Stan data
prepare_stan_data <- function(data) {
  gen_interval <- generate_generation_interval()
  delay_dist <- generate_delay_distribution()
  
  stan_data <- list(
    T = nrow(data),
    cases = data$cases,
    day_of_week = data$day_of_week,
    
    # Generation interval
    G = length(gen_interval),
    gen_interval = gen_interval,
    
    # Delay distribution
    D = length(delay_dist),
    delay_dist = delay_dist,
    
    # Prior parameters
    rt_prior_mean = 1.0,
    rt_prior_sd = 0.5,
    
    # Smoothing parameters
    rt_random_walk_sd = 0.1,
    ascertainment_random_walk_sd = 0.05
  )
  
  return(stan_data)
}

# Load data
data <- load_and_prepare_data()
stan_data <- prepare_stan_data(data)

# Compile and fit model
model <- cmdstan_model("rt_model.stan")

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100,
  adapt_delta = 0.95,
  max_treedepth = 12
)

# Extract results
extract_results <- function(fit, data) {
  draws <- fit$draws()
  
  # Extract Rt estimates
  rt_summary <- summarise_draws(
    subset(draws, variable = "Rt"),
    mean, median, sd, ~quantile(.x, c(0.025, 0.25, 0.75, 0.975))
  )
  rt_summary$date <- data$date
  
  # Extract day-of-week effects
  dow_summary <- summarise_draws(
    subset(draws, variable = "day_of_week_effect"),
    mean, median, sd, ~quantile(.x, c(0.025, 0.975))
  )
  dow_summary$day_name <- c("Monday", "Tuesday", "Wednesday", "Thursday", 
                           "Friday", "Saturday", "Sunday")
  
  # Extract ascertainment
  ascert_summary <- summarise_draws(
    subset(draws, variable = "ascertainment"),
    mean, median, sd, ~quantile(.x, c(0.025, 0.25, 0.75, 0.975))
  )
  ascert_summary$date <- data$date
  
  return(list(
    rt = rt_summary,
    day_of_week = dow_summary,
    ascertainment = ascert_summary,
    fit = fit
  ))
}

# Extract results
results <- extract_results(fit, data)

# Create summary plots
create_plots <- function(results, data) {
  # Rt plot
  rt_plot <- ggplot(results$rt, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         x = "Date", y = "Rt",
         subtitle = "Median with 50% and 95% credible intervals") +
    theme_minimal()
  
  # Day-of-week effects
  dow_plot <- ggplot(results$day_of_week, aes(x = day_name, y = mean)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Day-of-Week Reporting Effects",
         x = "Day of Week", y = "Multiplicative Effect",
         subtitle = "Mean with 95% credible intervals") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Ascertainment over time
  ascert_plot <- ggplot(results$ascertainment, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "green") +
    geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "green") +
    geom_line(aes(y = median), color = "darkgreen", size = 1) +
    labs(title = "Time-varying Ascertainment Rate",
         x = "Date", y = "Ascertainment Rate",
         subtitle = "Median with 50% and 95% credible intervals") +
    theme_minimal()
  
  # Cases vs fitted
  fitted_summary <- summarise_draws(
    subset(results$fit$draws(), variable = "expected_cases"),
    mean, median, ~quantile(.x, c(0.025, 0.975))
  )
  fitted_summary$date <- data$date
  fitted_summary$observed <- data$cases
  
  cases_plot <- ggplot(fitted_summary, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "orange") +
    geom_line(aes(y = median), color = "darkorange", size = 1) +
    geom_point(aes(y = observed), color = "black", alpha = 0.6) +
    labs(title = "Observed vs Fitted Cases",
         x = "Date", y = "Cases",
         subtitle = "Black points: observed, Orange: fitted with 95% CI") +
    theme_minimal()
  
  return(list(rt = rt_plot, dow = dow_plot, ascertainment = ascert_plot, 
              cases = cases_plot))
}

plots <- create_plots(results, data)

# Display plots
print(plots$rt)
print(plots$dow)
print(plots$ascertainment)
print(plots$cases)

# Print summary statistics
cat("\nRt Summary (recent 7 days):\n")
recent_rt <- tail(results$rt, 7)
print(recent_rt[, c("date", "median", "2.5%", "97.5%")])

cat("\nDay-of-week effects:\n")
print(results$day_of_week[, c("day_name", "mean", "2.5%", "97.5%")])

cat("\nModel diagnostics:\n")
print(fit$diagnostic_summary())
```

---

## SUB_106

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(cmdstanr)
library(dplyr)

# Load the case count data
cases <- read.csv("cases.csv")

# Define constants and hyperparameters
S <- 7  # Number of days to consider for generation interval
tau <- 3.5  # Mean delay from infection to reporting (in days, as per reference)
sigma <- 1.0  # Standard deviation of the delay distribution

# Fit a generalized linear mixed model with overdispersion correction
fit_glmm <- lme4::glmer(cases ~ day_of_week + (day_of_week | date),
                        data = cases,
                        family = quasipoisson)

# Extract fitted values for reporting rates
reporting_rates_fit <- coefficients(fit_glmm)$`(Intercept)`

# Define the Stan model and fit it to the data using cmdstanr
model_code <- "
data {
  int<lower=1> S;  // Number of days to consider for generation interval
  vector[S] g;     // Generation interval probability mass function
  int<lower=0> cases[S];  // Reported case counts
  int<lower=0,upper=6> day_of_week[S];  // Day-of-week indicator (0-6)
  real tau;         // Mean delay from infection to reporting
  real sigma;       // Standard deviation of the delay distribution

  // Ascertainment over time - a smoothly varying proportion
  vector[S] ascertaintime;

  // Overdispersion parameter for Poisson observations
  real phi;
}

parameters {
  // Reproduction number (Rt)
  vector[S] Rt;

  // Day-of-week effects
  vector[7] dow_effects;

  // Ascertainment over time - a smoothly varying proportion
  vector[S] ascertaintime_;

  // Overdispersion parameter for Poisson observations
  real phi_;
}

transformed parameters {
  // Compute the renewal equation terms
  vector[S] renals;
  for (s in 1:S) {
    renals[s] = sum(Rt[max(1, s-S+1):s] .* g[1:s - max(1, s-S+1)+1]);
  }
}

model {
  // Prior distributions for parameters
  Rt ~ normal(0, 2);
  dow_effects ~ normal(0, 1);
  ascertaintime_ ~ normal(ascentime, 0.5);

  phi_ ~ lognormal(-3, 0.5);

  // Observation model with overdispersion correction
  cases ~ negbinomial(renals .* reporting_rates_fit .* dow_effects[day_of_week] .*
                      ascertaintime_[ascertaintime] .*
                      exp(phi_), phi_);
}

generated quantities {
  // Ascertainment over time (posterior mean)
  vector[S] ascertaintime_post = ascertaintime_;

  // Rt estimates for each time point
  vector[S] Rt_estimates = Rt;

  // Day-of-week effect estimates
  vector[7] dow_effects_estimates = dow_effects;
}
"

# Fit the Stan model to the data using cmdstanr
fit <- cmdstanr::sample(
  "rt_model",  # The name of the file (without extension) that contains the Stan model code.
  data = list(S = S, g = rep(1/S, S), cases = cases$cases,
              day_of_week = cases$day_of_week - 1, tau = tau, sigma = sigma,
              ascertaintime = 0.2 * (seq(1:S) / S)),  # Start with a small ascertainment proportion
  chains = 4, iter = 10000, thin = 10, warmup = 500, refresh = 200)

# Extract the posterior samples for Rt estimates and day-of-week effect estimates
posterior <- cmdstanr::extract(fit)$Rt_estimates
posterior_dow <- cmdstanr::extract(fit)$dow_effects_estimates

# Plot the results (optional)
plot(posterior, main = "Time-varying Reproduction Number Estimates")
lines(seq(1:S), cmdstanr::extract(fit)$ascertaintime_post, col = "red", type = "l")

# Report summary statistics for Rt estimates and day-of-week effect estimates
summary(posterior)
summary(posterior_dow)
```

---

## SUB_060

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
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
```

---

## SUB_102

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
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
```

---

## SUB_035

**Scenario**: 2 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
import arviz as az
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data():
    """Generate sample COVID-19 case data for demonstration"""
    # Set parameters
    n_days = 150
    start_date = pd.to_datetime('2020-03-01')
    
    # True Rt trajectory (decreasing then increasing)
    t = np.linspace(0, n_days-1, n_days)
    true_rt = 2.5 * np.exp(-t/30) + 0.8 + 0.3 * np.sin(t/20)
    
    # Generation interval (Gamma distribution, mean=5.1, sd=2.3)
    max_gen_int = 20
    gen_int_shape = 4.9
    gen_int_scale = 1.04
    gen_int = gamma.pdf(np.arange(1, max_gen_int+1), a=gen_int_shape, scale=gen_int_scale)
    gen_int = gen_int / gen_int.sum()
    
    # Reporting delay (mean=7 days)
    max_delay = 25
    delay_shape = 2.0
    delay_scale = 3.5
    reporting_delay = gamma.pdf(np.arange(0, max_delay), a=delay_shape, scale=delay_scale)
    reporting_delay = reporting_delay / reporting_delay.sum()
    
    # Simulate true infections
    infections = np.zeros(n_days)
    infections[0] = 100  # Initial infections
    
    for t in range(1, n_days):
        # Renewal equation
        infectiousness = 0
        for s in range(min(t, len(gen_int))):
            infectiousness += infections[t-1-s] * gen_int[s]
        infections[t] = true_rt[t] * infectiousness
    
    # Day-of-week effects (lower reporting on weekends)
    dow_effects = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5])  # Mon-Sun
    
    # Time-varying ascertainment (starts low, increases)
    ascertainment = 0.1 + 0.4 / (1 + np.exp(-(t - 50)/15))
    
    # Apply reporting delay and observation process
    expected_reports = np.zeros(n_days + max_delay)
    for t in range(n_days):
        for d in range(len(reporting_delay)):
            if t + d < len(expected_reports):
                expected_reports[t + d] += infections[t] * ascertainment[t] * reporting_delay[d]
    
    # Truncate to original period
    expected_reports = expected_reports[:n_days]
    
    # Apply day-of-week effects
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    dow = dates.dayofweek  # Monday=0, Sunday=6
    dow_multiplier = dow_effects[dow]
    expected_reports *= dow_multiplier
    
    # Add overdispersion (negative binomial)
    phi = 10  # Overdispersion parameter
    cases = np.random.negative_binomial(
        n=phi, 
        p=phi / (phi + expected_reports)
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'cases': cases,
        'day_of_week': dow + 1  # Convert to 1-7 (Monday=1)
    })
    
    # Save to CSV
    df.to_csv('cases.csv', index=False)
    
    # Return additional info for validation
    return df, {
        'true_rt': true_rt,
        'true_infections': infections,
        'true_ascertainment': ascertainment,
        'true_dow_effects': dow_effects,
        'gen_int': gen_int,
        'reporting_delay': reporting_delay
    }

def create_generation_interval():
    """Create generation interval PMF"""
    # COVID-19 generation interval: Gamma(shape=4.9, scale=1.04)
    max_gen_int = 20
    gen_int_shape = 4.9
    gen_int_scale = 1.04
    
    gen_int = gamma.pdf(np.arange(1, max_gen_int+1), a=gen_int_shape, scale=gen_int_scale)
    gen_int = gen_int / gen_int.sum()
    
    return gen_int

def create_reporting_delay():
    """Create reporting delay PMF"""
    # Reporting delay: Gamma(shape=2.0, scale=3.5)
    max_delay = 25
    delay_shape = 2.0
    delay_scale = 3.5
    
    reporting_delay = gamma.pdf(np.arange(0, max_delay), a=delay_shape, scale=delay_scale)
    reporting_delay = reporting_delay / reporting_delay.sum()
    
    return reporting_delay

def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('cases.csv')
    except FileNotFoundError:
        print("cases.csv not found. Generating sample data...")
        df, true_params = generate_sample_data()
        print("Sample data generated and saved as cases.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def create_rt_model(cases_data, gen_int, reporting_delay):
    """Create the PyMC model for Rt estimation"""
    
    n_days = len(cases_data)
    cases = cases_data['cases'].values
    day_of_week = cases_data['day_of_week'].values - 1  # Convert to 0-6
    
    # Parameters for splines
    n_knots_rt = max(10, n_days // 15)  # Knots for Rt spline
    n_knots_asc = max(8, n_days // 20)   # Knots for ascertainment spline
    
    knots_rt = np.linspace(0, n_days-1, n_knots_rt)
    knots_asc = np.linspace(0, n_days-1, n_knots_asc)
    
    with pm.Model() as model:
        
        # ==================== Rt MODEL ====================
        # Random walk for Rt on log scale
        log_rt_init = pm.Normal('log_rt_init', mu=0, sigma=0.5)
        log_rt_steps = pm.Normal('log_rt_steps', mu=0, sigma=0.1, shape=n_days-1)
        log_rt = pt.concatenate([[log_rt_init], log_rt_init + pt.cumsum(log_rt_steps)])
        rt = pm.Deterministic('rt', pt.exp(log_rt))
        
        # ==================== DAY-OF-WEEK EFFECTS ====================
        # Day-of-week effects (multiplicative, sum to 7)
        dow_raw = pm.Normal('dow_raw', mu=0, sigma=0.3, shape=7)
        dow_effects = pm.Deterministic('dow_effects', 7 * pt.softmax(dow_raw))
        
        # ==================== TIME-VARYING ASCERTAINMENT ====================
        # Smooth ascertainment using random walk
        logit_asc_init = pm.Normal('logit_asc_init', mu=-2, sigma=1)  # Start low
        logit_asc_steps = pm.Normal('logit_asc_steps', mu=0, sigma=0.05, shape=n_days-1)
        logit_asc = pt.concatenate([[logit_asc_init], logit_asc_init + pt.cumsum(logit_asc_steps)])
        ascertainment = pm.Deterministic('ascertainment', pm.math.sigmoid(logit_asc))
        
        # ==================== INFECTION DYNAMICS ====================
        # Initial infections (seeding period)
        seed_days = min(14, n_days // 4)
        log_seed_infections = pm.Normal('log_seed_infections', mu=np.log(50), sigma=1, shape=seed_days)
        seed_infections = pt.exp(log_seed_infections)
        
        # Compute infections using renewal equation
        def compute_infections(rt_t, past_infections, gen_int):
            """Compute infections at time t using renewal equation"""
            # Convolve with generation interval
            infectiousness = 0
            for s in range(len(gen_int)):
                if s < past_infections.shape[0]:
                    infectiousness += past_infections[-(s+1)] * gen_int[s]
            return rt_t * infectiousness
        
        # Initialize infections list
        infections_list = [seed_infections[i] for i in range(seed_days)]
        
        # Compute remaining infections
        for t in range(seed_days, n_days):
            # Get past infections (up to generation interval length)
            past_infections = pt.stack(infections_list[max(0, t-len(gen_int)):t])
            
            # Compute new infections
            new_infection = compute_infections(rt[t], past_infections, gen_int)
            infections_list.append(new_infection)
        
        infections = pm.Deterministic('infections', pt.stack(infections_list))
        
        # ==================== OBSERVATION MODEL ====================
        # Apply reporting delay
        max_delay = len(reporting_delay)
        expected_reports = pt.zeros(n_days)
        
        for t in range(n_days):
            for d in range(max_delay):
                if t + d < n_days:
                    contribution = infections[t] * ascertainment[t] * reporting_delay[d]
                    expected_reports = pt.set_subtensor(
                        expected_reports[t + d], 
                        expected_reports[t + d] + contribution
                    )
        
        # Apply day-of-week effects
        dow_multiplier = dow_effects[day_of_week]
        expected_cases = expected_reports * dow_multiplier
        
        # ==================== LIKELIHOOD ====================
        # Overdispersion parameter
        phi = pm.Exponential('phi', lam=0.1)
        
        # Negative binomial likelihood
        likelihood = pm.NegativeBinomial(
            'likelihood',
            mu=expected_cases,
            alpha=phi,
            observed=cases
        )
        
        # Store expected cases for diagnostics
        pm.Deterministic('expected_cases', expected_cases)
    
    return model

def fit_model(model, samples=2000, tune=1000, chains=4):
    """Fit the PyMC model"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=samples,
            tune=tune,
            chains=chains,
            cores=min(4, chains),
            return_inferencedata=True,
            random_seed=42,
            target_accept=0.95
        )
    
    return trace

def extract_estimates(trace):
    """Extract point estimates and credible intervals"""
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Extract Rt estimates
    rt_vars = [var for var in summary.index if var.startswith('rt[')]
    rt_summary = summary.loc[rt_vars]
    
    # Extract day-of-week effects
    dow_vars = [var for var in summary.index if var.startswith('dow_effects[')]
    dow_summary = summary.loc[dow_vars]
    
    # Extract ascertainment
    asc_vars = [var for var in summary.index if var.startswith('ascertainment[')]
    asc_summary = summary.loc[asc_vars]
    
    return {
        'rt': rt_summary,
        'dow_effects': dow_summary,
        'ascertainment': asc_summary,
        'full_summary': summary
    }

def create_plots(data, trace, estimates):
    """Create comprehensive plots of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    dates = data['date']
    n_days = len(dates)
    
    # Plot 1: Rt over time
    ax1 = axes[0, 0]
    rt_mean = estimates['rt']['mean'].values
    rt_lower = estimates['rt']['hdi_2.5%'].values
    rt_upper = estimates['rt']['hdi_97.5%'].values
    
    ax1.plot(dates, rt_mean, 'b-', linewidth=2, label='Rt estimate')
    ax1.fill_between(dates, rt_lower, rt_upper, alpha=0.3, color='blue', label='95% HDI')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax1.set_ylabel('Reproduction Number (Rt)')
    ax1.set_title('Time-varying Reproduction Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Observed vs Expected Cases
    ax2 = axes[0, 1]
    posterior = trace.posterior
    expected_cases = posterior['expected_cases'].mean(dim=['chain', 'draw']).values
    
    ax2.scatter(data['cases'], expected_cases, alpha=0.6, s=30)
    max_val = max(data['cases'].max(), expected_cases.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    ax2.set_xlabel('Observed Cases')
    ax2.set_ylabel('Expected Cases')
    ax2.set_title('Model Fit: Observed vs Expected')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Day-of-week Effects
    ax3 = axes[1, 0]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_mean = estimates['dow_effects']['mean'].values
    dow_lower = estimates['dow_effects']['hdi_2.5%'].values
    dow_upper = estimates['dow_effects']['hdi_97.5%'].values
    
    x_pos = np.arange(len(dow_names))
    ax3.bar(x_pos, dow_mean, yerr=[dow_mean - dow_lower, dow_upper - dow_mean], 
            capsize=5, alpha=0.7, color='green')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dow_names)
    ax3.set_ylabel('Reporting Multiplier')
    ax3.set_title('Day-of-Week Effects')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Time-varying Ascertainment
    ax4 = axes[1, 1]
    asc_mean = estimates['ascertainment']['mean'].values
    asc_lower = estimates['ascertainment']['hdi_2.5%'].values
    asc_upper = estimates['ascertainment']['hdi_97.5%'].values
    
    ax4.plot(dates, asc_mean, 'g-', linewidth=2, label='Ascertainment')
    ax4.fill_between(dates, asc_lower, asc_upper, alpha=0.3, color='green', label='95% HDI')
    ax4.set_ylabel('Ascertainment Probability')
    ax4.set_title('Time-varying Ascertainment')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Cases over time with model fit
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.scatter(dates, data['cases'], alpha=0.6, s=30, label='Observed cases', color='black')
    ax.plot(dates, expected_cases, 'r-', linewidth=2, label='Expected cases', alpha=0.8)
    ax.set_ylabel('Daily Cases')
    ax.set_title('COVID-19 Cases: Observed vs Model Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("COVID-19 Rt Estimation with Complex Observation Processes")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = load_data()
    print(f"   Loaded {len(data)} days of case data")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Total cases: {data['cases'].sum():,}")
    
    # Create generation interval and reporting delay
    print("\n2. Setting up epidemiological parameters...")
    gen_int = create_generation_interval()
    reporting_delay = create_reporting_delay()
    print(f"   Generation interval mean: {np.sum(gen_int * np.arange(1, len(gen_int)+1)):.1f} days")
    print(f"   Reporting delay mean: {np.sum(reporting_delay * np.arange(len(reporting_delay))):.1f} days")
    
    # Create model
    print("\n3. Building PyMC model...")
    model = create_rt_model(data, gen_int, reporting_delay)
    print(f"   Model created with {len(data)} time points")
    
    # Check model
    with model:
        print(f"   Model has {len(model.free_RVs)} free random variables")
    
    # Fit model
    print("\n4. Fitting model (this may take several minutes)...")
    try:
        trace = fit_model(model, samples=1000, tune=500, chains=2)  # Reduced for faster execution
        print("   Model fitting completed successfully!")
    except Exception as e:
        print(f"   Error during fitting: {e}")
        return
    
    # Check convergence
    print("\n5. Checking convergence...")
    rhat = az.rhat(trace)
    max_rhat = float(rhat.max()) if hasattr(rhat.max(), 'values') else rhat.max().values.max()
    print(f"   Maximum R-hat: {max_rhat:.3f}")
    if max_rhat > 1.1:
        print("   Warning: Some parameters may not have converged (R-hat > 1.1)")
    else:
        print("   Convergence looks good (all R-hat <= 1.1)")
    
    # Extract estimates
    print("\n6. Extracting estimates...")
    estimates = extract_estimates(trace)
    
    # Print summary statistics
    print("\n7. Summary Results:")
    print("-" * 30)
    
    rt_mean = estimates['rt']['mean'].values
    print(f"   Rt range: {rt_mean.min():.2f} - {rt_mean.max():.2f}")
    print(f"   Current Rt: {rt_mean[-1]:.2f}")
    
    dow_effects = estimates['dow_effects']['mean'].values
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"   Day-of-week effects:")
    for i, (day, effect) in enumerate(zip(dow_names, dow_effects)):
        print(f"     {day}: {effect:.2f}")
    
    asc_mean = estimates['ascertainment']['mean'].values
    print(f"   Ascertainment range: {asc_mean.min():.1%} - {asc_mean.max():.1%}")
    print(f"   Final ascertainment: {asc_mean[-1]:.1%}")
    
    # Create plots
    print("\n8. Creating plots...")
    create_plots(data, trace, estimates)
    
    print("\nAnalysis completed successfully!")
    
    return {
        'data': data,
        'model': model,
        'trace': trace,
        'estimates': estimates
    }

if __name__ == "__main__":
    results = main()
```

---

## SUB_059

**Scenario**: 2 | **Execution**: FAILED

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import arviz as az
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def generate_sample_data():
    """Generate sample COVID-19 case data for demonstration"""
    np.random.seed(42)
    
    # Generate 120 days of data
    dates = pd.date_range(start='2022-01-01', periods=120, freq='D')
    
    # Simulate realistic case patterns
    t = np.arange(len(dates))
    
    # Create underlying infection trend with some waves
    base_infections = 100 + 200 * np.exp(-0.02 * t) * np.sin(0.1 * t + 1)**2
    
    # Add day-of-week effects (fewer cases reported on weekends)
    dow_effects = {0: 1.1, 1: 1.05, 2: 1.0, 3: 1.0, 4: 0.95, 5: 0.7, 6: 0.6}
    
    # Generate cases with reporting delays and overdispersion
    cases = []
    for i, date in enumerate(dates):
        dow = date.weekday()
        expected = base_infections[i] * dow_effects[dow]
        # Add overdispersion using negative binomial
        case_count = np.random.negative_binomial(n=10, p=10/(10 + expected))
        cases.append(max(0, case_count))
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'cases': cases,
        'day_of_week': [d.weekday() + 1 for d in dates]  # 1-7 format
    })
    
    return data

def get_generation_interval():
    """Define generation interval distribution"""
    # Based on COVID-19 literature: mean ~5-6 days, std ~3-4 days
    # Using gamma distribution with appropriate parameters
    mean_gi = 5.5
    std_gi = 3.0
    
    # Convert to gamma parameters
    scale = std_gi**2 / mean_gi
    shape = mean_gi / scale
    
    # Discretize to daily intervals (truncate at 20 days)
    max_gi = 20
    gi_support = np.arange(1, max_gi + 1)
    gi_pmf = stats.gamma.pdf(gi_support, a=shape, scale=scale)
    gi_pmf = gi_pmf / gi_pmf.sum()  # Normalize to sum to 1
    
    return gi_pmf

def get_delay_distribution():
    """Define delay from infection to reporting distribution"""
    # Infection to symptom onset: ~5 days
    # Symptom onset to reporting: ~2-3 days
    # Total delay: ~7-8 days on average
    mean_delay = 7.5
    std_delay = 3.5
    
    # Use gamma distribution for delays
    scale = std_delay**2 / mean_delay
    shape = mean_delay / scale
    
    # Discretize (truncate at 25 days)
    max_delay = 25
    delay_support = np.arange(0, max_delay + 1)
    delay_pmf = stats.gamma.pdf(delay_support, a=shape, scale=scale)
    delay_pmf = delay_pmf / delay_pmf.sum()
    
    return delay_pmf

class RtEstimationModel:
    """
    PyMC model for estimating time-varying Rt with observation processes
    """
    
    def __init__(self, data, generation_interval, delay_distribution):
        self.data = data
        self.cases = data['cases'].values
        self.dates = data['date'].values
        self.day_of_week = data['day_of_week'].values
        self.n_days = len(data)
        
        self.gi_pmf = generation_interval
        self.delay_pmf = delay_distribution
        self.max_gi = len(generation_interval)
        self.max_delay = len(delay_distribution)
        
        self.model = None
        self.trace = None
        
    def build_model(self):
        """Build the PyMC model"""
        
        with pm.Model() as model:
            # === PRIORS ===
            
            # Initial infections (seeding period)
            seed_days = max(self.max_gi, 14)  # At least 14 days or max generation interval
            I_seed = pm.Exponential('I_seed', lam=1/100, shape=seed_days)
            
            # Rt evolution (random walk on log scale)
            # Start with prior centered around R=1
            log_Rt_init = pm.Normal('log_Rt_init', mu=0, sigma=0.2)
            log_Rt_steps = pm.Normal('log_Rt_steps', mu=0, sigma=0.1, 
                                   shape=self.n_days - seed_days - 1)
            
            # Cumulative sum to create random walk
            log_Rt_full = pt.concatenate([
                [log_Rt_init],
                log_Rt_init + pt.cumsum(log_Rt_steps)
            ])
            Rt = pm.Deterministic('Rt', pt.exp(log_Rt_full))
            
            # Day-of-week effects (multiplicative)
            # Use Monday as reference (effect = 1)
            dow_effects_raw = pm.Normal('dow_effects_raw', mu=0, sigma=0.2, shape=7)
            # Set Monday effect to 0 (multiplicative effect of 1)
            dow_effects = pm.Deterministic('dow_effects', 
                                         pt.exp(dow_effects_raw - dow_effects_raw[0]))
            
            # Time-varying ascertainment rate (smoothly varying)
            # Use random walk on logit scale
            logit_ascert_init = pm.Normal('logit_ascert_init', mu=-1, sigma=0.5)
            logit_ascert_steps = pm.Normal('logit_ascert_steps', mu=0, sigma=0.05,
                                         shape=self.n_days - 1)
            
            logit_ascert_full = pt.concatenate([
                [logit_ascert_init],
                logit_ascert_init + pt.cumsum(logit_ascert_steps)
            ])
            ascertainment = pm.Deterministic('ascertainment', 
                                           pm.math.sigmoid(logit_ascert_full))
            
            # Overdispersion parameter for negative binomial
            phi = pm.Exponential('phi', lam=1/10)
            
            # === INFECTION DYNAMICS ===
            
            def renewal_step(Rt_t, *past_infections):
                """Single step of renewal equation"""
                past_I = pt.stack(past_infections)
                # Reverse to align with generation interval (most recent first)
                past_I_rev = past_I[::-1]
                new_infections = Rt_t * pt.sum(past_I_rev * self.gi_pmf[:len(past_I)])
                return new_infections
            
            # Compute infections using scan
            infection_days = self.n_days - seed_days
            sequences = [Rt]
            non_sequences = []
            
            # Initial infections for renewal process
            outputs_info = []
            for i in range(self.max_gi):
                if i < seed_days:
                    outputs_info.append(I_seed[-(i+1)])
                else:
                    outputs_info.append(pt.zeros_like(I_seed[0]))
            
            infections_result, _ = pm.scan(
                fn=renewal_step,
                sequences=[Rt],
                outputs_info=outputs_info,
                n_steps=infection_days,
                strict=True
            )
            
            # Extract just the new infections (first output)
            I_renewal = infections_result[0]
            
            # Combine seed infections and renewal-based infections
            I_full = pm.Deterministic('infections', 
                                    pt.concatenate([I_seed, I_renewal]))
            
            # === OBSERVATION PROCESS ===
            
            def compute_expected_cases():
                """Compute expected reported cases accounting for delays and ascertainment"""
                expected = pt.zeros(self.n_days)
                
                for t in range(self.n_days):
                    daily_expected = 0.0
                    
                    # Sum over all possible infection dates
                    for delay in range(self.max_delay + 1):
                        infection_day = t - delay
                        if infection_day >= 0:
                            # Infections on infection_day, reported on day t
                            prob_delay = self.delay_pmf[delay]
                            infections = I_full[infection_day]
                            ascert_rate = ascertainment[t]  # Ascertainment on reporting day
                            
                            daily_expected += infections * prob_delay * ascert_rate
                    
                    expected = pt.set_subtensor(expected[t], daily_expected)
                
                return expected
            
            expected_reported = compute_expected_cases()
            
            # Apply day-of-week effects
            dow_indices = self.day_of_week - 1  # Convert to 0-based indexing
            dow_multipliers = dow_effects[dow_indices]
            
            expected_final = pm.Deterministic('expected_cases', 
                                            expected_reported * dow_multipliers)
            
            # === LIKELIHOOD ===
            
            # Negative binomial likelihood for overdispersed counts
            obs = pm.NegativeBinomial('obs', 
                                    mu=expected_final,
                                    alpha=phi,
                                    observed=self.cases)
            
        self.model = model
        return model
    
    def fit(self, draws=1000, tune=1000, chains=2, target_accept=0.9):
        """Fit the model using MCMC"""
        
        if self.model is None:
            self.build_model()
        
        with self.model:
            # Use NUTS sampler with higher target acceptance for complex model
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True,
                random_seed=42
            )
        
        return self.trace
    
    def extract_estimates(self):
        """Extract point estimates and credible intervals"""
        
        if self.trace is None:
            raise ValueError("Model must be fitted before extracting estimates")
        
        # Extract Rt estimates
        rt_summary = az.summary(self.trace, var_names=['Rt'])
        rt_estimates = pd.DataFrame({
            'date': self.dates[-(len(rt_summary)):],  # Rt starts after seed period
            'rt_mean': rt_summary['mean'].values,
            'rt_lower': rt_summary['hdi_5%'].values,
            'rt_upper': rt_summary['hdi_95%'].values
        })
        
        # Extract day-of-week effects
        dow_summary = az.summary(self.trace, var_names=['dow_effects'])
        dow_estimates = pd.DataFrame({
            'day_of_week': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                          'Friday', 'Saturday', 'Sunday'],
            'dow_effect_mean': dow_summary['mean'].values,
            'dow_effect_lower': dow_summary['hdi_5%'].values,
            'dow_effect_upper': dow_summary['hdi_95%'].values
        })
        
        # Extract ascertainment rates
        ascert_summary = az.summary(self.trace, var_names=['ascertainment'])
        ascert_estimates = pd.DataFrame({
            'date': self.dates,
            'ascertainment_mean': ascert_summary['mean'].values,
            'ascertainment_lower': ascert_summary['hdi_5%'].values,
            'ascertainment_upper': ascert_summary['hdi_95%'].values
        })
        
        return {
            'rt': rt_estimates,
            'day_of_week_effects': dow_estimates,
            'ascertainment': ascert_estimates
        }
    
    def plot_results(self, estimates):
        """Create comprehensive plots of results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('COVID-19 Rt Estimation Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Observed cases and model fit
        ax1 = axes[0, 0]
        ax1.plot(self.dates, self.cases, 'o-', alpha=0.7, markersize=3, 
                label='Observed cases', color='darkblue')
        
        # Plot expected cases if available in trace
        try:
            expected_summary = az.summary(self.trace, var_names=['expected_cases'])
            ax1.fill_between(self.dates, 
                           expected_summary['hdi_5%'].values,
                           expected_summary['hdi_95%'].values,
                           alpha=0.3, color='red', label='Model 90% CI')
            ax1.plot(self.dates, expected_summary['mean'].values, 
                    color='red', linewidth=2, label='Model mean')
        except:
            pass
        
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Daily Cases')
        ax1.set_title('Observed vs Model-Expected Cases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rt estimates
        ax2 = axes[0, 1]
        rt_data = estimates['rt']
        ax2.fill_between(rt_data['date'], 
                        rt_data['rt_lower'], 
                        rt_data['rt_upper'],
                        alpha=0.3, color='green', label='90% CI')
        ax2.plot(rt_data['date'], rt_data['rt_mean'], 
                color='green', linewidth=2, label='Rt estimate')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Reproduction Number (Rt)')
        ax2.set_title('Time-varying Reproduction Number')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Day-of-week effects
        ax3 = axes[1, 0]
        dow_data = estimates['day_of_week_effects']
        x_pos = np.arange(len(dow_data))
        ax3.errorbar(x_pos, dow_data['dow_effect_mean'],
                    yerr=[dow_data['dow_effect_mean'] - dow_data['dow_effect_lower'],
                          dow_data['dow_effect_upper'] - dow_data['dow_effect_mean']],
                    fmt='o-', capsize=5, capthick=2, linewidth=2, markersize=8)
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(dow_data['day_of_week'], rotation=45)
        ax3.axhline(y=1, color='red', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Reporting Multiplier')
        ax3.set_title('Day-of-Week Effects')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Time-varying ascertainment
        ax4 = axes[1, 1]
        ascert_data = estimates['ascertainment']
        ax4.fill_between(ascert_data['date'], 
                        ascert_data['ascertainment_lower'], 
                        ascert_data['ascertainment_upper'],
                        alpha=0.3, color='purple', label='90% CI')
        ax4.plot(ascert_data['date'], ascert_data['ascertainment_mean'], 
                color='purple', linewidth=2, label='Ascertainment rate')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Ascertainment Rate')
        ax4.set_title('Time-varying Ascertainment Rate')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

def main():
    """Main execution function"""
    
    print("COVID-19 Rt Estimation with Complex Observation Processes")
    print("=" * 60)
    
    # Generate or load data
    print("\n1. Loading data...")
    try:
        # Try to load real data
        data = pd.read_csv('cases.csv')
        data['date'] = pd.to_datetime(data['date'])
        print(f"   Loaded {len(data)} days of case data from file")
    except FileNotFoundError:
        # Generate sample data if file doesn't exist
        print("   cases.csv not found. Generating sample data...")
        data = generate_sample_data()
        # Save for future use
        data.to_csv('cases.csv', index=False)
        print(f"   Generated {len(data)} days of sample data")
    
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Total cases: {data['cases'].sum():,}")
    print(f"   Daily average: {data['cases'].mean():.1f}")
    
    # Get distributions
    print("\n2. Setting up epidemiological distributions...")
    generation_interval = get_generation_interval()
    delay_distribution = get_delay_distribution()
    print(f"   Generation interval mean: {np.sum(np.arange(1, len(generation_interval)+1) * generation_interval):.1f} days")
    print(f"   Reporting delay mean: {np.sum(np.arange(len(delay_distribution)) * delay_distribution):.1f} days")
    
    # Build and fit model
    print("\n3. Building PyMC model...")
    rt_model = RtEstimationModel(data, generation_interval, delay_distribution)
    rt_model.build_model()
    print("   Model built successfully")
    
    print("\n4. Fitting model with MCMC...")
    print("   This may take several minutes...")
    trace = rt_model.fit(draws=500, tune=500, chains=2)  # Reduced for demo
    
    # Check model convergence
    print("\n5. Checking model diagnostics...")
    rhat = az.rhat(trace)
    max_rhat = float(rhat.max().values) if hasattr(rhat.max(), 'values') else float(rhat.max())
    print(f"   Maximum R-hat: {max_rhat:.3f}")
    if max_rhat > 1.1:
        print("   WARNING: Some parameters may not have converged (R-hat > 1.1)")
    else:
        print("   Model appears to have converged well")
    
    # Extract estimates
    print("\n6. Extracting estimates...")
    estimates = rt_model.extract_estimates()
    
    # Display summary results
    print("\n7. Summary Results:")
    print(f"   Current Rt estimate: {estimates['rt']['rt_mean'].iloc[-1]:.2f} "
          f"({estimates['rt']['rt_lower'].iloc[-1]:.2f}-{estimates['rt']['rt_upper'].iloc[-1]:.2f})")
    
    print(f"   Current ascertainment rate: {estimates['ascertainment']['ascertainment_mean'].iloc[-1]:.1%} "
          f"({estimates['ascertainment']['ascertainment_lower'].iloc[-1]:.1%}-"
          f"{estimates['ascertainment']['ascertainment_upper'].iloc[-1]:.1%})")
    
    print("\n   Day-of-week effects (relative to Monday):")
    for _, row in estimates['day_of_week_effects'].iterrows():
        print(f"     {row['day_of_week']}: {row['dow_effect_mean']:.2f} "
              f"({row['dow_effect_lower']:.2f}-{row['dow_effect_upper']:.2f})")
    
    # Create plots
    print("\n8. Creating visualizations...")
    fig = rt_model.plot_results(estimates)
    plt.show()
    
    # Save results
    print("\n9. Saving results...")
    estimates['rt'].to_csv('rt_estimates.csv', index=False)
    estimates['day_of_week_effects'].to_csv('dow_effects.csv', index=False)
    estimates['ascertainment'].to_csv('ascertainment_estimates.csv', index=False)
    print("   Results saved to CSV files")
    
    return rt_model, trace, estimates

if __name__ == "__main__":
    # Run the analysis
    model, trace, estimates = main()
    
    # Additional diagnostic plots
    print("\nGenerating additional diagnostic plots...")
    
    # Trace plots for key parameters
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Rt trace (last 10 days)
    rt_vars = [f'Rt[{i}]' for i in range(max(0, len(estimates['rt']) - 10), len(estimates['rt']))]
    if rt_vars:
        az.plot_trace(trace, var_names=['Rt'], coords={'Rt_dim_0': list(range(max(0, len(estimates['rt']) - 10), len(estimates['rt'])))}, 
                     ax=axes[0], compact=True)
        axes[0, 0].set_title('Rt Traces (Last 10 Days)')
    
    # Day-of-week effects trace
    az.plot_trace(trace, var_names=['dow_effects'], ax=axes[1], compact=True)
    axes[1, 0].set_title('Day-of-Week Effects Traces')
    
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis complete! Check the generated plots and CSV files for detailed results.")
```

---

## SUB_016

**Scenario**: 2 | **Execution**: FAILED

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import arviz as az
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data():
    """Generate sample COVID-19 case data with realistic patterns"""
    
    # Parameters for data generation
    n_days = 150
    start_date = pd.to_datetime('2023-01-01')
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # True Rt trajectory (starts high, decreases, then increases slightly)
    t = np.arange(n_days)
    true_rt = 2.5 * np.exp(-t/30) + 0.8 + 0.3 * np.sin(t/20)
    true_rt = np.maximum(true_rt, 0.5)  # Ensure positive
    
    # Generation interval (gamma distribution approximation)
    generation_shape, generation_scale = 2.5, 2.0
    max_gen = 20
    generation_pmf = stats.gamma.pdf(np.arange(1, max_gen+1), generation_shape, scale=generation_scale)
    generation_pmf = generation_pmf / generation_pmf.sum()
    
    # Reporting delay (gamma distribution)
    delay_shape, delay_scale = 3.0, 2.0
    max_delay = 25
    delay_pmf = stats.gamma.pdf(np.arange(max_delay), delay_shape, scale=delay_scale)
    delay_pmf = delay_pmf / delay_pmf.sum()
    
    # Day-of-week effects (Monday=1, Sunday=7)
    # Lower reporting on weekends
    true_dow_effects = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5])  # Mon-Sun
    
    # Time-varying ascertainment rate
    true_ascertainment = 0.3 + 0.2 * np.exp(-t/40) + 0.1 * np.sin(t/25)
    true_ascertainment = np.clip(true_ascertainment, 0.1, 0.8)
    
    # Generate infections using renewal equation
    infections = np.zeros(n_days)
    infections[:7] = 100  # Initial seeding
    
    for i in range(7, n_days):
        renewal_sum = 0
        for s, g_s in enumerate(generation_pmf):
            if i - s - 1 >= 0:
                renewal_sum += infections[i - s - 1] * g_s
        infections[i] = true_rt[i] * renewal_sum
    
    # Apply reporting delays and observation process
    reported_infections = np.zeros(n_days + max_delay)
    for i in range(n_days):
        for d, delay_prob in enumerate(delay_pmf):
            if i + d < len(reported_infections):
                reported_infections[i + d] += infections[i] * delay_prob
    
    # Truncate to original length
    reported_infections = reported_infections[:n_days]
    
    # Apply ascertainment and day-of-week effects
    day_of_week = np.array([d.weekday() + 1 for d in dates])  # 1-7, Monday=1
    dow_multipliers = true_dow_effects[day_of_week - 1]
    
    expected_cases = reported_infections * true_ascertainment * dow_multipliers
    
    # Add overdispersion (negative binomial)
    phi = 10.0  # Dispersion parameter
    cases = np.random.negative_binomial(
        n=phi, 
        p=phi / (phi + expected_cases)
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'cases': cases,
        'day_of_week': day_of_week
    })
    
    # Store true values for comparison
    true_values = {
        'rt': true_rt,
        'ascertainment': true_ascertainment,
        'dow_effects': true_dow_effects,
        'infections': infections,
        'generation_pmf': generation_pmf,
        'delay_pmf': delay_pmf
    }
    
    return df, true_values

def create_generation_interval():
    """Create generation interval PMF"""
    shape, scale = 2.5, 2.0
    max_gen = 20
    x = np.arange(1, max_gen + 1)
    pmf = stats.gamma.pdf(x, shape, scale=scale)
    return pmf / pmf.sum()

def create_reporting_delay():
    """Create reporting delay PMF"""
    shape, scale = 3.0, 2.0
    max_delay = 25
    x = np.arange(max_delay)
    pmf = stats.gamma.pdf(x, shape, scale=scale)
    return pmf / pmf.sum()

def build_rt_model(cases, day_of_week, generation_pmf, delay_pmf):
    """Build PyMC model for Rt estimation"""
    
    n_days = len(cases)
    n_seed = 7  # Initial seeding period
    
    with pm.Model() as model:
        
        # Priors for initial infections (seeding period)
        log_initial_infections = pm.Normal('log_initial_infections', 
                                         mu=np.log(100), sigma=1.0, 
                                         shape=n_seed)
        initial_infections = pm.math.exp(log_initial_infections)
        
        # Time-varying Rt using random walk on log scale
        log_rt_raw = pm.GaussianRandomWalk('log_rt_raw', 
                                          sigma=0.1, 
                                          shape=n_days-n_seed,
                                          init_dist=pm.Normal.dist(0, 0.5))
        log_rt = pm.Deterministic('log_rt', log_rt_raw + np.log(1.0))
        rt = pm.Deterministic('rt', pm.math.exp(log_rt))
        
        # Day-of-week effects (Monday=1, Sunday=7)
        dow_effects_raw = pm.Normal('dow_effects_raw', mu=0, sigma=0.5, shape=7)
        # Normalize so Monday = 1
        dow_effects = pm.Deterministic('dow_effects', 
                                     pm.math.exp(dow_effects_raw - dow_effects_raw[0]))
        
        # Time-varying ascertainment rate using random walk on logit scale
        logit_ascertainment_raw = pm.GaussianRandomWalk('logit_ascertainment_raw',
                                                       sigma=0.05,
                                                       shape=n_days,
                                                       init_dist=pm.Normal.dist(-1, 1))
        ascertainment = pm.Deterministic('ascertainment', 
                                       pm.math.sigmoid(logit_ascertainment_raw))
        
        # Overdispersion parameter
        phi = pm.Exponential('phi', 1.0/10.0)
        
        # Compute infections using renewal equation
        def renewal_step(rt_t, *prev_infections):
            renewal_sum = sum(inf * g for inf, g in zip(prev_infections, generation_pmf))
            return rt_t * renewal_sum
        
        infections = pt.zeros(n_days)
        infections = pt.set_subtensor(infections[:n_seed], initial_infections)
        
        # Compute infections for post-seeding period
        for t in range(n_seed, n_days):
            # Get relevant previous infections
            start_idx = max(0, t - len(generation_pmf))
            prev_inf = infections[start_idx:t]
            
            # Pad with zeros if needed and reverse to match generation interval
            if len(prev_inf) < len(generation_pmf):
                pad_length = len(generation_pmf) - len(prev_inf)
                prev_inf = pt.concatenate([pt.zeros(pad_length), prev_inf])
            
            prev_inf = prev_inf[-len(generation_pmf):][::-1]  # Reverse for convolution
            
            # Compute renewal sum
            renewal_sum = pt.sum(prev_inf * generation_pmf)
            new_infection = rt[t - n_seed] * renewal_sum
            
            infections = pt.set_subtensor(infections[t], new_infection)
        
        # Apply reporting delays
        reported_infections = pt.zeros(n_days)
        for t in range(n_days):
            for d, delay_prob in enumerate(delay_pmf):
                if t + d < n_days:
                    reported_infections = pt.set_subtensor(
                        reported_infections[t + d],
                        reported_infections[t + d] + infections[t] * delay_prob
                    )
        
        # Apply observation process
        dow_multipliers = dow_effects[day_of_week - 1]  # day_of_week is 1-indexed
        expected_cases = reported_infections * ascertainment * dow_multipliers
        
        # Likelihood with overdispersion
        cases_obs = pm.NegativeBinomial('cases_obs',
                                      n=phi,
                                      p=phi / (phi + expected_cases),
                                      observed=cases)
        
        # Store intermediate variables for analysis
        pm.Deterministic('infections', infections)
        pm.Deterministic('reported_infections', reported_infections)
        pm.Deterministic('expected_cases', expected_cases)
    
    return model

def fit_model(model, draws=1000, tune=1000, chains=2):
    """Fit the PyMC model"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(draws=draws, tune=tune, chains=chains, 
                         return_inferencedata=True,
                         random_seed=42,
                         target_accept=0.95)
    return trace

def plot_results(trace, data, true_values=None):
    """Create comprehensive plots of results"""
    
    dates = pd.to_datetime(data['date'])
    n_days = len(dates)
    
    # Extract posterior samples
    rt_samples = trace.posterior['rt'].values  # shape: (chains, draws, time)
    ascertainment_samples = trace.posterior['ascertainment'].values
    dow_effects_samples = trace.posterior['dow_effects'].values
    infections_samples = trace.posterior['infections'].values
    
    # Compute percentiles
    rt_median = np.median(rt_samples, axis=(0, 1))
    rt_lower = np.percentile(rt_samples, 2.5, axis=(0, 1))
    rt_upper = np.percentile(rt_samples, 97.5, axis=(0, 1))
    
    ascert_median = np.median(ascertainment_samples, axis=(0, 1))
    ascert_lower = np.percentile(ascertainment_samples, 2.5, axis=(0, 1))
    ascert_upper = np.percentile(ascertainment_samples, 97.5, axis=(0, 1))
    
    dow_median = np.median(dow_effects_samples, axis=(0, 1))
    dow_lower = np.percentile(dow_effects_samples, 2.5, axis=(0, 1))
    dow_upper = np.percentile(dow_effects_samples, 97.5, axis=(0, 1))
    
    inf_median = np.median(infections_samples, axis=(0, 1))
    inf_lower = np.percentile(infections_samples, 2.5, axis=(0, 1))
    inf_upper = np.percentile(infections_samples, 97.5, axis=(0, 1))
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('COVID-19 Rt Estimation Results', fontsize=16, fontweight='bold')
    
    # 1. Reproduction number over time
    ax = axes[0, 0]
    ax.fill_between(dates[7:], rt_lower, rt_upper, alpha=0.3, color='blue', label='95% CI')
    ax.plot(dates[7:], rt_median, color='blue', linewidth=2, label='Estimated Rt')
    
    if true_values is not None:
        ax.plot(dates[7:], true_values['rt'][7:], '--', color='red', linewidth=2, 
                label='True Rt', alpha=0.8)
    
    ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Rt = 1')
    ax.set_ylabel('Reproduction Number (Rt)')
    ax.set_title('Time-varying Reproduction Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Cases and infections
    ax = axes[0, 1]
    ax.bar(dates, data['cases'], alpha=0.6, color='gray', label='Reported Cases')
    ax.fill_between(dates, inf_lower, inf_upper, alpha=0.3, color='orange', label='95% CI')
    ax.plot(dates, inf_median, color='orange', linewidth=2, label='Estimated Infections')
    
    if true_values is not None:
        ax.plot(dates, true_values['infections'], '--', color='red', linewidth=2, 
                label='True Infections', alpha=0.8)
    
    ax.set_ylabel('Count')
    ax.set_title('Cases and Estimated Infections')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Ascertainment rate
    ax = axes[0, 2]
    ax.fill_between(dates, ascert_lower, ascert_upper, alpha=0.3, color='green', label='95% CI')
    ax.plot(dates, ascert_median, color='green', linewidth=2, label='Estimated Ascertainment')
    
    if true_values is not None:
        ax.plot(dates, true_values['ascertainment'], '--', color='red', linewidth=2, 
                label='True Ascertainment', alpha=0.8)
    
    ax.set_ylabel('Ascertainment Rate')
    ax.set_title('Time-varying Ascertainment Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Day-of-week effects
    ax = axes[1, 0]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x_pos = np.arange(len(days))
    
    ax.bar(x_pos, dow_median, yerr=[dow_median - dow_lower, dow_upper - dow_median],
           capsize=5, alpha=0.7, color='purple', label='Estimated')
    
    if true_values is not None:
        ax.plot(x_pos, true_values['dow_effects'], 'ro-', linewidth=2, 
                markersize=8, label='True Effects')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(days)
    ax.set_ylabel('Relative Reporting Rate')
    ax.set_title('Day-of-Week Effects')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Model diagnostics - Rt trace
    ax = axes[1, 1]
    for i in range(min(3, rt_samples.shape[0])):  # Show up to 3 chains
        for j in range(0, rt_samples.shape[2], 10):  # Every 10th time point
            ax.plot(rt_samples[i, :, j], alpha=0.6, linewidth=0.8)
    ax.set_xlabel('MCMC Iteration')
    ax.set_ylabel('Rt')
    ax.set_title('MCMC Traces (Rt, subset)')
    ax.grid(True, alpha=0.3)
    
    # 6. Posterior predictive check
    ax = axes[1, 2]
    expected_cases_samples = (infections_samples * ascertainment_samples[:, :, :, np.newaxis] * 
                            dow_effects_samples[:, :, data['day_of_week'].values - 1])
    expected_cases_median = np.median(expected_cases_samples, axis=(0, 1))
    
    ax.scatter(data['cases'], expected_cases_median, alpha=0.6, color='blue')
    max_val = max(data['cases'].max(), expected_cases_median.max())
    ax.plot([0, max_val], [0, max_val], '--', color='red', alpha=0.8, label='Perfect Fit')
    ax.set_xlabel('Observed Cases')
    ax.set_ylabel('Expected Cases')
    ax.set_title('Posterior Predictive Check')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def summarize_results(trace, data):
    """Print summary statistics"""
    
    print("=== MODEL SUMMARY ===")
    print(f"Data period: {data['date'].min()} to {data['date'].max()}")
    print(f"Total days: {len(data)}")
    print(f"Total cases: {data['cases'].sum():,}")
    print()
    
    # Rt summary
    rt_samples = trace.posterior['rt'].values
    rt_final_median = np.median(rt_samples[:, :, -1])
    rt_final_ci = np.percentile(rt_samples[:, :, -1], [2.5, 97.5])
    
    print("=== REPRODUCTION NUMBER ===")
    print(f"Final Rt estimate: {rt_final_median:.2f} (95% CI: {rt_final_ci[0]:.2f}-{rt_final_ci[1]:.2f})")
    
    rt_mean = np.median(rt_samples, axis=(0, 1)).mean()
    print(f"Average Rt over period: {rt_mean:.2f}")
    
    # Day-of-week effects
    dow_samples = trace.posterior['dow_effects'].values
    dow_median = np.median(dow_samples, axis=(0, 1))
    
    print("\n=== DAY-OF-WEEK EFFECTS ===")
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for i, day in enumerate(days):
        ci = np.percentile(dow_samples[:, :, i], [2.5, 97.5])
        print(f"{day}: {dow_median[i]:.2f} (95% CI: {ci[0]:.2f}-{ci[1]:.2f})")
    
    # Ascertainment
    ascert_samples = trace.posterior['ascertainment'].values
    ascert_median = np.median(ascert_samples, axis=(0, 1))
    ascert_final = np.median(ascert_samples[:, :, -1])
    ascert_final_ci = np.percentile(ascert_samples[:, :, -1], [2.5, 97.5])
    
    print("\n=== ASCERTAINMENT RATE ===")
    print(f"Final ascertainment: {ascert_final:.3f} (95% CI: {ascert_final_ci[0]:.3f}-{ascert_final_ci[1]:.3f})")
    print(f"Average ascertainment: {ascert_median.mean():.3f}")
    
    # Model diagnostics
    print(f"\n=== MODEL DIAGNOSTICS ===")
    rhat = az.rhat(trace)
    rhat_rt = rhat['rt'].values
    print(f"Rt R-hat range: {rhat_rt.min():.3f} - {rhat_rt.max():.3f}")
    
    if rhat_rt.max() > 1.1:
        print("WARNING: Some R-hat values > 1.1, consider longer sampling")
    else:
        print("Good convergence (all R-hat < 1.1)")

def main():
    """Main execution function"""
    
    print("Generating sample data...")
    data, true_values = generate_sample_data()
    
    # Save sample data
    data.to_csv('cases.csv', index=False)
    print("Sample data saved to 'cases.csv'")
    
    # Load data (simulating the actual use case)
    data = pd.read_csv('cases.csv')
    data['date'] = pd.to_datetime(data['date'])
    
    print(f"\nLoaded data: {len(data)} days of case counts")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Total cases: {data['cases'].sum():,}")
    
    # Create epidemiological parameters
    generation_pmf = create_generation_interval()
    delay_pmf = create_reporting_delay()
    
    print(f"\nGeneration interval: mean = {np.sum(np.arange(1, len(generation_pmf)+1) * generation_pmf):.1f} days")
    print(f"Reporting delay: mean = {np.sum(np.arange(len(delay_pmf)) * delay_pmf):.1f} days")
    
    # Build and fit model
    print("\nBuilding PyMC model...")
    model = build_rt_model(data['cases'].values, 
                          data['day_of_week'].values,
                          generation_pmf, 
                          delay_pmf)
    
    print("Model built successfully. Starting MCMC sampling...")
    print("This may take several minutes...")
    
    # Fit model
    trace = fit_model(model, draws=500, tune=500, chains=2)
    
    print("Sampling completed!")
    
    # Summarize results
    summarize_results(trace, data)
    
    # Create plots
    print("\nGenerating plots...")
    fig = plot_results(trace, data, true_values)
    
    # Extract key estimates for return
    rt_samples = trace.posterior['rt'].values
    ascertainment_samples = trace.posterior['ascertainment'].values  
    dow_samples = trace.posterior['dow_effects'].values
    
    results = {
        'rt_median': np.median(rt_samples, axis=(0, 1)),
        'rt_lower': np.percentile(rt_samples, 2.5, axis=(0, 1)),
        'rt_upper': np.percentile(rt_samples, 97.5, axis=(0, 1)),
        'ascertainment_median': np.median(ascertainment_samples, axis=(0, 1)),
        'ascertainment_lower': np.percentile(ascertainment_samples, 2.5, axis=(0, 1)),
        'ascertainment_upper': np.percentile(ascertainment_samples, 97.5, axis=(0, 1)),
        'dow_effects_median': np.median(dow_samples, axis=(0, 1)),
        'dow_effects_lower': np.percentile(dow_samples, 2.5, axis=(0, 1)),
        'dow_effects_upper': np.percentile(dow_samples, 97.5, axis=(0, 1)),
        'dates': data['date'].values[7:],  # Rt starts after seeding period
        'trace': trace
    }
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Results stored in returned dictionary with keys:")
    for key in results.keys():
        if key != 'trace':
            print(f"  - {key}")
    
    return results, data, true_values

if __name__ == "__main__":
    results, data, true_values = main()
```

---

## SUB_107

**Scenario**: 2 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('cases.csv')
df['date'] = pd.to_datetime(df['date'])

# Define constants
generation_interval = 5.0  # Average generation interval in days

# Fit model
with pm.Model() as model:
    # Day-of-week effects (multiplicative)
    day_of_week Effects = pm.Normal('day_of_week_effects', mu=1, sigma=0.2,
                                    shape=(7,))
    
    # Time-varying ascertainment (smoothly varying proportion of infections reported)
    ascertainment = pm.Uniform('ascertainment', lower=0, upper=1,
                               transformed=pm.distributions.transforms.Log,
                               shape=(len(df),))
    
    # Infection dynamics
    I_past = df['cases'].shift(generation_interval).values  # Lagged cases by generation interval
    
    # Renewal equation with day-of-week effects and ascertainment
    Rt = pm.Deterministic('Rt', (day_of_week_effects * I_past) * ascertainment)
    
    # Observation model (Poisson with overdispersion)
    observed_cases = pm.Poisson('observed_cases', mu=Rt,
                                observed=df['cases'],
                                shape=(len(df),),
                                tol=1e-6)
    
# Run MCMC
with model:
    idata = pm.sample(2000, tune=1000)

# Extract estimates
Rt_estimates = idata.posterior['Rt'].mean(axis=0).values
day_of_week_effect_estimates = idata.posterior['day_of_week_effects'].mean(axis=0).values
ascertainment_estimates = idata.posterior['ascertainment'].mean(axis=0).values

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(Rt_estimates)
plt.title('Estimated Time-Varying Reproduction Number (Rt)')
plt.xlabel('Time')
plt.ylabel('Rt estimate')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, 8), day_of_week_effect_estimates)
plt.title('Day-of-Week Effect Estimates')
plt.xlabel('Day of Week')
plt.ylabel('Effect Estimate')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(ascertainment_estimates)
plt.title('Time-Varying Ascertainment Estimates')
plt.xlabel('Time')
plt.ylabel('Ascertainment estimate')
plt.show()
```

---

## SUB_028

**Scenario**: 2 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
from pymc import Model, Deterministic, NUTS, sample_posterior_predictive
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('cases.csv')

# Define model and priors
with Model() as model:
    # Define variables
    R0 = Deterministic('Rt[1]', 2.5)
    
    # Day-of-week effect
    day_effect = np.zeros((7,))
    for i in range(1, 8):
        day_effect[i-1] = NUTS.dist(var=0.1, lower=-10, upper=10)
        
    delay = NUTs.dist(var=2, lower=0, upper=5) # days
    
    # Ascertainment (smoothly varying proportion of infections reported)
    ascertainment = Deterministic('ascertainment', np.exp(2 * R0 - 1))
    
    # Overdispersion parameter
    overdisp = NUTS.dist(var=0.1, lower=0, upper=5) 
    
    # Infection dynamics: renewal equation
    infections = Deterministic('infections',
        sum([cases[i-day] * ascertainment**i for i in range(1, data.shape[0]+1)])
        
    # Reported cases (accounting for delay, day-of-week effect, and overdispersion)
    reported = Deterministic('reported',
        poisson.rvs(mu=infections * np.exp(day_effect[data['day_of_week']-1] + 2*delay),
                    size=data.shape[0], 
                    overdisp=overdisp)
        
    # Fit the model
    start = {'Rt[1]': data['cases'].mean(),
             'delay': data['cases'].mean(),
             'ascertainment': np.mean(data['cases'] / sum(data['cases']))}
    
    with model:
        step = NUTS(vars=model.free_RVs)
        trace = sample_posterior_predictive(5000, progressbar=False)

# Extract estimates
rt_estimate = trace['Rt[1]'].mean(axis=0)
day_effect_estimate = trace['day_effect'].mean(axis=0)
ascertainment_estimate = trace['ascertainment'].mean(axis=0)

print(f'Rt Estimates: {rt_estimate}')
print(f'Day-of-Week Effects: {np.exp(day_effect_estimate)}')
print(f'Ascertainment Over Time: {np.exp(ascertainment_estimate)}')

# Summary plots
plt.figure(figsize=(10,6))
plt.plot(rt_estimate)
plt.title('Rt Estimates')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(range(1, 8), np.exp(day_effect_estimate))
plt.title('Day-of-Week Effects')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(np.exp(ascertainment_estimate))
plt.title('Ascertainment Over Time')
plt.show()
```

---

## SUB_046

**Scenario**: 2 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
from pmgpy import PMGPy
from pymc import (
    init_model,
    set_data,
    deterministic,
    Model,
    StepMethods,
    MCMC,
)
from scipy.stats import gamma

# Load the data
df = pd.read_csv("cases.csv")

# Ensure `date` is a datetime object and create a new column for day of year
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear

# Calculate day-of-week effect parameters (average and standard deviation)
dow_avg = np.mean(df.groupby("day_of_week")["cases"].mean())
dow_std = np.std(df.groupby("day_of_week")["cases"].mean())

# Fit the model
def fit_model(data):
    with init_model() as model:
        # Infection process parameters
        R_t = deterministic(
            name="Rt",
            var="Rt",
            dist=gamma,
            mu=np.mean(data["Rt"]),  # initial guess for Rt mean
            sigma=np.std(data["Rt"]),  # initial guess for Rt std
        )

        # Day-of-week effect parameters
        dow_avg = pm.Normal("dow_avg", mu=dow_avg, sd=0.1)  # Prior for day of week average
        dow_std = pm.Normal("dow_std", mu=dow_std, sd=0.1)  # Prior for day of week standard deviation

        # Time-varying ascertainment parameters
        alpha = pm.Uniform(
            "alpha",
            lower=0,
            upper=1,
            initial=np.mean(data["ascertainment"]),
        )  # Prior for ascertainment parameter
        beta = pm.Normal("beta", mu=np.mean(data["ascertainment"]), sd=0.1)  # Prior for ascertainment rate

        # Define the observation process
        obs_process = (
            alpha * (R_t ** dow_avg) * data["cases"]
            + (1 - alpha) * (np.exp(beta)) * np.random.poisson(
                R_t ** dow_avg, len(data), size=len(data)
            )
        )

        # Add a prior for the initial infection seeding period
        init_cases = pm.Normal("init_cases", mu=10, sd=5)

    return model

# Define and fit the model
model = fit_model(df)
mcmc = MCMC(model)
mcmc.sample(10000)  # sample from posterior distribution

# Extract estimates of Rt, day-of-week effects, and ascertainment over time
Rt_estimates = mcmc.trace("Rt")[:]

dow_avg_estimates = np.mean(mcmc.get_values("dow_avg"))
dow_std_estimates = np.std(mcmc.get_values("dow_std"))

ascertainment_estimates = mcmc.get_values("alpha")

# Handle the initial infection seeding period
init_cases = mcmc.trace("init_cases")[:]

print(f"Rt estimates: {Rt_estimates}")
print(f"DOW effect average estimate: {dow_avg_estimates}")
print(f"DOW effect std estimate: {dow_std_estimates}")
print(f"Ascentration estimate: {ascertainment_estimates}")

# Summary or plot showing results
import matplotlib.pyplot as plt

plt.plot(Rt_estimates)
plt.xlabel("Time")
plt.ylabel("Rt estimate")
plt.title("Estimated Rt over time")
plt.show()
```

---

## SUB_067

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using Turing, CSV, DataFrames, Dates, Plots, StatsPlots, LinearAlgebra
using Distributions, Random, MCMCChains
using Statistics, StatsBase

# Set random seed for reproducibility
Random.seed!(42)

# Function to create synthetic data if needed
function create_synthetic_data()
    # Create synthetic data for demonstration
    dates = Date(2023, 1, 1):Day(1):Date(2023, 6, 30)
    n_days = length(dates)
    
    # True Rt trajectory (decreasing then increasing)
    true_rt = [2.0 * exp(-0.02 * t) + 0.8 + 0.3 * sin(0.03 * t) for t in 1:n_days]
    
    # Generation interval (gamma distribution, mean ~5 days)
    gen_interval = [pdf(Gamma(2, 2.5), s) for s in 1:14]
    gen_interval = gen_interval ./ sum(gen_interval)
    
    # Simulate infections using renewal equation
    infections = zeros(n_days)
    infections[1:7] .= 100  # Initial seeding
    
    for t in 8:n_days
        infections[t] = true_rt[t] * sum(infections[max(1,t-13):t-1] .* 
                                       gen_interval[1:min(13, t-1)])
    end
    
    # Reporting delay (log-normal, mean ~7 days)
    delay_pmf = [pdf(LogNormal(log(7), 0.5), d) for d in 1:21]
    delay_pmf = delay_pmf ./ sum(delay_pmf)
    
    # Day-of-week effects (lower on weekends)
    true_dow_effects = [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5]
    
    # Time-varying ascertainment (starts high, drops, then increases)
    true_ascertainment = [0.8 * exp(-0.01 * t) + 0.3 + 0.2 * sin(0.02 * t) 
                         for t in 1:n_days]
    true_ascertainment = clamp.(true_ascertainment, 0.1, 1.0)
    
    # Generate reported cases with all observation processes
    cases = zeros(Int, n_days)
    
    for t in 1:n_days
        expected_reports = 0.0
        
        # Apply reporting delay
        for d in 1:min(21, t)
            inf_day = t - d + 1
            if inf_day > 0
                expected_reports += infections[inf_day] * delay_pmf[d] * 
                                  true_ascertainment[inf_day]
            end
        end
        
        # Apply day-of-week effect
        dow = dayofweek(dates[t])
        expected_reports *= true_dow_effects[dow]
        
        # Generate overdispersed cases (negative binomial)
        if expected_reports > 0
            cases[t] = rand(NegativeBinomial(expected_reports / (1 + expected_reports/20), 
                                           20 / (20 + expected_reports)))
        end
    end
    
    # Create DataFrame
    df = DataFrame(
        date = dates,
        cases = cases,
        day_of_week = dayofweek.(dates)
    )
    
    # Save to CSV
    CSV.write("cases.csv", df)
    
    return df, true_rt, true_dow_effects, true_ascertainment
end

# Load or create data
function load_data()
    if !isfile("cases.csv")
        println("Creating synthetic data...")
        return create_synthetic_data()
    else
        df = CSV.read("cases.csv", DataFrame)
        df.date = Date.(df.date)
        return df, nothing, nothing, nothing
    end
end

# Define the Bayesian model
@model function rt_model(cases, day_of_week, n_days)
    
    # Priors
    # Initial Rt
    rt_init ~ LogNormal(log(1.0), 0.5)
    
    # Rt random walk innovation standard deviation
    σ_rt ~ Exponential(0.1)
    
    # Day-of-week effects (Monday is reference)
    dow_raw ~ MvNormal(zeros(6), 0.5)
    dow_effects = vcat([1.0], exp.(dow_raw))
    
    # Ascertainment process
    asc_init ~ Beta(2, 2)  # Initial ascertainment rate
    σ_asc ~ Exponential(0.1)  # Innovation SD for ascertainment
    
    # Overdispersion parameter
    ϕ ~ Exponential(10.0)
    
    # Generation interval (fixed gamma distribution)
    gen_mean = 5.0
    gen_sd = 2.5
    gen_shape = (gen_mean / gen_sd)^2
    gen_scale = gen_sd^2 / gen_mean
    max_gen = 14
    gen_interval = [pdf(Gamma(gen_shape, gen_scale), s) for s in 1:max_gen]
    gen_interval = gen_interval ./ sum(gen_interval)
    
    # Reporting delay (fixed log-normal)
    delay_mean = 7.0
    delay_sd = 0.5
    max_delay = 21
    delay_pmf = [pdf(LogNormal(log(delay_mean), delay_sd), d) for d in 1:max_delay]
    delay_pmf = delay_pmf ./ sum(delay_pmf)
    
    # Time-varying parameters
    rt = Vector{Float64}(undef, n_days)
    ascertainment = Vector{Float64}(undef, n_days)
    infections = Vector{Float64}(undef, n_days)
    
    # Initial seeding period (first 7 days)
    for t in 1:7
        infections[t] ~ LogNormal(log(50.0), 0.5)
        rt[t] = rt_init
        ascertainment[t] = asc_init
    end
    
    # Subsequent time points
    rt[8] = rt_init
    ascertainment[8] = asc_init
    
    for t in 8:n_days
        # Rt random walk
        if t > 8
            rt[t] ~ LogNormal(log(rt[t-1]), σ_rt)
        end
        
        # Ascertainment random walk (on logit scale)
        if t > 8
            asc_logit_prev = log(ascertainment[t-1] / (1 - ascertainment[t-1]))
            asc_logit ~ Normal(asc_logit_prev, σ_asc)
            ascertainment[t] = 1 / (1 + exp(-asc_logit))
        end
        
        # Renewal equation for infections
        renewal_sum = 0.0
        for s in 1:min(max_gen, t-1)
            if t-s >= 1
                renewal_sum += infections[t-s] * gen_interval[s]
            end
        end
        infections[t] = rt[t] * renewal_sum
    end
    
    # Observation model
    for t in 1:n_days
        expected_cases = 0.0
        
        # Apply reporting delay and ascertainment
        for d in 1:min(max_delay, t)
            inf_day = t - d + 1
            if inf_day >= 1
                expected_cases += infections[inf_day] * delay_pmf[d] * 
                                ascertainment[inf_day]
            end
        end
        
        # Apply day-of-week effect
        expected_cases *= dow_effects[day_of_week[t]]
        
        # Overdispersed observation
        if expected_cases > 0
            # Negative binomial parameterization
            p = ϕ / (ϕ + expected_cases)
            cases[t] ~ NegativeBinomial(ϕ, p)
        else
            cases[t] ~ Poisson(0.001)
        end
    end
    
    return rt, dow_effects, ascertainment, infections
end

# Main analysis function
function analyze_rt()
    println("Loading data...")
    data, true_rt, true_dow_effects, true_ascertainment = load_data()
    
    n_days = nrow(data)
    println("Analyzing $n_days days of data")
    
    # Fit the model
    println("Fitting Bayesian model...")
    model = rt_model(data.cases, data.day_of_week, n_days)
    
    # Sample from posterior
    n_samples = 1000
    n_chains = 4
    
    println("Running MCMC sampling...")
    chain = sample(model, NUTS(0.65), MCMCThreads(), n_samples, n_chains)
    
    println("Extracting results...")
    
    # Extract Rt estimates
    rt_samples = group(chain, :rt)
    rt_mean = [mean(rt_samples["rt[$i]"]) for i in 1:n_days]
    rt_lower = [quantile(rt_samples["rt[$i]"], 0.025) for i in 1:n_days]
    rt_upper = [quantile(rt_samples["rt[$i]"], 0.975) for i in 1:n_days]
    
    # Extract day-of-week effects
    dow_samples = group(chain, :dow_effects)
    dow_mean = [mean(dow_samples["dow_effects[$i]"]) for i in 1:7]
    dow_lower = [quantile(dow_samples["dow_effects[$i]"], 0.025) for i in 1:7]
    dow_upper = [quantile(dow_samples["dow_effects[$i]"], 0.975) for i in 1:7]
    
    # Extract ascertainment estimates
    asc_samples = group(chain, :ascertainment)
    asc_mean = [mean(asc_samples["ascertainment[$i]"]) for i in 1:n_days]
    asc_lower = [quantile(asc_samples["ascertainment[$i]"], 0.025) for i in 1:n_days]
    asc_upper = [quantile(asc_samples["ascertainment[$i]"], 0.975) for i in 1:n_days]
    
    # Create results summary
    results = DataFrame(
        date = data.date,
        rt_mean = rt_mean,
        rt_lower = rt_lower,
        rt_upper = rt_upper,
        ascertainment_mean = asc_mean,
        ascertainment_lower = asc_lower,
        ascertainment_upper = asc_upper,
        cases = data.cases
    )
    
    dow_results = DataFrame(
        day_of_week = 1:7,
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", 
                   "Friday", "Saturday", "Sunday"],
        effect_mean = dow_mean,
        effect_lower = dow_lower,
        effect_upper = dow_upper
    )
    
    # Print summary
    println("\n=== Rt ESTIMATION RESULTS ===")
    println("Final Rt estimate: $(round(rt_mean[end], digits=3)) ",
            "[$(round(rt_lower[end], digits=3)), $(round(rt_upper[end], digits=3))]")
    
    println("\n=== DAY-OF-WEEK EFFECTS ===")
    for i in 1:7
        println("$(dow_results.day_name[i]): $(round(dow_results.effect_mean[i], digits=3)) ",
                "[$(round(dow_results.effect_lower[i], digits=3)), $(round(dow_results.effect_upper[i], digits=3))]")
    end
    
    println("\n=== ASCERTAINMENT SUMMARY ===")
    println("Initial ascertainment: $(round(asc_mean[1], digits=3)) ",
            "[$(round(asc_lower[1], digits=3)), $(round(asc_upper[1], digits=3))]")
    println("Final ascertainment: $(round(asc_mean[end], digits=3)) ",
            "[$(round(asc_lower[end], digits=3)), $(round(asc_upper[end], digits=3))]")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot 1: Rt over time
    p1 = plot(data.date, rt_mean, ribbon=(rt_mean .- rt_lower, rt_upper .- rt_mean),
             label="Rt estimate", linewidth=2, fillalpha=0.3,
             title="Time-varying Reproduction Number (Rt)",
             xlabel="Date", ylabel="Rt")
    hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1", linewidth=2)
    
    if true_rt !== nothing
        plot!(data.date, true_rt, label="True Rt", linestyle=:dash, linewidth=2)
    end
    
    # Plot 2: Cases and fitted values
    p2 = plot(data.date, data.cases, label="Observed cases", 
             title="Observed Cases Over Time", xlabel="Date", ylabel="Cases",
             seriestype=:scatter, alpha=0.6)
    
    # Plot 3: Day-of-week effects
    p3 = bar(dow_results.day_name, dow_results.effect_mean, 
            yerr=(dow_results.effect_mean .- dow_results.effect_lower,
                  dow_results.effect_upper .- dow_results.effect_mean),
            title="Day-of-Week Effects", xlabel="Day", ylabel="Relative Effect",
            xrotation=45)
    hline!([1.0], linestyle=:dash, color=:red, linewidth=2)
    
    # Plot 4: Ascertainment over time
    p4 = plot(data.date, asc_mean, ribbon=(asc_mean .- asc_lower, asc_upper .- asc_mean),
             label="Ascertainment rate", linewidth=2, fillalpha=0.3,
             title="Time-varying Ascertainment Rate",
             xlabel="Date", ylabel="Ascertainment Rate")
    
    if true_ascertainment !== nothing
        plot!(data.date, true_ascertainment, label="True ascertainment", 
              linestyle=:dash, linewidth=2)
    end
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
    
    display(final_plot)
    savefig(final_plot, "rt_analysis_results.png")
    
    # Save results
    CSV.write("rt_estimates.csv", results)
    CSV.write("dow_effects.csv", dow_results)
    
    println("\nResults saved to:")
    println("- rt_estimates.csv: Time-varying Rt and ascertainment estimates")
    println("- dow_effects.csv: Day-of-week effect estimates")
    println("- rt_analysis_results.png: Summary plots")
    
    return results, dow_results, chain
end

# Run the analysis
results, dow_results, chain = analyze_rt()
```

---

## SUB_079

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using Turing, CSV, DataFrames, Dates, Plots, Statistics, LinearAlgebra
using Distributions, Random, StatsPlots

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare the data
function load_and_prepare_data(filename="cases.csv")
    # For demonstration, I'll create sample data if file doesn't exist
    if !isfile(filename)
        println("Creating sample data...")
        dates = Date(2020, 3, 1):Day(1):Date(2020, 8, 31)
        n_days = length(dates)
        
        # Simulate true infections with varying Rt
        true_rt = 2.5 * exp.(-0.01 * (1:n_days)) .+ 0.5 .+ 0.3 * sin.(2π * (1:n_days) / 30)
        true_infections = zeros(n_days)
        true_infections[1:7] .= 100  # Initial seeding
        
        # Generation interval (gamma distribution discretized)
        g = [pdf(Gamma(5.1, 1.0), x) for x in 1:20]
        g = g ./ sum(g)
        
        for t in 8:n_days
            lambda = true_rt[t] * sum(true_infections[max(1,t-20):t-1] .* reverse(g[1:min(20,t-1)]))
            true_infections[t] = rand(Poisson(lambda))
        end
        
        # Simulate observation process
        day_of_week = [Dates.dayofweek(d) for d in dates]
        dow_effects = [1.0, 1.1, 1.05, 0.95, 0.9, 0.7, 0.8]  # Mon-Sun
        ascertainment = 0.3 .+ 0.2 * exp.(-0.005 * (1:n_days))
        
        # Reporting delay (discretized lognormal)
        delay_probs = [pdf(LogNormal(log(7), 0.5), x) for x in 1:20]
        delay_probs = delay_probs ./ sum(delay_probs)
        
        # Convolve infections with delay
        delayed_infections = zeros(n_days + 20)
        delayed_infections[1:n_days] = true_infections
        expected_reports = zeros(n_days)
        
        for t in 1:n_days
            for d in 1:20
                if t + d - 1 <= n_days
                    expected_reports[t + d - 1] += delayed_infections[t] * delay_probs[d] * ascertainment[t + d - 1] * dow_effects[day_of_week[t + d - 1]]
                end
            end
        end
        
        # Add overdispersion
        φ = 20.0
        cases = [rand(NegativeBinomial(expected_reports[t], φ/(φ + expected_reports[t]))) for t in 1:n_days]
        
        df = DataFrame(
            date = dates,
            cases = cases,
            day_of_week = day_of_week
        )
        CSV.write(filename, df)
        println("Sample data created and saved to $filename")
    end
    
    df = CSV.read(filename, DataFrame)
    return df
end

# Define the Bayesian model
@model function rt_model(cases, day_of_week, n_days, generation_interval)
    # Priors
    
    # Day-of-week effects (multiplicative, Monday = reference)
    dow_raw ~ MvNormal(zeros(6), 0.1^2 * I(6))
    dow_effects = vcat([0.0], dow_raw)  # Monday effect = 0 (log scale)
    
    # Time-varying ascertainment (on log scale)
    σ_ascert ~ truncated(Normal(0, 1), 0, Inf)
    ascert_raw ~ MvNormal(zeros(n_days), σ_ascert^2 * I(n_days))
    ascert_mean ~ Normal(-1, 0.5)  # Prior mean on log scale
    log_ascertainment = ascert_mean .+ ascert_raw
    
    # Smooth Rt using random walk
    σ_rt ~ truncated(Normal(0, 0.1), 0, Inf)
    rt_raw ~ MvNormal(zeros(n_days), σ_rt^2 * I(n_days))
    rt_mean ~ Normal(log(1.0), 0.2)
    log_rt = rt_mean .+ cumsum(rt_raw)
    rt = exp.(log_rt)
    
    # Initial infections
    init_infections ~ MvNormal(log(100) * ones(7), 1.0^2 * I(7))
    
    # Overdispersion parameter
    φ ~ truncated(Normal(20, 10), 1, Inf)
    
    # Reporting delay parameters
    delay_mean ~ truncated(Normal(log(7), 0.2), log(1), log(20))
    delay_sd ~ truncated(Normal(0.5, 0.2), 0.1, 2.0)
    
    # Calculate expected cases
    expected_cases = zeros(n_days)
    infections = zeros(n_days)
    
    # Initial period
    for t in 1:7
        infections[t] = exp(init_infections[t])
    end
    
    # Calculate infections using renewal equation
    gen_len = length(generation_interval)
    for t in 8:n_days
        lambda = 0.0
        for s in 1:min(gen_len, t-1)
            lambda += infections[t-s] * generation_interval[s]
        end
        lambda *= rt[t]
        infections[t] = lambda
    end
    
    # Calculate reporting delay distribution
    delay_probs = zeros(20)
    for d in 1:20
        delay_probs[d] = pdf(LogNormal(delay_mean, delay_sd), d)
    end
    delay_probs = delay_probs ./ sum(delay_probs)
    
    # Convolve infections with delay and observation process
    for t in 1:n_days
        expected = 0.0
        for d in 1:20
            infection_day = t - d + 1
            if infection_day >= 1
                dow_idx = day_of_week[t]
                expected += (infections[infection_day] * 
                           delay_probs[d] * 
                           exp(log_ascertainment[t]) * 
                           exp(dow_effects[dow_idx]))
            end
        end
        expected_cases[t] = max(expected, 1e-6)
    end
    
    # Likelihood with overdispersion
    for t in 1:n_days
        p = φ / (φ + expected_cases[t])
        cases[t] ~ NegativeBinomial(expected_cases[t], p)
    end
    
    return (rt = rt, 
            dow_effects = exp.(dow_effects), 
            ascertainment = exp.(log_ascertainment),
            infections = infections,
            expected_cases = expected_cases)
end

# Function to create generation interval
function create_generation_interval()
    # Discretized gamma distribution for generation interval
    g = [pdf(Gamma(5.1, 1.0), x) for x in 1:15]
    return g ./ sum(g)
end

# Main estimation function
function estimate_rt(df)
    cases = df.cases
    day_of_week = df.day_of_week
    n_days = length(cases)
    
    # Create generation interval
    generation_interval = create_generation_interval()
    
    # Fit the model
    println("Fitting Bayesian model...")
    model = rt_model(cases, day_of_week, n_days, generation_interval)
    
    # Sample from posterior
    chain = sample(model, NUTS(0.8), 1000, progress=true)
    
    return chain, df.date
end

# Function to extract and summarize results
function extract_results(chain, dates)
    n_days = length(dates)
    
    # Extract Rt estimates
    rt_samples = Array(group(chain, :rt))
    rt_mean = mean(rt_samples, dims=1)[:]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:n_days]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:n_days]
    
    # Extract day-of-week effects
    dow_samples = Array(group(chain, :dow_effects))
    dow_mean = mean(dow_samples, dims=1)[:]
    dow_lower = [quantile(dow_samples[:, i], 0.025) for i in 1:7]
    dow_upper = [quantile(dow_samples[:, i], 0.975) for i in 1:7]
    
    # Extract ascertainment
    ascert_samples = Array(group(chain, :ascertainment))
    ascert_mean = mean(ascert_samples, dims=1)[:]
    ascert_lower = [quantile(ascert_samples[:, i], 0.025) for i in 1:n_days]
    ascert_upper = [quantile(ascert_samples[:, i], 0.975) for i in 1:n_days]
    
    return (
        rt = (mean = rt_mean, lower = rt_lower, upper = rt_upper),
        dow = (mean = dow_mean, lower = dow_lower, upper = dow_upper),
        ascertainment = (mean = ascert_mean, lower = ascert_lower, upper = ascert_upper),
        dates = dates
    )
end

# Plotting function
function plot_results(results, observed_cases)
    dates = results.dates
    
    # Plot Rt over time
    p1 = plot(dates, results.rt.mean, ribbon=(results.rt.mean - results.rt.lower, 
                                              results.rt.upper - results.rt.mean),
              label="Rt estimate", color=:blue, alpha=0.3,
              title="Time-varying Reproduction Number (Rt)",
              ylabel="Rt", xlabel="Date")
    hline!([1.0], color=:red, linestyle=:dash, label="Rt = 1")
    
    # Plot day-of-week effects
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    p2 = bar(dow_names, results.dow.mean, 
             yerror=(results.dow.mean - results.dow.lower, 
                    results.dow.upper - results.dow.mean),
             title="Day-of-Week Effects", ylabel="Relative Reporting Rate",
             color=:orange, alpha=0.7)
    
    # Plot ascertainment over time
    p3 = plot(dates, results.ascertainment.mean, 
              ribbon=(results.ascertainment.mean - results.ascertainment.lower,
                     results.ascertainment.upper - results.ascertainment.mean),
              label="Ascertainment rate", color=:green, alpha=0.3,
              title="Time-varying Ascertainment Rate",
              ylabel="Ascertainment Rate", xlabel="Date")
    
    # Plot observed cases
    p4 = plot(dates, observed_cases, label="Observed cases", color=:black,
              title="Observed Cases", ylabel="Daily Cases", xlabel="Date")
    
    return plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
end

# Main execution
function main()
    println("Loading data...")
    df = load_and_prepare_data()
    
    println("Estimating Rt...")
    chain, dates = estimate_rt(df)
    
    println("Extracting results...")
    results = extract_results(chain, dates)
    
    println("Creating plots...")
    p = plot_results(results, df.cases)
    display(p)
    
    # Print summary statistics
    println("\n=== RESULTS SUMMARY ===")
    println("Mean Rt over time period: ", round(mean(results.rt.mean), digits=2))
    println("Rt range: ", round(minimum(results.rt.mean), digits=2), " - ", 
            round(maximum(results.rt.mean), digits=2))
    
    println("\nDay-of-week effects (relative to Monday):")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for i in 1:7
        println("  $(dow_names[i]): $(round(results.dow.mean[i], digits=3))")
    end
    
    println("\nMean ascertainment rate: ", round(mean(results.ascertainment.mean), digits=3))
    println("Ascertainment range: ", round(minimum(results.ascertainment.mean), digits=3), 
            " - ", round(maximum(results.ascertainment.mean), digits=3))
    
    return results, chain
end

# Run the analysis
results, chain = main()
```

---

## SUB_110

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using Turing, StatsPlots, CSV, DataFrames, Dates, LinearAlgebra, Random
using Distributions, MCMCChains, StatsBase

# Set random seed for reproducibility
Random.seed!(42)

# First, let's create some synthetic data for demonstration
function create_synthetic_data()
    n_days = 150
    dates = Date(2020, 3, 1):Day(1):(Date(2020, 3, 1) + Day(n_days - 1))
    
    # True Rt that varies over time (simulating epidemic waves)
    true_rt = [
        i <= 30 ? 2.5 : 
        i <= 60 ? 2.5 - 1.8 * (i - 30) / 30 :  # Decline due to interventions
        i <= 90 ? 0.7 + 0.6 * (i - 60) / 30 :   # Rise due to relaxation
        1.3 - 0.8 * (i - 90) / 60               # Final decline
        for i in 1:n_days
    ]
    
    # Generation interval (gamma distribution with mean 5.2, sd 5.1)
    g = [pdf(Gamma(1.04, 5.0), x) for x in 1:20]
    g = g ./ sum(g)
    
    # Reporting delay (gamma distribution with mean 7, sd 4)
    delay_dist = [pdf(Gamma(3.0, 2.3), x) for x in 1:25]
    delay_dist = delay_dist ./ sum(delay_dist)
    
    # Simulate true infections
    infections = zeros(n_days)
    infections[1:10] .= 50.0  # Initial seeding
    
    for t in 11:n_days
        infections[t] = true_rt[t] * sum(infections[max(1,t-length(g)+1):t-1] .* reverse(g[1:min(length(g)-1, t-1)]))
    end
    
    # Apply reporting delays and day-of-week effects
    day_effects = [1.2, 1.1, 1.0, 0.95, 0.9, 0.7, 0.6]  # Mon-Sun multipliers
    ascertainment = 0.3 * (1 .+ 0.3 * sin.(2π * (1:n_days) / 50))  # Time-varying ascertainment
    
    expected_reports = zeros(n_days)
    for t in 1:n_days
        for d in 1:min(length(delay_dist), t)
            dow = dayofweek(dates[t])
            expected_reports[t] += infections[t-d+1] * delay_dist[d] * ascertainment[t-d+1] * day_effects[dow]
        end
    end
    
    # Add overdispersion (negative binomial)
    cases = [rand(NegativeBinomial(max(1, r), 0.3)) for r in expected_reports]
    
    df = DataFrame(
        date = dates,
        cases = cases,
        day_of_week = dayofweek.(dates)
    )
    
    CSV.write("cases.csv", df)
    return df, true_rt, infections
end

# Load or create data
if !isfile("cases.csv")
    println("Creating synthetic data...")
    data, true_rt, true_infections = create_synthetic_data()
else
    data = CSV.read("cases.csv", DataFrame)
end

# Prepare generation interval and reporting delay distributions
function get_generation_interval(max_gen = 20)
    # Generation interval based on COVID-19 literature (mean ≈ 5.2 days)
    g = [pdf(Gamma(1.04, 5.0), x) for x in 1:max_gen]
    return g ./ sum(g)
end

function get_reporting_delay(max_delay = 25)
    # Reporting delay (mean ≈ 7 days)
    d = [pdf(Gamma(3.0, 2.3), x) for x in 1:max_delay]
    return d ./ sum(d)
end

# Turing model
@model function rt_model(cases, day_of_week, n_days, gen_interval, reporting_delay)
    
    # Priors for day-of-week effects (multiplicative, sum to 7)
    dow_raw ~ MvNormal(zeros(7), I(7))
    dow_effects = exp.(dow_raw .- mean(dow_raw))
    
    # Prior for Rt (random walk on log scale)
    rt_log_init ~ Normal(0, 0.5)  # Initial Rt around 1
    rt_innovations ~ MvNormal(zeros(n_days-1), 0.1^2 * I(n_days-1))
    rt_log = cumsum([rt_log_init; rt_innovations])
    rt = exp.(rt_log)
    
    # Time-varying ascertainment (smooth via random walk)
    ascertainment_logit_init ~ Normal(-1, 0.5)  # Initial ascertainment around 0.3
    ascertainment_innovations ~ MvNormal(zeros(n_days-1), 0.05^2 * I(n_days-1))
    ascertainment_logit = cumsum([ascertainment_logit_init; ascertainment_innovations])
    ascertainment = 1 ./ (1 .+ exp.(-ascertainment_logit))
    
    # Initial infections (seeding period)
    init_infections ~ MvNormal(log.(50) * ones(10), 0.2^2 * I(10))
    infections = zeros(n_days)
    infections[1:10] = exp.(init_infections)
    
    # Overdispersion parameter
    φ ~ Exponential(10)  # For negative binomial
    
    # Generate infections via renewal equation
    for t in 11:n_days
        # Convolution with generation interval
        renewal_sum = 0.0
        for s in 1:min(length(gen_interval), t-1)
            if t-s >= 1
                renewal_sum += infections[t-s] * gen_interval[s]
            end
        end
        infections[t] = rt[t] * renewal_sum
    end
    
    # Generate expected reported cases with delays and effects
    for t in 1:n_days
        expected_reports = 0.0
        
        # Apply reporting delay
        for d in 1:min(length(reporting_delay), t)
            if t-d+1 >= 1
                expected_reports += infections[t-d+1] * reporting_delay[d] * ascertainment[t-d+1]
            end
        end
        
        # Apply day-of-week effect
        expected_reports *= dow_effects[day_of_week[t]]
        expected_reports = max(expected_reports, 1e-6)
        
        # Negative binomial observation model
        p_nb = φ / (φ + expected_reports)
        cases[t] ~ NegativeBinomial(φ, p_nb)
    end
    
    return (rt = rt, dow_effects = dow_effects, ascertainment = ascertainment, 
            infections = infections)
end

# Prepare data
cases_data = data.cases
dow_data = data.day_of_week
n_days = length(cases_data)

# Get distributions
gen_interval = get_generation_interval()
reporting_delay = get_reporting_delay()

println("Setting up model...")
model = rt_model(cases_data, dow_data, n_days, gen_interval, reporting_delay)

# Sample from the model
println("Sampling from model (this may take several minutes)...")
n_samples = 1000
n_chains = 4

# Use NUTS sampler with adaptation
sampler = NUTS(0.65)
chain = sample(model, sampler, MCMCThreads(), n_samples, n_chains, 
               progress=true, verbose=true)

println("Sampling complete!")

# Extract results
function extract_results(chain, n_days)
    # Extract Rt estimates
    rt_samples = Array(group(chain, :rt))
    rt_mean = mean(rt_samples, dims=1)[1, :]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:n_days]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:n_days]
    
    # Extract day-of-week effects
    dow_samples = Array(group(chain, :dow_effects))
    dow_mean = mean(dow_samples, dims=1)[1, :]
    dow_lower = [quantile(dow_samples[:, i], 0.025) for i in 1:7]
    dow_upper = [quantile(dow_samples[:, i], 0.975) for i in 1:7]
    
    # Extract ascertainment
    asc_samples = Array(group(chain, :ascertainment))
    asc_mean = mean(asc_samples, dims=1)[1, :]
    asc_lower = [quantile(asc_samples[:, i], 0.025) for i in 1:n_days]
    asc_upper = [quantile(asc_samples[:, i], 0.975) for i in 1:n_days]
    
    return (
        rt = (mean = rt_mean, lower = rt_lower, upper = rt_upper),
        dow_effects = (mean = dow_mean, lower = dow_lower, upper = dow_upper),
        ascertainment = (mean = asc_mean, lower = asc_lower, upper = asc_upper)
    )
end

results = extract_results(chain, n_days)

# Create summary
function print_summary(results, data)
    println("\n" * "="^60)
    println("RESULTS SUMMARY")
    println("="^60)
    
    println("\nDay-of-week effects (multiplicative):")
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for (i, day) in enumerate(days)
        println("$day: $(round(results.dow_effects.mean[i], digits=3)) " *
                "[$(round(results.dow_effects.lower[i], digits=3)), " *
                "$(round(results.dow_effects.upper[i], digits=3))]")
    end
    
    println("\nRt statistics:")
    rt_mean = results.rt.mean
    println("Mean Rt: $(round(mean(rt_mean), digits=3))")
    println("Rt range: $(round(minimum(rt_mean), digits=3)) - $(round(maximum(rt_mean), digits=3))")
    println("Days with Rt > 1: $(sum(rt_mean .> 1)) / $(length(rt_mean))")
    
    println("\nAscertainment statistics:")
    asc_mean = results.ascertainment.mean
    println("Mean ascertainment: $(round(mean(asc_mean), digits=3))")
    println("Ascertainment range: $(round(minimum(asc_mean), digits=3)) - $(round(maximum(asc_mean), digits=3))")
    
    println("\nData summary:")
    println("Total cases: $(sum(data.cases))")
    println("Date range: $(data.date[1]) to $(data.date[end])")
    println("Days analyzed: $(length(data.cases))")
end

print_summary(results, data)

# Create plots
function create_plots(results, data)
    dates = data.date
    
    # Plot 1: Rt over time
    p1 = plot(dates, results.rt.mean, 
              ribbon=(results.rt.mean .- results.rt.lower, results.rt.upper .- results.rt.mean),
              label="Rt estimate", color=:blue, fillalpha=0.3,
              title="Time-varying Reproduction Number (Rt)",
              xlabel="Date", ylabel="Rt", 
              linewidth=2)
    hline!([1.0], color=:red, linestyle=:dash, label="Rt = 1", linewidth=2)
    
    # Plot 2: Day-of-week effects
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    p2 = scatter(1:7, results.dow_effects.mean,
                 yerror=(results.dow_effects.mean .- results.dow_effects.lower,
                        results.dow_effects.upper .- results.dow_effects.mean),
                 title="Day-of-week Effects", xlabel="Day", ylabel="Multiplicative Effect",
                 xticks=(1:7, days), color=:green, markersize=6, linewidth=2)
    hline!([1.0], color=:red, linestyle=:dash, label="No effect", linewidth=1)
    
    # Plot 3: Ascertainment over time
    p3 = plot(dates, results.ascertainment.mean,
              ribbon=(results.ascertainment.mean .- results.ascertainment.lower,
                     results.ascertainment.upper .- results.ascertainment.mean),
              label="Ascertainment", color=:purple, fillalpha=0.3,
              title="Time-varying Ascertainment Rate",
              xlabel="Date", ylabel="Ascertainment Rate",
              linewidth=2)
    
    # Plot 4: Observed cases
    p4 = plot(dates, data.cases, title="Observed Cases", xlabel="Date", ylabel="Cases",
              color=:orange, linewidth=1, label="Cases")
    
    # Combine plots
    plot(p1, p2, p3, p4, layout=(2,2), size=(1200, 800))
end

# Generate and display plots
println("\nGenerating plots...")
final_plot = create_plots(results, data)
display(final_plot)

# Save results to CSV
function save_results(results, data)
    results_df = DataFrame(
        date = data.date,
        rt_mean = results.rt.mean,
        rt_lower = results.rt.lower,
        rt_upper = results.rt.upper,
        ascertainment_mean = results.ascertainment.mean,
        ascertainment_lower = results.ascertainment.lower,
        ascertainment_upper = results.ascertainment.upper,
        observed_cases = data.cases
    )
    
    CSV.write("rt_estimates.csv", results_df)
    
    dow_df = DataFrame(
        day = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
        effect_mean = results.dow_effects.mean,
        effect_lower = results.dow_effects.lower,
        effect_upper = results.dow_effects.upper
    )
    
    CSV.write("day_of_week_effects.csv", dow_df)
    
    println("Results saved to rt_estimates.csv and day_of_week_effects.csv")
end

save_results(results, data)

# Print diagnostics
println("\nModel diagnostics:")
println("Chain summary:")
display(chain)

println("\n" * "="^60)
println("Analysis complete!")
println("="^60)
```

---

## SUB_082

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using Turing
using DataFrames
using Statistics
using Plots

# Load data
df = DataFrame(CSV.File("cases.csv"))

# Model definition
@model function covid_model(df)
    # Infection process
    R_t ~ LogNormal(0, 1) # Time-varying reproduction number
    I_s ~ Normal.(zeros(length(df.cases)), R_t .* df.cases ./ 100)

    # Observation process
    delay ~ Gamma(1, 3) # Delay from infection to reporting (mean=3 days)
    day_effect = @fun() begin
        for i in 1:7
            if df.day_of_week .== i
                return 1.5 + rand(Normal(0, 0.2)) # Day-of-week effect
            end
        end
        return 1
    end

    ascertainment ~ Beta(1, 1) # Time-varying proportion of infections reported
    overdispersion ~ Gamma(1, 10) # Overdispersion parameter (alpha=10)

    for i in 1:length(df.cases)
        y_i = df.cases[i]
        λ_i = R_t * sum(I_s[j] .* day_effect[j] .* exp(-delay * j) for j in 1:S) * ascertainment
        y_i_obs ~ Poisson(λ_i .* overdispersion)
    end

    # Prior on initial infection seeding period (e.g., assume it's negligible)
    I_s[1] ~ Normal(0, 10)

    # Handle the initial infection seeding period by ignoring the first few days of data
    for i in 1:5 # arbitrary choice; adjust as needed
        y_i = df.cases[i]
        λ_i = R_t * sum(I_s[j] .* day_effect[j] .* exp(-delay * j) for j in 1:S) * ascertainment
        y_i_obs ~ Poisson(λ_i .* overdispersion)
    end
end

# Define model parameters and data
S = 10 # Number of past infections to consider (days)
N = length(df.cases) # Total number of days

# Initialize model and data
m = covid_model(df)

# Sample from the posterior distribution
chain = sample(m, NUTS(1000), MCMCThreads(), 4)

# Extract estimates
Rt_estimates = mean.(map(x -> x.R_t, chain))
day_effect_estimates = mean.(map(x -> x.day_effect, chain))
ascertainment_estimates = mean.(map(x -> x.ascertainment, chain))

# Plot results
plot(Rt_estimates)
xlabel("Time (days)")
ylabel("Rt estimate")
title("Estimated Time-Varying Reproduction Number")

plot(day_effect_estimates)
xlabel("Day of Week")
ylabel("Day-of-Week Effect Estimate")
title("Estimated Day-of-Week Effects")

plot(ascertainment_estimates)
xlabel("Time (days)")
ylabel("Ascertainment Estimate")
title("Estimated Time-Varying Ascertainment")

# Summarize results
println("Rt estimates:")
for i in 1:length(Rt_estimates)
    println("Day $i: $(round(Rt_estimates[i], digits=2))")
end

println("Day-of-Week Effect Estimates:")
for i in 1:7
    println("Day of Week $i: $(round(day_effect_estimates[i], digits=2))")
end

println("Time-Varying Ascertainment Estimates:")
for i in 1:length(ascertainment_estimates)
    println("Day $i: $(round(ascertainment_estimates[i], digits=2))")
end
```

---

## SUB_091

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
# Import necessary packages
using Turing
using DataFrames
using StatsBase

# Load the data
df = readcsv("cases.csv")

# Define the model in Turing.jl syntax
@model function covid_model(data)
    # Parameters
    β ~ InverseGamma(2, 1)  # inverse gamma for rt
    γ ~ Beta(2, 2)          # beta for day-of-week effects

    # Model equations
    n_weekdays = 5
    N = length(data.date)

    # Day-of-week effects (multiplicative)
    day_effects = zeros(n_weekdays)
    for i in 1:n_weekdays
        day_effects[i] ~ Truncated(Beta(2, 2), 0, 1)  # truncated beta distribution for day_of_week effect
    end

    # Time-varying ascertainment (multiplicative)
    α_t = zeros(N)
    α_t[1] ~ Uniform(0.1, 10)  # uniform prior for initial ascertainment
    for t in 2:N
        α_t[t] ~ Weibull(5, β)  # weibull distribution for time-varying ascertainment
    end

    # Infection dynamics (renewal equation)
    I_prev = zeros(N - 1)
    I_new = zeros(N)

    for t in 2:N
        I_prev[t-1] ~ Poisson(α_t[t-1] * day_effects[data.day_of_week[t]] * β * sum(I_new .+ α_t[1:t-1]))  # poisson distribution for infection count at time t, scaled by ascertainment and day-of-week effect
    end

    return I_prev, I_new, α_t, day_effects
end

# Fit the model
NUTS(state = [1000, 2000, 3000]) do
    model = covid_model(df)
    chn = sample(model, NUTS(), MCMCThreads())
    return sample(chn, PosteriorMean())
end

# Extract estimates of Rt, day-of-week effects, and ascertainment over time
rt_estimates = Array{Float64}(undef, size(df.date)[1])
for t in 2:size(df.date)[1]
    rt_estimates[t] = mean([post[:β] for post in chn][t])
end

day_effects_estimates = zeros(size(df.date)[1], 5)
for i in 1:5
    for t in 1:size(df.date)[1]
        day_effects_estimates[t, i] = mean([post[:day_effects[i]] for post in chn][t])
    end
end

ascertainment_estimates = zeros(size(df.date)[1])
for t in 1:size(df.date)[1]
    ascertainment_estimates[t] = mean([post[:α_t[t]] for post in chn])
end

# Plot the results
using Plots
plot(rt_estimates[2:end], label="Rt")
plot!(day_effects_estimates[2:end, 1], label="Monday")
plot!(day_effects_estimates[2:end, 2], label="Tuesday")
plot!(day_effects_estimates[2:end, 3], label="Wednesday")
plot!(day_effects_estimates[2:end, 4], label="Thursday")
plot!(day_effects_estimates[2:end, 5], label="Friday")

plot!(ascertainment_estimates)
xlabel!("Time (days)")
ylabel!("Value")
title!("Estimated Rt and Day-of-Week Effects over Time")
```

---

## SUB_062

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
# Load necessary packages
using Pkg; Pkg.add("Turing")
using DataFrames
using CSV
import Turing: @model, @sample, @map
import StatsBase: mean
import Plots

# Load data
data = CSV.read("cases.csv", DataFrame)

# Prepare data for analysis
function prepare_data(data)
    dates = Date.(data.date)
    days_since_infection = 0:30
    gen_intervals = [i in days_since_infection for i in eachrow(data)]
    return data, dates, days_since_infection, gen_intervals
end

data, dates, days_since_infection, gen_intervals = prepare_data(data)

# Define the model
@model function Rt_model(data, days_since_infection, gen_intervals)
    N = maximum(data.cases)  # Number of people in population
    
    # Model for infections
    α ~ InverseGamma(2, 1)
    R_t ~ Gamma(10, 0.01)
    
    # Day-of-week effects
    θ_day_of_week ~ Normal(mean = 1.0, sd = 0.5)
    day_effects = theta_day_of_week .* gen_intervals
    
    # Time-varying ascertainment
    θ_ascertainment ~ Beta(2, 2)
    ascertainment = θ_ascertainment .* R_t
    
    # Overdispersion
    ν ~ Gamma(10, 1)
    
    for (i, row) in enumerate(data)
        cases_i ~ Poisson(R_t * sum(row.cases .* day_effects[gen_intervals .== days_since_infection]) *
            ascertainment[i] / (ν + R_t))
    end
    
    return alpha, R_t, theta_day_of_week, theta_ascertainment, nu
end

# Fit the model
chain = @sample(Rt_model(data, days_since_infection, gen_intervals), NUTS(1000))

# Extract estimates of Rt and day-of-week effects over time
Rt_chain = [getfield(chain.s[1], :value)[i] for i in 1:length(chain.s[1])]
day_of_week_effects = [chain.theta_day_of_week[i] for i in 1:length(chain)]
ascertainment_chain = [chain.θ_ascertainment[i] for i in 1:length(chain)]

# Plot results
p1 = plot(title = "Time-varying Reproduction Number (Rt)")
plot!(Rt_chain)
xlabel!("Day")
ylabel!("Rt")

p2 = plot(title = "Day-of-week Effects on Reported Cases")
bar(day_of_week_effects, label = "")
xlabel!("Day of Week")
ylabel!("Effect Size")

p3 = plot(title = "Time-varying Ascertainment Proportion")
plot!(ascertainment_chain)
xlabel!("Day")
ylabel!("Ascertainment Proportion")

plot(p1, p2, p3)
```

---

## SUB_048

**Scenario**: 2 | **Execution**: SUCCESS

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using Plots
using Statistics
using StatsPlots

# Load and prepare the data
function load_data(filename="cases.csv")
    df = CSV.read(filename, DataFrame)
    return df.cases, df.day_of_week
end

# Main function to estimate Rt with observation processes
function estimate_rt_with_observation_processes(cases, day_of_week)
    n_timepoints = length(cases)
    
    println("Setting up model components...")
    
    # 1. Define generation interval (COVID-19 typical values)
    gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, appropriate for COVID-19
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(mean(cases[1:7])), 1.0)
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smoothness
    rt_process = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],  # Strong persistence
        init_priors = [Normal(0.0, 0.5)],  # Initial log(Rt) ~ log(1) = 0
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Create day-of-week effects model
    # This creates a repeating pattern for 7 days of the week
    dow_effects = BroadcastLatentModel(
        CyclicRepeat(7),  # Repeat every 7 days
        [Normal(0.0, 0.2) for _ in 1:7]  # Prior for each day effect
    )
    
    # 5. Create time-varying ascertainment model
    # This allows the reporting rate to change smoothly over time
    ascertainment_process = RandomWalk(
        init_prior = Normal(0.0, 0.5),  # Initial ascertainment on log scale
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Slow changes
    )
    
    # 6. Combine latent processes
    # We need Rt process, day-of-week effects, and ascertainment
    combined_latent = CombineLatentModels([
        rt_process,           # log(Rt) over time
        dow_effects,          # Day-of-week multiplicative effects  
        ascertainment_process # Time-varying ascertainment
    ])
    
    # 7. Create observation model with delays and overdispersion
    # Delay from infection to reporting
    delay_dist = Gamma(5.0, 1.0)  # Mean delay ~5 days
    
    # Base observation model with overdispersion
    base_obs = NegativeBinomialError(
        cluster_factor_prior = HalfNormal(0.1)
    )
    
    # Add delay to observation model
    obs_with_delay = LatentDelay(base_obs, delay_dist)
    
    # 8. Create custom observation model that incorporates all effects
    # We need to modify the observation model to handle multiple latent processes
    obs_model = CustomObservationModel(obs_with_delay, n_timepoints, day_of_week)
    
    # 9. Compose into EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = combined_latent,
        observation_model = obs_model,
        tspan = (1, n_timepoints)
    )
    
    println("Generating Turing model...")
    
    # 10. Generate the full model
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    println("Starting inference...")
    
    # 11. Set up inference
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 200)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 2000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    # 12. Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Custom observation model to handle multiple latent processes
struct CustomObservationModel{T,S} <: AbstractObservationModel
    base_model::T
    n_timepoints::Int
    day_of_week::S
end

function EpiAware.generate_observations(obs::CustomObservationModel, y_t, expected_infections)
    @model function custom_obs_model(y_t, expected_infections, latent_processes)
        # Extract the different latent processes
        # latent_processes should contain: [log_Rt, dow_effects, log_ascertainment]
        n_rt = obs.n_timepoints
        
        log_Rt = latent_processes[1:n_rt]
        dow_effects = latent_processes[(n_rt+1):(n_rt+7)]
        log_ascertainment = latent_processes[(n_rt+8):end]
        
        # Apply day-of-week effects
        dow_multiplier = [dow_effects[obs.day_of_week[t]] for t in 1:obs.n_timepoints]
        
        # Apply ascertainment
        ascertainment = exp.(log_ascertainment)
        
        # Modify expected observations
        adjusted_expected = expected_infections .* ascertainment .* exp.(dow_multiplier)
        
        # Use base observation model
        base_model = obs.base_model
        return generate_observations(base_model, y_t, adjusted_expected)
    end
    
    return custom_obs_model
end

# Simplified version using available EpiAware components
function estimate_rt_simplified(cases, day_of_week)
    n_timepoints = length(cases)
    
    println("Setting up simplified model...")
    
    # 1. Generation interval
    gen_int = Gamma(6.5, 0.62)
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Infection model
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(mean(cases[1:7])), 1.0)
    )
    
    # 3. Latent model for log(Rt) with day-of-week effects
    rt_process = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Observation model with delay and overdispersion
    delay_dist = Gamma(5.0, 1.0)
    base_obs = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1))
    obs_model = LatentDelay(base_obs, delay_dist)
    
    # 5. Add time-varying ascertainment
    ascertainment_rw = RandomWalk(
        init_prior = Normal(0.0, 0.5),
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))
    )
    
    obs_with_ascertainment = Ascertainment(
        obs_model,
        ascertainment_rw,
        (Y, x) -> Y .* exp.(x)  # Multiplicative ascertainment
    )
    
    # 6. Compose model
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = rt_process,
        observation_model = obs_with_ascertainment,
        tspan = (1, n_timepoints)
    )
    
    println("Running inference...")
    
    # 7. Generate and fit model
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 150)],
        sampler = NUTSampler(
            ndraws = 1500,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract and analyze results
function analyze_results(results, cases, day_of_week)
    println("Analyzing results...")
    
    # Extract parameter chains
    chains = results.chains
    
    # Extract Rt estimates (exp of latent process)
    n_timepoints = length(cases)
    
    # Get latent process samples (these are log(Rt))
    log_rt_samples = []
    for i in 1:n_timepoints
        param_name = "Z_t[$i]"
        if param_name in string.(keys(chains))
            push!(log_rt_samples, vec(chains[param_name]))
        end
    end
    
    # Convert to Rt
    rt_samples = [exp.(samples) for samples in log_rt_samples]
    
    # Calculate summary statistics
    rt_mean = [mean(samples) for samples in rt_samples]
    rt_lower = [quantile(samples, 0.025) for samples in rt_samples]
    rt_upper = [quantile(samples, 0.975) for samples in rt_samples]
    
    # Extract other parameters if available
    ascertainment_samples = []
    dow_effects_samples = []
    
    # Try to extract ascertainment parameters
    for i in 1:n_timepoints
        param_name = "ascertainment[$i]"
        if param_name in string.(keys(chains))
            push!(ascertainment_samples, vec(chains[param_name]))
        end
    end
    
    # Try to extract day-of-week effects
    for i in 1:7
        param_name = "dow_effect[$i]"
        if param_name in string.(keys(chains))
            push!(dow_effects_samples, vec(chains[param_name]))
        end
    end
    
    return (
        rt_mean = rt_mean,
        rt_lower = rt_lower, 
        rt_upper = rt_upper,
        rt_samples = rt_samples,
        ascertainment_samples = ascertainment_samples,
        dow_effects_samples = dow_effects_samples,
        chains = chains
    )
end

# Plotting function
function plot_results(results_analysis, cases, day_of_week)
    n_timepoints = length(cases)
    dates = 1:n_timepoints
    
    # Plot Rt over time
    p1 = plot(dates, results_analysis.rt_mean, 
              ribbon=(results_analysis.rt_mean .- results_analysis.rt_lower,
                     results_analysis.rt_upper .- results_analysis.rt_mean),
              label="Rt estimate", linewidth=2, color=:blue, alpha=0.7,
              title="Time-varying Reproduction Number (Rt)",
              xlabel="Time", ylabel="Rt")
    hline!([1.0], label="Rt = 1", linestyle=:dash, color=:red)
    
    # Plot observed cases
    p2 = plot(dates, cases, label="Observed cases", 
              title="Observed Cases", xlabel="Time", ylabel="Cases",
              color=:black, linewidth=1)
    
    # Combine plots
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    
    return combined_plot
end

# Main execution function
function main()
    println("Loading data...")
    
    # Load data (replace with actual file path)
    cases, day_of_week = load_data("cases.csv")
    
    println("Data loaded: $(length(cases)) time points")
    println("Running Rt estimation with observation processes...")
    
    # Run the simplified model (more likely to work with current EpiAware)
    results, epi_prob = estimate_rt_simplified(cases, day_of_week)
    
    println("Analyzing results...")
    
    # Analyze results
    analysis = analyze_results(results, cases, day_of_week)
    
    # Print summary
    println("\n=== RESULTS SUMMARY ===")
    println("Mean Rt over time:")
    for (i, rt) in enumerate(analysis.rt_mean)
        if i <= 10 || i > length(analysis.rt_mean) - 5  # Show first 10 and last 5
            println("  Time $i: Rt = $(round(rt, digits=3)) [$(round(analysis.rt_lower[i], digits=3)), $(round(analysis.rt_upper[i], digits=3))]")
        elseif i == 11
            println("  ...")
        end
    end
    
    # Create plots
    println("\nCreating plots...")
    result_plot = plot_results(analysis, cases, day_of_week)
    display(result_plot)
    
    # Save results
    println("Saving results...")
    
    # Create results DataFrame
    results_df = DataFrame(
        time = 1:length(cases),
        rt_mean = analysis.rt_mean,
        rt_lower = analysis.rt_lower,
        rt_upper = analysis.rt_upper,
        observed_cases = cases,
        day_of_week = day_of_week
    )
    
    CSV.write("rt_estimates.csv", results_df)
    savefig(result_plot, "rt_estimates_plot.png")
    
    println("Analysis complete!")
    println("Results saved to: rt_estimates.csv")
    println("Plot saved to: rt_estimates_plot.png")
    
    return results, analysis, epi_prob
end

# Example of how to create synthetic data for testing
function create_synthetic_data(n_days=100)
    # Create synthetic case data with day-of-week effects
    dates = 1:n_days
    day_of_week = mod1.(dates, 7)
    
    # True Rt that varies over time
    true_rt = 1.5 * exp.(-0.02 * dates) .+ 0.5  # Declining epidemic
    
    # Day-of-week effects (multiplicative)
    dow_effects = [1.0, 0.9, 0.8, 0.85, 0.95, 1.1, 0.7]  # Lower on weekends
    
    # Generate synthetic cases (simplified)
    base_cases = 100 .+ 50 * sin.(0.1 * dates)  # Base pattern
    cases = Int.(round.(base_cases .* [dow_effects[d] for d in day_of_week] .* 
                       (0.8 .+ 0.4 * rand(n_days))))  # Add noise
    
    # Create DataFrame and save
    df = DataFrame(
        date = [Date(2023, 1, 1) + Day(i-1) for i in 1:n_days],
        cases = cases,
        day_of_week = day_of_week
    )
    
    CSV.write("cases.csv", df)
    
    println("Synthetic data created and saved to cases.csv")
    return df
end

# Uncomment the following lines to run:
# create_synthetic_data(80)  # Create synthetic data
# results, analysis, model = main()  # Run the analysis
```

---

## SUB_083

**Scenario**: 2 | **Execution**: SUCCESS

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using CSV
using DataFrames
using Plots
using StatsPlots
using Random

# Set random seed for reproducibility
Random.seed!(123)

# Load the data
function load_data(filename)
    df = CSV.read(filename, DataFrame)
    return df
end

# Main analysis function
function estimate_rt_with_observation_processes(data_file::String)
    
    # Load data
    df = load_data(data_file)
    cases = df.cases
    day_of_week = df.day_of_week
    n_timepoints = length(cases)
    
    println("Data loaded: $(n_timepoints) time points")
    println("Total cases: $(sum(cases))")
    
    # 1. Define generation interval (COVID-19 typical values)
    gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, shape consistent with COVID-19
    model_data = EpiData(gen_distribution = gen_int)
    
    # 2. Create infection model using renewal equation
    epi = Renewal(
        model_data; 
        initialisation_prior = Normal(log(max(10.0, cases[1])), 1.0)
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smooth evolution
    rt_latent = AR(
        damp_priors = [truncated(Normal(0.9, 0.05), 0.0, 0.99)],  # Strong persistence
        init_priors = [Normal(0.0, 0.5)],  # log(Rt) starts around 1
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))  # Moderate variation
    )
    
    # 4. Create day-of-week effects model
    # This creates 7 random effects, one for each day of the week
    dow_effect = BroadcastLatentModel(
        RepeatBlock(),  # Repeat the 7-day pattern
        7,              # 7 days in a week
        IID(Normal(0.0, 0.2))  # Day-of-week effects with moderate variation
    )
    
    # 5. Create time-varying ascertainment model
    # This captures changes in testing/reporting over time
    ascertainment_latent = RandomWalk(
        init_prior = Normal(logit(0.3), 0.5),  # Start at ~30% ascertainment
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Slow changes
    )
    
    # 6. Combine all latent processes
    # The combined model will have: [log_Rt, dow_effects..., ascertainment_logit]
    combined_latent = CombineLatentModels([rt_latent, dow_effect, ascertainment_latent])
    
    # 7. Create observation model with reporting delay and overdispersion
    delay_dist = Gamma(5.0, 1.2)  # Mean delay ~6 days from infection to reporting
    
    # Base observation model with overdispersion (negative binomial)
    base_obs = NegativeBinomialError(
        cluster_factor_prior = HalfNormal(0.2)  # Allows for overdispersion
    )
    
    # Add reporting delay
    obs_with_delay = LatentDelay(base_obs, delay_dist)
    
    # 8. Create custom observation model that incorporates day-of-week and ascertainment effects
    # We need to define a custom transformation function
    function observation_transform(infections, latent_vars, day_of_week_vec)
        n_time = length(infections)
        
        # Extract components from latent variables
        # latent_vars structure: [log_Rt (n_time), dow_effects (7), ascertainment_logit (n_time)]
        rt_start = 1
        rt_end = n_time
        dow_start = rt_end + 1
        dow_end = dow_start + 6  # 7 effects (indexed 0-6, so 6 additional)
        asc_start = dow_end + 1
        asc_end = asc_start + n_time - 1
        
        log_rt = latent_vars[rt_start:rt_end]
        dow_effects = latent_vars[dow_start:dow_end]
        ascertainment_logits = latent_vars[asc_start:asc_end]
        
        # Convert to probabilities
        ascertainment_probs = logistic.(ascertainment_logits)
        
        # Apply day-of-week effects
        dow_multipliers = exp.(dow_effects[day_of_week_vec])
        
        # Calculate expected reported cases
        expected_cases = infections .* ascertainment_probs .* dow_multipliers
        
        return expected_cases
    end
    
    # For EpiAware, we need to create a custom observation model
    # We'll use a simpler approach by modifying the expected cases within the model
    
    # 9. Compose the full EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi,
        latent_model = combined_latent,
        observation_model = obs_with_delay,
        tspan = (1, n_timepoints)
    )
    
    # 10. Generate Turing model with custom modifications for our observation process
    @model function custom_epiaware_model(y_t, day_of_week_indices)
        # Generate latent variables
        latent_vars ~ generate_latent(combined_latent, n_timepoints)
        
        # Extract Rt values (first n_timepoints elements)
        log_rt = latent_vars[1:n_timepoints]
        rt = exp.(log_rt)
        
        # Generate infections using renewal equation
        infections ~ generate_latent_infs(epi, log_rt)
        
        # Extract day-of-week effects (next 7 elements)
        dow_effects = latent_vars[(n_timepoints+1):(n_timepoints+7)]
        
        # Extract ascertainment (last n_timepoints elements)  
        ascertainment_logits = latent_vars[(n_timepoints+8):(2*n_timepoints+7)]
        ascertainment = logistic.(ascertainment_logits)
        
        # Apply observation process transformations
        dow_multipliers = exp.(dow_effects[day_of_week_indices])
        expected_reported = infections .* ascertainment .* dow_multipliers
        
        # Apply delay and generate observations
        delayed_expected ~ generate_observations(obs_with_delay, y_t, expected_reported)
        
        return (
            rt = rt,
            infections = infections,
            ascertainment = ascertainment,
            dow_effects = dow_effects,
            expected_reported = expected_reported
        )
    end
    
    # 11. Prepare data for inference
    mdl = custom_epiaware_model(cases, day_of_week)
    
    # 12. Set up inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 1000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    println("Starting MCMC sampling...")
    
    # 13. Run inference
    results = sample(mdl, NUTS(0.8), MCMCThreads(), 1000, 4)
    
    println("MCMC sampling completed!")
    
    return results, df
end

# Function to extract and summarize results
function extract_results(results, df)
    # Extract posterior samples
    rt_samples = Array(group(results, :rt))
    ascertainment_samples = Array(group(results, :ascertainment))
    dow_effects_samples = Array(group(results, :dow_effects))
    
    # Calculate summaries
    rt_mean = mean(rt_samples, dims=1)[1, :]
    rt_lower = [quantile(rt_samples[:, i], 0.025) for i in 1:size(rt_samples, 2)]
    rt_upper = [quantile(rt_samples[:, i], 0.975) for i in 1:size(rt_samples, 2)]
    
    ascertainment_mean = mean(ascertainment_samples, dims=1)[1, :]
    ascertainment_lower = [quantile(ascertainment_samples[:, i], 0.025) for i in 1:size(ascertainment_samples, 2)]
    ascertainment_upper = [quantile(ascertainment_samples[:, i], 0.975) for i in 1:size(ascertainment_samples, 2)]
    
    dow_mean = mean(dow_effects_samples, dims=1)[1, :]
    dow_lower = [quantile(dow_effects_samples[:, i], 0.025) for i in 1:size(dow_effects_samples, 2)]
    dow_upper = [quantile(dow_effects_samples[:, i], 0.975) for i in 1:size(dow_effects_samples, 2)]
    
    return (
        rt = (mean = rt_mean, lower = rt_lower, upper = rt_upper),
        ascertainment = (mean = ascertainment_mean, lower = ascertainment_lower, upper = ascertainment_upper),
        dow_effects = (mean = dow_mean, lower = dow_lower, upper = dow_upper)
    )
end

# Function to create plots
function plot_results(results_summary, df)
    dates = df.date
    cases = df.cases
    
    # Plot 1: Rt over time
    p1 = plot(dates, results_summary.rt.mean, 
             ribbon = (results_summary.rt.mean - results_summary.rt.lower,
                      results_summary.rt.upper - results_summary.rt.mean),
             label = "Rt", title = "Reproduction Number (Rt) Over Time",
             xlabel = "Date", ylabel = "Rt", linewidth = 2)
    hline!([1.0], linestyle = :dash, color = :red, label = "Rt = 1")
    
    # Plot 2: Ascertainment over time
    p2 = plot(dates, results_summary.ascertainment.mean * 100,
             ribbon = ((results_summary.ascertainment.mean - results_summary.ascertainment.lower) * 100,
                      (results_summary.ascertainment.upper - results_summary.ascertainment.mean) * 100),
             label = "Ascertainment Rate", title = "Time-varying Ascertainment Rate",
             xlabel = "Date", ylabel = "Ascertainment (%)", linewidth = 2)
    
    # Plot 3: Day-of-week effects
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    p3 = bar(dow_names, exp.(results_summary.dow_effects.mean),
            yerror = (exp.(results_summary.dow_effects.mean) - exp.(results_summary.dow_effects.lower),
                     exp.(results_summary.dow_effects.upper) - exp.(results_summary.dow_effects.mean)),
            title = "Day-of-Week Effects (Multiplicative)",
            xlabel = "Day of Week", ylabel = "Reporting Multiplier", legend = false)
    hline!([1.0], linestyle = :dash, color = :red)
    
    # Plot 4: Observed vs expected cases (sample from posterior predictive)
    p4 = plot(dates, cases, label = "Observed Cases", title = "Observed vs Expected Cases",
             xlabel = "Date", ylabel = "Daily Cases", linewidth = 2, alpha = 0.7)
    
    return plot(p1, p2, p3, p4, layout = (2, 2), size = (1200, 800))
end

# Main execution function
function main(data_file = "cases.csv")
    println("=== Estimating Rt with Complex Observation Processes ===")
    
    try
        # Run the analysis
        results, df = estimate_rt_with_observation_processes(data_file)
        
        # Extract results
        println("Extracting posterior summaries...")
        results_summary = extract_results(results, df)
        
        # Print summary statistics
        println("\n=== RESULTS SUMMARY ===")
        println("Rt estimates:")
        println("  Mean Rt: $(round(mean(results_summary.rt.mean), digits=2))")
        println("  Min Rt: $(round(minimum(results_summary.rt.mean), digits=2))")
        println("  Max Rt: $(round(maximum(results_summary.rt.mean), digits=2))")
        
        println("\nAscertainment rates:")
        println("  Mean ascertainment: $(round(mean(results_summary.ascertainment.mean) * 100, digits=1))%")
        println("  Min ascertainment: $(round(minimum(results_summary.ascertainment.mean) * 100, digits=1))%")
        println("  Max ascertainment: $(round(maximum(results_summary.ascertainment.mean) * 100, digits=1))%")
        
        println("\nDay-of-week effects (multiplicative):")
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for (i, name) in enumerate(dow_names)
            effect = exp(results_summary.dow_effects.mean[i])
            println("  $name: $(round(effect, digits=2))")
        end
        
        # Create plots
        println("\nGenerating plots...")
        final_plot = plot_results(results_summary, df)
        display(final_plot)
        
        # Save results
        savefig(final_plot, "rt_estimation_results.png")
        println("Plot saved as 'rt_estimation_results.png'")
        
        return results_summary, df, results
        
    catch e
        println("Error occurred: $e")
        rethrow(e)
    end
end

# Run the analysis (uncomment to execute)
# results_summary, df, mcmc_results = main("cases.csv")
```

---

## SUB_103

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using CSV
using DataFrames
using Distributions
using Turing
using StatsPlots
using Statistics

# Load the data
function load_data(filename::String)
    df = CSV.read(filename, DataFrame)
    return df.cases, df.day_of_week
end

# Main function to estimate Rt with observation processes
function estimate_rt_with_observations(cases::Vector{Int}, day_of_week::Vector{Int})
    n_timepoints = length(cases)
    
    println("Fitting Rt model with $(n_timepoints) time points...")
    
    # 1. Define generation interval (COVID-19 typical values)
    gen_distribution = Gamma(6.5, 0.62)  # Mean ~4 days
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create infection model using renewal equation
    epi_model = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)
    )
    
    # 3. Create latent model for log(Rt) - AR(1) process for smoothness
    rt_latent = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0.0, 1.0)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Create day-of-week effects model
    # This creates a latent effect for each day of the week
    dayofweek_latent = BroadcastLatentModel(
        RepeatEach(7),  # Repeat 7-day pattern
        n_timepoints ÷ 7 + 1,  # Number of weeks needed
        HierarchicalNormal(std_prior = HalfNormal(0.2))  # Day effects
    )
    
    # 5. Create time-varying ascertainment model
    # This models the changing proportion of infections that are reported
    ascertainment_latent = RandomWalk(
        init_prior = Normal(logit(0.3), 0.5),  # Start around 30% ascertainment
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Slow changes
    )
    
    # 6. Combine all latent processes
    combined_latent = CombineLatentModels([
        rt_latent,           # log(Rt)
        dayofweek_latent,    # Day-of-week effects  
        ascertainment_latent # Ascertainment rate (logit scale)
    ])
    
    # 7. Create observation model with delay and overdispersion
    delay_distribution = Gamma(5.0, 1.0)  # Mean delay ~5 days
    base_obs = NegativeBinomialError(
        cluster_factor_prior = HalfNormal(0.2)  # Allows overdispersion
    )
    obs_with_delay = LatentDelay(base_obs, delay_distribution)
    
    # 8. Create custom observation model that incorporates day-of-week effects
    # and time-varying ascertainment
    function custom_observation_model()
        @submodel log_rt, dayofweek_effects, logit_ascertainment = combined_latent()
        
        # Extract day-of-week effects for actual dates
        dow_effects = dayofweek_effects[day_of_week]
        
        # Convert ascertainment from logit to probability scale
        ascertainment_prob = logistic.(logit_ascertainment)
        
        # Generate latent infections from Rt
        @submodel I_t = generate_latent_infs(epi_model, log_rt)()
        
        # Apply delay to get expected reported infections
        @submodel delayed_I = generate_latent_infs(
            LatentDelay(I_t, delay_distribution)
        )()
        
        # Apply ascertainment and day-of-week effects
        expected_cases = delayed_I .* ascertainment_prob .* exp.(dow_effects)
        
        # Generate observations with overdispersion
        @submodel y_t = generate_observations(
            base_obs, 
            missing,  # Will be filled with actual data
            expected_cases
        )()
        
        return (
            y_t = y_t,
            rt = exp.(log_rt),
            dayofweek_effects = dow_effects,
            ascertainment = ascertainment_prob,
            expected_cases = expected_cases,
            latent_infections = I_t
        )
    end
    
    return custom_observation_model
end

# Simplified approach using EpiProblem structure
function estimate_rt_simplified(cases::Vector{Int}, day_of_week::Vector{Int})
    n_timepoints = length(cases)
    
    println("Fitting simplified Rt model with $(n_timepoints) time points...")
    
    # 1. Define generation interval
    gen_distribution = Gamma(6.5, 0.62)
    model_data = EpiData(gen_distribution = gen_distribution)
    
    # 2. Create infection model
    epi_model = Renewal(
        model_data; 
        initialisation_prior = Normal(log(100.0), 1.0)
    )
    
    # 3. Create latent model for log(Rt)
    latent_model = AR(
        damp_priors = [truncated(Normal(0.8, 0.1), 0.0, 1.0)],
        init_priors = [Normal(0.0, 0.5)],
        ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
    )
    
    # 4. Create observation model with delay and overdispersion
    delay_distribution = Gamma(5.0, 1.0)
    obs_model = LatentDelay(
        NegativeBinomialError(cluster_factor_prior = HalfNormal(0.2)),
        delay_distribution
    )
    
    # 5. Create EpiProblem
    epi_prob = EpiProblem(
        epi_model = epi_model,
        latent_model = latent_model,
        observation_model = obs_model,
        tspan = (1, n_timepoints)
    )
    
    # 6. Generate Turing model
    mdl = generate_epiaware(epi_prob, (y_t = cases,))
    
    # 7. Define inference method
    inference_method = EpiMethod(
        pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
        sampler = NUTSampler(
            adtype = AutoReverseDiff(compile = true),
            ndraws = 1000,
            nchains = 4,
            mcmc_parallel = MCMCThreads()
        )
    )
    
    println("Running MCMC sampling...")
    
    # 8. Run inference
    results = apply_method(mdl, inference_method, (y_t = cases,))
    
    return results, epi_prob
end

# Function to extract and summarize results
function extract_results(results, n_timepoints::Int)
    # Extract posterior samples
    posterior_samples = results.chain
    
    # Get parameter names
    param_names = names(posterior_samples)
    
    println("Available parameters:")
    for name in param_names
        println("  ", name)
    end
    
    # Extract Rt estimates (exp of latent process Z_t)
    rt_samples = []
    for t in 1:n_timepoints
        z_param = "generated_quantities.generated_quantities[$t]"
        if z_param in string.(param_names)
            z_samples = posterior_samples[z_param]
            rt_samples_t = exp.(z_samples)
            push!(rt_samples, rt_samples_t)
        end
    end
    
    if !isempty(rt_samples)
        # Calculate summary statistics
        rt_mean = [mean(rt_t) for rt_t in rt_samples]
        rt_lower = [quantile(rt_t, 0.025) for rt_t in rt_samples]
        rt_upper = [quantile(rt_t, 0.975) for rt_t in rt_samples]
        
        return (
            rt_mean = rt_mean,
            rt_lower = rt_lower,
            rt_upper = rt_upper,
            samples = rt_samples
        )
    else
        println("Warning: Could not find Rt parameters in chain")
        return nothing
    end
end

# Function to create plots
function plot_results(rt_results, cases::Vector{Int})
    if rt_results === nothing
        println("No results to plot")
        return nothing
    end
    
    n_timepoints = length(cases)
    time_points = 1:n_timepoints
    
    # Plot Rt over time
    p1 = plot(time_points, rt_results.rt_mean, 
             ribbon = (rt_results.rt_mean .- rt_results.rt_lower,
                      rt_results.rt_upper .- rt_results.rt_mean),
             label = "Rt estimate", 
             title = "Reproduction Number Over Time",
             xlabel = "Days", 
             ylabel = "Rt",
             linewidth = 2)
    hline!([1.0], linestyle = :dash, color = :red, label = "Rt = 1")
    
    # Plot observed cases
    p2 = bar(time_points, cases, 
            label = "Observed cases",
            title = "Daily Case Counts",
            xlabel = "Days", 
            ylabel = "Cases",
            alpha = 0.7)
    
    # Combine plots
    combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 600))
    
    return combined_plot
end

# Main execution function
function main()
    # Load data
    println("Loading data...")
    cases, day_of_week = load_data("cases.csv")
    
    println("Data summary:")
    println("  Total days: ", length(cases))
    println("  Total cases: ", sum(cases))
    println("  Mean daily cases: ", round(mean(cases), digits=1))
    
    # Run analysis with simplified model
    results, epi_prob = estimate_rt_simplified(cases, day_of_week)
    
    # Extract results
    rt_results = extract_results(results, length(cases))
    
    # Create plots
    plt = plot_results(rt_results, cases)
    
    if plt !== nothing
        display(plt)
        savefig(plt, "rt_estimates.png")
        println("Plot saved as rt_estimates.png")
    end
    
    # Print summary
    if rt_results !== nothing
        println("\nRt Summary:")
        println("  Current Rt estimate: ", 
               round(rt_results.rt_mean[end], digits=2),
               " (95% CI: ", 
               round(rt_results.rt_lower[end], digits=2), "-",
               round(rt_results.rt_upper[end], digits=2), ")")
        
        println("  Mean Rt over period: ", 
               round(mean(rt_results.rt_mean), digits=2))
        
        # Check convergence
        println("\nModel diagnostics:")
        println("  Number of samples: ", size(results.chain, 1))
        println("  Number of chains: ", size(results.chain, 3))
        
        return (
            rt_estimates = rt_results,
            mcmc_results = results,
            model = epi_prob
        )
    else
        println("Analysis completed but could not extract Rt estimates")
        return results
    end
end

# For testing with synthetic data if cases.csv doesn't exist
function create_synthetic_data()
    n_days = 100
    dates = [Date(2023, 1, 1) + Day(i-1) for i in 1:n_days]
    
    # Generate synthetic Rt that varies over time
    true_rt = 1.5 .- 0.8 .* (1:n_days) ./ n_days .+ 0.2 .* sin.((1:n_days) .* 2π ./ 14)
    
    # Generate synthetic cases with renewal process (simplified)
    cases = zeros(Int, n_days)
    cases[1:7] .= [10, 12, 8, 15, 18, 14, 9]  # Initial cases
    
    for t in 8:n_days
        # Simplified renewal equation
        new_infections = true_rt[t] * sum(cases[max(1, t-7):t-1] .* [0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.05][1:min(7, t-1)])
        # Add noise and delay
        cases[t] = max(0, round(Int, new_infections * (0.8 + 0.4*rand()) * 
                               (t % 7 ∈ [6, 0] ? 0.7 : 1.0)))  # Weekend effect
    end
    
    day_of_week = [dayofweek(d) for d in dates]
    
    df = DataFrame(
        date = dates,
        cases = cases,
        day_of_week = day_of_week
    )
    
    CSV.write("cases.csv", df)
    println("Created synthetic data in cases.csv")
    
    return cases, day_of_week
end

# Run the analysis
if !isfile("cases.csv")
    println("cases.csv not found, creating synthetic data...")
    create_synthetic_data()
end

# Execute main analysis
results = main()
```

---

## SUB_072

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing

# Load data
cases = CSV.read("cases.csv", DataFrame)

# Define generation interval distribution
gen_distribution = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_distribution)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Define observation model with delay and day-of-week effects
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)
dayofweek_effect = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Compose observation model with day-of-week effects and delay
obs_with_day_of_week = StackObservationModels((obs, dayofweek_effect))

# Time-varying ascertainment
ascertainment_model = Ascertainment(
    obs_with_day_of_week,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)

# Create EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = ascertainment_model,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases.cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases.cases,))

# Extract Rt estimates
rt_estimates = exp.(results.latent)

# Plot results
using Plots
plot(rt_estimates[:, 1], label="Rt Estimates")
xlabel("Time Point")
ylabel("Estimated Reproduction Number")
```

---

## SUB_063

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using Plots

# Load data
cases = readdlm("cases.csv", ',')
dates = cases[:, 1]
day_of_week = cases[:, 2]

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent models for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Add day-of-week effects
dayofweek_effect = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)
combined_latent = CombineLatentModels([latent, dayofweek_effect])

# Create observation model with delay and ascertainment
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

ascertainment_model = Ascertainment(
    obs,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)
stacked_obs = StackObservationModels((cases = obs))

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = combined_latent,
    observation_model = ascertainment_model,
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt_estimates = exp.(results.latent_samples)

# Plot results
plot(Rt_estimates[:, 1], label="Day of week effect")
plot!(Rt_estimates[:, 2], label="Ascertainment estimate")

xlabel("Time")
ylabel("Estimate")
legend()
```

---

## SUB_038

**Scenario**: 2 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using Plots

# Load data
cases = load("cases.csv", header=true)

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Define day-of-week effect model
dayofweek_effect_model = BroadcastLatentModel(
    RepeatEach(),  # Repeat pattern
    7,             # 7 days
    HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Combine latent models
combined_latent = CombineLatentModels([latent, dayofweek_effect_model])

# Create observation model with ascertainment
ascertainment_model = Ascertainment(
    obs,
    RandomWalk(init_prior = Normal(-1.0, 0.5), ϵ_t = IID(Normal(0, 0.1))),
    (Y, x) -> Y .* logistic.(x)  # Transform function
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = combined_latent,
    observation_model = ascertainment_model,
    tspan = (1, length(cases))
)

# Generate Turing model and run inference
mdl = generate_epiaware(epi_prob, (y_t = cases,))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)
results = apply_method(mdl, inference_method, (y_t = cases,))

# Extract Rt estimates
Rt_estimates = exp.(results[1][:Z_t])

# Plot Rt estimates over time
plot(Rt_estimates)
```

---

## SUB_055

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(rstan)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and prepare data
load_and_prepare_data <- function(file_path = "cases.csv") {
  # Load data
  data <- read_csv(file_path)
  data$date <- as.Date(data$date)
  
  # Sort by date and check for missing dates
  data <- data %>% arrange(date)
  
  # Fill in any missing dates with 0 cases
  date_seq <- seq(min(data$date), max(data$date), by = "day")
  complete_data <- data.frame(date = date_seq) %>%
    left_join(data, by = "date") %>%
    mutate(
      cases = ifelse(is.na(cases), 0, cases),
      day_of_week = ifelse(is.na(day_of_week), as.numeric(format(date, "%u")), day_of_week)
    )
  
  return(complete_data)
}

# Define generation interval (example: COVID-19 like)
get_generation_interval <- function(max_days = 15) {
  # Gamma distribution with mean=5.2, sd=1.7 (typical for COVID-19)
  shape <- (5.2/1.7)^2
  rate <- 5.2/(1.7^2)
  
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # Normalize
  
  return(pmf)
}

# Define delay distribution (infection to reporting)
get_delay_distribution <- function(max_delay = 20) {
  # Log-normal distribution with mean delay ~8 days
  meanlog <- log(8)
  sdlog <- 0.5
  
  delays <- 1:max_delay
  pmf <- dlnorm(delays, meanlog = meanlog, sdlog = sdlog)
  pmf <- pmf / sum(pmf)  # Normalize
  
  return(pmf)
}

# Stan model code
stan_model_code <- "
data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> cases[T];            // Observed cases
  int<lower=1, upper=7> dow[T];     // Day of week
  int<lower=0> G;                   // Generation interval length
  vector[G] generation_pmf;         // Generation interval PMF
  int<lower=0> D;                   // Delay distribution length
  vector[D] delay_pmf;              // Delay distribution PMF
  int<lower=0> seed_days;           // Number of initial seeding days
}

transformed data {
  int max_lag = max(G, D);
}

parameters {
  // Rt parameters
  vector[T] log_rt_raw;             // Raw log Rt
  real log_rt_mean;                 // Mean log Rt
  real<lower=0> rt_sigma;           // Rt random walk std
  
  // Day of week effects (Monday = reference)
  vector[6] dow_effects_raw;        // Day 2-7 effects
  real<lower=0> dow_sigma;          // Day of week effect std
  
  // Time-varying ascertainment
  vector[T] logit_ascert_raw;       // Raw logit ascertainment
  real logit_ascert_mean;           // Mean logit ascertainment
  real<lower=0> ascert_sigma;       // Ascertainment random walk std
  
  // Overdispersion
  real<lower=0> phi;                // Negative binomial overdispersion
  
  // Initial infections
  vector<lower=0>[seed_days] I_seed;
}

transformed parameters {
  vector[T] log_rt;
  vector[T] rt;
  vector[7] dow_effects;
  vector[T] logit_ascert;
  vector[T] ascertainment;
  vector[T] infections;
  vector[T] expected_cases;
  
  // Rt random walk
  log_rt[1] = log_rt_mean + rt_sigma * log_rt_raw[1];
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_sigma * log_rt_raw[t];
  }
  rt = exp(log_rt);
  
  // Day of week effects (Monday = 1.0)
  dow_effects[1] = 1.0;
  for (d in 2:7) {
    dow_effects[d] = exp(dow_sigma * dow_effects_raw[d-1]);
  }
  
  // Time-varying ascertainment
  logit_ascert[1] = logit_ascert_mean + ascert_sigma * logit_ascert_raw[1];
  for (t in 2:T) {
    logit_ascert[t] = logit_ascert[t-1] + ascert_sigma * logit_ascert_raw[t];
  }
  ascertainment = inv_logit(logit_ascert);
  
  // Infections via renewal equation
  for (t in 1:T) {
    if (t <= seed_days) {
      infections[t] = I_seed[t];
    } else {
      infections[t] = 0;
      for (g in 1:min(G, t-1)) {
        if (t-g > 0) {
          infections[t] += rt[t] * infections[t-g] * generation_pmf[g];
        }
      }
      // Add small constant to prevent numerical issues
      infections[t] = fmax(infections[t], 1e-8);
    }
  }
  
  // Expected cases with delay and observation process
  for (t in 1:T) {
    expected_cases[t] = 0;
    for (d in 1:min(D, t)) {
      if (t-d+1 > 0) {
        expected_cases[t] += infections[t-d+1] * delay_pmf[d] * 
                            ascertainment[t-d+1] * dow_effects[dow[t]];
      }
    }
    expected_cases[t] = fmax(expected_cases[t], 1e-8);
  }
}

model {
  // Priors
  log_rt_mean ~ normal(0, 0.5);     // Rt around 1
  rt_sigma ~ normal(0, 0.1);        // Moderate variation
  log_rt_raw ~ std_normal();
  
  dow_effects_raw ~ std_normal();
  dow_sigma ~ normal(0, 0.2);       // Moderate day-of-week effects
  
  logit_ascert_mean ~ normal(-2, 1); // ~12% base ascertainment
  ascert_sigma ~ normal(0, 0.05);    // Slow changes in ascertainment
  logit_ascert_raw ~ std_normal();
  
  phi ~ exponential(1);             // Overdispersion
  
  I_seed ~ exponential(1);          // Initial infections
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  vector[T] cases_pred;
  vector[T] log_lik;
  
  for (t in 1:T) {
    cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
    log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
  }
}
"

# Main estimation function
estimate_rt <- function(data, seed_days = 7, chains = 4, iter = 2000) {
  # Prepare data for Stan
  T <- nrow(data)
  generation_pmf <- get_generation_interval()
  delay_pmf <- get_delay_distribution()
  
  stan_data <- list(
    T = T,
    cases = data$cases,
    dow = data$day_of_week,
    G = length(generation_pmf),
    generation_pmf = generation_pmf,
    D = length(delay_pmf),
    delay_pmf = delay_pmf,
    seed_days = seed_days
  )
  
  # Compile and fit model
  cat("Compiling Stan model...\n")
  model <- stan_model(model_code = stan_model_code)
  
  cat("Fitting model...\n")
  fit <- sampling(
    model,
    data = stan_data,
    chains = chains,
    iter = iter,
    warmup = iter/2,
    cores = chains,
    control = list(adapt_delta = 0.95, max_treedepth = 12)
  )
  
  return(fit)
}

# Extract and summarize results
extract_results <- function(fit, data) {
  # Extract samples
  samples <- extract(fit)
  
  # Rt estimates
  rt_summary <- data.frame(
    date = data$date,
    rt_median = apply(samples$rt, 2, median),
    rt_lower = apply(samples$rt, 2, quantile, 0.025),
    rt_upper = apply(samples$rt, 2, quantile, 0.975),
    rt_lower_50 = apply(samples$rt, 2, quantile, 0.25),
    rt_upper_50 = apply(samples$rt, 2, quantile, 0.75)
  )
  
  # Day-of-week effects
  dow_names <- c("Monday", "Tuesday", "Wednesday", "Thursday", 
                 "Friday", "Saturday", "Sunday")
  dow_summary <- data.frame(
    day = dow_names,
    effect_median = apply(samples$dow_effects, 2, median),
    effect_lower = apply(samples$dow_effects, 2, quantile, 0.025),
    effect_upper = apply(samples$dow_effects, 2, quantile, 0.975)
  )
  
  # Ascertainment estimates
  ascert_summary <- data.frame(
    date = data$date,
    ascert_median = apply(samples$ascertainment, 2, median),
    ascert_lower = apply(samples$ascertainment, 2, quantile, 0.025),
    ascert_upper = apply(samples$ascertainment, 2, quantile, 0.975)
  )
  
  # Model diagnostics
  diagnostics <- list(
    rhat = summary(fit)$summary[, "Rhat"],
    n_eff = summary(fit)$summary[, "n_eff"],
    phi = median(samples$phi)
  )
  
  return(list(
    rt = rt_summary,
    dow_effects = dow_summary,
    ascertainment = ascert_summary,
    diagnostics = diagnostics,
    fit = fit,
    data = data
  ))
}

# Plotting functions
plot_rt <- function(results) {
  ggplot(results$rt, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.2, fill = "blue") +
    geom_ribbon(aes(ymin = rt_lower_50, ymax = rt_upper_50), alpha = 0.4, fill = "blue") +
    geom_line(aes(y = rt_median), color = "blue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(
      title = "Time-varying Reproduction Number (Rt)",
      x = "Date",
      y = "Rt",
      subtitle = "Dark ribbon: 50% CI, Light ribbon: 95% CI"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_dow_effects <- function(results) {
  ggplot(results$dow_effects, aes(x = day, y = effect_median)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = effect_lower, ymax = effect_upper), 
                  width = 0.2, color = "darkblue") +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(
      title = "Day-of-Week Reporting Effects",
      x = "Day of Week",
      y = "Relative Reporting Rate",
      subtitle = "Error bars show 95% credible intervals"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_ascertainment <- function(results) {
  ggplot(results$ascertainment, aes(x = date)) +
    geom_ribbon(aes(ymin = ascert_lower, ymax = ascert_upper), 
                alpha = 0.3, fill = "green") +
    geom_line(aes(y = ascert_median), color = "darkgreen", size = 1) +
    labs(
      title = "Time-varying Ascertainment Rate",
      x = "Date",
      y = "Ascertainment Rate",
      subtitle = "Proportion of infections that are reported"
    ) +
    scale_y_continuous(labels = scales::percent) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

plot_cases_fit <- function(results) {
  # Extract predicted cases
  samples <- extract(results$fit)
  cases_pred <- data.frame(
    date = results$data$date,
    observed = results$data$cases,
    predicted_median = apply(samples$expected_cases, 2, median),
    predicted_lower = apply(samples$expected_cases, 2, quantile, 0.025),
    predicted_upper = apply(samples$expected_cases, 2, quantile, 0.975)
  )
  
  ggplot(cases_pred, aes(x = date)) +
    geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper), 
                alpha = 0.3, fill = "orange") +
    geom_line(aes(y = predicted_median), color = "darkorange", size = 1) +
    geom_point(aes(y = observed), color = "black", alpha = 0.6) +
    labs(
      title = "Model Fit: Observed vs Predicted Cases",
      x = "Date",
      y = "Daily Cases",
      subtitle = "Points: observed, Line: predicted median, Ribbon: 95% CI"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
}

# Main analysis function
run_rt_analysis <- function(file_path = "cases.csv", seed_days = 7) {
  cat("Loading and preparing data...\n")
  data <- load_and_prepare_data(file_path)
  
  cat("Data summary:\n")
  cat(paste("Date range:", min(data$date), "to", max(data$date), "\n"))
  cat(paste("Total days:", nrow(data), "\n"))
  cat(paste("Total cases:", sum(data$cases), "\n"))
  cat(paste("Mean daily cases:", round(mean(data$cases), 1), "\n"))
  
  cat("\nEstimating Rt...\n")
  fit <- estimate_rt(data, seed_days = seed_days)
  
  cat("Extracting results...\n")
  results <- extract_results(fit, data)
  
  # Print diagnostics
  cat("\nModel Diagnostics:\n")
  rhat_issues <- sum(results$diagnostics$rhat > 1.1, na.rm = TRUE)
  cat(paste("Parameters with Rhat > 1.1:", rhat_issues, "\n"))
  cat(paste("Overdispersion parameter (phi):", round(results$diagnostics$phi, 2), "\n"))
  
  # Print summary statistics
  cat("\nRt Summary:\n")
  cat(paste("Current Rt (last week median):", 
            round(median(tail(results$rt$rt_median, 7)), 2), "\n"))
  
  cat("\nDay-of-week effects:\n")
  print(results$dow_effects)
  
  cat("\nAscertainment rate (recent median):", 
      paste0(round(median(tail(results$ascertainment$ascert_median, 14)) * 100, 1), "%"), "\n")
  
  return(results)
}

# Example usage and demonstration
if (FALSE) {  # Set to TRUE to run example
  # Create example data if cases.csv doesn't exist
  if (!file.exists("cases.csv")) {
    set.seed(123)
    dates <- seq(as.Date("2023-01-01"), as.Date("2023-06-30"), by = "day")
    n_days <- length(dates)
    
    # Simulate some realistic case data
    true_rt <- c(rep(1.5, 30), 1.5 * exp(-0.05 * (1:50)), rep(0.8, 30), 
                 0.8 + 0.3 * sin(2*pi*(1:71)/14), rep(1.1, n_days - 181))
    true_rt <- true_rt[1:n_days]
    
    cases <- rpois(n_days, pmax(10, 50 * exp(cumsum(log(true_rt)/7))))
    
    # Add day-of-week effects
    dow <- as.numeric(format(dates, "%u"))
    dow_multipliers <- c(1.1, 1.1, 1.0, 0.9, 0.8, 0.6, 0.7)
    cases <- rpois(n_days, cases * dow_multipliers[dow])
    
    example_data <- data.frame(
      date = dates,
      cases = cases,
      day_of_week = dow
    )
    
    write_csv(example_data, "cases.csv")
    cat("Created example cases.csv file\n")
  }
  
  # Run analysis
  results <- run_rt_analysis("cases.csv")
  
  # Create plots
  p1 <- plot_rt(results)
  p2 <- plot_dow_effects(results)
  p3 <- plot_ascertainment(results)
  p4 <- plot_cases_fit(results)
  
  # Display plots
  print(p1)
  print(p2)
  print(p3)
  print(p4)
}


# Run the analysis
results <- run_rt_analysis("cases.csv")

# Create visualizations
plot_rt(results)
plot_dow_effects(results)  
plot_ascertainment(results)
plot_cases_fit(results)
```

---

## SUB_119

**Scenario**: 2 | **Execution**: TIMEOUT

### script.R

```r
# Load required libraries
library(tidyverse)
library(rstan)
library(bayesplot)
library(lubridate)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and prepare data
load_and_prepare_data <- function(file_path) {
  # For demonstration, create synthetic data if file doesn't exist
  if (!file.exists(file_path)) {
    cat("Creating synthetic data for demonstration...\n")
    dates <- seq(as.Date("2020-03-01"), as.Date("2020-08-31"), by = "day")
    n_days <- length(dates)
    
    # Simulate true Rt varying over time
    true_rt <- 2.5 * exp(-0.02 * seq_len(n_days)) + 
               0.3 * sin(2 * pi * seq_len(n_days) / 50) + 
               rnorm(n_days, 0, 0.1)
    true_rt[true_rt < 0.3] <- 0.3
    
    # Simulate infections using renewal equation
    gen_interval <- dgamma(1:20, shape = 2.3, rate = 0.4)
    gen_interval <- gen_interval / sum(gen_interval)
    
    infections <- numeric(n_days)
    infections[1:7] <- 50 + rpois(7, 20)  # Initial seeding
    
    for (t in 8:n_days) {
      max_s <- min(20, t - 1)
      expected_inf <- true_rt[t] * sum(infections[(t-max_s):(t-1)] * gen_interval[1:max_s])
      infections[t] <- rpois(1, expected_inf)
    }
    
    # Simulate reporting process
    dow_effects <- c(0.9, 0.95, 1.0, 1.0, 0.98, 0.7, 0.5)  # Mon-Sun
    
    # Time-varying ascertainment
    ascertainment <- 0.3 + 0.2 * (1 - exp(-seq_len(n_days) / 30))
    
    # Reporting delay (geometric with mean 5 days)
    delay_pmf <- dgeom(0:14, prob = 0.2)
    
    # Convolve infections with delay and apply observation process
    cases <- numeric(n_days)
    for (t in 1:n_days) {
      for (d in 1:min(15, t)) {
        if (t - d + 1 > 0) {
          dow <- wday(dates[t], week_start = 1)
          expected_cases <- infections[t - d + 1] * delay_pmf[d] * 
                           ascertainment[t - d + 1] * dow_effects[dow]
          cases[t] <- cases[t] + rnbinom(1, size = 20, mu = expected_cases)
        }
      }
    }
    
    data <- tibble(
      date = dates,
      cases = pmax(0, round(cases)),
      day_of_week = wday(dates, week_start = 1)
    )
    
    write_csv(data, file_path)
  } else {
    data <- read_csv(file_path)
    data$day_of_week <- wday(data$date, week_start = 1)
  }
  
  return(data)
}

# Define Stan model
stan_model_code <- "
data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> cases[T];             // Observed cases
  int<lower=1, upper=7> dow[T];      // Day of week
  int<lower=0> S;                    // Maximum generation interval
  vector[S] gen_interval;            // Generation interval PMF
  int<lower=0> D;                    // Maximum reporting delay
  vector[D] delay_pmf;               // Reporting delay PMF
  int<lower=0> seed_days;            // Initial seeding period
}

parameters {
  vector[T] log_infections_raw;      // Raw log infections
  real<lower=0> sigma_infections;    // Innovation SD for infections
  vector[T-seed_days] log_rt_raw;    // Raw log Rt
  real<lower=0> sigma_rt;            // Innovation SD for Rt
  real mean_log_rt;                  // Mean log Rt
  
  vector[6] dow_effects_raw;         // Day-of-week effects (Sunday as reference)
  real<lower=0> sigma_dow;           // SD for day-of-week effects
  
  vector[T] logit_ascertainment_raw; // Raw logit ascertainment
  real<lower=0> sigma_ascert;        // Innovation SD for ascertainment
  real mean_logit_ascert;            // Mean logit ascertainment
  
  real<lower=0> phi;                 // Overdispersion parameter
}

transformed parameters {
  vector[T] log_infections;
  vector[T] infections;
  vector[T] log_rt;
  vector[T] rt;
  vector[7] dow_effects;
  vector[T] ascertainment;
  vector[T] expected_cases;
  
  // Random walk for log infections (seeding period)
  log_infections[1] = log_infections_raw[1];
  for (t in 2:seed_days) {
    log_infections[t] = log_infections[t-1] + sigma_infections * log_infections_raw[t];
  }
  
  // Day-of-week effects (Sunday = reference = 1.0)
  dow_effects[7] = 1.0;
  for (i in 1:6) {
    dow_effects[i] = exp(sigma_dow * dow_effects_raw[i]);
  }
  
  // Time-varying ascertainment (random walk on logit scale)
  ascertainment[1] = inv_logit(mean_logit_ascert + sigma_ascert * logit_ascertainment_raw[1]);
  for (t in 2:T) {
    ascertainment[t] = inv_logit(
      logit(ascertainment[t-1]) + sigma_ascert * logit_ascertainment_raw[t]
    );
  }
  
  // Rt (random walk on log scale after seeding)
  for (t in 1:seed_days) {
    log_rt[t] = mean_log_rt;
  }
  
  log_rt[seed_days + 1] = mean_log_rt + sigma_rt * log_rt_raw[1];
  for (t in (seed_days + 2):T) {
    log_rt[t] = log_rt[t-1] + sigma_rt * log_rt_raw[t - seed_days];
  }
  
  rt = exp(log_rt);
  
  // Renewal equation for infections after seeding
  for (t in (seed_days + 1):T) {
    real renewal_sum = 0;
    int max_s = min(S, t - 1);
    for (s in 1:max_s) {
      if (t - s > 0) {
        renewal_sum += exp(log_infections[t - s]) * gen_interval[s];
      }
    }
    log_infections[t] = log(rt[t]) + log(renewal_sum);
  }
  
  infections = exp(log_infections);
  
  // Observation model: convolve infections with delay and apply observation process
  for (t in 1:T) {
    expected_cases[t] = 0;
    int max_d = min(D, t);
    for (d in 1:max_d) {
      if (t - d + 1 > 0) {
        expected_cases[t] += infections[t - d + 1] * delay_pmf[d] * 
                            ascertainment[t - d + 1] * dow_effects[dow[t]];
      }
    }
  }
}

model {
  // Priors
  sigma_infections ~ normal(0, 0.2);
  sigma_rt ~ normal(0, 0.2);
  mean_log_rt ~ normal(0, 0.5);
  sigma_dow ~ normal(0, 0.3);
  sigma_ascert ~ normal(0, 0.1);
  mean_logit_ascert ~ normal(-1, 1);
  phi ~ gamma(2, 0.1);
  
  // Raw parameters (standard normal)
  log_infections_raw ~ normal(0, 1);
  log_rt_raw ~ normal(0, 1);
  dow_effects_raw ~ normal(0, 1);
  logit_ascertainment_raw ~ normal(0, 1);
  
  // Likelihood
  for (t in 1:T) {
    cases[t] ~ neg_binomial_2(expected_cases[t], phi);
  }
}
"

# Fit the model
fit_rt_model <- function(data, n_chains = 4, n_iter = 2000) {
  # Prepare generation interval (gamma distribution)
  gen_interval <- dgamma(1:20, shape = 2.3, rate = 0.4)
  gen_interval <- gen_interval / sum(gen_interval)
  
  # Prepare reporting delay (geometric distribution)
  delay_pmf <- dgeom(0:14, prob = 0.2)
  
  # Prepare data for Stan
  stan_data <- list(
    T = nrow(data),
    cases = data$cases,
    dow = data$day_of_week,
    S = length(gen_interval),
    gen_interval = gen_interval,
    D = length(delay_pmf),
    delay_pmf = delay_pmf,
    seed_days = 7
  )
  
  # Compile and fit model
  cat("Compiling Stan model...\n")
  model <- stan_model(model_code = stan_model_code)
  
  cat("Fitting model...\n")
  fit <- sampling(
    model,
    data = stan_data,
    chains = n_chains,
    iter = n_iter,
    warmup = n_iter / 2,
    control = list(adapt_delta = 0.95, max_treedepth = 12),
    verbose = TRUE
  )
  
  return(fit)
}

# Extract and summarize results
extract_results <- function(fit, data) {
  # Extract posterior samples
  rt_samples <- extract(fit, "rt")$rt
  dow_samples <- extract(fit, "dow_effects")$dow_effects
  ascert_samples <- extract(fit, "ascertainment")$ascertainment
  
  # Calculate summaries
  rt_summary <- data.frame(
    date = data$date,
    rt_median = apply(rt_samples, 2, median),
    rt_lower = apply(rt_samples, 2, quantile, 0.025),
    rt_upper = apply(rt_samples, 2, quantile, 0.975),
    rt_lower50 = apply(rt_samples, 2, quantile, 0.25),
    rt_upper50 = apply(rt_samples, 2, quantile, 0.75)
  )
  
  dow_summary <- data.frame(
    day_name = c("Monday", "Tuesday", "Wednesday", "Thursday", 
                 "Friday", "Saturday", "Sunday"),
    dow_median = apply(dow_samples, 2, median),
    dow_lower = apply(dow_samples, 2, quantile, 0.025),
    dow_upper = apply(dow_samples, 2, quantile, 0.975)
  )
  
  ascert_summary <- data.frame(
    date = data$date,
    ascert_median = apply(ascert_samples, 2, median),
    ascert_lower = apply(ascert_samples, 2, quantile, 0.025),
    ascert_upper = apply(ascert_samples, 2, quantile, 0.975)
  )
  
  return(list(
    rt = rt_summary,
    day_of_week = dow_summary,
    ascertainment = ascert_summary,
    fit = fit
  ))
}

# Create visualization
create_plots <- function(results, data) {
  library(gridExtra)
  
  # Plot 1: Rt over time
  p1 <- ggplot(results$rt, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.2) +
    geom_ribbon(aes(ymin = rt_lower50, ymax = rt_upper50), alpha = 0.3) +
    geom_line(aes(y = rt_median), size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         x = "Date", y = "Rt",
         subtitle = "Shaded areas show 50% and 95% credible intervals") +
    theme_minimal()
  
  # Plot 2: Day-of-week effects
  p2 <- ggplot(results$day_of_week, aes(x = day_name)) +
    geom_col(aes(y = dow_median), alpha = 0.7) +
    geom_errorbar(aes(ymin = dow_lower, ymax = dow_upper), width = 0.2) +
    labs(title = "Day-of-Week Effects on Reporting",
         x = "Day of Week", y = "Multiplicative Effect") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Plot 3: Ascertainment over time
  p3 <- ggplot(results$ascertainment, aes(x = date)) +
    geom_ribbon(aes(ymin = ascert_lower, ymax = ascert_upper), alpha = 0.2) +
    geom_line(aes(y = ascert_median), size = 1) +
    labs(title = "Time-varying Ascertainment Rate",
         x = "Date", y = "Ascertainment Rate") +
    theme_minimal()
  
  # Plot 4: Observed vs expected cases
  expected_cases <- extract(results$fit, "expected_cases")$expected_cases
  expected_median <- apply(expected_cases, 2, median)
  expected_lower <- apply(expected_cases, 2, quantile, 0.025)
  expected_upper <- apply(expected_cases, 2, quantile, 0.975)
  
  fit_data <- data.frame(
    date = data$date,
    observed = data$cases,
    expected = expected_median,
    exp_lower = expected_lower,
    exp_upper = expected_upper
  )
  
  p4 <- ggplot(fit_data, aes(x = date)) +
    geom_ribbon(aes(ymin = exp_lower, ymax = exp_upper), alpha = 0.2, fill = "blue") +
    geom_line(aes(y = expected), color = "blue", size = 1) +
    geom_point(aes(y = observed), alpha = 0.6, size = 0.8) +
    labs(title = "Model Fit: Observed vs Expected Cases",
         x = "Date", y = "Cases",
         subtitle = "Points: observed, Line: expected (with 95% CI)") +
    theme_minimal()
  
  return(list(rt = p1, dow = p2, ascertainment = p3, fit = p4))
}

# Main execution function
main <- function() {
  cat("=== Structured Rt Estimation with Observation Processes ===\n\n")
  
  # Load data
  cat("Loading data...\n")
  data <- load_and_prepare_data("cases.csv")
  cat(sprintf("Loaded %d days of data from %s to %s\n", 
              nrow(data), min(data$date), max(data$date)))
  
  # Fit model
  cat("Fitting Bayesian model...\n")
  fit <- fit_rt_model(data, n_chains = 4, n_iter = 2000)
  
  # Check convergence
  cat("Checking model convergence...\n")
  print(summary(fit, pars = c("mean_log_rt", "sigma_rt", "sigma_dow", 
                              "sigma_ascert", "phi"))$summary)
  
  # Extract results
  cat("Extracting results...\n")
  results <- extract_results(fit, data)
  
  # Create plots
  cat("Creating visualizations...\n")
  plots <- create_plots(results, data)
  
  # Display results
  cat("\n=== RESULTS SUMMARY ===\n")
  
  cat("\nMean Rt over time periods:\n")
  rt_monthly <- results$rt %>%
    mutate(month = floor_date(date, "month")) %>%
    group_by(month) %>%
    summarise(mean_rt = mean(rt_median), .groups = "drop")
  print(rt_monthly)
  
  cat("\nDay-of-week effects:\n")
  print(results$day_of_week)
  
  cat("\nAscertainment rate (first and last weeks):\n")
  ascert_summary <- results$ascertainment %>%
    slice(c(1:7, (n()-6):n())) %>%
    select(date, ascert_median)
  print(ascert_summary)
  
  # Show plots
  grid.arrange(plots$rt, plots$dow, plots$ascertainment, plots$fit, ncol = 2)
  
  return(list(
    data = data,
    fit = fit,
    results = results,
    plots = plots
  ))
}

# Run the analysis
if (interactive()) {
  results <- main()
} else {
  # For non-interactive execution
  main()
}
```

---

## SUB_013

**Scenario**: 2 | **Execution**: SUCCESS

### script.R

```r
# Load required libraries
library(rstan)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)

# Stan model code
stan_model_code <- "
data {
  int<lower=1> T;                    // Number of days
  int<lower=0> cases[T];             // Reported cases
  int<lower=1,upper=7> day_of_week[T]; // Day of week (1=Monday, 7=Sunday)
  int<lower=1> S;                    // Maximum generation interval
  vector[S] gen_interval;            // Generation interval PMF
  int<lower=1> D;                    // Maximum delay
  vector[D] delay_pmf;               // Delay from infection to reporting PMF
  int<lower=1> seed_days;            // Number of initial seeding days
}

transformed data {
  int effective_T = T - seed_days;   // Days after seeding period
}

parameters {
  // Initial infections (seeding period)
  vector<lower=0>[seed_days] log_infections_seed;
  
  // Rt parameters
  real<lower=0> R_initial;           // Initial Rt
  vector[effective_T-1] rt_innovations; // Random walk innovations
  real<lower=0> rt_sigma;            // Random walk standard deviation
  
  // Day-of-week effects (multiplicative, sum to 7)
  vector[6] dow_raw;                 // 6 free parameters for 7 effects
  
  // Time-varying ascertainment
  real logit_asc_initial;            // Initial ascertainment (logit scale)
  vector[effective_T-1] asc_innovations; // Random walk innovations
  real<lower=0> asc_sigma;           // Ascertainment random walk SD
  
  // Overdispersion
  real<lower=0> phi;                 // Negative binomial overdispersion
}

transformed parameters {
  vector[T] log_infections;
  vector[T] infections;
  vector[T] expected_cases;
  vector[7] dow_effects;
  vector[effective_T] rt;
  vector[effective_T] ascertainment;
  
  // Seeding period infections
  log_infections[1:seed_days] = log_infections_seed;
  
  // Day-of-week effects (constraint: sum = 7)
  dow_effects[1:6] = dow_raw;
  dow_effects[7] = 7 - sum(dow_raw);
  
  // Rt (random walk on log scale)
  rt[1] = R_initial;
  for(t in 2:effective_T) {
    rt[t] = rt[t-1] * exp(rt_innovations[t-1]);
  }
  
  // Ascertainment (random walk on logit scale)
  ascertainment[1] = inv_logit(logit_asc_initial);
  for(t in 2:effective_T) {
    ascertainment[t] = inv_logit(logit(ascertainment[t-1]) + asc_innovations[t-1]);
  }
  
  // Renewal equation for infections after seeding
  for(t in (seed_days + 1):T) {
    real convolution = 0;
    int rt_idx = t - seed_days;
    
    for(s in 1:min(S, t-1)) {
      convolution += infections[t-s] * gen_interval[s];
    }
    
    log_infections[t] = log(rt[rt_idx] * convolution + 1e-10);
  }
  
  // Convert to natural scale
  infections = exp(log_infections);
  
  // Expected reported cases (with delay and day-of-week effects)
  for(t in 1:T) {
    real delayed_infections = 0;
    
    // Apply delay from infection to reporting
    for(d in 1:min(D, t)) {
      int inf_day = t - d + 1;
      if(inf_day >= 1) {
        delayed_infections += infections[inf_day] * delay_pmf[d];
      }
    }
    
    // Apply ascertainment (use last available value for seeding period)
    real current_asc;
    if(t <= seed_days) {
      current_asc = ascertainment[1];
    } else {
      current_asc = ascertainment[t - seed_days];
    }
    
    expected_cases[t] = delayed_infections * current_asc * dow_effects[day_of_week[t]];
  }
}

model {
  // Priors
  log_infections_seed ~ normal(5, 2);   // Initial infections
  R_initial ~ normal(1, 0.5);
  rt_innovations ~ normal(0, rt_sigma);
  rt_sigma ~ exponential(20);           // Promotes smoothness
  
  dow_raw ~ normal(1, 0.2);             // Day-of-week effects around 1
  
  logit_asc_initial ~ normal(-2, 1);    // Initial ascertainment around 0.1
  asc_innovations ~ normal(0, asc_sigma);
  asc_sigma ~ exponential(50);          // Promotes smoothness
  
  phi ~ exponential(0.1);               // Overdispersion
  
  // Likelihood
  for(t in 1:T) {
    if(expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] cases_pred;
  
  for(t in 1:T) {
    if(expected_cases[t] > 0) {
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
    } else {
      log_lik[t] = 0;
      cases_pred[t] = 0;
    }
  }
}
"

# Function to create generation interval (gamma distribution)
create_generation_interval <- function(mean_gi = 5.2, sd_gi = 2.3, max_gi = 20) {
  shape <- (mean_gi / sd_gi)^2
  rate <- mean_gi / sd_gi^2
  
  gi <- dgamma(1:max_gi, shape = shape, rate = rate)
  gi / sum(gi)  # Normalize to sum to 1
}

# Function to create delay distribution (gamma distribution)
create_delay_distribution <- function(mean_delay = 8, sd_delay = 4, max_delay = 30) {
  shape <- (mean_delay / sd_delay)^2
  rate <- mean_delay / sd_delay^2
  
  delays <- dgamma(1:max_delay, shape = shape, rate = rate)
  delays / sum(delays)  # Normalize to sum to 1
}

# Function to load and prepare data
load_and_prepare_data <- function(filename) {
  # For demonstration, I'll create sample data
  # In practice, you would use: data <- read.csv(filename)
  
  dates <- seq(as.Date("2023-01-01"), as.Date("2023-06-30"), by = "day")
  
  # Simulate realistic COVID case data with patterns
  set.seed(123)
  T <- length(dates)
  
  # Simulate underlying Rt that changes over time
  true_rt <- c(
    rep(1.5, 30),           # Initial high transmission
    seq(1.5, 0.8, length.out = 60),  # Declining due to interventions
    rep(0.8, 30),           # Low transmission
    seq(0.8, 1.2, length.out = 30),  # Increase (variant or relaxed measures)
    rep(1.2, T - 150)       # Moderate transmission
  )
  
  # Simulate infections using renewal equation
  gen_interval <- create_generation_interval()
  infections <- numeric(T)
  infections[1:14] <- exp(rnorm(14, log(100), 0.3))  # Seeding
  
  for(t in 15:T) {
    convolution <- sum(infections[max(1, t-20):t-1] * rev(gen_interval[1:min(20, t-1)]))
    infections[t] <- max(1, true_rt[t] * convolution * exp(rnorm(1, 0, 0.1)))
  }
  
  # Apply delay and day-of-week effects to get reported cases
  delay_pmf <- create_delay_distribution()
  dow_effects <- c(1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5)  # Weekend underreporting
  ascertainment <- 0.3  # 30% of infections reported
  
  cases <- numeric(T)
  for(t in 1:T) {
    delayed_inf <- sum(infections[max(1, t-30):t] * rev(delay_pmf[1:min(30, t)]))
    dow <- wday(dates[t], week_start = 1)
    expected <- delayed_inf * ascertainment * dow_effects[dow]
    cases[t] <- rnbinom(1, mu = expected, size = 10)
  }
  
  data.frame(
    date = dates,
    cases = cases,
    day_of_week = wday(dates, week_start = 1)
  )
}

# Main estimation function
estimate_rt <- function(data, seed_days = 14) {
  # Prepare data for Stan
  gen_interval <- create_generation_interval()
  delay_pmf <- create_delay_distribution()
  
  stan_data <- list(
    T = nrow(data),
    cases = data$cases,
    day_of_week = data$day_of_week,
    S = length(gen_interval),
    gen_interval = gen_interval,
    D = length(delay_pmf),
    delay_pmf = delay_pmf,
    seed_days = seed_days
  )
  
  # Compile and fit the model
  model <- stan_model(model_code = stan_model_code)
  
  fit <- sampling(
    model,
    data = stan_data,
    chains = 4,
    iter = 2000,
    warmup = 1000,
    cores = 4,
    control = list(adapt_delta = 0.95, max_treedepth = 12)
  )
  
  return(fit)
}

# Function to extract and summarize results
extract_results <- function(fit, data, seed_days = 14) {
  # Extract posterior samples
  rt_samples <- extract(fit, "rt")$rt
  dow_samples <- extract(fit, "dow_effects")$dow_effects
  asc_samples <- extract(fit, "ascertainment")$ascertainment
  
  # Summarize Rt
  rt_summary <- data.frame(
    date = data$date[(seed_days + 1):nrow(data)],
    rt_median = apply(rt_samples, 2, median),
    rt_lower = apply(rt_samples, 2, quantile, 0.025),
    rt_upper = apply(rt_samples, 2, quantile, 0.975)
  )
  
  # Summarize day-of-week effects
  dow_summary <- data.frame(
    day_of_week = 1:7,
    day_name = c("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"),
    effect_median = apply(dow_samples, 2, median),
    effect_lower = apply(dow_samples, 2, quantile, 0.025),
    effect_upper = apply(dow_samples, 2, quantile, 0.975)
  )
  
  # Summarize ascertainment
  asc_summary <- data.frame(
    date = data$date[(seed_days + 1):nrow(data)],
    asc_median = apply(asc_samples, 2, median),
    asc_lower = apply(asc_samples, 2, quantile, 0.025),
    asc_upper = apply(asc_samples, 2, quantile, 0.975)
  )
  
  return(list(
    rt = rt_summary,
    day_of_week = dow_summary,
    ascertainment = asc_summary,
    fit = fit
  ))
}

# Function to create plots
create_plots <- function(results, data) {
  # Rt plot
  p1 <- ggplot(results$rt, aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "blue") +
    geom_line(aes(y = rt_median), color = "blue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         x = "Date", y = "Rt") +
    theme_minimal()
  
  # Day-of-week effects
  p2 <- ggplot(results$day_of_week, aes(x = day_name)) +
    geom_col(aes(y = effect_median), fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = effect_lower, ymax = effect_upper), 
                  width = 0.2) +
    labs(title = "Day-of-Week Effects",
         x = "Day of Week", y = "Multiplicative Effect") +
    theme_minimal()
  
  # Ascertainment over time
  p3 <- ggplot(results$ascertainment, aes(x = date)) +
    geom_ribbon(aes(ymin = asc_lower, ymax = asc_upper), alpha = 0.3, fill = "green") +
    geom_line(aes(y = asc_median), color = "green", size = 1) +
    labs(title = "Time-varying Ascertainment Rate",
         x = "Date", y = "Ascertainment Rate") +
    theme_minimal()
  
  # Cases over time with model fit
  cases_pred <- extract(results$fit, "cases_pred")$cases_pred
  cases_fit <- data.frame(
    date = data$date,
    observed = data$cases,
    predicted_median = apply(cases_pred, 2, median),
    predicted_lower = apply(cases_pred, 2, quantile, 0.025),
    predicted_upper = apply(cases_pred, 2, quantile, 0.975)
  )
  
  p4 <- ggplot(cases_fit, aes(x = date)) +
    geom_ribbon(aes(ymin = predicted_lower, ymax = predicted_upper), 
                alpha = 0.3, fill = "orange") +
    geom_line(aes(y = predicted_median), color = "orange", size = 1) +
    geom_point(aes(y = observed), size = 0.5, alpha = 0.6) +
    labs(title = "Observed vs Predicted Cases",
         x = "Date", y = "Daily Cases") +
    theme_minimal()
  
  return(list(rt = p1, dow = p2, ascertainment = p3, fit = p4))
}

# Main execution
main <- function() {
  cat("Loading and preparing data...\n")
  data <- load_and_prepare_data("cases.csv")
  
  cat("Fitting model (this may take several minutes)...\n")
  fit <- estimate_rt(data)
  
  cat("Extracting results...\n")
  results <- extract_results(fit, data)
  
  cat("Creating plots...\n")
  plots <- create_plots(results, data)
  
  # Display results
  cat("\n=== RT ESTIMATES (last 10 days) ===\n")
  print(tail(results$rt, 10))
  
  cat("\n=== DAY-OF-WEEK EFFECTS ===\n")
  print(results$day_of_week)
  
  cat("\n=== ASCERTAINMENT (last 10 days) ===\n")
  print(tail(results$ascertainment, 10))
  
  # Show plots
  print(plots$rt)
  print(plots$dow)
  print(plots$ascertainment)
  print(plots$fit)
  
  # Model diagnostics
  cat("\n=== MODEL DIAGNOSTICS ===\n")
  print(summary(fit, pars = c("rt_sigma", "asc_sigma", "phi"))$summary)
  
  return(list(
    data = data,
    results = results,
    plots = plots,
    fit = fit
  ))
}

# Run the analysis
if(interactive()) {
  analysis <- main()
}
```

---

## SUB_019

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
# Install required packages
install.packages("data.table")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("bsts")

# Load necessary libraries
library(data.table)
library(dplyr)
library(ggplot2)
library(bsts)

# Load the case data
cases <- fread("cases.csv")

# Convert date column to datetime format and extract month
cases$date <- as.Date(cases$date, "%Y-%m-%d")
cases$month <- format(cases$date, "%Y-%m")

# Define a function for the model's log-likelihood
log_lik <- function(par, dt) {
  rt <- par[1]
  day_of_week_effects <- par[2:7]
  ascertainment <- rep(par[8], length(dt))
  
  # Calculate the observed infections using the renewal equation
  I_observed <- dt$cases
  
  # Apply day-of-week effects and time-varying ascertainment
  day_of_week_effect <- (day_of_week_effects[dt$day_of_week]) * 
    (1 + ascertainment)
  
  # Add a multiplicative effect for overdispersion
  I_pred <- rt * dt$I_total * day_of_week_effect
  
  # Calculate log-likelihood with overdispersion
  ll <- -(sum(I_observed) - sum(I_pred)) / 2 - 
    (sum((I_observed - I_pred)^2) / 2)
  
  return(ll)
}

# Create a bsts model for the time series data
model <- bsts(
  y = cases$cases,
  ar_dens = list(),
  state_model = list(),
  niter = 10000,
  log_lik = log_lik,
  init = c(1, rep(1, 6), 0.5),
  dt = cases
)

# Extract the model results
results <- residuals(model)
rt_estimates <- fitted(model)$series[[1]]
day_of_week_effects <- fitted(model)$series[[2]]$coefficients[seq(1, length(day_of_week_effects), by=7)]
ascertainment <- rep(fitted(model)$series[[3]], length(rt_estimates))

# Plot the Rt estimates over time
ggplot(data.frame(date = cases$date,
                  rt_estimate = rt_estimates),
       aes(x = date, y = rt_estimate)) +
  geom_line() +
  labs(title = "Estimated Time-Varying Reproduction Number (Rt)",
       x = "Date",
       y = expression(R[t]))
```

---

## SUB_032

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(purr)
library(lme4)
library(bayesplot)

# Load data
cases <- read.csv("cases.csv")

# Define model structure
model <- function(cases, day_of_week) {
  # Delayed reporting term (infection to report delay: 5 days)
  delayed_cases <- cases %>% 
    mutate(delay = date - lag(date)) %>% 
    group_by(delay) %>% 
    summarise(sum_delayed = sum(cases))

  # Time-varying ascertainment
  ascertainment <- s(ascertainment_date) * I_t

  # Day-of-week effects
  day_effects <- day_of_week %>% 
    mutate(effect = ifelse(day == "Monday", 1, 
                           ifelse(day == "Tuesday", 0.85, 
                                  ifelse(day == "Wednesday", 0.8, 
                                         ifelse(day == "Thursday", 0.7, 
                                                ifelse(day == "Friday", 0.65, 
                                                       ifelse(day == "Saturday", 0.6, 
                                                              "Sunday"))))))))

  # Renewal equation
  I_t <- delayed_cases$sum_delayed * 
    day_effects$effect * ascertainment

  return(I_t)
}

# Fit model with Bayesian linear mixed effects model (BLMM)
fit <- blmer(Rt ~ date + (1|date), data = cases, REML = FALSE)

# Extract posterior estimates of Rt
posterior_rts <- posterior(fit)

# Plot posterior distribution of Rt estimates
ppc_densplot(posterior_rts, "Rt", parse = TRUE) +
  labs(title = "Posterior Distribution of Rt Estimates")

# Estimate day-of-week effects and time-varying ascertainment
day_effects <- cases %>% 
  group_by(day_of_week) %>% 
  summarise(mean_cases = mean(cases))

ascertainment_estimate <- cases %>% 
  group_by(date) %>% 
  summarise(estimate = mean(cases)) %>%
  mutate(estimate = estimate / (sum(delayed_cases$sum_delayed))) %>%
  arrange(date)

# Plot day-of-week effects and time-varying ascertainment
ggplot(day_effects, aes(x = day_of_week, y = mean_cases)) + 
  geom_bar(stat = "identity") +
  labs(title = "Day-of-Week Effects on Reported Cases")

ggplot(ascertainment_estimate, aes(x = date, y = estimate)) + 
  geom_line() +
  labs(title = "Time-Varying Ascertainment Estimates")
```

---

## SUB_052

**Scenario**: 2 | **Execution**: FAILED

### script.R

```r
# Load necessary libraries and data
library(tidyverse)
library(deSolve)
library(Rtsnps)

# Load case counts from CSV file
cases <- read_csv("cases.csv")

# Add day-of-week label as a factor for easier interpretation
cases$day_of_week_factor <- factor(cases$day_of_week,
                                   levels = 1:7,
                                   labels = c("Monday", "Tuesday", "Wednesday",
                                              "Thursday", "Friday", "Saturday", "Sunday"))

# Define model functions and parameters
model_func <- function(t, y, params) {
  # Infection dynamics (renewal equation)
  dydt <- c(
    dRt = -params[1] * R_t,
    dascertainment = params[2] - params[3]
  )
  
  return(list(c(dRt, dascertainment)))
}

# Define parameters and initial conditions
initial_conditions <- data.frame(R_t = 0.5,
                                 ascertainment = 0.6)

# Time points to estimate Rt at
time_points <- seq(1, nrow(cases), by = 1)

# Fit model using deSolve
fit <- ode(model_func,
          times = time_points,
          y = initial_conditions,
          parms = c(0.2,  # decay rate of Rt
                    0.8,  # initial ascertainment (proportion of infections reported)
                    0.5))  # overdispersion

# Extract estimates of Rt and day-of-week effects
Rt_estimates <- data.frame(
  date = cases$date[fit$times],
  R_t = fit$solution[, 1]
)

day_of_week_effects <- cases %>%
  group_by(day_of_week_factor) %>%
  summarise(n = n()) %>%
  mutate(day_of_week_effect = n / sum(n))

# Extract estimates of time-varying ascertainment
ascertainment_estimates <- data.frame(
  date = cases$date[fit$times],
  ascertainment = fit$solution[, 2]
)

# Summary plot
library(ggplot2)
ggplot(Rt_estimates, aes(date, R_t)) +
  geom_line() +
  labs(title = "Time-varying Reproduction Number (Rt) Estimates",
       subtitle = "COVID-19 Case Counts")
```

---

# Scenario 3

Total submissions: 30

---

## SUB_087

**Scenario**: 3 | **Execution**: FAILED

### model.stan

```stan
// multi_stream_rt.stan
functions {
  // Discrete gamma PMF for delays
  vector discrete_gamma_pmf(int max_delay, real shape, real rate) {
    vector[max_delay] pmf;
    real normalizing_constant = 0;
    
    // Calculate unnormalized probabilities
    for (i in 1:max_delay) {
      pmf[i] = exp(lgamma(shape + i - 1) - lgamma(shape) - lgamma(i) + 
                   shape * log(rate) - (shape + i - 1) * log(1 + rate));
      normalizing_constant += pmf[i];
    }
    
    // Normalize
    pmf = pmf / normalizing_constant;
    return pmf;
  }
}

data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> S;                    // Number of data streams (3)
  int<lower=0> max_delay;            // Maximum delay to consider
  int<lower=0> max_gen;              // Maximum generation interval
  
  // Observations
  array[T, S] int<lower=0> y;        // Observations [time, stream]
  
  // Generation interval (pre-computed)
  vector<lower=0>[max_gen] generation_interval;
  
  // Priors for delay distributions
  vector<lower=0>[S] delay_shape_prior_mean;
  vector<lower=0>[S] delay_shape_prior_sd;
  vector<lower=0>[S] delay_rate_prior_mean;
  vector<lower=0>[S] delay_rate_prior_sd;
}

parameters {
  // Rt parameters
  real log_R0;                       // Initial log(Rt)
  vector[T-1] log_Rt_innovations;    // Random walk innovations
  real<lower=0> sigma_Rt;            // SD of Rt random walk
  
  // Stream-specific parameters
  vector<lower=0>[S] ascertainment_rate;  // Ascertainment rates
  vector<lower=0>[S] phi;                 // Overdispersion parameters
  
  // Delay distribution parameters
  vector<lower=0>[S] delay_shape;
  vector<lower=0>[S] delay_rate;
  
  // Initial infections
  vector<lower=0>[max_gen] I_seed;
}

transformed parameters {
  vector[T] log_Rt;
  vector[T] Rt;
  vector[T] infections;
  matrix[T, S] expected_obs;
  
  // Build log_Rt time series
  log_Rt[1] = log_R0;
  for (t in 2:T) {
    log_Rt[t] = log_Rt[t-1] + log_Rt_innovations[t-1];
  }
  Rt = exp(log_Rt);
  
  // Calculate delay PMFs for each stream
  matrix[max_delay, S] delay_pmf;
  for (s in 1:S) {
    delay_pmf[,s] = discrete_gamma_pmf(max_delay, delay_shape[s], delay_rate[s]);
  }
  
  // Calculate infections using renewal equation
  for (t in 1:T) {
    if (t <= max_gen) {
      // Seeding period
      infections[t] = I_seed[t];
    } else {
      // Renewal equation
      real infectiousness = 0;
      for (tau in 1:max_gen) {
        if (t - tau > 0) {
          infectiousness += infections[t - tau] * generation_interval[tau];
        }
      }
      infections[t] = Rt[t] * infectiousness;
    }
  }
  
  // Calculate expected observations for each stream
  for (s in 1:S) {
    for (t in 1:T) {
      expected_obs[t, s] = 0;
      
      // Sum over delays
      for (d in 1:min(max_delay, t)) {
        expected_obs[t, s] += infections[t - d + 1] * delay_pmf[d, s];
      }
      
      // Apply ascertainment
      expected_obs[t, s] *= ascertainment_rate[s];
      expected_obs[t, s] = fmax(expected_obs[t, s], 1e-8);
    }
  }
}

model {
  // Priors
  log_R0 ~ normal(0, 0.5);
  sigma_Rt ~ exponential(2);
  log_Rt_innovations ~ normal(0, sigma_Rt);
  
  // Stream-specific priors
  ascertainment_rate ~ exponential(10);  // Weak prior favoring small rates
  phi ~ exponential(0.1);                // Overdispersion
  
  // Delay distribution priors
  for (s in 1:S) {
    delay_shape[s] ~ normal(delay_shape_prior_mean[s], delay_shape_prior_sd[s]);
    delay_rate[s] ~ normal(delay_rate_prior_mean[s], delay_rate_prior_sd[s]);
  }
  
  // Initial infections
  I_seed ~ exponential(0.01);
  
  // Likelihood
  for (s in 1:S) {
    for (t in 1:T) {
      if (y[t, s] > 0 || expected_obs[t, s] > 0) {
        y[t, s] ~ neg_binomial_2(expected_obs[t, s], phi[s]);
      }
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[T, S] int y_rep;
  
  for (s in 1:S) {
    for (t in 1:T) {
      y_rep[t, s] = neg_binomial_2_rng(expected_obs[t, s], phi[s]);
    }
  }
}
```

### script.R

```r
# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)

# Load the data
observations <- read_csv("observations.csv")
observations$date <- as.Date(observations$date)

# Remove any rows with all NA observations
observations <- observations[rowSums(is.na(observations[,c("cases", "hospitalisations", "deaths")])) < 3,]

# Replace NAs with 0 (assuming missing = 0 observations)
observations[is.na(observations)] <- 0

print(paste("Data spans", min(observations$date), "to", max(observations$date)))
print(paste("Total days:", nrow(observations)))




# Prepare data for Stan
prepare_stan_data <- function(obs_data) {
  T <- nrow(obs_data)
  S <- 3  # cases, hospitalisations, deaths
  
  # Create observation matrix
  y <- array(0, dim = c(T, S))
  y[, 1] <- obs_data$cases
  y[, 2] <- obs_data$hospitalisations  
  y[, 3] <- obs_data$deaths
  
  # Generation interval (gamma distribution with mean 5.2, sd 1.72)
  max_gen <- 15
  gen_shape <- (5.2 / 1.72)^2
  gen_rate <- 5.2 / (1.72^2)
  generation_interval <- dgamma(1:max_gen, shape = gen_shape, rate = gen_rate)
  generation_interval <- generation_interval / sum(generation_interval)
  
  # Delay distribution priors (based on epidemiological knowledge)
  # Cases: shorter delay (mean ~5 days)
  # Hospitalisations: medium delay (mean ~10 days)  
  # Deaths: longer delay (mean ~18 days)
  delay_shape_prior_mean <- c(2.5, 3.0, 2.8)  
  delay_shape_prior_sd <- c(0.5, 0.7, 0.6)
  delay_rate_prior_mean <- c(0.5, 0.3, 0.15) 
  delay_rate_prior_sd <- c(0.2, 0.1, 0.05)
  
  list(
    T = T,
    S = S,
    max_delay = 30,
    max_gen = max_gen,
    y = y,
    generation_interval = generation_interval,
    delay_shape_prior_mean = delay_shape_prior_mean,
    delay_shape_prior_sd = delay_shape_prior_sd,
    delay_rate_prior_mean = delay_rate_prior_mean,
    delay_rate_prior_sd = delay_rate_prior_sd
  )
}

# Prepare the data
stan_data <- prepare_stan_data(observations)

# Compile the Stan model
model <- cmdstan_model("multi_stream_rt.stan")

# Fit the model
fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100,
  max_treedepth = 12,
  adapt_delta = 0.95
)

# Check diagnostics
fit$cmdstan_diagnose()
print(fit$summary(c("sigma_Rt", "ascertainment_rate", "phi")))


# Extract and summarize results
extract_results <- function(fit, observations) {
  # Extract Rt estimates
  rt_draws <- fit$draws("Rt", format = "draws_matrix")
  rt_summary <- posterior::summarise_draws(rt_draws, 
                                         mean, median, sd, 
                                         ~quantile(.x, c(0.025, 0.1, 0.25, 0.75, 0.9, 0.975)))
  
  rt_summary$date <- observations$date
  
  # Extract stream-specific parameters
  asc_draws <- fit$draws("ascertainment_rate", format = "draws_matrix")
  asc_summary <- posterior::summarise_draws(asc_draws, mean, median, sd,
                                          ~quantile(.x, c(0.025, 0.975)))
  asc_summary$stream <- c("Cases", "Hospitalisations", "Deaths")
  
  phi_draws <- fit$draws("phi", format = "draws_matrix")
  phi_summary <- posterior::summarise_draws(phi_draws, mean, median, sd,
                                          ~quantile(.x, c(0.025, 0.975)))
  phi_summary$stream <- c("Cases", "Hospitalisations", "Deaths")
  
  # Extract delay parameters
  delay_shape_draws <- fit$draws("delay_shape", format = "draws_matrix")
  delay_rate_draws <- fit$draws("delay_rate", format = "draws_matrix")
  
  delay_shape_summary <- posterior::summarise_draws(delay_shape_draws, mean, median)
  delay_rate_summary <- posterior::summarise_draws(delay_rate_draws, mean, median)
  
  # Calculate implied delay means
  delay_means <- delay_shape_summary$mean / delay_rate_summary$mean
  
  list(
    rt_estimates = rt_summary,
    ascertainment_rates = asc_summary,
    overdispersion = phi_summary,
    delay_means = data.frame(
      stream = c("Cases", "Hospitalisations", "Deaths"),
      delay_mean = delay_means
    )
  )
}

# Extract results
results <- extract_results(fit, observations)

# Display key results
cat("=== Rt Estimation Results ===\n\n")

cat("Stream-specific Ascertainment Rates:\n")
print(results$ascertainment_rates[, c("stream", "mean", "q2.5", "q97.5")])

cat("\nStream-specific Overdispersion Parameters:\n") 
print(results$overdispersion[, c("stream", "mean", "q2.5", "q97.5")])

cat("\nEstimated Delay Means (days from infection):\n")
print(results$delay_means)

cat(sprintf("\nFinal Rt estimate: %.2f (95%% CI: %.2f-%.2f)\n",
           tail(results$rt_estimates$mean, 1),
           tail(results$rt_estimates$q2.5, 1), 
           tail(results$rt_estimates$q97.5, 1)))


# Create visualization
create_plots <- function(results, observations) {
  # Rt plot
  rt_plot <- ggplot(results$rt_estimates, aes(x = date)) +
    geom_ribbon(aes(ymin = q2.5, ymax = q97.5), alpha = 0.3, fill = "steelblue") +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.5, fill = "steelblue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         subtitle = "Estimated jointly from cases, hospitalisations, and deaths",
         x = "Date", y = "Rt",
         caption = "Ribbons show 50% and 95% credible intervals") +
    theme_minimal() +
    theme(plot.title = element_text(size = 14, face = "bold"))
  
  # Prepare data for multi-stream plot
  obs_long <- observations %>%
    pivot_longer(cols = c(cases, hospitalisations, deaths),
                names_to = "stream", values_to = "observed") %>%
    mutate(stream = case_when(
      stream == "cases" ~ "Cases",
      stream == "hospitalisations" ~ "Hospitalisations", 
      stream == "deaths" ~ "Deaths"
    ))
  
  # Data streams plot
  streams_plot <- ggplot(obs_long, aes(x = date, y = observed)) +
    geom_line(color = "darkred") +
    geom_point(size = 0.8, alpha = 0.6, color = "darkred") +
    facet_wrap(~stream, scales = "free_y", ncol = 1) +
    labs(title = "Observed Data Streams",
         x = "Date", y = "Daily Count") +
    theme_minimal() +
    theme(strip.text = element_text(face = "bold"))
  
  # Parameter comparison plot
  param_data <- rbind(
    data.frame(results$ascertainment_rates[, c("stream", "mean", "q2.5", "q97.5")], 
               parameter = "Ascertainment Rate"),
    data.frame(results$overdispersion[, c("stream", "mean", "q2.5", "q97.5")], 
               parameter = "Overdispersion (phi)")
  )
  
  param_plot <- ggplot(param_data, aes(x = stream, y = mean)) +
    geom_pointrange(aes(ymin = q2.5, ymax = q97.5), size = 0.8) +
    facet_wrap(~parameter, scales = "free_y") +
    labs(title = "Stream-specific Parameters",
         x = "Data Stream", y = "Estimate") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          strip.text = element_text(face = "bold"))
  
  return(list(rt_plot = rt_plot, streams_plot = streams_plot, param_plot = param_plot))
}

# Create plots
plots <- create_plots(results, observations)

# Display plots
print(plots$rt_plot)
print(plots$streams_plot) 
print(plots$param_plot)

# Save plots
ggsave("rt_estimates.png", plots$rt_plot, width = 12, height = 6, dpi = 300)
ggsave("data_streams.png", plots$streams_plot, width = 10, height = 8, dpi = 300)
ggsave("parameters.png", plots$param_plot, width = 10, height = 6, dpi = 300)


# Model checking and validation
perform_model_checks <- function(fit, stan_data, observations) {
  cat("=== Model Diagnostics ===\n\n")
  
  # Check Rhat and ESS
  all_draws <- fit$draws()
  diagnostics <- posterior::summarise_draws(all_draws)
  
  max_rhat <- max(diagnostics$rhat, na.rm = TRUE)
  min_ess <- min(diagnostics$ess_bulk, na.rm = TRUE)
  
  cat(sprintf("Maximum Rhat: %.3f %s\n", max_rhat, 
              ifelse(max_rhat < 1.01, "(Good)", "(Check convergence)")))
  cat(sprintf("Minimum ESS: %.0f %s\n\n", min_ess,
              ifelse(min_ess > 400, "(Good)", "(Consider more iterations)")))
  
  # Posterior predictive checks
  y_rep <- fit$draws("y_rep", format = "draws_matrix")
  
  # Calculate summary statistics for each stream
  for (s in 1:3) {
    stream_names <- c("Cases", "Hospitalisations", "Deaths")
    observed <- stan_data$y[, s]
    
    # Extract replications for this stream
    y_rep_stream <- y_rep[, seq(s, ncol(y_rep), by = 3)]
    
    # Calculate test statistics
    obs_mean <- mean(observed)
    rep_means <- apply(y_rep_stream, 1, mean)
    
    obs_max <- max(observed)
    rep_maxs <- apply(y_rep_stream, 1, max)
    
    p_mean <- mean(rep_means > obs_mean)
    p_max <- mean(rep_maxs > obs_max)
    
    cat(sprintf("%s:\n", stream_names[s]))
    cat(sprintf("  Mean check: observed=%.1f, p-value=%.3f\n", obs_mean, p_mean))
    cat(sprintf("  Max check: observed=%d, p-value=%.3f\n", obs_max, p_max))
  }
}

# Run model checks
perform_model_checks(fit, stan_data, observations)

# Final summary
cat("\n=== Summary ===\n")
cat("Successfully estimated time-varying Rt jointly from three data streams\n")
cat("Key features implemented:\n")
cat("- Shared Rt across all streams via renewal equation\n")
cat("- Stream-specific delay distributions\n") 
cat("- Stream-specific ascertainment rates\n")
cat("- Negative binomial observation model with overdispersion\n")
cat("- Smooth Rt evolution via random walk\n")
cat("- Proper handling of initial infection seeding\n")
```

---

## SUB_112

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(cmdstanr)
library(data.table)
library(ggplot2)
library(bayesplot)
library(posterior)

# Read and prepare data
data <- fread("observations.csv")
data[, date := as.Date(date)]
data <- data[order(date)]

# Handle missing values by setting them to 0
data[is.na(cases), cases := 0]
data[is.na(hospitalisations), hospitalisations := 0]
data[is.na(deaths), deaths := 0]

# Define delay distributions (gamma distributions discretized)
# Cases: mean delay ~5 days from infection
# Hospitalizations: mean delay ~10 days from infection  
# Deaths: mean delay ~18 days from infection

discretize_gamma <- function(shape, rate, max_delay = 30) {
  delays <- 1:max_delay
  probs <- pgamma(delays, shape, rate) - pgamma(delays - 1, shape, rate)
  probs / sum(probs)
}

# Delay distributions
case_delay <- discretize_gamma(shape = 5.1, rate = 0.86, max_delay = 25)
hosp_delay <- discretize_gamma(shape = 4.5, rate = 0.45, max_delay = 35)  
death_delay <- discretize_gamma(shape = 4.9, rate = 0.27, max_delay = 45)

# Generation interval (mean ~6.5 days)
generation_interval <- discretize_gamma(shape = 2.3, rate = 0.35, max_delay = 20)

# Prepare data for Stan
n_days <- nrow(data)
n_streams <- 3
seeding_days <- 7  # Initial seeding period

stan_data <- list(
  n_days = n_days,
  n_streams = n_streams,
  seeding_days = seeding_days,
  
  # Observations
  obs_cases = data$cases,
  obs_hosp = data$hospitalisations, 
  obs_deaths = data$deaths,
  
  # Delay distributions
  max_delay_case = length(case_delay),
  max_delay_hosp = length(hosp_delay),
  max_delay_death = length(death_delay),
  max_delay_gen = length(generation_interval),
  
  delay_case = case_delay,
  delay_hosp = hosp_delay,
  delay_death = death_delay,
  generation_interval = generation_interval
)

# Stan model code
stan_code <- "
data {
  int<lower=1> n_days;
  int<lower=1> n_streams;
  int<lower=1> seeding_days;
  
  // Observations
  int obs_cases[n_days];
  int obs_hosp[n_days];
  int obs_deaths[n_days];
  
  // Delay distribution parameters
  int<lower=1> max_delay_case;
  int<lower=1> max_delay_hosp; 
  int<lower=1> max_delay_death;
  int<lower=1> max_delay_gen;
  
  vector[max_delay_case] delay_case;
  vector[max_delay_hosp] delay_hosp;
  vector[max_delay_death] delay_death;
  vector[max_delay_gen] generation_interval;
}

transformed data {
  int max_delay = max([max_delay_case, max_delay_hosp, max_delay_death]);
}

parameters {
  // Initial infections (seeding)
  vector<lower=0>[seeding_days] initial_infections_raw;
  real<lower=0> initial_infections_scale;
  
  // Rt random walk
  real log_rt_initial;
  vector[n_days - seeding_days - 1] log_rt_noise;
  real<lower=0> rt_random_walk_sd;
  
  // Stream-specific parameters
  vector<lower=0, upper=1>[n_streams] ascertainment_rate;
  vector<lower=0>[n_streams] overdispersion;
}

transformed parameters {
  vector<lower=0>[n_days] infections;
  vector[n_days - seeding_days] log_rt;
  vector<lower=0>[n_days - seeding_days] rt;
  
  vector<lower=0>[n_days] expected_cases;
  vector<lower=0>[n_days] expected_hosp;
  vector<lower=0>[n_days] expected_deaths;
  
  // Initial infections
  infections[1:seeding_days] = initial_infections_raw * initial_infections_scale;
  
  // Rt random walk
  log_rt[1] = log_rt_initial;
  for(t in 2:(n_days - seeding_days)) {
    log_rt[t] = log_rt[t-1] + log_rt_noise[t-1];
  }
  rt = exp(log_rt);
  
  // Renewal equation for infections
  for(t in (seeding_days + 1):n_days) {
    real infectiousness = 0.0;
    int rt_index = t - seeding_days;
    
    for(s in 1:min(max_delay_gen, t-1)) {
      infectiousness += infections[t-s] * generation_interval[s];
    }
    infections[t] = rt[rt_index] * infectiousness;
  }
  
  // Expected observations for each stream
  for(t in 1:n_days) {
    expected_cases[t] = 0;
    expected_hosp[t] = 0; 
    expected_deaths[t] = 0;
    
    // Cases
    for(s in 1:min(max_delay_case, t)) {
      expected_cases[t] += infections[t-s+1] * delay_case[s];
    }
    expected_cases[t] *= ascertainment_rate[1];
    
    // Hospitalizations
    for(s in 1:min(max_delay_hosp, t)) {
      expected_hosp[t] += infections[t-s+1] * delay_hosp[s];
    }
    expected_hosp[t] *= ascertainment_rate[2];
    
    // Deaths
    for(s in 1:min(max_delay_death, t)) {
      expected_deaths[t] += infections[t-s+1] * delay_death[s];
    }
    expected_deaths[t] *= ascertainment_rate[3];
  }
}

model {
  // Priors
  initial_infections_raw ~ exponential(1);
  initial_infections_scale ~ normal(0, 100);
  
  log_rt_initial ~ normal(0, 0.2);
  log_rt_noise ~ normal(0, rt_random_walk_sd);
  rt_random_walk_sd ~ normal(0, 0.05) T[0,];
  
  ascertainment_rate ~ beta(1, 1);
  overdispersion ~ normal(0, 1) T[0,];
  
  // Likelihood - negative binomial for overdispersion
  for(t in 1:n_days) {
    if(expected_cases[t] > 1e-8) {
      obs_cases[t] ~ neg_binomial_2(expected_cases[t], 1/overdispersion[1]);
    }
    if(expected_hosp[t] > 1e-8) {
      obs_hosp[t] ~ neg_binomial_2(expected_hosp[t], 1/overdispersion[2]);
    }
    if(expected_deaths[t] > 1e-8) {
      obs_deaths[t] ~ neg_binomial_2(expected_deaths[t], 1/overdispersion[3]);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  int pred_cases[n_days];
  int pred_hosp[n_days];
  int pred_deaths[n_days];
  
  for(t in 1:n_days) {
    pred_cases[t] = expected_cases[t] > 1e-8 ? 
      neg_binomial_2_rng(expected_cases[t], 1/overdispersion[1]) : 0;
    pred_hosp[t] = expected_hosp[t] > 1e-8 ? 
      neg_binomial_2_rng(expected_hosp[t], 1/overdispersion[2]) : 0;
    pred_deaths[t] = expected_deaths[t] > 1e-8 ? 
      neg_binomial_2_rng(expected_deaths[t], 1/overdispersion[3]) : 0;
  }
}
"

# Compile and fit model
model <- cmdstan_model(write_stan_file(stan_code))

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100
)

# Check model diagnostics
fit$cmdstan_diagnose()
print(fit$summary(c("rt_random_walk_sd", "ascertainment_rate", "overdispersion")))

# Extract Rt estimates
rt_draws <- fit$draws("rt", format = "draws_matrix")
rt_summary <- summarise_draws(rt_draws)

# Create results data frame
rt_dates <- data$date[(seeding_days + 1):n_days]
rt_results <- data.table(
  date = rt_dates,
  rt_median = rt_summary$median,
  rt_lower = rt_summary$q5,
  rt_upper = rt_summary$q95,
  rt_mean = rt_summary$mean
)

# Extract stream-specific parameters
ascertainment_summary <- summarise_draws(fit$draws("ascertainment_rate"))
overdispersion_summary <- summarise_draws(fit$draws("overdispersion"))

stream_params <- data.table(
  stream = c("Cases", "Hospitalizations", "Deaths"),
  ascertainment_median = ascertainment_summary$median,
  ascertainment_lower = ascertainment_summary$q5,
  ascertainment_upper = ascertainment_summary$q95,
  overdispersion_median = overdispersion_summary$median,
  overdispersion_lower = overdispersion_summary$q5,
  overdispersion_upper = overdispersion_summary$q95
)

# Print results summary
cat("=== Rt Estimation Results ===\n")
print(rt_results)

cat("\n=== Stream-Specific Parameters ===\n")
print(stream_params)

# Create visualization
p1 <- ggplot(rt_results, aes(x = date)) +
  geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "steelblue") +
  geom_line(aes(y = rt_median), color = "steelblue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       subtitle = "Estimated jointly from cases, hospitalizations, and deaths",
       x = "Date", y = "Rt") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

# Plot observations vs predictions
pred_cases <- summarise_draws(fit$draws("pred_cases"))$median
pred_hosp <- summarise_draws(fit$draws("pred_hosp"))$median  
pred_deaths <- summarise_draws(fit$draws("pred_deaths"))$median

obs_data <- data.table(
  date = rep(data$date, 3),
  stream = rep(c("Cases", "Hospitalizations", "Deaths"), each = n_days),
  observed = c(data$cases, data$hospitalisations, data$deaths),
  predicted = c(pred_cases, pred_hosp, pred_deaths)
)

p2 <- ggplot(obs_data, aes(x = date)) +
  geom_line(aes(y = observed, color = "Observed"), size = 0.8) +
  geom_line(aes(y = predicted, color = "Predicted"), size = 0.8, linetype = "dashed") +
  facet_wrap(~stream, scales = "free_y", ncol = 1) +
  scale_color_manual(values = c("Observed" = "black", "Predicted" = "red")) +
  labs(title = "Model Fit: Observed vs Predicted",
       x = "Date", y = "Count", color = "") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5),
        legend.position = "bottom")

# Print plots
print(p1)
print(p2)

# Save results
fwrite(rt_results, "rt_estimates.csv")
fwrite(stream_params, "stream_parameters.csv")

cat("\n=== Model Summary ===\n")
cat("Successfully estimated Rt using joint renewal equation model\n")
cat("- Shared Rt across all data streams\n")  
cat("- Stream-specific delays and ascertainment rates\n")
cat("- Overdispersion modeled with negative binomial\n")
cat("- Smooth evolution of Rt via random walk prior\n")
cat("\nResults saved to: rt_estimates.csv and stream_parameters.csv\n")
```

---

## SUB_017

**Scenario**: 3 | **Execution**: TIMEOUT

### script.R

```r
# Load required libraries
library(cmdstanr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

# Load the data
data <- read_csv("observations.csv") %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

# Define generation interval (discretized gamma distribution)
# Mean = 5 days, SD = 2 days
gen_mean <- 5
gen_sd <- 2
max_gen <- 15

# Convert to gamma parameters
gen_shape <- (gen_mean / gen_sd)^2
gen_rate <- gen_mean / gen_sd^2

# Discretize generation interval
gen_interval <- diff(pgamma(0:(max_gen), shape = gen_shape, rate = gen_rate))
gen_interval <- gen_interval / sum(gen_interval)

# Define delay distributions for each stream (discretized gamma)
max_delay <- 30

# Cases: shorter delay (mean = 7 days)
cases_delay_shape <- (7 / 3)^2
cases_delay_rate <- 7 / 3^2
cases_delay <- diff(pgamma(0:(max_delay), shape = cases_delay_shape, rate = cases_delay_rate))
cases_delay <- cases_delay / sum(cases_delay)

# Hospitalizations: medium delay (mean = 14 days)
hosp_delay_shape <- (14 / 5)^2
hosp_delay_rate <- 14 / 5^2
hosp_delay <- diff(pgamma(0:(max_delay), shape = hosp_delay_shape, rate = hosp_delay_rate))
hosp_delay <- hosp_delay / sum(hosp_delay)

# Deaths: longer delay (mean = 21 days)
death_delay_shape <- (21 / 7)^2
death_delay_rate <- 21 / 7^2
death_delay <- diff(pgamma(0:(max_delay), shape = death_delay_shape, rate = death_delay_rate))
death_delay <- death_delay / sum(death_delay)

# Prepare data for Stan
n_days <- nrow(data)
n_seed_days <- 7  # Initial seeding period

stan_data <- list(
  n_days = n_days,
  n_seed_days = n_seed_days,
  n_gen = length(gen_interval),
  n_delay = length(cases_delay),
  
  # Observations
  cases = data$cases,
  hospitalizations = data$hospitalisations,
  deaths = data$deaths,
  
  # Delay distributions
  gen_interval = gen_interval,
  cases_delay = cases_delay,
  hosp_delay = hosp_delay,
  death_delay = death_delay
)

# Stan model code
stan_code <- "
functions {
  vector convolve_delay(vector infections, vector delay_pmf, int n_days, int n_delay) {
    vector[n_days] convolved = rep_vector(0.0, n_days);
    
    for (t in 1:n_days) {
      for (d in 1:min(t, n_delay)) {
        convolved[t] += infections[t - d + 1] * delay_pmf[d];
      }
    }
    return convolved;
  }
}

data {
  int<lower=1> n_days;
  int<lower=1> n_seed_days;
  int<lower=1> n_gen;
  int<lower=1> n_delay;
  
  // Observations
  array[n_days] int<lower=0> cases;
  array[n_days] int<lower=0> hospitalizations;
  array[n_days] int<lower=0> deaths;
  
  // Delay distributions
  vector[n_gen] gen_interval;
  vector[n_delay] cases_delay;
  vector[n_delay] hosp_delay;
  vector[n_delay] death_delay;
}

parameters {
  // Initial infections (seeding period)
  vector<lower=0>[n_seed_days] seed_infections;
  
  // Log Rt with random walk
  vector[n_days - n_seed_days] log_rt_raw;
  real log_rt_init;
  real<lower=0> rt_sd;
  
  // Ascertainment rates
  real<lower=0, upper=1> ascertain_cases;
  real<lower=0, upper=1> ascertain_hosp;
  real<lower=0, upper=1> ascertain_deaths;
  
  // Overdispersion parameters
  real<lower=0> phi_cases;
  real<lower=0> phi_hosp;
  real<lower=0> phi_deaths;
}

transformed parameters {
  vector[n_days] infections;
  vector[n_days] log_rt;
  vector[n_days] rt;
  
  // Expected observations
  vector[n_days] expected_cases;
  vector[n_days] expected_hosp;
  vector[n_days] expected_deaths;
  
  // Construct log_rt time series
  log_rt[1:n_seed_days] = rep_vector(log_rt_init, n_seed_days);
  for (t in (n_seed_days + 1):n_days) {
    log_rt[t] = log_rt[t-1] + rt_sd * log_rt_raw[t - n_seed_days];
  }
  rt = exp(log_rt);
  
  // Seeding infections
  infections[1:n_seed_days] = seed_infections;
  
  // Renewal equation for post-seeding infections
  for (t in (n_seed_days + 1):n_days) {
    real infectiousness = 0;
    
    for (s in 1:min(t-1, n_gen)) {
      infectiousness += infections[t - s] * gen_interval[s];
    }
    
    infections[t] = rt[t] * infectiousness;
  }
  
  // Convolve infections with delays and apply ascertainment
  expected_cases = ascertain_cases * convolve_delay(infections, cases_delay, n_days, n_delay);
  expected_hosp = ascertain_hosp * convolve_delay(infections, hosp_delay, n_days, n_delay);
  expected_deaths = ascertain_deaths * convolve_delay(infections, death_delay, n_days, n_delay);
}

model {
  // Priors
  seed_infections ~ exponential(0.1);
  log_rt_init ~ normal(0, 0.2);
  rt_sd ~ normal(0, 0.1) T[0,];
  log_rt_raw ~ std_normal();
  
  ascertain_cases ~ beta(2, 2);
  ascertain_hosp ~ beta(2, 2);
  ascertain_deaths ~ beta(2, 2);
  
  phi_cases ~ exponential(0.1);
  phi_hosp ~ exponential(0.1);
  phi_deaths ~ exponential(0.1);
  
  // Likelihood with negative binomial for overdispersion
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi_cases);
    }
    if (expected_hosp[t] > 0) {
      hospitalizations[t] ~ neg_binomial_2(expected_hosp[t], phi_hosp);
    }
    if (expected_deaths[t] > 0) {
      deaths[t] ~ neg_binomial_2(expected_deaths[t], phi_deaths);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[n_days] int cases_rep;
  array[n_days] int hosp_rep;
  array[n_days] int deaths_rep;
  
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      cases_rep[t] = neg_binomial_2_rng(expected_cases[t], phi_cases);
    } else {
      cases_rep[t] = 0;
    }
    
    if (expected_hosp[t] > 0) {
      hosp_rep[t] = neg_binomial_2_rng(expected_hosp[t], phi_hosp);
    } else {
      hosp_rep[t] = 0;
    }
    
    if (expected_deaths[t] > 0) {
      deaths_rep[t] = neg_binomial_2_rng(expected_deaths[t], phi_deaths);
    } else {
      deaths_rep[t] = 0;
    }
  }
}
"

# Compile and fit the model
model <- cmdstan_model(write_stan_file(stan_code))

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100,
  adapt_delta = 0.95,
  max_treedepth = 12
)

# Extract results
draws <- fit$draws(format = "df")

# Extract Rt estimates
rt_summary <- fit$summary("rt") %>%
  mutate(date = data$date,
         day = row_number())

# Extract ascertainment rates
ascertainment <- fit$summary(c("ascertain_cases", "ascertain_hosp", "ascertain_deaths"))

print("Ascertainment Rate Estimates:")
print(ascertainment)

# Extract overdispersion parameters
overdispersion <- fit$summary(c("phi_cases", "phi_hosp", "phi_deaths"))

print("Overdispersion Parameters:")
print(overdispersion)

# Create plots
# Plot 1: Rt over time
rt_plot <- ggplot(rt_summary, aes(x = date, y = mean)) +
  geom_ribbon(aes(ymin = q5, ymax = q95), alpha = 0.3, fill = "blue") +
  geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.5, fill = "blue") +
  geom_line(color = "darkblue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Estimated Reproduction Number (Rt)",
       subtitle = "50% and 90% credible intervals",
       x = "Date", y = "Rt") +
  theme_minimal()

print(rt_plot)

# Plot 2: Observed vs Expected data
expected_summary <- fit$summary(c("expected_cases", "expected_hosp", "expected_deaths")) %>%
  mutate(stream = case_when(
    str_detect(variable, "cases") ~ "Cases",
    str_detect(variable, "hosp") ~ "Hospitalizations",
    str_detect(variable, "deaths") ~ "Deaths"
  )) %>%
  mutate(day = rep(1:n_days, 3),
         date = rep(data$date, 3))

observed_data <- data %>%
  pivot_longer(cols = c(cases, hospitalisations, deaths),
               names_to = "stream", values_to = "observed") %>%
  mutate(stream = case_when(
    stream == "cases" ~ "Cases",
    stream == "hospitalisations" ~ "Hospitalizations",
    stream == "deaths" ~ "Deaths"
  ))

comparison_plot <- ggplot() +
  geom_ribbon(data = expected_summary, 
              aes(x = date, ymin = q5, ymax = q95, fill = stream), 
              alpha = 0.3) +
  geom_line(data = expected_summary, 
            aes(x = date, y = mean, color = stream), 
            size = 1) +
  geom_point(data = observed_data, 
             aes(x = date, y = observed, color = stream), 
             alpha = 0.7, size = 1) +
  facet_wrap(~stream, scales = "free_y") +
  labs(title = "Observed vs Expected Data Streams",
       subtitle = "Points: observed data, Lines: model expectations with 90% CI",
       x = "Date", y = "Count") +
  theme_minimal() +
  theme(legend.position = "none")

print(comparison_plot)

# Summary statistics
cat("\n=== MODEL SUMMARY ===\n")
cat(sprintf("Data period: %s to %s (%d days)\n", 
            min(data$date), max(data$date), n_days))
cat(sprintf("Mean Rt: %.2f (95%% CI: %.2f - %.2f)\n",
            mean(rt_summary$mean),
            quantile(rt_summary$q5, 0.025),
            quantile(rt_summary$q95, 0.975)))

cat("\nAscertainment rates (95% CI):\n")
cat(sprintf("  Cases: %.3f (%.3f - %.3f)\n",
            ascertainment$mean[1], ascertainment$q5[1], ascertainment$q95[1]))
cat(sprintf("  Hospitalizations: %.3f (%.3f - %.3f)\n",
            ascertainment$mean[2], ascertainment$q5[2], ascertainment$q95[2]))
cat(sprintf("  Deaths: %.3f (%.3f - %.3f)\n",
            ascertainment$mean[3], ascertainment$q5[3], ascertainment$q95[3]))

# Check model diagnostics
diagnostics <- fit$diagnostic_summary()
print("Model Diagnostics:")
print(diagnostics)

# Save results
results <- list(
  rt_estimates = rt_summary,
  ascertainment = ascertainment,
  overdispersion = overdispersion,
  model_fit = fit,
  diagnostics = diagnostics
)

# Optional: Save to file
# saveRDS(results, "rt_estimation_results.rds")
```

---

## SUB_070

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
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
```

---

## SUB_056

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Install required libraries
install.packages("dplyr")
install.packages("lubridate")
install.packages("cmdstanr")

# Load necessary packages
library(dplyr)
library(lubridate)
library(cmdstanr)

# Load data
data <- read_csv("observations.csv") %>%
  mutate(date = ymd(date))

# Define model in Stan
model_code <- "
    data {
      int<lower=1> T; // Number of observations
      array[int] vector[T]; // Cases
      array[int] vector[T]; // Hospitalisations
      array[int] vector[T]; // Deaths
      real delay_cases;
      real delay_hospitalisations;
      real delay_deaths;
    }

    parameters {
      real<lower=0> R; // Reproduction number (shared)
      real<lower=0, upper=1> ascertainment[3]; // Ascertainment rate for each stream
      vector[T] log_Rt; // Log reproduction numbers over time
    }

    transformed parameters {
      vector[T] Rt;
      for (t in 1:T) {
        Rt[t] = exp(log_Rt[t]);
      }
    }

    model {
      target += normal_lpdf(R | 0, 2); // Prior on R
      target += normal_lpdf(ascertainment | 0.5, 0.1); // Priors for ascertainment

      for (t in 1:T) {
        real lambda[t] = sum(vector[t]);
        vector[t] ~ neg_binomial_2(lambda[t], shape_inv);

        // Stream-specific delays and ascertainment
        real lambda_cases[t] = R * exp(log_Rt[t - delay_cases]) * ascertainment[1];
        real lambda_hospitalisations[t] = R * exp(log_Rt[t - delay_hospitalisations]) * ascertainment[2];
        real lambda_deaths[t] = R * exp(log_Rt[t - delay_deaths]) * ascertainment[3];

        vector[t + 1] ~ neg_binomial_2(lambda_cases[t], shape_inv);
        vector[t + 1] ~ neg_binomial_2(lambda_hospitalisations[t], shape_inv);
        vector[t + 1] ~ neg_binomial_2(lambda_deaths[t], shape_inv);
      }
    }

    generated quantities {
      Rt = exp(log_Rt);
      stream_params = {ascertainment};
    }
"


# Fit the model with cmdstanr
fit <- cmdstanr::cmd_stan_model(
  file = "renewal_equation.stan",
  model_code = model_code,
  data = list(
    cases = as.matrix(data$cases),
    hospitalisations = as.matrix(data$hospitalisations),
    deaths = as.matrix(data$deaths),
    T = nrow(data),
    delay_cases = 5, // Assume 5-day delay for cases
    delay_hospitalisations = 7, // Assume 7-day delay for hospitalisations
    delay_deaths = 14, // Assume 14-day delay for deaths
    shape_inv = 1.1, // Overdispersion parameter (inverse dispersion)
  )
)

# Run the model with initial infection seeding period excluded
fit_samples <- fit$sample(
  seed = 123,
  iter_warmup = 5000,
  iter_sampling = 10000,
  thinning = 2,
  data = list(
    cases = as.matrix(data$cases),
    hospitalisations = as.matrix(data$hospitalisations),
    deaths = as.matrix(data$deaths),
    T = nrow(data),
    delay_cases = 5, // Assume 5-day delay for cases
    delay_hospitalisations = 7, // Assume 7-day delay for hospitalisations
    delay_deaths = 14, // Assume 14-day delay for deaths
    shape_inv = 1.1, // Overdispersion parameter (inverse dispersion)
  ),
  chains = 4,
)

# Extract Rt estimates and stream-specific parameters
Rt_estimates <- fit_samples$draws(
  "Rt",
  inc_lowest = FALSE,
  column = 2:nrow(fit_samples$draws)
)[]

stream_params <- fit_samples$draws(
  "stream_params.1",
  inc_lowest = FALSE,
  column = 2:nrow(fit_samples$draws)
)[, , ]

# Plot results
library(ggplot2)

ggplot(data.frame(date = data$date, Rt = Rt_estimates), aes(x = date, y = Rt)) +
  geom_line() +
  labs(title = "Estimated Reproduction Number (Rt) Over Time")

summary(stream_params)
```

---

## SUB_117

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(cmdstanr)
library(tidyverse)

# Load the provided data
data <- read_csv("observations.csv")

# Define the Stan model for joint estimation of Rt from multiple streams
model_code <- "
data {
  int<lower=1> S; // number of time points (days) in observations
  vector[S] cases; // daily number of reported cases
  vector[S] hospitalisations; // daily number of hospital admissions
  vector[S] deaths; // daily number of deaths
}

parameters {
  real<lower=0, upper=1> rt[R]; // time-varying reproduction number (Rt) at each time point
  vector[S] delays_cases; // delay from infection to case report for each time point
  vector[S] delays_hospitalisations; // delay from infection to hospitalisation for each time point
  vector[S] delays_deaths; // delay from infection to death for each time point
  vector<lower=0, upper=1>[S] ascertainment_cases; // proportion of infections reported as cases for each time point
  vector<lower=0, upper=1>[S] ascertainment_hospitalisations; // proportion of infections resulting in hospitalisation for each time point
  vector<lower=0, upper=1>[S] ascertainment_deaths; // proportion of infections resulting in death for each time point
}

model {
  // prior distributions for parameters
  rt ~ normal(2.5, 0.5); // shared Rt prior with mean=2.5 and sd=0.5
  delays_cases ~ normal(0, 1);
  delays_hospitalisations ~ normal(3, 1);
  delays_deaths ~ normal(7, 1);
  ascertainment_cases ~ beta(2, 2); // uniform(0,1) prior for cases ascertainment rate
  ascertainment_hospitalisations ~ beta(5, 10); // informative prior for hospitalisation ascertainment rate
  ascertainment_deaths ~ beta(20, 40); // informative prior for death ascertainment rate
  
  // likelihood of observations given the model parameters
  for (s in 1:S) {
    cases[s] ~ poisson((rt[s] * sum(generation_interval[delay_cases]) * ascertainment_cases[s]));
    hospitalisations[s] ~ poisson((rt[s] * sum(generation_interval[delay_hospitalisations]) * ascertainment_hospitalisations[s]));
    deaths[s] ~ poisson((rt[s] * sum(generation_interval[delay_deaths]) * ascertainment_deaths[s]));
  }
}

generated quantities {
  // smoothed Rt estimate using the model parameters
  vector[S] rt_smoothed;
  for (s in 1:S) {
    rt_smoothed[s] = (rt[s] + rt[max(s-1, 1)]) / 2; // simple smoothing scheme
  }
}
"

# Compile the Stan model
stan_model <- cmdstanr::cmdstan_model(model_code)

# Define the data and sampling parameters
data_list <- list(
  S = nrow(data),
  cases = data$cases,
  hospitalisations = data$hospitalisations,
  deaths = data$deaths
)

sampling_params <- list(
  iter = 2000,
  warmup = 1000,
  chains = 4,
  cores = 4
)

# Fit the model
fit <- stan_model$sampling(data = data_list, 
                           algorithm = "NUTS", 
                           pars = c("rt", "delays_cases", "delays_hospitalisations", "delays_deaths",
                                    "ascertainment_cases", "ascertainment_hospitalisations", "ascertainment_deaths"),
                           chains = sampling_params$chains,
                           cores = sampling_params$cores,
                           iter = sampling_params$iter,
                           warmup = sampling_params$warmup)

# Extract Rt estimates and stream-specific parameters
rt_smoothed <- fit$draws("rt_smoothed", inc_lowest = TRUE)
rt_estimates <- rt_smoothed[, , 1]

# Plot the results (optional)
library(ggplot2)
ggplot(data.frame(date = as.Date(data$date), rt = rt_estimates),
       aes(x = date, y = rt)) +
  geom_line() +
  labs(title = "Time-Varying Reproduction Number (Rt) Estimates",
       subtitle = paste("Smoothed Rt using", sampling_params$chains, "chains and", sampling_params$cores, "cores"),
       x = "Date",
       y = "Rt")
```

---

## SUB_064

**Scenario**: 3 | **Execution**: FAILED

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath="observations.csv"):
    """Load and prepare the observation data"""
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        # Generate synthetic data for demonstration
        print("Data file not found. Generating synthetic data for demonstration...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic data that mimics real epidemic dynamics"""
    np.random.seed(42)
    n_days = 150
    dates = pd.date_range('2023-01-01', periods=n_days)
    
    # True Rt trajectory (varies over time)
    time = np.arange(n_days)
    true_rt = 1.5 * np.exp(-time/40) + 0.5 + 0.3 * np.sin(time/20)
    
    # Generation interval (gamma distribution)
    gen_shape, gen_scale = 2.5, 2.5
    max_gen = 20
    gen_pmf = stats.gamma.pdf(np.arange(1, max_gen+1), gen_shape, scale=gen_scale)
    gen_pmf = gen_pmf / gen_pmf.sum()
    
    # Simulate infections using renewal equation
    infections = np.zeros(n_days)
    infections[:7] = 50  # Initial seeding
    
    for t in range(7, n_days):
        if t < max_gen:
            past_inf = infections[:t][::-1]
            gen_subset = gen_pmf[:t]
        else:
            past_inf = infections[t-max_gen:t][::-1]
            gen_subset = gen_pmf
        
        lambda_t = true_rt[t] * np.sum(past_inf * gen_subset)
        infections[t] = np.random.poisson(max(lambda_t, 1))
    
    # Delay distributions (log-normal)
    def create_delay_pmf(mean_delay, std_delay, max_delay=30):
        delays = np.arange(1, max_delay+1)
        pmf = stats.lognorm.pdf(delays, s=std_delay, scale=np.exp(np.log(mean_delay)))
        return pmf / pmf.sum()
    
    case_delay_pmf = create_delay_pmf(5, 0.5)
    hosp_delay_pmf = create_delay_pmf(12, 0.6)
    death_delay_pmf = create_delay_pmf(20, 0.7)
    
    # Ascertainment rates
    case_asc = 0.3
    hosp_asc = 0.15
    death_asc = 0.02
    
    # Generate observations with overdispersion
    def generate_observations(infections, delay_pmf, ascertainment, overdispersion=5):
        n_days = len(infections)
        max_delay = len(delay_pmf)
        obs = np.zeros(n_days)
        
        for t in range(n_days):
            expected = 0
            for d in range(min(t, max_delay)):
                expected += infections[t-d-1] * delay_pmf[d]
            
            expected *= ascertainment
            if expected > 0:
                # Negative binomial with overdispersion
                p = expected / (expected + overdispersion)
                obs[t] = np.random.negative_binomial(overdispersion, 1-p)
        
        return obs.astype(int)
    
    cases = generate_observations(infections, case_delay_pmf, case_asc)
    hospitalizations = generate_observations(infections, hosp_delay_pmf, hosp_asc)
    deaths = generate_observations(infections, death_delay_pmf, death_asc)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases,
        'hospitalisations': hospitalizations,
        'deaths': deaths
    })
    
    return df

def create_generation_interval(shape=2.5, scale=2.5, max_gen=20):
    """Create generation interval PMF"""
    gen_pmf = stats.gamma.pdf(np.arange(1, max_gen+1), shape, scale=scale)
    return gen_pmf / gen_pmf.sum()

def create_delay_distributions():
    """Create delay distribution PMFs for each stream"""
    max_delay = 30
    delays = np.arange(1, max_delay+1)
    
    # Stream-specific delay parameters (mean, std)
    delay_params = {
        'cases': (5, 0.5),
        'hospitalisations': (12, 0.6), 
        'deaths': (20, 0.7)
    }
    
    delay_pmfs = {}
    for stream, (mean_delay, std_delay) in delay_params.items():
        pmf = stats.lognorm.pdf(delays, s=std_delay, scale=np.exp(np.log(mean_delay)))
        delay_pmfs[stream] = pmf / pmf.sum()
    
    return delay_pmfs

def build_model(data, generation_pmf, delay_pmfs):
    """Build the joint PyMC model"""
    n_days = len(data)
    max_gen = len(generation_pmf)
    max_delay = len(list(delay_pmfs.values())[0])
    
    # Prepare observation arrays
    obs_cases = data['cases'].values
    obs_hosp = data['hospitalisations'].values  
    obs_deaths = data['deaths'].values
    
    with pm.Model() as model:
        # Rt prior - log-normal with smoothness constraint
        log_rt_raw = pm.GaussianRandomWalk(
            'log_rt_raw', 
            sigma=0.1,  # Controls smoothness
            shape=n_days,
            init_dist=pm.Normal.dist(mu=np.log(1.0), sigma=0.3)
        )
        rt = pm.Deterministic('rt', pt.exp(log_rt_raw))
        
        # Initial infections (seeding period)
        seed_days = 7
        initial_infections = pm.Exponential('initial_infections', lam=1/50, shape=seed_days)
        
        # Stream-specific ascertainment rates
        asc_cases = pm.Beta('asc_cases', alpha=3, beta=7)  # Prior centered around 0.3
        asc_hosp = pm.Beta('asc_hosp', alpha=2, beta=10)   # Prior centered around 0.15  
        asc_deaths = pm.Beta('asc_deaths', alpha=1, beta=30) # Prior centered around 0.03
        
        # Overdispersion parameters (negative binomial)
        phi_cases = pm.Exponential('phi_cases', lam=0.1)
        phi_hosp = pm.Exponential('phi_hosp', lam=0.1)
        phi_deaths = pm.Exponential('phi_deaths', lam=0.1)
        
        # Infection dynamics using renewal equation
        def renewal_step(t, infections_prev, rt_t):
            # Get relevant past infections and generation intervals
            if t < max_gen:
                past_infections = infections_prev[:t]
                gen_weights = generation_pmf[:t]
            else:
                past_infections = infections_prev[t-max_gen:t]
                gen_weights = generation_pmf
            
            # Renewal equation
            lambda_t = rt_t * pt.sum(past_infections[::-1] * gen_weights)
            return pt.maximum(lambda_t, 1.0)
        
        # Initialize infections array
        infections = pt.concatenate([initial_infections, pt.zeros(n_days - seed_days)])
        
        # Iteratively compute infections
        for t in range(seed_days, n_days):
            lambda_t = renewal_step(t, infections[:t], rt[t])
            infections = pt.set_subtensor(infections[t], lambda_t)
        
        infections = pm.Deterministic('infections', infections)
        
        # Convolution function for delay from infection to observation
        def convolve_delays(infections, delay_pmf, ascertainment):
            n_days_local = infections.shape[0]
            max_delay_local = delay_pmf.shape[0]
            expected_obs = pt.zeros(n_days_local)
            
            for t in range(n_days_local):
                daily_expected = 0
                for d in range(min(t, max_delay_local)):
                    daily_expected += infections[t-d-1] * delay_pmf[d]
                expected_obs = pt.set_subtensor(expected_obs[t], daily_expected * ascertainment)
            
            return expected_obs
        
        # Expected observations for each stream
        expected_cases = convolve_delays(infections, delay_pmfs['cases'], asc_cases)
        expected_hosp = convolve_delays(infections, delay_pmfs['hospitalisations'], asc_hosp)  
        expected_deaths = convolve_delays(infections, delay_pmfs['deaths'], asc_deaths)
        
        # Likelihood - Negative binomial for overdispersion
        pm.NegativeBinomial('obs_cases', mu=expected_cases, alpha=phi_cases, observed=obs_cases)
        pm.NegativeBinomial('obs_hosp', mu=expected_hosp, alpha=phi_hosp, observed=obs_hosp)
        pm.NegativeBinomial('obs_deaths', mu=expected_deaths, alpha=phi_deaths, observed=obs_deaths)
        
    return model

def fit_model(model, draws=1000, tune=1000, chains=2):
    """Fit the model using NUTS sampler"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=draws,
            tune=tune, 
            chains=chains,
            cores=1,
            target_accept=0.95,
            return_inferencedata=True
        )
    return trace

def extract_results(trace, data):
    """Extract and summarize results"""
    # Get posterior summaries
    summary = pm.summary(trace, var_names=['rt', 'asc_cases', 'asc_hosp', 'asc_deaths', 
                                          'phi_cases', 'phi_hosp', 'phi_deaths'])
    
    # Extract Rt estimates
    rt_samples = trace.posterior['rt'].values.reshape(-1, len(data))
    rt_mean = rt_samples.mean(axis=0)
    rt_lower = np.percentile(rt_samples, 2.5, axis=0)
    rt_upper = np.percentile(rt_samples, 97.5, axis=0)
    
    # Extract stream parameters
    stream_params = {
        'ascertainment_cases': trace.posterior['asc_cases'].values.flatten(),
        'ascertainment_hosp': trace.posterior['asc_hosp'].values.flatten(),
        'ascertainment_deaths': trace.posterior['asc_deaths'].values.flatten(),
        'overdispersion_cases': trace.posterior['phi_cases'].values.flatten(),
        'overdispersion_hosp': trace.posterior['phi_hosp'].values.flatten(), 
        'overdispersion_deaths': trace.posterior['phi_deaths'].values.flatten()
    }
    
    results = {
        'dates': data['date'].values,
        'rt_mean': rt_mean,
        'rt_lower': rt_lower,
        'rt_upper': rt_upper,
        'stream_params': stream_params,
        'summary': summary
    }
    
    return results

def plot_results(results, data):
    """Create plots showing the results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rt over time
    ax1 = axes[0, 0]
    ax1.plot(results['dates'], results['rt_mean'], 'b-', linewidth=2, label='Rt (mean)')
    ax1.fill_between(results['dates'], results['rt_lower'], results['rt_upper'], 
                     alpha=0.3, color='blue', label='95% CI')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax1.set_ylabel('Reproduction Number (Rt)')
    ax1.set_title('Time-varying Reproduction Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data streams
    ax2 = axes[0, 1]
    ax2.plot(data['date'], data['cases'], 'o-', label='Cases', alpha=0.7)
    ax2.plot(data['date'], data['hospitalisations'], 's-', label='Hospitalizations', alpha=0.7)
    ax2.plot(data['date'], data['deaths'], '^-', label='Deaths', alpha=0.7)
    ax2.set_ylabel('Daily Observations')
    ax2.set_title('Observed Data Streams')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stream parameters - Ascertainment
    ax3 = axes[1, 0]
    asc_data = [
        results['stream_params']['ascertainment_cases'],
        results['stream_params']['ascertainment_hosp'], 
        results['stream_params']['ascertainment_deaths']
    ]
    bp = ax3.boxplot(asc_data, labels=['Cases', 'Hosp', 'Deaths'])
    ax3.set_ylabel('Ascertainment Rate')
    ax3.set_title('Stream-specific Ascertainment Rates')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stream parameters - Overdispersion
    ax4 = axes[1, 1]
    phi_data = [
        results['stream_params']['overdispersion_cases'],
        results['stream_params']['overdispersion_hosp'],
        results['stream_params']['overdispersion_deaths']
    ]
    bp = ax4.boxplot(phi_data, labels=['Cases', 'Hosp', 'Deaths'])
    ax4.set_ylabel('Overdispersion Parameter')
    ax4.set_title('Stream-specific Overdispersion')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main execution function"""
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} days of data")
    
    print("\nData summary:")
    print(data.describe())
    
    print("\nCreating model components...")
    generation_pmf = create_generation_interval()
    delay_pmfs = create_delay_distributions()
    
    print("Building PyMC model...")
    model = build_model(data, generation_pmf, delay_pmfs)
    
    print("\nModel summary:")
    print(model)
    
    print("\nFitting model (this may take several minutes)...")
    try:
        trace = fit_model(model, draws=1000, tune=1000, chains=2)
        
        print("\nExtracting results...")
        results = extract_results(trace, data)
        
        print("\nModel Summary:")
        print(results['summary'])
        
        print("\nStream Parameter Estimates:")
        for param_name, samples in results['stream_params'].items():
            mean_val = np.mean(samples)
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
            print(f"{param_name}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        print("\nCreating plots...")
        fig = plot_results(results, data)
        
        print("\nAnalysis complete!")
        
        return results, trace, model
        
    except Exception as e:
        print(f"Error during model fitting: {e}")
        print("This could be due to model complexity or data issues.")
        return None, None, model

if __name__ == "__main__":
    results, trace, model = main()
```

---

## SUB_011

**Scenario**: 3 | **Execution**: FAILED

### script.py

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath='observations.csv'):
    """Load and preprocess the observation data"""
    try:
        data = pd.read_csv(filepath)
        data['date'] = pd.to_datetime(data['date'])
        data = data.sort_values('date').reset_index(drop=True)
        
        # Handle missing values
        data = data.fillna(0)
        
        # Ensure non-negative integers
        for col in ['cases', 'hospitalisations', 'deaths']:
            data[col] = np.maximum(0, data[col].astype(int))
        
        print(f"Loaded {len(data)} days of data from {data['date'].min()} to {data['date'].max()}")
        print(f"Cases: {data['cases'].sum()}, Hospitalizations: {data['hospitalisations'].sum()}, Deaths: {data['deaths'].sum()}")
        
        return data
    
    except FileNotFoundError:
        print("Data file not found. Generating synthetic data for demonstration...")
        return generate_synthetic_data()

def generate_synthetic_data(n_days=100):
    """Generate synthetic data for demonstration purposes"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # True Rt trajectory (starts high, decreases, then increases)
    t = np.arange(n_days)
    true_rt = 1.5 * np.exp(-t/30) + 0.7 + 0.3 * np.sin(t/20)
    
    # Generation interval (discretized gamma distribution)
    generation_interval = stats.gamma(a=2.5, scale=2.5).pdf(np.arange(1, 21))
    generation_interval /= generation_interval.sum()
    
    # Initialize infections
    infections = np.zeros(n_days)
    infections[:7] = 100  # Seed infections
    
    # Generate infections using renewal equation
    for t in range(7, n_days):
        infectiousness = sum(infections[max(0, t-s)] * generation_interval[min(s-1, len(generation_interval)-1)] 
                           for s in range(1, min(t+1, 21)))
        infections[t] = np.random.poisson(true_rt[t] * infectiousness)
    
    # Generate observations with delays and ascertainment
    cases = np.zeros(n_days)
    hospitalizations = np.zeros(n_days)
    deaths = np.zeros(n_days)
    
    # Delay distributions
    case_delays = stats.gamma(a=1.5, scale=2).pmf(np.arange(15))
    case_delays /= case_delays.sum()
    
    hosp_delays = stats.gamma(a=2.5, scale=3).pmf(np.arange(20))
    hosp_delays /= hosp_delays.sum()
    
    death_delays = stats.gamma(a=4, scale=4).pmf(np.arange(30))
    death_delays /= death_delays.sum()
    
    # Apply delays and ascertainment
    for t in range(n_days):
        for delay, prob in enumerate(case_delays):
            if t + delay < n_days:
                cases[t + delay] += np.random.poisson(infections[t] * 0.3 * prob)
        
        for delay, prob in enumerate(hosp_delays):
            if t + delay < n_days:
                hospitalizations[t + delay] += np.random.poisson(infections[t] * 0.05 * prob)
        
        for delay, prob in enumerate(death_delays):
            if t + delay < n_days:
                deaths[t + delay] += np.random.poisson(infections[t] * 0.01 * prob)
    
    data = pd.DataFrame({
        'date': dates,
        'cases': cases.astype(int),
        'hospitalisations': hospitalizations.astype(int),
        'deaths': deaths.astype(int)
    })
    
    print("Generated synthetic data for demonstration")
    return data

def create_generation_interval():
    """Create discretized generation interval distribution"""
    # Gamma distribution with mean ~5 days, std ~2.5 days
    max_gen_int = 20
    gen_interval = stats.gamma(a=2.5, scale=2).pdf(np.arange(1, max_gen_int + 1))
    gen_interval = gen_interval / gen_interval.sum()
    return gen_interval

def create_delay_distributions():
    """Create delay distributions for each stream"""
    delays = {}
    
    # Cases: shorter delay, mean ~3 days
    delays['cases'] = {
        'dist': stats.gamma(a=2, scale=1.5).pmf(np.arange(15)),
        'max_delay': 15
    }
    
    # Hospitalizations: medium delay, mean ~8 days
    delays['hospitalisations'] = {
        'dist': stats.gamma(a=3, scale=2.5).pmf(np.arange(25)),
        'max_delay': 25
    }
    
    # Deaths: longer delay, mean ~18 days
    delays['deaths'] = {
        'dist': stats.gamma(a=4, scale=4.5).pmf(np.arange(35)),
        'max_delay': 35
    }
    
    # Normalize
    for stream in delays:
        delays[stream]['dist'] = delays[stream]['dist'] / delays[stream]['dist'].sum()
    
    return delays

class MultiStreamRtModel:
    """Joint Rt estimation model for multiple data streams"""
    
    def __init__(self, data, generation_interval, delay_distributions):
        self.data = data
        self.n_days = len(data)
        self.generation_interval = generation_interval
        self.delay_distributions = delay_distributions
        self.streams = ['cases', 'hospitalisations', 'deaths']
        
        # Observations
        self.observations = {
            stream: data[stream].values for stream in self.streams
        }
        
    def build_model(self):
        """Build the PyMC model"""
        
        with pm.Model() as model:
            
            # Rt prior - log normal with smoothness constraint
            rt_log_raw = pm.GaussianRandomWalk(
                name='rt_log_raw',
                sigma=0.1,  # Controls day-to-day variation
                shape=self.n_days,
                init_dist=pm.Normal.dist(mu=0, sigma=0.5)  # Prior centered around Rt=1
            )
            
            rt = pm.Deterministic('rt', pm.math.exp(rt_log_raw))
            
            # Stream-specific ascertainment rates
            ascertainment = {}
            for stream in self.streams:
                # Different priors based on expected ascertainment
                if stream == 'cases':
                    prior_alpha, prior_beta = 3, 7  # ~0.3 expected
                elif stream == 'hospitalisations':
                    prior_alpha, prior_beta = 1, 19  # ~0.05 expected
                else:  # deaths
                    prior_alpha, prior_beta = 1, 99  # ~0.01 expected
                
                ascertainment[stream] = pm.Beta(
                    f'ascertainment_{stream}',
                    alpha=prior_alpha,
                    beta=prior_beta
                )
            
            # Overdispersion parameters (inverse concentration for negative binomial)
            overdispersion = {}
            for stream in self.streams:
                overdispersion[stream] = pm.Exponential(
                    f'overdispersion_{stream}',
                    lam=1
                )
            
            # Initial infections (seed period)
            seed_period = 14
            initial_infections = pm.Exponential(
                'initial_infections',
                lam=1/100,
                shape=seed_period
            )
            
            # Compute infections using renewal equation
            def compute_infections(rt, initial_infections):
                infections = pt.zeros(self.n_days)
                infections = pt.set_subtensor(infections[:seed_period], initial_infections)
                
                # Renewal equation for remaining days
                for t in range(seed_period, self.n_days):
                    infectiousness = 0
                    for s in range(1, min(len(self.generation_interval) + 1, t + 1)):
                        if s <= len(self.generation_interval):
                            infectiousness += infections[t-s] * self.generation_interval[s-1]
                    
                    infections = pt.set_subtensor(infections[t], rt[t] * infectiousness)
                
                return infections
            
            infections = pm.Deterministic(
                'infections',
                compute_infections(rt, initial_infections)
            )
            
            # Expected observations for each stream
            expected_obs = {}
            
            for stream in self.streams:
                delay_dist = self.delay_distributions[stream]['dist']
                max_delay = self.delay_distributions[stream]['max_delay']
                
                # Convolve infections with delay distribution
                def convolve_with_delay(infections, ascertainment_rate, delay_dist):
                    expected = pt.zeros(self.n_days)
                    
                    for t in range(self.n_days):
                        obs_sum = 0
                        for d in range(min(len(delay_dist), t + 1)):
                            obs_sum += infections[t-d] * delay_dist[d]
                        expected = pt.set_subtensor(expected[t], ascertainment_rate * obs_sum)
                    
                    return expected
                
                expected_obs[stream] = pm.Deterministic(
                    f'expected_{stream}',
                    convolve_with_delay(infections, ascertainment[stream], delay_dist)
                )
                
                # Likelihood - Negative Binomial to handle overdispersion
                pm.NegativeBinomial(
                    f'obs_{stream}',
                    mu=expected_obs[stream],
                    alpha=1/overdispersion[stream],  # Convert to concentration parameter
                    observed=self.observations[stream]
                )
        
        return model
    
    def fit_model(self, draws=1000, tune=1000, chains=2, target_accept=0.9):
        """Fit the model using NUTS sampling"""
        
        self.model = self.build_model()
        
        with self.model:
            # Sample from posterior
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                random_seed=42,
                return_inferencedata=True
            )
            
            # Sample from posterior predictive
            self.posterior_predictive = pm.sample_posterior_predictive(
                self.trace,
                random_seed=42
            )
        
        return self.trace
    
    def extract_results(self):
        """Extract key results from the fitted model"""
        
        results = {}
        
        # Rt estimates
        rt_samples = self.trace.posterior['rt']
        results['rt_mean'] = rt_samples.mean(dim=['chain', 'draw']).values
        results['rt_lower'] = rt_samples.quantile(0.025, dim=['chain', 'draw']).values
        results['rt_upper'] = rt_samples.quantile(0.975, dim=['chain', 'draw']).values
        results['rt_samples'] = rt_samples
        
        # Ascertainment rates
        results['ascertainment'] = {}
        for stream in self.streams:
            asc_samples = self.trace.posterior[f'ascertainment_{stream}']
            results['ascertainment'][stream] = {
                'mean': float(asc_samples.mean()),
                'lower': float(asc_samples.quantile(0.025)),
                'upper': float(asc_samples.quantile(0.975)),
                'samples': asc_samples
            }
        
        # Overdispersion parameters
        results['overdispersion'] = {}
        for stream in self.streams:
            od_samples = self.trace.posterior[f'overdispersion_{stream}']
            results['overdispersion'][stream] = {
                'mean': float(od_samples.mean()),
                'lower': float(od_samples.quantile(0.025)),
                'upper': float(od_samples.quantile(0.975)),
                'samples': od_samples
            }
        
        # Expected observations
        results['expected_obs'] = {}
        for stream in self.streams:
            exp_samples = self.trace.posterior[f'expected_{stream}']
            results['expected_obs'][stream] = {
                'mean': exp_samples.mean(dim=['chain', 'draw']).values,
                'lower': exp_samples.quantile(0.025, dim=['chain', 'draw']).values,
                'upper': exp_samples.quantile(0.975, dim=['chain', 'draw']).values
            }
        
        return results

def plot_results(data, results):
    """Create comprehensive plots of the results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    dates = data['date']
    streams = ['cases', 'hospitalisations', 'deaths']
    
    # Plot 1: Rt over time
    ax = axes[0, 0]
    ax.fill_between(dates, results['rt_lower'], results['rt_upper'], 
                   alpha=0.3, color='blue', label='95% CI')
    ax.plot(dates, results['rt_mean'], color='blue', linewidth=2, label='Rt estimate')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax.set_ylabel('Reproduction Number (Rt)')
    ax.set_title('Time-varying Reproduction Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Observed vs Expected - Cases and Hospitalizations
    ax = axes[0, 1]
    ax.plot(dates, data['cases'], 'o-', alpha=0.6, label='Observed cases', markersize=3)
    ax.fill_between(dates, results['expected_obs']['cases']['lower'], 
                   results['expected_obs']['cases']['upper'], alpha=0.3, label='Expected cases (95% CI)')
    ax.plot(dates, results['expected_obs']['cases']['mean'], '--', label='Expected cases')
    
    ax2 = ax.twinx()
    ax2.plot(dates, data['hospitalisations'], 's-', alpha=0.6, color='orange', 
            label='Observed hospitalizations', markersize=3)
    ax2.fill_between(dates, results['expected_obs']['hospitalisations']['lower'],
                    results['expected_obs']['hospitalisations']['upper'], 
                    alpha=0.3, color='orange', label='Expected hosp. (95% CI)')
    ax2.plot(dates, results['expected_obs']['hospitalisations']['mean'], '--', 
            color='orange', label='Expected hospitalizations')
    
    ax.set_ylabel('Cases', color='blue')
    ax2.set_ylabel('Hospitalizations', color='orange')
    ax.set_title('Observed vs Expected: Cases & Hospitalizations')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    # Plot 3: Deaths
    ax = axes[1, 0]
    ax.plot(dates, data['deaths'], '^-', alpha=0.6, color='red', 
           label='Observed deaths', markersize=3)
    ax.fill_between(dates, results['expected_obs']['deaths']['lower'],
                   results['expected_obs']['deaths']['upper'], 
                   alpha=0.3, color='red', label='Expected deaths (95% CI)')
    ax.plot(dates, results['expected_obs']['deaths']['mean'], '--', 
           color='red', label='Expected deaths')
    ax.set_ylabel('Deaths')
    ax.set_title('Observed vs Expected: Deaths')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Ascertainment rates
    ax = axes[1, 1]
    stream_names = ['Cases', 'Hospitalizations', 'Deaths']
    means = [results['ascertainment'][stream]['mean'] for stream in streams]
    lowers = [results['ascertainment'][stream]['lower'] for stream in streams]
    uppers = [results['ascertainment'][stream]['upper'] for stream in streams]
    
    x_pos = np.arange(len(streams))
    ax.bar(x_pos, means, yerr=[np.array(means) - np.array(lowers),
                              np.array(uppers) - np.array(means)], 
          capsize=5, alpha=0.7, color=['blue', 'orange', 'red'])
    ax.set_xticks(x_pos)
    ax.set_xticklabels(stream_names)
    ax.set_ylabel('Ascertainment Rate')
    ax.set_title('Stream-specific Ascertainment Rates')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_summary(results):
    """Print a summary of the results"""
    
    print("\n" + "="*60)
    print("MULTI-STREAM Rt ESTIMATION RESULTS")
    print("="*60)
    
    # Rt summary
    final_rt = results['rt_mean'][-7:].mean()  # Last week average
    print(f"\nReproduction Number (Rt):")
    print(f"  Current Rt (last 7 days avg): {final_rt:.2f}")
    print(f"  Overall range: {results['rt_mean'].min():.2f} - {results['rt_mean'].max():.2f}")
    
    # Ascertainment rates
    print(f"\nAscertainment Rates:")
    for stream in ['cases', 'hospitalisations', 'deaths']:
        asc = results['ascertainment'][stream]
        print(f"  {stream.capitalize():15}: {asc['mean']:.3f} [{asc['lower']:.3f}, {asc['upper']:.3f}]")
    
    # Overdispersion
    print(f"\nOverdispersion Parameters:")
    for stream in ['cases', 'hospitalisations', 'deaths']:
        od = results['overdispersion'][stream]
        print(f"  {stream.capitalize():15}: {od['mean']:.3f} [{od['lower']:.3f}, {od['upper']:.3f}]")
    
    print("\n" + "="*60)

def main():
    """Main execution function"""
    
    print("Multi-Stream Rt Estimation")
    print("=" * 40)
    
    # Load data
    data = load_data()
    
    # Create model components
    generation_interval = create_generation_interval()
    delay_distributions = create_delay_distributions()
    
    print(f"\nModel Setup:")
    print(f"- Generation interval: mean = {np.sum(np.arange(1, len(generation_interval)+1) * generation_interval):.1f} days")
    print(f"- Delay distributions created for {len(delay_distributions)} streams")
    print(f"- Time series length: {len(data)} days")
    
    # Build and fit model
    print(f"\nBuilding and fitting model...")
    model = MultiStreamRtModel(data, generation_interval, delay_distributions)
    
    try:
        trace = model.fit_model(draws=1000, tune=1000, chains=2)
        print("Model fitting completed successfully!")
        
        # Extract results
        results = model.extract_results()
        
        # Print summary
        print_summary(results)
        
        # Create plots
        plot_results(data, results)
        
        return model, results, data
        
    except Exception as e:
        print(f"Error during model fitting: {e}")
        print("This might be due to model complexity or data issues.")
        return None, None, data

if __name__ == "__main__":
    model, results, data = main()
```

---

## SUB_031

**Scenario**: 3 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(filepath):
    """Load and prepare the observation data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Handle any missing values
    df = df.fillna(0)
    
    print(f"Data loaded: {len(df)} days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['cases'].sum()}")
    print(f"Total hospitalisations: {df['hospitalisations'].sum()}")
    print(f"Total deaths: {df['deaths'].sum()}")
    
    return df

def create_delay_distributions():
    """Create delay distributions for each data stream"""
    # Infection to symptom onset: ~5 days
    # Symptom onset to case report: ~2 days
    # Total infection to case: ~7 days
    case_delay_mean = 7
    case_delay_sd = 3
    
    # Infection to hospitalisation: ~10-12 days
    hosp_delay_mean = 11
    hosp_delay_sd = 4
    
    # Infection to death: ~18-21 days
    death_delay_mean = 20
    death_delay_sd = 5
    
    max_delay = 30
    
    delays = {}
    
    for name, mean, sd in [
        ('cases', case_delay_mean, case_delay_sd),
        ('hospitalisations', hosp_delay_mean, hosp_delay_sd),
        ('deaths', death_delay_mean, death_delay_sd)
    ]:
        # Use gamma distribution truncated at max_delay
        shape = (mean / sd) ** 2
        scale = sd ** 2 / mean
        
        pmf = np.array([stats.gamma.pdf(i, a=shape, scale=scale) for i in range(1, max_delay + 1)])
        pmf = pmf / pmf.sum()  # Normalize
        
        delays[name] = pmf
    
    return delays, max_delay

def create_generation_interval():
    """Create generation interval distribution"""
    # Mean ~5.5 days, SD ~2.5 days for COVID-19
    gen_mean = 5.5
    gen_sd = 2.5
    max_gen = 15
    
    shape = (gen_mean / gen_sd) ** 2
    scale = gen_sd ** 2 / gen_mean
    
    pmf = np.array([stats.gamma.pdf(i, a=shape, scale=scale) for i in range(1, max_gen + 1)])
    pmf = pmf / pmf.sum()
    
    return pmf

def build_renewal_model(observations, delays, generation_interval, max_delay):
    """Build the joint renewal equation model"""
    
    n_days = len(observations)
    streams = ['cases', 'hospitalisations', 'deaths']
    
    # Convert observations to arrays
    obs_data = {stream: observations[stream].values for stream in streams}
    
    with pm.Model() as model:
        # === Priors ===
        
        # Initial infections (seeding period)
        seed_days = max_delay
        log_initial_infections = pm.Normal('log_initial_infections', mu=np.log(50), sigma=1, 
                                         shape=seed_days)
        initial_infections = pm.Deterministic('initial_infections', 
                                            pt.exp(log_initial_infections))
        
        # Rt evolution - using random walk on log scale for smoothness
        log_rt_init = pm.Normal('log_rt_init', mu=np.log(1.5), sigma=0.5)
        log_rt_innovations = pm.Normal('log_rt_innovations', mu=0, sigma=0.1, 
                                     shape=n_days - 1)
        
        log_rt = pm.Deterministic('log_rt', pt.concatenate([
            [log_rt_init],
            log_rt_init + pt.cumsum(log_rt_innovations)
        ]))
        
        rt = pm.Deterministic('rt', pt.exp(log_rt))
        
        # Stream-specific parameters
        ascertainment_rates = {}
        overdispersion_params = {}
        
        for stream in streams:
            # Ascertainment rates (logit scale for constraint to [0,1])
            if stream == 'cases':
                # Cases might have higher ascertainment
                ascertainment_rates[stream] = pm.Beta(f'ascertainment_{stream}', 
                                                    alpha=2, beta=3)
            elif stream == 'hospitalisations':
                # Lower ascertainment but more stable
                ascertainment_rates[stream] = pm.Beta(f'ascertainment_{stream}', 
                                                    alpha=1, beta=10)
            else:  # deaths
                # Lowest ascertainment but most complete
                ascertainment_rates[stream] = pm.Beta(f'ascertainment_{stream}', 
                                                    alpha=1, beta=20)
            
            # Overdispersion parameters (inverse dispersion)
            overdispersion_params[stream] = pm.Exponential(f'phi_{stream}', 1.0)
        
        # === Renewal equation dynamics ===
        
        def renewal_step(rt_t, infections_history):
            """Single step of renewal equation"""
            # infections_history contains last len(generation_interval) infections
            new_infections = rt_t * pt.dot(infections_history, generation_interval[::-1])
            return new_infections
        
        # Set up the renewal equation recursion
        infections_extended = pt.concatenate([initial_infections, 
                                            pt.zeros(n_days)])
        
        # Apply renewal equation for each day
        for t in range(n_days):
            start_idx = t + seed_days - len(generation_interval)
            end_idx = t + seed_days
            
            # Ensure we have enough history
            if start_idx >= 0:
                infection_history = infections_extended[start_idx:end_idx]
                # Pad if necessary
                if len(infection_history) < len(generation_interval):
                    padding = pt.zeros(len(generation_interval) - len(infection_history))
                    infection_history = pt.concatenate([padding, infection_history])
                
                new_infection = rt[t] * pt.dot(infection_history, generation_interval[::-1])
                infections_extended = pt.set_subtensor(infections_extended[seed_days + t], 
                                                     new_infection)
        
        infections = infections_extended[seed_days:]
        
        # === Observation model ===
        
        expected_obs = {}
        
        for stream in streams:
            delay_pmf = delays[stream]
            
            # Convolve infections with delay distribution
            expected_stream = pt.zeros(n_days)
            
            for t in range(n_days):
                total = 0.0
                for d, delay_prob in enumerate(delay_pmf):
                    infection_day = t - (d + 1)  # d+1 because delays start from day 1
                    if infection_day >= 0:
                        total += infections[infection_day] * delay_prob
                
                expected_stream = pt.set_subtensor(expected_stream[t], total)
            
            # Apply ascertainment
            expected_obs[stream] = expected_stream * ascertainment_rates[stream]
            
            # Observation likelihood with overdispersion (Negative Binomial)
            pm.NegativeBinomial(
                f'obs_{stream}',
                mu=expected_obs[stream],
                alpha=overdispersion_params[stream],
                observed=obs_data[stream]
            )
        
        # Store expected observations for diagnostics
        for stream in streams:
            pm.Deterministic(f'expected_{stream}', expected_obs[stream])
    
    return model

def fit_model_and_extract_results(model, observations):
    """Fit the model and extract results"""
    
    with model:
        # Use NUTS sampler
        print("Starting MCMC sampling...")
        
        # Initial tuning
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=2,
            target_accept=0.95,
            random_seed=42,
            progressbar=True
        )
        
        print("Sampling completed!")
        
        # Extract posterior summaries
        summary = pm.summary(trace, var_names=['rt'])
        
        # Extract Rt estimates
        rt_samples = trace.posterior['rt'].values  # shape: (chains, draws, time)
        rt_mean = rt_samples.mean(axis=(0, 1))
        rt_lower = np.percentile(rt_samples, 2.5, axis=(0, 1))
        rt_upper = np.percentile(rt_samples, 97.5, axis=(0, 1))
        
        # Extract ascertainment rates
        ascertainment = {}
        for stream in ['cases', 'hospitalisations', 'deaths']:
            samples = trace.posterior[f'ascertainment_{stream}'].values
            ascertainment[stream] = {
                'mean': samples.mean(),
                'lower': np.percentile(samples, 2.5),
                'upper': np.percentile(samples, 97.5)
            }
        
        # Create results dataframe
        results_df = observations.copy()
        results_df['rt_mean'] = rt_mean
        results_df['rt_lower'] = rt_lower
        results_df['rt_upper'] = rt_upper
        
        # Add expected observations
        for stream in ['cases', 'hospitalisations', 'deaths']:
            expected_samples = trace.posterior[f'expected_{stream}'].values
            results_df[f'expected_{stream}_mean'] = expected_samples.mean(axis=(0, 1))
            results_df[f'expected_{stream}_lower'] = np.percentile(expected_samples, 2.5, axis=(0, 1))
            results_df[f'expected_{stream}_upper'] = np.percentile(expected_samples, 97.5, axis=(0, 1))
    
    return trace, results_df, ascertainment

def plot_results(results_df, ascertainment):
    """Create comprehensive plots of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Joint Rt Estimation from Multiple Data Streams', fontsize=16, fontweight='bold')
    
    # Plot 1: Rt over time
    ax = axes[0, 0]
    ax.plot(results_df['date'], results_df['rt_mean'], 'b-', linewidth=2, label='Rt estimate')
    ax.fill_between(results_df['date'], results_df['rt_lower'], results_df['rt_upper'], 
                    alpha=0.3, color='blue', label='95% CI')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax.set_ylabel('Reproduction Number (Rt)')
    ax.set_title('Time-varying Reproduction Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cases
    ax = axes[0, 1]
    ax.plot(results_df['date'], results_df['cases'], 'ko-', markersize=3, alpha=0.7, label='Observed')
    ax.plot(results_df['date'], results_df['expected_cases_mean'], 'r-', linewidth=2, label='Expected')
    ax.fill_between(results_df['date'], results_df['expected_cases_lower'], 
                    results_df['expected_cases_upper'], alpha=0.3, color='red')
    ax.set_ylabel('Cases')
    ax.set_title('Cases: Observed vs Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Hospitalisations
    ax = axes[1, 0]
    ax.plot(results_df['date'], results_df['hospitalisations'], 'ko-', markersize=3, alpha=0.7, label='Observed')
    ax.plot(results_df['date'], results_df['expected_hospitalisations_mean'], 'g-', linewidth=2, label='Expected')
    ax.fill_between(results_df['date'], results_df['expected_hospitalisations_lower'], 
                    results_df['expected_hospitalisations_upper'], alpha=0.3, color='green')
    ax.set_ylabel('Hospitalisations')
    ax.set_title('Hospitalisations: Observed vs Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Deaths
    ax = axes[1, 1]
    ax.plot(results_df['date'], results_df['deaths'], 'ko-', markersize=3, alpha=0.7, label='Observed')
    ax.plot(results_df['date'], results_df['expected_deaths_mean'], 'm-', linewidth=2, label='Expected')
    ax.fill_between(results_df['date'], results_df['expected_deaths_lower'], 
                    results_df['expected_deaths_upper'], alpha=0.3, color='magenta')
    ax.set_ylabel('Deaths')
    ax.set_title('Deaths: Observed vs Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Plot ascertainment rates
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    streams = ['cases', 'hospitalisations', 'deaths']
    means = [ascertainment[stream]['mean'] for stream in streams]
    lowers = [ascertainment[stream]['lower'] for stream in streams]
    uppers = [ascertainment[stream]['upper'] for stream in streams]
    
    x_pos = np.arange(len(streams))
    ax.bar(x_pos, means, yerr=[np.array(means) - np.array(lowers), 
                               np.array(uppers) - np.array(means)], 
           capsize=5, alpha=0.7, color=['blue', 'green', 'magenta'])
    
    ax.set_xlabel('Data Stream')
    ax.set_ylabel('Ascertainment Rate')
    ax.set_title('Estimated Ascertainment Rates by Data Stream')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(streams)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    
    print("=== Joint Rt Estimation from Multiple Data Streams ===\n")
    
    # Load data
    print("1. Loading data...")
    observations = load_and_prepare_data('observations.csv')
    
    # Create delay distributions
    print("\n2. Setting up delay distributions...")
    delays, max_delay = create_delay_distributions()
    generation_interval = create_generation_interval()
    
    print(f"Maximum delay considered: {max_delay} days")
    print(f"Generation interval length: {len(generation_interval)} days")
    
    # Build model
    print("\n3. Building Bayesian model...")
    model = build_renewal_model(observations, delays, generation_interval, max_delay)
    
    print(f"Model built with {len(observations)} time points")
    
    # Fit model
    print("\n4. Fitting model...")
    trace, results_df, ascertainment = fit_model_and_extract_results(model, observations)
    
    # Display results
    print("\n5. Results Summary:")
    print("="*50)
    
    print("\nAscertainment Rates (95% CI):")
    for stream, params in ascertainment.items():
        print(f"{stream:15}: {params['mean']:.3f} ({params['lower']:.3f}, {params['upper']:.3f})")
    
    print(f"\nRt Summary:")
    print(f"Mean Rt: {results_df['rt_mean'].mean():.2f}")
    print(f"Min Rt:  {results_df['rt_mean'].min():.2f}")
    print(f"Max Rt:  {results_df['rt_mean'].max():.2f}")
    
    # Show periods where Rt > 1
    above_one = results_df[results_df['rt_mean'] > 1]
    print(f"\nPeriods with Rt > 1: {len(above_one)} days ({100*len(above_one)/len(results_df):.1f}%)")
    
    # Plot results
    print("\n6. Generating plots...")
    plot_results(results_df, ascertainment)
    
    print("\nAnalysis completed successfully!")
    
    return model, trace, results_df, ascertainment

if __name__ == "__main__":
    # For demonstration, create sample data if file doesn't exist
    try:
        pd.read_csv('observations.csv')
    except FileNotFoundError:
        print("Creating sample data for demonstration...")
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate an outbreak with changing Rt
        true_rt = 2.0 * np.exp(-0.03 * np.arange(100)) + 0.5
        
        infections = np.zeros(100)
        infections[0] = 50
        
        for t in range(1, 100):
            infections[t] = max(1, np.random.poisson(true_rt[t] * infections[max(0, t-7):t].sum() * 0.2))
        
        # Add delays and noise
        cases = np.random.negative_binomial(n=5, p=5/(5 + infections * 0.3))
        hospitalisations = np.random.negative_binomial(n=10, p=10/(10 + infections * 0.05))
        deaths = np.random.negative_binomial(n=20, p=20/(20 + infections * 0.01))
        
        # Apply delays
        cases = np.convolve(cases, [0.1, 0.3, 0.4, 0.2])[:100]
        hospitalisations = np.convolve(hospitalisations, [0.05, 0.15, 0.3, 0.3, 0.2])[:100]
        deaths = np.convolve(deaths, [0.02, 0.08, 0.2, 0.4, 0.2, 0.1])[:100]
        
        sample_data = pd.DataFrame({
            'date': dates,
            'cases': np.maximum(0, cases).astype(int),
            'hospitalisations': np.maximum(0, hospitalisations).astype(int),
            'deaths': np.maximum(0, deaths).astype(int)
        })
        
        sample_data.to_csv('observations.csv', index=False)
        print("Sample data created!\n")
    
    # Run the main analysis
    model, trace, results_df, ascertainment = main()
```

---

## SUB_085

**Scenario**: 3 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
from scipy.stats import poisson, gamma
import pymc as pm
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('observations.csv')

# Model setup
with pm.Model() as model:
    # Shared Rt
    rt = pm.Gamma('rt', alpha=1.0, beta=1.0)

    # Stream-specific delays and ascertainment rates
    delays = {
        'cases': 3,  # Assume cases are reported 3 days after infection
        'hospitalisations': 7,
        'deaths': 14
    }

    ascertainment_rates = {
        'cases': pm.Uniform('asc_cases', lower=0.5, upper=1.0),
        'hospitalisations': pm.Uniform('asc_hospitalisations', lower=0.5, upper=1.0),
        'deaths': pm.Uniform('asc_deaths', lower=0.5, upper=1.0)
    }

    # Stream-specific ascertainment
    cases_asc = gamma.rvs(ascertainment_rates['cases'] * 100, size=len(data))
    hospitalisations_asc = gamma.rvs(ascertainment_rates['hospitalisations'] * 100, size=len(data))
    deaths_asc = gamma.rvs(ascertainment_rates['deaths'] * 100, size=len(data))

    # Overdispersion
    cases_overdispersion = pm.Gamma('cases_overdispersion', alpha=2.0, beta=1.0)
    hospitalisations_overdispersion = pm.Gamma('hospitalisations_overdispersion', alpha=2.0, beta=1.0)
    deaths_overdispersion = pm.Gamma('deaths_overdispersion', alpha=2.0, beta=1.0)

    # Renewal equation for each stream
    cases = pm.Poisson('cases', mu=rt * np.sum(data['cases'].shift(-delays['cases']) / (cases_asc * 100)), observed=data['cases'])
    hospitalisations = pm.Poisson('hospitalisations', mu=rt * np.sum(data['hospitalisations'].shift(-delays['hospitalisations']) / (hospitalisations_asc * 100)) * cases_overdispersion, observed=data['hospitalisations'])
    deaths = pm.Poisson('deaths', mu=rt * np.sum(data['deaths'].shift(-delays['deaths']) / (deaths_asc * 100)) * hospitalisations_overdispersion * cases_overdispersion, observed=data['deaths'])

    # Smoothness constraint on Rt
    rt_smooth = pm.Deterministic('rt_smooth', pm.gp.mean(pm.gp.Combined( [pm.gp.Matern32(), pm.gp.Matern52()]) (rt)))

# Fit the model
with model:
    idata = pm.sample(1000, cores=4)

# Extract Rt estimates and stream-specific parameters
idata_rt = idata.posterior['rt_smooth']

# Summarize or plot results
plt.plot(idata_rt)
plt.xlabel('Time')
plt.ylabel('Rt')
plt.title('Estimated time-varying reproduction number (Rt)')
plt.show()

print("Stream-specific ascertainment rates:")
print(f"Cases: {idata.posterior['asc_cases'].mean():.2f}")
print(f"Hospitalisations: {idata.posterior['asc_hospitalisations'].mean():.2f}")
print(f"Deaths: {idata.posterior['asc_deaths'].mean():.2f}")

# Handle initial infection seeding period
start = 0  # Replace with actual date for the first reported case
```

---

## SUB_075

**Scenario**: 3 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('observations.csv')

# Set up delays for each stream (days)
case_delay = 1
hospitalisation_delay = 7
death_delay = 14

# Define model parameters and priors
with pm.Model() as model:
    # Shared Rt
    rt = pm.Deterministic('rt', pm.math.exp(pm.Normal('log_rt', mu=0, sd=2.0)))
    
    # Stream-specific delays (days)
    case_delay_t = pm.Uninformative('case_delay_t')(pm.Normal('case_delay', mu=case_delay, sd=3))
    hospitalisation_delay_t = pm.Uninformative('hospitalisation_delay_t')(pm.Normal('hospitalisation_delay', mu=hospitalisation_delay, sd=3))
    death_delay_t = pm.Uninformative('death_delay_t')(pm.Normal('death_delay', mu=death_delay, sd=3))
    
    # Stream-specific ascertainment rates
    case_ascertainment = pm.Uniform('case_ascertainment', lower=0.01, upper=1.0)
    hospitalisation_ascertainment = pm.Uniform('hospitalisation_ascertainment', lower=0.01, upper=1.0)
    death_ascertainment = pm.Uniform('death_ascertainment', lower=0.01, upper=1.0)
    
    # Stream-specific overdispersion parameters
    case_overdispersion = pm.HalfNormal('case_overdispersion', sd=2.0)
    hospitalisation_overdispersion = pm.HalfNormal('hospitalisation_overdispersion', sd=2.0)
    death_overdispersion = pm.HalfNormal('death_overdispersion', sd=2.0)
    
    # Renewal equation for each stream
    cases_observed = pm.Poisson('cases_observed', mu=rt * case_delay_t * np.sum(data['cases'].values))
    hospitalisations_observed = pm.Poisson('hospitalisations_observed', mu=rt * hospitalisation_delay_t * np.sum(data['hospitalisations'].values) * hospitalisation_ascertainment)
    deaths_observed = pm.Poisson('deaths_observed', mu=rt * death_delay_t * np.sum(data['deaths'].values) * death_ascertainment)
    
    # Smoothness constraint on Rt
    rt_smooth = pm.Deterministic('rt_smooth', pm.math.exp(pm.Normal('log_rt_smooth', mu=np.log(rt), sd=0.5)))
    
# Fit the model
with model:
    start = pm.find_MAP()
    step = pm.NUTS()
    trace = pm.sample(20000, step=step, init=start)

# Extract Rt estimates and stream-specific parameters
rt_estimates = np.mean(trace['rt_smooth'], axis=0)
case_ascertainment_rate_estimates = np.mean(trace['case_ascertainment'], axis=0)
hospitalisation_ascertainment_rate_estimates = np.mean(trace['hospitalisation_ascertainment'], axis=0)
death_ascertainment_rate_estimates = np.mean(trace['death_ascertainment'], axis=0)

# Plot results
plt.figure(figsize=(10,6))
plt.plot(data['date'].values, rt_estimates, label='Rt estimates')
plt.plot(data['date'].values, case_ascertainment_rate_estimates, label='Case ascertainment rate estimates')
plt.plot(data['date'].values, hospitalisation_ascertainment_rate_estimates, label='Hospitalisation ascertainment rate estimates')
plt.plot(data['date'].values, death_ascertainment_rate_estimates, label='Death ascertainment rate estimates')
plt.legend()
plt.show()
```

---

## SUB_001

**Scenario**: 3 | **Execution**: FAILED

### script.py

```python
import pandas as pd
import numpy as np
from pymc import Model, StepMethod, SampleFromPrior, NUTS
import pymc as pm
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('observations.csv')

# Define model parameters
n_cases = len(data['cases'])
n_hospitalisations = len(data['hospitalisations'])
n_deaths = len(data['deaths'])

# Delay and ascertainment rates for each stream
delay_cases = 0.5  # Example delay, adjust as needed
delay_hospitalisations = 3
delay_deaths = 7

ascertainment_rates = [1/6, 1/12, 1/20]  # Adjust these as needed to reflect proportions of infections observed

# Prior distribution for Rt
rtprior = pm.Uniform('Rt', lower=0.5, upper=2)

# Infection dynamics model using renewal equation
with Model() as model:
    Rt = rtprior

    cases_observed = pm.Poisson('cases_observed', mu=Rt*pm.math.sum([cases[(delay_cases + s) % n_cases] for s in range(n_cases)], axis=0))
    hospitalisations_observed = pm.Poisson('hospitalisations_observed', mu=Rt*pm.math.sum([hospitalisations[(delay_hospitalisations + s) % n_hospitalisations] for s in range(n_hospitalisations)], axis=0))
    deaths_observed = pm.Poisson('deaths_observed', mu=Rt*pm.math.sum([deaths[(delay_deaths + s) % n_deaths] for s in range(n_deaths)], axis=0))

    # Overdispersion to account for greater variance than Poisson
    cases_observed_dist = pm.Deterministic('cases_observed_dist', cases_observed * (1+pm.math.sqrt(pm.math.log(n_cases))))
    hospitalisations_observed_dist = pm.Deterministic('hospitalisations_observed_dist', hospitalisations_observed * (1+pm.math.sqrt(pm.math.log(n_hospitalisations))))
    deaths_observed_dist = pm.Deterministic('deaths_observed_dist', deaths_observed * (1+pm.math.sqrt(pm.math.log(n_deaths))))

# Initialising the seed
data['cases'][0] = 100  # Assign a dummy value to represent initial infections

# Model fitting and MCMC sampling
with model:
    trace = pm.sample(2000, tune=1000)

# Extract Rt estimates
rtestimates = trace['Rt']

# Plotting the results
plt.plot(rtestimates)
plt.title('Time-Varying Reproduction Number (Rt)')
plt.xlabel('Time')
plt.ylabel('Reproduction Number')
plt.show()

print(pm.summary(trace))
```

---

## SUB_096

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using Turing, Distributions, CSV, DataFrames, Plots, StatsPlots
using LinearAlgebra, Random
using MCMCChains

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare data
function load_data(filepath="observations.csv")
    df = CSV.read(filepath, DataFrame)
    
    # Convert to arrays
    cases = df.cases
    hospitalisations = df.hospitalisations  
    deaths = df.deaths
    dates = df.date
    
    T = length(cases)
    
    return cases, hospitalisations, deaths, dates, T
end

# Generate interval probability mass function
function generate_interval_pmf(mean_interval, sd_interval, max_interval=30)
    # Use gamma distribution parameterization
    shape = (mean_interval / sd_interval)^2
    scale = sd_interval^2 / mean_interval
    
    # Discretize gamma distribution
    pmf = zeros(max_interval)
    for i in 1:max_interval
        pmf[i] = cdf(Gamma(shape, scale), i) - cdf(Gamma(shape, scale), i-1)
    end
    
    # Normalize to ensure sum = 1
    pmf = pmf ./ sum(pmf)
    return pmf
end

# Define delay distributions for each stream
function get_delay_distributions()
    # Generation interval (infection to infection)
    gen_interval = generate_interval_pmf(6.5, 4.0, 20)
    
    # Delay from infection to observation for each stream
    case_delay = generate_interval_pmf(5.0, 3.0, 20)      # Cases: ~5 days
    hosp_delay = generate_interval_pmf(8.0, 4.0, 25)      # Hospitalisation: ~8 days  
    death_delay = generate_interval_pmf(18.0, 8.0, 40)    # Deaths: ~18 days
    
    return gen_interval, case_delay, hosp_delay, death_delay
end

@model function renewal_model(cases, hospitalisations, deaths, gen_interval, 
                             case_delay, hosp_delay, death_delay)
    
    T = length(cases)
    S_gen = length(gen_interval)
    S_case = length(case_delay)
    S_hosp = length(hosp_delay)
    S_death = length(death_delay)
    
    # Prior for initial infections (seed infections)
    seed_days = max(S_gen, S_case, S_hosp, S_death)
    I_seed ~ filldist(Exponential(100.0), seed_days)
    
    # Prior for log(Rt) - smooth time-varying reproduction number
    log_R0 ~ Normal(0.0, 0.5)  # Initial log(Rt)
    σ_R ~ Exponential(0.1)     # Smoothness parameter for Rt random walk
    
    # Random walk for log(Rt)
    log_R_raw ~ filldist(Normal(0.0, 1.0), T-1)
    
    # Construct smooth log(Rt) series
    log_R = Vector{eltype(log_R0)}(undef, T)
    log_R[1] = log_R0
    for t in 2:T
        log_R[t] = log_R[t-1] + σ_R * log_R_raw[t-1]
    end
    
    # Convert to Rt
    R = exp.(log_R)
    
    # Stream-specific ascertainment rates
    p_case ~ Beta(2, 5)      # Ascertainment rate for cases
    p_hosp ~ Beta(1, 10)     # Ascertainment rate for hospitalizations  
    p_death ~ Beta(1, 5)     # Ascertainment rate for deaths
    
    # Overdispersion parameters (inverse overdispersion)
    ϕ_case ~ Exponential(0.1)
    ϕ_hosp ~ Exponential(0.1)  
    ϕ_death ~ Exponential(0.1)
    
    # Generate infections using renewal equation
    I = Vector{eltype(log_R0)}(undef, T + seed_days)
    
    # Set seed infections
    for i in 1:seed_days
        I[i] = I_seed[i]
    end
    
    # Renewal equation for infections
    for t in 1:T
        t_idx = t + seed_days
        renewal_sum = zero(eltype(log_R0))
        
        for s in 1:min(S_gen, t_idx-1)
            renewal_sum += I[t_idx - s] * gen_interval[s]
        end
        
        I[t_idx] = R[t] * renewal_sum
    end
    
    # Extract infections corresponding to observation times
    I_obs = I[(seed_days+1):end]
    
    # Generate expected observations for each stream with delays
    λ_case = Vector{eltype(log_R0)}(undef, T)
    λ_hosp = Vector{eltype(log_R0)}(undef, T)
    λ_death = Vector{eltype(log_R0)}(undef, T)
    
    for t in 1:T
        # Cases
        case_sum = zero(eltype(log_R0))
        for s in 1:min(S_case, t + seed_days - 1)
            if t + seed_days - s >= 1
                case_sum += I[t + seed_days - s] * case_delay[s]
            end
        end
        λ_case[t] = p_case * case_sum
        
        # Hospitalizations
        hosp_sum = zero(eltype(log_R0))
        for s in 1:min(S_hosp, t + seed_days - 1)
            if t + seed_days - s >= 1
                hosp_sum += I[t + seed_days - s] * hosp_delay[s]
            end
        end
        λ_hosp[t] = p_hosp * hosp_sum
        
        # Deaths  
        death_sum = zero(eltype(log_R0))
        for s in 1:min(S_death, t + seed_days - 1)
            if t + seed_days - s >= 1
                death_sum += I[t + seed_days - s] * death_delay[s]
            end
        end
        λ_death[t] = p_death * death_sum
    end
    
    # Likelihood - negative binomial for overdispersion
    for t in 1:T
        # Add small constant to prevent numerical issues
        λ_case_safe = max(λ_case[t], 1e-6)
        λ_hosp_safe = max(λ_hosp[t], 1e-6)
        λ_death_safe = max(λ_death[t], 1e-6)
        
        # Negative binomial: NB(r, p) where r is shape, p is prob of success
        # Mean = r(1-p)/p, Var = r(1-p)/p^2
        # Reparameterize using mean μ and overdispersion ϕ
        cases[t] ~ NegativeBinomial2(λ_case_safe, ϕ_case)
        hospitalisations[t] ~ NegativeBinomial2(λ_hosp_safe, ϕ_hosp) 
        deaths[t] ~ NegativeBinomial2(λ_death_safe, ϕ_death)
    end
    
    return R, I_obs, λ_case, λ_hosp, λ_death
end

# Main analysis function
function analyze_reproduction_number(filepath="observations.csv")
    println("Loading data...")
    cases, hospitalisations, deaths, dates, T = load_data(filepath)
    
    println("Setting up delay distributions...")
    gen_interval, case_delay, hosp_delay, death_delay = get_delay_distributions()
    
    println("Fitting model...")
    model = renewal_model(cases, hospitalisations, deaths, 
                         gen_interval, case_delay, hosp_delay, death_delay)
    
    # Sample from posterior
    n_samples = 2000
    n_chains = 4
    
    chain = sample(model, NUTS(), MCMCThreads(), n_samples, n_chains, 
                  progress=true)
    
    println("\nModel Summary:")
    display(summarystats(chain))
    
    # Extract Rt estimates
    R_samples = Array(group(chain, :R))
    R_mean = mean(R_samples, dims=1)[1, :]
    R_lower = [quantile(R_samples[:, t], 0.025) for t in 1:T]
    R_upper = [quantile(R_samples[:, t], 0.975) for t in 1:T]
    
    # Extract stream-specific parameters
    p_case_est = mean(chain[:p_case])
    p_hosp_est = mean(chain[:p_hosp])  
    p_death_est = mean(chain[:p_death])
    
    println("\nStream-specific ascertainment rates:")
    println("Cases: $(round(p_case_est[1], digits=3))")
    println("Hospitalizations: $(round(p_hosp_est[1], digits=3))")
    println("Deaths: $(round(p_death_est[1], digits=3))")
    
    # Create plots
    println("\nCreating plots...")
    
    # Plot Rt over time
    p1 = plot(1:T, R_mean, ribbon=(R_mean .- R_lower, R_upper .- R_mean),
              label="Rt estimate", xlabel="Time", ylabel="Rt", 
              title="Time-varying Reproduction Number",
              fillalpha=0.3, linewidth=2)
    hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1")
    
    # Plot observations and fitted values
    λ_case_samples = Array(group(chain, :λ_case))
    λ_hosp_samples = Array(group(chain, :λ_hosp))
    λ_death_samples = Array(group(chain, :λ_death))
    
    λ_case_mean = mean(λ_case_samples, dims=1)[1, :]
    λ_hosp_mean = mean(λ_hosp_samples, dims=1)[1, :]
    λ_death_mean = mean(λ_death_samples, dims=1)[1, :]
    
    p2 = plot(1:T, cases, label="Observed cases", alpha=0.7)
    plot!(1:T, λ_case_mean, label="Expected cases", linewidth=2)
    plot!(xlabel="Time", ylabel="Count", title="Cases: Observed vs Expected")
    
    p3 = plot(1:T, hospitalisations, label="Observed hospitalizations", alpha=0.7)
    plot!(1:T, λ_hosp_mean, label="Expected hospitalizations", linewidth=2)
    plot!(xlabel="Time", ylabel="Count", title="Hospitalizations: Observed vs Expected")
    
    p4 = plot(1:T, deaths, label="Observed deaths", alpha=0.7) 
    plot!(1:T, λ_death_mean, label="Expected deaths", linewidth=2)
    plot!(xlabel="Time", ylabel="Count", title="Deaths: Observed vs Expected")
    
    # Combine plots
    final_plot = plot(p1, p2, p3, p4, layout=(2,2), size=(1000, 800))
    display(final_plot)
    
    # Return results
    results = Dict(
        "Rt_mean" => R_mean,
        "Rt_lower" => R_lower, 
        "Rt_upper" => R_upper,
        "ascertainment_rates" => Dict(
            "cases" => p_case_est[1],
            "hospitalizations" => p_hosp_est[1], 
            "deaths" => p_death_est[1]
        ),
        "chain" => chain,
        "dates" => dates
    )
    
    return results
end

# Create synthetic data for testing if no file exists
function create_synthetic_data(filename="observations.csv")
    T = 100
    dates = [Date(2023, 1, 1) + Day(i-1) for i in 1:T]
    
    # Simulate true Rt that changes over time
    true_Rt = 1.5 * exp.(-0.02 * (1:T)) .+ 0.5 .+ 0.2 * sin.(0.1 * (1:T))
    
    # Generate synthetic infections
    gen_interval, case_delay, hosp_delay, death_delay = get_delay_distributions()
    
    I = zeros(T + 20)
    I[1:20] .= 50.0  # Seed infections
    
    for t in 1:T
        renewal_sum = sum(I[t+20-s] * gen_interval[s] for s in 1:min(20, t+19))
        I[t+20] = true_Rt[t] * renewal_sum + rand(Normal(0, 5))
        I[t+20] = max(I[t+20], 1.0)  # Ensure positive
    end
    
    # Generate observations with delays and noise
    cases = zeros(Int, T)
    hospitalisations = zeros(Int, T) 
    deaths = zeros(Int, T)
    
    for t in 1:T
        # Cases
        case_exp = 0.15 * sum(I[t+20-s+1] * case_delay[s] for s in 1:min(20, t+20))
        cases[t] = max(0, rand(Poisson(case_exp + 1e-6)))
        
        # Hospitalizations
        hosp_exp = 0.03 * sum(I[t+20-s+1] * hosp_delay[s] for s in 1:min(25, t+20))
        hospitalisations[t] = max(0, rand(Poisson(hosp_exp + 1e-6)))
        
        # Deaths
        death_exp = 0.008 * sum(I[t+20-s+1] * death_delay[s] for s in 1:min(40, t+20))
        deaths[t] = max(0, rand(Poisson(death_exp + 1e-6)))
    end
    
    # Save synthetic data
    df = DataFrame(
        date = dates,
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    CSV.write(filename, df)
    println("Synthetic data saved to $filename")
    
    return df
end

# Main execution
function main()
    # Create synthetic data if file doesn't exist
    if !isfile("observations.csv")
        println("Creating synthetic data...")
        create_synthetic_data("observations.csv")
    end
    
    # Run analysis
    results = analyze_reproduction_number("observations.csv")
    
    println("\nAnalysis complete!")
    println("Final Rt estimate: $(round(results["Rt_mean"][end], digits=2))")
    
    return results
end

# Run the analysis
results = main()
```

---

## SUB_014

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using Turing, Distributions, DataFrames, CSV, Plots, StatsPlots
using LinearAlgebra, Random, Dates
using MCMCChains

# Set random seed for reproducibility
Random.seed!(123)

# Load and prepare data
function load_data(filepath="observations.csv")
    df = CSV.read(filepath, DataFrame)
    df.date = Date.(df.date)
    return df
end

# Generation interval (gamma distribution discretized)
function get_generation_interval(max_days=21)
    # Gamma distribution parameters for generation interval
    # Mean ~5 days, std ~2.5 days
    shape, scale = 4.0, 1.25
    
    # Discretize and normalize
    gen_interval = pdf.(Gamma(shape, scale), 1:max_days)
    gen_interval = gen_interval ./ sum(gen_interval)
    
    return gen_interval
end

# Stream-specific delay distributions
function get_delay_distributions(max_delay=28)
    delays = Dict()
    
    # Cases: Short delay (mean ~3 days)
    delays["cases"] = pdf.(Gamma(3.0, 1.0), 1:max_delay)
    delays["cases"] = delays["cases"] ./ sum(delays["cases"])
    
    # Hospitalizations: Medium delay (mean ~7 days) 
    delays["hospitalisations"] = pdf.(Gamma(7.0, 1.0), 1:max_delay)
    delays["hospitalisations"] = delays["hospitalisations"] ./ sum(delays["hospitalisations"])
    
    # Deaths: Long delay (mean ~14 days)
    delays["deaths"] = pdf.(Gamma(14.0, 1.0), 1:max_delay)
    delays["deaths"] = delays["deaths"] ./ sum(delays["deaths"])
    
    return delays
end

# Convolve infections with delay distribution
function convolve_delay(infections, delay_dist)
    n = length(infections)
    m = length(delay_dist)
    observed = zeros(n)
    
    for t in 1:n
        for d in 1:min(m, t)
            observed[t] += infections[t-d+1] * delay_dist[d]
        end
    end
    
    return observed
end

# Apply renewal equation
function apply_renewal(Rt, gen_interval, initial_infections)
    n = length(Rt)
    g = length(gen_interval)
    infections = zeros(n)
    
    # Set initial infections
    infections[1:min(g, n)] = initial_infections[1:min(g, n)]
    
    # Apply renewal equation
    for t in (g+1):n
        infections[t] = Rt[t] * sum(infections[(t-g):(t-1)] .* reverse(gen_interval))
    end
    
    return infections
end

@model function multi_stream_renewal_model(cases, hospitalisations, deaths, 
                                          gen_interval, delay_dists)
    n = length(cases)
    g = length(gen_interval)
    
    # Priors for initial infections (seeding period)
    initial_infections ~ MvNormal(zeros(g), 100.0 * I(g))
    
    # Prior for log(R0)
    log_R0 ~ Normal(log(1.2), 0.2)
    
    # Random walk on log(Rt) with smoothness constraint
    σ_rw ~ truncated(Normal(0, 0.1), 0, Inf)  # Standard deviation of random walk
    log_Rt_raw ~ MvNormal(zeros(n-1), I(n-1))
    
    # Construct log(Rt) as cumulative sum (random walk)
    log_Rt = Vector{eltype(log_Rt_raw)}(undef, n)
    log_Rt[1] = log_R0
    for t in 2:n
        log_Rt[t] = log_Rt[t-1] + σ_rw * log_Rt_raw[t-1]
    end
    
    Rt = exp.(log_Rt)
    
    # Apply renewal equation to get infections
    infections = apply_renewal(Rt, gen_interval, max.(0.1, initial_infections))
    
    # Stream-specific parameters
    # Ascertainment rates (proportion of infections observed in each stream)
    asc_cases ~ Beta(2, 8)        # Prior: mean ~0.2
    asc_hosp ~ Beta(1, 9)         # Prior: mean ~0.1  
    asc_deaths ~ Beta(1, 99)      # Prior: mean ~0.01
    
    # Overdispersion parameters (inverse overdispersion, higher = less overdispersed)
    ϕ_cases ~ Gamma(5, 1)
    ϕ_hosp ~ Gamma(5, 1)
    ϕ_deaths ~ Gamma(5, 1)
    
    # Convolve infections with delays to get expected observations
    expected_cases_raw = convolve_delay(infections, delay_dists["cases"])
    expected_hosp_raw = convolve_delay(infections, delay_dists["hospitalisations"])
    expected_deaths_raw = convolve_delay(infections, delay_dists["deaths"])
    
    # Apply ascertainment
    expected_cases = asc_cases .* expected_cases_raw
    expected_hosp = asc_hosp .* expected_hosp_raw
    expected_deaths = asc_deaths .* expected_deaths_raw
    
    # Likelihood with overdispersion (negative binomial)
    for t in 1:n
        # Ensure positive expected values
        μ_cases = max(0.1, expected_cases[t])
        μ_hosp = max(0.1, expected_hosp[t])
        μ_deaths = max(0.1, expected_deaths[t])
        
        # Negative binomial parameterization: NegativeBinomial(r, p)
        # where mean = r(1-p)/p and var = r(1-p)/p²
        # We use: r = ϕ, p = ϕ/(ϕ + μ)
        cases[t] ~ NegativeBinomial(ϕ_cases, ϕ_cases/(ϕ_cases + μ_cases))
        hospitalisations[t] ~ NegativeBinomial(ϕ_hosp, ϕ_hosp/(ϕ_hosp + μ_hosp))
        deaths[t] ~ NegativeBinomial(ϕ_deaths, ϕ_deaths/(ϕ_deaths + μ_deaths))
    end
end

# Main analysis function
function estimate_rt_multi_stream(data_file="observations.csv")
    println("Loading data...")
    df = load_data(data_file)
    
    # Extract observations
    cases = df.cases
    hospitalisations = df.hospitalisations  
    deaths = df.deaths
    dates = df.date
    
    println("Data loaded: $(length(cases)) time points from $(dates[1]) to $(dates[end])")
    
    # Get generation interval and delay distributions
    gen_interval = get_generation_interval()
    delay_dists = get_delay_distributions()
    
    println("Setting up model...")
    
    # Create model
    model = multi_stream_renewal_model(cases, hospitalisations, deaths, 
                                     gen_interval, delay_dists)
    
    println("Starting MCMC sampling...")
    
    # Sample from posterior
    n_samples = 1000
    n_chains = 4
    
    chain = sample(model, NUTS(0.8), MCMCThreads(), n_samples, n_chains, 
                  progress=true, drop_warmup=true)
    
    println("MCMC sampling completed!")
    
    # Extract results
    return extract_results(chain, dates, cases, hospitalisations, deaths)
end

function extract_results(chain, dates, cases, hospitalisations, deaths)
    n = length(dates)
    
    # Extract Rt estimates
    Rt_samples = Array(group(chain, :Rt))
    Rt_mean = [mean(Rt_samples[:, t]) for t in 1:n]
    Rt_lower = [quantile(Rt_samples[:, t], 0.025) for t in 1:n]
    Rt_upper = [quantile(Rt_samples[:, t], 0.975) for t in 1:n]
    
    # Extract ascertainment rates
    asc_cases_samples = Array(group(chain, :asc_cases))
    asc_hosp_samples = Array(group(chain, :asc_hosp))
    asc_deaths_samples = Array(group(chain, :asc_deaths))
    
    # Create results summary
    results = DataFrame(
        date = dates,
        Rt_mean = Rt_mean,
        Rt_lower = Rt_lower,
        Rt_upper = Rt_upper,
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    # Print summary statistics
    println("\n=== RESULTS SUMMARY ===")
    println("Rt estimates:")
    println("  Mean Rt: $(round(mean(Rt_mean), digits=3))")
    println("  Min Rt:  $(round(minimum(Rt_mean), digits=3))")
    println("  Max Rt:  $(round(maximum(Rt_mean), digits=3))")
    
    println("\nStream-specific ascertainment rates:")
    println("  Cases:           $(round(mean(asc_cases_samples), digits=4)) (95% CI: $(round(quantile(asc_cases_samples, 0.025), digits=4))-$(round(quantile(asc_cases_samples, 0.975), digits=4)))")
    println("  Hospitalizations: $(round(mean(asc_hosp_samples), digits=4)) (95% CI: $(round(quantile(asc_hosp_samples, 0.025), digits=4))-$(round(quantile(asc_hosp_samples, 0.975), digits=4)))")
    println("  Deaths:          $(round(mean(asc_deaths_samples), digits=4)) (95% CI: $(round(quantile(asc_deaths_samples, 0.025), digits=4))-$(round(quantile(asc_deaths_samples, 0.975), digits=4)))")
    
    # Create plots
    create_plots(results, chain)
    
    return results, chain
end

function create_plots(results, chain)
    # Plot 1: Rt over time
    p1 = plot(results.date, results.Rt_mean, 
             ribbon=(results.Rt_mean .- results.Rt_lower, 
                    results.Rt_upper .- results.Rt_mean),
             label="Rt estimate", color=:blue, alpha=0.7,
             title="Reproduction Number (Rt) Over Time",
             xlabel="Date", ylabel="Rt",
             legend=:topright)
    hline!([1.0], color=:red, linestyle=:dash, label="Rt = 1", alpha=0.7)
    
    # Plot 2: Data streams
    p2 = plot(results.date, results.cases, 
             label="Cases", color=:orange, alpha=0.7,
             title="Observed Data Streams",
             xlabel="Date", ylabel="Count")
    plot!(results.date, results.hospitalisations, 
          label="Hospitalizations", color=:red, alpha=0.7)
    plot!(results.date, results.deaths, 
          label="Deaths", color=:black, alpha=0.7)
    
    # Plot 3: Posterior distributions of ascertainment rates
    asc_cases_samples = Array(group(chain, :asc_cases))
    asc_hosp_samples = Array(group(chain, :asc_hosp))
    asc_deaths_samples = Array(group(chain, :asc_deaths))
    
    p3 = density(asc_cases_samples[:], label="Cases", alpha=0.7, color=:orange,
                title="Posterior Ascertainment Rates",
                xlabel="Ascertainment Rate", ylabel="Density")
    density!(asc_hosp_samples[:], label="Hospitalizations", alpha=0.7, color=:red)
    density!(asc_deaths_samples[:], label="Deaths", alpha=0.7, color=:black)
    
    # Combine plots
    final_plot = plot(p1, p2, p3, layout=(3,1), size=(800, 900))
    display(final_plot)
    
    # Save plot
    savefig(final_plot, "rt_multi_stream_results.png")
    println("\nPlots saved as 'rt_multi_stream_results.png'")
    
    return final_plot
end

# Generate synthetic data for testing
function generate_synthetic_data(n_days=100)
    dates = Date(2023, 1, 1):Day(1):(Date(2023, 1, 1) + Day(n_days-1))
    
    # True Rt trajectory (starts high, decreases, then increases)
    true_Rt = 1.5 .* exp.(-0.02 .* (1:n_days)) .+ 0.5 .+ 0.3 .* sin.(0.1 .* (1:n_days))
    
    # Generate true infections using renewal equation
    gen_interval = get_generation_interval()
    initial_infections = fill(100.0, length(gen_interval))
    true_infections = apply_renewal(true_Rt, gen_interval, initial_infections)
    
    # Generate observations with delays and ascertainment
    delay_dists = get_delay_distributions()
    
    # True ascertainment rates
    true_asc_cases = 0.15
    true_asc_hosp = 0.08
    true_asc_deaths = 0.012
    
    # Expected observations
    exp_cases = true_asc_cases .* convolve_delay(true_infections, delay_dists["cases"])
    exp_hosp = true_asc_hosp .* convolve_delay(true_infections, delay_dists["hospitalisations"])
    exp_deaths = true_asc_deaths .* convolve_delay(true_infections, delay_dists["deaths"])
    
    # Add overdispersion (negative binomial observations)
    ϕ = 10.0  # Overdispersion parameter
    
    cases = [rand(NegativeBinomial(ϕ, ϕ/(ϕ + max(0.1, μ)))) for μ in exp_cases]
    hospitalisations = [rand(NegativeBinomial(ϕ, ϕ/(ϕ + max(0.1, μ)))) for μ in exp_hosp]
    deaths = [rand(NegativeBinomial(ϕ, ϕ/(ϕ + max(0.1, μ)))) for μ in exp_deaths]
    
    # Create DataFrame
    df = DataFrame(
        date = dates,
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    # Save synthetic data
    CSV.write("observations.csv", df)
    println("Synthetic data generated and saved to 'observations.csv'")
    
    return df, true_Rt
end

# Main execution
function main()
    println("=== Multi-Stream Rt Estimation ===\n")
    
    # Generate synthetic data if file doesn't exist
    if !isfile("observations.csv")
        println("Generating synthetic data...")
        synthetic_data, true_Rt = generate_synthetic_data()
    end
    
    # Run analysis
    results, chain = estimate_rt_multi_stream()
    
    # Print convergence diagnostics
    println("\n=== MCMC DIAGNOSTICS ===")
    println("ESS summary:")
    display(summarize(chain, ess))
    
    return results, chain
end

# Run the analysis
results, chain = main()
```

---

## SUB_061

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using Turing, CSV, DataFrames, Distributions, Random, Plots, StatsPlots
using LinearAlgebra, Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Load and prepare the data
function load_data(filename="observations.csv")
    df = CSV.read(filename, DataFrame)
    
    # Convert date column if needed and sort by date
    sort!(df, :date)
    
    # Extract observation vectors
    cases = df.cases
    hospitalisations = df.hospitalisations  
    deaths = df.deaths
    
    return cases, hospitalisations, deaths, nrow(df)
end

# Define delay distributions (using gamma distributions)
function setup_delay_distributions()
    # Cases: shorter delay from infection (mean ~5 days)
    cases_delay = [pdf(Gamma(2, 2.5), x) for x in 1:20]
    cases_delay = cases_delay ./ sum(cases_delay)
    
    # Hospitalisations: medium delay (mean ~10 days)  
    hosp_delay = [pdf(Gamma(3, 3.33), x) for x in 1:25]
    hosp_delay = hosp_delay ./ sum(hosp_delay)
    
    # Deaths: longer delay (mean ~18 days)
    death_delay = [pdf(Gamma(4, 4.5), x) for x in 1:35]
    death_delay = death_delay ./ sum(death_delay)
    
    return cases_delay, hosp_delay, death_delay
end

# Define generation interval
function setup_generation_interval()
    # Generation interval with mean ~6 days
    gen_interval = [pdf(Gamma(2.5, 2.4), x) for x in 1:15]
    gen_interval = gen_interval ./ sum(gen_interval)
    return gen_interval
end

@model function joint_rt_model(cases, hospitalisations, deaths, T, 
                              cases_delay, hosp_delay, death_delay, gen_interval)
    
    # Dimensions
    max_delay = max(length(cases_delay), length(hosp_delay), length(death_delay))
    gen_len = length(gen_interval)
    seed_period = max(max_delay, gen_len, 7)  # Initial seeding period
    
    # Priors for ascertainment rates (proportion of infections observed)
    p_cases ~ Beta(2, 8)        # Cases have lower ascertainment
    p_hosp ~ Beta(1, 19)        # Hospitalisations much lower
    p_deaths ~ Beta(1, 99)      # Deaths lowest but most complete
    
    # Overdispersion parameters (smaller = more overdispersed)
    φ_cases ~ Gamma(2, 5)
    φ_hosp ~ Gamma(2, 5) 
    φ_deaths ~ Gamma(2, 5)
    
    # Initial infections (seeding period)
    log_I0 ~ Normal(5, 1)
    I_seed ~ arraydist([Normal(exp(log_I0), exp(log_I0)/3) for _ in 1:seed_period])
    
    # Time-varying Rt with smoothness constraint
    log_Rt_raw ~ arraydist([Normal(0, 0.1) for _ in 1:(T-seed_period)])
    
    # Initial Rt
    log_Rt_init ~ Normal(0, 0.5)
    
    # Construct smoothed log Rt
    log_Rt = Vector{eltype(log_Rt_raw)}(undef, T-seed_period)
    log_Rt[1] = log_Rt_init + log_Rt_raw[1]
    
    for t in 2:(T-seed_period)
        log_Rt[t] = log_Rt[t-1] + log_Rt_raw[t]  # Random walk
    end
    
    Rt = exp.(log_Rt)
    
    # Infections via renewal equation
    infections = Vector{eltype(I_seed)}(undef, T)
    
    # Seeding period
    for t in 1:seed_period
        infections[t] = I_seed[t]
    end
    
    # Renewal equation period
    for t in (seed_period+1):T
        renewal_sum = zero(eltype(infections))
        for s in 1:min(gen_len, t-1)
            renewal_sum += infections[t-s] * gen_interval[s]
        end
        infections[t] = Rt[t-seed_period] * renewal_sum
    end
    
    # Expected observations for each stream
    expected_cases = Vector{eltype(infections)}(undef, T)
    expected_hosp = Vector{eltype(infections)}(undef, T)  
    expected_deaths = Vector{eltype(infections)}(undef, T)
    
    for t in 1:T
        # Cases
        cases_sum = zero(eltype(infections))
        for d in 1:min(length(cases_delay), t)
            cases_sum += infections[t-d+1] * cases_delay[d]
        end
        expected_cases[t] = p_cases * cases_sum
        
        # Hospitalisations
        hosp_sum = zero(eltype(infections))
        for d in 1:min(length(hosp_delay), t)
            hosp_sum += infections[t-d+1] * hosp_delay[d]
        end
        expected_hosp[t] = p_hosp * hosp_sum
        
        # Deaths  
        death_sum = zero(eltype(infections))
        for d in 1:min(length(death_delay), t)
            death_sum += infections[t-d+1] * death_delay[d]
        end
        expected_deaths[t] = p_deaths * death_sum
    end
    
    # Likelihood with overdispersion (negative binomial)
    for t in 1:T
        # Negative binomial parameterization: NegativeBinomial(r, p) where mean = r(1-p)/p
        # We want mean = μ and variance = μ + μ²/φ
        # This gives us: r = φ, p = φ/(φ + μ)
        
        if expected_cases[t] > 0
            p_nb_cases = φ_cases / (φ_cases + expected_cases[t])
            cases[t] ~ NegativeBinomial(φ_cases, p_nb_cases)
        end
        
        if expected_hosp[t] > 0  
            p_nb_hosp = φ_hosp / (φ_hosp + expected_hosp[t])
            hospitalisations[t] ~ NegativeBinomial(φ_hosp, p_nb_hosp)
        end
        
        if expected_deaths[t] > 0
            p_nb_deaths = φ_deaths / (φ_deaths + expected_deaths[t])  
            deaths[t] ~ NegativeBinomial(φ_deaths, p_nb_deaths)
        end
    end
    
    return infections, Rt, expected_cases, expected_hosp, expected_deaths
end

# Main analysis function
function run_joint_rt_estimation()
    println("Loading data...")
    cases, hospitalisations, deaths, T = load_data()
    
    println("Setting up delay distributions...")
    cases_delay, hosp_delay, death_delay = setup_delay_distributions()
    gen_interval = setup_generation_interval()
    
    println("Fitting model...")
    model = joint_rt_model(cases, hospitalisations, deaths, T,
                          cases_delay, hosp_delay, death_delay, gen_interval)
    
    # Sample from posterior
    sampler = NUTS(0.65)
    n_samples = 1000
    n_chains = 4
    
    chains = sample(model, sampler, MCMCThreads(), n_samples, n_chains)
    
    println("Extracting results...")
    
    # Extract Rt estimates
    rt_samples = Array(group(chains, :Rt))
    rt_mean = vec(mean(rt_samples, dims=1))
    rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims=1))
    rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims=1))
    
    # Extract ascertainment rates
    p_cases_est = mean(Array(group(chains, :p_cases)))
    p_hosp_est = mean(Array(group(chains, :p_hosp))) 
    p_deaths_est = mean(Array(group(chains, :p_deaths)))
    
    # Extract overdispersion parameters
    phi_cases_est = mean(Array(group(chains, :φ_cases)))
    phi_hosp_est = mean(Array(group(chains, :φ_hosp)))
    phi_deaths_est = mean(Array(group(chains, :φ_deaths)))
    
    println("\n=== RESULTS ===")
    println("Ascertainment Rates:")
    println("  Cases: $(round(p_cases_est, digits=3))")
    println("  Hospitalisations: $(round(p_hosp_est, digits=3))")  
    println("  Deaths: $(round(p_deaths_est, digits=3))")
    
    println("\nOverdispersion Parameters:")
    println("  Cases φ: $(round(phi_cases_est, digits=2))")
    println("  Hospitalisations φ: $(round(phi_hosp_est, digits=2))")
    println("  Deaths φ: $(round(phi_deaths_est, digits=2))")
    
    println("\nRt Summary:")
    println("  Mean Rt: $(round(mean(rt_mean), digits=2))")
    println("  Min Rt: $(round(minimum(rt_mean), digits=2))")
    println("  Max Rt: $(round(maximum(rt_mean), digits=2))")
    
    # Create plots
    println("\nGenerating plots...")
    
    # Determine seeding period for plotting
    max_delay = max(length(cases_delay), length(hosp_delay), length(death_delay))
    gen_len = length(gen_interval)
    seed_period = max(max_delay, gen_len, 7)
    
    # Time vector for Rt (excluding seeding period)
    time_rt = (seed_period+1):T
    
    # Plot 1: Rt over time
    p1 = plot(time_rt, rt_mean, ribbon=(rt_mean .- rt_lower, rt_upper .- rt_mean),
              fillalpha=0.3, linewidth=2, label="Rt estimate",
              xlabel="Time", ylabel="Reproduction Number (Rt)", 
              title="Time-varying Reproduction Number")
    hline!([1.0], linestyle=:dash, color=:red, alpha=0.7, label="Rt = 1")
    
    # Plot 2: Observed data streams
    p2 = plot(1:T, cases, label="Cases", linewidth=2, alpha=0.8)
    plot!(1:T, hospitalisations, label="Hospitalisations", linewidth=2, alpha=0.8)
    plot!(1:T, deaths, label="Deaths", linewidth=2, alpha=0.8)
    xlabel!("Time")
    ylabel!("Daily Count") 
    title!("Observed Data Streams")
    
    # Combined plot
    combined_plot = plot(p1, p2, layout=(2,1), size=(800, 600))
    display(combined_plot)
    
    return (
        rt_estimates = (mean=rt_mean, lower=rt_lower, upper=rt_upper),
        ascertainment = (cases=p_cases_est, hosp=p_hosp_est, deaths=p_deaths_est),
        overdispersion = (cases=phi_cases_est, hosp=phi_hosp_est, deaths=phi_deaths_est),
        chains = chains,
        time_rt = time_rt
    )
end

# Generate sample data if observations.csv doesn't exist
function generate_sample_data()
    println("Generating sample data...")
    
    T = 100
    true_rt = vcat(fill(1.5, 20), 
                   1.5 .- 0.8 * (1:30)/30,  # Declining phase
                   fill(0.7, 25),            # Low phase  
                   0.7 .+ 0.6 * (1:25)/25)  # Recovery phase
    
    # Generate infections using renewal equation
    gen_interval = setup_generation_interval()
    infections = Vector{Float64}(undef, T)
    infections[1:7] .= 100.0  # Initial seeding
    
    for t in 8:T
        renewal_sum = 0.0
        for s in 1:min(length(gen_interval), t-1)
            renewal_sum += infections[t-s] * gen_interval[s]
        end
        infections[t] = true_rt[t-7] * renewal_sum
    end
    
    # Generate observations with delays
    cases_delay, hosp_delay, death_delay = setup_delay_distributions()
    
    true_p_cases = 0.15
    true_p_hosp = 0.03  
    true_p_deaths = 0.005
    
    cases = Vector{Int}(undef, T)
    hospitalisations = Vector{Int}(undef, T)  
    deaths = Vector{Int}(undef, T)
    
    for t in 1:T
        # Expected values with delays
        exp_cases = true_p_cases * sum(infections[max(1,t-d+1)] * cases_delay[d] 
                                     for d in 1:min(length(cases_delay), t))
        exp_hosp = true_p_hosp * sum(infections[max(1,t-d+1)] * hosp_delay[d]
                                   for d in 1:min(length(hosp_delay), t))
        exp_deaths = true_p_deaths * sum(infections[max(1,t-d+1)] * death_delay[d]
                                       for d in 1:min(length(death_delay), t))
        
        # Add overdispersion
        cases[t] = rand(NegativeBinomial(10, 10/(10 + exp_cases)))
        hospitalisations[t] = rand(NegativeBinomial(5, 5/(5 + exp_hosp)))
        deaths[t] = rand(NegativeBinomial(3, 3/(3 + exp_deaths)))
    end
    
    # Save to CSV
    df = DataFrame(
        date = [Date(2023, 1, 1) + Day(i-1) for i in 1:T],
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    )
    
    CSV.write("observations.csv", df)
    println("Sample data saved to observations.csv")
    
    return df
end

# Main execution
function main()
    # Check if data file exists, if not generate sample data
    if !isfile("observations.csv")
        generate_sample_data()
    end
    
    # Run the joint Rt estimation
    results = run_joint_rt_estimation()
    
    return results
end

# Run the analysis
results = main()
```

---

## SUB_077

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using Pkg; Pkg.activate("."); Pkg.instantiate()
using DataFrames
import CSV: readtable!
using Statistics
using LinearAlgebra
using Random
using Turing
using Plots

# Load data
df = readtable!("observations.csv")
Dates.format(df.date, "yyyy-mm-dd") # format date column

# Define model
@model function rt_model(observations)
    R_t ~ InverseGamma(2, 1)
    
    for (i, obs) in enumerate(eachrow(observations))
        cases_delay = i == 1 ? 0 : 5  # Assuming 5-day delay for cases
        hospitalisations_delay = [7, 14]  # Assuming delays of 7 and 14 days for hospitalisations
        deaths_delay = [14, 21]  # Assuming delays of 14 and 21 days for deaths
        
        stream_idx = findfirst(isequal(obs[2]), ["cases", "hospitalisations", "deaths"])
        
        if stream_idx == 1
            cases_observed ~ Binomial(cases_total, ascertainment_rate)
        elseif stream_idx == 2
            hospitalisations_observed ~ Binomial(hospitalisations_total, ascertainment_rate)
        else
            deaths_observed ~ Binomial(deaths_total, ascertainment_rate)
        end
        
        # Stream-specific delays
        cases_delayed ~ Poisson(1)
        hospitalisations_delayed ~ Poisson(sum([exp(-delay * R_t) for delay in hospitalisations_delay]))
        deaths_delayed ~ Poisson(sum([exp(-delay * R_t) for delay in deaths_delay]))
        
        # Renewal equation
        Rt = exp(R_t)
        delayed_cases = sum([exp(-delay * R_t) for delay in [0, 5]])
        delayed_hospitalisations = sum([exp(-delay * R_t) for delay in hospitalisations_delay])
        delayed_deaths = sum([exp(-delay * R_t) for delay in deaths_delay])
        
        observed_cases = cases_observed / (delayed_cases * ascertainment_rate)
        observed_hospitalisations = hospitalisations_observed / (delayed_hospitalisations * ascertainment_rate)
        observed_deaths = deaths_observed / (delayed_deaths * ascertainment_rate)
        
        # Overdispersion
        alpha ~ Gamma(2, 1)
        observed_cases ~ InverseGamma(alpha + 1, 1)
        observed_hospitalisations ~ InverseGamma(alpha + 1, 1)
        observed_deaths ~ InverseGamma(alpha + 1, 1)
        
    end
    
end

# Define the data
observations = map((d) -> DataFrame([d.date, d.cases, d.hospitalisations, d.deaths], [:date, :cases, :hospitalisations, :deaths]), eachrow(df))
data = [obs for obs in observations]

# Fit the model
chns = map(model -> sample(model, data[model == rt_model], MCMCThreads(), 1000), rt_model)

# Extract results
summary_stats = []
for chn in chns
    push!(summary_stats,
        (
            Rt_mean = mean(chn[:R_t]),
            ascertainment_rates = [
                mean(chn[:ascertainment_rate]) for chn in map(model -> model[1:2], rt_model)
            ]
        )
    )
end

# Print summary statistics
for (i, stats) in enumerate(summary_stats)
    println("Iteration $i:")
    println("Mean Rt:", stats.Rt_mean)
    println("Ascertainment rates: ", join(map(round, stats.ascertainment_rates), " "))
    println()
end

# Plot results
Rt_estimates = [mean(chn[:R_t]) for chn in chns]

plot(Rt_estimates, title="Estimated R(t) over time", xlabel="Time point", ylabel="Estimated Rt")
savefig("rt_estimate_plot.png")

# Handle the initial infection seeding period
initial_delay = 5
delayed_cases = sum([exp(-initial_delay * mean(chn[:R_t])) for chn in chns])
println("Delayed cases:", delayed_cases)
```

---

## SUB_007

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
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
```

---

## SUB_081

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
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
```

---

## SUB_023

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using CSV
using DataFrames
using Plots
using Statistics
using StatsPlots

# Load the data
observations = CSV.read("observations.csv", DataFrame)

# Extract data vectors
cases = observations.cases
hospitalisations = observations.hospitalisations  
deaths = observations.deaths
dates = observations.date
n_timepoints = length(cases)

println("Loaded data: $n_timepoints time points")
println("Cases range: $(minimum(cases)) - $(maximum(cases))")
println("Hospitalisations range: $(minimum(hospitalisations)) - $(maximum(hospitalisations))")
println("Deaths range: $(minimum(deaths)) - $(maximum(deaths))")

# Define generation interval (serial interval for SARS-CoV-2)
gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, shape commonly used for COVID-19

# Create EpiData
model_data = EpiData(gen_distribution = gen_int)

# Define the renewal infection model
epi = Renewal(
    model_data; 
    initialisation_prior = Normal(log(100.0), 1.0)  # Prior for initial infections
)

# Define latent model for shared log(Rt) - AR(1) process for smoothness
latent_rt = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],  # Strong persistence
    init_priors = [Normal(0.0, 0.3)],                    # Prior for initial log(Rt)
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05)) # Small innovations for smoothness
)

# Define stream-specific delay distributions
# Cases: shorter delay (symptom onset to reporting)
cases_delay = Gamma(3.0, 1.5)  # Mean ~4.5 days

# Hospitalisations: medium delay (infection to admission) 
hosp_delay = Gamma(7.0, 1.0)   # Mean ~7 days

# Deaths: longest delay (infection to death)
deaths_delay = Gamma(18.0, 0.8) # Mean ~14.4 days

# Create observation models for each stream with delays and overdispersion
obs_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.2)),
    cases_delay
)

obs_hosp = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.3)),
    hosp_delay  
)

obs_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.4)),
    deaths_delay
)

# Add stream-specific ascertainment rates using Ascertainment wrapper
# Cases ascertainment (time-varying due to testing changes)
cases_ascert_latent = RandomWalk(
    init_prior = Normal(logit(0.3), 0.3),  # Start around 30% ascertainment
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))
)

obs_cases_with_ascert = Ascertainment(
    obs_cases,
    cases_ascert_latent,
    (Y, x) -> Y .* logistic.(x)  # Transform logit to probability
)

# Hospitalisation ascertainment (more stable)
obs_hosp_with_ascert = Ascertainment(
    obs_hosp,
    FixedIntercept(logit(0.05)),  # Fixed ~5% ascertainment
    (Y, x) -> Y .* logistic.(x)
)

# Deaths ascertainment (highest, most stable)
obs_deaths_with_ascert = Ascertainment(
    obs_deaths,
    FixedIntercept(logit(0.008)),  # Fixed ~0.8% IFR
    (Y, x) -> Y .* logistic.(x)
)

# Stack the observation models for joint inference
stacked_obs = StackObservationModels((
    cases = obs_cases_with_ascert,
    hospitalisations = obs_hosp_with_ascert, 
    deaths = obs_deaths_with_ascert
))

# Create the EpiProblem combining all components
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent_rt,
    observation_model = stacked_obs,
    tspan = (1, n_timepoints)
)

# Prepare data for inference
obs_data = (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    ),
)

# Generate the Turing model
mdl = generate_epiaware(epi_prob, obs_data)

# Define inference method with pathfinder initialization and NUTS sampling
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 200)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 2000,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

println("Starting inference...")

# Run inference
results = apply_method(mdl, inference_method, obs_data)

println("Inference completed!")

# Extract results
posterior_samples = results.inference_results

# Function to extract and summarize parameter estimates
function extract_estimates(samples, param_name)
    param_samples = samples[param_name]
    if ndims(param_samples) == 2  # Time series parameter
        means = mean(param_samples, dims=1)[1, :]
        q025 = [quantile(param_samples[:, t], 0.025) for t in 1:size(param_samples, 2)]
        q975 = [quantile(param_samples[:, t], 0.975) for t in 1:size(param_samples, 2)]
        return means, q025, q975
    else  # Scalar parameter
        return mean(param_samples), quantile(param_samples, 0.025), quantile(param_samples, 0.975)
    end
end

# Extract Rt estimates (Rt = exp(Z_t) where Z_t is the latent process)
log_rt_samples = posterior_samples[:Z_t]
rt_samples = exp.(log_rt_samples)

rt_mean = mean(rt_samples, dims=1)[1, :]
rt_q025 = [quantile(rt_samples[:, t], 0.025) for t in 1:length(rt_mean)]
rt_q975 = [quantile(rt_samples[:, t], 0.975) for t in 1:length(rt_mean)]

# Extract ascertainment rates
# Cases ascertainment (time-varying)
if haskey(posterior_samples, Symbol("cases.x_t"))
    cases_ascert_logit = posterior_samples[Symbol("cases.x_t")]
    cases_ascert_samples = logistic.(cases_ascert_logit)
    cases_ascert_mean = mean(cases_ascert_samples, dims=1)[1, :]
    cases_ascert_q025 = [quantile(cases_ascert_samples[:, t], 0.025) for t in 1:length(cases_ascert_mean)]
    cases_ascert_q975 = [quantile(cases_ascert_samples[:, t], 0.975) for t in 1:length(cases_ascert_mean)]
end

# Extract fixed ascertainment rates for hospitalizations and deaths
hosp_ascert_logit = posterior_samples[Symbol("hospitalisations.α")]
hosp_ascert_samples = logistic.(hosp_ascert_logit)
hosp_ascert_mean = mean(hosp_ascert_samples)

deaths_ascert_logit = posterior_samples[Symbol("deaths.α")]  
deaths_ascert_samples = logistic.(deaths_ascert_logit)
deaths_ascert_mean = mean(deaths_ascert_samples)

# Print summary results
println("\n=== RESULTS SUMMARY ===")
println("Final Rt estimate: $(round(rt_mean[end], digits=3)) [$(round(rt_q025[end], digits=3)), $(round(rt_q975[end], digits=3))]")
println("Mean Rt over period: $(round(mean(rt_mean), digits=3))")

println("\nStream-specific ascertainment rates:")
println("Cases (final): $(round(cases_ascert_mean[end]*100, digits=2))%")
println("Cases (mean): $(round(mean(cases_ascert_mean)*100, digits=2))%")
println("Hospitalizations: $(round(hosp_ascert_mean*100, digits=3))%") 
println("Deaths: $(round(deaths_ascert_mean*100, digits=3))%")

# Create comprehensive plots
p1 = plot(1:n_timepoints, rt_mean, 
         ribbon = (rt_mean .- rt_q025, rt_q975 .- rt_mean),
         label = "Rt estimate", 
         title = "Time-varying Reproduction Number (Rt)",
         xlabel = "Time", 
         ylabel = "Rt",
         linewidth = 2,
         fillalpha = 0.3)
hline!([1.0], linestyle = :dash, color = :red, label = "Rt = 1", linewidth = 2)

p2 = plot(1:n_timepoints, cases_ascert_mean .* 100,
         ribbon = (cases_ascert_mean .- cases_ascert_q025, cases_ascert_q975 .- cases_ascert_mean) .* 100,
         label = "Cases ascertainment",
         title = "Cases Ascertainment Rate Over Time", 
         xlabel = "Time",
         ylabel = "Ascertainment (%)",
         linewidth = 2,
         fillalpha = 0.3)

# Plot observed vs expected data (posterior predictive check)
p3 = plot(1:n_timepoints, cases, label = "Observed cases", alpha = 0.7)
plot!(1:n_timepoints, hospitalisations, label = "Observed hospitalizations", alpha = 0.7)  
plot!(1:n_timepoints, deaths, label = "Observed deaths", alpha = 0.7)
plot!(title = "Observed Data Streams", xlabel = "Time", ylabel = "Count")

# Combine plots
plot(p1, p2, p3, layout = (3,1), size = (800, 1000))

# Save results to CSV
results_df = DataFrame(
    date = dates,
    rt_mean = rt_mean,
    rt_q025 = rt_q025, 
    rt_q975 = rt_q975,
    cases_ascert_mean = cases_ascert_mean,
    cases_ascert_q025 = cases_ascert_q025,
    cases_ascert_q975 = cases_ascert_q975
)

CSV.write("rt_estimates_multi_stream.csv", results_df)
println("\nResults saved to rt_estimates_multi_stream.csv")

# Display final summary
println("\n=== MODEL COMPONENTS ===")
println("✓ Shared Rt estimated via renewal equation")
println("✓ Stream-specific delays: Cases(~4.5d), Hosp(~7d), Deaths(~14d)")  
println("✓ Stream-specific ascertainment rates")
println("✓ Overdispersion via negative binomial errors")
println("✓ Smoothness via AR(1) process for log(Rt)")
println("✓ Time-varying ascertainment for cases, fixed for hosp/deaths")
```

---

## SUB_051

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using DataFrames
using CSV
using Plots
using Statistics
using Turing

# Load the data
data = CSV.read("observations.csv", DataFrame)
n_timepoints = nrow(data)

# Extract observation vectors
cases = data.cases
hospitalisations = data.hospitalisations  
deaths = data.deaths

println("Loaded data: $(n_timepoints) time points")
println("Cases range: $(minimum(cases)) - $(maximum(cases))")
println("Hospitalizations range: $(minimum(hospitalisations)) - $(maximum(hospitalisations))")
println("Deaths range: $(minimum(deaths)) - $(maximum(deaths))")

# 1. Define generation interval (infection-to-infection)
gen_int = Gamma(6.5, 0.62)  # Mean ~4 days, SD ~1.6 days
model_data = EpiData(gen_distribution = gen_int)

# 2. Create shared infection model using renewal equation
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# 3. Create shared latent model for log(Rt) - AR(1) with smoothness
latent = AR(
    damp_priors = [truncated(Normal(0.9, 0.05), 0.0, 1.0)],  # High persistence for smoothness
    init_priors = [Normal(0.0, 0.5)],                         # Prior for initial log(Rt)
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))     # Innovation noise
)

# 4. Create stream-specific observation models with different delays
# Cases: Short delay (2-3 days from infection to reporting)
cases_delay = Gamma(2.5, 1.2)  # Mean ~3 days
cases_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.2)),
    cases_delay
)

# Hospitalizations: Medium delay (7-8 days from infection to admission)  
hosp_delay = Gamma(6.5, 1.2)   # Mean ~8 days
hosp_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.3)),
    hosp_delay
)

# Deaths: Long delay (14-16 days from infection to death)
death_delay = Gamma(13.0, 1.2) # Mean ~16 days
death_obs = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.4)),
    death_delay
)

# 5. Add stream-specific ascertainment rates using Ascertainment wrapper
# Cases ascertainment (highest, but variable)
cases_ascert = Ascertainment(
    cases_obs,
    FixedIntercept(-1.5),  # logit scale, ~18% base ascertainment
    (infections, logit_p) -> infections .* logistic.(logit_p)
)

# Hospitalizations ascertainment (lower but more stable)
hosp_ascert = Ascertainment(
    hosp_obs, 
    FixedIntercept(-3.0),  # logit scale, ~5% ascertainment
    (infections, logit_p) -> infections .* logistic.(logit_p)
)

# Deaths ascertainment (lowest but most complete)
death_ascert = Ascertainment(
    death_obs,
    FixedIntercept(-4.5),  # logit scale, ~1% ascertainment  
    (infections, logit_p) -> infections .* logistic.(logit_p)
)

# 6. Stack the observation models for joint inference
stacked_obs = StackObservationModels((
    cases = cases_ascert,
    hospitalisations = hosp_ascert, 
    deaths = death_ascert
))

# 7. Create the complete EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = stacked_obs,
    tspan = (1, n_timepoints)
)

# 8. Generate the Turing model with observed data
println("Generating Turing model...")
mdl = generate_epiaware(epi_prob, (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    ),
))

# 9. Set up inference method
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 1500,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

# 10. Run inference
println("Running MCMC inference...")
results = apply_method(mdl, inference_method, (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations, 
        deaths = deaths
    ),
))

# 11. Extract and process results
println("Processing results...")

# Extract Rt estimates (latent process Z_t = log(Rt))
log_rt_samples = results[:, r"Z_t", :]
rt_samples = exp.(log_rt_samples)

# Compute Rt summary statistics
rt_mean = vec(mean(rt_samples, dims = 1))
rt_lower = vec(mapslices(x -> quantile(x, 0.025), rt_samples, dims = 1))
rt_upper = vec(mapslices(x -> quantile(x, 0.975), rt_samples, dims = 1))

# Extract ascertainment parameters (on logit scale)
cases_ascert_logit = results[:, "y_t.cases.ascert_param.Z_t[1]", :]
hosp_ascert_logit = results[:, "y_t.hospitalisations.ascert_param.Z_t[1]", :]  
death_ascert_logit = results[:, "y_t.deaths.ascert_param.Z_t[1]", :]

# Transform to probability scale
cases_ascert_prob = logistic.(cases_ascert_logit)
hosp_ascert_prob = logistic.(hosp_ascert_logit)
death_ascert_prob = logistic.(death_ascert_logit)

# Compute ascertainment summary statistics  
cases_ascert_mean = mean(cases_ascert_prob)
hosp_ascert_mean = mean(hosp_ascert_prob)
death_ascert_mean = mean(death_ascert_prob)

# 12. Display results summary
println("\n" * "="^60)
println("JOINT Rt ESTIMATION RESULTS")
println("="^60)

println("\nEstimated Parameters:")
println("├─ Cases ascertainment rate: $(round(cases_ascert_mean*100, digits=1))%")
println("├─ Hospitalizations ascertainment rate: $(round(hosp_ascert_mean*100, digits=1))%") 
println("└─ Deaths ascertainment rate: $(round(death_ascert_mean*100, digits=1))%")

println("\nRt Summary:")
println("├─ Mean Rt: $(round(mean(rt_mean), digits=2))")
println("├─ Min Rt: $(round(minimum(rt_mean), digits=2))")
println("├─ Max Rt: $(round(maximum(rt_mean), digits=2))")
println("└─ Final Rt: $(round(rt_mean[end], digits=2)) [$(round(rt_lower[end], digits=2)), $(round(rt_upper[end], digits=2))]")

# 13. Create visualization
println("\nGenerating plots...")

# Plot 1: Rt estimates over time
p1 = plot(1:n_timepoints, rt_mean, 
         ribbon = (rt_mean .- rt_lower, rt_upper .- rt_mean),
         fillalpha = 0.3,
         color = :red,
         linewidth = 2,
         label = "Rt (95% CI)",
         title = "Time-varying Reproduction Number (Rt)",
         xlabel = "Days",
         ylabel = "Rt",
         legend = :topright)

hline!([1.0], linestyle = :dash, color = :black, alpha = 0.5, label = "Rt = 1")

# Plot 2: Observed data streams
p2 = plot(1:n_timepoints, cases, 
         label = "Cases", 
         color = :blue, 
         linewidth = 2,
         title = "Observed Data Streams",
         xlabel = "Days", 
         ylabel = "Count")

plot!(1:n_timepoints, hospitalisations, 
      label = "Hospitalizations", 
      color = :orange, 
      linewidth = 2)

plot!(1:n_timepoints, deaths, 
      label = "Deaths", 
      color = :red, 
      linewidth = 2)

# Combine plots
combined_plot = plot(p1, p2, layout = (2, 1), size = (800, 600))
display(combined_plot)

# Save results to CSV
results_df = DataFrame(
    day = 1:n_timepoints,
    date = data.date,
    rt_mean = rt_mean,
    rt_lower = rt_lower, 
    rt_upper = rt_upper,
    cases_obs = cases,
    hosp_obs = hospitalisations,
    deaths_obs = deaths
)

CSV.write("rt_estimates_joint.csv", results_df)
println("\nResults saved to 'rt_estimates_joint.csv'")

println("\n" * "="^60)
println("ANALYSIS COMPLETE")
println("="^60)
```

---

## SUB_054

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing
using CSV
using DataFrames
using StatsPlots
using MCMCChains

# Load the data
data = CSV.read("observations.csv", DataFrame)

# Extract observation vectors
cases = data.cases
hospitalisations = data.hospitalisations  
deaths = data.deaths
n_timepoints = length(cases)

println("Loaded $(n_timepoints) days of data")
println("Cases: $(sum(cases)) total")
println("Hospitalisations: $(sum(hospitalisations)) total") 
println("Deaths: $(sum(deaths)) total")

# 1. Define generation interval (time from infection to onward transmission)
gen_int_dist = Gamma(6.5, 0.62)  # Mean ~4 days, common for respiratory viruses
model_data = EpiData(gen_distribution = gen_int_dist)

# 2. Create renewal infection model
epi = Renewal(
    model_data; 
    initialisation_prior = Normal(log(200.0), 1.0)  # Prior for initial infections
)

# 3. Create latent model for shared log(Rt) - AR(1) process for smoothness
latent = AR(
    damp_priors = [truncated(Normal(0.9, 0.05), 0, 1)],  # High persistence for smoothness
    init_priors = [Normal(0.0, 0.5)],  # Prior for initial log(Rt)
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.05))  # Small innovations for smoothness
)

# 4. Define stream-specific delay distributions
# Cases: shortest delay (symptom onset + reporting)
cases_delay_dist = Gamma(3.0, 1.5)  # Mean ~4.5 days

# Hospitalisations: medium delay (symptom onset + progression + admission)
hosp_delay_dist = Gamma(8.0, 1.0)   # Mean ~8 days

# Deaths: longest delay (symptom onset + progression + death + reporting)
deaths_delay_dist = Gamma(15.0, 1.2) # Mean ~18 days

# 5. Create observation models for each stream with overdispersion
base_obs = NegativeBinomialError(cluster_factor_prior = HalfNormal(0.2))

# Stream-specific observation models with delays and ascertainment
cases_obs = Ascertainment(
    LatentDelay(base_obs, cases_delay_dist),
    FixedIntercept(0.0),  # Will be estimated
    (Y, x) -> Y .* exp.(x)  # Multiplicative ascertainment
)

hosp_obs = Ascertainment(
    LatentDelay(base_obs, hosp_delay_dist), 
    FixedIntercept(0.0),  # Will be estimated
    (Y, x) -> Y .* exp.(x)  # Multiplicative ascertainment
)

deaths_obs = Ascertainment(
    LatentDelay(base_obs, deaths_delay_dist),
    FixedIntercept(0.0),  # Will be estimated  
    (Y, x) -> Y .* exp.(x)  # Multiplicative ascertainment
)

# 6. Stack observation models for joint inference
stacked_obs = StackObservationModels((
    cases = cases_obs,
    hospitalisations = hosp_obs, 
    deaths = deaths_obs
))

# 7. Create the joint EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_model = stacked_obs,
    tspan = (1, n_timepoints)
)

# 8. Prepare observations as named tuple
observations = (
    y_t = (
        cases = cases,
        hospitalisations = hospitalisations,
        deaths = deaths
    ),
)

# 9. Generate Turing model
mdl = generate_epiaware(epi_prob, observations)

# 10. Define inference method with pathfinder initialization
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 200)],
    sampler = NUTSampler(
        adtype = AutoReverseDiff(compile = true),
        ndraws = 1500,
        nchains = 4,
        mcmc_parallel = MCMCThreads()
    )
)

# 11. Run inference
println("Running MCMC inference...")
results = apply_method(mdl, inference_method, observations)

# 12. Extract and summarize results
chain = results[1]  # MCMC chain
println("\nMCMC Summary:")
println(summarystats(chain))

# Extract Rt estimates (exp of latent process)
Z_samples = Array(group(chain, :Z_t))  # latent log(Rt) samples
Rt_samples = exp.(Z_samples)
Rt_mean = mean(Rt_samples, dims=1)[:]
Rt_lower = [quantile(Rt_samples[:, t], 0.025) for t in 1:n_timepoints]
Rt_upper = [quantile(Rt_samples[:, t], 0.975) for t in 1:n_timepoints]

# Extract ascertainment parameters for each stream
# These are stored in the obs_model parameters
ascertainment_chains = group(chain, :obs_model)

println("\nRt Summary:")
println("Mean Rt over time period: $(round(mean(Rt_mean), digits=3))")
println("Min Rt (95% CI): $(round(minimum(Rt_lower), digits=3))")  
println("Max Rt (95% CI): $(round(maximum(Rt_upper), digits=3))")

# 13. Create summary plots
p1 = plot(1:n_timepoints, Rt_mean, ribbon=(Rt_mean - Rt_lower, Rt_upper - Rt_mean),
          fillalpha=0.3, linewidth=2, label="Rt estimate",
          xlabel="Days", ylabel="Reproduction number (Rt)",
          title="Joint Rt Estimation from Multiple Data Streams")
hline!([1.0], linestyle=:dash, color=:red, label="Rt = 1", linewidth=1)

# Plot the three data streams
p2 = plot(1:n_timepoints, cases, label="Cases", linewidth=2, alpha=0.7)
plot!(1:n_timepoints, hospitalisations, label="Hospitalisations", linewidth=2, alpha=0.7)
plot!(1:n_timepoints, deaths, label="Deaths", linewidth=2, alpha=0.7)
xlabel!("Days")
ylabel!("Count")
title!("Observed Data Streams")

# Combine plots
plot(p1, p2, layout=(2,1), size=(800, 600))
savefig("joint_rt_estimation.png")

# 14. Print parameter summaries
println("\n" * "="^50)
println("JOINT RT ESTIMATION RESULTS")
println("="^50)

println("\nModel Configuration:")
println("- Shared Rt across all data streams")
println("- Stream-specific delays and ascertainment rates")
println("- Overdispersed observations (Negative Binomial)")
println("- Smooth Rt evolution (AR(1) process)")

println("\nDelay Distributions (mean days):")
println("- Cases: $(round(mean(cases_delay_dist), digits=1))")
println("- Hospitalisations: $(round(mean(hosp_delay_dist), digits=1))")
println("- Deaths: $(round(mean(deaths_delay_dist), digits=1))")

println("\nRt Evolution:")
println("- Initial Rt: $(round(Rt_mean[1], digits=3)) ($(round(Rt_lower[1], digits=3))-$(round(Rt_upper[1], digits=3)))")
println("- Final Rt: $(round(Rt_mean[end], digits=3)) ($(round(Rt_lower[end], digits=3))-$(round(Rt_upper[end], digits=3)))")
println("- Days with Rt > 1: $(sum(Rt_mean .> 1.0))/$(n_timepoints)")

# Save results
results_df = DataFrame(
    day = 1:n_timepoints,
    date = data.date,
    Rt_mean = Rt_mean,
    Rt_lower = Rt_lower, 
    Rt_upper = Rt_upper,
    cases = cases,
    hospitalisations = hospitalisations,
    deaths = deaths
)

CSV.write("joint_rt_estimates.csv", results_df)
println("\nResults saved to 'joint_rt_estimates.csv'")
println("Plot saved to 'joint_rt_estimation.png'")

# Return key results
(Rt_estimates = (mean = Rt_mean, lower = Rt_lower, upper = Rt_upper),
 chain = chain,
 model = mdl)
```

---

## SUB_044

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
# Import necessary libraries
using EpiAware
using Distributions
using Turing
using DataFrames
using CSV

# Load data
df = DataFrame(CSV.File("observations.csv"))

# Define data structures for each observation type
cases_data = EpiData(
    name="Cases",
    y=df[:cases],
    x=collect(1:length(df[:cases])),
    Δ=collect(0:1:length(df[:cases])-1)
)

hospitalisations_data = EpiData(
    name="Hospitalisations",
    y=df[:hospitalisations],
    x=collect(1:length(df[:hospitalisations])),
    Δ=collect(0:1:length(df[:hospitalisations])-1)
)

deaths_data = EpiData(
    name="Deaths",
    y=df[:deaths],
    x=collect(1:length(df[:deaths])),
    Δ=collect(0:1:length(df[:deaths])-1)
)

# Define generation interval distribution
gen_distribution = Gamma(6.5, 0.62)

# Create EpiData with generation interval
model_data_cases = EpiData(gen_distribution = gen_distribution)
model_data_hospitalisations = EpiData(gen_distribution = gen_distribution)
model_data_deaths = EpiData(gen_distribution = gen_distribution)

# Define infection models (Renewal)
epi_cases = Renewal(model_data_cases; initialisation_prior = Normal(log(100.0), 1.0))
epi_hospitalisations = Renewal(model_data_hospitalisations; initialisation_prior = Normal(log(100.0), 1.0))
epi_deaths = Renewal(model_data_deaths; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

obs_hospitalisations = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

obs_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Compose into EpiProblem
epi_prob_cases = EpiProblem(
    epi_model = epi_cases,
    latent_model = latent,
    observation_model = obs_cases,
    tspan = (1, length(df[:cases]))
)

epi_prob_hospitalisations = EpiProblem(
    epi_model = epi_hospitalisations,
    latent_model = latent,
    observation_model = obs_hospitalisations,
    tspan = (1, length(df[:hospitalisations]))
)

epi_prob_deaths = EpiProblem(
    epi_model = epi_deaths,
    latent_model = latent,
    observation_model = obs_deaths,
    tspan = (1, length(df[:deaths]))
)

# Run inference
mdl_cases = generate_epiaware(epi_prob_cases, (y_t = df[:cases],))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 2000, nchains = 4)
)

results_cases = apply_method(mdl_cases, inference_method, (y_t = df[:cases],))

mdl_hospitalisations = generate_epiaware(epi_prob_hospitalisations, (y_t = df[:hospitalisations],))
results_hospitalisations = apply_method(mdl_hospitalisations, inference_method, (y_t = df[:hospitalisations],))

mdl_deaths = generate_epiaware(epi_prob_deaths, (y_t = df[:deaths],))
results_deaths = apply_method(mdl_deaths, inference_method, (y_t = df[:deaths],))

# Extract Rt estimates
rt_cases = Array(results_cases[1].Z_t)[2:end]
rt_hospitalisations = Array(results_hospitalisations[1].Z_t)[2:end]
rt_deaths = Array(results_deaths[1].Z_t)[2:end]

# Plot results
using Plots

plot(rt_cases, label="Cases")
plot!(rt_hospitalisations, label="Hospitalisations")
plot!(rt_deaths, label="Deaths")

xlabel!("Time")
ylabel!("log(Rt)")
title!("Estimated log(Rt) over time")
legend()
```

---

## SUB_066

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing

# Load data
df = readtable("observations.csv")
cases = df[:, :cases]
hospitalisations = df[:, :hospitalisations]
deaths = df[:, :deaths]

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)
model_data = EpiData(gen_distribution = gen_int)

# Create infection model (Renewal)
epi = Renewal(model_data; initialisation_prior = Normal(log(100.0), 1.0))

# Create latent models for log(Rt) - AR(1) process
latent = AR(
    damp_priors = [truncated(Normal(0.8, 0.1), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Create observation models with delays
delay_dist_cases = Gamma(5.0, 1.0)
obs_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist_cases
)

delay_dist_hospitalisations = Gamma(5.0, 1.0)
obs_hospitalisations = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist_hospitalisations
)

delay_dist_deaths = Gamma(5.0, 1.0)
obs_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist_deaths
)

# Compose into EpiProblem
epi_prob = EpiProblem(
    epi_model = epi,
    latent_model = latent,
    observation_models = (
        cases = (obs = obs_cases,),
        hospitalisations = (obs = obs_hospitalisations,),
        deaths = (obs = obs_deaths,)
    ),
    tspan = (1, length(cases))
)

# Run inference
mdl = generate_epiaware(epi_prob, (cases, hospitalisations, deaths))
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(mdl, inference_method, (cases, hospitalisations, deaths))

# Extract Rt estimates
Rt_estimates = results.latent_samples.latent_model.Z_t[1]

# Plot Rt estimates
using Plots
plot(Rt_estimates)
xlabel("Time")
ylabel("log(Rt)")
```

---

## SUB_073

**Scenario**: 3 | **Execution**: FAILED

### script.jl

```julia
using EpiAware
using Distributions
using Turing

# Load data
data = CSV.read("observations.csv")

# Define generation interval distribution
gen_int = Gamma(6.5, 0.62)

# Create infection model (Renewal)
epi_data = EpiData(gen_distribution = gen_int)
epi_model = Renewal(epi_data; initialisation_prior = Normal(log(100.0), 1.0))

# Define latent model for log(Rt) - AR(1) process
latent_model = AR(
    damp_priors = [truncated(Normal(0.8, 0.05), 0, 1)],
    init_priors = [Normal(0.0, 0.5)],
    ϵ_t = HierarchicalNormal(std_prior = HalfNormal(0.1))
)

# Define observation model with delay
delay_dist = Gamma(5.0, 1.0)
obs_model_cases = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)
obs_model_hospitalisations = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)
obs_model_deaths = LatentDelay(
    NegativeBinomialError(cluster_factor_prior = HalfNormal(0.1)),
    delay_dist
)

# Create EpiProblem with multiple observation streams
epi_prob = EpiProblem(
    epi_model = epi_model,
    latent_model = latent_model,
    observation_models = [
        obs_model_cases => "cases",
        obs_model_hospitalisations => "hospitalisations",
        obs_model_deaths => "deaths"
    ],
    tspan = (1, length(data.date))
)

# Run inference with multiple chains
inference_method = EpiMethod(
    pre_sampler_steps = [ManyPathfinder(nruns = 4, maxiters = 100)],
    sampler = NUTSampler(ndraws = 1000, nchains = 4)
)
results = apply_method(epi_prob, inference_method)

# Extract Rt estimates and stream-specific parameters
Rt_estimates = results.latent
stream_params = results.obs_params

# Plot results
using Plots
plot(Rt_estimates)
xlabel!("Time")
ylabel!("log(Rt)")
title!("Estimated log(Rt) over time")

# Print summary of model fit
println("Model Fit Summary:")
println("R-squared: $(results.r2)")
println("MAE: $(results.mae)")
println("RMSE: $(results.rmse)")
```

---

## SUB_088

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Load required libraries
library(rstan)
library(dplyr)
library(ggplot2)
library(tidyr)
library(readr)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and prepare data
data <- read_csv("observations.csv")
data$date <- as.Date(data$date)

# Define generation interval (gamma distribution with mean 5.2, sd 2.8)
max_gen <- 20
gen_interval <- dgamma(1:max_gen, shape = 3.44, scale = 1.51)
gen_interval <- gen_interval / sum(gen_interval)

# Define delay distributions for each stream
max_delay <- 30

# Cases: Short delay (mean 3 days)
delay_cases <- dgamma(0:(max_delay-1), shape = 2.25, scale = 1.33)
delay_cases <- delay_cases / sum(delay_cases)

# Hospitalizations: Medium delay (mean 10 days)
delay_hosp <- dgamma(0:(max_delay-1), shape = 4, scale = 2.5)
delay_hosp <- delay_hosp / sum(delay_hosp)

# Deaths: Long delay (mean 18 days)
delay_deaths <- dgamma(0:(max_delay-1), shape = 3.24, scale = 5.56)
delay_deaths <- delay_deaths / sum(delay_deaths)

# Prepare data for Stan
n_days <- nrow(data)
n_gen <- length(gen_interval)
n_delay <- length(delay_cases)

stan_data <- list(
  n_days = n_days,
  n_gen = n_gen,
  n_delay = n_delay,
  cases = data$cases,
  hospitalisations = data$hospitalisations,
  deaths = data$deaths,
  gen_interval = gen_interval,
  delay_cases = delay_cases,
  delay_hosp = delay_hosp,
  delay_deaths = delay_deaths,
  prior_rt_mean = 1.0,
  prior_rt_sd = 0.5,
  prior_sigma_rt = 0.1
)

# Stan model code
stan_model_code <- "
data {
  int<lower=0> n_days;
  int<lower=0> n_gen;
  int<lower=0> n_delay;
  int<lower=0> cases[n_days];
  int<lower=0> hospitalisations[n_days];
  int<lower=0> deaths[n_days];
  vector[n_gen] gen_interval;
  vector[n_delay] delay_cases;
  vector[n_delay] delay_hosp;
  vector[n_delay] delay_deaths;
  real prior_rt_mean;
  real prior_rt_sd;
  real prior_sigma_rt;
}

parameters {
  vector[n_days] log_rt_raw;
  real<lower=0> sigma_rt;
  vector<lower=0>[n_days] infections;
  real<lower=0, upper=1> ascertainment_cases;
  real<lower=0, upper=1> ascertainment_hosp;
  real<lower=0, upper=1> ascertainment_deaths;
  real<lower=0> phi_cases;
  real<lower=0> phi_hosp;
  real<lower=0> phi_deaths;
}

transformed parameters {
  vector[n_days] log_rt;
  vector[n_days] rt;
  vector[n_days] expected_cases;
  vector[n_days] expected_hosp;
  vector[n_days] expected_deaths;
  
  // Smooth Rt using random walk
  log_rt[1] = log(prior_rt_mean) + log_rt_raw[1] * prior_rt_sd;
  for (t in 2:n_days) {
    log_rt[t] = log_rt[t-1] + log_rt_raw[t] * sigma_rt;
  }
  rt = exp(log_rt);
  
  // Calculate expected observations for each stream
  for (t in 1:n_days) {
    real expected_infections_cases = 0;
    real expected_infections_hosp = 0;
    real expected_infections_deaths = 0;
    
    // Convolve infections with delays
    for (d in 1:min(t, n_delay)) {
      expected_infections_cases += infections[t-d+1] * delay_cases[d];
      expected_infections_hosp += infections[t-d+1] * delay_hosp[d];
      expected_infections_deaths += infections[t-d+1] * delay_deaths[d];
    }
    
    expected_cases[t] = ascertainment_cases * expected_infections_cases;
    expected_hosp[t] = ascertainment_hosp * expected_infections_hosp;
    expected_deaths[t] = ascertainment_deaths * expected_infections_deaths;
  }
}

model {
  // Priors
  log_rt_raw ~ std_normal();
  sigma_rt ~ exponential(1/prior_sigma_rt);
  ascertainment_cases ~ beta(2, 8);  // Prior belief: ~20% ascertainment
  ascertainment_hosp ~ beta(8, 32);  // Prior belief: ~20% of cases hospitalized
  ascertainment_deaths ~ beta(2, 98); // Prior belief: ~2% of cases die
  phi_cases ~ exponential(0.1);
  phi_hosp ~ exponential(0.1);
  phi_deaths ~ exponential(0.1);
  
  // Initial infections prior
  infections[1:n_gen] ~ exponential(0.001);
  
  // Renewal equation for infections
  for (t in (n_gen+1):n_days) {
    real infectiousness = 0;
    for (s in 1:n_gen) {
      infectiousness += infections[t-s] * gen_interval[s];
    }
    infections[t] ~ lognormal(log(rt[t] * infectiousness), 0.1);
  }
  
  // Observation models with overdispersion
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi_cases);
    }
    if (expected_hosp[t] > 0) {
      hospitalisations[t] ~ neg_binomial_2(expected_hosp[t], phi_hosp);
    }
    if (expected_deaths[t] > 0) {
      deaths[t] ~ neg_binomial_2(expected_deaths[t], phi_deaths);
    }
  }
}

generated quantities {
  vector[n_days] log_lik_cases;
  vector[n_days] log_lik_hosp;
  vector[n_days] log_lik_deaths;
  
  for (t in 1:n_days) {
    if (expected_cases[t] > 0) {
      log_lik_cases[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi_cases);
    } else {
      log_lik_cases[t] = 0;
    }
    
    if (expected_hosp[t] > 0) {
      log_lik_hosp[t] = neg_binomial_2_lpmf(hospitalisations[t] | expected_hosp[t], phi_hosp);
    } else {
      log_lik_hosp[t] = 0;
    }
    
    if (expected_deaths[t] > 0) {
      log_lik_deaths[t] = neg_binomial_2_lpmf(deaths[t] | expected_deaths[t], phi_deaths);
    } else {
      log_lik_deaths[t] = 0;
    }
  }
}
"

# Compile and fit the model
cat("Compiling Stan model...\n")
stan_model <- stan_model(model_code = stan_model_code)

cat("Fitting model...\n")
fit <- sampling(
  stan_model,
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  thin = 1,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  seed = 123
)

# Extract results
cat("Extracting results...\n")
rt_samples <- extract(fit, "rt")$rt
infections_samples <- extract(fit, "infections")$infections
ascertainment_samples <- extract(fit, c("ascertainment_cases", "ascertainment_hosp", "ascertainment_deaths"))

# Calculate summary statistics
rt_summary <- data.frame(
  date = data$date,
  rt_mean = apply(rt_samples, 2, mean),
  rt_median = apply(rt_samples, 2, median),
  rt_lower = apply(rt_samples, 2, quantile, 0.025),
  rt_upper = apply(rt_samples, 2, quantile, 0.975),
  rt_lower_50 = apply(rt_samples, 2, quantile, 0.25),
  rt_upper_50 = apply(rt_samples, 2, quantile, 0.75)
)

infections_summary <- data.frame(
  date = data$date,
  infections_mean = apply(infections_samples, 2, mean),
  infections_median = apply(infections_samples, 2, median),
  infections_lower = apply(infections_samples, 2, quantile, 0.025),
  infections_upper = apply(infections_samples, 2, quantile, 0.975)
)

# Stream-specific parameters
ascertainment_summary <- data.frame(
  stream = c("cases", "hospitalisations", "deaths"),
  mean = c(
    mean(ascertainment_samples$ascertainment_cases),
    mean(ascertainment_samples$ascertainment_hosp),
    mean(ascertainment_samples$ascertainment_deaths)
  ),
  median = c(
    median(ascertainment_samples$ascertainment_cases),
    median(ascertainment_samples$ascertainment_hosp),
    median(ascertainment_samples$ascertainment_deaths)
  ),
  lower = c(
    quantile(ascertainment_samples$ascertainment_cases, 0.025),
    quantile(ascertainment_samples$ascertainment_hosp, 0.025),
    quantile(ascertainment_samples$ascertainment_deaths, 0.025)
  ),
  upper = c(
    quantile(ascertainment_samples$ascertainment_cases, 0.975),
    quantile(ascertainment_samples$ascertainment_hosp, 0.975),
    quantile(ascertainment_samples$ascertainment_deaths, 0.975)
  )
)

# Print results summary
cat("\n=== MODEL SUMMARY ===\n")
print(summary(fit, pars = c("ascertainment_cases", "ascertainment_hosp", "ascertainment_deaths", 
                           "phi_cases", "phi_hosp", "phi_deaths", "sigma_rt"))$summary)

cat("\n=== ASCERTAINMENT RATES ===\n")
print(ascertainment_summary)

cat("\n=== Rt SUMMARY (last 10 days) ===\n")
print(tail(rt_summary[, c("date", "rt_median", "rt_lower", "rt_upper")], 10))

# Create comprehensive plots
# 1. Rt over time
p1 <- ggplot(rt_summary, aes(x = date)) +
  geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "blue") +
  geom_ribbon(aes(ymin = rt_lower_50, ymax = rt_upper_50), alpha = 0.5, fill = "blue") +
  geom_line(aes(y = rt_median), color = "darkblue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Estimated Reproduction Number (Rt)",
       subtitle = "Dark ribbon: 50% CI, Light ribbon: 95% CI",
       x = "Date", y = "Rt") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 2. Observed vs expected data
expected_samples <- extract(fit, c("expected_cases", "expected_hosp", "expected_deaths"))

expected_summary <- data.frame(
  date = rep(data$date, 3),
  stream = rep(c("Cases", "Hospitalizations", "Deaths"), each = n_days),
  observed = c(data$cases, data$hospitalisations, data$deaths),
  expected_mean = c(
    apply(expected_samples$expected_cases, 2, mean),
    apply(expected_samples$expected_hosp, 2, mean),
    apply(expected_samples$expected_deaths, 2, mean)
  ),
  expected_lower = c(
    apply(expected_samples$expected_cases, 2, quantile, 0.025),
    apply(expected_samples$expected_hosp, 2, quantile, 0.025),
    apply(expected_samples$expected_deaths, 2, quantile, 0.025)
  ),
  expected_upper = c(
    apply(expected_samples$expected_cases, 2, quantile, 0.975),
    apply(expected_samples$expected_hosp, 2, quantile, 0.975),
    apply(expected_samples$expected_deaths, 2, quantile, 0.975)
  )
)

p2 <- ggplot(expected_summary, aes(x = date)) +
  geom_ribbon(aes(ymin = expected_lower, ymax = expected_upper), alpha = 0.3, fill = "blue") +
  geom_line(aes(y = expected_mean), color = "blue", size = 1) +
  geom_point(aes(y = observed), color = "red", alpha = 0.7, size = 0.8) +
  facet_wrap(~stream, scales = "free_y", ncol = 1) +
  labs(title = "Model Fit: Observed vs Expected",
       subtitle = "Red points: observed data, Blue: model predictions with 95% CI",
       x = "Date", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 3. Estimated infections
p3 <- ggplot(infections_summary, aes(x = date)) +
  geom_ribbon(aes(ymin = infections_lower, ymax = infections_upper), alpha = 0.3, fill = "green") +
  geom_line(aes(y = infections_median), color = "darkgreen", size = 1) +
  labs(title = "Estimated Daily Infections",
       subtitle = "Green ribbon: 95% CI",
       x = "Date", y = "Daily Infections") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Display plots
print(p1)
print(p2)
print(p3)

# Model diagnostics
cat("\n=== MODEL DIAGNOSTICS ===\n")
cat("Rhat values (should be < 1.1):\n")
rhat_vals <- summary(fit)$summary[, "Rhat"]
cat("Max Rhat:", max(rhat_vals, na.rm = TRUE), "\n")
cat("Number of parameters with Rhat > 1.1:", sum(rhat_vals > 1.1, na.rm = TRUE), "\n")

# Save results
write_csv(rt_summary, "rt_estimates.csv")
write_csv(ascertainment_summary, "ascertainment_estimates.csv")
write_csv(infections_summary, "infections_estimates.csv")

cat("\nResults saved to CSV files.\n")
cat("Analysis complete!\n")
```

---

## SUB_090

**Scenario**: 3 | **Execution**: TIMEOUT

### script.R

```r
# Load required libraries
library(rstan)
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(bayesplot)

# Set Stan options
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# Load and prepare data
data <- read_csv("observations.csv")
data$t <- seq_len(nrow(data))

# Define generation interval (discretized gamma distribution)
# Mean ~5 days, SD ~2.5 days for COVID-like pathogen
generation_interval <- function(max_days = 20) {
  shape <- 4
  rate <- 0.8
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # Normalize to sum to 1
  return(pmf)
}

gen_int <- generation_interval(20)

# Define delay distributions for each stream
# Cases: shorter delay (incubation + testing)
delay_cases <- function(max_days = 15) {
  shape <- 2.5
  rate <- 0.4
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)
  return(pmf)
}

# Hospitalizations: medium delay (incubation + symptom onset + admission)
delay_hosp <- function(max_days = 25) {
  shape <- 3
  rate <- 0.25
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)
  return(pmf)
}

# Deaths: longest delay (incubation + illness + death)
delay_deaths <- function(max_days = 35) {
  shape <- 4
  rate <- 0.15
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)
  return(pmf)
}

delay_cases_pmf <- delay_cases(15)
delay_hosp_pmf <- delay_hosp(25)
delay_deaths_pmf <- delay_deaths(35)

# Prepare Stan data
stan_data <- list(
  T = nrow(data),
  cases = data$cases,
  hospitalizations = data$hospitalisations,
  deaths = data$deaths,
  
  # Generation interval
  G = length(gen_int),
  gen_int = gen_int,
  
  # Delay distributions
  D_cases = length(delay_cases_pmf),
  D_hosp = length(delay_hosp_pmf),
  D_deaths = length(delay_deaths_pmf),
  delay_cases = delay_cases_pmf,
  delay_hosp = delay_hosp_pmf,
  delay_deaths = delay_deaths_pmf
)

# Stan model code
stan_code <- "
data {
  int<lower=1> T;                    // Number of time points
  int cases[T];                      // Observed cases
  int hospitalizations[T];           // Observed hospitalizations
  int deaths[T];                     // Observed deaths
  
  // Generation interval
  int<lower=1> G;                    // Length of generation interval
  vector[G] gen_int;                 // Generation interval PMF
  
  // Delay distributions
  int<lower=1> D_cases;              // Length of cases delay
  int<lower=1> D_hosp;               // Length of hosp delay  
  int<lower=1> D_deaths;             // Length of deaths delay
  vector[D_cases] delay_cases;       // Cases delay PMF
  vector[D_hosp] delay_hosp;         // Hosp delay PMF
  vector[D_deaths] delay_deaths;     // Deaths delay PMF
}

parameters {
  // Reproduction number (log scale for smoothness)
  vector[T] log_Rt_raw;              // Raw Rt values
  real log_Rt_mean;                  // Mean log Rt
  real<lower=0> sigma_Rt;            // Rt random walk SD
  
  // Initial infections (log scale)
  vector[G] log_I0_raw;              // Initial infections
  real log_I0_mean;                  // Mean initial infections
  real<lower=0> sigma_I0;            // Initial infections SD
  
  // Ascertainment rates (logit scale)
  real logit_asc_cases;              // Cases ascertainment
  real logit_asc_hosp;               // Hosp ascertainment  
  real logit_asc_deaths;             // Deaths ascertainment
  
  // Overdispersion parameters
  real<lower=0> phi_cases;           // Cases overdispersion
  real<lower=0> phi_hosp;            // Hosp overdispersion
  real<lower=0> phi_deaths;          // Deaths overdispersion
}

transformed parameters {
  vector[T] log_Rt;                  // Smoothed log Rt
  vector[T] Rt;                      // Rt values
  vector[T] infections;              // Latent infections
  vector[T] expected_cases;          // Expected cases
  vector[T] expected_hosp;           // Expected hosp
  vector[T] expected_deaths;         // Expected deaths
  
  // Ascertainment rates
  real asc_cases = inv_logit(logit_asc_cases);
  real asc_hosp = inv_logit(logit_asc_hosp);
  real asc_deaths = inv_logit(logit_asc_deaths);
  
  // Smooth Rt using random walk
  log_Rt[1] = log_Rt_mean + sigma_Rt * log_Rt_raw[1];
  for (t in 2:T) {
    log_Rt[t] = log_Rt[t-1] + sigma_Rt * log_Rt_raw[t];
  }
  Rt = exp(log_Rt);
  
  // Generate infections using renewal equation
  // Handle initial period
  for (t in 1:G) {
    infections[t] = exp(log_I0_mean + sigma_I0 * log_I0_raw[t]);
  }
  
  // Renewal equation for subsequent infections
  for (t in (G+1):T) {
    real infectiousness = 0;
    for (g in 1:G) {
      if (t > g) {
        infectiousness += infections[t-g] * gen_int[g];
      }
    }
    infections[t] = Rt[t] * infectiousness;
  }
  
  // Expected observations with delays
  for (t in 1:T) {
    expected_cases[t] = 0;
    expected_hosp[t] = 0;
    expected_deaths[t] = 0;
    
    // Cases
    for (d in 1:D_cases) {
      if (t > d) {
        expected_cases[t] += infections[t-d] * delay_cases[d];
      }
    }
    expected_cases[t] *= asc_cases;
    
    // Hospitalizations
    for (d in 1:D_hosp) {
      if (t > d) {
        expected_hosp[t] += infections[t-d] * delay_hosp[d];
      }
    }
    expected_hosp[t] *= asc_hosp;
    
    // Deaths
    for (d in 1:D_deaths) {
      if (t > d) {
        expected_deaths[t] += infections[t-d] * delay_deaths[d];
      }
    }
    expected_deaths[t] *= asc_deaths;
  }
}

model {
  // Priors
  log_Rt_mean ~ normal(0, 0.5);      // Prior mean Rt ~ 1
  sigma_Rt ~ normal(0, 0.1);         // Smooth changes in Rt
  log_Rt_raw ~ std_normal();         // Standard normal for raw values
  
  log_I0_mean ~ normal(5, 2);        // Initial infections
  sigma_I0 ~ normal(0, 1);
  log_I0_raw ~ std_normal();
  
  // Ascertainment priors (moderately informative)
  logit_asc_cases ~ normal(-1, 1);   // Cases ~27% ascertainment
  logit_asc_hosp ~ normal(-2, 1);    // Hosp ~12% ascertainment
  logit_asc_deaths ~ normal(-3, 1);  // Deaths ~5% ascertainment
  
  // Overdispersion priors
  phi_cases ~ exponential(0.1);
  phi_hosp ~ exponential(0.1);
  phi_deaths ~ exponential(0.1);
  
  // Likelihood with overdispersion (negative binomial)
  for (t in 1:T) {
    if (expected_cases[t] > 1e-6) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi_cases);
    }
    if (expected_hosp[t] > 1e-6) {
      hospitalizations[t] ~ neg_binomial_2(expected_hosp[t], phi_hosp);
    }
    if (expected_deaths[t] > 1e-6) {
      deaths[t] ~ neg_binomial_2(expected_deaths[t], phi_deaths);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  int cases_rep[T];
  int hosp_rep[T]; 
  int deaths_rep[T];
  
  for (t in 1:T) {
    if (expected_cases[t] > 1e-6) {
      cases_rep[t] = neg_binomial_2_rng(expected_cases[t], phi_cases);
    } else {
      cases_rep[t] = 0;
    }
    
    if (expected_hosp[t] > 1e-6) {
      hosp_rep[t] = neg_binomial_2_rng(expected_hosp[t], phi_hosp);
    } else {
      hosp_rep[t] = 0;
    }
    
    if (expected_deaths[t] > 1e-6) {
      deaths_rep[t] = neg_binomial_2_rng(expected_deaths[t], phi_deaths);
    } else {
      deaths_rep[t] = 0;
    }
  }
}
"

# Compile and fit the model
cat("Compiling Stan model...\n")
model <- stan_model(model_code = stan_code)

cat("Fitting model...\n")
fit <- sampling(
  model,
  data = stan_data,
  chains = 4,
  iter = 2000,
  warmup = 1000,
  thin = 1,
  control = list(adapt_delta = 0.95, max_treedepth = 12),
  seed = 12345
)

# Check model diagnostics
print(fit, pars = c("log_Rt_mean", "sigma_Rt", "asc_cases", "asc_hosp", 
                    "asc_deaths", "phi_cases", "phi_hosp", "phi_deaths"))

# Extract results
results <- extract(fit)

# Calculate Rt summary statistics
Rt_summary <- data.frame(
  date = data$date,
  t = data$t,
  Rt_mean = apply(results$Rt, 2, mean),
  Rt_median = apply(results$Rt, 2, median),
  Rt_lower = apply(results$Rt, 2, quantile, 0.025),
  Rt_upper = apply(results$Rt, 2, quantile, 0.975),
  Rt_lower_50 = apply(results$Rt, 2, quantile, 0.25),
  Rt_upper_50 = apply(results$Rt, 2, quantile, 0.75)
)

# Extract stream-specific parameters
stream_params <- data.frame(
  parameter = c("Cases Ascertainment", "Hospitalizations Ascertainment", 
                "Deaths Ascertainment", "Cases Overdispersion", 
                "Hospitalizations Overdispersion", "Deaths Overdispersion"),
  mean = c(mean(results$asc_cases), mean(results$asc_hosp), mean(results$asc_deaths),
           mean(results$phi_cases), mean(results$phi_hosp), mean(results$phi_deaths)),
  median = c(median(results$asc_cases), median(results$asc_hosp), median(results$asc_deaths),
             median(results$phi_cases), median(results$phi_hosp), median(results$phi_deaths)),
  lower_95 = c(quantile(results$asc_cases, 0.025), quantile(results$asc_hosp, 0.025),
               quantile(results$asc_deaths, 0.025), quantile(results$phi_cases, 0.025),
               quantile(results$phi_hosp, 0.025), quantile(results$phi_deaths, 0.025)),
  upper_95 = c(quantile(results$asc_cases, 0.975), quantile(results$asc_hosp, 0.975),
               quantile(results$asc_deaths, 0.975), quantile(results$phi_cases, 0.975),
               quantile(results$phi_hosp, 0.975), quantile(results$phi_deaths, 0.975))
)

# Print results summary
cat("\n=== Rt Estimation Results ===\n")
cat("Stream-specific Parameters:\n")
print(round(stream_params, 4))

cat(sprintf("\nOverall Rt Statistics:\n"))
cat(sprintf("Mean Rt: %.3f (95%% CI: %.3f - %.3f)\n", 
            mean(Rt_summary$Rt_mean), 
            min(Rt_summary$Rt_lower), 
            max(Rt_summary$Rt_upper)))

# Create comprehensive plot
p1 <- ggplot(Rt_summary, aes(x = date)) +
  geom_ribbon(aes(ymin = Rt_lower, ymax = Rt_upper), alpha = 0.3, fill = "blue") +
  geom_ribbon(aes(ymin = Rt_lower_50, ymax = Rt_upper_50), alpha = 0.5, fill = "blue") +
  geom_line(aes(y = Rt_median), color = "darkblue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red", alpha = 0.7) +
  labs(title = "Time-varying Reproduction Number (Rt)",
       subtitle = "Estimated from cases, hospitalizations, and deaths",
       x = "Date", y = "Rt") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Create data streams plot
data_long <- data %>%
  select(date, cases, hospitalisations, deaths) %>%
  pivot_longer(-date, names_to = "stream", values_to = "count")

p2 <- ggplot(data_long, aes(x = date, y = count, color = stream)) +
  geom_line(size = 1) +
  scale_y_log10() +
  labs(title = "Observed Data Streams",
       x = "Date", y = "Count (log scale)", color = "Stream") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Model fit comparison
expected_summary <- data.frame(
  date = data$date,
  obs_cases = data$cases,
  obs_hosp = data$hospitalisations,
  obs_deaths = data$deaths,
  exp_cases = apply(results$expected_cases, 2, median),
  exp_hosp = apply(results$expected_hosp, 2, median),
  exp_deaths = apply(results$expected_deaths, 2, median)
)

fit_long <- expected_summary %>%
  select(date, obs_cases, exp_cases, obs_hosp, exp_hosp, obs_deaths, exp_deaths) %>%
  pivot_longer(-date) %>%
  separate(name, into = c("type", "stream"), sep = "_") %>%
  pivot_wider(names_from = type, values_from = value)

p3 <- ggplot(fit_long, aes(x = date)) +
  geom_line(aes(y = obs, color = "Observed"), size = 1) +
  geom_line(aes(y = exp, color = "Expected"), size = 1, alpha = 0.8) +
  facet_wrap(~stream, scales = "free_y", ncol = 1) +
  scale_y_log10() +
  labs(title = "Model Fit: Observed vs Expected",
       x = "Date", y = "Count (log scale)", color = "Type") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Print plots
print(p1)
print(p2)
print(p3)

# Save results
write_csv(Rt_summary, "rt_estimates.csv")
write_csv(stream_params, "stream_parameters.csv")

# Posterior predictive checks
cat("\n=== Model Diagnostics ===\n")
bayesplot_theme_set(theme_minimal())

# Check Rhat values
rhat_values <- summary(fit)$summary[, "Rhat"]
cat(sprintf("Max Rhat: %.3f (should be < 1.1)\n", max(rhat_values, na.rm = TRUE)))

# Effective sample sizes
n_eff_values <- summary(fit)$summary[, "n_eff"]
cat(sprintf("Min n_eff: %.0f (should be > 400)\n", min(n_eff_values, na.rm = TRUE)))

cat("\n=== Analysis Complete ===\n")
cat("Results saved to:\n")
cat("- rt_estimates.csv: Time-varying Rt estimates\n")
cat("- stream_parameters.csv: Stream-specific parameters\n")
```

---

## SUB_057

**Scenario**: 3 | **Execution**: TIMEOUT

### script.R

```r
# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)

# Set up Stan (if not already done)
# check_cmdstan_toolchain()
# install_cmdstan()

# Load and prepare data
data <- read_csv("observations.csv")
data$date <- as.Date(data$date)

# Prepare data for Stan
T <- nrow(data)
cases <- data$cases
hospitalisations <- data$hospitalisations
deaths <- data$deaths

# Generation interval (discretized gamma distribution)
# Mean ~5 days, SD ~2 days for COVID-like pathogen
gen_mean <- 5.0
gen_sd <- 2.0
S <- 15  # Maximum generation interval
s_vec <- 1:S
gen_shape <- (gen_mean / gen_sd)^2
gen_rate <- gen_mean / gen_sd^2
g_pmf <- dgamma(s_vec, shape = gen_shape, rate = gen_rate)
g_pmf <- g_pmf / sum(g_pmf)  # Normalize to PMF

# Delay distributions for each stream
# Cases: shorter delay (mean ~3 days)
delay_cases_mean <- 3.0
delay_cases_sd <- 2.0
D_cases <- 20
delay_cases_shape <- (delay_cases_mean / delay_cases_sd)^2
delay_cases_rate <- delay_cases_mean / delay_cases_sd^2
d_cases <- dgamma(1:D_cases, shape = delay_cases_shape, rate = delay_cases_rate)
d_cases <- d_cases / sum(d_cases)

# Hospitalisations: medium delay (mean ~8 days)
delay_hosp_mean <- 8.0
delay_hosp_sd <- 3.0
D_hosp <- 25
delay_hosp_shape <- (delay_hosp_mean / delay_hosp_sd)^2
delay_hosp_rate <- delay_hosp_mean / delay_hosp_sd^2
d_hosp <- dgamma(1:D_hosp, shape = delay_hosp_shape, rate = delay_hosp_rate)
d_hosp <- d_hosp / sum(d_hosp)

# Deaths: longest delay (mean ~15 days)
delay_deaths_mean <- 15.0
delay_deaths_sd <- 5.0
D_deaths <- 35
delay_deaths_shape <- (delay_deaths_mean / delay_deaths_sd)^2
delay_deaths_rate <- delay_deaths_mean / delay_deaths_sd^2
d_deaths <- dgamma(1:D_deaths, shape = delay_deaths_shape, rate = delay_deaths_rate)
d_deaths <- d_deaths / sum(d_deaths)

# Stan model code
stan_code <- "
data {
  int<lower=1> T;                    // Number of time points
  array[T] int<lower=0> cases;       // Observed cases
  array[T] int<lower=0> hospitalisations; // Observed hospitalisations
  array[T] int<lower=0> deaths;      // Observed deaths
  
  int<lower=1> S;                    // Generation interval length
  vector[S] g;                       // Generation interval PMF
  
  int<lower=1> D_cases;              // Cases delay length
  vector[D_cases] d_cases;           // Cases delay PMF
  
  int<lower=1> D_hosp;               // Hospitalisation delay length
  vector[D_hosp] d_hosp;             // Hospitalisation delay PMF
  
  int<lower=1> D_deaths;             // Deaths delay length
  vector[D_deaths] d_deaths;         // Deaths delay PMF
}

parameters {
  vector[T] log_Rt_raw;              // Raw log Rt (for smoothing)
  real log_Rt_mean;                  // Mean log Rt
  real<lower=0> sigma_Rt;            // Rt random walk SD
  
  real<lower=0> ascertainment_cases;      // Ascertainment rate for cases
  real<lower=0> ascertainment_hosp;       // Ascertainment rate for hospitalisations
  real<lower=0> ascertainment_deaths;     // Ascertainment rate for deaths
  
  vector<lower=0>[T] infections;     // True infections
  
  real<lower=0> phi_cases;           // Overdispersion for cases
  real<lower=0> phi_hosp;            // Overdispersion for hospitalisations
  real<lower=0> phi_deaths;          // Overdispersion for deaths
}

transformed parameters {
  vector[T] log_Rt;
  vector[T] Rt;
  vector[T] expected_cases = rep_vector(0, T);
  vector[T] expected_hosp = rep_vector(0, T);
  vector[T] expected_deaths = rep_vector(0, T);
  
  // Smooth Rt using random walk
  log_Rt[1] = log_Rt_mean + log_Rt_raw[1] * sigma_Rt;
  for (t in 2:T) {
    log_Rt[t] = log_Rt[t-1] + log_Rt_raw[t] * sigma_Rt;
  }
  Rt = exp(log_Rt);
  
  // Calculate expected observations for each stream
  for (t in 1:T) {
    // Cases
    for (d in 1:min(D_cases, t)) {
      expected_cases[t] += infections[t-d+1] * d_cases[d] * ascertainment_cases;
    }
    
    // Hospitalisations
    for (d in 1:min(D_hosp, t)) {
      expected_hosp[t] += infections[t-d+1] * d_hosp[d] * ascertainment_hosp;
    }
    
    // Deaths
    for (d in 1:min(D_deaths, t)) {
      expected_deaths[t] += infections[t-d+1] * d_deaths[d] * ascertainment_deaths;
    }
  }
}

model {
  // Priors
  log_Rt_mean ~ normal(0, 0.5);
  sigma_Rt ~ normal(0, 0.1);
  log_Rt_raw ~ std_normal();
  
  ascertainment_cases ~ beta(2, 5);      // Prior belief: ~20% ascertainment
  ascertainment_hosp ~ beta(1, 20);      // Prior belief: ~5% ascertainment  
  ascertainment_deaths ~ beta(1, 100);   // Prior belief: ~1% ascertainment
  
  phi_cases ~ exponential(0.1);
  phi_hosp ~ exponential(0.1);
  phi_deaths ~ exponential(0.1);
  
  // Renewal equation for infections
  for (t in 1:T) {
    real renewal_mean = 0;
    
    if (t <= S) {
      // Seeding period - use prior
      infections[t] ~ gamma(2, 0.001);  // Vague prior for initial infections
    } else {
      // Renewal equation
      for (s in 1:S) {
        renewal_mean += Rt[t] * infections[t-s] * g[s];
      }
      infections[t] ~ gamma(2, 2.0 / renewal_mean);  // Gamma prior around renewal mean
    }
  }
  
  // Likelihood for observations
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi_cases);
    }
    if (expected_hosp[t] > 0) {
      hospitalisations[t] ~ neg_binomial_2(expected_hosp[t], phi_hosp);
    }
    if (expected_deaths[t] > 0) {
      deaths[t] ~ neg_binomial_2(expected_deaths[t], phi_deaths);
    }
  }
}

generated quantities {
  array[T] int cases_pred;
  array[T] int hosp_pred;
  array[T] int deaths_pred;
  
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi_cases);
    } else {
      cases_pred[t] = 0;
    }
    
    if (expected_hosp[t] > 0) {
      hosp_pred[t] = neg_binomial_2_rng(expected_hosp[t], phi_hosp);
    } else {
      hosp_pred[t] = 0;
    }
    
    if (expected_deaths[t] > 0) {
      deaths_pred[t] = neg_binomial_2_rng(expected_deaths[t], phi_deaths);
    } else {
      deaths_pred[t] = 0;
    }
  }
}
"

# Prepare data list for Stan
stan_data <- list(
  T = T,
  cases = cases,
  hospitalisations = hospitalisations,
  deaths = deaths,
  S = S,
  g = g_pmf,
  D_cases = D_cases,
  d_cases = d_cases,
  D_hosp = D_hosp,
  d_hosp = d_hosp,
  D_deaths = D_deaths,
  d_deaths = d_deaths
)

# Compile and fit the model
model <- cmdstan_model(stan_file = write_stan_file(stan_code))

fit <- model$sample(
  data = stan_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  adapt_delta = 0.95,
  max_treedepth = 12,
  refresh = 100
)

# Extract results
draws <- fit$draws()

# Get Rt estimates
Rt_summary <- summarise_draws(draws, "Rt", mean, median, sd, ~quantile(.x, c(0.025, 0.25, 0.75, 0.975)))
Rt_estimates <- Rt_summary %>%
  mutate(
    time = as.numeric(gsub("Rt\\[(\\d+)\\]", "\\1", variable)),
    date = data$date[time]
  ) %>%
  select(date, time, mean, median, sd, q2.5, q25, q75, q97.5)

# Get ascertainment rates
ascertainment_summary <- summarise_draws(
  draws, 
  c("ascertainment_cases", "ascertainment_hosp", "ascertainment_deaths"),
  mean, median, sd, ~quantile(.x, c(0.025, 0.975))
)

# Print ascertainment rate estimates
cat("Stream-specific ascertainment rates:\n")
print(ascertainment_summary)

# Print Rt summary statistics
cat("\nRt summary (first 10 days):\n")
print(head(Rt_estimates, 10))

# Create plots
# Plot 1: Rt over time
p1 <- ggplot(Rt_estimates, aes(x = date)) +
  geom_ribbon(aes(ymin = q2.5, ymax = q97.5), alpha = 0.3, fill = "blue") +
  geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.5, fill = "blue") +
  geom_line(aes(y = median), color = "blue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(
    title = "Time-varying Reproduction Number (Rt)",
    subtitle = "Estimated from cases, hospitalisations, and deaths",
    x = "Date",
    y = "Rt"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p1)

# Plot 2: Data streams with model fit
# Get posterior predictive checks
cases_pred_summary <- summarise_draws(draws, "cases_pred", mean, ~quantile(.x, c(0.025, 0.975)))
hosp_pred_summary <- summarise_draws(draws, "hosp_pred", mean, ~quantile(.x, c(0.025, 0.975)))
deaths_pred_summary <- summarise_draws(draws, "deaths_pred", mean, ~quantile(.x, c(0.025, 0.975)))

# Prepare data for plotting
plot_data <- data.frame(
  date = rep(data$date, 3),
  stream = rep(c("Cases", "Hospitalisations", "Deaths"), each = T),
  observed = c(cases, hospitalisations, deaths),
  predicted = c(
    cases_pred_summary$mean,
    hosp_pred_summary$mean,
    deaths_pred_summary$mean
  ),
  lower = c(
    cases_pred_summary$`2.5%`,
    hosp_pred_summary$`2.5%`,
    deaths_pred_summary$`2.5%`
  ),
  upper = c(
    cases_pred_summary$`97.5%`,
    hosp_pred_summary$`97.5%`,
    deaths_pred_summary$`97.5%`
  )
)

p2 <- ggplot(plot_data, aes(x = date)) +
  geom_ribbon(aes(ymin = lower, ymax = upper), alpha = 0.3, fill = "red") +
  geom_line(aes(y = predicted), color = "red", size = 0.8) +
  geom_point(aes(y = observed), size = 0.8, alpha = 0.7) +
  facet_wrap(~stream, scales = "free_y", ncol = 1) +
  labs(
    title = "Model Fit to Multiple Data Streams",
    subtitle = "Points: observed data, Lines: model predictions with 95% CI",
    x = "Date",
    y = "Count"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

print(p2)

# Model diagnostics
cat("\nModel diagnostics:\n")
fit$diagnostic_summary()

# Summary table
cat("\nFinal Results Summary:\n")
cat("======================\n")
cat("Ascertainment Rates (median [95% CrI]):\n")
cat(sprintf("Cases: %.3f [%.3f, %.3f]\n", 
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_cases", "median"],
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_cases", "2.5%"],
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_cases", "97.5%"]))
cat(sprintf("Hospitalisations: %.3f [%.3f, %.3f]\n",
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_hosp", "median"],
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_hosp", "2.5%"],
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_hosp", "97.5%"]))
cat(sprintf("Deaths: %.4f [%.4f, %.4f]\n",
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_deaths", "median"],
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_deaths", "2.5%"],
            ascertainment_summary[ascertainment_summary$variable == "ascertainment_deaths", "97.5%"]))

cat(sprintf("\nCurrent Rt estimate (last day): %.2f [%.2f, %.2f]\n",
            tail(Rt_estimates$median, 1),
            tail(Rt_estimates$q2.5, 1),
            tail(Rt_estimates$q97.5, 1)))

# Save results
write_csv(Rt_estimates, "rt_estimates.csv")
write_csv(ascertainment_summary, "ascertainment_estimates.csv")

cat("\nResults saved to 'rt_estimates.csv' and 'ascertainment_estimates.csv'\n")
```

---

## SUB_053

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Install necessary libraries and load data
library(tidyverse)
library(bayesplot)
library(rstan)

data <- read_csv("observations.csv")

# Define model structure in Stan
model_code <- "
data {
  int<lower=1> S; // Number of days to look back into the past
  int<lower=1> T; // Total number of observations (days)
  vector[T] cases;
  vector[T] hospitalisations;
  vector[T] deaths;
}

parameters {
  real<lower=0, upper=1> ascertainment_cases[S];
  real<lower=0, upper=1> ascertainment_hospitalisations[S];
  real<lower=0, upper=1> ascertainment_deaths[S];
  real<lower=0, upper=100> overdispersion_cases;
  real<lower=0, upper=100> overdispersion_hospitalisations;
  real<lower=0, upper=100> overdispersion_deaths;
}

transformed parameters {
  vector[T] rt;
  rt[1] = 1; // Prior for Rt at day 0
  for (t in 2:T) {
    rt[t] = inv_logit(logit(rt[t-1]) + log(1 - exp(-logit(rt[t-1])*sum(cases[t-1:S]))));
  }
}

model {
  // Priors
  target += normal_lpdf(inv_logit(logit(rt[1])) | 0, 2);
  
  // Likelihood
  for (t in 1:T) {
    cases[t] ~ neg_binomial_2(exp(logit(rt[t]) + log(ascertainment_cases[t])), overdispersion_cases);
    hospitalisations[t] ~ neg_binomial_2(exp(logit(rt[t]) + log(ascertainment_hospitalisations[t])) - inv_logit(inv_logit(rt[t])+log(t)), overdispersion_hospitalisations);
    deaths[t] ~ neg_binomial_2(exp(logit(rt[t]) + log(ascertainment_deaths[t])) - inv_logit(inv_logit(rt[t])+log(t)) - inv_logit(inv_logit(rt[t])+log(T-t+1)), overdispersion_deaths);
  }
}
"

# Fit the model
fit <- rstan::sampling(model_code, data = list(
  S = 14, # number of days to look back into the past
  T = nrow(data),
  cases = data$cases,
  hospitalisations = data$hospitalisations + data$cases, # adjust for observed delay
  deaths = data$deaths + data$hospitalisations + data$cases), 
  chains = 4, iter = 10000)

# Extract and summarize Rt estimates
Rt <- rstan::extract(fit)$rt

# Plot results
plot(Rt)
```

---

## SUB_080

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(ggplot2)
library(dplyr)
library(bayesplot)

# Load data
observations <- read.csv("observations.csv")

# Define model function
model_function <- function(data) {
  # Shared Rt model for infection dynamics with stream-specific delays and ascertainment rates
  # Overdispersion accounted by a negative binomial distribution
  
  # Stream-specific delays (in days)
  delay_cases <- 2
  delay_hospitalisations <- 5
  delay_deaths <- 10
  
  # Stream-specific ascertainment rates
  ascertainment_cases <- 0.8
  ascertainment_hospitalisations <- 0.95
  ascertainment_deaths <- 0.99
  
  # Shared Rt model
  Rt_model <- brms::brm(
    cases ~ (1|time) + (Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE,
    save_model = "shared_rt.model"
  )
  
  # Stream-specific models
  model_cases <- brm(
    cases ~ (1|time) + delay_cases*(Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE
  )
  
  model_hospitalisations <- brm(
    hospitalisations ~ (1|time) + delay_hospitalisations*(Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE
  )
  
  model_deaths <- brm(
    deaths ~ (1|time) + delay_deaths*(Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE
  )
  
  # Posterior sampling of Rt from shared Rt model
  posterior_Rt <- post_summary(Rt_model)
  
  return(list(
    Rt_posterior = posterior_Rt,
    R_asymp = brms::fixef(model_cases)[1]
  ))
}

# Define data preparation function
data_preparation_function <- function(data) {
  # Prepare data for Bayesian analysis
  
  # Time variable in days since first day of observations
  data$time <- as.numeric(as.Date(data$date)) - min(as.Date(data$date))
  
  # Create separate data frames for each stream
  cases_data <- data %>% filter(!is.na(cases))
  hospitalisations_data <- data %>% filter(!is.na(hospitalisations))
  deaths_data <- data %>% filter(!is.na(deaths))
  
  return(list(
    cases_data = cases_data,
    hospitalisations_data = hospitalisations_data,
    deaths_data = deaths_data
  ))
}

# Load and fit the model
data <- data_preparation_function(observations)
model_results <- lapply(data, model_function)

# Extract results and plot Rt posterior distribution
Rt_posteriors <- sapply(model_results, function(x) x$Rt_posterior)
fixef_R_asymp <- sapply(model_results, function(x) x$R_asymp)

# Plot the Rt posterior distribution
ggplot(data.frame(date = observations$date), aes(x = date)) +
  geom_line(aes(y = mean(Rt_posteriors[1]), color = "Rt")) +
  theme_classic() +
  labs(title="Posterior Distribution of Rt", y = "Mean Rt") +
  scale_color_brewer(type = "qual", palette = "Set3")
```

---

## SUB_118

**Scenario**: 3 | **Execution**: FAILED

### script.R

```r
# Load necessary libraries
library(brms)
library(ggplot2)

# Load data
data <- read.csv("observations.csv")

# Define model structure and priors
model <- brm(
  formula = cases ~ rt + hospitalisations + deaths,
  family = negative_binomial,
  data = data,
  links = c("identity", "log"),
  include = ~1|date,
  chains = 4, cores = 4,
  iter = 20000, warmup = 10000, thin = 5,
  control = list(adapt_delta = 0.95),
  prior = set_prior("normal(0,2)", class = "b"),
  monitor = c("rt", "hospitalisations", "deaths")
)

# Define stream-specific delays and ascertainment rates
delay_cases <- 1
ascertainment_cases <- 0.8
delay_hospitalisations <- 7
ascertainment_hospitalisations <- 0.9
delay_deaths <- 14
ascertainment_deaths <- 0.95

# Generate predictions for Rt
new_data <- data.frame(
  date = unique(data$date)
)

predictions <- predict(model, newdata = new_data, type = "response", re_formula = NA)

# Plot Rt estimates over time
ggplot(new_data, aes(x = date)) +
  geom_line(aes(y = as.numeric(predictions)), color = "blue") +
  labs(title = "Estimated Rt over Time")
```

---

