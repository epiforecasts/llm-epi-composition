
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/stan/claude-sonnet-4-20250514/run_03")
options(warn = 1)

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

