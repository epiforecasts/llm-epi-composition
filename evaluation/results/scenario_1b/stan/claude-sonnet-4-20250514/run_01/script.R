
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/stan/claude-sonnet-4-20250514/run_01")
options(warn = 1)

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

