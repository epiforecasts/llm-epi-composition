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

