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

