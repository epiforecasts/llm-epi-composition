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

