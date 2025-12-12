I'll create a complete solution for estimating time-varying reproduction number (Rt) jointly from multiple data streams using a Bayesian approach with Stan.

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

This comprehensive solution provides:

## Key Features:

1. **Renewal Equation**: Implements the renewal equation with shared Rt across all streams
2. **Stream-Specific Delays**: Different delay distributions for cases, hospitalizations, and deaths
3. **Stream-Specific Ascertainment**: Separate ascertainment rates for each data stream
4. **Overdispersion**: Uses negative binomial distributions to handle extra variance
5. **Smooth Rt**: Random walk prior ensures smooth temporal variation
6. **Proper Initialization**: Handles the seeding period appropriately

## Model Components:

- **Generation Interval**: Gamma distribution (mean ~5.2 days)
- **Delay Distributions**: Stream-specific delays from infection to observation
- **Ascertainment Rates**: Beta priors reflecting realistic expectations
- **Overdispersion Parameters**: Stream-specific φ parameters
- **Smoothness**: Random walk on log(Rt) with estimated variance

## Outputs:

1. **Rt estimates** with uncertainty intervals
2. **Stream-specific ascertainment rates**
3. **Model fit diagnostics** and visualizations
4. **Estimated infections** over time
5. **Comprehensive plots** showing results and model fit

The model jointly uses all three data streams to provide a robust estimate of Rt while accounting for the different characteristics of each observation type.