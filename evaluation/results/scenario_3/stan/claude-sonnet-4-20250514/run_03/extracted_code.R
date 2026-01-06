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

