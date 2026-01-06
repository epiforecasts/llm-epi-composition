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

