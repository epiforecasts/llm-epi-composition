
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/stan/llama3.1:8b/run_02")
options(warn = 1)

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

