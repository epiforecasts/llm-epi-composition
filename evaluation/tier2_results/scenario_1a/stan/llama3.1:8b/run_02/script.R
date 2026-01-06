
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/stan/llama3.1:8b/run_02")
options(warn = 1)

# Install and load required libraries
install.packages("cmdstanr")
library(cmdstanr)
library(dplyr)
library(readr)


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

