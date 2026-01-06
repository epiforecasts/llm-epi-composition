# Load necessary libraries
library(cmdstanr)
library(tidyverse)

# Load the provided data
data <- read_csv("observations.csv")

# Define the Stan model for joint estimation of Rt from multiple streams
model_code <- "
data {
  int<lower=1> S; // number of time points (days) in observations
  vector[S] cases; // daily number of reported cases
  vector[S] hospitalisations; // daily number of hospital admissions
  vector[S] deaths; // daily number of deaths
}

parameters {
  real<lower=0, upper=1> rt[R]; // time-varying reproduction number (Rt) at each time point
  vector[S] delays_cases; // delay from infection to case report for each time point
  vector[S] delays_hospitalisations; // delay from infection to hospitalisation for each time point
  vector[S] delays_deaths; // delay from infection to death for each time point
  vector<lower=0, upper=1>[S] ascertainment_cases; // proportion of infections reported as cases for each time point
  vector<lower=0, upper=1>[S] ascertainment_hospitalisations; // proportion of infections resulting in hospitalisation for each time point
  vector<lower=0, upper=1>[S] ascertainment_deaths; // proportion of infections resulting in death for each time point
}

model {
  // prior distributions for parameters
  rt ~ normal(2.5, 0.5); // shared Rt prior with mean=2.5 and sd=0.5
  delays_cases ~ normal(0, 1);
  delays_hospitalisations ~ normal(3, 1);
  delays_deaths ~ normal(7, 1);
  ascertainment_cases ~ beta(2, 2); // uniform(0,1) prior for cases ascertainment rate
  ascertainment_hospitalisations ~ beta(5, 10); // informative prior for hospitalisation ascertainment rate
  ascertainment_deaths ~ beta(20, 40); // informative prior for death ascertainment rate
  
  // likelihood of observations given the model parameters
  for (s in 1:S) {
    cases[s] ~ poisson((rt[s] * sum(generation_interval[delay_cases]) * ascertainment_cases[s]));
    hospitalisations[s] ~ poisson((rt[s] * sum(generation_interval[delay_hospitalisations]) * ascertainment_hospitalisations[s]));
    deaths[s] ~ poisson((rt[s] * sum(generation_interval[delay_deaths]) * ascertainment_deaths[s]));
  }
}

generated quantities {
  // smoothed Rt estimate using the model parameters
  vector[S] rt_smoothed;
  for (s in 1:S) {
    rt_smoothed[s] = (rt[s] + rt[max(s-1, 1)]) / 2; // simple smoothing scheme
  }
}
"

# Compile the Stan model
stan_model <- cmdstanr::cmdstan_model(model_code)

# Define the data and sampling parameters
data_list <- list(
  S = nrow(data),
  cases = data$cases,
  hospitalisations = data$hospitalisations,
  deaths = data$deaths
)

sampling_params <- list(
  iter = 2000,
  warmup = 1000,
  chains = 4,
  cores = 4
)

# Fit the model
fit <- stan_model$sampling(data = data_list, 
                           algorithm = "NUTS", 
                           pars = c("rt", "delays_cases", "delays_hospitalisations", "delays_deaths",
                                    "ascertainment_cases", "ascertainment_hospitalisations", "ascertainment_deaths"),
                           chains = sampling_params$chains,
                           cores = sampling_params$cores,
                           iter = sampling_params$iter,
                           warmup = sampling_params$warmup)

# Extract Rt estimates and stream-specific parameters
rt_smoothed <- fit$draws("rt_smoothed", inc_lowest = TRUE)
rt_estimates <- rt_smoothed[, , 1]

# Plot the results (optional)
library(ggplot2)
ggplot(data.frame(date = as.Date(data$date), rt = rt_estimates),
       aes(x = date, y = rt)) +
  geom_line() +
  labs(title = "Time-Varying Reproduction Number (Rt) Estimates",
       subtitle = paste("Smoothed Rt using", sampling_params$chains, "chains and", sampling_params$cores, "cores"),
       x = "Date",
       y = "Rt")

