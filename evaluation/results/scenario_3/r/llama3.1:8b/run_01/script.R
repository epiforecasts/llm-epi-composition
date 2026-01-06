
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/r/llama3.1:8b/run_01")
options(warn = 1)

# Install necessary libraries and load data
library(tidyverse)
library(bayesplot)
library(rstan)

data <- read_csv("observations.csv")

# Define model structure in Stan
model_code <- "
data {
  int<lower=1> S; // Number of days to look back into the past
  int<lower=1> T; // Total number of observations (days)
  vector[T] cases;
  vector[T] hospitalisations;
  vector[T] deaths;
}

parameters {
  real<lower=0, upper=1> ascertainment_cases[S];
  real<lower=0, upper=1> ascertainment_hospitalisations[S];
  real<lower=0, upper=1> ascertainment_deaths[S];
  real<lower=0, upper=100> overdispersion_cases;
  real<lower=0, upper=100> overdispersion_hospitalisations;
  real<lower=0, upper=100> overdispersion_deaths;
}

transformed parameters {
  vector[T] rt;
  rt[1] = 1; // Prior for Rt at day 0
  for (t in 2:T) {
    rt[t] = inv_logit(logit(rt[t-1]) + log(1 - exp(-logit(rt[t-1])*sum(cases[t-1:S]))));
  }
}

model {
  // Priors
  target += normal_lpdf(inv_logit(logit(rt[1])) | 0, 2);
  
  // Likelihood
  for (t in 1:T) {
    cases[t] ~ neg_binomial_2(exp(logit(rt[t]) + log(ascertainment_cases[t])), overdispersion_cases);
    hospitalisations[t] ~ neg_binomial_2(exp(logit(rt[t]) + log(ascertainment_hospitalisations[t])) - inv_logit(inv_logit(rt[t])+log(t)), overdispersion_hospitalisations);
    deaths[t] ~ neg_binomial_2(exp(logit(rt[t]) + log(ascertainment_deaths[t])) - inv_logit(inv_logit(rt[t])+log(t)) - inv_logit(inv_logit(rt[t])+log(T-t+1)), overdispersion_deaths);
  }
}
"

# Fit the model
fit <- rstan::sampling(model_code, data = list(
  S = 14, # number of days to look back into the past
  T = nrow(data),
  cases = data$cases,
  hospitalisations = data$hospitalisations + data$cases, # adjust for observed delay
  deaths = data$deaths + data$hospitalisations + data$cases), 
  chains = 4, iter = 10000)

# Extract and summarize Rt estimates
Rt <- rstan::extract(fit)$rt

# Plot results
plot(Rt)

