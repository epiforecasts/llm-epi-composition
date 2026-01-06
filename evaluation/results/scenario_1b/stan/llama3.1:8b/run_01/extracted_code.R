# Load required libraries
library(cmdstanr)
library(readr)

# Load case count data from CSV file
cases_data <- read_csv("cases.csv")


# Create a new data frame with lagged cases
case_lags <- cases_data %>%
  mutate(lag_1 = cases,
         lag_2 = lead(cases),
         lag_3 = lead(lead(cases)),
         lag_4 = lead(lead(lead(cases))),
         lag_5 = lead(lead(lead(lead(cases)))))

# Select only the required columns
case_lags <- select(case_lags, date, cases, lag_1:lag_5)


# Define the model in Stan
model_code <- "
data {
  int<lower=1> N; // number of observations
  vector[N] cases; // observed case counts
  real<lower=0, upper=1> S;
}

parameters {
  real<lower=0, upper=Inf> R_t[1]; // time-varying reproduction number
  vector<lower=0>[5] g; // generation interval probability mass function
  real<lower=0, upper=S> theta[1]; // observation model parameter
}

model {
  // Prior distributions for parameters
  target += normal_lpdf(R_t | 2, 1);
  target += dirichlet_lpdf(g | rep_vector(1, 5));
  target += beta_lpdf(theta[1] | 2, 2);

  // Observation model
  for (t in 1:N) {
    cases[t] ~ neg_binomial_2(R_t[1] * g[case_lags[lag_(t)][lag_id]] * theta[1], alpha);
  }
}
"

# Fit the model using cmdstanr
fit <- cmdstanr::sample(model_code, data = list(N = nrow(case_lags), cases = case_lags$cases,
                                                 S = 1),
                        chains = 4, iter_warmup = 2000, iter_sampling = 2000)


# Extract Rt estimates for each time point
Rt_estimates <- fit$summary[, "mean"]


# Exclude initial seeding period (e.g., first 7 days)
Rt_estimates <- Rt_estimates[case_lags$date > "2020-03-01"]


# Plot Rt estimates over time
plot(Rt_estimates ~ case_lags$date, type = "l")

