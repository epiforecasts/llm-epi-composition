library(cmdstanr)
library(readr)
library(ggplot2)


# Load data
data <- read_csv("cases.csv")

# Ensure date is in a format that can be used for time-series analysis
data$date <- as.Date(data$date)

# Handle missing or invalid dates
data %<>%
  filter(!is.na(date))


# Model definition (in Stan)
model_code <- "
data {
  int<lower=0> T; // Number of time points
  vector[T] cases; // Case counts at each time point
  real<lower=0, upper=1> S; // Generation interval distribution parameter
}

parameters {
  real<lower=0, upper=10> R_t[T]; // Time-varying reproduction number
  real<lower=0, upper=S> g[3]; // Generation interval distribution parameters (e.g., mean and standard deviation)
}

model {
  // Prior on Rt (lognormal with constant mean and variance)
  target += normal_lpdf(log(R_t), 2.5, 1);

  // Prior on g (uniform within the specified range)
  for (i in 1:3) {
    target += uniform_lpdf(g[i], 0, S);
  }

  // Likelihood of case counts given Rt and generation interval distribution
  for (t in 1:T) {
    if (cases[t] > 0 && t > 1) { // Exclude initial infection seeding period
      cases[t] ~ poisson_log(lambda = log(sum(exp(log(R_t[u]) + normal_lpdf(g[3] * u, 0, g[2])))));
    }
  }

  // Priors for the generation interval distribution parameters
  target += uniform_lpdf(g[1], 0, S);
}
"

# Fit the model using cmdstanr
fit <- stan(
  file = "renewal_equation.stan",
  data = list(T = nrow(data), cases = data$cases, S = 10),
  chains = 4,
  iter = 2000,
  warmup = 1000,
  thin = 1
)


# Extract Rt estimates for each time point
Rt_estimates <- extract(fit, "R_t")[, , 1]

# Convert to long format and add date information
Rt_long <- data.frame(
  Date = data$date,
  Rt_estimate = Rt_estimates,
  model = rep("Renewal Equation", nrow(data))
)

# Plot the results (optional)
ggplot(Rt_long, aes(x = Date, y = Rt_estimate)) +
  geom_line() +
  labs(title = "Estimated Reproduction Number (Rt)", x = "Date") +
  theme_classic()

