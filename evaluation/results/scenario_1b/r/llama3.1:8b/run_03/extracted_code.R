# Load necessary libraries
library(readr)
library(dplyr)
library(ggplot2)

# Load data
data <- read_csv("cases.csv")

# Define the generation interval probability mass function
gen_int_pdf <- function(s, S, tau) {
  # Assume exponential distribution for simplicity
  exp(-s / tau)
}

# Define the observation model (Poisson for simplicity)
observation_model <- function(cases_observed, Rt, g_s, s_max) {
  sum(Rt * cases_observed[cases_observed > 0] *
      gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))))
}

# Define the renewal equation model
renewal_equation_model <- function(cases_observed, Rt, g_s, s_max) {
  I_t <- rep(0, length(cases_observed))
  for (t in seq_along(cases_observed)) {
    I_t[t] <- sum(Rt * cases_observed[(t - s_max):t] *
                  gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))))
  }
  return(I_t)
}

# Define the model fit function
fit_model <- function(cases_observed) {
  # Initialize parameters (e.g., Rt and g_s)
  init_params <- list(Rt = rep(1, length(unique(as.Date(data$date)))),
                      g_s = gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))))
  
  # Fit the model using optimization (e.g., maximum likelihood)
  fit <- optim(init_params,
               function(params) {
                 -sum(observation_model(cases_observed, params$Rt, gen_int_pdf(s = s_max, S = max(s_max), tau = mean(gen_int_pdf(s = 1:10, S = 10, tau = 5))), s_max)^2)
               },
               method = "BFGS",
               control = list(reltol = 1e-4))
  
  # Extract Rt estimates
  Rt_estimates <- fit$par[, 1]
  
  return(Rt_estimates)
}

# Load necessary variables
s_max <- 7  # Maximum generation interval (in days)
data$date <- as.Date(data$date)  # Convert date to Date type

# Handle initial infection seeding period (e.g., ignore the first week)
cases_observed <- data %>%
  filter(date > min(data$date, na.rm = TRUE) + weeks(1)) %>%
  pull(cases)

# Fit and extract Rt estimates
Rt_estimates <- fit_model(cases_observed)

# Plot Rt over time
ggplot(data.frame(date = unique(data$date), Rt = Rt_estimates),
       aes(x = date, y = Rt)) +
  geom_line() +
  labs(title = "Estimated Time-Varying Reproduction Number (Rt)",
       subtitle = "(Renewal Equation Framework)",
       x = "Date",
       y = "Rt")

