# Load necessary libraries and data
library(tidyverse)
library(deSolve)
library(Rtsnps)

# Load case counts from CSV file
cases <- read_csv("cases.csv")

# Add day-of-week label as a factor for easier interpretation
cases$day_of_week_factor <- factor(cases$day_of_week,
                                   levels = 1:7,
                                   labels = c("Monday", "Tuesday", "Wednesday",
                                              "Thursday", "Friday", "Saturday", "Sunday"))

# Define model functions and parameters
model_func <- function(t, y, params) {
  # Infection dynamics (renewal equation)
  dydt <- c(
    dRt = -params[1] * R_t,
    dascertainment = params[2] - params[3]
  )
  
  return(list(c(dRt, dascertainment)))
}

# Define parameters and initial conditions
initial_conditions <- data.frame(R_t = 0.5,
                                 ascertainment = 0.6)

# Time points to estimate Rt at
time_points <- seq(1, nrow(cases), by = 1)

# Fit model using deSolve
fit <- ode(model_func,
          times = time_points,
          y = initial_conditions,
          parms = c(0.2,  # decay rate of Rt
                    0.8,  # initial ascertainment (proportion of infections reported)
                    0.5))  # overdispersion

# Extract estimates of Rt and day-of-week effects
Rt_estimates <- data.frame(
  date = cases$date[fit$times],
  R_t = fit$solution[, 1]
)

day_of_week_effects <- cases %>%
  group_by(day_of_week_factor) %>%
  summarise(n = n()) %>%
  mutate(day_of_week_effect = n / sum(n))

# Extract estimates of time-varying ascertainment
ascertainment_estimates <- data.frame(
  date = cases$date[fit$times],
  ascertainment = fit$solution[, 2]
)

# Summary plot
library(ggplot2)
ggplot(Rt_estimates, aes(date, R_t)) +
  geom_line() +
  labs(title = "Time-varying Reproduction Number (Rt) Estimates",
       subtitle = "COVID-19 Case Counts")

