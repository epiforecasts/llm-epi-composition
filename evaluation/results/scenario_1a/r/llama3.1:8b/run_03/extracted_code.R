# Install required libraries if not already installed
install.packages("dplyr")
install.packages("lubridate")
install.packages("deSolve")

# Load necessary libraries
library(dplyr)
library(lubridate)
library(deSolve)

# Load the daily case counts from a CSV file
cases <- read.csv("cases.csv", stringsAsFactors = FALSE) %>%
  mutate(date = as.Date(date))

# Set up parameters for the model (you may need to adjust these based on your data and understanding of the model)
parameters <- list(
  R0 = 2.5, # Initial reproduction number
  gamma = 1/7, # Rate at which infected individuals recover or die
  sigma = 1/3, # Rate at which susceptible individuals become infected
  N = sum(cases$cases), # Total population (estimated from case counts)
  I0 = cases$cases[1], # Initial number of infected individuals (first reported case count)
  S0 = sum(cases$cases) - cases$cases[1] # Initial number of susceptible individuals
)

# Define the model function
model <- function(t, y, parameters) {
  with(as.list(c(y, parameters)), {
    dSdt <- -beta * S * I / N
    dIdt <- beta * S * I / N - gamma * I
    return(list(c(dSdt, dIdt)))
  })
}

# Define the solve function to fit the model
solve <- function(parameters) {
  # Discretize the time points (you may need to adjust this based on your data and computational resources)
  t <- seq(0, max(cases$date), by = "day")
  
  # Initial conditions
  init <- c(S = parameters$S0, I = parameters$I0)
  
  # Solve ODEs
  output <- ode(y = init, times = t, func = model, parms = parameters)
  
  # Extract Rt estimates for each time point (using the estimated beta and gamma values)
  Rt <- (parameters$beta * output$solve[, "S"] / output$solve[, "I"]) + parameters$gamma
  
  return(Rt)
}

# Fit the model
Rt_estimates <- solve(parameters)

# Plot or summarize the results
plot(cases$date, Rt_estimates, type = "l", 
     xlab = "Date", ylab = expression("Estimated R[italic(t)]"))

summary(Rt_estimates)

