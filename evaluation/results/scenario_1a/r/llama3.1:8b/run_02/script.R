
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/r/llama3.1:8b/run_02")
options(warn = 1)

# Install required packages if not already installed
install.packages("dplyr")
install.packages("tidyr")
install.packages("lubridate")
install.packages("bats")

# Load required libraries
library(dplyr)
library(tidyr)
library(lubridate)
library(bats)


# Load the data from cases.csv
cases_data <- read_csv("cases.csv")

# Convert date column to Date format
cases_data$date <- ymd(cases_data$date)

# Ensure that cases are in integer format
cases_data$cases <- as.integer(cases_data$cases)


# Define a new function for Rt estimation using bats
Rt_estimation <- function(x) {
  # x should be a tibble with date and cases columns
  
  # Calculate lagged values of cases (for autoregression)
  x$lag_cases <- lag(x, x$date, -1)$cases
  
  # Fit the model
  model <- bats(
    formula = cases ~ lag_cases + (1|date),
    data = x,
    family = "poisson"
  )
  
  # Extract and return Rt estimates
  Rt_estimates <- predict(model, newdata = x, type = "Rt")
  return(Rt_estimates)
}

# Apply the function to each time point
estimates <- cases_data %>%
  group_by(date) %>%
  summarise(
    Rt_estimate = Rt_estimation(.[, c("date", "cases")])
  )


# Extract Rt estimates for plotting
Rt_over_time <- estimates$Rt_estimate

# Plot Rt over time (optional)
plot(Rt_over_time,
     main = "Time-Varying Reproduction Number (Rt)",
     xlab = "Date",
     ylab = "Estimated Rt")

