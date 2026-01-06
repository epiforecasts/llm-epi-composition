
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/r/llama3.1:8b/run_02")
options(warn = 1)

# Load necessary libraries
library(readr)
library(ggplot2)

# Set seed for reproducibility
set.seed(1234)

# Function to estimate Rt from daily case counts

estimate_rt <- function(case_data, beta) {
  # Read the data into a data frame
  df <- read_csv(case_data)

  # Ensure 'date' is of date class and convert it to year-month format for easier indexing
  df$date <- as.Date(df$date)
  
  # Initialize Rt as NA for each day
  rt_estimates <- rep(NA, nrow(df))
  
  # Calculate the generation interval probability mass function (PMF) values
  gen_interval_pmf <- function(s) {
    if (s == 0) { return(1 - exp(-beta)) }
    else { return(exp(-beta * s)) }
  }

  # Apply the renewal equation to estimate Rt at each time point
  for (i in seq_len(nrow(df))) {
    s_values <- seq_len(i)
    denominator <- sum(gen_interval_pmf(s) * df$cases[df$date == df$date[i] - s + 1], na.rm = TRUE)
    
    # Check if denominator is not zero to avoid division by zero error
    if (denominator > 0) {
      rt_estimates[i] <- denominator / df$cases[i]
    }
  }

  return(rt_estimates)
}

# Example usage:
case_data <- "cases.csv"
beta <- 1/5 # Rate parameter for exponential generation interval distribution

rt_estimates <- estimate_rt(case_data, beta)

# Plot Rt estimates over time
ggplot(data.frame(date = unique(df$date), rt_estimate = rt_estimates), aes(x = date, y = rt_estimate)) +
  geom_line() + 
  labs(title = "Time-Varying Reproduction Number (Rt) Estimates", x = "Date", y = "Rt") +
  theme_classic()

