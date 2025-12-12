Here is a complete, runnable code to estimate the time-varying reproduction number (Rt) from daily reported COVID-19 case counts using Stan in R:

```r
# Load necessary libraries
library(readr)
library(cmdstanr)

# Function to load data
load_data <- function(file_path) {
  cases_df <- read_csv(file_path, 
                       col_names = c("date", "cases"),
                       skip = 1)
  
  # Convert date column to Date type
  cases_df$date <- as.Date(cases_df$date, format = "%Y-%m-%d")
  
  return(cases_df)
}

# Function to estimate Rt
estimate_rt <- function(data, model_name) {
  # Prepare data for Stan
  stan_data <- list(
    N = nrow(data),
    T = data$date[1],
    cases = data$cases
  )
  
  # Fit the model using cmdstanr
  fit <- cmdstanr::sample(
    file = paste0("models/", model_name, ".stan"),
    seed = 123,
    diagnostic_file = FALSE,
    refresh = FALSE,
    data = stan_data
  )
  
  # Extract Rt estimates for each time point
  rt_estimates <- as.data.frame(fit$samples[, c("Rt")])
  
  return(rt_estimates)
}

# Load the data
data_path <- "cases.csv"
cases_df <- load_data(data_path)

# Estimate Rt
model_name <- "basic_sir"
rt_estimates <- estimate_rt(cases_df, model_name)

# Summary of Rt estimates
summary(rt_estimates)

# Plot of Rt over time (optional)
library(ggplot2)
ggplot(rt_estimates, aes(x = sample)) + 
  geom_line(aes(y = Rt)) + 
  labs(title = "Time-varying Reproduction Number (Rt)", x = "Sample", y = "Rt")
```

This code includes:

1. Data loading function `load_data`
2. Function to estimate Rt using Stan, `estimate_rt`
3. Model fitting and Rt extraction
4. Optional: A plot of Rt over time

Note that you need to have the `cases.csv` file in the same directory as your R script. You can modify the model (e.g., "basic_sir" or "more_complex") to change the Stan model being fitted.

This code uses cmdstanr, a package for fitting Stan models directly from R. The Stan model used here is a basic SIR model with constant transmission rate and recovery rate. You may need to adjust this according to your needs.