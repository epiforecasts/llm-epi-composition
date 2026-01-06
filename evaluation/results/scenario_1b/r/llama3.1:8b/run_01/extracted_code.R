# Install required libraries
install.packages("dplyr")
install.packages("ggplot2")

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Load the data from cases.csv into a dataframe called df
df <- read_csv("cases.csv")

# Ensure date is in a format that can be used for time-series analysis
df$date <- as.Date(df$date, "%Y-%m-%d")

# Remove any rows with missing case counts
df <- df %>% 
  filter(!is.na(cases))

# Define the generation interval probability mass function (pmf)
g_s <- dpois(1:10, lambda = 7) # Assume mean generation time is 7 days

# Function to estimate Rt using the renewal equation
estimate_rt <- function(df, g_s){
  # Initialize variables
  n_days <- max(df$date) - min(df$date) + 1
  n_cases <- rep(0, n_days)
  
  # Loop over each day and calculate cumulative cases
  for (i in 2:n_days){
    start_date <- as.Date(min(df$date), "%Y-%m-%d") + i - 2
    end_date <- as.Date(max(df$date), "%Y-%m-%d")
    
    # Filter data to only include relevant days and calculate cumulative cases
    sub_df <- df %>% 
      filter(date >= start_date & date <= end_date)
    
    n_cases[i] <- sum(sub_df$cases)
  }
  
  # Calculate Rt for each day using the renewal equation
  rt_estimates <- rep(0, length(n_days))
  for (i in 2:n_days){
    if (n_cases[i - 1] > 0){
      rt_estimates[i] <- n_cases[i] / (g_s * sum(g_s * n_cases[i - 1]))
    } else {
      rt_estimates[i] <- NA
    }
  }
  
  # Return Rt estimates and corresponding dates
  return(data.frame(date = as.Date(min(df$date), "%Y-%m-%d") + 0: (n_days-1),
                    rt_estimate = rt_estimates))
}

# Estimate Rt using the renewal equation framework
rt_df <- estimate_rt(df, g_s)

# Plot Rt over time
ggplot(rt_df, aes(x = date, y = rt_estimate)) +
  geom_line() +
  theme_classic()

