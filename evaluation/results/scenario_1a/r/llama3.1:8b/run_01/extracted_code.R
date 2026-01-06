# Install required packages if not already installed
install.packages("dplyr")
install.packages("ggplot2")

# Load necessary libraries
library(dplyr)
library(ggplot2)

# Load data from the CSV file
cases <- read.csv("cases.csv", stringsAsFactors = FALSE)

# Convert date to Date format for proper sorting and plotting
cases$date <- as.Date(cases$date)

# Define a function to estimate Rt using SIR model
estimate_rt <- function(data) {
  # Fit an ARIMA model to the case counts
  arima_model <- auto.arima(data$cases, order = c(1, 1, 0), seasonal = FALSE)
  
  # Extract parameters of the fitted ARIMA model
  phi <- coefficients(arima_model)[1]
  theta <- coefficients(arima_model)[2]
  
  # Calculate Rt using formula from SIR model and ARIMA parameters
  rt <- (phi + theta) / (1 - phi * theta)
  
  return(rt)
}

# Apply the function to each time point and store estimates in a new data frame
cases_rt <- cases %>%
  group_by(date) %>%
  summarise(rts = estimate_rt(.$cases))

# Plot Rt over time
ggplot(cases_rt, aes(x = date, y = rts)) +
  geom_line() +
  labs(title = "Time-varying Reproduction Number (Rt)", x = "Date", y = "Rt")

