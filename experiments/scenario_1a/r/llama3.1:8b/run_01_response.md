Here's a step-by-step solution in R:

```r
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
```

**How it works:**

1. Load the `dplyr` and `ggplot2` packages for data manipulation and visualization.
2. Load the daily case count data from the CSV file into a data frame called `cases`.
3. Convert the date column to Date format using `as.Date()`, which is necessary for proper sorting and plotting.
4. Define an R function named `estimate_rt()` that estimates Rt at each time point using the SIR model, assuming a simple ARIMA(1,1,0) process with seasonality removed (set to `FALSE`). This involves fitting an ARIMA model to the case counts using the `auto.arima()` function and extracting its parameters.
5. Apply this function to each time point in the data frame using `group_by()` and `summarise()`, storing the estimates of Rt in a new data frame called `cases_rt`.
6. Visualize the estimated values of Rt over time using a simple line plot.

**Note:** This is a basic implementation for illustration purposes only. For more advanced analyses or real-world datasets, you might need to consider additional factors like time-varying effects, non-linear relationships, and uncertainty quantification, which would require modifications to this code.