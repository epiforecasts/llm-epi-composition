# Load required packages
if (!require(EpiEstim)) {
  install.packages("EpiEstim")
  library(EpiEstim)
}
if (!require(ggplot2)) {
  install.packages("ggplot2")
  library(ggplot2)
}
if (!require(dplyr)) {
  install.packages("dplyr")
  library(dplyr)
}
if (!require(lubridate)) {
  install.packages("lubridate")
  library(lubridate)
}

# Load the data
cases_data <- read.csv("cases.csv", stringsAsFactors = FALSE)

# Convert date column to Date format
cases_data$date <- as.Date(cases_data$date)

# Sort by date to ensure chronological order
cases_data <- cases_data[order(cases_data$date), ]

# Create a complete date sequence to handle any missing dates
date_seq <- seq.Date(from = min(cases_data$date), 
                     to = max(cases_data$date), 
                     by = "day")

# Create a complete dataset with all dates
complete_data <- data.frame(date = date_seq)
complete_data <- merge(complete_data, cases_data, by = "date", all.x = TRUE)

# Replace NA values with 0 for missing dates
complete_data$cases[is.na(complete_data$cases)] <- 0

# Ensure non-negative case counts
complete_data$cases[complete_data$cases < 0] <- 0

# Prepare data for EpiEstim (requires specific format)
# EpiEstim expects a data frame with columns: dates, I (incidence)
epi_data <- data.frame(
  dates = complete_data$date,
  I = complete_data$cases
)

# Define serial interval distribution
# Using gamma distribution parameters for COVID-19
# Mean serial interval: ~5.2 days, SD: ~5.1 days (based on literature)
mean_si <- 5.2
std_si <- 5.1

# Calculate shape and scale parameters for gamma distribution
# For gamma distribution: mean = shape * scale, variance = shape * scale^2
# std^2 = shape * scale^2, so shape = mean^2 / std^2, scale = std^2 / mean
si_shape <- (mean_si^2) / (std_si^2)
si_scale <- (std_si^2) / mean_si

# Create serial interval configuration
si_config <- make_config(
  list(
    mean_si = mean_si,
    std_si = std_si,
    si_parametric_distr = "G",  # Gamma distribution
    mcmc_control = make_mcmc_control(
      burnin = 1000,
      thin = 10,
      seed = 1
    )
  )
)

# Estimate Rt using a sliding window approach
# Window size of 7 days (weekly estimates)
window_size <- 7

# Ensure we have enough data points
if (nrow(epi_data) < window_size + 1) {
  stop("Insufficient data points for Rt estimation. Need at least ", 
       window_size + 1, " days of data.")
}

# Estimate Rt
rt_estimates <- estimate_R(
  incid = epi_data,
  method = "parametric_si",
  config = si_config
)

# Extract Rt results
rt_results <- rt_estimates$R

# Add dates to the results
# The first estimate corresponds to day (window_size + 1)
rt_results$date <- epi_data$dates[(window_size + 1):nrow(epi_data)]

# Create summary statistics
rt_summary <- rt_results %>%
  select(date, `Mean(R)`, `Quantile.0.025(R)`, `Quantile.0.975(R)`) %>%
  rename(
    Rt_mean = `Mean(R)`,
    Rt_lower = `Quantile.0.025(R)`,
    Rt_upper = `Quantile.0.975(R)`
  )

# Display summary
print("=== Rt Estimation Summary ===")
print(paste("Data period:", min(epi_data$dates), "to", max(epi_data$dates)))
print(paste("Total days:", nrow(epi_data)))
print(paste("Rt estimates available from:", min(rt_summary$date)))
print(paste("Serial interval - Mean:", mean_si, "days, SD:", std_si, "days"))
print("")
print("First 10 Rt estimates:")
print(head(rt_summary, 10))
print("")
print("Last 10 Rt estimates:")
print(tail(rt_summary, 10))
print("")
print("Overall Rt statistics:")
print(summary(rt_summary[, c("Rt_mean", "Rt_lower", "Rt_upper")]))

# Create visualization
p1 <- ggplot(complete_data, aes(x = date, y = cases)) +
  geom_col(fill = "steelblue", alpha = 0.7) +
  labs(title = "Daily COVID-19 Case Counts",
       x = "Date",
       y = "Number of Cases") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p2 <- ggplot(rt_summary, aes(x = date)) +
  geom_ribbon(aes(ymin = Rt_lower, ymax = Rt_upper), 
              fill = "gray", alpha = 0.3) +
  geom_line(aes(y = Rt_mean), color = "red", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       subtitle = "Red line: Mean Rt, Gray ribbon: 95% Credible Interval",
       x = "Date",
       y = "Rt") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  ylim(0, max(rt_summary$Rt_upper) * 1.1)

# Display plots
print(p1)
print(p2)

# Save results to CSV
write.csv(rt_summary, "rt_estimates.csv", row.names = FALSE)

# Additional analysis: periods where Rt > 1
epidemic_periods <- rt_summary %>%
  filter(Rt_lower > 1) %>%
  summarise(
    periods_above_1 = n(),
    max_rt = max(Rt_mean),
    date_max_rt = date[which.max(Rt_mean)]
  )

declining_periods <- rt_summary %>%
  filter(Rt_upper < 1) %>%
  summarise(
    periods_below_1 = n(),
    min_rt = min(Rt_mean),
    date_min_rt = date[which.min(Rt_mean)]
  )

print("=== Epidemic Analysis ===")
print(paste("Days with Rt significantly > 1 (lower CI > 1):", 
            ifelse(nrow(epidemic_periods) > 0, epidemic_periods$periods_above_1, 0)))
print(paste("Days with Rt significantly < 1 (upper CI < 1):", 
            ifelse(nrow(declining_periods) > 0, declining_periods$periods_below_1, 0)))

if (nrow(epidemic_periods) > 0 && epidemic_periods$periods_above_1 > 0) {
  print(paste("Maximum Rt:", round(epidemic_periods$max_rt, 2), 
              "on", epidemic_periods$date_max_rt))
}

if (nrow(declining_periods) > 0 && declining_periods$periods_below_1 > 0) {
  print(paste("Minimum Rt:", round(declining_periods$min_rt, 2), 
              "on", declining_periods$date_min_rt))
}

# Function to get Rt estimate for a specific date
get_rt_for_date <- function(target_date) {
  target_date <- as.Date(target_date)
  result <- rt_summary[rt_summary$date == target_date, ]
  if (nrow(result) > 0) {
    return(result)
  } else {
    return(paste("No Rt estimate available for", target_date))
  }
}

# Example usage of the function
print("=== Example: Get Rt for specific date ===")
if (nrow(rt_summary) > 10) {
  example_date <- rt_summary$date[10]
  print(paste("Rt estimate for", example_date, ":"))
  print(get_rt_for_date(example_date))
}

# Create a combined plot
library(gridExtra)
combined_plot <- grid.arrange(p1, p2, ncol = 1, heights = c(1, 1))

# Print final message
print("=== Analysis Complete ===")
print("Results saved to 'rt_estimates.csv'")
print("Rt estimates with 95% credible intervals are available in rt_summary dataframe")

