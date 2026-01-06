# Load required libraries
library(tidyverse)
library(lubridate)
library(ggplot2)

# Function to estimate Rt from case counts
estimate_rt <- function(cases_data, 
                       serial_interval_mean = 5.2,
                       serial_interval_sd = 2.8,
                       window_size = 7,
                       prior_mean = 1,
                       prior_sd = 2) {
  
  # Prepare data
  cases_data <- cases_data %>%
    arrange(date) %>%
    mutate(
      day = as.numeric(date - min(date)) + 1,
      cases = pmax(cases, 0)  # Ensure non-negative cases
    )
  
  n_days <- nrow(cases_data)
  
  # Generate serial interval distribution (discretized gamma distribution)
  max_si <- 20  # Maximum serial interval length
  si_distr <- dgamma(1:max_si, 
                    shape = (serial_interval_mean/serial_interval_sd)^2,
                    rate = serial_interval_mean/(serial_interval_sd^2))
  si_distr <- si_distr / sum(si_distr)  # Normalize
  
  # Calculate infectivity (convolution of cases with serial interval)
  infectivity <- numeric(n_days)
  
  for (t in 1:n_days) {
    infectivity[t] <- sum(
      cases_data$cases[pmax(1, t - length(si_distr) + 1):t] * 
      rev(si_distr[1:min(length(si_distr), t)])
    )
  }
  
  # Estimate Rt using sliding window approach
  rt_estimates <- data.frame(
    date = cases_data$date,
    day = cases_data$day,
    cases = cases_data$cases,
    infectivity = infectivity,
    rt_mean = NA,
    rt_lower = NA,
    rt_upper = NA
  )
  
  # Start estimation after sufficient data points
  start_day <- max(window_size, length(si_distr))
  
  for (t in start_day:n_days) {
    # Define window
    window_start <- max(1, t - window_size + 1)
    window_end <- t
    
    # Get cases and infectivity for window
    window_cases <- cases_data$cases[window_start:window_end]
    window_infectivity <- infectivity[window_start:window_end]
    
    # Remove days with zero infectivity to avoid division by zero
    valid_days <- window_infectivity > 0
    
    if (sum(valid_days) > 0) {
      window_cases <- window_cases[valid_days]
      window_infectivity <- window_infectivity[valid_days]
      
      # Bayesian estimation using Gamma-Poisson conjugacy
      # Prior: Rt ~ Gamma(alpha_prior, beta_prior)
      alpha_prior <- (prior_mean / prior_sd)^2
      beta_prior <- prior_mean / (prior_sd^2)
      
      # Posterior parameters
      alpha_post <- alpha_prior + sum(window_cases)
      beta_post <- beta_prior + sum(window_infectivity)
      
      # Posterior statistics
      rt_mean <- alpha_post / beta_post
      rt_var <- alpha_post / (beta_post^2)
      
      # 95% credible interval
      rt_lower <- qgamma(0.025, alpha_post, beta_post)
      rt_upper <- qgamma(0.975, alpha_post, beta_post)
      
      # Store results
      rt_estimates$rt_mean[t] <- rt_mean
      rt_estimates$rt_lower[t] <- rt_lower
      rt_estimates$rt_upper[t] <- rt_upper
    }
  }
  
  return(rt_estimates)
}

# Function to load and process data
load_case_data <- function(file_path) {
  data <- read.csv(file_path) %>%
    mutate(date = as.Date(date)) %>%
    arrange(date) %>%
    filter(!is.na(cases), cases >= 0)
  
  return(data)
}

# Function to create visualization
plot_rt_estimates <- function(rt_data) {
  
  # Plot 1: Cases over time
  p1 <- ggplot(rt_data, aes(x = date)) +
    geom_bar(aes(y = cases), stat = "identity", fill = "steelblue", alpha = 0.7) +
    labs(title = "Daily COVID-19 Cases",
         x = "Date", y = "Cases") +
    theme_minimal()
  
  # Plot 2: Rt estimates over time
  p2 <- rt_data %>%
    filter(!is.na(rt_mean)) %>%
    ggplot(aes(x = date)) +
    geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), 
                fill = "red", alpha = 0.3) +
    geom_line(aes(y = rt_mean), color = "red", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         subtitle = "Red line: Mean estimate, Gray ribbon: 95% credible interval",
         x = "Date", y = "Rt") +
    theme_minimal() +
    ylim(0, max(rt_data$rt_upper, na.rm = TRUE) * 1.1)
  
  return(list(cases_plot = p1, rt_plot = p2))
}

# Main execution function
main <- function() {
  
  # Check if data file exists, if not create sample data
  if (!file.exists("cases.csv")) {
    cat("Creating sample data file 'cases.csv'...\n")
    
    # Generate sample data
    set.seed(123)
    dates <- seq(from = as.Date("2023-01-01"), 
                to = as.Date("2023-06-30"), 
                by = "day")
    
    # Simulate epidemic curve with changing Rt
    n_days <- length(dates)
    true_rt <- c(rep(2.5, 30), rep(0.8, 60), rep(1.2, 40), rep(0.6, n_days - 130))
    
    cases <- numeric(n_days)
    cases[1:7] <- c(10, 12, 15, 18, 22, 28, 35)  # Initial cases
    
    # Simple simulation
    for (i in 8:n_days) {
      lambda <- sum(cases[max(1, i-10):(i-1)] * rev(dgamma(1:min(10, i-1), 2, 0.4)))
      cases[i] <- max(0, rpois(1, true_rt[min(i, length(true_rt))] * lambda * 0.1))
    }
    
    sample_data <- data.frame(
      date = dates,
      cases = cases
    )
    
    write.csv(sample_data, "cases.csv", row.names = FALSE)
  }
  
  # Load the data
  cat("Loading case data...\n")
  cases_data <- load_case_data("cases.csv")
  
  cat(sprintf("Loaded %d days of case data from %s to %s\n", 
              nrow(cases_data), 
              min(cases_data$date), 
              max(cases_data$date)))
  
  # Estimate Rt
  cat("Estimating Rt...\n")
  rt_results <- estimate_rt(cases_data)
  
  # Print summary
  rt_summary <- rt_results %>%
    filter(!is.na(rt_mean)) %>%
    summarise(
      n_estimates = n(),
      mean_rt = mean(rt_mean, na.rm = TRUE),
      median_rt = median(rt_mean, na.rm = TRUE),
      min_rt = min(rt_mean, na.rm = TRUE),
      max_rt = max(rt_mean, na.rm = TRUE),
      days_above_1 = sum(rt_mean > 1, na.rm = TRUE),
      days_below_1 = sum(rt_mean < 1, na.rm = TRUE)
    )
  
  cat("\n=== Rt Estimation Summary ===\n")
  cat(sprintf("Number of Rt estimates: %d\n", rt_summary$n_estimates))
  cat(sprintf("Mean Rt: %.2f\n", rt_summary$mean_rt))
  cat(sprintf("Median Rt: %.2f\n", rt_summary$median_rt))
  cat(sprintf("Range: %.2f - %.2f\n", rt_summary$min_rt, rt_summary$max_rt))
  cat(sprintf("Days with Rt > 1: %d (%.1f%%)\n", 
              rt_summary$days_above_1,
              rt_summary$days_above_1 / rt_summary$n_estimates * 100))
  cat(sprintf("Days with Rt < 1: %d (%.1f%%)\n", 
              rt_summary$days_below_1,
              rt_summary$days_below_1 / rt_summary$n_estimates * 100))
  
  # Create and display plots
  plots <- plot_rt_estimates(rt_results)
  print(plots$cases_plot)
  print(plots$rt_plot)
  
  # Save results
  write.csv(rt_results, "rt_estimates.csv", row.names = FALSE)
  cat("\nResults saved to 'rt_estimates.csv'\n")
  
  # Return results for further analysis
  return(rt_results)
}

# Run the analysis
rt_results <- main()

# Additional analysis functions
analyze_rt_trends <- function(rt_data) {
  rt_clean <- rt_data %>% filter(!is.na(rt_mean))
  
  # Identify periods of growth/decline
  rt_clean <- rt_clean %>%
    mutate(
      growth_phase = rt_mean > 1,
      rt_trend = case_when(
        rt_mean > 1.2 ~ "High transmission (Rt > 1.2)",
        rt_mean > 1.0 ~ "Growing (1.0 < Rt ≤ 1.2)",
        rt_mean > 0.8 ~ "Declining (0.8 < Rt ≤ 1.0)",
        TRUE ~ "Low transmission (Rt ≤ 0.8)"
      )
    )
  
  trend_summary <- rt_clean %>%
    count(rt_trend) %>%
    mutate(percentage = n / sum(n) * 100)
  
  cat("\n=== Transmission Phase Summary ===\n")
  print(trend_summary)
  
  return(rt_clean)
}

# Run additional analysis
rt_trends <- analyze_rt_trends(rt_results)

