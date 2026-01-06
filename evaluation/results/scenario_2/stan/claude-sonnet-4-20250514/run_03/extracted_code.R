library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(lubridate)

# Load and prepare data
load_and_prepare_data <- function(file_path = "cases.csv") {
  # Load data
  data <- read.csv(file_path)
  data$date <- as.Date(data$date)
  data <- data[order(data$date), ]
  
  # Ensure we have complete time series
  date_seq <- seq(min(data$date), max(data$date), by = "day")
  complete_data <- data.frame(
    date = date_seq,
    day_of_week = as.numeric(format(date_seq, "%u"))
  )
  
  # Merge with observed data
  complete_data <- merge(complete_data, data[, c("date", "cases")], 
                        by = "date", all.x = TRUE)
  complete_data$cases[is.na(complete_data$cases)] <- 0
  
  return(complete_data)
}

# Generate generation interval (discretized gamma distribution)
generate_generation_interval <- function(max_days = 20, shape = 2.5, rate = 0.5) {
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # normalize
  return(pmf)
}

# Generate delay distribution (infection to reporting)
generate_delay_distribution <- function(max_days = 21, shape = 2.0, rate = 0.3) {
  days <- 0:(max_days-1)
  pmf <- dgamma(days + 1, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # normalize
  return(pmf)
}

# Prepare Stan data
prepare_stan_data <- function(data) {
  gen_interval <- generate_generation_interval()
  delay_dist <- generate_delay_distribution()
  
  stan_data <- list(
    T = nrow(data),
    cases = data$cases,
    day_of_week = data$day_of_week,
    
    # Generation interval
    G = length(gen_interval),
    gen_interval = gen_interval,
    
    # Delay distribution
    D = length(delay_dist),
    delay_dist = delay_dist,
    
    # Prior parameters
    rt_prior_mean = 1.0,
    rt_prior_sd = 0.5,
    
    # Smoothing parameters
    rt_random_walk_sd = 0.1,
    ascertainment_random_walk_sd = 0.05
  )
  
  return(stan_data)
}

# Load data
data <- load_and_prepare_data()
stan_data <- prepare_stan_data(data)

# Compile and fit model
model <- cmdstan_model("rt_model.stan")

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100,
  adapt_delta = 0.95,
  max_treedepth = 12
)

# Extract results
extract_results <- function(fit, data) {
  draws <- fit$draws()
  
  # Extract Rt estimates
  rt_summary <- summarise_draws(
    subset(draws, variable = "Rt"),
    mean, median, sd, ~quantile(.x, c(0.025, 0.25, 0.75, 0.975))
  )
  rt_summary$date <- data$date
  
  # Extract day-of-week effects
  dow_summary <- summarise_draws(
    subset(draws, variable = "day_of_week_effect"),
    mean, median, sd, ~quantile(.x, c(0.025, 0.975))
  )
  dow_summary$day_name <- c("Monday", "Tuesday", "Wednesday", "Thursday", 
                           "Friday", "Saturday", "Sunday")
  
  # Extract ascertainment
  ascert_summary <- summarise_draws(
    subset(draws, variable = "ascertainment"),
    mean, median, sd, ~quantile(.x, c(0.025, 0.25, 0.75, 0.975))
  )
  ascert_summary$date <- data$date
  
  return(list(
    rt = rt_summary,
    day_of_week = dow_summary,
    ascertainment = ascert_summary,
    fit = fit
  ))
}

# Extract results
results <- extract_results(fit, data)

# Create summary plots
create_plots <- function(results, data) {
  # Rt plot
  rt_plot <- ggplot(results$rt, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "blue") +
    geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "blue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         x = "Date", y = "Rt",
         subtitle = "Median with 50% and 95% credible intervals") +
    theme_minimal()
  
  # Day-of-week effects
  dow_plot <- ggplot(results$day_of_week, aes(x = day_name, y = mean)) +
    geom_col(fill = "steelblue", alpha = 0.7) +
    geom_errorbar(aes(ymin = `2.5%`, ymax = `97.5%`), width = 0.2) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Day-of-Week Reporting Effects",
         x = "Day of Week", y = "Multiplicative Effect",
         subtitle = "Mean with 95% credible intervals") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Ascertainment over time
  ascert_plot <- ggplot(results$ascertainment, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "green") +
    geom_ribbon(aes(ymin = `25%`, ymax = `75%`), alpha = 0.5, fill = "green") +
    geom_line(aes(y = median), color = "darkgreen", size = 1) +
    labs(title = "Time-varying Ascertainment Rate",
         x = "Date", y = "Ascertainment Rate",
         subtitle = "Median with 50% and 95% credible intervals") +
    theme_minimal()
  
  # Cases vs fitted
  fitted_summary <- summarise_draws(
    subset(results$fit$draws(), variable = "expected_cases"),
    mean, median, ~quantile(.x, c(0.025, 0.975))
  )
  fitted_summary$date <- data$date
  fitted_summary$observed <- data$cases
  
  cases_plot <- ggplot(fitted_summary, aes(x = date)) +
    geom_ribbon(aes(ymin = `2.5%`, ymax = `97.5%`), alpha = 0.3, fill = "orange") +
    geom_line(aes(y = median), color = "darkorange", size = 1) +
    geom_point(aes(y = observed), color = "black", alpha = 0.6) +
    labs(title = "Observed vs Fitted Cases",
         x = "Date", y = "Cases",
         subtitle = "Black points: observed, Orange: fitted with 95% CI") +
    theme_minimal()
  
  return(list(rt = rt_plot, dow = dow_plot, ascertainment = ascert_plot, 
              cases = cases_plot))
}

plots <- create_plots(results, data)

# Display plots
print(plots$rt)
print(plots$dow)
print(plots$ascertainment)
print(plots$cases)

# Print summary statistics
cat("\nRt Summary (recent 7 days):\n")
recent_rt <- tail(results$rt, 7)
print(recent_rt[, c("date", "median", "2.5%", "97.5%")])

cat("\nDay-of-week effects:\n")
print(results$day_of_week[, c("day_name", "mean", "2.5%", "97.5%")])

cat("\nModel diagnostics:\n")
print(fit$diagnostic_summary())

