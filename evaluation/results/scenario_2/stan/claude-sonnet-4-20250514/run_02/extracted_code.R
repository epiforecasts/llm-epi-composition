library(cmdstanr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)

# Function to create generation interval (gamma distribution discretized)
discretize_gamma <- function(shape = 2.5, rate = 0.5, max_days = 20) {
  days <- 1:max_days
  pmf <- dgamma(days, shape = shape, rate = rate)
  pmf <- pmf / sum(pmf)  # normalize to sum to 1
  return(pmf)
}

# Load and prepare data
prepare_data <- function(file_path = "cases.csv") {
  # If file doesn't exist, create sample data
  if (!file.exists(file_path)) {
    set.seed(123)
    dates <- seq(as.Date("2020-03-01"), as.Date("2020-12-31"), by = "day")
    n_days <- length(dates)
    
    # Simulate realistic COVID-like data
    true_rt <- c(rep(2.5, 30), 
                 2.5 * exp(-0.05 * (1:60)),  # decline
                 rep(0.8, 50),               # low
                 0.8 + 0.8 * plogis((1:100) - 50)/2,  # rise
                 rep(1.2, n_days - 240))     # moderate
    true_rt <- true_rt[1:n_days]
    
    # Simulate infections using renewal equation
    gen_interval <- discretize_gamma()
    infections <- numeric(n_days)
    infections[1:7] <- c(10, 15, 20, 25, 30, 35, 40)  # seed
    
    for(t in 8:n_days) {
      lambda <- true_rt[t] * sum(infections[max(1, t-20):(t-1)] * 
                                 rev(gen_interval[1:min(20, t-1)]))
      infections[t] <- rpois(1, lambda)
    }
    
    # Add reporting delays and day-of-week effects
    dow_effects <- c(0.8, 0.9, 1.0, 1.0, 1.0, 0.7, 0.5)  # Mon-Sun
    ascertainment <- 0.3 * (1 + 0.5 * sin(2*pi*(1:n_days)/365))  # seasonal
    
    cases <- numeric(n_days)
    for(t in 1:n_days) {
      # Reporting delay (mean 7 days)
      delay_dist <- dpois(0:14, 7)
      delay_dist <- delay_dist / sum(delay_dist)
      
      expected_cases <- 0
      for(d in 1:min(15, t)) {
        if(t-d+1 >= 1) {
          dow <- ((t-1) %% 7) + 1
          expected_cases <- expected_cases + 
            infections[t-d+1] * delay_dist[d] * ascertainment[t] * dow_effects[dow]
        }
      }
      cases[t] <- rnbinom(1, size = 10, mu = expected_cases)
    }
    
    data <- data.frame(
      date = dates,
      cases = pmax(0, cases),
      day_of_week = ((as.numeric(dates) - 1) %% 7) + 1
    )
    write.csv(data, "cases.csv", row.names = FALSE)
  } else {
    data <- read.csv(file_path)
    data$date <- as.Date(data$date)
  }
  
  return(data)
}

# Prepare Stan data
prepare_stan_data <- function(case_data) {
  gen_interval <- discretize_gamma()
  delay_pmf <- dpois(0:14, 7)
  delay_pmf <- delay_pmf / sum(delay_pmf)
  
  list(
    T = nrow(case_data),
    cases = case_data$cases,
    day_of_week = case_data$day_of_week,
    S = length(gen_interval),
    D = length(delay_pmf),
    generation_pmf = gen_interval,
    delay_pmf = delay_pmf,
    seeding_days = 14
  )
}

# Main execution
case_data <- prepare_data()
stan_data <- prepare_stan_data(case_data)

print(paste("Data spans", min(case_data$date), "to", max(case_data$date)))
print(paste("Total cases:", sum(case_data$cases)))


# Write Stan model to file
stan_code <- '
[Insert the Stan code from above here]
'
writeLines(stan_code, "rt_model.stan")

# Compile and fit model
model <- cmdstan_model("rt_model.stan")

fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 200,
  max_treedepth = 12,
  adapt_delta = 0.95
)

# Extract results
draws <- fit$draws()
summary_stats <- fit$summary()

# Extract key parameters
rt_summary <- summary_stats[grepl("^rt\\[", summary_stats$variable), ]
dow_summary <- summary_stats[grepl("^dow_effects\\[", summary_stats$variable), ]
ascertainment_summary <- summary_stats[grepl("^ascertainment\\[", summary_stats$variable), ]

# Create results dataframe
results <- case_data %>%
  mutate(
    rt_median = rt_summary$median,
    rt_lower = rt_summary$q5,
    rt_upper = rt_summary$q95,
    ascertainment_median = ascertainment_summary$median,
    ascertainment_lower = ascertainment_summary$q5,
    ascertainment_upper = ascertainment_summary$q95
  )

# Day of week effects
dow_effects <- data.frame(
  day = c("Monday", "Tuesday", "Wednesday", "Thursday", 
          "Friday", "Saturday", "Sunday"),
  effect = dow_summary$median,
  lower = dow_summary$q5,
  upper = dow_summary$q95
)

# Print summary
print("Day of Week Effects:")
print(dow_effects)

print("\nOverdispersion Parameter (phi):")
phi_summary <- summary_stats[summary_stats$variable == "phi", ]
print(phi_summary[c("median", "q5", "q95")])

# Create plots
p1 <- ggplot(results, aes(x = date)) +
  geom_ribbon(aes(ymin = rt_lower, ymax = rt_upper), alpha = 0.3, fill = "blue") +
  geom_line(aes(y = rt_median), color = "blue", size = 1) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
  labs(title = "Time-varying Reproduction Number (Rt)",
       x = "Date", y = "Rt") +
  theme_minimal()

p2 <- ggplot(results, aes(x = date)) +
  geom_ribbon(aes(ymin = ascertainment_lower, ymax = ascertainment_upper), 
              alpha = 0.3, fill = "green") +
  geom_line(aes(y = ascertainment_median), color = "green", size = 1) +
  labs(title = "Time-varying Ascertainment Rate",
       x = "Date", y = "Ascertainment") +
  theme_minimal()

p3 <- ggplot(dow_effects, aes(x = day, y = effect)) +
  geom_col(fill = "orange", alpha = 0.7) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.2) +
  geom_hline(yintercept = 1, linetype = "dashed") +
  labs(title = "Day of Week Effects",
       x = "Day of Week", y = "Relative Reporting") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

p4 <- ggplot(results, aes(x = date)) +
  geom_col(aes(y = cases), alpha = 0.5, fill = "gray") +
  labs(title = "Daily Reported Cases",
       x = "Date", y = "Cases") +
  theme_minimal()

# Display plots
print(p1)
print(p2)
print(p3)
print(p4)

# Model diagnostics
print("\nModel Diagnostics:")
print(fit$diagnostic_summary())

# Save results
write.csv(results, "rt_estimates.csv", row.names = FALSE)
write.csv(dow_effects, "dow_effects.csv", row.names = FALSE)

print("\nResults saved to rt_estimates.csv and dow_effects.csv")

