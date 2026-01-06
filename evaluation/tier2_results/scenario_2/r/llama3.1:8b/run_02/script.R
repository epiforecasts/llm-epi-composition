
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/r/llama3.1:8b/run_02")
options(warn = 1)

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(purrr)
library(lme4)
library(bayesplot)

# Load data
cases <- read.csv("cases.csv")

# Define model structure
model <- function(cases, day_of_week) {
  # Delayed reporting term (infection to report delay: 5 days)
  delayed_cases <- cases %>% 
    mutate(delay = date - lag(date)) %>% 
    group_by(delay) %>% 
    summarise(sum_delayed = sum(cases))

  # Time-varying ascertainment
  ascertainment <- s(ascertainment_date) * I_t

  # Day-of-week effects
  day_effects <- day_of_week %>% 
    mutate(effect = ifelse(day == "Monday", 1, 
                           ifelse(day == "Tuesday", 0.85, 
                                  ifelse(day == "Wednesday", 0.8, 
                                         ifelse(day == "Thursday", 0.7, 
                                                ifelse(day == "Friday", 0.65, 
                                                       ifelse(day == "Saturday", 0.6, 
                                                              "Sunday"))))))))

  # Renewal equation
  I_t <- delayed_cases$sum_delayed * 
    day_effects$effect * ascertainment

  return(I_t)
}

# Fit model with Bayesian linear mixed effects model (BLMM)
fit <- blmer(Rt ~ date + (1|date), data = cases, REML = FALSE)

# Extract posterior estimates of Rt
posterior_rts <- posterior(fit)

# Plot posterior distribution of Rt estimates
ppc_densplot(posterior_rts, "Rt", parse = TRUE) +
  labs(title = "Posterior Distribution of Rt Estimates")

# Estimate day-of-week effects and time-varying ascertainment
day_effects <- cases %>% 
  group_by(day_of_week) %>% 
  summarise(mean_cases = mean(cases))

ascertainment_estimate <- cases %>% 
  group_by(date) %>% 
  summarise(estimate = mean(cases)) %>%
  mutate(estimate = estimate / (sum(delayed_cases$sum_delayed))) %>%
  arrange(date)

# Plot day-of-week effects and time-varying ascertainment
ggplot(day_effects, aes(x = day_of_week, y = mean_cases)) + 
  geom_bar(stat = "identity") +
  labs(title = "Day-of-Week Effects on Reported Cases")

ggplot(ascertainment_estimate, aes(x = date, y = estimate)) + 
  geom_line() +
  labs(title = "Time-Varying Ascertainment Estimates")

