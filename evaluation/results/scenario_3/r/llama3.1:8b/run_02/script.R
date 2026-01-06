
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/r/llama3.1:8b/run_02")
options(warn = 1)

# Load necessary libraries
library(ggplot2)
library(dplyr)
library(bayesplot)

# Load data
observations <- read.csv("observations.csv")

# Define model function
model_function <- function(data) {
  # Shared Rt model for infection dynamics with stream-specific delays and ascertainment rates
  # Overdispersion accounted by a negative binomial distribution
  
  # Stream-specific delays (in days)
  delay_cases <- 2
  delay_hospitalisations <- 5
  delay_deaths <- 10
  
  # Stream-specific ascertainment rates
  ascertainment_cases <- 0.8
  ascertainment_hospitalisations <- 0.95
  ascertainment_deaths <- 0.99
  
  # Shared Rt model
  Rt_model <- brms::brm(
    cases ~ (1|time) + (Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE,
    save_model = "shared_rt.model"
  )
  
  # Stream-specific models
  model_cases <- brm(
    cases ~ (1|time) + delay_cases*(Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE
  )
  
  model_hospitalisations <- brm(
    hospitalisations ~ (1|time) + delay_hospitalisations*(Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE
  )
  
  model_deaths <- brm(
    deaths ~ (1|time) + delay_deaths*(Rt),
    data = data,
    family = negative.binomial(link="log"),
    chains = 4, cores = 4,
    refresh = 0,
    stan_var_partitions = TRUE
  )
  
  # Posterior sampling of Rt from shared Rt model
  posterior_Rt <- post_summary(Rt_model)
  
  return(list(
    Rt_posterior = posterior_Rt,
    R_asymp = brms::fixef(model_cases)[1]
  ))
}

# Define data preparation function
data_preparation_function <- function(data) {
  # Prepare data for Bayesian analysis
  
  # Time variable in days since first day of observations
  data$time <- as.numeric(as.Date(data$date)) - min(as.Date(data$date))
  
  # Create separate data frames for each stream
  cases_data <- data %>% filter(!is.na(cases))
  hospitalisations_data <- data %>% filter(!is.na(hospitalisations))
  deaths_data <- data %>% filter(!is.na(deaths))
  
  return(list(
    cases_data = cases_data,
    hospitalisations_data = hospitalisations_data,
    deaths_data = deaths_data
  ))
}

# Load and fit the model
data <- data_preparation_function(observations)
model_results <- lapply(data, model_function)

# Extract results and plot Rt posterior distribution
Rt_posteriors <- sapply(model_results, function(x) x$Rt_posterior)
fixef_R_asymp <- sapply(model_results, function(x) x$R_asymp)

# Plot the Rt posterior distribution
ggplot(data.frame(date = observations$date), aes(x = date)) +
  geom_line(aes(y = mean(Rt_posteriors[1]), color = "Rt")) +
  theme_classic() +
  labs(title="Posterior Distribution of Rt", y = "Mean Rt") +
  scale_color_brewer(type = "qual", palette = "Set3")

