
setwd("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_3/r/llama3.1:8b/run_03")
options(warn = 1)

# Load necessary libraries
library(brms)
library(ggplot2)

# Load data
data <- read.csv("observations.csv")

# Define model structure and priors
model <- brm(
  formula = cases ~ rt + hospitalisations + deaths,
  family = negative_binomial,
  data = data,
  links = c("identity", "log"),
  include = ~1|date,
  chains = 4, cores = 4,
  iter = 20000, warmup = 10000, thin = 5,
  control = list(adapt_delta = 0.95),
  prior = set_prior("normal(0,2)", class = "b"),
  monitor = c("rt", "hospitalisations", "deaths")
)

# Define stream-specific delays and ascertainment rates
delay_cases <- 1
ascertainment_cases <- 0.8
delay_hospitalisations <- 7
ascertainment_hospitalisations <- 0.9
delay_deaths <- 14
ascertainment_deaths <- 0.95

# Generate predictions for Rt
new_data <- data.frame(
  date = unique(data$date)
)

predictions <- predict(model, newdata = new_data, type = "response", re_formula = NA)

# Plot Rt estimates over time
ggplot(new_data, aes(x = date)) +
  geom_line(aes(y = as.numeric(predictions)), color = "blue") +
  labs(title = "Estimated Rt over Time")

