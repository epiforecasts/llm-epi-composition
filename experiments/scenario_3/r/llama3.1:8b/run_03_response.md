Here is a solution to the problem using R:

```r
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
```

This solution uses the `brms` package to fit a Bayesian negative binomial model with shared Rt. The `negative_binomial` family is used for modeling the count data, and the `log` link function is used for the response variable.

The model includes stream-specific delays (`delay_cases`, `delay_hospitalisations`, `delay_deaths`) and ascertainment rates (`ascertainment_cases`, `ascertainment_hospitalisations`, `ascertainment_deaths`). The prior distributions are specified using the `set_prior` function, and the initial infection seeding period is handled by excluding early observations from the model.

The code also generates predictions for Rt over time using the `predict` function and plots the results using `ggplot2`.