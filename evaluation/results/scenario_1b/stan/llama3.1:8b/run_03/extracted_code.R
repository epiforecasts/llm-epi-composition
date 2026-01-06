# Load libraries
library(cmdstanr)
library(tidyverse)

# Load case counts from CSV
cases <- read_csv("cases.csv")

# Define Stan model for renewal equation (model.stan)
cat(read_file("renewal_model.stan"), file = "renewal_model.stan")

# Fit the model using cmdstanr
fit <- stan(
  file = "renewal_model.stan",
  data = list(n_cases = nrow(cases), 
              cases = cases$cases, 
              dates = as.numeric(as.Date(cases$date)), 
              S = 7, # 7-day generation interval distribution 
              T = 365) # Time horizon
)

# Extract Rt estimates for each time point
rt_estimates <- data.frame(date = fit$summary$Rt)

# Create a plot of Rt over time ( example )
ggplot(rt_estimates, aes(x = date)) + 
  geom_line() + 
  theme_classic()

