Here's a step-by-step solution in R using cmdstanr package.

```R
# Install necessary packages
install.packages("cmdstanr")
library(cmdstanr)

# Load required libraries
library(readr)
library(ggplot2)

# Load the data from CSV file
data <- read_csv("cases.csv")

# Define the model in Stan
model_code <- "
  data {
    int<lower=0> T; // number of time points
    vector[T] cases; // daily reported case counts
    real<lower=0,upper=1> gamma; // average reporting rate for each day
  }

  parameters {
    real<lower=0, upper=T> phi[T]; // Rt at each time point
  }

  transformed parameters {
    real<lower=0, upper=T> log_phi[T];
    for (t in 1:T) {
      log_phi[t] = normal_lpdf(phi[t]| -2.5, 1);
    }
  }

  model {
    cases ~ neg_binomial_2(log(phil), exp(-phi));
    phil ~ normal(0, 1);

    // prior for reporting rate
    gamma ~ uniform(0, 1);
  }
"

# Fit the model to data using cmdstanr
fit <- stan(
  file = "rt_model.stan", 
  data = list(cases = data$cases, T = nrow(data)), 
  chains = 4, iter = 10000, warmup = 5000, thin = 2
)

# Extract Rt estimates from posterior samples
posterior <- as.data.frame(fit)

Rt_estimates <- posterior[, "phi"]
date_Rt <- paste(posterior$date[1:nrow(posterior)], 
                 posterior$time[1:nrow(posterior)])

# Plot the estimated Rt over time (optional)
library(ggplot2)
ggplot(data.frame(date = date_Rt, rt = Rt_estimates), aes(x = date, y = rt)) +
  geom_line() + labs(title = "Estimated Reproduction Number Over Time", x = "", y = "") +
  theme_classic()
```

This code first loads the required libraries and data. It then defines a Stan model in string form (`model_code`), which is used to estimate Rt at each time point from daily case counts with an average reporting rate (gamma) as an input parameter.

We fit this model using `stan()` function from cmdstanr package, specifying chains = 4 for better posterior uncertainty and thinning = 2 for more efficient computation. After fitting the model, we extract Rt estimates from the posterior samples of the model's 'phi' parameters. We also save time points at which these estimates are made.

Finally, we create a plot showing Rt over time using ggplot2 package to visualize our results. Note that you need to install cmdstanr and rstan packages if they're not already installed in your R environment.

Remember that Stan model code should be saved as 'rt_model.stan' (or any other .stan file name) for the fitting process to work properly.

Please note: The provided example assumes a basic familiarity with Stan modeling, cmdstanr package and ggplot2 library.