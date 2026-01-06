# Load required libraries
library(cmdstanr)
library(posterior)
library(bayesplot)
library(ggplot2)
library(dplyr)
library(readr)
library(tidyr)

# Load the data
observations <- read_csv("observations.csv")
observations$date <- as.Date(observations$date)

# Remove any rows with all NA observations
observations <- observations[rowSums(is.na(observations[,c("cases", "hospitalisations", "deaths")])) < 3,]

# Replace NAs with 0 (assuming missing = 0 observations)
observations[is.na(observations)] <- 0

print(paste("Data spans", min(observations$date), "to", max(observations$date)))
print(paste("Total days:", nrow(observations)))


# Prepare data for Stan
prepare_stan_data <- function(obs_data) {
  T <- nrow(obs_data)
  S <- 3  # cases, hospitalisations, deaths
  
  # Create observation matrix
  y <- array(0, dim = c(T, S))
  y[, 1] <- obs_data$cases
  y[, 2] <- obs_data$hospitalisations  
  y[, 3] <- obs_data$deaths
  
  # Generation interval (gamma distribution with mean 5.2, sd 1.72)
  max_gen <- 15
  gen_shape <- (5.2 / 1.72)^2
  gen_rate <- 5.2 / (1.72^2)
  generation_interval <- dgamma(1:max_gen, shape = gen_shape, rate = gen_rate)
  generation_interval <- generation_interval / sum(generation_interval)
  
  # Delay distribution priors (based on epidemiological knowledge)
  # Cases: shorter delay (mean ~5 days)
  # Hospitalisations: medium delay (mean ~10 days)  
  # Deaths: longer delay (mean ~18 days)
  delay_shape_prior_mean <- c(2.5, 3.0, 2.8)  
  delay_shape_prior_sd <- c(0.5, 0.7, 0.6)
  delay_rate_prior_mean <- c(0.5, 0.3, 0.15) 
  delay_rate_prior_sd <- c(0.2, 0.1, 0.05)
  
  list(
    T = T,
    S = S,
    max_delay = 30,
    max_gen = max_gen,
    y = y,
    generation_interval = generation_interval,
    delay_shape_prior_mean = delay_shape_prior_mean,
    delay_shape_prior_sd = delay_shape_prior_sd,
    delay_rate_prior_mean = delay_rate_prior_mean,
    delay_rate_prior_sd = delay_rate_prior_sd
  )
}

# Prepare the data
stan_data <- prepare_stan_data(observations)

# Compile the Stan model
model <- cmdstan_model("multi_stream_rt.stan")

# Fit the model
fit <- model$sample(
  data = stan_data,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000,
  refresh = 100,
  max_treedepth = 12,
  adapt_delta = 0.95
)

# Check diagnostics
fit$cmdstan_diagnose()
print(fit$summary(c("sigma_Rt", "ascertainment_rate", "phi")))


# Extract and summarize results
extract_results <- function(fit, observations) {
  # Extract Rt estimates
  rt_draws <- fit$draws("Rt", format = "draws_matrix")
  rt_summary <- posterior::summarise_draws(rt_draws, 
                                         mean, median, sd, 
                                         ~quantile(.x, c(0.025, 0.1, 0.25, 0.75, 0.9, 0.975)))
  
  rt_summary$date <- observations$date
  
  # Extract stream-specific parameters
  asc_draws <- fit$draws("ascertainment_rate", format = "draws_matrix")
  asc_summary <- posterior::summarise_draws(asc_draws, mean, median, sd,
                                          ~quantile(.x, c(0.025, 0.975)))
  asc_summary$stream <- c("Cases", "Hospitalisations", "Deaths")
  
  phi_draws <- fit$draws("phi", format = "draws_matrix")
  phi_summary <- posterior::summarise_draws(phi_draws, mean, median, sd,
                                          ~quantile(.x, c(0.025, 0.975)))
  phi_summary$stream <- c("Cases", "Hospitalisations", "Deaths")
  
  # Extract delay parameters
  delay_shape_draws <- fit$draws("delay_shape", format = "draws_matrix")
  delay_rate_draws <- fit$draws("delay_rate", format = "draws_matrix")
  
  delay_shape_summary <- posterior::summarise_draws(delay_shape_draws, mean, median)
  delay_rate_summary <- posterior::summarise_draws(delay_rate_draws, mean, median)
  
  # Calculate implied delay means
  delay_means <- delay_shape_summary$mean / delay_rate_summary$mean
  
  list(
    rt_estimates = rt_summary,
    ascertainment_rates = asc_summary,
    overdispersion = phi_summary,
    delay_means = data.frame(
      stream = c("Cases", "Hospitalisations", "Deaths"),
      delay_mean = delay_means
    )
  )
}

# Extract results
results <- extract_results(fit, observations)

# Display key results
cat("=== Rt Estimation Results ===\n\n")

cat("Stream-specific Ascertainment Rates:\n")
print(results$ascertainment_rates[, c("stream", "mean", "q2.5", "q97.5")])

cat("\nStream-specific Overdispersion Parameters:\n") 
print(results$overdispersion[, c("stream", "mean", "q2.5", "q97.5")])

cat("\nEstimated Delay Means (days from infection):\n")
print(results$delay_means)

cat(sprintf("\nFinal Rt estimate: %.2f (95%% CI: %.2f-%.2f)\n",
           tail(results$rt_estimates$mean, 1),
           tail(results$rt_estimates$q2.5, 1), 
           tail(results$rt_estimates$q97.5, 1)))


# Create visualization
create_plots <- function(results, observations) {
  # Rt plot
  rt_plot <- ggplot(results$rt_estimates, aes(x = date)) +
    geom_ribbon(aes(ymin = q2.5, ymax = q97.5), alpha = 0.3, fill = "steelblue") +
    geom_ribbon(aes(ymin = q25, ymax = q75), alpha = 0.5, fill = "steelblue") +
    geom_line(aes(y = median), color = "darkblue", size = 1) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "red") +
    labs(title = "Time-varying Reproduction Number (Rt)",
         subtitle = "Estimated jointly from cases, hospitalisations, and deaths",
         x = "Date", y = "Rt",
         caption = "Ribbons show 50% and 95% credible intervals") +
    theme_minimal() +
    theme(plot.title = element_text(size = 14, face = "bold"))
  
  # Prepare data for multi-stream plot
  obs_long <- observations %>%
    pivot_longer(cols = c(cases, hospitalisations, deaths),
                names_to = "stream", values_to = "observed") %>%
    mutate(stream = case_when(
      stream == "cases" ~ "Cases",
      stream == "hospitalisations" ~ "Hospitalisations", 
      stream == "deaths" ~ "Deaths"
    ))
  
  # Data streams plot
  streams_plot <- ggplot(obs_long, aes(x = date, y = observed)) +
    geom_line(color = "darkred") +
    geom_point(size = 0.8, alpha = 0.6, color = "darkred") +
    facet_wrap(~stream, scales = "free_y", ncol = 1) +
    labs(title = "Observed Data Streams",
         x = "Date", y = "Daily Count") +
    theme_minimal() +
    theme(strip.text = element_text(face = "bold"))
  
  # Parameter comparison plot
  param_data <- rbind(
    data.frame(results$ascertainment_rates[, c("stream", "mean", "q2.5", "q97.5")], 
               parameter = "Ascertainment Rate"),
    data.frame(results$overdispersion[, c("stream", "mean", "q2.5", "q97.5")], 
               parameter = "Overdispersion (phi)")
  )
  
  param_plot <- ggplot(param_data, aes(x = stream, y = mean)) +
    geom_pointrange(aes(ymin = q2.5, ymax = q97.5), size = 0.8) +
    facet_wrap(~parameter, scales = "free_y") +
    labs(title = "Stream-specific Parameters",
         x = "Data Stream", y = "Estimate") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          strip.text = element_text(face = "bold"))
  
  return(list(rt_plot = rt_plot, streams_plot = streams_plot, param_plot = param_plot))
}

# Create plots
plots <- create_plots(results, observations)

# Display plots
print(plots$rt_plot)
print(plots$streams_plot) 
print(plots$param_plot)

# Save plots
ggsave("rt_estimates.png", plots$rt_plot, width = 12, height = 6, dpi = 300)
ggsave("data_streams.png", plots$streams_plot, width = 10, height = 8, dpi = 300)
ggsave("parameters.png", plots$param_plot, width = 10, height = 6, dpi = 300)


# Model checking and validation
perform_model_checks <- function(fit, stan_data, observations) {
  cat("=== Model Diagnostics ===\n\n")
  
  # Check Rhat and ESS
  all_draws <- fit$draws()
  diagnostics <- posterior::summarise_draws(all_draws)
  
  max_rhat <- max(diagnostics$rhat, na.rm = TRUE)
  min_ess <- min(diagnostics$ess_bulk, na.rm = TRUE)
  
  cat(sprintf("Maximum Rhat: %.3f %s\n", max_rhat, 
              ifelse(max_rhat < 1.01, "(Good)", "(Check convergence)")))
  cat(sprintf("Minimum ESS: %.0f %s\n\n", min_ess,
              ifelse(min_ess > 400, "(Good)", "(Consider more iterations)")))
  
  # Posterior predictive checks
  y_rep <- fit$draws("y_rep", format = "draws_matrix")
  
  # Calculate summary statistics for each stream
  for (s in 1:3) {
    stream_names <- c("Cases", "Hospitalisations", "Deaths")
    observed <- stan_data$y[, s]
    
    # Extract replications for this stream
    y_rep_stream <- y_rep[, seq(s, ncol(y_rep), by = 3)]
    
    # Calculate test statistics
    obs_mean <- mean(observed)
    rep_means <- apply(y_rep_stream, 1, mean)
    
    obs_max <- max(observed)
    rep_maxs <- apply(y_rep_stream, 1, max)
    
    p_mean <- mean(rep_means > obs_mean)
    p_max <- mean(rep_maxs > obs_max)
    
    cat(sprintf("%s:\n", stream_names[s]))
    cat(sprintf("  Mean check: observed=%.1f, p-value=%.3f\n", obs_mean, p_mean))
    cat(sprintf("  Max check: observed=%d, p-value=%.3f\n", obs_max, p_max))
  }
}

# Run model checks
perform_model_checks(fit, stan_data, observations)

# Final summary
cat("\n=== Summary ===\n")
cat("Successfully estimated time-varying Rt jointly from three data streams\n")
cat("Key features implemented:\n")
cat("- Shared Rt across all streams via renewal equation\n")
cat("- Stream-specific delay distributions\n") 
cat("- Stream-specific ascertainment rates\n")
cat("- Negative binomial observation model with overdispersion\n")
cat("- Smooth Rt evolution via random walk\n")
cat("- Proper handling of initial infection seeding\n")

