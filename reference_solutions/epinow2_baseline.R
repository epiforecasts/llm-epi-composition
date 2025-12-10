# Reference Solution: EpiNow2 Package Baseline
#
# This provides a "package baseline" estimate of Rt using the established
# EpiNow2 package. LLM-generated model outputs can be compared against this.

library(EpiNow2)
library(data.table)
library(ggplot2)

# ---- Load Data ----
cases <- fread("data/cases.csv")
cases <- cases[, .(date = as.Date(date), confirm = cases)]

# ---- Define Epidemiological Parameters ----

# Generation interval (as provided in prompts)
# PMF: [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
# This corresponds approximately to a Gamma with mean ~3.5 days
generation_time <- LogNormal(
  mean = Normal(3.5, 0.5),
  sd = Normal(1.5, 0.3),
  max = 10
)

# Reporting delay (as provided in prompts)
# PMF: [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]
# This corresponds approximately to a Gamma with mean ~4 days
reporting_delay <- LogNormal(
  mean = Normal(4.0, 0.5),
  sd = Normal(1.5, 0.3),
  max = 10
)

# Incubation period (standard COVID-19 estimate)
incubation_period <- LogNormal(
  mean = Normal(5.0, 0.5),
  sd = Normal(1.5, 0.3),
  max = 14
)

# ---- Run EpiNow2 ----
message("Running EpiNow2 estimation...")

estimates <- epinow(
  data = cases,
  generation_time = generation_time_opts(generation_time),
  delays = delay_opts(incubation_period + reporting_delay),
  rt = rt_opts(
    prior = list(mean = 1.0, sd = 0.5),
    rw = 7  # Weekly random walk
  ),
  stan = stan_opts(
    cores = 4,
    chains = 4,
    samples = 2000,
    warmup = 500
  ),
  verbose = interactive()
)

# ---- Extract Results ----
rt_estimates <- estimates$estimates$summarised[variable == "R"]

# Save results
fwrite(rt_estimates, "outputs/epinow2_rt_estimates.csv")
message("Results saved to outputs/epinow2_rt_estimates.csv")

# ---- Plot ----
p <- plot(estimates, type = "R")
ggsave("outputs/epinow2_rt_plot.png", p, width = 10, height = 6)
message("Plot saved to outputs/epinow2_rt_plot.png")

# ---- Summary ----
message("\nEpiNow2 Rt estimation complete")
message("Date range: ", min(rt_estimates$date), " to ", max(rt_estimates$date))
message("Median Rt range: ",
        round(min(rt_estimates$median, na.rm = TRUE), 2), " to ",
        round(max(rt_estimates$median, na.rm = TRUE), 2))
