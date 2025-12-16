# Setup R environment for experiment execution
# Run with: Rscript setup/setup_r.R

message("Setting up R environment...")

# CRAN packages
cran_packages <- c(
  "tidyverse",
  "tidyr",
  "EpiEstim",
  "readr",
  "dplyr",
  "ggplot2",
  "lubridate",
  "jsonlite",
  "bayesplot",
  "posterior",
  "httr2"
)

# Install CRAN packages
for (pkg in cran_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(sprintf("%s already installed", pkg))
  }
}

# EpiNow2 (may need additional dependencies)
if (!requireNamespace("EpiNow2", quietly = TRUE)) {
  message("Installing EpiNow2...")
  install.packages("EpiNow2", repos = "https://cloud.r-project.org")
}

# cmdstanr for Stan models
if (!requireNamespace("cmdstanr", quietly = TRUE)) {
  message("Installing cmdstanr...")
  install.packages("cmdstanr", repos = c(
    "https://mc-stan.org/r-packages/",
    "https://cloud.r-project.org"
  ))
}

# Check if CmdStan is installed
if (requireNamespace("cmdstanr", quietly = TRUE)) {
  cmdstan_path <- tryCatch(
    cmdstanr::cmdstan_path(),
    error = function(e) NULL
  )

  if (is.null(cmdstan_path)) {
    message("CmdStan not found. Installing...")
    cmdstanr::install_cmdstan()
  } else {
    message(sprintf("CmdStan found at: %s", cmdstan_path))
  }
}

# rstan as alternative
if (!requireNamespace("rstan", quietly = TRUE)) {
  message("Installing rstan...")
  install.packages("rstan", repos = "https://cloud.r-project.org")
}

message("\nR environment setup complete!")
message("Installed packages:")
for (pkg in c(cran_packages, "EpiNow2", "cmdstanr", "rstan")) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    v <- as.character(packageVersion(pkg))
    message(sprintf("  %s: %s", pkg, v))
  }
}
