# Run GPT-4o experiments with extended delays
# This script runs only GPT-4o experiments with longer delays to avoid rate limits

source("experiments/run_experiment.R")

# Experiment configuration
scenarios <- c("1a", "1b", "2", "3")
conditions <- c("stan", "pymc", "turing", "epiaware")
llm <- "gpt-4o"
n_runs <- 3

# Track progress
total <- length(scenarios) * length(conditions) * n_runs
completed <- 0
failed <- list()

message(sprintf("Starting %d GPT-4o experiments...", total))
message(sprintf("Scenarios: %s", paste(scenarios, collapse = ", ")))
message(sprintf("Conditions: %s", paste(conditions, collapse = ", ")))
message("Using 60 second delay between requests to avoid rate limits")
message("")

for (scenario in scenarios) {
  for (condition in conditions) {
    for (run_id in 1:n_runs) {
      completed <- completed + 1
      message(sprintf("\n[%d/%d] scenario=%s, condition=%s, llm=%s, run=%d",
                     completed, total, scenario, condition, llm, run_id))

      # Check if already done
      exp_dir <- file.path("experiments", paste0("scenario_", scenario), condition, llm)
      output_file <- file.path(exp_dir, sprintf("run_%02d.json", run_id))

      if (file.exists(output_file)) {
        message("  -> Already completed, skipping")
        next
      }

      # Run experiment with error handling
      result <- tryCatch({
        run_experiment(
          scenario = scenario,
          condition = condition,
          llm = llm,
          run_id = run_id
        )
        "success"
      }, error = function(e) {
        message(sprintf("  -> ERROR: %s", e$message))
        failed <<- c(failed, list(list(
          scenario = scenario,
          condition = condition,
          llm = llm,
          run_id = run_id,
          error = e$message
        )))
        "failed"
      })

      # Extended delay between GPT-4o requests (60 seconds)
      if (result == "success") {
        message("  -> Waiting 60 seconds before next request...")
        Sys.sleep(60)
      } else {
        # Even longer delay after failure
        message("  -> Waiting 120 seconds after failure...")
        Sys.sleep(120)
      }
    }
  }
}

message("\n========================================")
message(sprintf("Completed: %d/%d experiments", completed - length(failed), total))

if (length(failed) > 0) {
  message(sprintf("Failed: %d experiments", length(failed)))
  message("\nFailed experiments:")
  for (f in failed) {
    message(sprintf("  - %s/%s/%s/run_%02d: %s",
                   f$scenario, f$condition, f$llm, f$run_id, f$error))
  }
}

message("========================================")
