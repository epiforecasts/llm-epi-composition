# Run R condition experiments
# This script runs experiments for the new "R" condition (using R packages like EpiEstim/EpiNow2)

source("experiments/run_experiment.R")

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  llms_to_run <- c("claude-sonnet-4-20250514", "llama3.1:8b")
} else {
  llms_to_run <- strsplit(args[1], ",")[[1]]
}

# Experiment configuration - only the R condition
scenarios <- c("1a", "1b", "2", "3")
condition <- "r"
n_runs <- 3

# Track progress
total <- length(scenarios) * length(llms_to_run) * n_runs
completed <- 0
failed <- list()

message(sprintf("Starting %d R condition experiments...", total))
message(sprintf("Scenarios: %s", paste(scenarios, collapse = ", ")))
message(sprintf("Condition: %s", condition))
message(sprintf("LLMs: %s", paste(llms_to_run, collapse = ", ")))
message("")

for (scenario in scenarios) {
  for (llm in llms_to_run) {
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

      # Rate limiting - wait between requests
      if (result == "success") {
        if (grepl("^(claude|gpt)", llm)) {
          Sys.sleep(10)  # 10 second delay for commercial APIs
        } else {
          Sys.sleep(2)  # 2 second delay for local models
        }
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
