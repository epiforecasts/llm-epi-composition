# Run all experiments
# This script runs the full experiment matrix

source("experiments/run_experiment.R")

# Experiment configuration
scenarios <- c("1a", "1b", "2", "3")
conditions <- c("stan", "pymc", "turing", "epiaware")
llms <- c(
  "claude-sonnet-4-20250514",
  "gpt-4o",
  "llama3.1:8b"  # Local via ollama - LMIC-relevant open source option
)
n_runs <- 3

# Track progress
total <- length(scenarios) * length(conditions) * length(llms) * n_runs
completed <- 0
failed <- list()

message(sprintf("Starting %d experiments...", total))
message(sprintf("Scenarios: %s", paste(scenarios, collapse = ", ")))
message(sprintf("Conditions: %s", paste(conditions, collapse = ", ")))
message(sprintf("LLMs: %s", paste(llms, collapse = ", ")))
message("")

for (scenario in scenarios) {
  for (condition in conditions) {
    for (llm in llms) {
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
          Sys.sleep(2)  # 2 second delay between successful requests
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
