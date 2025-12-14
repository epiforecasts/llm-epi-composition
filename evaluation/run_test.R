# Test evaluation on one experiment per condition
# Use this to verify the setup before running full evaluation

source("evaluation/run_evaluation.R")

# Override results dir for test
RESULTS_DIR <<- file.path(PROJECT_DIR, "evaluation/test_results")

# Find one experiment per condition
conditions <- c("stan", "pymc", "turing", "epiaware", "r")

test_files <- sapply(conditions, function(cond) {
  files <- list.files(
    EXPERIMENTS_DIR,
    pattern = sprintf("/%s/.*run_01\\.json$", cond),
    recursive = TRUE,
    full.names = TRUE
  )
  if (length(files) > 0) files[1] else NA
})

test_files <- test_files[!is.na(test_files)]

print_diagnostics()

message(sprintf("Running test on %d experiments (one per condition)...\n", length(test_files)))

dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

results <- list()
for (f in test_files) {
  result <- tryCatch(
    run_single_experiment(f, RESULTS_DIR),
    error = function(e) {
      message(sprintf("ERROR: %s", e$message))
      list(error = e$message)
    }
  )
  results[[length(results) + 1]] <- result
}

message("\n========================================")
message("Test Summary:")
message("========================================")

for (r in results) {
  if (!is.null(r$condition)) {
    status <- if (isTRUE(r$tier1_success)) "SUCCESS" else "FAILED"
    message(sprintf("  %s: %s", r$condition, status))
  }
}
