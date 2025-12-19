# Tier 2 Evaluation: Re-run failed experiments with minimal fixes
# This script:
# 1. Reads Tier 1 results
# 2. For failed experiments, applies allowed fixes (missing imports only)
# 3. Re-runs and records Tier 2 results

library(jsonlite)
library(dplyr)

# Source the fixes
source("evaluation/tier2_fixes.R")

# Define paths (same as run_evaluation.R)
PROJECT_DIR <- getwd()
DATA_DIR <- file.path(PROJECT_DIR, "data")
EXPERIMENTS_DIR <- file.path(PROJECT_DIR, "experiments")
RESULTS_DIR <- file.path(PROJECT_DIR, "evaluation", "results")

# Helper to write JSON
write_json_file <- function(x, path) {
  writeLines(toJSON(x, pretty = TRUE, auto_unbox = TRUE), path)
}

# Determine language from condition
get_language <- function(condition) {
  switch(condition,
    "r" = "r",
    "stan" = "r",  # Stan condition uses R wrapper
    "pymc" = "python",
    "turing" = "julia",
    "epiaware" = "julia",
    "r"  # default
 )
}

# Run Tier 2 evaluation on a single experiment
run_tier2_experiment <- function(experiment_file, tier1_result_dir, tier2_results_dir) {
  # Load experiment metadata
  experiment <- fromJSON(experiment_file)

  # Extract path components
  path_parts <- strsplit(experiment_file, "/")[[1]]
  scenario <- path_parts[length(path_parts) - 3]
  condition <- path_parts[length(path_parts) - 2]
  llm <- path_parts[length(path_parts) - 1]
  run_file <- path_parts[length(path_parts)]
  run_id <- gsub("run_|\\.json", "", run_file)

  # Create Tier 2 work directory
  tier2_work_dir <- file.path(tier2_results_dir, scenario, condition, llm, paste0("run_", run_id))
  dir.create(tier2_work_dir, recursive = TRUE, showWarnings = FALSE)

  # Read the original extracted code from Tier 1
  tier1_work_dir <- file.path(tier1_result_dir, scenario, condition, llm, paste0("run_", run_id))

  # Determine script extension and language
  language <- get_language(condition)
  script_ext <- switch(language, "r" = "R", "python" = "py", "julia" = "jl", "R")
  script_file <- file.path(tier1_work_dir, paste0("script.", script_ext))

  if (!file.exists(script_file)) {
    # Check for extracted_code files
    code_files <- list.files(tier1_work_dir, pattern = "extracted_code", full.names = TRUE)
    if (length(code_files) > 0) {
      script_file <- code_files[1]
      script_ext <- tools::file_ext(script_file)
      language <- switch(script_ext, "R" = "r", "py" = "python", "jl" = "julia", "r")
    } else {
      return(list(
        scenario = scenario,
        condition = condition,
        llm = llm,
        run_id = as.integer(run_id),
        tier2_attempted = FALSE,
        tier2_success = FALSE,
        fixes_applied = list(),
        error = "No script file found"
      ))
    }
  }

  # Read original code
  original_code <- paste(readLines(script_file, warn = FALSE), collapse = "\n")

  # Apply Tier 2 fixes
  fix_result <- apply_tier2_fixes(original_code, language)
  fixed_code <- fix_result$code
  fixes_applied <- fix_result$fixes

  # If no fixes were applied, skip re-running
  if (length(fixes_applied) == 0) {
    return(list(
      scenario = scenario,
      condition = condition,
      llm = llm,
      run_id = as.integer(run_id),
      tier2_attempted = FALSE,
      tier2_success = FALSE,
      fixes_applied = list(),
      error = "No fixes applicable"
    ))
  }

  # Write fixed code
  fixed_script <- file.path(tier2_work_dir, paste0("script.", script_ext))
  writeLines(fixed_code, fixed_script)

  # Write fixes log
  writeLines(fixes_applied, file.path(tier2_work_dir, "fixes_applied.txt"))

  # Copy data files
  data_files <- c("cases.csv", "cases_dow.csv", "observations.csv")
  for (df in data_files) {
    src <- file.path(DATA_DIR, df)
    if (file.exists(src)) {
      file.copy(src, tier2_work_dir, overwrite = TRUE)
    }
  }

  # Execute fixed code
  timeout <- 600  # 10 minutes
  start_time <- Sys.time()

  result <- tryCatch({
    if (language == "r") {
      cmd <- sprintf(
        "cd '%s' && timeout %d Rscript script.R > output.txt 2> error.txt",
        tier2_work_dir, timeout
      )
    } else if (language == "python") {
      python_path <- Sys.which("python")
      if (nchar(python_path) == 0) {
        python_path <- "python"
      }
      cmd <- sprintf(
        "cd '%s' && timeout %d %s script.py > output.txt 2> error.txt",
        tier2_work_dir, timeout, python_path
      )
    } else if (language == "julia") {
      cmd <- sprintf(
        "cd '%s' && timeout %d julia script.jl > output.txt 2> error.txt",
        tier2_work_dir, timeout
      )
    }

    exit_code <- system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)
    list(success = (exit_code == 0), timed_out = (exit_code == 124))
  }, error = function(e) {
    list(success = FALSE, timed_out = FALSE, error = conditionMessage(e))
  })

  end_time <- Sys.time()
  duration <- as.numeric(difftime(end_time, start_time, units = "secs"))

  # Read error output if failed
  error_file <- file.path(tier2_work_dir, "error.txt")
  error_msg <- ""
  if (file.exists(error_file)) {
    error_msg <- paste(readLines(error_file, warn = FALSE, n = 10), collapse = "\n")
  }

  # Save result
  result_data <- list(
    scenario = scenario,
    condition = condition,
    llm = llm,
    run_id = as.integer(run_id),
    tier2_attempted = TRUE,
    tier2_success = result$success,
    tier2_timed_out = result$timed_out,
    tier2_duration = duration,
    fixes_applied = fixes_applied,
    error_preview = if (!result$success) substr(error_msg, 1, 500) else ""
  )

  write_json_file(result_data, file.path(tier2_work_dir, "result.json"))

  return(result_data)
}

# Main execution
main <- function() {
  # Load Tier 1 summary
  tier1_summary <- read.csv("evaluation/results/summary.csv")

  # Filter to failed experiments only
  failed_experiments <- tier1_summary %>%
    filter(!tier1_success)

  cat(sprintf("Found %d failed Tier 1 experiments to process\n", nrow(failed_experiments)))

  # Create Tier 2 results directory
  tier2_results_dir <- "evaluation/tier2_results"
  dir.create(tier2_results_dir, recursive = TRUE, showWarnings = FALSE)

  # Process each failed experiment
  results <- list()

  for (i in seq_len(nrow(failed_experiments))) {
    row <- failed_experiments[i, ]

    # Construct experiment file path
    experiment_file <- file.path(
      EXPERIMENTS_DIR,
      paste0("scenario_", row$scenario),
      row$condition,
      row$llm,
      paste0("run_0", row$run_id, ".json")
    )

    cat(sprintf("\n[%d/%d] Processing: %s/%s/%s/run_%d\n",
                i, nrow(failed_experiments),
                row$scenario, row$condition, row$llm, row$run_id))

    result <- run_tier2_experiment(
      experiment_file,
      "evaluation/results",
      tier2_results_dir
    )

    if (result$tier2_attempted) {
      cat(sprintf("  Fixes applied: %s\n", paste(result$fixes_applied, collapse = ", ")))
      cat(sprintf("  Result: %s\n", if (result$tier2_success) "SUCCESS" else "FAILED"))
    } else {
      cat(sprintf("  Skipped: %s\n", result$error))
    }

    results[[i]] <- result
  }

  # Create summary
  summary_df <- do.call(rbind, lapply(results, function(r) {
    data.frame(
      scenario = r$scenario,
      condition = r$condition,
      llm = r$llm,
      run_id = r$run_id,
      tier2_attempted = r$tier2_attempted,
      tier2_success = if (is.null(r$tier2_success)) FALSE else r$tier2_success,
      tier2_timed_out = if (is.null(r$tier2_timed_out)) FALSE else r$tier2_timed_out,
      fixes_applied = paste(unlist(r$fixes_applied), collapse = "; "),
      stringsAsFactors = FALSE
    )
  }))

  write.csv(summary_df, file.path(tier2_results_dir, "tier2_summary.csv"), row.names = FALSE)

  # Print summary
  cat("\n=== Tier 2 Summary ===\n")
  cat(sprintf("Total failed Tier 1 experiments: %d\n", nrow(failed_experiments)))
  cat(sprintf("Tier 2 fixes attempted: %d\n", sum(summary_df$tier2_attempted)))
  cat(sprintf("Tier 2 successes: %d\n", sum(summary_df$tier2_success)))

  # Combined results
  tier1_successes <- sum(tier1_summary$tier1_success)
  tier2_additional <- sum(summary_df$tier2_success)
  total <- nrow(tier1_summary)

  cat(sprintf("\n=== Combined Results ===\n"))
  cat(sprintf("Tier 1 successes: %d/%d (%.1f%%)\n", tier1_successes, total, 100 * tier1_successes / total))
  cat(sprintf("Tier 2 additional: %d\n", tier2_additional))
  cat(sprintf("Total with fixes: %d/%d (%.1f%%)\n",
              tier1_successes + tier2_additional, total,
              100 * (tier1_successes + tier2_additional) / total))
}

# Run if executed directly
if (!interactive()) {
  main()
}
