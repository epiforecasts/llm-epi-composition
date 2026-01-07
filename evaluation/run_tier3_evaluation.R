# Tier 3 Evaluation: LLM Self-Fix (Multi-shot)
# Let LLMs see their errors and attempt to fix their own code
# This mirrors how people actually use LLMs (iterative debugging)
# Runs on ALL experiments, tracking which attempt produces working code

library(jsonlite)
library(stringr)
library(httr2)
library(readr)

# Configuration
MAX_FIX_ATTEMPTS <- 4  # Number of fix attempts (so 5 total: original + 4 fixes)
TIMEOUT_SECONDS <- 600  # 10 minutes per execution
PROJECT_DIR <- getwd()
DATA_DIR <- file.path(PROJECT_DIR, "data")
EXPERIMENTS_DIR <- file.path(PROJECT_DIR, "experiments")
RESULTS_DIR <- file.path(PROJECT_DIR, "evaluation", "results")
TIER3_DIR <- file.path(PROJECT_DIR, "evaluation", "tier3_results")

# Source the API functions from run_experiment.R
source(file.path(PROJECT_DIR, "experiments", "run_experiment.R"))

# Extract code blocks from markdown response
extract_code_blocks <- function(response, condition) {
  pattern <- "```([a-zA-Z]*)\\s*\\n([\\s\\S]*?)```"
  matches <- str_match_all(response, pattern)[[1]]

  if (nrow(matches) == 0) {
    return(list())
  }

  code_blocks <- list()
  for (i in seq_len(nrow(matches))) {
    lang <- tolower(matches[i, 2])
    code <- matches[i, 3]

    if (lang == "" || lang == "py") lang <- switch(condition,
      "pymc" = "python",
      "turing" = "julia",
      "epiaware" = "julia",
      "stan" = "r",
      "r" = "r",
      "r"
    )
    if (lang == "jl") lang <- "julia"

    code_blocks[[length(code_blocks) + 1]] <- list(language = lang, code = code)
  }
  code_blocks
}

# Create fix prompt - no constraints, let LLM change whatever it wants
create_fix_prompt <- function(original_prompt, original_code, error_output, language, attempt_num) {
  sprintf(
"I previously asked you to write code for the following task:

---
%s
---

You provided this %s code:

```%s
%s
```

However, when I ran it, I got the following error:

```
%s
```

Please fix the code so it runs successfully and produces the requested output (Rt estimates saved to a CSV file). Provide the complete corrected code.

This is fix attempt %d of %d.",
    original_prompt,
    language,
    language,
    original_code,
    error_output,
    attempt_num,
    MAX_FIX_ATTEMPTS
  )
}

# Execute code and capture output
execute_code <- function(code, language, work_dir, attempt_num = 0, timeout = TIMEOUT_SECONDS) {
  # Write code to file
  ext <- switch(language,
    "r" = "R",
    "python" = "py",
    "julia" = "jl",
    "R"
  )
  script_file <- file.path(work_dir, paste0("script.", ext))
  writeLines(code, script_file)

  # Build command - combine stdout and stderr to capture all output in order
  # This ensures we get the actual error message before any crash
  output_file <- file.path(work_dir, "output.txt")
  script_path <- file.path(work_dir, paste0("script.", ext))

  # Use absolute paths everywhere to avoid working directory issues
  cmd <- switch(language,
    "r" = sprintf("Rscript '%s'", script_path),
    "python" = sprintf("python '%s'", script_path),
    "julia" = sprintf("julia --project=@. --startup-file=no '%s'", script_path),
    sprintf("Rscript '%s'", script_path)
  )

  # Execute with timeout, combining stdout and stderr
  start_time <- Sys.time()

  # Build the command based on language
  # Use system2() for more reliable process handling
  output_file <- file.path(work_dir, "output.txt")

  message(sprintf("    Running: %s in %s", language, work_dir))

  # Change to work directory, run command, capture all output
  old_wd <- getwd()
  setwd(work_dir)

  result <- tryCatch({
    if (language == "julia") {
      # For Julia, use env -i to clear R's environment which can conflict with Julia libraries
      # Only keep essential vars like PATH and HOME
      out <- system2(
        "env",
        args = c(
          "-i",
          paste0("PATH=", Sys.getenv("PATH")),
          paste0("HOME=", Sys.getenv("HOME")),
          paste0("JULIA_DEPOT_PATH=", Sys.getenv("JULIA_DEPOT_PATH", unset = "~/.julia")),
          "timeout", as.character(timeout),
          "julia", "--project=@.", script_path
        ),
        stdout = TRUE,
        stderr = TRUE,
        timeout = timeout + 30
      )
      exit_code <- attr(out, "status")
      if (is.null(exit_code)) exit_code <- 0
      list(output = paste(out, collapse = "\n"), exit_code = exit_code)
    } else if (language == "python") {
      out <- system2(
        "timeout",
        args = c(as.character(timeout), "python", script_path),
        stdout = TRUE,
        stderr = TRUE,
        timeout = timeout + 30
      )
      exit_code <- attr(out, "status")
      if (is.null(exit_code)) exit_code <- 0
      list(output = paste(out, collapse = "\n"), exit_code = exit_code)
    } else {
      out <- system2(
        "timeout",
        args = c(as.character(timeout), "Rscript", script_path),
        stdout = TRUE,
        stderr = TRUE,
        timeout = timeout + 30
      )
      exit_code <- attr(out, "status")
      if (is.null(exit_code)) exit_code <- 0
      list(output = paste(out, collapse = "\n"), exit_code = exit_code)
    }
  }, error = function(e) {
    list(output = e$message, exit_code = -1)
  })

  setwd(old_wd)

  duration <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))

  # Save output to attempt-specific file for reference
  output_file_attempt <- file.path(work_dir, sprintf("output_attempt_%d.txt", attempt_num))
  writeLines(result$output, output_file_attempt)

  exit_code <- result$exit_code
  output_content <- result$output

  message(sprintf("    Exit code: %d, Output length: %d chars", exit_code, nchar(output_content)))

  list(
    success = exit_code == 0,
    exit_code = exit_code,
    output = output_content,
    timed_out = exit_code == 124,
    duration = duration
  )
}

# Check if Rt output file was produced
check_output_produced <- function(work_dir) {
  output_files <- list.files(work_dir, pattern = "rt_estimate|Rt_|_rt\\.", ignore.case = TRUE)
  length(output_files) > 0
}

# Run Tier 3 for a single experiment
run_tier3_single <- function(scenario, condition, llm, run_id, skip_completed = TRUE) {
  message(sprintf("\n=== Tier 3: %s/%s/%s/run_%02d ===", scenario, condition, llm, run_id))

  # Setup work directory path
  work_dir <- file.path(TIER3_DIR, paste0("scenario_", scenario), condition, llm,
                        sprintf("run_%02d", run_id))

  # Check if already completed (success or failure - to avoid bias from re-running failures)
  result_file <- file.path(work_dir, "tier3_result.json")
  if (skip_completed && file.exists(result_file)) {
    existing_result <- tryCatch(fromJSON(result_file), error = function(e) NULL)
    if (!is.null(existing_result)) {
      status <- if (isTRUE(existing_result$tier3_success)) "successfully" else "with failure"
      message(sprintf("  Already completed %s, skipping", status))
      return(existing_result)
    }
  }

  # Load original experiment
  exp_file <- file.path(EXPERIMENTS_DIR, paste0("scenario_", scenario), condition, llm,
                        sprintf("run_%02d.json", run_id))
  if (!file.exists(exp_file)) {
    message("  Experiment file not found, skipping")
    return(NULL)
  }

  exp_data <- fromJSON(exp_file)
  original_prompt <- exp_data$prompt
  original_response <- exp_data$response

  # Extract original code
  code_blocks <- extract_code_blocks(original_response, condition)
  if (length(code_blocks) == 0) {
    message("  No code blocks found, skipping")
    return(NULL)
  }

  # Get primary language
  language <- switch(condition,
    "pymc" = "python",
    "turing" = "julia",
    "epiaware" = "julia",
    "stan" = "r",
    "r" = "r",
    "r"
  )

  # Combine code blocks of the right language
  relevant_code <- paste(
    sapply(code_blocks, function(b) if (b$language == language) b$code else ""),
    collapse = "\n\n"
  )
  relevant_code <- trimws(relevant_code)

  if (nchar(relevant_code) == 0) {
    message("  No relevant code found, skipping")
    return(NULL)
  }

  # Create work directory if needed
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)

  # Copy data files from Tier 1 directory
  tier1_dir <- file.path(RESULTS_DIR, paste0("scenario_", scenario), condition, llm,
                         sprintf("run_%02d", run_id))
  data_files <- c("cases.csv", "cases_dow.csv", "observations.csv")
  for (f in data_files) {
    src <- file.path(tier1_dir, f)
    if (file.exists(src)) file.copy(src, work_dir, overwrite = TRUE)
  }

  # Track results for each attempt
  results <- list(
    scenario = scenario,
    condition = condition,
    llm = llm,
    run_id = run_id,
    attempts = list()
  )

  current_code <- relevant_code

  # Attempt 0: Run original code
  message("  Running original code (attempt 0)...")
  exec_result <- execute_code(current_code, language, work_dir, attempt_num = 0)
  output_produced <- check_output_produced(work_dir)

  results$attempts[[1]] <- list(
    attempt = 0,
    type = "original",
    success = exec_result$success && output_produced,
    exit_code = exec_result$exit_code,
    timed_out = exec_result$timed_out,
    duration = exec_result$duration,
    output_produced = output_produced
  )

  # Save original code
  writeLines(current_code, file.path(work_dir, sprintf("code_attempt_0.%s",
    switch(language, "python" = "py", "julia" = "jl", "R"))))

  if (results$attempts[[1]]$success) {
    message("  SUCCESS on original code!")
    results$tier3_success <- TRUE
    results$successful_attempt <- 0
  } else {
    message(sprintf("  Original failed (exit=%d, output=%s)", exec_result$exit_code, output_produced))

    current_error <- exec_result$output
    if (nchar(current_error) > 4000) {
      current_error <- paste0(substr(current_error, 1, 2000), "\n...[truncated]...\n",
                              substr(current_error, nchar(current_error) - 1500, nchar(current_error)))
    }

    # Try fix attempts
    for (attempt in 1:MAX_FIX_ATTEMPTS) {
      message(sprintf("  Fix attempt %d/%d...", attempt, MAX_FIX_ATTEMPTS))

      # Create fix prompt
      fix_prompt <- create_fix_prompt(original_prompt, current_code, current_error,
                                      language, attempt)

      # Call LLM
      fix_success <- FALSE
      tryCatch({
        if (grepl("^claude", llm)) {
          response <- call_claude(fix_prompt, model = llm)
          fix_response <- response$content[[1]]$text
        } else if (grepl("^llama", llm)) {
          response <- call_ollama(fix_prompt, model = llm)
          fix_response <- response$response
        } else {
          message("    Unknown LLM type, skipping")
          break
        }

        # Extract fixed code
        fix_blocks <- extract_code_blocks(fix_response, condition)
        fixed_code <- paste(
          sapply(fix_blocks, function(b) if (b$language == language) b$code else ""),
          collapse = "\n\n"
        )
        fixed_code <- trimws(fixed_code)

        if (nchar(fixed_code) == 0) {
          message("    No code in response")
          results$attempts[[attempt + 1]] <- list(
            attempt = attempt,
            type = "fix",
            success = FALSE,
            reason = "no_code_in_response"
          )
          next
        }

        # Save fixed code and response
        writeLines(fixed_code, file.path(work_dir, sprintf("code_attempt_%d.%s",
          attempt, switch(language, "python" = "py", "julia" = "jl", "R"))))
        writeLines(fix_response, file.path(work_dir, sprintf("response_attempt_%d.md", attempt)))

        # Execute fixed code
        message("    Executing fixed code...")
        exec_result <- execute_code(fixed_code, language, work_dir, attempt_num = attempt)
        output_produced <- check_output_produced(work_dir)

        attempt_result <- list(
          attempt = attempt,
          type = "fix",
          success = exec_result$success && output_produced,
          exit_code = exec_result$exit_code,
          timed_out = exec_result$timed_out,
          duration = exec_result$duration,
          output_produced = output_produced
        )
        results$attempts[[attempt + 1]] <- attempt_result

        if (attempt_result$success) {
          message(sprintf("    SUCCESS on fix attempt %d!", attempt))
          fix_success <- TRUE
          break
        } else {
          message(sprintf("    Failed (exit=%d, output=%s)", exec_result$exit_code, output_produced))
          current_code <- fixed_code
          current_error <- exec_result$output
          if (nchar(current_error) > 4000) {
            current_error <- paste0(substr(current_error, 1, 2000), "\n...[truncated]...\n",
                                    substr(current_error, nchar(current_error) - 1500, nchar(current_error)))
          }
        }

      }, error = function(e) {
        message(sprintf("    API error: %s", e$message))
        results$attempts[[attempt + 1]] <- list(
          attempt = attempt,
          type = "fix",
          success = FALSE,
          reason = "api_error",
          error = e$message
        )
      })

      if (fix_success) break
    }

    # Determine final status
    results$tier3_success <- any(sapply(results$attempts, function(a) isTRUE(a$success)))
    results$successful_attempt <- if (results$tier3_success) {
      attempts_success <- sapply(results$attempts, function(a) isTRUE(a$success))
      results$attempts[[which(attempts_success)[1]]]$attempt
    } else {
      NA
    }
  }

  # Save results
  result_file <- file.path(work_dir, "tier3_result.json")
  write_json(results, result_file, auto_unbox = TRUE, pretty = TRUE)

  message(sprintf("  Final: %s (attempt %s)",
                 if (results$tier3_success) "SUCCESS" else "FAILED",
                 if (is.na(results$successful_attempt)) "none" else results$successful_attempt))

  results
}

# Load all experiments
load_all_experiments <- function() {
  summary_file <- file.path(RESULTS_DIR, "summary.csv")
  if (!file.exists(summary_file)) {
    stop("summary.csv not found. Run Tier 1 evaluation first.")
  }

  summary <- read.csv(summary_file)
  message(sprintf("Found %d experiments to process", nrow(summary)))
  summary
}

# Main execution
run_tier3_evaluation <- function() {
  message("=== Tier 3 Evaluation: LLM Self-Fix (Multi-shot) ===")
  message(sprintf("Max fix attempts: %d (so %d total attempts per experiment)\n",
                 MAX_FIX_ATTEMPTS, MAX_FIX_ATTEMPTS + 1))

  dir.create(TIER3_DIR, recursive = TRUE, showWarnings = FALSE)

  # Load ALL experiments
  experiments <- load_all_experiments()

  all_results <- list()

  for (i in seq_len(nrow(experiments))) {
    row <- experiments[i, ]

    # Handle scenario naming
    scenario <- gsub("^scenario_", "", as.character(row$scenario))

    result <- tryCatch({
      run_tier3_single(
        scenario = scenario,
        condition = as.character(row$condition),
        llm = as.character(row$llm),
        run_id = row$run_id
      )
    }, error = function(e) {
      message(sprintf("  ERROR: %s", e$message))
      NULL
    })

    if (!is.null(result)) {
      all_results[[length(all_results) + 1]] <- result
    }

    # Save progress periodically
    if (i %% 10 == 0) {
      message(sprintf("\n=== Progress: %d/%d processed ===\n", i, nrow(experiments)))
    }
  }

  # Create summary
  summary_df <- do.call(rbind, lapply(all_results, function(r) {
    data.frame(
      scenario = r$scenario,
      condition = r$condition,
      llm = r$llm,
      run_id = r$run_id,
      tier3_success = r$tier3_success,
      successful_attempt = ifelse(is.na(r$successful_attempt), NA, r$successful_attempt),
      num_attempts = length(r$attempts),
      stringsAsFactors = FALSE
    )
  }))

  summary_file <- file.path(TIER3_DIR, "tier3_summary.csv")
  write.csv(summary_df, summary_file, row.names = FALSE)

  # Print summary
  message("\n========================================")
  message("=== Tier 3 Summary ===")
  message("========================================")
  message(sprintf("Total processed: %d", nrow(summary_df)))
  message(sprintf("Tier 3 successes: %d (%.1f%%)",
                 sum(summary_df$tier3_success),
                 100 * mean(summary_df$tier3_success)))

  message("\nSuccesses by attempt:")
  attempt_table <- table(summary_df$successful_attempt[summary_df$tier3_success])
  for (att in names(attempt_table)) {
    label <- if (att == "0") "Original (no fix needed)" else paste0("Fix attempt ", att)
    message(sprintf("  %s: %d", label, attempt_table[att]))
  }

  message("\nBy LLM:")
  for (llm in unique(summary_df$llm)) {
    subset <- summary_df[summary_df$llm == llm, ]
    message(sprintf("  %s: %d/%d (%.1f%%)",
                   llm, sum(subset$tier3_success), nrow(subset),
                   100 * mean(subset$tier3_success)))
  }

  message("\nBy condition:")
  for (cond in unique(summary_df$condition)) {
    subset <- summary_df[summary_df$condition == cond, ]
    message(sprintf("  %s: %d/%d (%.1f%%)",
                   cond, sum(subset$tier3_success), nrow(subset),
                   100 * mean(subset$tier3_success)))
  }

  message(sprintf("\nResults saved to: %s", TIER3_DIR))

  summary_df
}

# Run if called directly
if (!interactive() && sys.nframe() == 0L) {
  run_tier3_evaluation()
}
