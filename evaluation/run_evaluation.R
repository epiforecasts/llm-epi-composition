# Run evaluation of LLM-generated code
# Extracts code from experiment JSONs and executes them

library(jsonlite)
library(stringr)

# Configuration
TIMEOUT_SECONDS <- 600  # 10 minutes
DATA_DIR <- "data"
EXPERIMENTS_DIR <- "experiments"
RESULTS_DIR <- "evaluation/results"

# Extract code blocks from markdown response
extract_code_blocks <- function(response, condition) {
  # Match fenced code blocks with language specifier
  pattern <- "```([a-zA-Z]*)\\s*\\n([\\s\\S]*?)```"
  matches <- str_match_all(response, pattern)[[1]]

  if (nrow(matches) == 0) {
    return(list(code = NULL, language = NULL))
  }

  # Determine expected language based on condition
  expected_lang <- switch(condition,
    "stan" = c("stan", "r"),
    "pymc" = c("python", "py"),
    "turing" = c("julia", "jl"),
    "epiaware" = c("julia", "jl"),
    "r" = c("r", "R"),
    c("r", "python", "julia", "stan")  # default
  )

  # Collect all code blocks
  code_blocks <- list()
  for (i in seq_len(nrow(matches))) {
    lang <- tolower(matches[i, 2])
    code <- matches[i, 3]

    # Normalize language names
    lang <- switch(lang,
      "py" = "python",
      "jl" = "julia",
      "" = guess_language(code, condition),
      lang
    )

    code_blocks[[length(code_blocks) + 1]] <- list(
      language = lang,
      code = code
    )
  }

  code_blocks
}

# Guess language from code content if not specified
guess_language <- function(code, condition) {
  # Use condition as primary hint
  switch(condition,
    "stan" = if (grepl("data\\s*\\{|parameters\\s*\\{", code)) "stan" else "r",
    "pymc" = "python",
    "turing" = "julia",
    "epiaware" = "julia",
    "r" = "r",
    "r"  # default
  )
}

# Prepare code for execution
prepare_code <- function(code_blocks, condition, work_dir) {
  # For Stan condition, we need both Stan and R code
  if (condition == "stan") {
    stan_code <- NULL
    r_code <- NULL

    for (block in code_blocks) {
      if (block$language == "stan") {
        stan_code <- c(stan_code, block$code)
      } else if (block$language == "r") {
        r_code <- c(r_code, block$code)
      }
    }

    # Write Stan model to file
    if (!is.null(stan_code)) {
      writeLines(paste(stan_code, collapse = "\n\n"),
                 file.path(work_dir, "model.stan"))
    }

    return(list(
      language = "r",
      code = paste(r_code, collapse = "\n\n"),
      stan_file = if (!is.null(stan_code)) "model.stan" else NULL
    ))
  }

  # For other conditions, concatenate all code of the expected language
  expected_lang <- switch(condition,
    "pymc" = "python",
    "turing" = "julia",
    "epiaware" = "julia",
    "r" = "r",
    "r"
  )

  code_parts <- sapply(code_blocks, function(b) {
    if (b$language == expected_lang) b$code else NULL
  })
  code_parts <- code_parts[!sapply(code_parts, is.null)]

  list(
    language = expected_lang,
    code = paste(code_parts, collapse = "\n\n")
  )
}

# Execute R code
execute_r <- function(code, work_dir, timeout) {
  code_file <- file.path(work_dir, "script.R")
  output_file <- file.path(work_dir, "output.txt")
  error_file <- file.path(work_dir, "error.txt")

  # Prepend setup code
  setup_code <- sprintf('
setwd("%s")
options(warn = 1)
', work_dir)

  writeLines(paste0(setup_code, "\n", code), code_file)

  # Run with timeout
  cmd <- sprintf(
    "cd '%s' && timeout %d Rscript script.R > output.txt 2> error.txt",
    work_dir, timeout
  )

  start_time <- Sys.time()
  exit_code <- system(cmd)
  duration <- as.numeric(Sys.time() - start_time, units = "secs")

  list(
    exit_code = exit_code,
    duration = duration,
    stdout = if (file.exists(output_file)) readLines(output_file, warn = FALSE) else NULL,
    stderr = if (file.exists(error_file)) readLines(error_file, warn = FALSE) else NULL,
    timed_out = exit_code == 124
  )
}

# Execute Python code
execute_python <- function(code, work_dir, timeout) {
  code_file <- file.path(work_dir, "script.py")
  output_file <- file.path(work_dir, "output.txt")
  error_file <- file.path(work_dir, "error.txt")

  # Prepend setup code
  setup_code <- sprintf('
import os
os.chdir("%s")
', work_dir)

  writeLines(paste0(setup_code, "\n", code), code_file)

  # Use venv if available
  python_cmd <- if (file.exists("venv_pymc/bin/python")) {
    "venv_pymc/bin/python"
  } else {
    "python3"
  }

  cmd <- sprintf(
    "cd '%s' && timeout %d %s script.py > output.txt 2> error.txt",
    work_dir, timeout, python_cmd
  )

  start_time <- Sys.time()
  exit_code <- system(cmd)
  duration <- as.numeric(Sys.time() - start_time, units = "secs")

  list(
    exit_code = exit_code,
    duration = duration,
    stdout = if (file.exists(output_file)) readLines(output_file, warn = FALSE) else NULL,
    stderr = if (file.exists(error_file)) readLines(error_file, warn = FALSE) else NULL,
    timed_out = exit_code == 124
  )
}

# Execute Julia code
execute_julia <- function(code, work_dir, timeout) {
  code_file <- file.path(work_dir, "script.jl")
  output_file <- file.path(work_dir, "output.txt")
  error_file <- file.path(work_dir, "error.txt")

  # Prepend setup code
  setup_code <- sprintf('
cd("%s")
', work_dir)

  writeLines(paste0(setup_code, "\n", code), code_file)

  cmd <- sprintf(
    "cd '%s' && timeout %d julia script.jl > output.txt 2> error.txt",
    work_dir, timeout
  )

  start_time <- Sys.time()
  exit_code <- system(cmd)
  duration <- as.numeric(Sys.time() - start_time, units = "secs")

  list(
    exit_code = exit_code,
    duration = duration,
    stdout = if (file.exists(output_file)) readLines(output_file, warn = FALSE) else NULL,
    stderr = if (file.exists(error_file)) readLines(error_file, warn = FALSE) else NULL,
    timed_out = exit_code == 124
  )
}

# Run a single experiment
run_single_experiment <- function(experiment_path, results_dir) {
  message(sprintf("Processing: %s", experiment_path))

  # Load experiment
  exp <- fromJSON(experiment_path, simplifyVector = FALSE)

  # Create work directory
  work_dir <- file.path(
    results_dir,
    sprintf("scenario_%s", exp$scenario),
    exp$condition,
    exp$llm,
    sprintf("run_%02d", exp$run_id)
  )
  dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)

  # Copy data files to work directory
  file.copy(file.path(DATA_DIR, "cases.csv"), work_dir, overwrite = TRUE)
  file.copy(file.path(DATA_DIR, "cases_dow.csv"), work_dir, overwrite = TRUE)
  file.copy(file.path(DATA_DIR, "observations.csv"), work_dir, overwrite = TRUE)

  # Extract code
  code_blocks <- extract_code_blocks(exp$response, exp$condition)

  if (length(code_blocks) == 0) {
    result <- list(
      scenario = exp$scenario,
      condition = exp$condition,
      llm = exp$llm,
      run_id = exp$run_id,
      tier1_success = FALSE,
      tier1_error = "No code blocks found in response",
      code_extracted = FALSE
    )
    write_json(result, file.path(work_dir, "result.json"), pretty = TRUE)
    return(result)
  }

  # Prepare code
  prepared <- prepare_code(code_blocks, exp$condition, work_dir)

  if (is.null(prepared$code) || nchar(trimws(prepared$code)) == 0) {
    result <- list(
      scenario = exp$scenario,
      condition = exp$condition,
      llm = exp$llm,
      run_id = exp$run_id,
      tier1_success = FALSE,
      tier1_error = "No executable code found for condition",
      code_extracted = FALSE
    )
    write_json(result, file.path(work_dir, "result.json"), pretty = TRUE)
    return(result)
  }

  # Save extracted code
  code_ext <- switch(prepared$language,
    "r" = "R",
    "python" = "py",
    "julia" = "jl",
    "txt"
  )
  writeLines(prepared$code, file.path(work_dir, sprintf("extracted_code.%s", code_ext)))

  # Execute code (Tier 1)
  message(sprintf("  Executing %s code...", prepared$language))

  exec_result <- switch(prepared$language,
    "r" = execute_r(prepared$code, work_dir, TIMEOUT_SECONDS),
    "python" = execute_python(prepared$code, work_dir, TIMEOUT_SECONDS),
    "julia" = execute_julia(prepared$code, work_dir, TIMEOUT_SECONDS),
    list(exit_code = 1, error = "Unknown language")
  )

  # Check for output files (Rt estimates, plots)
  output_files <- list.files(work_dir, pattern = "\\.(csv|png|pdf|json)$")

  # Build result
  result <- list(
    scenario = exp$scenario,
    condition = exp$condition,
    llm = exp$llm,
    run_id = exp$run_id,
    code_extracted = TRUE,
    language = prepared$language,
    tier1_success = exec_result$exit_code == 0,
    tier1_exit_code = exec_result$exit_code,
    tier1_timed_out = exec_result$timed_out,
    tier1_duration = exec_result$duration,
    tier1_stdout = paste(exec_result$stdout, collapse = "\n"),
    tier1_stderr = paste(exec_result$stderr, collapse = "\n"),
    output_files = output_files
  )

  # Save result
  write_json(result, file.path(work_dir, "result.json"), pretty = TRUE, auto_unbox = TRUE)

  message(sprintf("  Result: %s (%.1fs)",
                  if (result$tier1_success) "SUCCESS" else "FAILED",
                  result$tier1_duration))

  result
}

# Main function
run_all_evaluations <- function() {
  # Create results directory
  dir.create(RESULTS_DIR, recursive = TRUE, showWarnings = FALSE)

  # Find all experiment files
  experiment_files <- list.files(
    EXPERIMENTS_DIR,
    pattern = "run_\\d+\\.json$",
    recursive = TRUE,
    full.names = TRUE
  )

  message(sprintf("Found %d experiments to evaluate", length(experiment_files)))

  # Run each experiment
  results <- list()
  for (i in seq_along(experiment_files)) {
    message(sprintf("\n[%d/%d]", i, length(experiment_files)))
    result <- tryCatch(
      run_single_experiment(experiment_files[i], RESULTS_DIR),
      error = function(e) {
        message(sprintf("  ERROR: %s", e$message))
        list(
          file = experiment_files[i],
          error = e$message
        )
      }
    )
    results[[i]] <- result
  }

  # Summary
  message("\n========================================")
  message("Evaluation complete!")

  successes <- sum(sapply(results, function(r) isTRUE(r$tier1_success)))
  failures <- sum(sapply(results, function(r) isFALSE(r$tier1_success)))

  message(sprintf("Tier 1 Success: %d/%d (%.0f%%)",
                  successes, length(results), 100 * successes / length(results)))
  message(sprintf("Tier 1 Failed: %d/%d", failures, length(results)))
  message("========================================")

  # Save summary
  summary_df <- do.call(rbind, lapply(results, function(r) {
    data.frame(
      scenario = r$scenario %||% NA,
      condition = r$condition %||% NA,
      llm = r$llm %||% NA,
      run_id = r$run_id %||% NA,
      code_extracted = r$code_extracted %||% FALSE,
      tier1_success = r$tier1_success %||% FALSE,
      tier1_timed_out = r$tier1_timed_out %||% FALSE,
      tier1_duration = r$tier1_duration %||% NA,
      stringsAsFactors = FALSE
    )
  }))

  write.csv(summary_df, file.path(RESULTS_DIR, "summary.csv"), row.names = FALSE)

  results
}

# Null coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# Run if executed directly
if (!interactive()) {
  results <- run_all_evaluations()
}
