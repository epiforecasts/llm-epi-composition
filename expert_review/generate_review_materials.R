# Generate anonymised review materials for expert assessment
# Creates blinded code samples with random IDs

library(dplyr)
library(readr)

set.seed(42)

PROJECT_DIR <- getwd()
RUNS_DIR <- file.path(PROJECT_DIR, "runs")
REVIEW_DIR <- file.path(PROJECT_DIR, "expert_review")

SCENARIOS <- c("1a", "1b", "2", "3")
CONDITIONS <- c("r", "python", "julia", "epiaware")
LLM <- "claude-sonnet-4-20250514"
CODE_EXTENSIONS <- c("R", "py", "jl", "stan")

# Walk runs/ directory to find all runs
get_all_runs <- function() {
  runs <- list()
  for (scenario in SCENARIOS) {
    for (condition in CONDITIONS) {
      for (run_num in 1:3) {
        run_dir <- file.path(
          RUNS_DIR, paste0("scenario_", scenario),
          condition, LLM, sprintf("run_%02d", run_num)
        )
        if (!dir.exists(run_dir)) next
        runs[[length(runs) + 1]] <- list(
          scenario = scenario,
          condition = condition,
          run_id = run_num,
          run_dir = run_dir
        )
      }
    }
  }
  runs
}

# Load manually curated mapping of final code files per run
load_final_files <- function() {
  read_csv(file.path(REVIEW_DIR, "final_files.csv"),
           col_types = cols(.default = "c"))
}

# Get the final code file(s) for a run from the curated mapping
get_final_code_files <- function(scenario, condition, run_id, run_dir, final_files_df) {
  row <- final_files_df |>
    filter(.data$scenario == .env$scenario,
           .data$condition == .env$condition,
           .data$run_id == as.character(.env$run_id))
  if (nrow(row) == 0 || is.na(row$final_file[1]) || row$final_file[1] == "") {
    return(character(0))
  }
  filenames <- trimws(strsplit(row$final_file[1], ";")[[1]])
  paths <- file.path(run_dir, filenames)
  paths[file.exists(paths)]
}

# Determine execution status: SUCCESS if any CSV with "rt" in name exists
get_execution_status <- function(run_dir) {
  csvs <- list.files(run_dir, pattern = "\\.csv$",
                     full.names = TRUE, recursive = TRUE)
  csvs <- csvs[!grepl("/(rt_env|data)/", csvs)]
  has_rt_csv <- any(grepl("rt", basename(csvs), ignore.case = TRUE))
  if (has_rt_csv) "SUCCESS" else "FAILED"
}

# Determine language label for a file extension
ext_to_lang <- function(ext) {
  switch(tolower(ext),
    "r" = "r",
    "py" = "python",
    "jl" = "julia",
    "stan" = "stan",
    ""
  )
}

generate_review_materials <- function() {
  message("Generating expert review materials...")

  runs <- get_all_runs()
  n <- length(runs)
  message(sprintf("Found %d runs", n))

  # Assign random blinded submission IDs
  random_ids <- sprintf("SUB_%03d", sample(seq_len(n)))

  # Build mapping
  mapping <- data.frame(
    submission_id = random_ids,
    scenario = vapply(runs, `[[`, character(1), "scenario"),
    condition = vapply(runs, `[[`, character(1), "condition"),
    run_id = vapply(runs, `[[`, integer(1), "run_id"),
    stringsAsFactors = FALSE
  )

  # Add execution status
  mapping$execution <- vapply(runs, function(r) {
    get_execution_status(r$run_dir)
  }, character(1))

  # Save confidential mapping
  write_csv(mapping, file.path(REVIEW_DIR, "CONFIDENTIAL_mapping.csv"))
  message("Saved CONFIDENTIAL_mapping.csv")

  # Generate all_code.md (ordered by submission ID)
  order_idx <- order(mapping$submission_id)

  code_lines <- c(
    "# Expert Review: Code Submissions",
    "",
    "Use alongside scoresheet.csv for review.",
    "See README.md for review guidelines.",
    "",
    "---",
    ""
  )

  final_files_df <- load_final_files()

  for (i in order_idx) {
    sub_id <- mapping$submission_id[i]
    scenario <- mapping$scenario[i]
    execution <- mapping$execution[i]
    run <- runs[[i]]

    code_files <- get_final_code_files(
      run$scenario, run$condition, run$run_id, run$run_dir, final_files_df
    )

    code_lines <- c(code_lines,
      sprintf("## %s", sub_id),
      "",
      sprintf("**Scenario**: %s | **Execution**: %s", scenario, execution),
      ""
    )

    if (length(code_files) == 0) {
      code_lines <- c(code_lines, "*No code files found.*", "")
    } else {
      for (cf in code_files) {
        ext <- tools::file_ext(cf)
        lang <- ext_to_lang(ext)
        file_content <- readLines(cf, warn = FALSE)

        code_lines <- c(code_lines,
          sprintf("```%s", lang),
          file_content,
          "```",
          ""
        )
      }
    }

    code_lines <- c(code_lines, "---", "")
  }

  reviewer_dir <- file.path(REVIEW_DIR, "for_reviewers")
  dir.create(reviewer_dir, recursive = TRUE, showWarnings = FALSE)

  writeLines(code_lines, file.path(reviewer_dir, "all_code.md"))
  message("Saved for_reviewers/all_code.md")

  # Generate scoresheet.csv with columns from README.md
  scoresheet <- data.frame(
    submission_id = mapping$submission_id[order_idx],
    scenario = mapping$scenario[order_idx],
    execution = mapping$execution[order_idx],
    method_1a_only = character(n),
    no_delay = character(n),
    fixed_gi = character(n),
    wrong_gi = character(n),
    si_not_gi = character(n),
    poisson = character(n),
    no_smoothing = character(n),
    negative_rt = character(n),
    no_uncertainty = character(n),
    wrong_likelihood = character(n),
    confused_rt_r = character(n),
    no_discretisation = character(n),
    other_departures = character(n),
    overall = character(n),
    notes = character(n),
    stringsAsFactors = FALSE
  )

  write_csv(scoresheet, file.path(reviewer_dir, "scoresheet.csv"))
  message("Saved for_reviewers/scoresheet.csv")

  # Copy README into reviewer directory
  file.copy(file.path(REVIEW_DIR, "README.md"),
            file.path(reviewer_dir, "README.md"), overwrite = TRUE)

  # Summary
  message("\nSummary:")
  summary_df <- mapping |>
    group_by(scenario) |>
    summarise(
      n = n(),
      n_success = sum(execution == "SUCCESS"),
      .groups = "drop"
    )
  print(as.data.frame(summary_df))

  message("\nDone! Materials ready for expert review.")
}

if (!interactive()) {
  generate_review_materials()
}
