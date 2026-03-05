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

# Collect code files from a run directory (top-level only, exclude data/ etc.)
get_code_files <- function(run_dir) {
  pattern <- paste0("\\.(", paste(CODE_EXTENSIONS, collapse = "|"), ")$")
  files <- list.files(run_dir, pattern = pattern, full.names = TRUE)
  # Sort for deterministic ordering
  sort(files)
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

  for (i in order_idx) {
    sub_id <- mapping$submission_id[i]
    scenario <- mapping$scenario[i]
    execution <- mapping$execution[i]
    run <- runs[[i]]

    code_files <- get_code_files(run$run_dir)

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
          sprintf("### %s", basename(cf)),
          "",
          sprintf("```%s", lang),
          file_content,
          "```",
          ""
        )
      }
    }

    code_lines <- c(code_lines, "---", "")
  }

  writeLines(code_lines, file.path(REVIEW_DIR, "all_code.md"))
  message("Saved all_code.md")

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
    count_A_equivalent = character(n),
    count_B_minor = character(n),
    count_C_major = character(n),
    count_D_fundamental = character(n),
    overall = character(n),
    notes = character(n),
    stringsAsFactors = FALSE
  )

  write_csv(scoresheet, file.path(REVIEW_DIR, "scoresheet.csv"))
  message("Saved scoresheet.csv")

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
