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

# Find the last code file written and executed by the LLM from conversation.jsonl
get_final_code_file <- function(run_dir) {
  conv_file <- file.path(run_dir, "conversation.jsonl")
  if (!file.exists(conv_file)) return(NULL)

  raw <- readLines(conv_file, warn = FALSE)
  raw <- paste(raw, collapse = "\n")

  exts <- paste(tolower(CODE_EXTENSIONS), collapse = "|")

  # Find all files that were written (Write tool calls)
  write_pattern <- paste0('"file_path":"[^"]*?/([^/"]+[.](?:', exts, '))","content"')
  write_matches <- gregexpr(write_pattern, raw, ignore.case = TRUE, perl = TRUE)
  written_files <- character(0)
  if (write_matches[[1]][1] != -1) {
    for (pos in write_matches[[1]]) {
      chunk <- substr(raw, pos, pos + 300)
      m <- regmatches(chunk, regexec(write_pattern, chunk,
                                     ignore.case = TRUE, perl = TRUE))[[1]]
      if (length(m) >= 2) written_files <- c(written_files, m[2])
    }
  }

  # Find all files that were executed (Bash command calls)
  exec_pattern <- paste0('"command":"[^"]*?([\\w_-]+[.](?:', exts, '))')
  exec_matches <- gregexpr(exec_pattern, raw, ignore.case = TRUE, perl = TRUE)
  executed_files <- character(0)
  if (exec_matches[[1]][1] != -1) {
    for (pos in exec_matches[[1]]) {
      chunk <- substr(raw, pos, pos + 500)
      m <- regmatches(chunk, regexec(exec_pattern, chunk,
                                     ignore.case = TRUE, perl = TRUE))[[1]]
      if (length(m) >= 2) executed_files <- c(executed_files, m[2])
    }
  }

  # Last file that was both written and executed, excluding auxiliary scripts
  aux_pattern <- "^(plot|create_plot|simple_plot|summary|final_summary|show_result|monitor|inspect|extract|fix_plot|test|explore|debug|install)"
  both <- intersect(written_files, executed_files)
  if (length(both) > 1) {
    non_aux <- both[!grepl(aux_pattern, both, ignore.case = TRUE)]
    if (length(non_aux) > 0) both <- non_aux
  }
  if (length(both) == 0) {
    if (length(written_files) == 0) return(NULL)
    both <- written_files
    non_aux <- both[!grepl(aux_pattern, both, ignore.case = TRUE)]
    if (length(non_aux) > 0) both <- non_aux
  }

  # Pick the last one (most recent in conversation order)
  final <- both[length(both)]

  filepath <- file.path(run_dir, final)
  if (file.exists(filepath)) filepath else NULL
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

    final_file <- get_final_code_file(run$run_dir)

    code_lines <- c(code_lines,
      sprintf("## %s", sub_id),
      "",
      sprintf("**Scenario**: %s | **Execution**: %s", scenario, execution),
      ""
    )

    if (is.null(final_file)) {
      code_lines <- c(code_lines, "*No code files found.*", "")
    } else {
      ext <- tools::file_ext(final_file)
      lang <- ext_to_lang(ext)
      file_content <- readLines(final_file, warn = FALSE)

      code_lines <- c(code_lines,
        sprintf("```%s", lang),
        file_content,
        "```",
        ""
      )
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
