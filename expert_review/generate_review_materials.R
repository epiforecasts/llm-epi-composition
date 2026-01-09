# Generate anonymized review materials for expert assessment
# Creates blinded code samples with random IDs

library(jsonlite)
library(dplyr)
library(tidyr)
library(readr)
library(stringr)

set.seed(42)  # For reproducible randomization

PROJECT_DIR <- getwd()
EXPERIMENTS_DIR <- file.path(PROJECT_DIR, "experiments")
RESULTS_DIR <- file.path(PROJECT_DIR, "evaluation", "results")
OUTPUT_DIR <- file.path(PROJECT_DIR, "expert_review", "submissions")

# Create output directory
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# Get all experiments
get_all_experiments <- function() {
  experiments <- list()

  for (scenario in c("1a", "1b", "2", "3")) {
    scenario_dir <- file.path(EXPERIMENTS_DIR, paste0("scenario_", scenario))
    if (!dir.exists(scenario_dir)) next

    for (condition in c("stan", "pymc", "turing", "epiaware", "r")) {
      condition_dir <- file.path(scenario_dir, condition)
      if (!dir.exists(condition_dir)) next

      for (llm in c("claude-sonnet-4-20250514", "llama3.1:8b")) {
        llm_dir <- file.path(condition_dir, llm)
        if (!dir.exists(llm_dir)) next

        for (run_id in 1:3) {
          exp_file <- file.path(llm_dir, sprintf("run_%02d.json", run_id))
          if (!file.exists(exp_file)) next

          experiments[[length(experiments) + 1]] <- list(
            scenario = scenario,
            condition = condition,
            llm = llm,
            run_id = run_id,
            exp_file = exp_file
          )
        }
      }
    }
  }

  experiments
}

# Extract code from experiment
extract_code <- function(exp_file, condition) {
  exp <- fromJSON(exp_file, simplifyVector = FALSE)
  response <- exp$response

  # Extract code blocks
  pattern <- "```([a-zA-Z]*)\\s*\\n([\\s\\S]*?)```"
  matches <- str_match_all(response, pattern)[[1]]

  if (nrow(matches) == 0) {
    return(list(code = "NO CODE BLOCKS FOUND", language = "unknown"))
  }

  # Determine expected language
  expected_lang <- switch(condition,
    "stan" = c("stan", "r"),
    "pymc" = c("python", "py"),
    "turing" = c("julia", "jl"),
    "epiaware" = c("julia", "jl"),
    "r" = c("r"),
    c("r", "python", "julia", "stan")
  )

  # Collect relevant code blocks
  code_parts <- list()
  for (i in seq_len(nrow(matches))) {
    lang <- tolower(matches[i, 2])
    code <- matches[i, 3]

    # Normalize language
    if (lang == "py") lang <- "python"
    if (lang == "jl") lang <- "julia"
    if (lang == "") {
      lang <- switch(condition,
        "pymc" = "python",
        "turing" = "julia",
        "epiaware" = "julia",
        "stan" = "r",
        "r" = "r",
        "r"
      )
    }

    code_parts[[length(code_parts) + 1]] <- list(language = lang, code = code)
  }

  # For Stan condition, separate Stan and R code
  if (condition == "stan") {
    stan_code <- paste(sapply(code_parts, function(x) if (x$language == "stan") x$code else ""), collapse = "\n\n")
    r_code <- paste(sapply(code_parts, function(x) if (x$language == "r") x$code else ""), collapse = "\n\n")
    return(list(
      stan_code = trimws(stan_code),
      r_code = trimws(r_code),
      language = "stan+r"
    ))
  }

  # For other conditions, combine all relevant code
  primary_lang <- switch(condition,
    "pymc" = "python",
    "turing" = "julia",
    "epiaware" = "julia",
    "r" = "r",
    "r"
  )

  code <- paste(sapply(code_parts, function(x) {
    if (x$language == primary_lang) x$code else ""
  }), collapse = "\n\n")

  list(code = trimws(code), language = primary_lang)
}

# Load execution results
load_results <- function(scenario, condition, llm, run_id) {
  result_dir <- file.path(RESULTS_DIR, paste0("scenario_", scenario), condition, llm, sprintf("run_%02d", run_id))
  result_file <- file.path(result_dir, "result.json")

  if (file.exists(result_file)) {
    result <- fromJSON(result_file)
    return(list(
      executed = TRUE,
      success = isTRUE(result$tier1_success),
      exit_code = result$tier1_exit_code,
      timed_out = isTRUE(result$tier1_timed_out),
      duration = result$tier1_duration
    ))
  }

  list(executed = FALSE, success = NA, exit_code = NA, timed_out = NA, duration = NA)
}

# Main function
generate_review_materials <- function() {
  message("Generating expert review materials...")

  # Get all experiments
  experiments <- get_all_experiments()
  message(sprintf("Found %d experiments", length(experiments)))

  # Generate random IDs (shuffled)
  n <- length(experiments)
  random_ids <- sprintf("SUB_%03d", sample(1:n))

  # Create mapping (for coordinator use only)
  mapping <- data.frame(
    submission_id = random_ids,
    scenario = sapply(experiments, function(x) x$scenario),
    condition = sapply(experiments, function(x) x$condition),
    llm = sapply(experiments, function(x) x$llm),
    run_id = sapply(experiments, function(x) x$run_id),
    stringsAsFactors = FALSE
  )

  # Add execution results to mapping
  for (i in seq_len(nrow(mapping))) {
    result <- load_results(mapping$scenario[i], mapping$condition[i],
                          mapping$llm[i], mapping$run_id[i])
    mapping$executed[i] <- result$executed
    mapping$success[i] <- result$success
    mapping$timed_out[i] <- result$timed_out
  }

  # Save mapping (CONFIDENTIAL - for coordinators only)
  write_csv(mapping, file.path(PROJECT_DIR, "expert_review", "CONFIDENTIAL_mapping.csv"))
  message("Saved confidential mapping to expert_review/CONFIDENTIAL_mapping.csv")

  # Generate individual submission files
  for (i in seq_len(n)) {
    exp <- experiments[[i]]
    sub_id <- random_ids[i]

    # Extract code
    code_data <- extract_code(exp$exp_file, exp$condition)

    # Get execution status
    result <- load_results(exp$scenario, exp$condition, exp$llm, exp$run_id)

    # Create submission file
    sub_dir <- file.path(OUTPUT_DIR, sub_id)
    dir.create(sub_dir, showWarnings = FALSE)

    # Write metadata (blinded)
    metadata <- list(
      submission_id = sub_id,
      scenario = exp$scenario,
      execution_status = if (result$success) "SUCCESS" else if (result$timed_out) "TIMEOUT" else "FAILED"
    )
    write_json(metadata, file.path(sub_dir, "metadata.json"), auto_unbox = TRUE, pretty = TRUE)

    # Write code file(s)
    if (exp$condition == "stan") {
      if (nchar(code_data$stan_code) > 0) {
        writeLines(code_data$stan_code, file.path(sub_dir, "model.stan"))
      }
      if (nchar(code_data$r_code) > 0) {
        writeLines(code_data$r_code, file.path(sub_dir, "script.R"))
      }
    } else {
      ext <- switch(code_data$language,
        "python" = "py",
        "julia" = "jl",
        "r" = "R",
        "txt"
      )
      writeLines(code_data$code, file.path(sub_dir, paste0("script.", ext)))
    }

    # Write review form template
    form <- sprintf('# Review Form: %s

## Scenario: %s

## Execution Status: %s

---

## 1. METHOD IDENTIFICATION (Scenario 1a only)

Method used:
- [ ] Renewal equation / Cori / EpiEstim
- [ ] Wallinga-Teunis
- [ ] Bettencourt-Ribeiro / SIR-based
- [ ] Naive ratio
- [ ] Other: _________________

---

## 2. DEPARTURES FROM REFERENCE

| # | Description | Category |
|---|-------------|----------|
| 1 |             | A/B/C/D  |
| 2 |             | A/B/C/D  |
| 3 |             | A/B/C/D  |

---

## 3. DEPARTURE COUNTS

- A (Equivalent alternative): ___
- B (Minor error): ___
- C (Major error): ___
- D (Fundamental misunderstanding): ___

---

## 4. OVERALL ASSESSMENT

- [ ] Acceptable
- [ ] Minor issues
- [ ] Major issues
- [ ] Incorrect

---

## 5. ADDITIONAL CRITERIA

- Uncertainty quantification provided: [ ] Yes [ ] No
- Appropriate epidemiological parameters: [ ] Yes [ ] No [ ] N/A
- Proper discretization handling: [ ] Yes [ ] No [ ] N/A

---

## 6. NOTES

(Any additional comments)

---

Reviewer: _________________
Date: _________________
', sub_id, exp$scenario, metadata$execution_status)

    writeLines(form, file.path(sub_dir, "review_form.md"))
  }

  message(sprintf("Generated %d submission packages in expert_review/submissions/", n))

  # Create summary by scenario for easier review batching
  summary_by_scenario <- mapping %>%
    group_by(scenario) %>%
    summarise(
      n = n(),
      n_success = sum(success, na.rm = TRUE),
      submissions = paste(submission_id, collapse = ", ")
    )

  write_csv(summary_by_scenario, file.path(PROJECT_DIR, "expert_review", "submissions_by_scenario.csv"))

  # Generate two files: code and scoresheet (for side-by-side review)
  message("\nGenerating consolidated review files...")

  # File 1: All code submissions
  code_content <- c(
    "# Expert Review: Code Submissions",
    "",
    "Use alongside scoresheet.md for side-by-side review.",
    "See instructions.md for review guidelines.",
    "",
    "---",
    ""
  )

  # File 2: Scoresheet only
  score_content <- c(
    "# Expert Review: Scoresheet",
    "",
    "Fill in scores while viewing code in all_code.md (same order).",
    "See instructions.md for review guidelines.",
    "",
    "---",
    ""
  )

  for (scenario_num in c("1a", "1b", "2", "3")) {
    scenario_subs <- mapping %>% filter(scenario == scenario_num)
    if (nrow(scenario_subs) == 0) next

    code_content <- c(code_content,
      sprintf("# Scenario %s", scenario_num),
      "",
      sprintf("Total submissions: %d", nrow(scenario_subs)),
      "",
      "---",
      ""
    )

    score_content <- c(score_content,
      sprintf("# Scenario %s", scenario_num),
      "",
      "---",
      ""
    )

    for (i in seq_len(nrow(scenario_subs))) {
      sub_id <- scenario_subs$submission_id[i]
      sub_dir <- file.path(OUTPUT_DIR, sub_id)

      # Get execution status
      result <- load_results(scenario_subs$scenario[i], scenario_subs$condition[i],
                            scenario_subs$llm[i], scenario_subs$run_id[i])
      exec_status <- if (isTRUE(result$success)) "SUCCESS" else if (isTRUE(result$timed_out)) "TIMEOUT" else "FAILED"

      # Code file entry
      code_content <- c(code_content,
        sprintf("## %s", sub_id),
        "",
        sprintf("**Scenario**: %s | **Execution**: %s", scenario_num, exec_status),
        ""
      )

      # Add code files
      code_files <- list.files(sub_dir, pattern = "\\.(R|py|jl|stan)$", full.names = TRUE)
      for (code_file in code_files) {
        ext <- tools::file_ext(code_file)
        lang <- switch(ext,
          "R" = "r",
          "py" = "python",
          "jl" = "julia",
          "stan" = "stan",
          ""
        )
        file_content <- readLines(code_file, warn = FALSE)

        code_content <- c(code_content,
          sprintf("### %s", basename(code_file)),
          "",
          sprintf("```%s", lang),
          file_content,
          "```",
          ""
        )
      }

      code_content <- c(code_content, "---", "")

      # Scoresheet entry
      score_content <- c(score_content,
        sprintf("## %s", sub_id),
        "",
        sprintf("**Scenario**: %s | **Execution**: %s", scenario_num, exec_status),
        "",
        "| Field | Value |",
        "|-------|-------|",
        "| Method (1a only) | |",
        "| Departures | |",
        "| Count A (Equivalent) | |",
        "| Count B (Minor) | |",
        "| Count C (Major) | |",
        "| Count D (Fundamental) | |",
        "| Overall | Acceptable / Minor / Major / Incorrect |",
        "| Uncertainty quantified | Yes / No |",
        "| Appropriate parameters | Yes / No / N/A |",
        "| Proper discretization | Yes / No / N/A |",
        "| Notes | |",
        "",
        "---",
        ""
      )
    }
  }

  writeLines(code_content, file.path(PROJECT_DIR, "expert_review", "all_code.md"))
  writeLines(score_content, file.path(PROJECT_DIR, "expert_review", "scoresheet.md"))
  message("Generated: expert_review/all_code.md and expert_review/scoresheet.md")

  message("\nSummary by scenario:")
  print(summary_by_scenario)

  message("\nDone! Materials ready for expert review.")
  message("  - Instructions: expert_review/instructions.md")
  message("  - Code: expert_review/all_code.md")
  message("  - Scoresheet: expert_review/scoresheet.md")
  message("  - Individual submissions: expert_review/submissions/SUB_XXX/")
  message("  - Mapping (confidential): expert_review/CONFIDENTIAL_mapping.csv")
}

# Run if called directly
if (!interactive()) {
  generate_review_materials()
}
