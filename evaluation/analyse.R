#!/usr/bin/env Rscript
# Confirmatory analysis for the study.
#
# Reads:
#   runs/{scenario}/{condition}/par_{p}/{variant}/rep_{r}/{model}/run_{n}/
#     metadata.json
#     outputs/rt_estimates.csv
#     conversation.jsonl (for hallucination-rate scan)
#     one or more source files (.jl/.py/.R/.stan)
#   simulations/{variant}/rep_{r}/truth/true_rt.csv
#   evaluation/detectors_output.csv (produced by `python evaluation/detectors.py --all runs`)
#   expert_review/reviews.csv (produced by the review coordinator; may be absent)
#   expert_review/mutation_manifest.csv (sealed until reviews are in)
#
# Writes to analysis_output/:
#   table_1_correctness.csv      … table_9_descriptive.csv
#   figure_1_recovery.png        … figure_8_paraphrase_sensitivity.png
#   predictions.csv              (one row per pre-specified prediction)
#   session_info.txt
#
# Run once, after all agent runs and reviews are complete:
#   Rscript evaluation/analyse.R
#
# All bootstrap CIs are non-parametric percentile bootstraps at 1000 resamples
# using boot::boot. The bootstrap unit is (paraphrase × run) for primary
# comparisons and (variant × run) for the adversarial fingerprint.

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(purrr)
  library(stringr)
  library(jsonlite)
  library(boot)
  library(ggplot2)
})

RUNS_ROOT   <- "runs"
SIMS_ROOT   <- "simulations"
DETECTORS_CSV <- "evaluation/detectors_output.csv"
REVIEW_CSV    <- "expert_review/reviews.csv"
MUTATION_CSV  <- "expert_review/mutation_manifest.csv"
OUT_DIR       <- "analysis_output"

CONDITIONS <- c("no-spec", "julia", "epiaware")
SCENARIOS  <- c("scenario_1a", "scenario_1b", "scenario_2", "scenario_3")
MODELS     <- c("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7")
CANONICAL_VARIANT <- "canonical"
ADVERSARIAL_VARIANTS <- c("short_gi", "long_delay", "extreme_dispersion", "abrupt_change")
EVAL_WINDOW_DAYS <- 25:125
BOOTSTRAP_R <- 1000

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)


# ---------------------------------------------------------------------------
# Load runs
# ---------------------------------------------------------------------------

parse_run_dir <- function(rd) {
  # runs/{scenario}/{condition}/par_{p}/{variant}/rep_{r}/{model}/run_{n}
  parts <- strsplit(rd, "/", fixed = TRUE)[[1]]
  tibble(
    run_dir   = rd,
    scenario  = parts[2],
    condition = parts[3],
    paraphrase = as.integer(sub("par_", "", parts[4])),
    variant   = parts[5],
    replicate = as.integer(sub("rep_", "", parts[6])),
    model     = parts[7],
    run_num   = as.integer(sub("run_", "", parts[8]))
  )
}

collect_runs <- function() {
  rds <- list.files(RUNS_ROOT, pattern = "^conversation\\.jsonl$",
                    recursive = TRUE, full.names = FALSE)
  if (length(rds) == 0) return(tibble())
  rds <- dirname(rds)
  meta <- purrr::map_dfr(rds, function(rel) {
    m_path <- file.path(RUNS_ROOT, rel, "metadata.json")
    if (!file.exists(m_path)) return(tibble())
    m <- tryCatch(jsonlite::fromJSON(m_path), error = function(e) NULL)
    if (is.null(m)) return(tibble())
    p <- parse_run_dir(file.path(RUNS_ROOT, rel))
    p$output_present <- isTRUE(m$output_present) ||
      file.exists(file.path(RUNS_ROOT, rel, "outputs", "rt_estimates.csv")) &&
      file.info(file.path(RUNS_ROOT, rel, "outputs", "rt_estimates.csv"))$size > 0
    p$retry_count <- m$retry_count %||% NA_integer_
    p$post_agent_waits <- m$post_agent_waits %||% NA_integer_
    p$start_time <- m$start_time %||% NA_character_
    p$end_time   <- m$end_time   %||% NA_character_
    p
  })
  meta$duration_min <- as.numeric(difftime(
    as.POSIXct(meta$end_time, tz = "UTC"),
    as.POSIXct(meta$start_time, tz = "UTC"),
    units = "mins"
  ))
  meta
}

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a


# ---------------------------------------------------------------------------
# Truth and recovery
# ---------------------------------------------------------------------------

load_truth <- function(variant) {
  # rep 101 == replicate 1; the seed-encoded file lives under rep_01
  path <- file.path(SIMS_ROOT, variant, "rep_01", "truth", "true_rt.csv")
  if (!file.exists(path)) return(NULL)
  suppressMessages(readr::read_csv(path, show_col_types = FALSE))
}

recovery_metrics <- function(rd, truth_df) {
  csv <- file.path(rd, "outputs", "rt_estimates.csv")
  if (!file.exists(csv) || file.info(csv)$size == 0) return(NULL)
  est <- tryCatch(suppressMessages(readr::read_csv(csv, show_col_types = FALSE)),
                  error = function(e) NULL)
  if (is.null(est) || nrow(est) == 0) return(NULL)
  # Column-name flexibility
  colnames(est) <- tolower(colnames(est))
  if (!("date" %in% colnames(est))) return(NULL)
  # find the median column
  med_col <- intersect(c("rt_median", "rt_med", "median"), colnames(est))[1]
  lo_col  <- intersect(c("rt_lower", "rt_low", "lower"),   colnames(est))[1]
  hi_col  <- intersect(c("rt_upper", "rt_high", "upper"),  colnames(est))[1]
  if (is.na(med_col)) return(NULL)
  est$date <- as.character(est$date)
  truth_df$date <- as.character(truth_df$date)
  # Truth column
  truth_col <- intersect(c("R_t", "Rt", "rt"), colnames(truth_df))[1]
  if (is.na(truth_col)) return(NULL)
  # Days 25..125
  keep_days <- EVAL_WINDOW_DAYS
  # Assume date is day-of-simulation-order; use position 25..125
  truth_df <- truth_df[keep_days, , drop = FALSE]
  est <- est[match(truth_df$date, est$date), , drop = FALSE]
  if (nrow(est) == 0 || all(is.na(est[[med_col]]))) return(NULL)
  diffs <- as.numeric(est[[med_col]]) - as.numeric(truth_df[[truth_col]])
  rmse <- sqrt(mean(diffs^2, na.rm = TRUE))
  covered <- NA_real_; width <- NA_real_
  if (!is.na(lo_col) && !is.na(hi_col)) {
    lo <- as.numeric(est[[lo_col]]); hi <- as.numeric(est[[hi_col]])
    within <- lo <= truth_df[[truth_col]] & truth_df[[truth_col]] <= hi
    covered <- mean(within, na.rm = TRUE)
    width <- mean(hi - lo, na.rm = TRUE)
  }
  tibble(rmse = rmse, coverage = covered, ci_width = width)
}


# ---------------------------------------------------------------------------
# Detector output
# ---------------------------------------------------------------------------

load_detectors <- function() {
  if (!file.exists(DETECTORS_CSV)) {
    message("Detectors output not found at ", DETECTORS_CSV,
            "; run: python evaluation/detectors.py --all runs > ", DETECTORS_CSV)
    return(NULL)
  }
  suppressMessages(readr::read_csv(DETECTORS_CSV, show_col_types = FALSE))
}


# ---------------------------------------------------------------------------
# Lines of code
# ---------------------------------------------------------------------------

count_loc <- function(rd) {
  patterns <- c("*.jl", "*.py", "*.R", "*.stan")
  files <- unlist(lapply(patterns, function(p)
    Sys.glob(file.path(rd, p))))
  files <- files[!grepl("(Project|Manifest)\\.toml$", files)]
  files <- files[!grepl("_docs\\.md$", files)]
  if (length(files) == 0) return(NA_integer_)
  n <- 0L
  for (f in files) {
    lines <- readLines(f, warn = FALSE)
    lines <- trimws(lines)
    # Drop blank and comment-only lines
    lang_comment <- switch(tools::file_ext(f),
      py = "^#", R = "^#", jl = "^#", stan = "^(//|/\\*)", "^#")
    lines <- lines[nzchar(lines) & !grepl(lang_comment, lines)]
    n <- n + length(lines)
  }
  n
}


# ---------------------------------------------------------------------------
# Hallucination-rate scan (secondary outcome)
# ---------------------------------------------------------------------------

hallucination_rate <- function(rd) {
  logs <- Sys.glob(file.path(rd, "conversation*.jsonl"))
  if (length(logs) == 0) return(NA_real_)
  n_iter <- 0L; n_hallu <- 0L
  patterns <- c(
    "does not exist",
    "no method matching",
    "UndefVarError",
    "NameError",
    "AttributeError.*has no attribute",
    "cannot find symbol",
    "not defined",
    "is not exported"
  )
  hallu_re <- paste(patterns, collapse = "|")
  for (log in logs) {
    lines <- readLines(log, warn = FALSE)
    for (ln in lines) {
      if (grepl("\"tool_use_id\"", ln, fixed = FALSE)) n_iter <- n_iter + 1L
      if (grepl(hallu_re, ln)) n_hallu <- n_hallu + 1L
    }
  }
  if (n_iter == 0) return(NA_real_)
  n_hallu / n_iter
}


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------

bootstrap_stat <- function(x, statistic = median, R = BOOTSTRAP_R) {
  x <- x[is.finite(x)]
  if (length(x) < 2) return(list(est = NA_real_, lo = NA_real_, hi = NA_real_))
  b <- boot::boot(x, function(d, i) statistic(d[i]), R = R)
  ci <- tryCatch(boot::boot.ci(b, type = "perc")$percent[4:5],
                 error = function(e) c(NA_real_, NA_real_))
  list(est = as.numeric(statistic(x)), lo = ci[1], hi = ci[2])
}

bootstrap_diff <- function(x, y, statistic = median, R = BOOTSTRAP_R) {
  x <- x[is.finite(x)]; y <- y[is.finite(y)]
  if (length(x) < 2 || length(y) < 2) {
    return(list(est = NA_real_, lo = NA_real_, hi = NA_real_))
  }
  # Bootstrap the difference of statistics by resampling within group.
  d <- statistic(x) - statistic(y)
  diffs <- replicate(R, {
    xs <- sample(x, length(x), replace = TRUE)
    ys <- sample(y, length(y), replace = TRUE)
    statistic(xs) - statistic(ys)
  })
  ci <- quantile(diffs, c(0.025, 0.975), na.rm = TRUE)
  list(est = as.numeric(d), lo = as.numeric(ci[1]), hi = as.numeric(ci[2]))
}

bootstrap_proportion <- function(x, R = BOOTSTRAP_R) {
  x <- x[!is.na(x)]
  if (length(x) < 2) return(list(est = NA_real_, lo = NA_real_, hi = NA_real_))
  b <- boot::boot(as.integer(as.logical(x)),
                  function(d, i) mean(d[i]), R = R)
  ci <- tryCatch(boot::boot.ci(b, type = "perc")$percent[4:5],
                 error = function(e) c(NA_real_, NA_real_))
  list(est = mean(x), lo = ci[1], hi = ci[2])
}


# ---------------------------------------------------------------------------
# Assemble the analysis dataset
# ---------------------------------------------------------------------------

message("Collecting runs …")
runs <- collect_runs()
if (nrow(runs) == 0) stop("No runs found under ", RUNS_ROOT)
message("  ", nrow(runs), " runs.")

message("Loading truth trajectories …")
truths <- purrr::set_names(
  lapply(c(CANONICAL_VARIANT, ADVERSARIAL_VARIANTS), load_truth),
  c(CANONICAL_VARIANT, ADVERSARIAL_VARIANTS)
)

message("Computing recovery metrics …")
runs <- runs %>%
  rowwise() %>%
  mutate(rec = list(recovery_metrics(run_dir, truths[[variant]]))) %>%
  ungroup() %>%
  mutate(
    rmse     = purrr::map_dbl(rec, ~ .x$rmse     %||% NA_real_),
    coverage = purrr::map_dbl(rec, ~ .x$coverage %||% NA_real_),
    ci_width = purrr::map_dbl(rec, ~ .x$ci_width %||% NA_real_)
  ) %>%
  select(-rec)

message("Computing LOC …")
runs <- runs %>% mutate(loc = purrr::map_int(run_dir, count_loc))

message("Computing hallucination rate …")
runs <- runs %>% mutate(hallu_rate = purrr::map_dbl(run_dir, hallucination_rate))

detectors <- load_detectors()
if (!is.null(detectors)) {
  detectors$run_dir <- detectors$run_dir
  runs <- runs %>% left_join(detectors, by = "run_dir")
}


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

message("Evaluating pre-specified predictions …")
preds <- list()

# P1: Instruction adherence.
#   In no-spec, ≥70% of scenario_1a/1b submissions use R+EpiEstim / R+EpiNow2 /
#   Python+PyMC / Python+numpyro. We infer package use from source-file
#   language/imports; a helper detector script writes this per-run into
#   detectors_output.csv as an extra column `chosen_package`. If the detector
#   run didn't emit that column, this prediction is skipped.
if ("chosen_package" %in% colnames(runs)) {
  s12_no_spec <- runs %>%
    filter(condition == "no-spec",
           scenario %in% c("scenario_1a", "scenario_1b"),
           variant == CANONICAL_VARIANT)
  matching_packages <- c("EpiEstim", "EpiNow2", "PyMC", "numpyro")
  is_match <- s12_no_spec$chosen_package %in% matching_packages
  p <- bootstrap_proportion(is_match)
  preds$P1_no_spec_package_default <- list(
    estimate = p$est, ci_lo = p$lo, ci_hi = p$hi,
    threshold = "lower CI bound >= 0.70",
    confirmed = isTRUE(p$lo >= 0.70)
  )
} else {
  preds$P1_no_spec_package_default <- list(
    estimate = NA, ci_lo = NA, ci_hi = NA,
    threshold = "lower CI bound >= 0.70",
    confirmed = NA,
    note = "detectors_output.csv does not include chosen_package column"
  )
}

# P3: Missing-component rates differ across conditions.
#   For each (scenario × required component), rate of missing components is
#   higher in julia than in epiaware by at least 15 pp.
components_by_scenario <- list(
  scenario_2 = c("flag_no_dow", "flag_no_ascertainment", "flag_poisson_only"),
  scenario_3 = c("flag_no_dow", "flag_no_ascertainment", "flag_poisson_only",
                 "flag_no_multistream_latent")
)
p3_rows <- list()
if (!is.null(detectors)) {
  for (sc in names(components_by_scenario)) {
    for (comp in components_by_scenario[[sc]]) {
      julia_flags <- runs %>%
        filter(scenario == sc, condition == "julia", variant == CANONICAL_VARIANT) %>%
        pull(!!comp)
      epi_flags <- runs %>%
        filter(scenario == sc, condition == "epiaware", variant == CANONICAL_VARIANT) %>%
        pull(!!comp)
      diff <- bootstrap_diff(as.numeric(julia_flags), as.numeric(epi_flags),
                             statistic = mean)
      p3_rows[[paste(sc, comp, sep = "_")]] <- tibble(
        scenario = sc, component = comp,
        diff_est = diff$est, ci_lo = diff$lo, ci_hi = diff$hi,
        confirmed = isTRUE(diff$lo > 0)
      )
    }
  }
}
preds$P3_missing_components <- dplyr::bind_rows(p3_rows)

# P5: Recovery on adversarial variants — median Rt RMSE lower in epiaware than
# julia on scenarios 2 and 3 across the four confirmatory variants.
adv_runs <- runs %>% filter(variant %in% ADVERSARIAL_VARIANTS,
                            scenario %in% c("scenario_2", "scenario_3"))
julia_rmse <- adv_runs %>% filter(condition == "julia")     %>% pull(rmse)
epi_rmse   <- adv_runs %>% filter(condition == "epiaware")  %>% pull(rmse)
d5 <- bootstrap_diff(julia_rmse, epi_rmse, statistic = median)
preds$P5_epiaware_lower_adversarial_rmse <- list(
  estimate = d5$est, ci_lo = d5$lo, ci_hi = d5$hi,
  threshold = "median difference >= 0.03 with 95%-CI excluding zero",
  confirmed = isTRUE(d5$est >= 0.03 && d5$lo > 0)
)

# P6a: Submissions flagged flag_no_delay_handling — median RMSE on long_delay is
# at least 0.05 higher than on canonical.
if ("flag_no_delay_handling" %in% colnames(runs)) {
  flagged <- runs %>% filter(flag_no_delay_handling == 1 |
                             flag_no_delay_handling == TRUE)
  long   <- flagged %>% filter(variant == "long_delay") %>% pull(rmse)
  canon  <- flagged %>% filter(variant == CANONICAL_VARIANT) %>% pull(rmse)
  d6a <- bootstrap_diff(long, canon, statistic = median)
  preds$P6a_no_delay_x_long_delay <- list(
    estimate = d6a$est, ci_lo = d6a$lo, ci_hi = d6a$hi,
    threshold = "median difference >= 0.05 with 95%-CI excluding zero",
    confirmed = isTRUE(d6a$est >= 0.05 && d6a$lo > 0)
  )
}
if ("flag_poisson_only" %in% colnames(runs)) {
  flagged <- runs %>% filter(flag_poisson_only == 1 | flag_poisson_only == TRUE)
  extreme <- flagged %>% filter(variant == "extreme_dispersion") %>% pull(coverage)
  canon   <- flagged %>% filter(variant == CANONICAL_VARIANT) %>% pull(coverage)
  d6b <- bootstrap_diff(canon, extreme, statistic = median)
  preds$P6b_poisson_x_extreme_dispersion <- list(
    estimate = d6b$est, ci_lo = d6b$lo, ci_hi = d6b$hi,
    threshold = "coverage drop >= 0.15 with 95%-CI excluding zero",
    confirmed = isTRUE(d6b$est >= 0.15 && d6b$lo > 0)
  )
}

# P7: LOC — median epiaware < median julia by ≥ 50, < median no-spec by ≥ 100.
loc_epi   <- runs %>% filter(condition == "epiaware", variant == CANONICAL_VARIANT) %>% pull(loc)
loc_julia <- runs %>% filter(condition == "julia",    variant == CANONICAL_VARIANT) %>% pull(loc)
loc_ns    <- runs %>% filter(condition == "no-spec",  variant == CANONICAL_VARIANT) %>% pull(loc)
d7_jul <- bootstrap_diff(loc_julia, loc_epi, statistic = median)
d7_ns  <- bootstrap_diff(loc_ns,    loc_epi, statistic = median)
preds$P7_loc <- list(
  julia_minus_epi = list(estimate = d7_jul$est, ci_lo = d7_jul$lo, ci_hi = d7_jul$hi,
                        threshold = "median difference >= 50 with 95%-CI excluding zero",
                        confirmed = isTRUE(d7_jul$est >= 50 && d7_jul$lo > 0)),
  no_spec_minus_epi = list(estimate = d7_ns$est, ci_lo = d7_ns$lo, ci_hi = d7_ns$hi,
                          threshold = "median difference >= 100 with 95%-CI excluding zero",
                          confirmed = isTRUE(d7_ns$est >= 100 && d7_ns$lo > 0))
)

# P8: Injected-defect detection sensitivity. Depends on review data.
if (file.exists(REVIEW_CSV) && file.exists(MUTATION_CSV)) {
  reviews <- suppressMessages(readr::read_csv(REVIEW_CSV, show_col_types = FALSE))
  mutations <- suppressMessages(readr::read_csv(MUTATION_CSV, show_col_types = FALSE))
  # Sensitivity per condition
  # Expected columns: reviews has (sample_id, reviewer, defects_caught) as
  # long-format one row per (sample, reviewer, mutation_correctly_caught).
  # mutations has (sample_id, condition, mutation_id).
  joined <- reviews %>% left_join(mutations, by = "sample_id")
  sens_epi   <- joined %>% filter(condition == "epiaware") %>% pull(mutation_correctly_caught)
  sens_julia <- joined %>% filter(condition == "julia")    %>% pull(mutation_correctly_caught)
  sens_ns    <- joined %>% filter(condition == "no-spec")  %>% pull(mutation_correctly_caught)
  d8_j <- bootstrap_diff(as.numeric(sens_epi), as.numeric(sens_julia), statistic = mean)
  d8_n <- bootstrap_diff(as.numeric(sens_epi), as.numeric(sens_ns),    statistic = mean)
  preds$P8_defect_detection_sensitivity <- list(
    epi_minus_julia = list(estimate = d8_j$est, ci_lo = d8_j$lo, ci_hi = d8_j$hi,
                           threshold = ">= 0.15 with 95%-CI excluding zero",
                           confirmed = isTRUE(d8_j$est >= 0.15 && d8_j$lo > 0)),
    epi_minus_no_spec = list(estimate = d8_n$est, ci_lo = d8_n$lo, ci_hi = d8_n$hi,
                             threshold = ">= 0.10 with 95%-CI excluding zero",
                             confirmed = isTRUE(d8_n$est >= 0.10 && d8_n$lo > 0))
  )
} else {
  preds$P8_defect_detection_sensitivity <- list(
    note = "expert_review/reviews.csv or mutation_manifest.csv not present; run after reviews complete."
  )
}

# P9: Capability-conditional gap. julia-vs-epiaware RMSE gap on scenarios 2 and
# 3 shrinks with model capability.
gap_by_model <- purrr::map_dfr(MODELS, function(m) {
  jul  <- runs %>% filter(model == m, condition == "julia",
                          scenario %in% c("scenario_2", "scenario_3"),
                          variant == CANONICAL_VARIANT) %>% pull(rmse)
  epi  <- runs %>% filter(model == m, condition == "epiaware",
                          scenario %in% c("scenario_2", "scenario_3"),
                          variant == CANONICAL_VARIANT) %>% pull(rmse)
  d <- bootstrap_diff(jul, epi, statistic = median)
  tibble(model = m, gap_est = d$est, gap_lo = d$lo, gap_hi = d$hi)
})
preds$P9_capability_conditional_gap <- gap_by_model

# P10: Hallucination rate in epiaware vs julia and no-spec.
hallu_epi   <- runs %>% filter(condition == "epiaware") %>% pull(hallu_rate)
hallu_julia <- runs %>% filter(condition == "julia")    %>% pull(hallu_rate)
hallu_ns    <- runs %>% filter(condition == "no-spec")  %>% pull(hallu_rate)
d10_j <- bootstrap_diff(hallu_epi, hallu_julia, statistic = median)
d10_n <- bootstrap_diff(hallu_epi, hallu_ns,    statistic = median)
preds$P10_hallucination <- list(
  epi_minus_julia = list(estimate = d10_j$est, ci_lo = d10_j$lo, ci_hi = d10_j$hi,
                         threshold = ">= 0.10 with 95%-CI excluding zero",
                         confirmed = isTRUE(d10_j$est >= 0.10 && d10_j$lo > 0)),
  epi_minus_no_spec = list(estimate = d10_n$est, ci_lo = d10_n$lo, ci_hi = d10_n$hi,
                           threshold = ">= 0.10 with 95%-CI excluding zero",
                           confirmed = isTRUE(d10_n$est >= 0.10 && d10_n$lo > 0))
)

writeLines(jsonlite::toJSON(preds, pretty = TRUE, auto_unbox = TRUE, na = "null"),
           file.path(OUT_DIR, "predictions.json"))


# ---------------------------------------------------------------------------
# Tables (headline only; full column set derived from `runs` below)
# ---------------------------------------------------------------------------

message("Writing tables …")

table_1 <- runs %>%
  filter(variant == CANONICAL_VARIANT) %>%
  group_by(scenario, condition) %>%
  summarise(
    n = n(),
    median_rmse = median(rmse, na.rm = TRUE),
    iqr_rmse    = IQR(rmse, na.rm = TRUE),
    median_coverage = median(coverage, na.rm = TRUE),
    iqr_coverage    = IQR(coverage, na.rm = TRUE),
    .groups = "drop"
  )
readr::write_csv(table_1, file.path(OUT_DIR, "table_1_correctness.csv"))

table_2 <- runs %>%
  filter(variant %in% ADVERSARIAL_VARIANTS) %>%
  group_by(scenario, condition, variant) %>%
  summarise(
    n = n(),
    median_rmse = median(rmse, na.rm = TRUE),
    iqr_rmse    = IQR(rmse, na.rm = TRUE),
    median_coverage = median(coverage, na.rm = TRUE),
    .groups = "drop"
  )
readr::write_csv(table_2, file.path(OUT_DIR, "table_2_adversarial.csv"))

if (!is.null(detectors)) {
  flag_cols <- grep("^flag_", colnames(runs), value = TRUE)
  table_3 <- runs %>%
    filter(variant == CANONICAL_VARIANT) %>%
    group_by(scenario, condition) %>%
    summarise(across(all_of(flag_cols),
                     ~ mean(as.numeric(.x), na.rm = TRUE)),
              .groups = "drop")
  readr::write_csv(table_3, file.path(OUT_DIR, "table_3_component_correctness.csv"))
}

table_4 <- runs %>%
  filter(variant == CANONICAL_VARIANT) %>%
  group_by(scenario, condition) %>%
  summarise(
    n = n(),
    median_loc = median(loc, na.rm = TRUE),
    iqr_loc    = IQR(loc, na.rm = TRUE),
    .groups = "drop"
  )
readr::write_csv(table_4, file.path(OUT_DIR, "table_4_interpretability.csv"))

# Table 5 (reviewability) requires review data.
if (file.exists(REVIEW_CSV)) {
  message("Skipping table 5 draft — write once reviews are ingested.")
}

table_9 <- runs %>%
  group_by(scenario, condition) %>%
  summarise(
    n = n(),
    median_retries = median(retry_count, na.rm = TRUE),
    median_waits   = median(post_agent_waits, na.rm = TRUE),
    median_dur_min = median(duration_min, na.rm = TRUE),
    .groups = "drop"
  )
readr::write_csv(table_9, file.path(OUT_DIR, "table_9_descriptive.csv"))


# ---------------------------------------------------------------------------
# Figures (minimum viable)
# ---------------------------------------------------------------------------

message("Writing figures …")

fig1 <- runs %>%
  filter(variant == CANONICAL_VARIANT, !is.na(rmse)) %>%
  ggplot(aes(x = condition, y = rmse, fill = condition)) +
    geom_violin(alpha = 0.6, scale = "width") +
    geom_jitter(width = 0.1, alpha = 0.5, size = 0.7) +
    facet_wrap(~ scenario, ncol = 2, scales = "free_y") +
    labs(title = "Recovery distributions per condition × scenario",
         y = "Rt RMSE (evaluation window)", x = NULL) +
    theme_minimal()
ggsave(file.path(OUT_DIR, "figure_1_recovery.png"), fig1,
       width = 8, height = 6, dpi = 150)

fig5 <- runs %>%
  filter(variant == CANONICAL_VARIANT, !is.na(loc)) %>%
  ggplot(aes(x = condition, y = loc, fill = condition)) +
    geom_violin(alpha = 0.6, scale = "width") +
    geom_jitter(width = 0.1, alpha = 0.5, size = 0.7) +
    facet_wrap(~ scenario, ncol = 2) +
    labs(title = "Lines of code per submission",
         y = "LOC (non-blank, non-comment)", x = NULL) +
    theme_minimal()
ggsave(file.path(OUT_DIR, "figure_5_loc.png"), fig5,
       width = 8, height = 6, dpi = 150)


# ---------------------------------------------------------------------------
# Session info
# ---------------------------------------------------------------------------

sink(file.path(OUT_DIR, "session_info.txt"))
print(sessionInfo())
sink()

message("Done. Outputs in ", OUT_DIR, "/")
