# Download UK COVID-19 data from UKHSA Dashboard API
# This script downloads cases, hospitalisations, and deaths data

library(httr2)
library(jsonlite)
library(dplyr)
library(readr)
library(lubridate)
library(tidyr)

# New UKHSA API endpoint
base_url <- "https://api.ukhsa-dashboard.data.gov.uk"

# Function to download data from UKHSA API (paginated)
download_ukhsa_metric <- function(metric, geography = "England", page_size = 365) {

  endpoint <- sprintf(
    "%s/themes/infectious_disease/sub_themes/respiratory/topics/COVID-19/geography_types/Nation/geographies/%s/metrics/%s",
    base_url, geography, metric
  )

  all_results <- list()
  page <- 1

  repeat {
    message(sprintf("  Fetching page %d...", page))

    resp <- request(endpoint) |>
      req_url_query(page_size = page_size, page = page) |>
      req_perform()

    data <- resp |>
      resp_body_json()

    if (length(data$results) == 0) break

    all_results <- c(all_results, data$results)

    if (is.null(data$`next`)) break
    page <- page + 1
  }

  # Convert to data frame
  df <- bind_rows(lapply(all_results, as.data.frame))
  return(df)
}

# Download cases (by specimen date)
message("Downloading case data...")
cases_raw <- download_ukhsa_metric("COVID-19_cases_casesByDay")
cases <- cases_raw |>
  select(date, cases = metric_value) |>
  mutate(date = as.Date(date)) |>
  arrange(date)

# Download hospital admissions
message("Downloading hospitalisation data...")
hosp_raw <- download_ukhsa_metric("COVID-19_healthcare_admissionByDay")
hospitalisations <- hosp_raw |>
  select(date, hospitalisations = metric_value) |>
  mutate(date = as.Date(date)) |>
  arrange(date)

# Download deaths (within 28 days of positive test)
message("Downloading death data...")
deaths_raw <- download_ukhsa_metric("COVID-19_deaths_ONSByDay")
deaths <- deaths_raw |>
  select(date, deaths = metric_value) |>
  mutate(date = as.Date(date)) |>
  arrange(date)

# Merge all data streams
message("Merging data...")
combined <- cases |>
  full_join(hospitalisations, by = "date") |>
  full_join(deaths, by = "date") |>
  arrange(date)

# Add day of week (for Scenario 2)
combined <- combined |>
  mutate(day_of_week = wday(date, week_start = 1))  # Monday = 1

# Select a reasonable date range (e.g., 2020-09-01 to 2021-03-31 - second wave)
# This gives a clear epidemic wave with Rt variation
date_start <- as.Date("2020-09-01")
date_end <- as.Date("2021-03-31")

combined_filtered <- combined |>
  filter(date >= date_start, date <= date_end) |>
  filter(!is.na(cases))

# Save combined data for Scenario 3
write_csv(combined_filtered, "observations.csv")
message("Saved: observations.csv")

# Save cases only for Scenarios 1a, 1b
cases_only <- combined_filtered |>
  select(date, cases)
write_csv(cases_only, "cases.csv")
message("Saved: cases.csv")

# Save cases with day of week for Scenario 2
cases_dow <- combined_filtered |>
  select(date, cases, day_of_week)
write_csv(cases_dow, "cases_dow.csv")
message("Saved: cases_dow.csv")

# Summary
message("\nData summary:")
message(sprintf("Date range: %s to %s", min(combined_filtered$date), max(combined_filtered$date)))
message(sprintf("Number of days: %d", nrow(combined_filtered)))
message(sprintf("Cases: %d total, range %d-%d daily",
                sum(combined_filtered$cases, na.rm = TRUE),
                min(combined_filtered$cases, na.rm = TRUE),
                max(combined_filtered$cases, na.rm = TRUE)))
message(sprintf("Hospitalisations: %d total", sum(combined_filtered$hospitalisations, na.rm = TRUE)))
message(sprintf("Deaths: %d total", sum(combined_filtered$deaths, na.rm = TRUE)))
