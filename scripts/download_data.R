# Download UK COVID-19 data for the LLM epidemic model composition study
#
# Data source: UKHSA COVID-19 Archive
# https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download
#
# This script downloads and processes UK COVID-19 data including:
# - Daily case counts
# - Hospital admissions
# - Deaths

library(httr)
library(jsonlite)
library(dplyr)
library(readr)
library(lubridate)

# Output directory
data_dir <- here::here("data")
dir.create(data_dir, showWarnings = FALSE, recursive = TRUE)

# ---- Download from legacy API ----
# The legacy API at coronavirus.data.gov.uk can still be used for historical data

base_url <- "https://api.coronavirus.data.gov.uk/v1/data"

# Function to download data from the API
download_covid_data <- function(metrics, area_type = "overview") {
  # Build structure parameter
  structure <- c(
    list(date = "date", areaName = "areaName"),
    setNames(as.list(metrics), metrics)
  )

  params <- list(
    filters = paste0("areaType=", area_type),
    structure = toJSON(structure, auto_unbox = TRUE),
    format = "csv"
  )

  response <- GET(base_url, query = params)


  if (status_code(response) == 200) {
    content(response, as = "text", encoding = "UTF-8") |>
      read_csv(show_col_types = FALSE)
  } else {
    stop("API request failed with status: ", status_code(response))
  }
}

# ---- Download cases ----
message("Downloading case data...")
cases_metrics <- c(
  "newCasesBySpecimenDate",
  "cumCasesBySpecimenDate"
)

cases_raw <- tryCatch(

download_covid_data(cases_metrics),
  error = function(e) {
    message("API download failed: ", e$message)
    message("Please download data manually from:")
    message("https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download")
    NULL
  }
)

# ---- Download hospitalisations ----
message("Downloading hospitalisation data...")
hosp_metrics <- c(
  "newAdmissions",
  "cumAdmissions"
)

hosp_raw <- tryCatch(
  download_covid_data(hosp_metrics, area_type = "nation"),
  error = function(e) {
    message("Hospitalisation data download failed: ", e$message)
    NULL
  }
)

# ---- Download deaths ----
message("Downloading death data...")
deaths_metrics <- c(
  "newDeaths28DaysByDeathDate",
  "cumDeaths28DaysByDeathDate"
)

deaths_raw <- tryCatch(
  download_covid_data(deaths_metrics),
  error = function(e) {
    message("Death data download failed: ", e$message)
    NULL
  }
)

# ---- Process and combine data ----
if (!is.null(cases_raw) && !is.null(deaths_raw)) {
  # Select a reasonable time period (e.g., 2020-2021 wave)
  start_date <- as.Date("2020-09-01")
  end_date <- as.Date("2021-03-31")

  # Process cases
  cases <- cases_raw |>
    filter(date >= start_date, date <= end_date) |>
    transmute(
      date = as.Date(date),
      cases = newCasesBySpecimenDate
    ) |>
    arrange(date) |>
    mutate(day_of_week = wday(date, week_start = 1))

  # Process deaths
  deaths <- deaths_raw |>
    filter(date >= start_date, date <= end_date) |>
    transmute(
      date = as.Date(date),
      deaths = newDeaths28DaysByDeathDate
    ) |>
    arrange(date)

  # Combine (hospitalisations may need separate handling due to nation-level data)
  combined <- cases |>
    left_join(deaths, by = "date")

  # Save individual files for scenarios
  write_csv(cases, file.path(data_dir, "cases.csv"))
  message("Saved: ", file.path(data_dir, "cases.csv"))

  if (!is.null(hosp_raw)) {
    hosp <- hosp_raw |>
      filter(areaName == "England") |>
      filter(date >= start_date, date <= end_date) |>
      transmute(
        date = as.Date(date),
        hospitalisations = newAdmissions
      ) |>
      arrange(date)

    observations <- combined |>
      left_join(hosp, by = "date")

    write_csv(observations, file.path(data_dir, "observations.csv"))
    message("Saved: ", file.path(data_dir, "observations.csv"))
  }

  message("\nData download complete!")
  message("Date range: ", start_date, " to ", end_date)
  message("Number of days: ", nrow(cases))

} else {
  message("\n--- Manual download required ---")
  message("1. Go to: https://ukhsa-dashboard.data.gov.uk/covid-19-archive-data-download")
  message("2. Download 'Cases metrics data', 'Deaths metrics data', and 'Healthcare metrics data'")
  message("3. Extract and place CSV files in: ", data_dir)
}
