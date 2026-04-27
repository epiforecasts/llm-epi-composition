# Estimate a shared time-varying reproduction number from multiple data streams

You are analysing daily counts from an infectious disease outbreak. Three different observation streams have been recorded: cases, hospitalisations, and deaths. All three are independent observations of the same underlying epidemic. Your task is to estimate a single time-varying effective reproduction number $R_t$ that is consistent with all three streams.

## Data

The observation period is 150 days, dated 2023-01-01 to 2023-05-30. The data are in:

- `data/cases.csv` — columns `date`, `cases`.
- `data/hospitalisations.csv` — columns `date`, `hospitalisations`.
- `data/deaths.csv` — columns `date`, `deaths`.

## Epidemiological parameters

These distributions come from external studies of the pathogen. Treat them as fixed knowns; no need to propagate uncertainty.

- Generation interval: Gamma distribution, mean 5.5 days, standard deviation 2 days.
- Reporting delay from infection to observation, log-normal:
  - Cases: mean 5 days, standard deviation 2 days.
  - Hospitalisations: mean 10 days, standard deviation 3 days.
  - Deaths: mean 20 days, standard deviation 5 days.

## Structural features of the data

- All three series are independent observations of the same underlying epidemic and share a single $R_t$ trajectory.
- Each stream has its own reporting delay (above) and its own time-varying ascertainment (the proportion of infections eventually reported in that stream). Ascertainment trajectories differ across streams and are unknown.
- Counts show a weekly reporting cycle.
- Observation noise is overdispersed relative to a homogeneous Poisson process.

## Task

Produce daily point estimates of a single $R_t$ trajectory consistent with all three streams. Use the renewal equation for the infection process. Account explicitly for stream-specific delays, time-varying stream-specific ascertainment, the day-of-week cycle, and observation overdispersion.

You may use any language or framework.

Save your final $R_t$ estimates to `outputs/rt_estimates.csv` with columns:

- `date` (YYYY-MM-DD)
- `Rt_median` (point estimate)
- `Rt_lower` (5th-percentile credible/confidence bound, optional but encouraged)
- `Rt_upper` (95th-percentile credible/confidence bound, optional but encouraged)

If your code raises errors, diagnose and fix them. If results look implausible, adjust the model and re-run.
