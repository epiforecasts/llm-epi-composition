# Estimate the time-varying reproduction number

You are analysing daily counts from an infectious disease outbreak. Your task is to estimate the time-varying effective reproduction number $R_t$ (the average number of secondary infections caused by an individual infected at time $t$) from the observed data.

## Data

The observation period is 150 days, dated 2023-01-01 to 2023-05-30. The data are in:

- `data/cases.csv` with columns `date` (YYYY-MM-DD) and `cases` (integer counts).

## Epidemiological parameters

These distributions come from external studies of the pathogen. Treat them as fixed knowns; no need to propagate uncertainty.

- Generation interval: Gamma distribution, mean 5.5 days, standard deviation 2 days.
- Reporting delay (infection to observation in cases): log-normal distribution, mean 5 days, standard deviation 2 days.

## Structural features of the data

- There is a reporting delay between infection and case observation, with the distribution given above.
- Counts are integer-valued and exhibit day-of-week fluctuations consistent with reduced reporting on weekends.
- Observation noise is greater than would be expected from a homogeneous Poisson model.

## Task

Produce daily point estimates of $R_t$ for the full observation period.

**Use the renewal equation as the basis of your model.** That is, use the relationship between $R_t$, the generation interval, and past infections to model the infection process, then connect the latent infection process to the observed cases via the reporting delay.

**Use Julia.** Standard Julia packages are available. A Turing.jl API reference is provided in `turing_api_docs.md`; use Turing.jl for any probabilistic / Bayesian modelling.

Save your final $R_t$ estimates to `outputs/rt_estimates.csv` with columns:

- `date` (YYYY-MM-DD)
- `Rt_median` (point estimate)
- `Rt_lower` (5th-percentile credible/confidence bound, optional but encouraged)
- `Rt_upper` (95th-percentile credible/confidence bound, optional but encouraged)

If your code raises errors, diagnose and fix them. If results look implausible, adjust the model and re-run.
