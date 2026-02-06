# Scenario 1a: Estimate Rt from Case Counts

## Task

Estimate the time-varying reproduction number (Rt) from daily COVID-19 case counts.

## Data

Daily COVID-19 case counts from England (`data/cases.csv`):
- `date`: Specimen date (date the test was taken), in YYYY-MM-DD format
- `cases`: Number of confirmed cases

Note: There is a delay between infection and specimen date. Recent data may be incomplete due to reporting delays.

## Requirements

1. Write complete, runnable code that estimates Rt over time
2. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates for each time point

## Output

Your code should produce:
- Rt estimates for each time point (the full trajectory)
- The current (most recent) Rt estimate with uncertainty
- A summary or plot of Rt over time

Save all results and any other output that may be useful to the user.

## Language

Use Python.
