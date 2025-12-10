# Scenario 1a: Estimate Rt from Case Counts

## Task

Estimate the time-varying reproduction number (Rt) from daily reported COVID-19 case counts.

## Data

You are provided with daily case counts (`cases.csv`): A CSV file with columns:
- `date`: Date in YYYY-MM-DD format
- `cases`: Number of reported cases on that date

## Requirements

1. Write complete, runnable code that estimates Rt over time
2. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates for each time point

## Output

Your code should produce:
- Rt estimates for each time point
- A summary or plot of Rt over time

## Language

Use Turing.jl for the model in Julia.
