# Scenario 1a: Estimate Rt from Case Counts (Open Method)

## Task

Estimate the time-varying reproduction number (Rt) from daily reported COVID-19 case counts.

## Background

The reproduction number Rt represents the average number of secondary infections caused by a single infected individual at time t. Estimating Rt from case data is a fundamental task in infectious disease epidemiology.

## Data

You are provided with:

1. **Daily case counts** (`cases.csv`): A CSV file with columns:
   - `date`: Date in YYYY-MM-DD format
   - `cases`: Number of reported cases on that date

2. **Generation interval distribution**: The time between successive infections in a transmission chain, discretised to daily probabilities:
   ```
   gen_int = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
   ```
   (Index 0 = 0 days, index 1 = 1 day, etc. Sums to 1.0)

3. **Reporting delay distribution**: The delay from infection to case report, discretised to daily probabilities:
   ```
   delay_pmf = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]
   ```
   (Index 0 = 0 days delay, etc. Sums to 1.0)

## Requirements

1. Write complete, runnable code that estimates Rt over time
2. The model should account for the observation process (cases are not infections)
3. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates for each time point

## Output

Your code should produce:
- Rt estimates for each time point
- A summary or plot of Rt over time

## Language

Use PyMC (version 5.x) for the model in Python.
