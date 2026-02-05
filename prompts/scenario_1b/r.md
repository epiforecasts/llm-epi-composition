# Scenario 1b: Estimate Rt Using the Renewal Equation

## Task

Estimate the time-varying reproduction number (Rt) from daily COVID-19 case counts using the renewal equation framework.

## Background

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function.

## Data

Daily COVID-19 case counts from England (`data/cases.csv`):
- `date`: Specimen date (date the test was taken), in YYYY-MM-DD format
- `cases`: Number of confirmed cases

Note: There is a delay between infection and specimen date. Recent data may be incomplete due to reporting delays.

## Requirements

1. Write complete, runnable code that estimates Rt over time using the renewal equation
2. The model should:
   - Use the renewal equation to model infection dynamics
   - Account for delays between infection and case reporting
   - Include an appropriate observation model for case counts
3. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates for each time point
4. Handle the initial infection seeding period appropriately

## Output

Your code should produce:
- Rt estimates for each time point (the full trajectory)
- The current (most recent) Rt estimate with uncertainty
- A summary or plot of Rt over time

Save all results and any other output that may be useful to the user.

## Language

Use R.
