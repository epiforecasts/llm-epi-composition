# Scenario 2: Structured Rt with Observation Processes

## Task

Estimate the time-varying reproduction number (Rt) from daily reported COVID-19 case counts, accounting for complex observation processes.

## Background

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function.

Reported cases are not a direct observation of infections. The observation process may include:
- A delay from infection to reporting
- Day-of-week effects (fewer cases reported on weekends)
- Time-varying ascertainment (the proportion of infections that become reported cases may change over time)
- Overdispersion (more variance than a Poisson distribution would predict)

## Data

You are provided with daily case counts (`cases.csv`): A CSV file with columns:
- `date`: Date in YYYY-MM-DD format
- `cases`: Number of reported cases on that date
- `day_of_week`: Integer 1-7 (Monday=1, Sunday=7)

## Requirements

1. Write complete, runnable code that estimates Rt over time using the renewal equation
2. The model should include:
   - The renewal equation for infection dynamics
   - Delays between infection and reporting
   - **Day-of-week effects**: A multiplicative effect on reported cases for each day of the week
   - **Time-varying ascertainment**: A smoothly varying proportion of infections that are reported
   - **Overdispersion**: Account for greater variance than Poisson in observations
3. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract estimates of Rt, day-of-week effects, and ascertainment over time
4. Handle the initial infection seeding period appropriately

## Output

Your code should produce:
- Rt estimates for each time point
- Day-of-week effect estimates
- Time-varying ascertainment estimates
- A summary or plot showing results

## Language

Use Turing.jl for the model in Julia.
