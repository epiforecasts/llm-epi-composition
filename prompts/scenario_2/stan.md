# Scenario 2: Structured Rt with Observation Processes

## Task

Estimate the time-varying reproduction number (Rt) from daily reported COVID-19 case counts, accounting for complex observation processes.

## Background

The reproduction number Rt represents the average number of secondary infections caused by a single infected individual at time t.

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function.

However, reported cases are not a direct observation of infections. The observation process includes:
- A delay from infection to reporting
- Day-of-week effects (fewer cases reported on weekends)
- Time-varying ascertainment (the proportion of infections that become reported cases may change over time)
- Overdispersion (more variance than a Poisson distribution would predict)

## Data

You are provided with:

1. **Daily case counts** (`cases.csv`): A CSV file with columns:
   - `date`: Date in YYYY-MM-DD format
   - `cases`: Number of reported cases on that date
   - `day_of_week`: Integer 1-7 (Monday=1, Sunday=7)

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

1. Write complete, runnable code that estimates Rt over time using the renewal equation
2. The model should include:
   - The renewal equation for infection dynamics
   - Convolution with the delay distribution
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

Use Stan for the model, with R for data preparation and model fitting (using cmdstanr or rstan).
