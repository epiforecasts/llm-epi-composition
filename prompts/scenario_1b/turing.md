# Scenario 1b: Estimate Rt Using the Renewal Equation

## Task

Estimate the time-varying reproduction number (Rt) from daily reported COVID-19 case counts using the renewal equation framework.

## Background

The reproduction number Rt represents the average number of secondary infections caused by a single infected individual at time t.

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function (the probability that the generation interval is exactly $s$ days).

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

1. Write complete, runnable code that estimates Rt over time using the renewal equation
2. The model should:
   - Use the renewal equation to model infection dynamics
   - Convolve infections with the delay distribution to get expected cases
   - Include an appropriate observation model for case counts
3. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates for each time point
4. Include uncertainty quantification (e.g., confidence/credible intervals)
5. Handle the initial infection seeding period appropriately

## Output

Your code should produce:
- Rt estimates for each time point
- A measure of uncertainty (e.g., 90% intervals)
- A summary or plot of Rt over time

## Language

Use Turing.jl for the model in Julia.
