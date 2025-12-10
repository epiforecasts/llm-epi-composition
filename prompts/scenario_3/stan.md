# Scenario 3: Multiple Data Streams

## Task

Estimate the time-varying reproduction number (Rt) jointly from three data streams: cases, hospitalisations, and deaths.

## Background

The reproduction number Rt represents the average number of secondary infections caused by a single infected individual at time t.

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function.

Different data streams provide complementary information about the underlying infection dynamics:
- **Cases**: Most timely but subject to testing behaviour and ascertainment changes
- **Hospitalisations**: More reliable denominator but delayed relative to infection
- **Deaths**: Most complete ascertainment but longest delay

Each stream has its own:
- Delay distribution (time from infection to observation)
- Ascertainment rate (proportion of infections observed)
- Observation noise characteristics

By jointly modelling these streams, we can leverage their complementary strengths.

## Data

You are provided with:

1. **Daily observations** (`observations.csv`): A CSV file with columns:
   - `date`: Date in YYYY-MM-DD format
   - `cases`: Number of reported cases on that date
   - `hospitalisations`: Number of hospital admissions on that date
   - `deaths`: Number of deaths on that date

2. **Generation interval distribution**: The time between successive infections in a transmission chain:
   ```
   gen_int = [0.0, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02]
   ```

3. **Delay distributions** (infection to observation):

   Cases:
   ```
   delay_cases = [0.0, 0.05, 0.15, 0.25, 0.25, 0.15, 0.1, 0.05]
   ```

   Hospitalisations:
   ```
   delay_hosp = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.18, 0.12, 0.08, 0.05, 0.03, 0.02]
   ```

   Deaths:
   ```
   delay_deaths = [0.0, 0.0, 0.01, 0.02, 0.04, 0.07, 0.10, 0.12, 0.14, 0.13, 0.11, 0.09, 0.07, 0.05, 0.03, 0.02]
   ```

## Requirements

1. Write complete, runnable code that estimates a single shared Rt over time using the renewal equation
2. The model should include:
   - The renewal equation for infection dynamics with a **shared Rt**
   - **Stream-specific delay convolutions**: Each observation type has its own delay
   - **Stream-specific ascertainment**: Each stream has a proportion of infections observed
   - **Overdispersion**: Account for greater variance than Poisson in observations
   - **Smoothness constraint on Rt**: Rt should vary smoothly over time
3. Provide code to:
   - Load the data
   - Define and fit the model
   - Extract Rt estimates and stream-specific parameters
4. Handle the initial infection seeding period appropriately

## Output

Your code should produce:
- Rt estimates for each time point
- Stream-specific ascertainment rate estimates
- A summary or plot showing results

## Language

Use Stan for the model, with R for data preparation and model fitting (using cmdstanr or rstan).
