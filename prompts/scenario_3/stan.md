# Scenario 3: Multiple Data Streams

## Task

Estimate the time-varying reproduction number (Rt) jointly from three data streams: cases, hospitalisations, and deaths.

## Background

The renewal equation relates infections at time t to past infections:

$$I_t = R_t \sum_{s=1}^{S} I_{t-s} \cdot g_s$$

where $g_s$ is the generation interval probability mass function.

Different data streams provide complementary information about the underlying infection dynamics:
- **Cases**: Most timely but subject to testing behaviour and ascertainment changes
- **Hospitalisations**: More reliable denominator but delayed relative to infection
- **Deaths**: Most complete ascertainment but longest delay

Each stream has its own delay from infection to observation, ascertainment rate, and observation noise characteristics.

## Data

You are provided with daily observations (`observations.csv`): A CSV file with columns:
- `date`: Date in YYYY-MM-DD format
- `cases`: Number of reported cases on that date
- `hospitalisations`: Number of hospital admissions on that date
- `deaths`: Number of deaths on that date

## Requirements

1. Write complete, runnable code that estimates a single shared Rt over time using the renewal equation
2. The model should include:
   - The renewal equation for infection dynamics with a **shared Rt**
   - **Stream-specific delays**: Each observation type has its own delay from infection
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
