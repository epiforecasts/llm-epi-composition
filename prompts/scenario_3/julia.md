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

Daily COVID-19 observations from England (`data/observations.csv`):
- `date`: Specimen/event date, in YYYY-MM-DD format
- `cases`: Number of confirmed cases (by specimen date)
- `hospitalisations`: Number of hospital admissions
- `deaths`: Number of deaths

Note: Each data stream has different delays from infection to observation. Recent data may be incomplete due to reporting delays.

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
- Rt estimates for each time point (the full trajectory)
- The current (most recent) Rt estimate with uncertainty
- Stream-specific ascertainment rate estimates
- A summary or plot showing results

Save all results and any other output that may be useful to the user.

## Language

Use Julia.
