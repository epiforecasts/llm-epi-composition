import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('cases.csv')
df['date'] = pd.to_datetime(df['date'])

# Define constants
generation_interval = 5.0  # Average generation interval in days

# Fit model
with pm.Model() as model:
    # Day-of-week effects (multiplicative)
    day_of_week Effects = pm.Normal('day_of_week_effects', mu=1, sigma=0.2,
                                    shape=(7,))
    
    # Time-varying ascertainment (smoothly varying proportion of infections reported)
    ascertainment = pm.Uniform('ascertainment', lower=0, upper=1,
                               transformed=pm.distributions.transforms.Log,
                               shape=(len(df),))
    
    # Infection dynamics
    I_past = df['cases'].shift(generation_interval).values  # Lagged cases by generation interval
    
    # Renewal equation with day-of-week effects and ascertainment
    Rt = pm.Deterministic('Rt', (day_of_week_effects * I_past) * ascertainment)
    
    # Observation model (Poisson with overdispersion)
    observed_cases = pm.Poisson('observed_cases', mu=Rt,
                                observed=df['cases'],
                                shape=(len(df),),
                                tol=1e-6)
    
# Run MCMC
with model:
    idata = pm.sample(2000, tune=1000)

# Extract estimates
Rt_estimates = idata.posterior['Rt'].mean(axis=0).values
day_of_week_effect_estimates = idata.posterior['day_of_week_effects'].mean(axis=0).values
ascertainment_estimates = idata.posterior['ascertainment'].mean(axis=0).values

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(Rt_estimates)
plt.title('Estimated Time-Varying Reproduction Number (Rt)')
plt.xlabel('Time')
plt.ylabel('Rt estimate')
plt.show()

plt.figure(figsize=(10, 6))
plt.bar(range(1, 8), day_of_week_effect_estimates)
plt.title('Day-of-Week Effect Estimates')
plt.xlabel('Day of Week')
plt.ylabel('Effect Estimate')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(ascertainment_estimates)
plt.title('Time-Varying Ascertainment Estimates')
plt.xlabel('Time')
plt.ylabel('Ascertainment estimate')
plt.show()

