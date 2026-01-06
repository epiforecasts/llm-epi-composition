import pandas as pd
import numpy as np
from pymc import Model, StepMethod, SampleFromPrior, NUTS
import pymc as pm
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('observations.csv')

# Define model parameters
n_cases = len(data['cases'])
n_hospitalisations = len(data['hospitalisations'])
n_deaths = len(data['deaths'])

# Delay and ascertainment rates for each stream
delay_cases = 0.5  # Example delay, adjust as needed
delay_hospitalisations = 3
delay_deaths = 7

ascertainment_rates = [1/6, 1/12, 1/20]  # Adjust these as needed to reflect proportions of infections observed

# Prior distribution for Rt
rtprior = pm.Uniform('Rt', lower=0.5, upper=2)

# Infection dynamics model using renewal equation
with Model() as model:
    Rt = rtprior

    cases_observed = pm.Poisson('cases_observed', mu=Rt*pm.math.sum([cases[(delay_cases + s) % n_cases] for s in range(n_cases)], axis=0))
    hospitalisations_observed = pm.Poisson('hospitalisations_observed', mu=Rt*pm.math.sum([hospitalisations[(delay_hospitalisations + s) % n_hospitalisations] for s in range(n_hospitalisations)], axis=0))
    deaths_observed = pm.Poisson('deaths_observed', mu=Rt*pm.math.sum([deaths[(delay_deaths + s) % n_deaths] for s in range(n_deaths)], axis=0))

    # Overdispersion to account for greater variance than Poisson
    cases_observed_dist = pm.Deterministic('cases_observed_dist', cases_observed * (1+pm.math.sqrt(pm.math.log(n_cases))))
    hospitalisations_observed_dist = pm.Deterministic('hospitalisations_observed_dist', hospitalisations_observed * (1+pm.math.sqrt(pm.math.log(n_hospitalisations))))
    deaths_observed_dist = pm.Deterministic('deaths_observed_dist', deaths_observed * (1+pm.math.sqrt(pm.math.log(n_deaths))))

# Initialising the seed
data['cases'][0] = 100  # Assign a dummy value to represent initial infections

# Model fitting and MCMC sampling
with model:
    trace = pm.sample(2000, tune=1000)

# Extract Rt estimates
rtestimates = trace['Rt']

# Plotting the results
plt.plot(rtestimates)
plt.title('Time-Varying Reproduction Number (Rt)')
plt.xlabel('Time')
plt.ylabel('Reproduction Number')
plt.show()

print(pm.summary(trace))

