
import os
os.chdir("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/pymc/llama3.1:8b/run_01")

import pandas as pd
import numpy as np
from pymc import Model, Deterministic, Normal
from pymc import MCMC, sample

# Load data
data = pd.read_csv('cases.csv')

# Ensure dates are in datetime format
data['date'] = pd.to_datetime(data['date'])

# Calculate time delta between days (to be used as index)
time_deltas = data['date'].diff().dt.days.values

# Define the model
with Model() as model:
    # Assume an exponential distribution for generation interval probabilities
    lam = 7.0  # mean of generation interval, assume gamma distributed with shape=3 and scale=1/7
    g_s = [np.exp(-lam * s) for s in range(31)]  # We'll only consider up to 30 days
    
    # Define the parameters for the renewal equation
    Rt = Deterministic('Rt', np.zeros(len(data)))
    
    # Renewal Equation Model
    infections = np.cumsum([g_s[day] * data['cases'][i] for i, day in enumerate(time_deltas)])
    
    # Observation model
    observed_cases = Normal('observed_cases', mu=data['cases'], sigma=10)
    
    # Define the Rt parameters
    R_params = Deterministic('Rt_parameters',
                             [0.1 + 0.9 * (infections[i] / sum(infections[:i+1])) for i in range(len(data))])
    
    # Add the Rt as a Deterministic variable
    Rt[...] = R_params
    
# Initialize the sampler and run it
with model:
    trace = MCMC(model)
    trace.init(
        random_seed=42, 
        start='adapt_full',
    )
    trace.sample(10000)

# Extract Rt estimates for each time point
Rt_estimates = np.array([trace.get_values('Rt_parameters')])

# Plot Rt over time
import matplotlib.pyplot as plt

plt.plot(Rt_estimates.mean(axis=0))
plt.xlabel('Time')
plt.ylabel('Estimated Reproduction Number (Rt)')
plt.title('Estimated Rt Over Time')
plt.show()

