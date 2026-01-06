
import os
os.chdir("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/pymc/llama3.1:8b/run_02")

import pandas as pd
import numpy as np
from pymc import Model, Deterministic, NUTS, sample_posterior_predictive
from scipy.stats import poisson
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('cases.csv')

# Define model and priors
with Model() as model:
    # Define variables
    R0 = Deterministic('Rt[1]', 2.5)
    
    # Day-of-week effect
    day_effect = np.zeros((7,))
    for i in range(1, 8):
        day_effect[i-1] = NUTS.dist(var=0.1, lower=-10, upper=10)
        
    delay = NUTs.dist(var=2, lower=0, upper=5) # days
    
    # Ascertainment (smoothly varying proportion of infections reported)
    ascertainment = Deterministic('ascertainment', np.exp(2 * R0 - 1))
    
    # Overdispersion parameter
    overdisp = NUTS.dist(var=0.1, lower=0, upper=5) 
    
    # Infection dynamics: renewal equation
    infections = Deterministic('infections',
        sum([cases[i-day] * ascertainment**i for i in range(1, data.shape[0]+1)])
        
    # Reported cases (accounting for delay, day-of-week effect, and overdispersion)
    reported = Deterministic('reported',
        poisson.rvs(mu=infections * np.exp(day_effect[data['day_of_week']-1] + 2*delay),
                    size=data.shape[0], 
                    overdisp=overdisp)
        
    # Fit the model
    start = {'Rt[1]': data['cases'].mean(),
             'delay': data['cases'].mean(),
             'ascertainment': np.mean(data['cases'] / sum(data['cases']))}
    
    with model:
        step = NUTS(vars=model.free_RVs)
        trace = sample_posterior_predictive(5000, progressbar=False)

# Extract estimates
rt_estimate = trace['Rt[1]'].mean(axis=0)
day_effect_estimate = trace['day_effect'].mean(axis=0)
ascertainment_estimate = trace['ascertainment'].mean(axis=0)

print(f'Rt Estimates: {rt_estimate}')
print(f'Day-of-Week Effects: {np.exp(day_effect_estimate)}')
print(f'Ascertainment Over Time: {np.exp(ascertainment_estimate)}')

# Summary plots
plt.figure(figsize=(10,6))
plt.plot(rt_estimate)
plt.title('Rt Estimates')
plt.show()

plt.figure(figsize=(10,6))
plt.bar(range(1, 8), np.exp(day_effect_estimate))
plt.title('Day-of-Week Effects')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(np.exp(ascertainment_estimate))
plt.title('Ascertainment Over Time')
plt.show()

