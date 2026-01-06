
import os
os.chdir("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1a/pymc/llama3.1:8b/run_02")

# Import necessary libraries
import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('cases.csv')

# Set date as datetime object
df['date'] = pd.to_datetime(df['date'])

# Calculate time difference between consecutive days in days
time_diff = (df['date'].diff().fillna(pd.DateOffset(days=1)).dt.days)

# Define a simple model for Rt: assume it follows an Ornstein-Uhlenbeck process
with pm.Model() as model:
    # Prior distribution on Rt0 (Rt at the start of the data)
    R0 = pm.Normal('R0', mu=2.5, sigma=1)

    # Process noise variance (hyperparameter)
    omega = pm.Uninformative('omega', lower=0, initval=10)

    # Transition matrix for Ornstein-Uhlenbeck process
    Rt = pm.OrnsteinUhlenbeck('Rt', mu=R0, sigma=np.sqrt(omega), dt=time_diff.values, shape=(len(time_diff), 1))

# Fit the model using MAP estimation (or MCMC if desired)
with model:
    step = pm.Metropolis()
    trace = pm.sample_posterior_predictive(1000, vars=[Rt], tune=500, discard_tuned_samples=True, step=step)

# Extract Rt estimates for each time point
rt_estimates = trace['Rt']

# Plot Rt over time (example using matplotlib)
plt.figure(figsize=(8, 6))
for i in range(rt_estimates.shape[1]):
    plt.plot(df['date'].values[1:], rt_estimates[:, i], label=f'Rt_{i+1}')
plt.xlabel('Date')
plt.ylabel('Rt')
plt.title('Estimated Reproduction Number Over Time')
plt.legend()
plt.show()

# Print a summary of Rt estimates
print(rt_estimates.mean(axis=0))

