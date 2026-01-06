# Import necessary libraries
import pandas as pd
import pymc as pm
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('cases.csv')

# Convert date to datetime format for easier manipulation
data['date'] = pd.to_datetime(data['date'])

# Create a time-series index and align it with the data
ts_index = pd.date_range(start=data['date'].min(), end=data['date'].max())
aligned_data = data.set_index('date').reindex(ts_index).fillna(0)  # Fill missing values with 0

# Define the model
def rt_model(d, n_cases, tau=1/3.8):  # Using a typical estimate for tau (mean generation time)
    mu = pm.Normal('mu', mu=n_cases, sigma=10, shape=len(n_cases))
    beta = pm.Deterministic('beta', 1 / (d + tau))  # Deterministic for computational efficiency
    rt = pm.Deterministic('rt', n_cases * beta)  
    return rt

# Fit the model using PyMC
with pm.Model() as model:
    d = aligned_data.index.to_numpy()
    n_cases = aligned_data['cases'].values
    rt_samples = pm.sample_posterior_predictive(rt_model(d, n_cases), samples=10000)

# Extract Rt estimates for each time point
rt_estimates = np.mean(rt_samples['rt'], axis=0)

# Plot Rt over time (optional)
plt.figure(figsize=(8,6))
plt.plot(aligned_data.index, rt_estimates, color='blue')
plt.fill_between(aligned_data.index, 0, rt_estimates, alpha=0.2, color='blue')
plt.xlabel('Date')
plt.ylabel('Rt')
plt.title('Time-Varying Reproduction Number (Rt) Estimate')
plt.show()

# Print summary statistics for Rt estimates
print(f"Mean Rt estimate: {np.mean(rt_estimates):.4f}")
print(f"Median Rt estimate: {np.median(rt_estimates):.4f}")

