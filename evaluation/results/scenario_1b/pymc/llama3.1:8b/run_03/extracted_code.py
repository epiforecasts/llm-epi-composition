import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.stats import gamma

# Load data
data = pd.read_csv('cases.csv')
data['date'] = pd.to_datetime(data['date'])

# Calculate time differences (days) between successive cases
time_diffs = data['date'].diff().dt.days.values

# Define generation interval probability mass function
def generation_interval(gamma_shape, gamma_scale):
    return gamma(gamma_shape, scale=gamma_scale)

# Define observation model for case counts
def obs_model(Rt, past_cases, time_diffs, gamma_shape, gamma_scale):
    # Apply delays between infection and case reporting
    delayed_past_cases = np.zeros(len(time_diffs))
    for i in range(1, len(delayed_past_cases)):
        start_idx = max(0, i - 7)  # Assuming a maximum delay of 7 days
        end_idx = min(i, len(past_cases) - start_idx)
        delayed_past_cases[i] = np.sum(past_cases[start_idx:end_idx])
    
    # Apply renewal equation model
    infection_counts = Rt * np.convolve(delayed_past_cases, time_diffs[:len(delayed_past_cases)], mode='full')
    
    # Apply observation noise (Poisson distribution)
    observed_cases = pm.Poisson('observed_cases', mu=infection_counts)
    return observed_cases

# Define model
with pm.Model() as model:
    Rt = pm.Gamma('Rt', alpha=1, beta=1)  # Prior for reproduction number
    
    gamma_shape = pm.Uniform('gamma_shape', lower=0.1, upper=5)
    gamma_scale = pm.Uniform('gamma_scale', lower=0.1, upper=10)
    
    past_cases = data['cases'].values[:-7]  # Exclude last week
    time_diffs = np.repeat(data['date'].diff().dt.days.values, len(past_cases))[:len(past_cases)]
    
    observed_cases = obs_model(Rt, past_cases, time_diffs, gamma_shape, gamma_scale)
    
    # Inference
    step = pm.NUTS(tune=1000, target_accept=0.9)
    trace = pm.sample(10000, step=step)

# Extract Rt estimates for each time point
Rt_estimates = np.mean(pm.sample_posterior_predictive(trace, var_names=['Rt']), axis=0)

# Plot Rt over time (use the last 7 days of data to smooth out noise)
data['date'] = pd.to_datetime(data['date'])
last_week_cases = data['cases'].values[-7:]
smoothed_Rt = np.convolve(Rt_estimates, np.ones(7)/7, mode='valid')
plt.plot(last_week_cases[:-6], smoothed_Rt[:len(last_week_cases)-6], label='Estimated Rt')
plt.xlabel('Number of reported cases (last 7 days)')
plt.ylabel('Reproduction number (Rt)')
plt.title('Time-varying reproduction number (Rt) estimates over time')
plt.legend()
plt.show()

# Print summary statistics for Rt
print(pm.summary(trace, var_names=['Rt']))

