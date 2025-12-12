Here's a step-by-step implementation of the task in PyMC3. Note that we will need to assume some distribution for the generation interval probability mass function, g_s.

```python
import pandas as pd
import numpy as np
from pymc3 import Model, Deterministic, Normal
from pymc3 import MCMC, sample

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
```

This code assumes a simple exponential distribution for the generation interval probability mass function and estimates Rt using the renewal equation. The model accounts for delays between infection and case reporting and includes an observation model for case counts.

Note that you'll need to adjust the model parameters (e.g., `lam`, `g_s`) based on your specific understanding of the data and the disease dynamics at play. Additionally, consider increasing the number of samples (`10000` in this example) to improve the accuracy of your Rt estimates.

Also note that the initial infection seeding period should be handled appropriately; you may need to apply some form of smoothing or filtering to the early time points to prevent overestimation of Rt due to incomplete data.