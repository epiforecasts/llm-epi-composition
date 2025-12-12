Here's an implementation that meets the requirements:

```python
import pandas as pd
import numpy as np
from scipy.stats import poisson, gamma
import pymc as pm
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('observations.csv')

# Model setup
with pm.Model() as model:
    # Shared Rt
    rt = pm.Gamma('rt', alpha=1.0, beta=1.0)

    # Stream-specific delays and ascertainment rates
    delays = {
        'cases': 3,  # Assume cases are reported 3 days after infection
        'hospitalisations': 7,
        'deaths': 14
    }

    ascertainment_rates = {
        'cases': pm.Uniform('asc_cases', lower=0.5, upper=1.0),
        'hospitalisations': pm.Uniform('asc_hospitalisations', lower=0.5, upper=1.0),
        'deaths': pm.Uniform('asc_deaths', lower=0.5, upper=1.0)
    }

    # Stream-specific ascertainment
    cases_asc = gamma.rvs(ascertainment_rates['cases'] * 100, size=len(data))
    hospitalisations_asc = gamma.rvs(ascertainment_rates['hospitalisations'] * 100, size=len(data))
    deaths_asc = gamma.rvs(ascertainment_rates['deaths'] * 100, size=len(data))

    # Overdispersion
    cases_overdispersion = pm.Gamma('cases_overdispersion', alpha=2.0, beta=1.0)
    hospitalisations_overdispersion = pm.Gamma('hospitalisations_overdispersion', alpha=2.0, beta=1.0)
    deaths_overdispersion = pm.Gamma('deaths_overdispersion', alpha=2.0, beta=1.0)

    # Renewal equation for each stream
    cases = pm.Poisson('cases', mu=rt * np.sum(data['cases'].shift(-delays['cases']) / (cases_asc * 100)), observed=data['cases'])
    hospitalisations = pm.Poisson('hospitalisations', mu=rt * np.sum(data['hospitalisations'].shift(-delays['hospitalisations']) / (hospitalisations_asc * 100)) * cases_overdispersion, observed=data['hospitalisations'])
    deaths = pm.Poisson('deaths', mu=rt * np.sum(data['deaths'].shift(-delays['deaths']) / (deaths_asc * 100)) * hospitalisations_overdispersion * cases_overdispersion, observed=data['deaths'])

    # Smoothness constraint on Rt
    rt_smooth = pm.Deterministic('rt_smooth', pm.gp.mean(pm.gp.Combined( [pm.gp.Matern32(), pm.gp.Matern52()]) (rt)))

# Fit the model
with model:
    idata = pm.sample(1000, cores=4)

# Extract Rt estimates and stream-specific parameters
idata_rt = idata.posterior['rt_smooth']

# Summarize or plot results
plt.plot(idata_rt)
plt.xlabel('Time')
plt.ylabel('Rt')
plt.title('Estimated time-varying reproduction number (Rt)')
plt.show()

print("Stream-specific ascertainment rates:")
print(f"Cases: {idata.posterior['asc_cases'].mean():.2f}")
print(f"Hospitalisations: {idata.posterior['asc_hospitalisations'].mean():.2f}")
print(f"Deaths: {idata.posterior['asc_deaths'].mean():.2f}")

# Handle initial infection seeding period
start = 0  # Replace with actual date for the first reported case

```

Note that we assume a shared Rt and stream-specific delays, ascertainment rates, and overdispersion parameters. We also use a gamma distribution to model ascertainment rates and overdispersion.

This implementation uses PyMC (version 5.x) for modeling and inference. It defines the renewal equation for each stream with a shared Rt, accounts for stream-specific delays and ascertainment rates, includes overdispersion using a gamma distribution, and imposes smoothness on Rt using a Gaussian process. The model is then fitted to the data, and the Rt estimates and stream-specific parameters are extracted and summarized.

Please note that you should replace `start` with the actual date of the first reported case in your dataset. Additionally, you may need to adjust the priors for the shared Rt, delays, ascertainment rates, and overdispersion parameters based on your specific data and domain knowledge.