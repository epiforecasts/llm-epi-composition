
import os
os.chdir("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_2/pymc/llama3.1:8b/run_03")

import pandas as pd
import numpy as np
from pmgpy import PMGPy
from pymc import (
    init_model,
    set_data,
    deterministic,
    Model,
    StepMethods,
    MCMC,
)
from scipy.stats import gamma

# Load the data
df = pd.read_csv("cases.csv")

# Ensure `date` is a datetime object and create a new column for day of year
df["date"] = pd.to_datetime(df["date"])
df["day_of_year"] = df["date"].dt.dayofyear

# Calculate day-of-week effect parameters (average and standard deviation)
dow_avg = np.mean(df.groupby("day_of_week")["cases"].mean())
dow_std = np.std(df.groupby("day_of_week")["cases"].mean())

# Fit the model
def fit_model(data):
    with init_model() as model:
        # Infection process parameters
        R_t = deterministic(
            name="Rt",
            var="Rt",
            dist=gamma,
            mu=np.mean(data["Rt"]),  # initial guess for Rt mean
            sigma=np.std(data["Rt"]),  # initial guess for Rt std
        )

        # Day-of-week effect parameters
        dow_avg = pm.Normal("dow_avg", mu=dow_avg, sd=0.1)  # Prior for day of week average
        dow_std = pm.Normal("dow_std", mu=dow_std, sd=0.1)  # Prior for day of week standard deviation

        # Time-varying ascertainment parameters
        alpha = pm.Uniform(
            "alpha",
            lower=0,
            upper=1,
            initial=np.mean(data["ascertainment"]),
        )  # Prior for ascertainment parameter
        beta = pm.Normal("beta", mu=np.mean(data["ascertainment"]), sd=0.1)  # Prior for ascertainment rate

        # Define the observation process
        obs_process = (
            alpha * (R_t ** dow_avg) * data["cases"]
            + (1 - alpha) * (np.exp(beta)) * np.random.poisson(
                R_t ** dow_avg, len(data), size=len(data)
            )
        )

        # Add a prior for the initial infection seeding period
        init_cases = pm.Normal("init_cases", mu=10, sd=5)

    return model

# Define and fit the model
model = fit_model(df)
mcmc = MCMC(model)
mcmc.sample(10000)  # sample from posterior distribution

# Extract estimates of Rt, day-of-week effects, and ascertainment over time
Rt_estimates = mcmc.trace("Rt")[:]

dow_avg_estimates = np.mean(mcmc.get_values("dow_avg"))
dow_std_estimates = np.std(mcmc.get_values("dow_std"))

ascertainment_estimates = mcmc.get_values("alpha")

# Handle the initial infection seeding period
init_cases = mcmc.trace("init_cases")[:]

print(f"Rt estimates: {Rt_estimates}")
print(f"DOW effect average estimate: {dow_avg_estimates}")
print(f"DOW effect std estimate: {dow_std_estimates}")
print(f"Ascentration estimate: {ascertainment_estimates}")

# Summary or plot showing results
import matplotlib.pyplot as plt

plt.plot(Rt_estimates)
plt.xlabel("Time")
plt.ylabel("Rt estimate")
plt.title("Estimated Rt over time")
plt.show()

