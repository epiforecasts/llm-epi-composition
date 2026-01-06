
import os
os.chdir("/home/eidesfun/code/llm-epi-composition/evaluation/results/scenario_1b/pymc/claude-sonnet-4-20250514/run_03")

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
from scipy.stats import gamma, lognorm
import arviz as az
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filename='cases.csv'):
    """Load and preprocess case data"""
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        # Generate synthetic data for demonstration
        print("cases.csv not found. Generating synthetic data for demonstration.")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic COVID-19 case data for demonstration"""
    dates = pd.date_range('2020-03-01', '2020-08-01', freq='D')
    n_days = len(dates)
    
    # Simulate true Rt values with realistic dynamics
    t = np.arange(n_days)
    true_rt = (2.5 * np.exp(-t/30) + 0.8 + 
               0.3 * np.sin(2*np.pi*t/14) * np.exp(-t/50))
    
    # Generation interval (discretized gamma distribution)
    generation_interval = get_generation_interval()
    
    # Reporting delay (discretized gamma distribution)
    reporting_delay = get_reporting_delay()
    
    # Simulate infections using renewal equation
    infections = np.zeros(n_days)
    infections[:7] = np.random.poisson(50, 7)  # Initial seeding
    
    for t in range(7, n_days):
        infectiousness = sum(infections[max(0, t-s)] * generation_interval[s] 
                           for s in range(1, min(t+1, len(generation_interval))))
        infections[t] = np.random.poisson(max(1, true_rt[t] * infectiousness))
    
    # Apply reporting delay and observation noise
    cases = np.zeros(n_days)
    for t in range(n_days):
        for d in range(len(reporting_delay)):
            if t + d < n_days:
                cases[t + d] += np.random.poisson(infections[t] * reporting_delay[d])
    
    # Add some additional observation noise
    cases = np.maximum(0, cases + np.random.normal(0, np.sqrt(cases + 1)))
    cases = cases.astype(int)
    
    df = pd.DataFrame({'date': dates, 'cases': cases})
    return df

def get_generation_interval(max_days=20):
    """
    Get discretized generation interval distribution
    Based on COVID-19 literature (mean ~5.5 days, std ~2.1 days)
    """
    # Parameters for gamma distribution
    mean_gi = 5.5
    std_gi = 2.1
    
    # Convert to gamma parameters
    shape = (mean_gi / std_gi) ** 2
    rate = mean_gi / (std_gi ** 2)
    
    # Discretize
    days = np.arange(1, max_days + 1)
    pmf = gamma.cdf(days + 0.5, shape, scale=1/rate) - gamma.cdf(days - 0.5, shape, scale=1/rate)
    pmf = pmf / pmf.sum()  # Normalize
    
    return pmf

def get_reporting_delay(max_days=15):
    """
    Get discretized reporting delay distribution
    Based on typical COVID-19 reporting patterns
    """
    # Parameters for gamma distribution (mean ~3 days, std ~2 days)
    mean_delay = 3.0
    std_delay = 2.0
    
    shape = (mean_delay / std_delay) ** 2
    rate = mean_delay / (std_delay ** 2)
    
    # Discretize
    days = np.arange(0, max_days)
    pmf = gamma.cdf(days + 0.5, shape, scale=1/rate) - gamma.cdf(days - 0.5, shape, scale=1/rate)
    pmf = pmf / pmf.sum()
    
    return pmf

def create_infectiousness_matrix(generation_interval, n_days):
    """Create matrix for computing infectiousness from past infections"""
    max_gi = len(generation_interval)
    infectiousness_matrix = np.zeros((n_days, n_days))
    
    for t in range(n_days):
        for s in range(1, min(t + 1, max_gi + 1)):
            if t - s >= 0:
                infectiousness_matrix[t, t - s] = generation_interval[s - 1]
    
    return infectiousness_matrix

def create_reporting_matrix(reporting_delay, n_days):
    """Create matrix for mapping infections to reported cases"""
    max_delay = len(reporting_delay)
    reporting_matrix = np.zeros((n_days, n_days))
    
    for t in range(n_days):
        for d in range(max_delay):
            if t + d < n_days:
                reporting_matrix[t + d, t] = reporting_delay[d]
    
    return reporting_matrix

def build_renewal_model(cases, generation_interval, reporting_delay, seed_days=7):
    """Build PyMC model for Rt estimation using renewal equation"""
    
    n_days = len(cases)
    
    # Create matrices for vectorized operations
    infectiousness_matrix = create_infectiousness_matrix(generation_interval, n_days)
    reporting_matrix = create_reporting_matrix(reporting_delay, n_days)
    
    with pm.Model() as model:
        # Prior for initial infections (seeding period)
        initial_infections = pm.Exponential('initial_infections', 
                                          lam=1/50, shape=seed_days)
        
        # Prior for Rt - allow for time-varying reproduction number
        # Use random walk to allow smooth changes over time
        rt_raw = pm.GaussianRandomWalk('rt_raw', 
                                      sigma=0.1, 
                                      shape=n_days - seed_days,
                                      init_dist=pm.Normal.dist(mu=0, sigma=0.5))
        
        # Transform to ensure Rt > 0
        rt = pm.Deterministic('rt', pm.math.exp(rt_raw))
        
        # Combine initial infections and computed infections
        infections = pt.zeros(n_days)
        infections = pt.set_subtensor(infections[:seed_days], initial_infections)
        
        # Compute infections for days after seeding using renewal equation
        for t in range(seed_days, n_days):
            # Compute infectiousness (sum of past infections weighted by generation interval)
            infectiousness = pt.sum(infections[:t] * infectiousness_matrix[t, :t])
            
            # Compute expected infections
            expected_infections = rt[t - seed_days] * infectiousness
            
            # Set infections for day t
            infections = pt.set_subtensor(infections[t], 
                                        pm.Poisson.dist(mu=pm.math.maximum(expected_infections, 0.1)))
        
        # Apply reporting delay to get expected reported cases
        expected_cases = pt.dot(reporting_matrix, infections)
        
        # Add small constant to avoid zero expected cases
        expected_cases = pm.math.maximum(expected_cases, 0.1)
        
        # Observation model with overdispersion
        phi = pm.Exponential('phi', lam=0.1)  # Overdispersion parameter
        
        # Negative binomial likelihood
        obs = pm.NegativeBinomial('obs', 
                                 mu=expected_cases, 
                                 alpha=phi,
                                 observed=cases)
        
        # Store matrices as constants for later use
        pm.ConstantData('infectiousness_matrix', infectiousness_matrix)
        pm.ConstantData('reporting_matrix', reporting_matrix)
        
    return model

def fit_model(model, samples=2000, tune=1000, chains=2, target_accept=0.9):
    """Fit the renewal equation model"""
    
    with model:
        # Use NUTS sampler with higher target acceptance for better sampling
        trace = pm.sample(
            samples, 
            tune=tune,
            chains=chains,
            target_accept=target_accept,
            return_inferencedata=True,
            random_seed=42
        )
    
    return trace

def extract_rt_estimates(trace, dates):
    """Extract Rt estimates with credible intervals"""
    
    rt_samples = trace.posterior['rt']
    
    # Compute summary statistics
    rt_mean = rt_samples.mean(dim=['chain', 'draw'])
    rt_lower = rt_samples.quantile(0.025, dim=['chain', 'draw'])
    rt_upper = rt_samples.quantile(0.975, dim=['chain', 'draw'])
    rt_median = rt_samples.quantile(0.5, dim=['chain', 'draw'])
    
    # Create results DataFrame (excluding seeding period)
    seed_days = len(dates) - len(rt_mean)
    
    results = pd.DataFrame({
        'date': dates[seed_days:],
        'rt_mean': rt_mean.values,
        'rt_median': rt_median.values,
        'rt_lower': rt_lower.values,
        'rt_upper': rt_upper.values
    })
    
    return results

def plot_results(df, rt_results, generation_interval, reporting_delay):
    """Create comprehensive plots of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Case counts over time
    axes[0,0].plot(df['date'], df['cases'], 'k-', alpha=0.7, linewidth=2)
    axes[0,0].set_title('Daily Reported Cases', fontsize=12, fontweight='bold')
    axes[0,0].set_ylabel('Cases')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates over time
    axes[0,1].plot(rt_results['date'], rt_results['rt_median'], 
                   'b-', linewidth=2, label='Median Rt')
    axes[0,1].fill_between(rt_results['date'], 
                          rt_results['rt_lower'], 
                          rt_results['rt_upper'],
                          alpha=0.3, color='blue', label='95% CI')
    axes[0,1].axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    axes[0,1].set_title('Time-varying Reproduction Number (Rt)', fontsize=12, fontweight='bold')
    axes[0,1].set_ylabel('Rt')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Plot 3: Generation interval distribution
    axes[1,0].bar(range(1, len(generation_interval) + 1), generation_interval, 
                  alpha=0.7, color='green')
    axes[1,0].set_title('Generation Interval Distribution', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Days')
    axes[1,0].set_ylabel('Probability')
    axes[1,0].grid(True, alpha=0.3)
    
    # Plot 4: Reporting delay distribution
    axes[1,1].bar(range(len(reporting_delay)), reporting_delay, 
                  alpha=0.7, color='orange')
    axes[1,1].set_title('Reporting Delay Distribution', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Days')
    axes[1,1].set_ylabel('Probability')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Rt trajectory with key statistics
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(rt_results['date'], rt_results['rt_median'], 'b-', linewidth=3, label='Median Rt')
    ax.fill_between(rt_results['date'], 
                   rt_results['rt_lower'], 
                   rt_results['rt_upper'],
                   alpha=0.3, color='blue', label='95% Credible Interval')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Rt = 1 (epidemic threshold)')
    
    # Highlight periods where Rt > 1
    above_one = rt_results['rt_lower'] > 1
    if above_one.any():
        ax.fill_between(rt_results['date'], 0, 3, 
                       where=above_one, alpha=0.1, color='red',
                       label='Likely growing (95% CI > 1)')
    
    ax.set_title('COVID-19 Reproduction Number (Rt) Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Reproduction Number (Rt)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, max(rt_results['rt_upper'].max() * 1.1, 3))
    
    plt.tight_layout()
    plt.show()

def print_summary(rt_results):
    """Print summary statistics of Rt estimates"""
    
    print("\n" + "="*60)
    print("RT ESTIMATION SUMMARY")
    print("="*60)
    
    print(f"Estimation period: {rt_results['date'].min().strftime('%Y-%m-%d')} to {rt_results['date'].max().strftime('%Y-%m-%d')}")
    print(f"Number of days estimated: {len(rt_results)}")
    
    print(f"\nOverall Rt statistics:")
    print(f"  Mean Rt: {rt_results['rt_mean'].mean():.2f}")
    print(f"  Median Rt: {rt_results['rt_median'].median():.2f}")
    print(f"  Min Rt (median): {rt_results['rt_median'].min():.2f}")
    print(f"  Max Rt (median): {rt_results['rt_median'].max():.2f}")
    
    # Periods of growth vs decline
    growing_days = (rt_results['rt_lower'] > 1).sum()
    likely_growing = (rt_results['rt_median'] > 1).sum()
    
    print(f"\nEpidemic dynamics:")
    print(f"  Days with Rt > 1 (median): {likely_growing} ({100*likely_growing/len(rt_results):.1f}%)")
    print(f"  Days with 95% CI > 1: {growing_days} ({100*growing_days/len(rt_results):.1f}%)")
    
    # Recent trend (last 7 days)
    if len(rt_results) >= 7:
        recent_rt = rt_results['rt_median'].iloc[-7:].mean()
        print(f"  Average Rt (last 7 days): {recent_rt:.2f}")
    
    print("="*60)

def main():
    """Main execution function"""
    
    print("COVID-19 Rt Estimation Using Renewal Equation")
    print("=" * 50)
    
    # Load data
    print("\n1. Loading case data...")
    df = load_data()
    print(f"   Loaded {len(df)} days of data from {df['date'].min()} to {df['date'].max()}")
    
    # Get epidemiological parameters
    print("\n2. Setting up epidemiological parameters...")
    generation_interval = get_generation_interval()
    reporting_delay = get_reporting_delay()
    print(f"   Generation interval: mean = {np.sum(np.arange(1, len(generation_interval)+1) * generation_interval):.1f} days")
    print(f"   Reporting delay: mean = {np.sum(np.arange(len(reporting_delay)) * reporting_delay):.1f} days")
    
    # Build model
    print("\n3. Building renewal equation model...")
    model = build_renewal_model(df['cases'].values, generation_interval, reporting_delay)
    print("   Model built successfully")
    
    # Fit model
    print("\n4. Fitting model using MCMC...")
    print("   This may take several minutes...")
    
    try:
        trace = fit_model(model, samples=1000, tune=1000, chains=2)
        print("   Model fitted successfully")
        
        # Check convergence
        rhat = az.rhat(trace)
        max_rhat = float(rhat['rt'].max())
        print(f"   Max R-hat: {max_rhat:.3f} {'(Good)' if max_rhat < 1.1 else '(Check convergence)'}")
        
    except Exception as e:
        print(f"   Error during model fitting: {str(e)}")
        print("   Try reducing the number of samples or adjusting model parameters")
        return
    
    # Extract results
    print("\n5. Extracting Rt estimates...")
    rt_results = extract_rt_estimates(trace, df['date'])
    print(f"   Extracted Rt estimates for {len(rt_results)} days")
    
    # Print summary
    print_summary(rt_results)
    
    # Create plots
    print("\n6. Creating visualizations...")
    plot_results(df, rt_results, generation_interval, reporting_delay)
    
    # Save results
    print("\n7. Saving results...")
    rt_results.to_csv('rt_estimates.csv', index=False)
    print("   Results saved to 'rt_estimates.csv'")
    
    print("\n" + "="*50)
    print("Analysis completed successfully!")
    print("="*50)
    
    return rt_results, trace, model

if __name__ == "__main__":
    rt_results, trace, model = main()

