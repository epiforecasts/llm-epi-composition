import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(filepath='cases.csv'):
    """Load and prepare the case data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any negative cases and handle missing values
    df['cases'] = df['cases'].clip(lower=0)
    df = df.dropna()
    
    print(f"Data loaded: {len(df)} days from {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['cases'].sum():,}")
    print(f"Max daily cases: {df['cases'].max():,}")
    
    return df

def get_generation_interval_pmf(max_days=15, mean_gi=5.2, std_gi=1.72):
    """
    Generate generation interval probability mass function
    Based on COVID-19 literature estimates
    """
    # Use log-normal distribution for generation interval
    days = np.arange(1, max_days + 1)
    
    # Convert mean and std to log-normal parameters
    mu = np.log(mean_gi**2 / np.sqrt(std_gi**2 + mean_gi**2))
    sigma = np.sqrt(np.log(1 + std_gi**2 / mean_gi**2))
    
    # Calculate PMF
    pmf = stats.lognorm.pdf(days, s=sigma, scale=np.exp(mu))
    pmf = pmf / pmf.sum()  # Normalize to sum to 1
    
    print(f"Generation interval: mean={np.sum(days * pmf):.2f}, std={np.sqrt(np.sum(days**2 * pmf) - np.sum(days * pmf)**2):.2f}")
    
    return pmf

def get_reporting_delay_pmf(max_days=21, mean_delay=5.1, std_delay=3.2):
    """
    Generate reporting delay probability mass function
    Time from infection to case reporting
    """
    days = np.arange(0, max_days)
    
    # Use gamma distribution for reporting delay
    # Convert mean and std to shape and scale parameters
    scale = std_delay**2 / mean_delay
    shape = mean_delay / scale
    
    pmf = stats.gamma.pdf(days, a=shape, scale=scale)
    pmf = pmf / pmf.sum()  # Normalize
    
    print(f"Reporting delay: mean={np.sum(days * pmf):.2f}, std={np.sqrt(np.sum(days**2 * pmf) - np.sum(days * pmf)**2):.2f}")
    
    return pmf

def create_rt_model(cases, generation_interval, reporting_delay):
    """
    Create PyMC model for estimating time-varying Rt using renewal equation
    """
    n_days = len(cases)
    max_gi = len(generation_interval)
    max_delay = len(reporting_delay)
    
    with pm.Model() as model:
        # Priors for initial infections (seeding period)
        seed_days = max_gi + max_delay
        log_initial_infections = pm.Normal(
            'log_initial_infections', 
            mu=np.log(np.maximum(cases[:seed_days], 1)), 
            sigma=1.0, 
            shape=seed_days
        )
        initial_infections = pm.Deterministic(
            'initial_infections', 
            pt.exp(log_initial_infections)
        )
        
        # Prior for Rt - assume it varies smoothly over time
        rt_raw = pm.GaussianRandomWalk(
            'rt_raw',
            mu=0,
            sigma=0.1,  # Controls smoothness of Rt over time
            shape=n_days - seed_days
        )
        
        # Transform to positive values with reasonable range
        rt = pm.Deterministic(
            'rt',
            pt.exp(rt_raw + np.log(1.0))  # Centered around 1.0
        )
        
        # Compute infections using renewal equation
        def compute_infections(rt_vals, initial_inf):
            infections = pt.zeros(n_days)
            infections = pt.set_subtensor(infections[:seed_days], initial_inf)
            
            for t in range(seed_days, n_days):
                # Compute convolution sum for renewal equation
                infection_sum = 0
                for s in range(1, min(max_gi + 1, t + 1)):
                    if t - s >= 0:
                        infection_sum += infections[t - s] * generation_interval[s - 1]
                
                infections = pt.set_subtensor(
                    infections[t], 
                    rt_vals[t - seed_days] * infection_sum
                )
            
            return infections
        
        infections = compute_infections(rt, initial_infections)
        
        # Compute expected cases accounting for reporting delay
        def compute_expected_cases(infections_t):
            expected_cases = pt.zeros(n_days)
            
            for t in range(n_days):
                case_sum = 0
                for d in range(min(max_delay, t + 1)):
                    if t - d >= 0:
                        case_sum += infections_t[t - d] * reporting_delay[d]
                expected_cases = pt.set_subtensor(expected_cases[t], case_sum)
            
            return expected_cases
        
        expected_cases = compute_expected_cases(infections)
        
        # Observation model - Negative Binomial to handle overdispersion
        phi = pm.Gamma('phi', alpha=2, beta=0.1)  # Overdispersion parameter
        
        # Convert mean and phi to alpha, beta for NegativeBinomial
        alpha = expected_cases / phi
        
        observed_cases = pm.NegativeBinomial(
            'observed_cases',
            mu=expected_cases,
            alpha=alpha,
            observed=cases
        )
        
        # Store variables for easy access
        model.add_coord('time', values=range(n_days), mutable=True)
        model.add_coord('rt_time', values=range(seed_days, n_days), mutable=True)
    
    return model, seed_days

def fit_model(model, draws=1000, tune=1000, chains=2):
    """Fit the PyMC model"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=2,
            target_accept=0.95,
            random_seed=42,
            return_inferencedata=True
        )
    
    return trace

def extract_rt_estimates(trace, dates, seed_days):
    """Extract Rt estimates from the trace"""
    rt_samples = trace.posterior['rt'].values  # shape: (chains, draws, time_points)
    rt_samples_flat = rt_samples.reshape(-1, rt_samples.shape[-1])  # flatten chains and draws
    
    # Calculate summary statistics
    rt_mean = np.mean(rt_samples_flat, axis=0)
    rt_median = np.median(rt_samples_flat, axis=0)
    rt_lower = np.percentile(rt_samples_flat, 2.5, axis=0)
    rt_upper = np.percentile(rt_samples_flat, 97.5, axis=0)
    rt_lower_50 = np.percentile(rt_samples_flat, 25, axis=0)
    rt_upper_50 = np.percentile(rt_samples_flat, 75, axis=0)
    
    # Create results dataframe
    rt_dates = dates[seed_days:]  # Rt estimates start after seeding period
    
    results_df = pd.DataFrame({
        'date': rt_dates,
        'rt_mean': rt_mean,
        'rt_median': rt_median,
        'rt_lower_95': rt_lower,
        'rt_upper_95': rt_upper,
        'rt_lower_50': rt_lower_50,
        'rt_upper_50': rt_upper_50
    })
    
    return results_df

def plot_results(df, rt_results, cases):
    """Plot the results"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cases over time
    ax1.bar(df['date'], df['cases'], alpha=0.7, color='steelblue', label='Observed cases')
    ax1.set_ylabel('Daily cases')
    ax1.set_title('Daily COVID-19 Cases')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates
    ax2.fill_between(
        rt_results['date'], 
        rt_results['rt_lower_95'], 
        rt_results['rt_upper_95'],
        alpha=0.3, color='red', label='95% CI'
    )
    ax2.fill_between(
        rt_results['date'], 
        rt_results['rt_lower_50'], 
        rt_results['rt_upper_50'],
        alpha=0.5, color='red', label='50% CI'
    )
    ax2.plot(rt_results['date'], rt_results['rt_median'], 'r-', linewidth=2, label='Median Rt')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Rt = 1')
    ax2.set_ylabel('Reproduction number (Rt)')
    ax2.set_xlabel('Date')
    ax2.set_title('Time-varying Reproduction Number (Rt)')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics
    print("\nRt Summary Statistics:")
    print(f"Mean Rt: {rt_results['rt_mean'].mean():.2f}")
    print(f"Median Rt: {rt_results['rt_median'].median():.2f}")
    print(f"Min Rt (median): {rt_results['rt_median'].min():.2f}")
    print(f"Max Rt (median): {rt_results['rt_median'].max():.2f}")
    
    # Days with Rt > 1
    days_above_1 = (rt_results['rt_median'] > 1).sum()
    total_days = len(rt_results)
    print(f"Days with Rt > 1: {days_above_1}/{total_days} ({100*days_above_1/total_days:.1f}%)")

def main():
    """Main analysis pipeline"""
    print("=== COVID-19 Rt Estimation using Renewal Equation ===\n")
    
    # Load data
    df = load_and_prepare_data('cases.csv')
    cases = df['cases'].values
    
    # Get generation interval and reporting delay distributions
    print("\n" + "="*50)
    print("Setting up epidemiological parameters...")
    generation_interval = get_generation_interval_pmf()
    reporting_delay = get_reporting_delay_pmf()
    
    # Create and fit model
    print("\n" + "="*50)
    print("Creating PyMC model...")
    model, seed_days = create_rt_model(cases, generation_interval, reporting_delay)
    
    print(f"Model created. Seeding period: {seed_days} days")
    print("Model summary:")
    print(model)
    
    print("\n" + "="*50)
    print("Fitting model (this may take several minutes)...")
    trace = fit_model(model, draws=1000, tune=1000, chains=2)
    
    # Extract results
    print("\n" + "="*50)
    print("Extracting Rt estimates...")
    rt_results = extract_rt_estimates(trace, df['date'].values, seed_days)
    
    # Display results
    print("\nFirst 10 Rt estimates:")
    print(rt_results.head(10).round(3))
    
    # Plot results
    print("\n" + "="*50)
    print("Generating plots...")
    plot_results(df, rt_results, cases)
    
    # Save results
    rt_results.to_csv('rt_estimates.csv', index=False)
    print("\nResults saved to 'rt_estimates.csv'")
    
    return rt_results, trace, model

# Example of creating sample data if cases.csv doesn't exist
def create_sample_data():
    """Create sample COVID-19 case data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', periods=100, freq='D')
    
    # Simulate epidemic curve with changing Rt
    true_rt = np.concatenate([
        np.linspace(2.5, 0.8, 30),  # Initial decline
        np.linspace(0.8, 1.3, 20),  # Increase
        np.linspace(1.3, 0.7, 30),  # Decline
        np.linspace(0.7, 1.1, 20)   # Slight increase
    ])
    
    # Simple simulation
    cases = []
    infections = [10] * 10  # Initial seeding
    
    gi_pmf = get_generation_interval_pmf()
    
    for t in range(len(dates)):
        if t < 10:
            new_infections = infections[t]
        else:
            infectiousness = sum(infections[max(0, t-s-1)] * gi_pmf[s] 
                               for s in range(min(len(gi_pmf), t)))
            new_infections = np.random.poisson(true_rt[min(t-10, len(true_rt)-1)] * infectiousness)
            infections.append(new_infections)
        
        # Add reporting noise and delay
        reported_cases = max(0, int(new_infections * np.random.uniform(0.7, 1.3)))
        cases.append(reported_cases)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases
    })
    
    df.to_csv('cases.csv', index=False)
    print("Sample data created in 'cases.csv'")
    return df

if __name__ == "__main__":
    # Uncomment the next line if you need to create sample data
    # create_sample_data()
    
    try:
        rt_results, trace, model = main()
    except FileNotFoundError:
        print("cases.csv not found. Creating sample data...")
        create_sample_data()
        rt_results, trace, model = main()

