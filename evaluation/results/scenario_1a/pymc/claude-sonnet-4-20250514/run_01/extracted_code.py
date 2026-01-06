import pandas as pd
import numpy as np
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

def load_and_prepare_data(filename='cases.csv'):
    """
    Load and prepare the case data for analysis
    """
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Remove any negative cases and fill NaNs with 0
    df['cases'] = df['cases'].fillna(0)
    df['cases'] = np.maximum(df['cases'], 0)
    
    return df

def get_generation_time_pmf(max_days=20):
    """
    Generate a discrete generation time distribution.
    Using a gamma distribution with mean ~5.1 days and std ~2.3 days
    (typical values for COVID-19 from literature)
    """
    # Parameters for COVID-19 generation time (Gamma distribution)
    mean_gt = 5.1
    std_gt = 2.3
    
    # Convert to shape and scale parameters
    shape = (mean_gt / std_gt) ** 2
    scale = std_gt ** 2 / mean_gt
    
    # Create discrete PMF
    days = np.arange(1, max_days + 1)
    pmf = stats.gamma.pdf(days, a=shape, scale=scale)
    pmf = pmf / pmf.sum()  # Normalize to sum to 1
    
    return pmf

def estimate_rt_pymc(cases, generation_pmf, n_samples=2000, n_tune=1000):
    """
    Estimate time-varying Rt using PyMC with a renewal equation approach
    """
    n_days = len(cases)
    gt_len = len(generation_pmf)
    
    # Smooth the cases slightly to avoid issues with zeros
    smoothed_cases = np.maximum(cases + 0.1, 1.0)
    
    with pm.Model() as model:
        # Prior for initial Rt
        rt_log_initial = pm.Normal('rt_log_initial', mu=np.log(1.0), sigma=0.5)
        
        # Random walk for log(Rt)
        rt_log_innovations = pm.Normal('rt_log_innovations', 
                                     mu=0, sigma=0.1, 
                                     shape=n_days-1)
        
        # Cumulative sum to get log(Rt) time series
        rt_log = pt.concatenate([[rt_log_initial], 
                                rt_log_initial + pt.cumsum(rt_log_innovations)])
        
        # Transform to get Rt
        rt = pm.Deterministic('rt', pt.exp(rt_log))
        
        # Calculate expected cases using renewal equation
        def calculate_infections(rt_vec, cases_obs):
            infections = pt.zeros(n_days)
            
            # Initialize first few days with observed cases
            init_days = min(7, n_days)
            infections = pt.set_subtensor(infections[:init_days], 
                                        cases_obs[:init_days])
            
            # Calculate subsequent infections using renewal equation
            for t in range(init_days, n_days):
                # Calculate infectiousness (convolution with generation time)
                start_idx = max(0, t - gt_len)
                relevant_infections = infections[start_idx:t]
                relevant_gt = generation_pmf[-(t-start_idx):]
                
                if len(relevant_infections) > 0 and len(relevant_gt) > 0:
                    # Ensure arrays have the same length
                    min_len = min(len(relevant_infections), len(relevant_gt))
                    infectiousness = pt.dot(relevant_infections[-min_len:], 
                                          relevant_gt[-min_len:])
                    
                    new_infections = rt_vec[t] * infectiousness
                    infections = pt.set_subtensor(infections[t], new_infections)
            
            return infections
        
        # Calculate expected infections
        expected_infections = calculate_infections(rt, smoothed_cases)
        
        # Add small constant to avoid numerical issues
        expected_cases = pm.math.maximum(expected_infections, 0.1)
        
        # Observation model - Negative Binomial to handle overdispersion
        alpha = pm.Exponential('alpha', 1.0)  # Overdispersion parameter
        
        # Likelihood
        obs = pm.NegativeBinomial('obs', 
                                mu=expected_cases, 
                                alpha=alpha,
                                observed=cases)
        
        # Sample from posterior
        trace = pm.sample(draws=n_samples, 
                         tune=n_tune,
                         target_accept=0.95,
                         random_seed=42,
                         return_inferencedata=True)
    
    return trace, model

def summarize_rt_estimates(trace, dates):
    """
    Extract and summarize Rt estimates
    """
    rt_samples = trace.posterior.rt.values
    
    # Reshape to (n_samples, n_days)
    rt_samples = rt_samples.reshape(-1, rt_samples.shape[-1])
    
    # Calculate summary statistics
    rt_mean = np.mean(rt_samples, axis=0)
    rt_median = np.median(rt_samples, axis=0)
    rt_lower = np.percentile(rt_samples, 2.5, axis=0)
    rt_upper = np.percentile(rt_samples, 97.5, axis=0)
    rt_lower_50 = np.percentile(rt_samples, 25, axis=0)
    rt_upper_50 = np.percentile(rt_samples, 75, axis=0)
    
    # Create summary dataframe
    summary_df = pd.DataFrame({
        'date': dates,
        'rt_mean': rt_mean,
        'rt_median': rt_median,
        'rt_lower_95': rt_lower,
        'rt_upper_95': rt_upper,
        'rt_lower_50': rt_lower_50,
        'rt_upper_50': rt_upper_50
    })
    
    return summary_df, rt_samples

def plot_results(df, rt_summary, cases):
    """
    Create comprehensive plots of the results
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Cases over time
    axes[0].bar(df['date'], cases, alpha=0.6, color='steelblue', 
                label='Observed cases')
    axes[0].set_ylabel('Daily Cases')
    axes[0].set_title('Daily COVID-19 Cases')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=45)
    
    # Plot 2: Rt estimates
    axes[1].fill_between(rt_summary['date'], 
                        rt_summary['rt_lower_95'], 
                        rt_summary['rt_upper_95'],
                        alpha=0.2, color='red', label='95% CI')
    axes[1].fill_between(rt_summary['date'], 
                        rt_summary['rt_lower_50'], 
                        rt_summary['rt_upper_50'],
                        alpha=0.4, color='red', label='50% CI')
    axes[1].plot(rt_summary['date'], rt_summary['rt_median'], 
                color='red', linewidth=2, label='Rt (median)')
    axes[1].axhline(y=1.0, color='black', linestyle='--', alpha=0.7, 
                   label='Rt = 1')
    axes[1].set_ylabel('Reproduction Number (Rt)')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Time-varying Reproduction Number (Rt)')
    axes[1].legend()
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylim(0, max(3, rt_summary['rt_upper_95'].max() * 1.1))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """
    Main function to run the complete analysis
    """
    print("Loading and preparing data...")
    df = load_and_prepare_data('cases.csv')
    cases = df['cases'].values.astype(float)
    dates = df['date'].values
    
    print(f"Data loaded: {len(cases)} days of case data")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    print(f"Total cases: {cases.sum():.0f}")
    
    # Get generation time distribution
    print("Setting up generation time distribution...")
    generation_pmf = get_generation_time_pmf()
    
    print(f"Generation time - Mean: {np.sum(np.arange(1, len(generation_pmf)+1) * generation_pmf):.2f} days")
    
    # Estimate Rt
    print("Estimating Rt using PyMC...")
    print("This may take a few minutes...")
    
    try:
        trace, model = estimate_rt_pymc(cases, generation_pmf)
        
        print("Sampling completed successfully!")
        print(f"Sampling summary:")
        print(f"  Effective sample size: {pm.ess(trace).rt.values.min():.0f}")
        print(f"  R-hat max: {pm.rhat(trace).rt.values.max():.3f}")
        
    except Exception as e:
        print(f"Error in sampling: {e}")
        return None, None
    
    # Summarize results
    print("Summarizing Rt estimates...")
    rt_summary, rt_samples = summarize_rt_estimates(trace, dates)
    
    # Print key results
    print(f"\nRt Summary Statistics:")
    print(f"  Mean Rt: {rt_summary['rt_mean'].mean():.2f}")
    print(f"  Median Rt: {rt_summary['rt_median'].median():.2f}")
    print(f"  Days with Rt > 1: {(rt_summary['rt_median'] > 1).sum()}")
    print(f"  Final Rt estimate: {rt_summary['rt_median'].iloc[-1]:.2f} "
          f"({rt_summary['rt_lower_95'].iloc[-1]:.2f}-{rt_summary['rt_upper_95'].iloc[-1]:.2f})")
    
    # Create plots
    print("Creating plots...")
    fig = plot_results(df, rt_summary, cases)
    
    # Save results
    rt_summary.to_csv('rt_estimates.csv', index=False)
    print("Results saved to 'rt_estimates.csv'")
    
    return rt_summary, trace

# Example of creating sample data if cases.csv doesn't exist
def create_sample_data():
    """
    Create sample case data for demonstration
    """
    np.random.seed(42)
    dates = pd.date_range('2020-03-01', periods=100, freq='D')
    
    # Simulate epidemic curve with changing Rt
    rt_true = np.concatenate([
        np.linspace(2.5, 1.5, 30),  # Initial growth
        np.linspace(1.5, 0.8, 40),  # Decline due to interventions
        np.linspace(0.8, 1.2, 30)   # Slight increase
    ])
    
    cases = []
    cases.append(10)  # Initial cases
    
    generation_pmf = get_generation_time_pmf()
    
    for t in range(1, len(dates)):
        # Simple renewal equation for simulation
        infectiousness = 0
        for s, gt_prob in enumerate(generation_pmf):
            if t - s - 1 >= 0:
                infectiousness += cases[t - s - 1] * gt_prob
        
        expected_cases = rt_true[t] * infectiousness
        new_cases = np.random.poisson(max(expected_cases, 0))
        cases.append(new_cases)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases
    })
    
    df.to_csv('cases.csv', index=False)
    print("Sample data created and saved to 'cases.csv'")
    
    return df

if __name__ == "__main__":
    # Create sample data if file doesn't exist
    import os
    if not os.path.exists('cases.csv'):
        print("Creating sample data...")
        create_sample_data()
    
    # Run the main analysis
    rt_estimates, trace = main()

