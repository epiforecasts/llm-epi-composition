I'll create a Bayesian model to jointly estimate Rt from multiple data streams using PyMC. Here's the complete solution:

```python
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_data(filepath="observations.csv"):
    """Load and prepare the observation data"""
    try:
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except FileNotFoundError:
        # Generate synthetic data for demonstration
        print("Data file not found. Generating synthetic data for demonstration...")
        return generate_synthetic_data()

def generate_synthetic_data():
    """Generate synthetic data that mimics real epidemic dynamics"""
    np.random.seed(42)
    n_days = 150
    dates = pd.date_range('2023-01-01', periods=n_days)
    
    # True Rt trajectory (varies over time)
    time = np.arange(n_days)
    true_rt = 1.5 * np.exp(-time/40) + 0.5 + 0.3 * np.sin(time/20)
    
    # Generation interval (gamma distribution)
    gen_shape, gen_scale = 2.5, 2.5
    max_gen = 20
    gen_pmf = stats.gamma.pdf(np.arange(1, max_gen+1), gen_shape, scale=gen_scale)
    gen_pmf = gen_pmf / gen_pmf.sum()
    
    # Simulate infections using renewal equation
    infections = np.zeros(n_days)
    infections[:7] = 50  # Initial seeding
    
    for t in range(7, n_days):
        if t < max_gen:
            past_inf = infections[:t][::-1]
            gen_subset = gen_pmf[:t]
        else:
            past_inf = infections[t-max_gen:t][::-1]
            gen_subset = gen_pmf
        
        lambda_t = true_rt[t] * np.sum(past_inf * gen_subset)
        infections[t] = np.random.poisson(max(lambda_t, 1))
    
    # Delay distributions (log-normal)
    def create_delay_pmf(mean_delay, std_delay, max_delay=30):
        delays = np.arange(1, max_delay+1)
        pmf = stats.lognorm.pdf(delays, s=std_delay, scale=np.exp(np.log(mean_delay)))
        return pmf / pmf.sum()
    
    case_delay_pmf = create_delay_pmf(5, 0.5)
    hosp_delay_pmf = create_delay_pmf(12, 0.6)
    death_delay_pmf = create_delay_pmf(20, 0.7)
    
    # Ascertainment rates
    case_asc = 0.3
    hosp_asc = 0.15
    death_asc = 0.02
    
    # Generate observations with overdispersion
    def generate_observations(infections, delay_pmf, ascertainment, overdispersion=5):
        n_days = len(infections)
        max_delay = len(delay_pmf)
        obs = np.zeros(n_days)
        
        for t in range(n_days):
            expected = 0
            for d in range(min(t, max_delay)):
                expected += infections[t-d-1] * delay_pmf[d]
            
            expected *= ascertainment
            if expected > 0:
                # Negative binomial with overdispersion
                p = expected / (expected + overdispersion)
                obs[t] = np.random.negative_binomial(overdispersion, 1-p)
        
        return obs.astype(int)
    
    cases = generate_observations(infections, case_delay_pmf, case_asc)
    hospitalizations = generate_observations(infections, hosp_delay_pmf, hosp_asc)
    deaths = generate_observations(infections, death_delay_pmf, death_asc)
    
    df = pd.DataFrame({
        'date': dates,
        'cases': cases,
        'hospitalisations': hospitalizations,
        'deaths': deaths
    })
    
    return df

def create_generation_interval(shape=2.5, scale=2.5, max_gen=20):
    """Create generation interval PMF"""
    gen_pmf = stats.gamma.pdf(np.arange(1, max_gen+1), shape, scale=scale)
    return gen_pmf / gen_pmf.sum()

def create_delay_distributions():
    """Create delay distribution PMFs for each stream"""
    max_delay = 30
    delays = np.arange(1, max_delay+1)
    
    # Stream-specific delay parameters (mean, std)
    delay_params = {
        'cases': (5, 0.5),
        'hospitalisations': (12, 0.6), 
        'deaths': (20, 0.7)
    }
    
    delay_pmfs = {}
    for stream, (mean_delay, std_delay) in delay_params.items():
        pmf = stats.lognorm.pdf(delays, s=std_delay, scale=np.exp(np.log(mean_delay)))
        delay_pmfs[stream] = pmf / pmf.sum()
    
    return delay_pmfs

def build_model(data, generation_pmf, delay_pmfs):
    """Build the joint PyMC model"""
    n_days = len(data)
    max_gen = len(generation_pmf)
    max_delay = len(list(delay_pmfs.values())[0])
    
    # Prepare observation arrays
    obs_cases = data['cases'].values
    obs_hosp = data['hospitalisations'].values  
    obs_deaths = data['deaths'].values
    
    with pm.Model() as model:
        # Rt prior - log-normal with smoothness constraint
        log_rt_raw = pm.GaussianRandomWalk(
            'log_rt_raw', 
            sigma=0.1,  # Controls smoothness
            shape=n_days,
            init_dist=pm.Normal.dist(mu=np.log(1.0), sigma=0.3)
        )
        rt = pm.Deterministic('rt', pt.exp(log_rt_raw))
        
        # Initial infections (seeding period)
        seed_days = 7
        initial_infections = pm.Exponential('initial_infections', lam=1/50, shape=seed_days)
        
        # Stream-specific ascertainment rates
        asc_cases = pm.Beta('asc_cases', alpha=3, beta=7)  # Prior centered around 0.3
        asc_hosp = pm.Beta('asc_hosp', alpha=2, beta=10)   # Prior centered around 0.15  
        asc_deaths = pm.Beta('asc_deaths', alpha=1, beta=30) # Prior centered around 0.03
        
        # Overdispersion parameters (negative binomial)
        phi_cases = pm.Exponential('phi_cases', lam=0.1)
        phi_hosp = pm.Exponential('phi_hosp', lam=0.1)
        phi_deaths = pm.Exponential('phi_deaths', lam=0.1)
        
        # Infection dynamics using renewal equation
        def renewal_step(t, infections_prev, rt_t):
            # Get relevant past infections and generation intervals
            if t < max_gen:
                past_infections = infections_prev[:t]
                gen_weights = generation_pmf[:t]
            else:
                past_infections = infections_prev[t-max_gen:t]
                gen_weights = generation_pmf
            
            # Renewal equation
            lambda_t = rt_t * pt.sum(past_infections[::-1] * gen_weights)
            return pt.maximum(lambda_t, 1.0)
        
        # Initialize infections array
        infections = pt.concatenate([initial_infections, pt.zeros(n_days - seed_days)])
        
        # Iteratively compute infections
        for t in range(seed_days, n_days):
            lambda_t = renewal_step(t, infections[:t], rt[t])
            infections = pt.set_subtensor(infections[t], lambda_t)
        
        infections = pm.Deterministic('infections', infections)
        
        # Convolution function for delay from infection to observation
        def convolve_delays(infections, delay_pmf, ascertainment):
            n_days_local = infections.shape[0]
            max_delay_local = delay_pmf.shape[0]
            expected_obs = pt.zeros(n_days_local)
            
            for t in range(n_days_local):
                daily_expected = 0
                for d in range(min(t, max_delay_local)):
                    daily_expected += infections[t-d-1] * delay_pmf[d]
                expected_obs = pt.set_subtensor(expected_obs[t], daily_expected * ascertainment)
            
            return expected_obs
        
        # Expected observations for each stream
        expected_cases = convolve_delays(infections, delay_pmfs['cases'], asc_cases)
        expected_hosp = convolve_delays(infections, delay_pmfs['hospitalisations'], asc_hosp)  
        expected_deaths = convolve_delays(infections, delay_pmfs['deaths'], asc_deaths)
        
        # Likelihood - Negative binomial for overdispersion
        pm.NegativeBinomial('obs_cases', mu=expected_cases, alpha=phi_cases, observed=obs_cases)
        pm.NegativeBinomial('obs_hosp', mu=expected_hosp, alpha=phi_hosp, observed=obs_hosp)
        pm.NegativeBinomial('obs_deaths', mu=expected_deaths, alpha=phi_deaths, observed=obs_deaths)
        
    return model

def fit_model(model, draws=1000, tune=1000, chains=2):
    """Fit the model using NUTS sampler"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=draws,
            tune=tune, 
            chains=chains,
            cores=1,
            target_accept=0.95,
            return_inferencedata=True
        )
    return trace

def extract_results(trace, data):
    """Extract and summarize results"""
    # Get posterior summaries
    summary = pm.summary(trace, var_names=['rt', 'asc_cases', 'asc_hosp', 'asc_deaths', 
                                          'phi_cases', 'phi_hosp', 'phi_deaths'])
    
    # Extract Rt estimates
    rt_samples = trace.posterior['rt'].values.reshape(-1, len(data))
    rt_mean = rt_samples.mean(axis=0)
    rt_lower = np.percentile(rt_samples, 2.5, axis=0)
    rt_upper = np.percentile(rt_samples, 97.5, axis=0)
    
    # Extract stream parameters
    stream_params = {
        'ascertainment_cases': trace.posterior['asc_cases'].values.flatten(),
        'ascertainment_hosp': trace.posterior['asc_hosp'].values.flatten(),
        'ascertainment_deaths': trace.posterior['asc_deaths'].values.flatten(),
        'overdispersion_cases': trace.posterior['phi_cases'].values.flatten(),
        'overdispersion_hosp': trace.posterior['phi_hosp'].values.flatten(), 
        'overdispersion_deaths': trace.posterior['phi_deaths'].values.flatten()
    }
    
    results = {
        'dates': data['date'].values,
        'rt_mean': rt_mean,
        'rt_lower': rt_lower,
        'rt_upper': rt_upper,
        'stream_params': stream_params,
        'summary': summary
    }
    
    return results

def plot_results(results, data):
    """Create plots showing the results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Rt over time
    ax1 = axes[0, 0]
    ax1.plot(results['dates'], results['rt_mean'], 'b-', linewidth=2, label='Rt (mean)')
    ax1.fill_between(results['dates'], results['rt_lower'], results['rt_upper'], 
                     alpha=0.3, color='blue', label='95% CI')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax1.set_ylabel('Reproduction Number (Rt)')
    ax1.set_title('Time-varying Reproduction Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Data streams
    ax2 = axes[0, 1]
    ax2.plot(data['date'], data['cases'], 'o-', label='Cases', alpha=0.7)
    ax2.plot(data['date'], data['hospitalisations'], 's-', label='Hospitalizations', alpha=0.7)
    ax2.plot(data['date'], data['deaths'], '^-', label='Deaths', alpha=0.7)
    ax2.set_ylabel('Daily Observations')
    ax2.set_title('Observed Data Streams')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stream parameters - Ascertainment
    ax3 = axes[1, 0]
    asc_data = [
        results['stream_params']['ascertainment_cases'],
        results['stream_params']['ascertainment_hosp'], 
        results['stream_params']['ascertainment_deaths']
    ]
    bp = ax3.boxplot(asc_data, labels=['Cases', 'Hosp', 'Deaths'])
    ax3.set_ylabel('Ascertainment Rate')
    ax3.set_title('Stream-specific Ascertainment Rates')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Stream parameters - Overdispersion
    ax4 = axes[1, 1]
    phi_data = [
        results['stream_params']['overdispersion_cases'],
        results['stream_params']['overdispersion_hosp'],
        results['stream_params']['overdispersion_deaths']
    ]
    bp = ax4.boxplot(phi_data, labels=['Cases', 'Hosp', 'Deaths'])
    ax4.set_ylabel('Overdispersion Parameter')
    ax4.set_title('Stream-specific Overdispersion')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """Main execution function"""
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} days of data")
    
    print("\nData summary:")
    print(data.describe())
    
    print("\nCreating model components...")
    generation_pmf = create_generation_interval()
    delay_pmfs = create_delay_distributions()
    
    print("Building PyMC model...")
    model = build_model(data, generation_pmf, delay_pmfs)
    
    print("\nModel summary:")
    print(model)
    
    print("\nFitting model (this may take several minutes)...")
    try:
        trace = fit_model(model, draws=1000, tune=1000, chains=2)
        
        print("\nExtracting results...")
        results = extract_results(trace, data)
        
        print("\nModel Summary:")
        print(results['summary'])
        
        print("\nStream Parameter Estimates:")
        for param_name, samples in results['stream_params'].items():
            mean_val = np.mean(samples)
            ci_lower = np.percentile(samples, 2.5)
            ci_upper = np.percentile(samples, 97.5)
            print(f"{param_name}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        
        print("\nCreating plots...")
        fig = plot_results(results, data)
        
        print("\nAnalysis complete!")
        
        return results, trace, model
        
    except Exception as e:
        print(f"Error during model fitting: {e}")
        print("This could be due to model complexity or data issues.")
        return None, None, model

if __name__ == "__main__":
    results, trace, model = main()
```

This comprehensive solution provides:

## Key Features:

1. **Joint Estimation**: Uses a single shared Rt estimated from all three data streams simultaneously

2. **Renewal Equation**: Implements the renewal equation with generation interval for infection dynamics

3. **Stream-Specific Components**:
   - Different delay distributions for cases (5 days), hospitalizations (12 days), deaths (20 days)
   - Stream-specific ascertainment rates
   - Stream-specific overdispersion parameters

4. **Bayesian Framework**:
   - Smooth Rt evolution using Gaussian random walk
   - Proper uncertainty quantification
   - Negative binomial likelihoods for overdispersion

5. **Robust Implementation**:
   - Handles initial infection seeding
   - Includes synthetic data generation for testing
   - Comprehensive visualization and summarization

## Model Structure:

- **Rt**: Smooth time-varying reproduction number (shared across streams)
- **Infections**: Generated via renewal equation
- **Observations**: Stream-specific delays and ascertainment from infections
- **Overdispersion**: Negative binomial likelihoods account for extra variance

The model jointly leverages information from all three data streams while accounting for their different characteristics, providing a robust estimate of the underlying epidemic dynamics.