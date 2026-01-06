import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_prepare_data(filepath):
    """Load and prepare the observation data"""
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Handle any missing values
    df = df.fillna(0)
    
    print(f"Data loaded: {len(df)} days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Total cases: {df['cases'].sum()}")
    print(f"Total hospitalisations: {df['hospitalisations'].sum()}")
    print(f"Total deaths: {df['deaths'].sum()}")
    
    return df

def create_delay_distributions():
    """Create delay distributions for each data stream"""
    # Infection to symptom onset: ~5 days
    # Symptom onset to case report: ~2 days
    # Total infection to case: ~7 days
    case_delay_mean = 7
    case_delay_sd = 3
    
    # Infection to hospitalisation: ~10-12 days
    hosp_delay_mean = 11
    hosp_delay_sd = 4
    
    # Infection to death: ~18-21 days
    death_delay_mean = 20
    death_delay_sd = 5
    
    max_delay = 30
    
    delays = {}
    
    for name, mean, sd in [
        ('cases', case_delay_mean, case_delay_sd),
        ('hospitalisations', hosp_delay_mean, hosp_delay_sd),
        ('deaths', death_delay_mean, death_delay_sd)
    ]:
        # Use gamma distribution truncated at max_delay
        shape = (mean / sd) ** 2
        scale = sd ** 2 / mean
        
        pmf = np.array([stats.gamma.pdf(i, a=shape, scale=scale) for i in range(1, max_delay + 1)])
        pmf = pmf / pmf.sum()  # Normalize
        
        delays[name] = pmf
    
    return delays, max_delay

def create_generation_interval():
    """Create generation interval distribution"""
    # Mean ~5.5 days, SD ~2.5 days for COVID-19
    gen_mean = 5.5
    gen_sd = 2.5
    max_gen = 15
    
    shape = (gen_mean / gen_sd) ** 2
    scale = gen_sd ** 2 / gen_mean
    
    pmf = np.array([stats.gamma.pdf(i, a=shape, scale=scale) for i in range(1, max_gen + 1)])
    pmf = pmf / pmf.sum()
    
    return pmf

def build_renewal_model(observations, delays, generation_interval, max_delay):
    """Build the joint renewal equation model"""
    
    n_days = len(observations)
    streams = ['cases', 'hospitalisations', 'deaths']
    
    # Convert observations to arrays
    obs_data = {stream: observations[stream].values for stream in streams}
    
    with pm.Model() as model:
        # === Priors ===
        
        # Initial infections (seeding period)
        seed_days = max_delay
        log_initial_infections = pm.Normal('log_initial_infections', mu=np.log(50), sigma=1, 
                                         shape=seed_days)
        initial_infections = pm.Deterministic('initial_infections', 
                                            pt.exp(log_initial_infections))
        
        # Rt evolution - using random walk on log scale for smoothness
        log_rt_init = pm.Normal('log_rt_init', mu=np.log(1.5), sigma=0.5)
        log_rt_innovations = pm.Normal('log_rt_innovations', mu=0, sigma=0.1, 
                                     shape=n_days - 1)
        
        log_rt = pm.Deterministic('log_rt', pt.concatenate([
            [log_rt_init],
            log_rt_init + pt.cumsum(log_rt_innovations)
        ]))
        
        rt = pm.Deterministic('rt', pt.exp(log_rt))
        
        # Stream-specific parameters
        ascertainment_rates = {}
        overdispersion_params = {}
        
        for stream in streams:
            # Ascertainment rates (logit scale for constraint to [0,1])
            if stream == 'cases':
                # Cases might have higher ascertainment
                ascertainment_rates[stream] = pm.Beta(f'ascertainment_{stream}', 
                                                    alpha=2, beta=3)
            elif stream == 'hospitalisations':
                # Lower ascertainment but more stable
                ascertainment_rates[stream] = pm.Beta(f'ascertainment_{stream}', 
                                                    alpha=1, beta=10)
            else:  # deaths
                # Lowest ascertainment but most complete
                ascertainment_rates[stream] = pm.Beta(f'ascertainment_{stream}', 
                                                    alpha=1, beta=20)
            
            # Overdispersion parameters (inverse dispersion)
            overdispersion_params[stream] = pm.Exponential(f'phi_{stream}', 1.0)
        
        # === Renewal equation dynamics ===
        
        def renewal_step(rt_t, infections_history):
            """Single step of renewal equation"""
            # infections_history contains last len(generation_interval) infections
            new_infections = rt_t * pt.dot(infections_history, generation_interval[::-1])
            return new_infections
        
        # Set up the renewal equation recursion
        infections_extended = pt.concatenate([initial_infections, 
                                            pt.zeros(n_days)])
        
        # Apply renewal equation for each day
        for t in range(n_days):
            start_idx = t + seed_days - len(generation_interval)
            end_idx = t + seed_days
            
            # Ensure we have enough history
            if start_idx >= 0:
                infection_history = infections_extended[start_idx:end_idx]
                # Pad if necessary
                if len(infection_history) < len(generation_interval):
                    padding = pt.zeros(len(generation_interval) - len(infection_history))
                    infection_history = pt.concatenate([padding, infection_history])
                
                new_infection = rt[t] * pt.dot(infection_history, generation_interval[::-1])
                infections_extended = pt.set_subtensor(infections_extended[seed_days + t], 
                                                     new_infection)
        
        infections = infections_extended[seed_days:]
        
        # === Observation model ===
        
        expected_obs = {}
        
        for stream in streams:
            delay_pmf = delays[stream]
            
            # Convolve infections with delay distribution
            expected_stream = pt.zeros(n_days)
            
            for t in range(n_days):
                total = 0.0
                for d, delay_prob in enumerate(delay_pmf):
                    infection_day = t - (d + 1)  # d+1 because delays start from day 1
                    if infection_day >= 0:
                        total += infections[infection_day] * delay_prob
                
                expected_stream = pt.set_subtensor(expected_stream[t], total)
            
            # Apply ascertainment
            expected_obs[stream] = expected_stream * ascertainment_rates[stream]
            
            # Observation likelihood with overdispersion (Negative Binomial)
            pm.NegativeBinomial(
                f'obs_{stream}',
                mu=expected_obs[stream],
                alpha=overdispersion_params[stream],
                observed=obs_data[stream]
            )
        
        # Store expected observations for diagnostics
        for stream in streams:
            pm.Deterministic(f'expected_{stream}', expected_obs[stream])
    
    return model

def fit_model_and_extract_results(model, observations):
    """Fit the model and extract results"""
    
    with model:
        # Use NUTS sampler
        print("Starting MCMC sampling...")
        
        # Initial tuning
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=2,
            cores=2,
            target_accept=0.95,
            random_seed=42,
            progressbar=True
        )
        
        print("Sampling completed!")
        
        # Extract posterior summaries
        summary = pm.summary(trace, var_names=['rt'])
        
        # Extract Rt estimates
        rt_samples = trace.posterior['rt'].values  # shape: (chains, draws, time)
        rt_mean = rt_samples.mean(axis=(0, 1))
        rt_lower = np.percentile(rt_samples, 2.5, axis=(0, 1))
        rt_upper = np.percentile(rt_samples, 97.5, axis=(0, 1))
        
        # Extract ascertainment rates
        ascertainment = {}
        for stream in ['cases', 'hospitalisations', 'deaths']:
            samples = trace.posterior[f'ascertainment_{stream}'].values
            ascertainment[stream] = {
                'mean': samples.mean(),
                'lower': np.percentile(samples, 2.5),
                'upper': np.percentile(samples, 97.5)
            }
        
        # Create results dataframe
        results_df = observations.copy()
        results_df['rt_mean'] = rt_mean
        results_df['rt_lower'] = rt_lower
        results_df['rt_upper'] = rt_upper
        
        # Add expected observations
        for stream in ['cases', 'hospitalisations', 'deaths']:
            expected_samples = trace.posterior[f'expected_{stream}'].values
            results_df[f'expected_{stream}_mean'] = expected_samples.mean(axis=(0, 1))
            results_df[f'expected_{stream}_lower'] = np.percentile(expected_samples, 2.5, axis=(0, 1))
            results_df[f'expected_{stream}_upper'] = np.percentile(expected_samples, 97.5, axis=(0, 1))
    
    return trace, results_df, ascertainment

def plot_results(results_df, ascertainment):
    """Create comprehensive plots of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Joint Rt Estimation from Multiple Data Streams', fontsize=16, fontweight='bold')
    
    # Plot 1: Rt over time
    ax = axes[0, 0]
    ax.plot(results_df['date'], results_df['rt_mean'], 'b-', linewidth=2, label='Rt estimate')
    ax.fill_between(results_df['date'], results_df['rt_lower'], results_df['rt_upper'], 
                    alpha=0.3, color='blue', label='95% CI')
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax.set_ylabel('Reproduction Number (Rt)')
    ax.set_title('Time-varying Reproduction Number')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cases
    ax = axes[0, 1]
    ax.plot(results_df['date'], results_df['cases'], 'ko-', markersize=3, alpha=0.7, label='Observed')
    ax.plot(results_df['date'], results_df['expected_cases_mean'], 'r-', linewidth=2, label='Expected')
    ax.fill_between(results_df['date'], results_df['expected_cases_lower'], 
                    results_df['expected_cases_upper'], alpha=0.3, color='red')
    ax.set_ylabel('Cases')
    ax.set_title('Cases: Observed vs Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Hospitalisations
    ax = axes[1, 0]
    ax.plot(results_df['date'], results_df['hospitalisations'], 'ko-', markersize=3, alpha=0.7, label='Observed')
    ax.plot(results_df['date'], results_df['expected_hospitalisations_mean'], 'g-', linewidth=2, label='Expected')
    ax.fill_between(results_df['date'], results_df['expected_hospitalisations_lower'], 
                    results_df['expected_hospitalisations_upper'], alpha=0.3, color='green')
    ax.set_ylabel('Hospitalisations')
    ax.set_title('Hospitalisations: Observed vs Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Deaths
    ax = axes[1, 1]
    ax.plot(results_df['date'], results_df['deaths'], 'ko-', markersize=3, alpha=0.7, label='Observed')
    ax.plot(results_df['date'], results_df['expected_deaths_mean'], 'm-', linewidth=2, label='Expected')
    ax.fill_between(results_df['date'], results_df['expected_deaths_lower'], 
                    results_df['expected_deaths_upper'], alpha=0.3, color='magenta')
    ax.set_ylabel('Deaths')
    ax.set_title('Deaths: Observed vs Expected')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels
    for ax in axes.flat:
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Plot ascertainment rates
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    streams = ['cases', 'hospitalisations', 'deaths']
    means = [ascertainment[stream]['mean'] for stream in streams]
    lowers = [ascertainment[stream]['lower'] for stream in streams]
    uppers = [ascertainment[stream]['upper'] for stream in streams]
    
    x_pos = np.arange(len(streams))
    ax.bar(x_pos, means, yerr=[np.array(means) - np.array(lowers), 
                               np.array(uppers) - np.array(means)], 
           capsize=5, alpha=0.7, color=['blue', 'green', 'magenta'])
    
    ax.set_xlabel('Data Stream')
    ax.set_ylabel('Ascertainment Rate')
    ax.set_title('Estimated Ascertainment Rates by Data Stream')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(streams)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    
    print("=== Joint Rt Estimation from Multiple Data Streams ===\n")
    
    # Load data
    print("1. Loading data...")
    observations = load_and_prepare_data('observations.csv')
    
    # Create delay distributions
    print("\n2. Setting up delay distributions...")
    delays, max_delay = create_delay_distributions()
    generation_interval = create_generation_interval()
    
    print(f"Maximum delay considered: {max_delay} days")
    print(f"Generation interval length: {len(generation_interval)} days")
    
    # Build model
    print("\n3. Building Bayesian model...")
    model = build_renewal_model(observations, delays, generation_interval, max_delay)
    
    print(f"Model built with {len(observations)} time points")
    
    # Fit model
    print("\n4. Fitting model...")
    trace, results_df, ascertainment = fit_model_and_extract_results(model, observations)
    
    # Display results
    print("\n5. Results Summary:")
    print("="*50)
    
    print("\nAscertainment Rates (95% CI):")
    for stream, params in ascertainment.items():
        print(f"{stream:15}: {params['mean']:.3f} ({params['lower']:.3f}, {params['upper']:.3f})")
    
    print(f"\nRt Summary:")
    print(f"Mean Rt: {results_df['rt_mean'].mean():.2f}")
    print(f"Min Rt:  {results_df['rt_mean'].min():.2f}")
    print(f"Max Rt:  {results_df['rt_mean'].max():.2f}")
    
    # Show periods where Rt > 1
    above_one = results_df[results_df['rt_mean'] > 1]
    print(f"\nPeriods with Rt > 1: {len(above_one)} days ({100*len(above_one)/len(results_df):.1f}%)")
    
    # Plot results
    print("\n6. Generating plots...")
    plot_results(results_df, ascertainment)
    
    print("\nAnalysis completed successfully!")
    
    return model, trace, results_df, ascertainment

if __name__ == "__main__":
    # For demonstration, create sample data if file doesn't exist
    try:
        pd.read_csv('observations.csv')
    except FileNotFoundError:
        print("Creating sample data for demonstration...")
        
        # Generate synthetic data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        # Simulate an outbreak with changing Rt
        true_rt = 2.0 * np.exp(-0.03 * np.arange(100)) + 0.5
        
        infections = np.zeros(100)
        infections[0] = 50
        
        for t in range(1, 100):
            infections[t] = max(1, np.random.poisson(true_rt[t] * infections[max(0, t-7):t].sum() * 0.2))
        
        # Add delays and noise
        cases = np.random.negative_binomial(n=5, p=5/(5 + infections * 0.3))
        hospitalisations = np.random.negative_binomial(n=10, p=10/(10 + infections * 0.05))
        deaths = np.random.negative_binomial(n=20, p=20/(20 + infections * 0.01))
        
        # Apply delays
        cases = np.convolve(cases, [0.1, 0.3, 0.4, 0.2])[:100]
        hospitalisations = np.convolve(hospitalisations, [0.05, 0.15, 0.3, 0.3, 0.2])[:100]
        deaths = np.convolve(deaths, [0.02, 0.08, 0.2, 0.4, 0.2, 0.1])[:100]
        
        sample_data = pd.DataFrame({
            'date': dates,
            'cases': np.maximum(0, cases).astype(int),
            'hospitalisations': np.maximum(0, hospitalisations).astype(int),
            'deaths': np.maximum(0, deaths).astype(int)
        })
        
        sample_data.to_csv('observations.csv', index=False)
        print("Sample data created!\n")
    
    # Run the main analysis
    model, trace, results_df, ascertainment = main()

