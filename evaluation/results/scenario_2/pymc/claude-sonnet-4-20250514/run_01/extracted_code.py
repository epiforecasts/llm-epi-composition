import pandas as pd
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gamma
import arviz as az
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def generate_sample_data():
    """Generate sample COVID-19 case data for demonstration"""
    # Set parameters
    n_days = 150
    start_date = pd.to_datetime('2020-03-01')
    
    # True Rt trajectory (decreasing then increasing)
    t = np.linspace(0, n_days-1, n_days)
    true_rt = 2.5 * np.exp(-t/30) + 0.8 + 0.3 * np.sin(t/20)
    
    # Generation interval (Gamma distribution, mean=5.1, sd=2.3)
    max_gen_int = 20
    gen_int_shape = 4.9
    gen_int_scale = 1.04
    gen_int = gamma.pdf(np.arange(1, max_gen_int+1), a=gen_int_shape, scale=gen_int_scale)
    gen_int = gen_int / gen_int.sum()
    
    # Reporting delay (mean=7 days)
    max_delay = 25
    delay_shape = 2.0
    delay_scale = 3.5
    reporting_delay = gamma.pdf(np.arange(0, max_delay), a=delay_shape, scale=delay_scale)
    reporting_delay = reporting_delay / reporting_delay.sum()
    
    # Simulate true infections
    infections = np.zeros(n_days)
    infections[0] = 100  # Initial infections
    
    for t in range(1, n_days):
        # Renewal equation
        infectiousness = 0
        for s in range(min(t, len(gen_int))):
            infectiousness += infections[t-1-s] * gen_int[s]
        infections[t] = true_rt[t] * infectiousness
    
    # Day-of-week effects (lower reporting on weekends)
    dow_effects = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5])  # Mon-Sun
    
    # Time-varying ascertainment (starts low, increases)
    ascertainment = 0.1 + 0.4 / (1 + np.exp(-(t - 50)/15))
    
    # Apply reporting delay and observation process
    expected_reports = np.zeros(n_days + max_delay)
    for t in range(n_days):
        for d in range(len(reporting_delay)):
            if t + d < len(expected_reports):
                expected_reports[t + d] += infections[t] * ascertainment[t] * reporting_delay[d]
    
    # Truncate to original period
    expected_reports = expected_reports[:n_days]
    
    # Apply day-of-week effects
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    dow = dates.dayofweek  # Monday=0, Sunday=6
    dow_multiplier = dow_effects[dow]
    expected_reports *= dow_multiplier
    
    # Add overdispersion (negative binomial)
    phi = 10  # Overdispersion parameter
    cases = np.random.negative_binomial(
        n=phi, 
        p=phi / (phi + expected_reports)
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'date': dates,
        'cases': cases,
        'day_of_week': dow + 1  # Convert to 1-7 (Monday=1)
    })
    
    # Save to CSV
    df.to_csv('cases.csv', index=False)
    
    # Return additional info for validation
    return df, {
        'true_rt': true_rt,
        'true_infections': infections,
        'true_ascertainment': ascertainment,
        'true_dow_effects': dow_effects,
        'gen_int': gen_int,
        'reporting_delay': reporting_delay
    }

def create_generation_interval():
    """Create generation interval PMF"""
    # COVID-19 generation interval: Gamma(shape=4.9, scale=1.04)
    max_gen_int = 20
    gen_int_shape = 4.9
    gen_int_scale = 1.04
    
    gen_int = gamma.pdf(np.arange(1, max_gen_int+1), a=gen_int_shape, scale=gen_int_scale)
    gen_int = gen_int / gen_int.sum()
    
    return gen_int

def create_reporting_delay():
    """Create reporting delay PMF"""
    # Reporting delay: Gamma(shape=2.0, scale=3.5)
    max_delay = 25
    delay_shape = 2.0
    delay_scale = 3.5
    
    reporting_delay = gamma.pdf(np.arange(0, max_delay), a=delay_shape, scale=delay_scale)
    reporting_delay = reporting_delay / reporting_delay.sum()
    
    return reporting_delay

def load_data():
    """Load and preprocess the data"""
    try:
        df = pd.read_csv('cases.csv')
    except FileNotFoundError:
        print("cases.csv not found. Generating sample data...")
        df, true_params = generate_sample_data()
        print("Sample data generated and saved as cases.csv")
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    return df

def create_rt_model(cases_data, gen_int, reporting_delay):
    """Create the PyMC model for Rt estimation"""
    
    n_days = len(cases_data)
    cases = cases_data['cases'].values
    day_of_week = cases_data['day_of_week'].values - 1  # Convert to 0-6
    
    # Parameters for splines
    n_knots_rt = max(10, n_days // 15)  # Knots for Rt spline
    n_knots_asc = max(8, n_days // 20)   # Knots for ascertainment spline
    
    knots_rt = np.linspace(0, n_days-1, n_knots_rt)
    knots_asc = np.linspace(0, n_days-1, n_knots_asc)
    
    with pm.Model() as model:
        
        # ==================== Rt MODEL ====================
        # Random walk for Rt on log scale
        log_rt_init = pm.Normal('log_rt_init', mu=0, sigma=0.5)
        log_rt_steps = pm.Normal('log_rt_steps', mu=0, sigma=0.1, shape=n_days-1)
        log_rt = pt.concatenate([[log_rt_init], log_rt_init + pt.cumsum(log_rt_steps)])
        rt = pm.Deterministic('rt', pt.exp(log_rt))
        
        # ==================== DAY-OF-WEEK EFFECTS ====================
        # Day-of-week effects (multiplicative, sum to 7)
        dow_raw = pm.Normal('dow_raw', mu=0, sigma=0.3, shape=7)
        dow_effects = pm.Deterministic('dow_effects', 7 * pt.softmax(dow_raw))
        
        # ==================== TIME-VARYING ASCERTAINMENT ====================
        # Smooth ascertainment using random walk
        logit_asc_init = pm.Normal('logit_asc_init', mu=-2, sigma=1)  # Start low
        logit_asc_steps = pm.Normal('logit_asc_steps', mu=0, sigma=0.05, shape=n_days-1)
        logit_asc = pt.concatenate([[logit_asc_init], logit_asc_init + pt.cumsum(logit_asc_steps)])
        ascertainment = pm.Deterministic('ascertainment', pm.math.sigmoid(logit_asc))
        
        # ==================== INFECTION DYNAMICS ====================
        # Initial infections (seeding period)
        seed_days = min(14, n_days // 4)
        log_seed_infections = pm.Normal('log_seed_infections', mu=np.log(50), sigma=1, shape=seed_days)
        seed_infections = pt.exp(log_seed_infections)
        
        # Compute infections using renewal equation
        def compute_infections(rt_t, past_infections, gen_int):
            """Compute infections at time t using renewal equation"""
            # Convolve with generation interval
            infectiousness = 0
            for s in range(len(gen_int)):
                if s < past_infections.shape[0]:
                    infectiousness += past_infections[-(s+1)] * gen_int[s]
            return rt_t * infectiousness
        
        # Initialize infections list
        infections_list = [seed_infections[i] for i in range(seed_days)]
        
        # Compute remaining infections
        for t in range(seed_days, n_days):
            # Get past infections (up to generation interval length)
            past_infections = pt.stack(infections_list[max(0, t-len(gen_int)):t])
            
            # Compute new infections
            new_infection = compute_infections(rt[t], past_infections, gen_int)
            infections_list.append(new_infection)
        
        infections = pm.Deterministic('infections', pt.stack(infections_list))
        
        # ==================== OBSERVATION MODEL ====================
        # Apply reporting delay
        max_delay = len(reporting_delay)
        expected_reports = pt.zeros(n_days)
        
        for t in range(n_days):
            for d in range(max_delay):
                if t + d < n_days:
                    contribution = infections[t] * ascertainment[t] * reporting_delay[d]
                    expected_reports = pt.set_subtensor(
                        expected_reports[t + d], 
                        expected_reports[t + d] + contribution
                    )
        
        # Apply day-of-week effects
        dow_multiplier = dow_effects[day_of_week]
        expected_cases = expected_reports * dow_multiplier
        
        # ==================== LIKELIHOOD ====================
        # Overdispersion parameter
        phi = pm.Exponential('phi', lam=0.1)
        
        # Negative binomial likelihood
        likelihood = pm.NegativeBinomial(
            'likelihood',
            mu=expected_cases,
            alpha=phi,
            observed=cases
        )
        
        # Store expected cases for diagnostics
        pm.Deterministic('expected_cases', expected_cases)
    
    return model

def fit_model(model, samples=2000, tune=1000, chains=4):
    """Fit the PyMC model"""
    with model:
        # Use NUTS sampler
        trace = pm.sample(
            draws=samples,
            tune=tune,
            chains=chains,
            cores=min(4, chains),
            return_inferencedata=True,
            random_seed=42,
            target_accept=0.95
        )
    
    return trace

def extract_estimates(trace):
    """Extract point estimates and credible intervals"""
    summary = az.summary(trace, hdi_prob=0.95)
    
    # Extract Rt estimates
    rt_vars = [var for var in summary.index if var.startswith('rt[')]
    rt_summary = summary.loc[rt_vars]
    
    # Extract day-of-week effects
    dow_vars = [var for var in summary.index if var.startswith('dow_effects[')]
    dow_summary = summary.loc[dow_vars]
    
    # Extract ascertainment
    asc_vars = [var for var in summary.index if var.startswith('ascertainment[')]
    asc_summary = summary.loc[asc_vars]
    
    return {
        'rt': rt_summary,
        'dow_effects': dow_summary,
        'ascertainment': asc_summary,
        'full_summary': summary
    }

def create_plots(data, trace, estimates):
    """Create comprehensive plots of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    dates = data['date']
    n_days = len(dates)
    
    # Plot 1: Rt over time
    ax1 = axes[0, 0]
    rt_mean = estimates['rt']['mean'].values
    rt_lower = estimates['rt']['hdi_2.5%'].values
    rt_upper = estimates['rt']['hdi_97.5%'].values
    
    ax1.plot(dates, rt_mean, 'b-', linewidth=2, label='Rt estimate')
    ax1.fill_between(dates, rt_lower, rt_upper, alpha=0.3, color='blue', label='95% HDI')
    ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Rt = 1')
    ax1.set_ylabel('Reproduction Number (Rt)')
    ax1.set_title('Time-varying Reproduction Number')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 2: Observed vs Expected Cases
    ax2 = axes[0, 1]
    posterior = trace.posterior
    expected_cases = posterior['expected_cases'].mean(dim=['chain', 'draw']).values
    
    ax2.scatter(data['cases'], expected_cases, alpha=0.6, s=30)
    max_val = max(data['cases'].max(), expected_cases.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
    ax2.set_xlabel('Observed Cases')
    ax2.set_ylabel('Expected Cases')
    ax2.set_title('Model Fit: Observed vs Expected')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Day-of-week Effects
    ax3 = axes[1, 0]
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_mean = estimates['dow_effects']['mean'].values
    dow_lower = estimates['dow_effects']['hdi_2.5%'].values
    dow_upper = estimates['dow_effects']['hdi_97.5%'].values
    
    x_pos = np.arange(len(dow_names))
    ax3.bar(x_pos, dow_mean, yerr=[dow_mean - dow_lower, dow_upper - dow_mean], 
            capsize=5, alpha=0.7, color='green')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(dow_names)
    ax3.set_ylabel('Reporting Multiplier')
    ax3.set_title('Day-of-Week Effects')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Time-varying Ascertainment
    ax4 = axes[1, 1]
    asc_mean = estimates['ascertainment']['mean'].values
    asc_lower = estimates['ascertainment']['hdi_2.5%'].values
    asc_upper = estimates['ascertainment']['hdi_97.5%'].values
    
    ax4.plot(dates, asc_mean, 'g-', linewidth=2, label='Ascertainment')
    ax4.fill_between(dates, asc_lower, asc_upper, alpha=0.3, color='green', label='95% HDI')
    ax4.set_ylabel('Ascertainment Probability')
    ax4.set_title('Time-varying Ascertainment')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Additional plot: Cases over time with model fit
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.scatter(dates, data['cases'], alpha=0.6, s=30, label='Observed cases', color='black')
    ax.plot(dates, expected_cases, 'r-', linewidth=2, label='Expected cases', alpha=0.8)
    ax.set_ylabel('Daily Cases')
    ax.set_title('COVID-19 Cases: Observed vs Model Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("COVID-19 Rt Estimation with Complex Observation Processes")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data...")
    data = load_data()
    print(f"   Loaded {len(data)} days of case data")
    print(f"   Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"   Total cases: {data['cases'].sum():,}")
    
    # Create generation interval and reporting delay
    print("\n2. Setting up epidemiological parameters...")
    gen_int = create_generation_interval()
    reporting_delay = create_reporting_delay()
    print(f"   Generation interval mean: {np.sum(gen_int * np.arange(1, len(gen_int)+1)):.1f} days")
    print(f"   Reporting delay mean: {np.sum(reporting_delay * np.arange(len(reporting_delay))):.1f} days")
    
    # Create model
    print("\n3. Building PyMC model...")
    model = create_rt_model(data, gen_int, reporting_delay)
    print(f"   Model created with {len(data)} time points")
    
    # Check model
    with model:
        print(f"   Model has {len(model.free_RVs)} free random variables")
    
    # Fit model
    print("\n4. Fitting model (this may take several minutes)...")
    try:
        trace = fit_model(model, samples=1000, tune=500, chains=2)  # Reduced for faster execution
        print("   Model fitting completed successfully!")
    except Exception as e:
        print(f"   Error during fitting: {e}")
        return
    
    # Check convergence
    print("\n5. Checking convergence...")
    rhat = az.rhat(trace)
    max_rhat = float(rhat.max()) if hasattr(rhat.max(), 'values') else rhat.max().values.max()
    print(f"   Maximum R-hat: {max_rhat:.3f}")
    if max_rhat > 1.1:
        print("   Warning: Some parameters may not have converged (R-hat > 1.1)")
    else:
        print("   Convergence looks good (all R-hat <= 1.1)")
    
    # Extract estimates
    print("\n6. Extracting estimates...")
    estimates = extract_estimates(trace)
    
    # Print summary statistics
    print("\n7. Summary Results:")
    print("-" * 30)
    
    rt_mean = estimates['rt']['mean'].values
    print(f"   Rt range: {rt_mean.min():.2f} - {rt_mean.max():.2f}")
    print(f"   Current Rt: {rt_mean[-1]:.2f}")
    
    dow_effects = estimates['dow_effects']['mean'].values
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    print(f"   Day-of-week effects:")
    for i, (day, effect) in enumerate(zip(dow_names, dow_effects)):
        print(f"     {day}: {effect:.2f}")
    
    asc_mean = estimates['ascertainment']['mean'].values
    print(f"   Ascertainment range: {asc_mean.min():.1%} - {asc_mean.max():.1%}")
    print(f"   Final ascertainment: {asc_mean[-1]:.1%}")
    
    # Create plots
    print("\n8. Creating plots...")
    create_plots(data, trace, estimates)
    
    print("\nAnalysis completed successfully!")
    
    return {
        'data': data,
        'model': model,
        'trace': trace,
        'estimates': estimates
    }

if __name__ == "__main__":
    results = main()

