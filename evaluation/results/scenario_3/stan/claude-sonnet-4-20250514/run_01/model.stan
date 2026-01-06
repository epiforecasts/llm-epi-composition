// multi_stream_rt.stan
functions {
  // Discrete gamma PMF for delays
  vector discrete_gamma_pmf(int max_delay, real shape, real rate) {
    vector[max_delay] pmf;
    real normalizing_constant = 0;
    
    // Calculate unnormalized probabilities
    for (i in 1:max_delay) {
      pmf[i] = exp(lgamma(shape + i - 1) - lgamma(shape) - lgamma(i) + 
                   shape * log(rate) - (shape + i - 1) * log(1 + rate));
      normalizing_constant += pmf[i];
    }
    
    // Normalize
    pmf = pmf / normalizing_constant;
    return pmf;
  }
}

data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> S;                    // Number of data streams (3)
  int<lower=0> max_delay;            // Maximum delay to consider
  int<lower=0> max_gen;              // Maximum generation interval
  
  // Observations
  array[T, S] int<lower=0> y;        // Observations [time, stream]
  
  // Generation interval (pre-computed)
  vector<lower=0>[max_gen] generation_interval;
  
  // Priors for delay distributions
  vector<lower=0>[S] delay_shape_prior_mean;
  vector<lower=0>[S] delay_shape_prior_sd;
  vector<lower=0>[S] delay_rate_prior_mean;
  vector<lower=0>[S] delay_rate_prior_sd;
}

parameters {
  // Rt parameters
  real log_R0;                       // Initial log(Rt)
  vector[T-1] log_Rt_innovations;    // Random walk innovations
  real<lower=0> sigma_Rt;            // SD of Rt random walk
  
  // Stream-specific parameters
  vector<lower=0>[S] ascertainment_rate;  // Ascertainment rates
  vector<lower=0>[S] phi;                 // Overdispersion parameters
  
  // Delay distribution parameters
  vector<lower=0>[S] delay_shape;
  vector<lower=0>[S] delay_rate;
  
  // Initial infections
  vector<lower=0>[max_gen] I_seed;
}

transformed parameters {
  vector[T] log_Rt;
  vector[T] Rt;
  vector[T] infections;
  matrix[T, S] expected_obs;
  
  // Build log_Rt time series
  log_Rt[1] = log_R0;
  for (t in 2:T) {
    log_Rt[t] = log_Rt[t-1] + log_Rt_innovations[t-1];
  }
  Rt = exp(log_Rt);
  
  // Calculate delay PMFs for each stream
  matrix[max_delay, S] delay_pmf;
  for (s in 1:S) {
    delay_pmf[,s] = discrete_gamma_pmf(max_delay, delay_shape[s], delay_rate[s]);
  }
  
  // Calculate infections using renewal equation
  for (t in 1:T) {
    if (t <= max_gen) {
      // Seeding period
      infections[t] = I_seed[t];
    } else {
      // Renewal equation
      real infectiousness = 0;
      for (tau in 1:max_gen) {
        if (t - tau > 0) {
          infectiousness += infections[t - tau] * generation_interval[tau];
        }
      }
      infections[t] = Rt[t] * infectiousness;
    }
  }
  
  // Calculate expected observations for each stream
  for (s in 1:S) {
    for (t in 1:T) {
      expected_obs[t, s] = 0;
      
      // Sum over delays
      for (d in 1:min(max_delay, t)) {
        expected_obs[t, s] += infections[t - d + 1] * delay_pmf[d, s];
      }
      
      // Apply ascertainment
      expected_obs[t, s] *= ascertainment_rate[s];
      expected_obs[t, s] = fmax(expected_obs[t, s], 1e-8);
    }
  }
}

model {
  // Priors
  log_R0 ~ normal(0, 0.5);
  sigma_Rt ~ exponential(2);
  log_Rt_innovations ~ normal(0, sigma_Rt);
  
  // Stream-specific priors
  ascertainment_rate ~ exponential(10);  // Weak prior favoring small rates
  phi ~ exponential(0.1);                // Overdispersion
  
  // Delay distribution priors
  for (s in 1:S) {
    delay_shape[s] ~ normal(delay_shape_prior_mean[s], delay_shape_prior_sd[s]);
    delay_rate[s] ~ normal(delay_rate_prior_mean[s], delay_rate_prior_sd[s]);
  }
  
  // Initial infections
  I_seed ~ exponential(0.01);
  
  // Likelihood
  for (s in 1:S) {
    for (t in 1:T) {
      if (y[t, s] > 0 || expected_obs[t, s] > 0) {
        y[t, s] ~ neg_binomial_2(expected_obs[t, s], phi[s]);
      }
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[T, S] int y_rep;
  
  for (s in 1:S) {
    for (t in 1:T) {
      y_rep[t, s] = neg_binomial_2_rng(expected_obs[t, s], phi[s]);
    }
  }
}

