data {
  int<lower=0> T;                    // Number of time points
  int<lower=0> cases[T];             // Observed cases
  int<lower=0> G;                    // Length of generation time distribution
  vector<lower=0>[G] w;              // Generation time distribution (discrete)
  real prior_Rt_mean;                // Prior mean for initial Rt
  real<lower=0> prior_Rt_sd;         // Prior SD for initial Rt
  real prior_sigma_mean;             // Prior mean for log(sigma)
  real<lower=0> prior_sigma_sd;      // Prior SD for log(sigma)
}

transformed data {
  vector[G] rev_w = reverse(w);      // Reverse generation time for convolution
}

parameters {
  real log_Rt_init;                  // Initial log(Rt)
  vector[T-1] log_Rt_noise;          // Random walk innovations
  real log_sigma;                    // Log of random walk standard deviation
}

transformed parameters {
  vector<lower=0>[T] Rt;             // Time-varying reproduction number
  vector<lower=0>[T] lambda;         // Expected number of cases
  real<lower=0> sigma = exp(log_sigma);
  
  // Random walk for log(Rt)
  vector[T] log_Rt;
  log_Rt[1] = log_Rt_init;
  for (t in 2:T) {
    log_Rt[t] = log_Rt[t-1] + sigma * log_Rt_noise[t-1];
  }
  
  // Transform to Rt
  Rt = exp(log_Rt);
  
  // Calculate expected cases using renewal equation
  for (t in 1:T) {
    if (t == 1) {
      lambda[t] = cases[1];  // Initialize with first observation
    } else {
      real convolution = 0;
      int max_lag = min(t-1, G);
      
      for (g in 1:max_lag) {
        convolution += lambda[t-g] * w[g];
      }
      
      lambda[t] = Rt[t] * convolution;
      
      // Ensure lambda is positive
      if (lambda[t] < 1e-10) {
        lambda[t] = 1e-10;
      }
    }
  }
}

model {
  // Priors
  log_Rt_init ~ normal(log(prior_Rt_mean), prior_Rt_sd);
  log_Rt_noise ~ std_normal();
  log_sigma ~ normal(prior_sigma_mean, prior_sigma_sd);
  
  // Likelihood
  for (t in 1:T) {
    cases[t] ~ poisson(lambda[t]);
  }
}

generated quantities {
  // Posterior predictive checks
  int<lower=0> cases_pred[T];
  vector[T] log_lik;
  
  for (t in 1:T) {
    cases_pred[t] = poisson_rng(lambda[t]);
    log_lik[t] = poisson_lpmf(cases[t] | lambda[t]);
  }
}

