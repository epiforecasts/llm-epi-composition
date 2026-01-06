data {
  int<lower=1> n_days;                    // Number of days
  array[n_days] int<lower=0> cases;       // Daily case counts
  int<lower=1> max_si;                    // Length of serial interval distribution
  vector<lower=0>[max_si] si_distr;       // Serial interval distribution
}

parameters {
  real rt_init;                           // Initial log Rt
  vector[n_days-1] rt_noise;              // Random walk innovations
  real<lower=0> sigma_rt;                 // Standard deviation of Rt random walk
}

transformed parameters {
  vector[n_days] log_rt;                  // Log reproduction number
  vector[n_days] Rt;                      // Reproduction number
  vector[n_days] lambda;                  // Expected cases (infection rate)
  
  // Random walk for log Rt
  log_rt[1] = rt_init;
  for (t in 2:n_days) {
    log_rt[t] = log_rt[t-1] + sigma_rt * rt_noise[t-1];
  }
  
  // Transform to Rt
  Rt = exp(log_rt);
  
  // Calculate expected cases using renewal equation
  for (t in 1:n_days) {
    lambda[t] = 0;
    for (s in 1:min(t-1, max_si)) {
      lambda[t] += Rt[t] * cases[t-s] * si_distr[s];
    }
    // Add small constant to avoid zero lambda
    lambda[t] = fmax(lambda[t], 1e-8);
  }
}

model {
  // Priors
  rt_init ~ normal(0, 1);                 // Prior for initial log Rt (Rt ~ lognormal(0,1))
  rt_noise ~ std_normal();                // Standard normal innovations
  sigma_rt ~ normal(0, 0.2);              // Prior for Rt volatility
  
  // Likelihood
  for (t in 1:n_days) {
    if (cases[t] > 0) {
      cases[t] ~ poisson(lambda[t]);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[n_days] int cases_pred;
  vector[n_days] log_lik;
  
  for (t in 1:n_days) {
    cases_pred[t] = poisson_rng(lambda[t]);
    if (cases[t] > 0) {
      log_lik[t] = poisson_lpmf(cases[t] | lambda[t]);
    } else {
      log_lik[t] = 0;
    }
  }
}

