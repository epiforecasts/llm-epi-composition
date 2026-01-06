data {
  int<lower=1> n_days;
  array[n_days] int<lower=0> cases;
  int<lower=1> n_serial;
  vector<lower=0>[n_serial] serial_interval;
  
  // Priors
  real rt_prior_mean;
  real<lower=0> rt_prior_sd;
  real<lower=0> rt_walk_sd;
}

parameters {
  vector[n_days] log_rt_raw;
  real<lower=0> phi; // Overdispersion parameter for negative binomial
}

transformed parameters {
  vector[n_days] log_rt;
  vector[n_days] Rt;
  vector[n_days] lambda; // Expected infections
  
  // Random walk for log(Rt)
  log_rt[1] = log(rt_prior_mean) + rt_walk_sd * log_rt_raw[1];
  for (t in 2:n_days) {
    log_rt[t] = log_rt[t-1] + rt_walk_sd * log_rt_raw[t];
  }
  
  Rt = exp(log_rt);
  
  // Calculate expected infections using renewal equation
  for (t in 1:n_days) {
    real infectiousness = 0;
    
    for (s in 1:min(t-1, n_serial)) {
      infectiousness += cases[t-s] * serial_interval[s];
    }
    
    if (t <= n_serial) {
      // Early days: use simple exponential growth
      lambda[t] = cases[1] * exp(0.1 * (t-1));
    } else {
      lambda[t] = Rt[t] * infectiousness;
    }
    
    // Ensure lambda is positive and bounded
    lambda[t] = fmax(lambda[t], 1e-6);
  }
}

model {
  // Priors
  log_rt_raw ~ std_normal();
  phi ~ exponential(0.1);
  
  // Likelihood
  for (t in 1:n_days) {
    if (lambda[t] > 0) {
      cases[t] ~ neg_binomial_2(lambda[t], phi);
    }
  }
}

generated quantities {
  array[n_days] int cases_pred;
  vector[n_days] log_lik;
  
  // Posterior predictive checks
  for (t in 1:n_days) {
    cases_pred[t] = neg_binomial_2_rng(lambda[t], phi);
    log_lik[t] = neg_binomial_2_lpmf(cases[t] | lambda[t], phi);
  }
}

