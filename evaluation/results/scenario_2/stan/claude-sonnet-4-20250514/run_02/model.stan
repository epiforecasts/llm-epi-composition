data {
  int<lower=1> T;                    // Number of time points
  array[T] int<lower=0> cases;       // Observed cases
  array[T] int<lower=1,upper=7> day_of_week;  // Day of week (1=Mon, 7=Sun)
  int<lower=1> S;                    // Length of generation interval
  int<lower=1> D;                    // Length of delay distribution
  vector<lower=0>[S] generation_pmf; // Generation interval PMF
  vector<lower=0>[D] delay_pmf;      // Reporting delay PMF
  int<lower=1> seeding_days;         // Days for infection seeding
}

transformed data {
  vector[S] generation_pmf_rev = reverse(generation_pmf);
}

parameters {
  // Rt parameters
  real log_rt_mean;                       // Overall mean log Rt
  real<lower=0> rt_sigma;                 // Std dev of Rt random walk
  vector[T-1] rt_noise;                   // Rt random walk innovations
  
  // Day of week effects (Monday = reference)
  vector[6] log_dow_effects_raw;          // Log effects for Tue-Sun
  
  // Time-varying ascertainment
  real logit_ascertainment_mean;          // Mean logit ascertainment
  real<lower=0> ascertainment_sigma;      // Std dev of ascertainment
  vector[T-1] ascertainment_noise;        // Ascertainment random walk
  
  // Overdispersion
  real<lower=0> phi_inv;                  // Inverse overdispersion parameter
  
  // Initial infections
  vector<lower=0>[seeding_days] log_infections_seed;
}

transformed parameters {
  vector[T] log_rt;
  vector[T] rt;
  vector[7] dow_effects;
  vector[T] logit_ascertainment;
  vector[T] ascertainment;
  vector[T] infections;
  vector[T] expected_cases;
  real phi = inv(phi_inv);
  
  // Rt evolution (random walk on log scale)
  log_rt[1] = log_rt_mean;
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_sigma * rt_noise[t-1];
  }
  rt = exp(log_rt);
  
  // Day of week effects (Monday = 1.0 reference)
  dow_effects[1] = 1.0;
  dow_effects[2:7] = exp(log_dow_effects_raw);
  
  // Ascertainment evolution (random walk on logit scale)
  logit_ascertainment[1] = logit_ascertainment_mean;
  for (t in 2:T) {
    logit_ascertainment[t] = logit_ascertainment[t-1] + 
                           ascertainment_sigma * ascertainment_noise[t-1];
  }
  ascertainment = inv_logit(logit_ascertainment);
  
  // Infection dynamics
  // Seed initial infections
  for (t in 1:seeding_days) {
    infections[t] = exp(log_infections_seed[t]);
  }
  
  // Renewal equation for subsequent infections
  for (t in (seeding_days+1):T) {
    real lambda = 0;
    int max_lag = min(t-1, S);
    for (s in 1:max_lag) {
      lambda += infections[t-s] * generation_pmf[s];
    }
    infections[t] = rt[t] * lambda;
  }
  
  // Expected cases with delays and observation process
  for (t in 1:T) {
    real expected = 0;
    int max_delay = min(t, D);
    for (d in 1:max_delay) {
      if (t-d+1 >= 1) {
        expected += infections[t-d+1] * delay_pmf[d];
      }
    }
    expected_cases[t] = expected * ascertainment[t] * dow_effects[day_of_week[t]];
  }
}

model {
  // Priors
  log_rt_mean ~ normal(0, 0.5);           // Rt around 1
  rt_sigma ~ normal(0, 0.1);              // Moderate variation in Rt
  rt_noise ~ std_normal();
  
  log_dow_effects_raw ~ normal(0, 0.2);   // Day effects close to 1
  
  logit_ascertainment_mean ~ normal(-1, 0.5);  // Around 25% ascertainment
  ascertainment_sigma ~ normal(0, 0.05);       // Slow ascertainment changes
  ascertainment_noise ~ std_normal();
  
  phi_inv ~ exponential(5);               // Overdispersion
  
  log_infections_seed ~ normal(3, 1);     // Initial infections around 20
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 1e-6) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  array[T] int cases_pred;
  vector[T] log_lik;
  
  // Posterior predictive samples
  for (t in 1:T) {
    if (expected_cases[t] > 1e-6) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = 0;
      log_lik[t] = 0;
    }
  }
}

