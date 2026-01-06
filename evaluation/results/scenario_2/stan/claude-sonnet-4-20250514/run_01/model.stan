data {
  int<lower=1> T;                    // Number of time points
  array[T] int<lower=0> cases;       // Observed case counts
  array[T] int<lower=1,upper=7> day_of_week;  // Day of week (1=Mon, 7=Sun)
  
  // Generation interval
  int<lower=1> G;                    // Length of generation interval
  vector<lower=0>[G] gen_pmf;        // Generation interval PMF
  
  // Delay distribution
  int<lower=1> D;                    // Length of delay distribution  
  vector<lower=0>[D] delay_pmf;      // Delay PMF
  
  // Priors
  real rt_prior_mean;
  real<lower=0> rt_prior_sd;
  real<lower=0> rt_rw_sd;
  real<lower=0> ascertainment_rw_sd;
}

transformed data {
  int seeding_days = max(G, D);      // Days needed for seeding
}

parameters {
  // Log Rt (random walk)
  vector[T] log_rt_raw;
  real log_rt_init;
  
  // Day of week effects (Monday is reference)
  vector[6] dow_effect_raw;          // Effects for Tue-Sun
  
  // Time-varying ascertainment (on logit scale, random walk)
  vector[T] logit_ascertainment_raw;
  real logit_ascertainment_init;
  
  // Overdispersion parameter
  real<lower=0> phi;
  
  // Initial infections (seeding period)
  vector<lower=0>[seeding_days] I_seed;
}

transformed parameters {
  vector[T] log_rt;
  vector[7] dow_effect;
  vector[T] logit_ascertainment;
  vector<lower=0>[T] infections;
  vector<lower=0>[T] expected_cases;
  
  // Rt random walk
  log_rt[1] = log_rt_init + rt_rw_sd * log_rt_raw[1];
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_rw_sd * log_rt_raw[t];
  }
  
  // Day of week effects (Monday = 1.0, others relative to Monday)
  dow_effect[1] = 1.0;  // Monday reference
  for (i in 1:6) {
    dow_effect[i+1] = exp(0.2 * dow_effect_raw[i]);  // Moderate effects
  }
  
  // Ascertainment random walk
  logit_ascertainment[1] = logit_ascertainment_init + 
                          ascertainment_rw_sd * logit_ascertainment_raw[1];
  for (t in 2:T) {
    logit_ascertainment[t] = logit_ascertainment[t-1] + 
                            ascertainment_rw_sd * logit_ascertainment_raw[t];
  }
  
  // Compute infections using renewal equation
  for (t in 1:T) {
    if (t <= seeding_days) {
      infections[t] = I_seed[t];
    } else {
      real renewal = 0;
      for (g in 1:G) {
        if (t - g >= 1) {
          renewal += infections[t - g] * gen_pmf[g];
        }
      }
      infections[t] = exp(log_rt[t]) * renewal;
    }
  }
  
  // Compute expected reported cases (with delays and observation process)
  for (t in 1:T) {
    real delayed_infections = 0;
    
    // Apply delay from infection to reporting
    for (d in 1:D) {
      if (t - d >= 1 && t - d <= T) {
        delayed_infections += infections[t - d] * delay_pmf[d];
      }
    }
    
    // Apply ascertainment and day-of-week effects
    expected_cases[t] = delayed_infections * 
                       inv_logit(logit_ascertainment[t]) * 
                       dow_effect[day_of_week[t]];
  }
}

model {
  // Priors
  log_rt_init ~ normal(log(rt_prior_mean), rt_prior_sd);
  log_rt_raw ~ std_normal();
  
  dow_effect_raw ~ std_normal();
  
  logit_ascertainment_init ~ normal(logit(0.3), 1);  // Prior: ~30% ascertainment
  logit_ascertainment_raw ~ std_normal();
  
  phi ~ gamma(2, 0.1);  // Overdispersion parameter
  
  I_seed ~ gamma(2, 0.1);  // Weakly informative prior on initial infections
  
  // Likelihood with overdispersion (negative binomial)
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    } else {
      cases[t] ~ poisson(1e-8);  // Avoid zero expected cases
    }
  }
}

generated quantities {
  vector<lower=0>[T] rt = exp(log_rt);
  vector<lower=0,upper=1>[T] ascertainment = inv_logit(logit_ascertainment);
  array[T] int cases_pred;
  vector[T] log_lik;
  
  // Posterior predictions and log-likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = poisson_rng(1e-8);
      log_lik[t] = poisson_lpmf(cases[t] | 1e-8);
    }
  }
}

