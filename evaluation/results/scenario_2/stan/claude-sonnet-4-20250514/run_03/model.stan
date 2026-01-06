functions {
  // Convolution for renewal equation
  vector convolve_infections(vector infections, vector gen_interval, int T, int G) {
    vector[T] result = rep_vector(0.0, T);
    for (t in 1:T) {
      for (g in 1:min(G, t)) {
        if (t - g + 1 >= 1) {
          result[t] += infections[t - g + 1] * gen_interval[g];
        }
      }
    }
    return result;
  }
  
  // Convolution for reporting delay
  vector convolve_delay(vector infections, vector delay_dist, int T, int D) {
    vector[T] result = rep_vector(0.0, T);
    for (t in 1:T) {
      for (d in 1:min(D, T - t + 1)) {
        if (t + d - 1 <= T) {
          result[t + d - 1] += infections[t] * delay_dist[d];
        }
      }
    }
    return result;
  }
}

data {
  int<lower=1> T;                          // Number of time points
  array[T] int<lower=0> cases;             // Observed cases
  array[T] int<lower=1,upper=7> day_of_week; // Day of week (1=Monday, 7=Sunday)
  
  // Generation interval
  int<lower=1> G;                          // Length of generation interval
  vector[G] gen_interval;                  // Generation interval PMF
  
  // Delay distribution
  int<lower=1> D;                          // Length of delay distribution
  vector[D] delay_dist;                    // Delay distribution PMF
  
  // Prior parameters
  real rt_prior_mean;
  real<lower=0> rt_prior_sd;
  real<lower=0> rt_random_walk_sd;
  real<lower=0> ascertainment_random_walk_sd;
}

parameters {
  // Log Rt (for numerical stability)
  vector[T] log_rt_raw;                    // Raw random walk innovations
  real log_rt_init;                        // Initial log Rt
  
  // Initial infections (for seeding)
  vector<lower=0>[G] init_infections;
  
  // Day-of-week effects
  vector[6] day_of_week_raw;               // 6 effects (Sunday as reference)
  
  // Time-varying ascertainment (logit scale)
  vector[T] logit_ascertainment_raw;       // Raw random walk innovations
  real logit_ascertainment_init;           // Initial logit ascertainment
  
  // Overdispersion parameter
  real<lower=0> phi;                       // Negative binomial overdispersion
}

transformed parameters {
  vector[T] log_rt;
  vector[T] Rt;
  vector[7] day_of_week_effect;
  vector[T] logit_ascertainment;
  vector[T] ascertainment;
  vector[T] infections;
  vector[T] delayed_infections;
  vector[T] expected_cases;
  
  // Random walk for log Rt
  log_rt[1] = log_rt_init;
  for (t in 2:T) {
    log_rt[t] = log_rt[t-1] + rt_random_walk_sd * log_rt_raw[t];
  }
  Rt = exp(log_rt);
  
  // Day-of-week effects (Sunday = reference = 1.0)
  day_of_week_effect[7] = 1.0;  // Sunday
  day_of_week_effect[1:6] = exp(day_of_week_raw);
  
  // Random walk for ascertainment (on logit scale)
  logit_ascertainment[1] = logit_ascertainment_init;
  for (t in 2:T) {
    logit_ascertainment[t] = logit_ascertainment[t-1] + 
                            ascertainment_random_walk_sd * logit_ascertainment_raw[t];
  }
  ascertainment = inv_logit(logit_ascertainment);
  
  // Renewal equation
  // Initialize infections with seeded values
  for (t in 1:min(G, T)) {
    if (t <= G) {
      infections[t] = init_infections[t];
    }
  }
  
  // Apply renewal equation for subsequent time points
  for (t in (G+1):T) {
    real renewal_sum = 0.0;
    for (g in 1:G) {
      renewal_sum += infections[t - g] * gen_interval[g];
    }
    infections[t] = Rt[t] * renewal_sum;
  }
  
  // Apply reporting delay
  delayed_infections = convolve_delay(infections, delay_dist, T, D);
  
  // Expected reported cases
  for (t in 1:T) {
    expected_cases[t] = delayed_infections[t] * ascertainment[t] * 
                       day_of_week_effect[day_of_week[t]];
  }
}

model {
  // Priors
  log_rt_init ~ normal(log(rt_prior_mean), rt_prior_sd);
  log_rt_raw[2:T] ~ std_normal();
  
  init_infections ~ exponential(0.1);
  
  day_of_week_raw ~ normal(0, 0.2);  // Weakly informative
  
  logit_ascertainment_init ~ normal(logit(0.1), 1);  // Prior belief about ascertainment
  logit_ascertainment_raw[2:T] ~ std_normal();
  
  phi ~ exponential(0.1);
  
  // Likelihood
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases[t] ~ neg_binomial_2(expected_cases[t], phi);
    }
  }
}

generated quantities {
  // Posterior predictive checks
  array[T] int cases_pred;
  vector[T] log_lik;
  
  for (t in 1:T) {
    if (expected_cases[t] > 0) {
      cases_pred[t] = neg_binomial_2_rng(expected_cases[t], phi);
      log_lik[t] = neg_binomial_2_lpmf(cases[t] | expected_cases[t], phi);
    } else {
      cases_pred[t] = 0;
      log_lik[t] = 0;
    }
  }
}

