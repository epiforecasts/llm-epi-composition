data {
  int<lower=0> n_cases;
  vector[n_cases] cases; // daily case counts
  real[365] dates; // date in days since first reported case
  int<lower=1> S; // number of generations to account for
  int<lower=1> T; // time horizon
  
  real mu_g; // mean generation interval (days)
}

parameters {
  real<lower=0, upper=2*mu_g> R_t[S]; // reproduction numbers over time
  real<lower=0, upper=T> t_infected[n_cases]; // times infected for each case
  vector[365] nu; // innovation noise (case count uncertainty)
}

model {
  // define prior distributions and likelihoods here
  
  // model the data using renewal equation:
  for (i in 1:n_cases) {
    cases[i] ~ normal(nu[t_infected[i]], 0.5); // observation model
    nu[t_infected[i]] = sum(R_t[s] * nu[t_infected[i] - s]) / S;
  }
}

