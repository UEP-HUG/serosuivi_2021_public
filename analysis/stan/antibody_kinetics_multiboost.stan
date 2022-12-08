//
// This Stan program infers antibody trajectories assuming a bi-phasic kintetic model
//
// Input data consists of obervations of antibody levels at given times after a reference point
// The kinetic model is based on XXX et al. 2021 ()
//
functions {
  // Function to simulate antibody kinetics
  real simKinetics(real t, real alpha, real beta, real gamma_s,
  real gamma_l, real nu, real rho, real delta) {
    real y_gen;
    real epsilon = t - delta;
    // Difference in antibody levels
    real da = beta * (rho * (exp(-gamma_s * epsilon) - exp(-nu * epsilon))/(nu - gamma_s) +
    (1-rho) * (exp(-gamma_l * epsilon) - exp(-nu * epsilon))/(nu - gamma_l));

    y_gen = alpha + max([da, 0]);
    return y_gen;
  }

  real logSimKinetics(real t, real loga, real beta, real gamma_s,
  real gamma_l, real nu, real rho, real delta) {
    real y_gen;
    real epsilon = t - delta;

    if (epsilon < 0) {
      y_gen = loga;
    } else {
      real a = min([nu, gamma_s]);
      real b = max([nu, gamma_s]);
      real c = min([nu, gamma_l]);
      real d = max([nu, gamma_l]);
      real ds = log(rho) + log_diff_exp(-a * epsilon, -b * epsilon) - log(b - a);
      real dl = log(1-rho) + log_diff_exp(-c * epsilon, -d * epsilon) - log(d - c);
      real logda = log(beta) + log_sum_exp([ds, dl]);

      y_gen = log_sum_exp([loga, logda]);
    }
    return y_gen;
  }
}

data {
  // Scalars
  int<lower=0> M;    // total number of observations
  int<lower=0> N;    // number of participants
  int<lower=1> B;    // maximum number of boosts
  int<lower=0> NB;   /// number of participants x boosts for each
  int<lower=0> K;    // number of covariate
  int<lower=0> J;    // number of unique covariate combinations
  int<lower=0> M_cens; // number of censored observations for which event times are known
  int<lower=0> N_interval; // number of participants for which the date of infection is unkown
  int<lower=0> M_interval; // number of observations for which the date of infection is unkown
  int<lower=0> M_cens_interval; // number of censored observations for which infection date is unknown
  int<lower=0> L; // maximum length of time intervals for unkown infection times

  // Observations
  vector<lower=0>[M-M_interval] y; // antibody values
  vector<lower=0>[M_interval] y_interval;   // antibody values for unknown dates
  real<lower=0> sigma;     // Uncertainty in observations, known through quality control

  // Boosts
  int<lower=1> n_boost[N]; // number of boosts for each participant
  int<lower=1, upper=M> starts[N];   // start of data for each participant
  int<lower=1, upper=M> ends[N];    // end of data each participant

  // Intervals
  int<lower=0> pind_known[N-N_interval];  // indices of participants with known dates
  int<lower=0> pind_interval[N_interval]; // indices of participants with unknown dates
  int<lower=0,upper=L> L_intervals[N_interval]; // number of interval sections for each participant
  matrix<lower=0>[N_interval, L] prob_interval; // the probability of infection in each interval based on reported cases

  // Censoring
  real<lower=0> y_max;     // censoring limit
  int<lower=0> ind_cens[M_cens]; // indices of censored observations
  int<lower=0> ind_uncens[M-M_interval-M_cens]; // indices of uncensored observations
  int<lower=0, upper=1> is_cens_interval[M_interval]; // indicateor vector of censored observations

  // Timings
  matrix<lower=0>[M, B] dt;   // times to references (either infection or vaccination)
  matrix<lower=0>[N_interval, L] dt_interval;   // times to infection for unknown infection times

  // Covariates
  matrix[NB, K] X;          // matrix of covariates
  matrix[J, K] X_u;        // matrix of unique covariate combinations

  // Mappings
  int<lower=0, upper=N> map_to_i[M];    // map from observations to participants
  int<lower=1, upper=B> map_to_boost[M];    // map from observations to boost number
  int<lower=1, upper=NB> map_to_i_boost[M];    // map from observations to boost number
  int<lower=0, upper=NB> map_i_b_to_ib[N, B];  // map from participant and boost to ib
  int<lower=0, upper=M_interval> map_interval[M]; // map from observation to interval observation;
  int<lower=0, upper=M-M_interval> map_known[M];
  int<lower=0, upper=M_cens_interval> map_cens_interval[M_interval]; // map from interval observation to censored interval observation id;

  // Locations of priors of mean parameters (intercepts)
  real prior_alpha_mu;
  real prior_alpha_sigma;

  real prior_beta_mu;
  real prior_beta_sigma;

  real prior_thalf_gamma_s_mu;
  real prior_thalf_gamma_s_sigma;

  real prior_thalf_gamma_l_mu;
  real prior_thalf_gamma_l_sigma;

  real prior_thalf_nu_mu;
  real prior_thalf_nu_sigma;

  real prior_delta_mu;
  real prior_delta_sigma;

  real prior_rho_a;
  real prior_rho_b;

  // Standard deviation of priors on random effects
  real sigma_sd_prior;

  // Standrad deviation of regressoin coefficients other than intercepts
  real coef_sd;
}
transformed data {
  real log_y_max = log(y_max);
  int<lower=0> N_known = N - N_interval;
  int<lower=0> M_known = M - M_interval;
  int<lower=0> M_known_uncens = M_known - M_cens;
  int<lower=0> M_uncens_interval = M_interval - M_cens_interval;
  real log_prob_interval[N_interval, L];
  real log_y[M_known];
  real log_y_interval[M_interval];

  for (i in 1:M_known) {
    log_y[i] = log(y[i]);
  }

  for (i in 1:M_interval) {
    log_y_interval[i] = log(y_interval[i]);
  }

  for (i in 1:N_interval) {
    for (l in 1:L_intervals[i]) {
      log_prob_interval[i, l] = log(prob_interval[i, l]);
    }
  }
}
parameters {

  // Censored observations
  real<lower=log_y_max, upper = log(2.5e5)> log_y_cens[M_cens];
  real<lower=log_y_max, upper = log(2.5e5)> log_y_cens_interval[M_cens_interval];

  real<lower = 0, upper = 1> alpha;
  real<lower = 1, upper = 5> delta;

  // uncentered Individual-level parameters
  vector[NB] log_betas_u;     // boosting * production
  vector[NB] log_gammas_s_u;     // decay of long-live cells
  vector[NB] log_gammas_l_u;     // decay of long-live cells
  vector[NB] log_nus_u;     // decay of long-live cells

  // regression coefficients for parameter means
  vector[K] b_beta;
  vector[K] b_gamma_s;
  real<lower=0> diff_b_gamma_l; // enforce gamma_l[1] > gamma_s[1]
  vector[K-1] b_gamma_l_other;
  vector[K] b_nu;
  vector[K] b_rho;

  // sd of pooling
  real<lower=0> sigma_beta;
  real<lower=0> sigma_gamma_l;
  real<lower=0> sigma_gamma_s;
  real<lower=0> sigma_nu;

}
transformed parameters {


  vector[K] b_gamma_l;
  vector[J] thalf_gamma_s;
  vector[J] thalf_gamma_l;
  vector[J] thalf_nu;

  vector[M_known] y_model;    // modeled antibody levels for each participant at each time point
  real loga = log(alpha);

  vector[M_interval] lp; // log-likelihood of observations for which infection dates are unknown

  b_gamma_l[1] = b_gamma_s[1] + diff_b_gamma_l;
  b_gamma_l[2:K] = b_gamma_l_other;
  thalf_gamma_s = exp(X_u * b_gamma_s);
  thalf_gamma_l = exp(X_u * b_gamma_l);
  thalf_nu = exp(X_u * b_nu);

  for (i in 1:M_known) {
    y_model[i] = 0;
  }

  for (i in 1:M_interval) {
    lp[i] = 0;
  }

  {
    vector[NB] betas;
    vector[NB] gammas_s;
    vector[NB] gammas_l;
    vector[NB] nus;
    vector[NB] rhos;
    vector[NB] deltas;

    // uncentered parametrization
    betas = exp(X * b_beta + log_betas_u * sigma_beta);
    gammas_s = log(2)./exp(X * b_gamma_s + log_gammas_s_u * sigma_gamma_s);
    gammas_l = log(2)./exp(X * b_gamma_l + log_gammas_l_u * sigma_gamma_l);
    nus = log(2)./exp(X * b_nu + log_nus_u * sigma_nu);
    rhos = inv_logit(X * b_rho);
    deltas = rep_vector(delta, NB);

    // Participants for which event dates are known
    for (pind in 1:N_known) {
      int i = pind_known[pind];  // index of participant

      for (m in starts[i]:ends[i]){

        int b = map_to_boost[m];    // boost number
        int ib = map_to_i_boost[m]; // boost and int number
        real x = 0; // accumulator

        for (j in 1:b) {
          int jb = map_i_b_to_ib[i, j];
          if (j == 1) {
            x = logSimKinetics(dt[m, j], loga, betas[jb], gammas_s[jb], gammas_l[jb], nus[jb], rhos[jb], deltas[jb]);
          } else {
            x = log_sum_exp([x, logSimKinetics(dt[m, j], loga, betas[jb], gammas_s[jb], gammas_l[jb], nus[jb], rhos[jb], deltas[jb])]);
          }
        }

        y_model[map_known[m]] = x;
      }
    }

    // Likelihoods for participants for which infection dates are unkown
    for (pind in 1:N_interval) {
      int i = pind_interval[pind];  // index of participant
      int lint = L_intervals[pind]; // length of intervals for this participant
      real ys[lint]; // values of modeled kinetics at each unkown interval
      vector[lint] lps; // log-likelihood of observations for which infection dates are unknown

      for (m in starts[i]:ends[i]){
        int b = map_to_boost[m];    // boost number
        int ib = map_to_i_boost[m]; // boost and int number
        real x = 0; // accumulator

        for (j in 1:b) {
          int jb = map_i_b_to_ib[i, j];
          if (j == 1) {
            // Compute simulated kinetics at each subinterval
            // (infection with unkown dates is assumed to always be the first boost)
            for (l in 1:lint) {
              ys[l] = logSimKinetics(dt[m, j] + dt_interval[pind, l], loga, betas[jb], gammas_s[jb], gammas_l[jb], nus[jb], rhos[jb], deltas[jb]);;
            }
          } else {
            // Cache kinetics for vaccinations which for which dates are known
            real xb = logSimKinetics(dt[m, j], loga, betas[jb], gammas_s[jb], gammas_l[jb], nus[jb], rhos[jb], deltas[jb]);
            for (l in 1:lint) {
              ys[l] =  log_sum_exp([ys[l], xb]);
            }
          }
        }

        // Likelihood computation

        // First set prior on infection dates
        for (l in 1:lint) {
          lps[l] = log_prob_interval[pind, l];
        }

        // Add loglik of observations
        if (is_cens_interval[map_interval[m]] == 0) {
          // int check = 0;
          // Uncensored observations
          for (l in 1:lint) {
            lps[l] += normal_lpdf(log_y_interval[map_interval[m]]| ys[l], sigma);
            // if (is_inf(lps[l]) || is_nan(lps[l]) && check == 0) {
            //   check = 1;
            //   print("Found inf at: pind: ", pind, " m: ", m, " l: ", l, " logy:", log_y_interval[map_interval[m]],
            //   " ys: ", ys[l], " logp:", log_prob_interval[pind, l]);
            // }
          }
        } else {
          // Censored observations
          for (l in 1:lint) {
            lps[l] += normal_lpdf(log_y_cens_interval[map_cens_interval[map_interval[m]]] | ys[l], sigma);
            // if (is_inf(lps[l])) {
            //   print("Found inf at: pind: ", pind, " m: ", m, " l: ", l);
            // }
          }
        }

        // Accumulate log-lik
        lp[map_interval[m]] += log_sum_exp(lps);
      }
    }
  }
}

model {

  // Uncensored observations
  log_y[ind_uncens] ~ normal(y_model[ind_uncens], sigma);

  if (M_cens > 0) {
    log_y_cens ~ normal(y_model[ind_cens], sigma);
  }

  // Add loglikelihoods of interval observations
  target += sum(lp);

  // Hierarchical model
  log_betas_u ~ std_normal();
  log_gammas_s_u ~ std_normal();
  log_gammas_l_u ~ std_normal();
  log_nus_u ~ std_normal();

  // Priors
  alpha ~ normal(prior_alpha_mu,  prior_alpha_sigma);
  delta ~ normal(prior_delta_mu, prior_delta_sigma);
  exp(b_beta[1]) ~ normal(prior_beta_mu,  prior_beta_sigma);

  // Priors on reference half-lives
  thalf_gamma_s[1] ~ normal(prior_thalf_gamma_s_mu, prior_thalf_gamma_s_sigma);
  thalf_gamma_l[1] ~ normal(prior_thalf_gamma_l_mu, prior_thalf_gamma_l_sigma);
  thalf_nu[1] ~ normal(prior_thalf_nu_mu, prior_thalf_nu_sigma);

  inv_logit(b_rho[1]) ~ beta(prior_rho_a, prior_rho_b);

  // Correct target likelihood
  target += b_beta[1];
  target += b_gamma_s[1];
  target += b_gamma_l[1];
  target += b_nu[1];
  target += b_rho[1] - 2 * log1p_exp(b_rho[1]);  //log(d/dx inv_logit(x))

  if (K > 1) {
    b_beta[2:K] ~ normal(0, coef_sd);
  }

  if (K > 1) {
    b_gamma_s[2:K] ~ normal(0, coef_sd);
    b_gamma_l[2:K] ~ normal(0, coef_sd);
    b_nu[2:K] ~ normal(0, coef_sd);
    b_rho[2:K] ~ normal(0, coef_sd);
  }

  sigma_beta ~ normal(0, sigma_sd_prior);
  sigma_gamma_s ~ normal(0, sigma_sd_prior);
  sigma_gamma_l ~ normal(0, sigma_sd_prior);
  sigma_nu ~ normal(0, sigma_sd_prior);

}
