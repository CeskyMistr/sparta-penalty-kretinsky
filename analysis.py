import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


from functions import load_data, test_mean_equality, calculate_referee_interventions, beta_binomial_posterior, fit_regression_models
from functions_charts import plot_sampled_penalties,visualize_and_test_gamma_dist, plot_posterior_distributions, plot_coeff_plot

#### Settings
half = False
# denominator = "utoky"   # "utoky" or "nebezpecne_utoky"
# Severity Coeffs
# faul = 1            # avg: 12.28
# zluta_karta = 3     # avg:  1.99
# cervena_karta = 6   # avg:  0.14 ; red_card_xg = lambda x: 1.4 * x ** 0.85
# penalta = 9         # avg:  0.12 ; yellow_card_xg = lambda x: red_card_xg(x) / 2


#### Load data
dt = load_data(half)

#### Feature engineering
# Křetínský factor
filtr = dt["match_datetime"] >= datetime.datetime(2023, 4, 3)
dt["kretinsky"] = 0
dt.loc[filtr, "kretinsky"] = 1
dt["kretinsky_sparta"] = dt["kretinsky"] * (dt["team"] == "Sparta Praha")
dt["kretinsky_sparta_opp"] = dt["kretinsky"] * (dt["team_opp"] == "Sparta Praha")
dt["kretinsky_slavia"] = dt["kretinsky"] * (dt["team"] == "Slavia Praha")
dt["kretinsky_slavia_opp"] = dt["kretinsky"] * (dt["team_opp"] == "Slavia Praha")

# Add referee interventions with specific severity
dt = calculate_referee_interventions(dt, faul=1, zluta_karta=3, cervena_karta=6, penalta=9, denominator="utoky")

#### Test mean equality
# Filtr Sparta matches
filtr = dt["team"] == "Sparta Praha"
dt_sparta = dt.loc[filtr,:].sort_values(by="match_datetime", ignore_index=True)
dt_sparta[["match_id", "match_datetime", "team", "team_opp", "kretinsky", "penalty", "penalty_opp"]]
equality_tests_sparta = test_mean_equality(dt_sparta)

# Filtr Slavia matches
filtr = dt["team"] == "Slavia Praha"
dt_slavia = dt.loc[filtr,:].sort_values(by="match_datetime", ignore_index=True)
equality_tests_slavia = test_mean_equality(dt_slavia)


#### Simulate # of penalties in 8 matches
np.random.seed(666)
sample_size = 10000
sample_penalties = lambda : dt_sparta["penalty"].sample(8, replace=True, ignore_index=True).sum()
hist_data = pd.Series([sample_penalties() for x in range(sample_size)]).value_counts() / sample_size * 100
plot_sampled_penalties(hist_data, 'sparta_penalty_sampling.png')


#### Bayesian Beta-Binomial model for pct of matches with penalty
sparta_pens_before_kretinsky = dt_sparta.loc[dt_sparta["kretinsky"] == 0, "penalty"]
sparta_pens_after_kretinsky  = dt_sparta.loc[dt_sparta["kretinsky"] == 1, "penalty"]

# Compute the posterior parameters
alpha_a, beta_a, alpha_b, beta_b, prob_b_greater = beta_binomial_posterior(sparta_pens_before_kretinsky, sparta_pens_after_kretinsky)

# Plot the posterior distributions
plot_posterior_distributions(alpha_a, beta_a, alpha_b, beta_b, prob_b_greater)


#### More complex model for referee interventions
# Visualize & Test distribution
visualize_and_test_gamma_dist(dt, "ref_interv_per_attack")

# Fit regression models
model_gamma, model_lm = fit_regression_models(dt)

# Plot coefficient estimates with error bars
plot_coeff_plot(model_gamma, "Coeff plot - efekt na pískání rozhodčího", "coeff_plot_gamma.png")
plot_coeff_plot(model_lm, "Coeff plot - efekt na zatížení pískáním rozhodčího", "coeff_plot_lm.png")

