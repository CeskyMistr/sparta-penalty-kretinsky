import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levene, ttest_ind, beta
import statsmodels.api as sm


def load_data(half = False):
    """
    Load data from parquet files and merge them into one dataframe.

    Parameters
    ----------
    half : bool
        Whether to use data from the second half of the season (True) or not (False).
    
    Returns
    -------
    dtc : pandas.DataFrame
        Dataframe with all the data.
    """


    #### Load data
    dta = pd.read_parquet("match_data.parquet")
    dtb = pd.read_parquet("event_data.parquet")
    dtc = pd.read_parquet("stats_data_2nd_half.parquet") if half else pd.read_parquet("stats_data.parquet")

    #### Data manipulation
    #   # Add match max time
    #   dtb["match_max_time"] = dtb.groupby("match_id")["time"].transform("max")
    #   dtb["match_max_time"] = np.maximum(dtb["match_max_time"], 94)
    #   
    #   # Add half max time
    #   filtr = dtb["half"] == 1
    #   dtb["half_max_time"] = dtb.loc[filtr,:].groupby(["match_id", "half"])["time"].transform("max")
    #   dtb["half_max_time"] = np.maximum(dtb["half_max_time"], 46).fillna(46)
    #   
    #   # Add score at event time
    #   dtb["goals_home_after_event"] = ((dtb["event"] == "goal") * (dtb["team"] == "home") * 1).groupby(dtb["match_id"]).cumsum()
    #   dtb["goals_away_after_event"] = ((dtb["event"] == "goal") * (dtb["team"] == "away") * 1).groupby(dtb["match_id"]).cumsum()
    #   dtb["goals_home_before_event"] = dtb["goals_home_after_event"].groupby(dtb["match_id"]).shift(1).fillna(0)
    #   dtb["goals_away_before_event"] = dtb["goals_away_after_event"].groupby(dtb["match_id"]).shift(1).fillna(0)
    #   
    #   # Control score
    #   dtb.reset_index(drop = False, inplace = True)
    #   filtr = dtb.groupby("match_id")["index"].max()
    #   tmp = dtb.loc[filtr,["match_id", "goals_home_after_event", "goals_away_after_event"]].merge(dta[["match_id", "match_datetime", "team_home", "team_away", "home_goals", "away_goals"]], on = "match_id", how = "left")
    #   sum(tmp["home_goals"] != tmp["goals_home_after_event"])
    #   sum(tmp["away_goals"] != tmp["goals_away_after_event"])
    #   dtb.drop(columns="index", inplace = True)

    # Add yellow score at event time
    dtb["yellow_home"] = ((dtb["event"] == "yellow") * (dtb["team"] == "home") * 1).groupby(dtb["match_id"]).cumsum()
    dtb["yellow_away"] = ((dtb["event"] == "yellow") * (dtb["team"] == "away") * 1).groupby(dtb["match_id"]).cumsum()

    # Add red score at event time
    dtb["red_home"] = ((dtb["event"] == "red") * (dtb["team"] == "home") * 1).groupby(dtb["match_id"]).cumsum()
    dtb["red_away"] = ((dtb["event"] == "red") * (dtb["team"] == "away") * 1).groupby(dtb["match_id"]).cumsum()

    # Add penalty score at event time
    dtb["penalty_home"] = (dtb["penalty"].fillna(0) * (dtb["team"] == "home") * 1).groupby(dtb["match_id"]).cumsum()
    dtb["penalty_away"] = (dtb["penalty"].fillna(0) * (dtb["team"] == "away") * 1).groupby(dtb["match_id"]).cumsum()

    #### Extract final match stats
    dtb.reset_index(drop = False, inplace = True)
    # For 2nd half I need to subtract match stats from 1st half stats
    if half:
        filtr_1st = dtb.loc[dtb["half"] == 1,:].groupby("match_id")["index"].max()
        filtr_2nd = dtb.loc[dtb["half"] == 2,:].groupby("match_id")["index"].max()
        dtb_1st = dtb.loc[filtr_1st,:].drop(columns = ["index", "event", "time", "half", "team", "penalty"]).reset_index(drop = True)
        dtb_2nd = dtb.loc[filtr_2nd,:].drop(columns = ["index", "event", "time", "half", "team", "penalty"]).reset_index(drop = True)
        dtb_fin = dtb_2nd.merge(dtb_1st, on = "match_id", how = "left", suffixes = ("_2nd", "_1st"))
        for column in dtb_2nd.columns:
            if column == "match_id":
                pass
            else:
                dtb_fin[column] = dtb_fin[column + "_2nd"] - dtb_fin[column + "_1st"].fillna(0)
                dtb_fin.drop(columns = [column + "_2nd", column + "_1st"], inplace = True)
    else:
        filtr = dtb.groupby("match_id")["index"].max()
        dtb_fin = dtb.loc[filtr,:].drop(columns = ["index", "event", "time", "half", "team", "penalty"]).reset_index(drop = True)


    #### Create final dataset
    dt = dtb_fin.merge(dtc, on = "match_id", how = "left") \
                .merge(dta, on = "match_id", how = "left")
    # Reorder columns fore better readability
    dt = dt[['match_id', 'match_datetime', 'referee', 'team_home', 'team_away', 'home_goals', 'away_goals', 'yellow_home', 'yellow_away',
             'red_home', 'red_away', 'penalty_home', 'penalty_away', 'fauly_home', 'fauly_away','utoky_home', 'utoky_away',
             'nebezpecne_utoky_home', 'nebezpecne_utoky_away']]

    #### Create dataset for model
    dt_model = pd.DataFrame({"match_id": pd.concat([dt["match_id"], dt["match_id"]]),
                "match_datetime": pd.concat([dt["match_datetime"], dt["match_datetime"]]),
                "team": pd.concat([dt["team_home"], dt["team_away"]]),
                "hga": ["H"]*len(dt) + ["A"]*len(dt),
                "team_opp": pd.concat([dt["team_away"], dt["team_home"]]),
                "goals": pd.concat([dt["home_goals"], dt["away_goals"]]),
                "goals_opp": pd.concat([dt["away_goals"], dt["home_goals"]]),
                "yellow": pd.concat([dt["yellow_home"], dt["yellow_away"]]),
                "yellow_opp": pd.concat([dt["yellow_away"], dt["yellow_home"]]),
                "red": pd.concat([dt["red_home"], dt["red_away"]]),
                "red_opp": pd.concat([dt["red_away"], dt["red_home"]]),
                "penalty": pd.concat([dt["penalty_home"], dt["penalty_away"]]),
                "penalty_opp": pd.concat([dt["penalty_away"], dt["penalty_home"]]),
                "fauly": pd.concat([dt["fauly_home"], dt["fauly_away"]]),
                "fauly_opp": pd.concat([dt["fauly_away"], dt["fauly_home"]]),
                "utoky": pd.concat([dt["utoky_home"], dt["utoky_away"]]),
                "utoky_opp": pd.concat([dt["utoky_away"], dt["utoky_home"]]),
                "nebezpecne_utoky": pd.concat([dt["nebezpecne_utoky_home"], dt["nebezpecne_utoky_away"]]),
                "nebezpecne_utoky_opp": pd.concat([dt["nebezpecne_utoky_away"], dt["nebezpecne_utoky_home"]])}).reset_index(drop = True)

    # Filter out matches with missing attack stats
    filtr = dt_model["utoky"] != 0
    dt_model = dt_model.loc[filtr,:].reset_index(drop = True)
    print(f"Deleted {sum(~filtr)/2} matches with missing attack stats.")

    return dt_model


def calculate_referee_interventions(dt: pd.DataFrame, faul: float, zluta_karta: float, cervena_karta:float, penalta: float,
                                    denominator: str) -> pd.DataFrame:
    """
    Calculate referee interventions per attack

    Parameters
    ----------
    dt : pd.DataFrame
        Loaded data
    penalta : float
        Weight of penalty severity
    cervena_karta : float
        Weight of red card severity
    zluta_karta : float
        Weight of yellow card severity
    faul : float
        Weight of foul severity
    denominator : str {'utoky', 'nebezpecne_utoky'}
        Denominator for referee interventions
    
    Returns
    -------
    dt : pd.DataFrame
        Data extended by referee interventions
        ['ref_interv', 'ref_interv_opp', 'ref_interv_per_attack', 'ref_interv_per_attack_opp', 'ref_interv_per_attack_diff']
    """

    # Copy data
    data = dt.copy()

    # Weighted referee interventions
    data["ref_interv"] = penalta * data["penalty_opp"] + \
                    cervena_karta * data["red"] + \
                    zluta_karta * data["yellow"] + \
                    faul * (data["fauly"] - data["penalty_opp"])     # cannot subtract red&yellow cards, not all are fouls
    data["ref_interv_opp"] = penalta * data["penalty"] + \
                        cervena_karta * data["red_opp"] + \
                        zluta_karta * data["yellow_opp"] + \
                        faul * (data["fauly_opp"] - data["penalty"]) # cannot subtract red&yellow cards, not all are fouls
    # Weighted referee interventions per number of attacks
    data["ref_interv_per_attack"] = data["ref_interv"] / data[denominator + "_opp"]
    data["ref_interv_per_attack_opp"] = data["ref_interv_opp"] / data[denominator]
    data["ref_interv_per_attack_diff"] = data["ref_interv"] / data[denominator + "_opp"] - data["ref_interv_opp"] / data[denominator]

    return data


def test_mean_equality(dt_team):
    #### Test mean equality
    # Variables to test
    columns = ["penalty", "penalty_opp", "red", "red_opp", "yellow", "yellow_opp", "fauly", "fauly_opp", "ref_interv", "ref_interv_opp",
            "ref_interv_per_attack", "ref_interv_per_attack_opp"]
    # Create empty dataframe
    equality_tests = pd.DataFrame(index = columns, columns = ["var_pval", "var_equal", "mean_before", "mean_after", "mean_pval", "mean_equal"])

    # Loop over variables
    for col in columns:

        # Filter samples before and after Kretinsky
        sample_before_kretinsky = dt_team.loc[dt_team["kretinsky"] == 0, col]
        sample_after_kretinsky  = dt_team.loc[dt_team["kretinsky"] == 1, col]

        # First test for equality of variances
        levene_test = levene(sample_before_kretinsky, sample_after_kretinsky)
        vars_equal = levene_test.pvalue > 0.05
        # Then test for equality of means
        mean_test = ttest_ind(sample_before_kretinsky, sample_after_kretinsky, equal_var=vars_equal)

        # Save results
        equality_tests.loc[col, "var_pval"] = levene_test.pvalue
        equality_tests.loc[col, "var_equal"] = vars_equal
        equality_tests.loc[col, "mean_before"] = sample_before_kretinsky.mean()
        equality_tests.loc[col, "mean_after"] = sample_after_kretinsky.mean()
        equality_tests.loc[col, "mean_pval"] = mean_test.pvalue
        equality_tests.loc[col, "mean_equal"] = mean_test.pvalue > 0.05

    return equality_tests


def beta_binomial_posterior(sample1, sample2):
    """
    Returns the posterior parameters of the Beta-Binomial model with uniform uninformative prior.

    Parameters
    ----------
    sample1 : array-like
        Sample of the first group.
    sample2 : array-like
        Sample of the second group.
    
    Returns
    -------
    alpha_a, beta_a, alpha_b, beta_b, prob_b_greater : float
        Posterior parameters of the Beta-Binomial model and the probability that the second group is greater.
    """
    # Prior parameters
    alpha_0, beta_0 = 1, 1

    # Sample size and number of successes
    x_a, n_a = sum(sample1 > 0), len(sample1)
    x_b, n_b = sum(sample2 > 0), len(sample2)

    # Posterior parameters
    alpha_a = alpha_0 + x_a
    beta_a = beta_0 + n_a - x_a
    alpha_b = alpha_0 + x_b
    beta_b = beta_0 + n_b - x_b

    # Generate random samples from the two beta distributions
    np.random.seed(666)
    num_samples = 10000
    samples_a = np.random.beta(alpha_a, beta_a, size=num_samples)
    samples_b = np.random.beta(alpha_b, beta_b, size=num_samples)

    # Probability that after Kretinsky, Sparta has a higher penalty rate
    prob_b_greater = (samples_a < samples_b).mean()


    return alpha_a, beta_a, alpha_b, beta_b, prob_b_greater


def fit_regression_models(dt):
    # Fit gamma regression model for referee interventions per attack
    model_gamma = sm.formula.glm(
        formula="ref_interv_per_attack ~ team + team_opp + hga + kretinsky + kretinsky_sparta + kretinsky_sparta_opp + kretinsky_slavia + kretinsky_slavia_opp",
        data=dt,
        subset=dt["ref_interv_per_attack"] > 0,
        family=sm.families.Gamma(link = sm.families.links.log())
    ).fit()
    print("\n\n~~~~~~~ GAMMA REGRESSION ~~~~~~~")
    print(model_gamma.summary())
    filtr = dt["ref_interv_per_attack"] <= 0
    print(f"Rows deleted due to non-positivity: {sum(filtr)}")

    # Fit regression model for referee interventions difference per attack
    model_lm = sm.formula.glm(
        formula="ref_interv_per_attack_diff ~ team + hga + kretinsky + kretinsky_sparta + kretinsky_slavia",
        data=dt,
        family=sm.families.Gaussian()
    ).fit()
    print("\n\n~~~~~~~ LINEAR REGRESSION ~~~~~~~")
    print(model_lm.summary())

    return model_gamma, model_lm

