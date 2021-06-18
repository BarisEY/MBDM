# Imports:
# External
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from SALib.analyze import sobol
# EMA Workbench
from ema_workbench import load_results, Policy, MultiprocessingEvaluator, ema_logging
from ema_workbench.analysis import parcoords
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS
# Internal
from quaquel_github_files_kopie.problem_formulation import get_model_for_problem_formulation

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    experiments, outcomes = load_results("../final_data/deep_uncertainty.tar.gz")

    # Setting different threshold. For the final findings, threshold_2 was used.
    thresholds = {'A.1 Total Costs': 1e8, 'A.1_Expected Number of Deaths' : 0.0, 'A.2 Total Costs': 1e8, 'A.2_Expected Number of Deaths' : 0.0}
    thresholds_2 = {'A.1_Expected Number of Deaths' : 0.0, 'A.2_Expected Number of Deaths' : 0.0, '0_RfR 0': 0,
                    '0_RfR 1': 0, '0_RfR 2': 0, '1_RfR 0': 0, '1_RfR 1': 0, '1_RfR 2': 0}

    # Calculating scores of the different policies
    overall_scores = {}
    for policy in experiments.policy.unique() :
        logical = experiments.policy == policy
        scores = {}
        for k, v in outcomes.items() :
            try :
                # Here, both thresholds and thresholds_2 were used on different occasions.
                a = v[logical] <= thresholds_2[k]
                n = np.sum(a)
            except KeyError :
                continue
            scores[k] = n / 1000
        overall_scores[policy] = scores
    overall_scores = pd.DataFrame(overall_scores).T

    # Taking the mean of each column and plotting it on a heatmap
    for col in overall_scores.columns:
        overall_scores[col] = overall_scores[col].mean()
    overall_scores = overall_scores.loc['scenario 0 option 0', :].to_frame().rename(columns={"scenario 0 option 0": "mean"}).T
    sns.heatmap(overall_scores, cmap='viridis', annot=True)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.show()

    # Calculating the regret for each of the unique policies
    overall_scores = {}
    regret = []
    for scenario in experiments.policy.unique() :
        logical = experiments.policy == scenario
        temp_results = {k : v[logical] for k, v in outcomes.items()}
        temp_results = pd.DataFrame(temp_results)
        temp_experiments = experiments[experiments.policy == scenario]

        best = temp_results.min()
        worst = temp_results.max()
        scenario_regret = worst - best
        scenario_regret['policy'] = temp_experiments['policy'].iloc[0]
        regret.append(scenario_regret)

    # Turn the list into a dataframe and sort on policies. Then, a for loop standardized regret in order to get values
    # between 0 and 10, giving a fairer comparison.
    regret = pd.DataFrame(regret)
    maxregret = regret.groupby('policy').max()
    for col in maxregret.columns:
        a = maxregret[col].max()
        try:
            b = math.log(a, 10)
        except:
            b = 0
        b = math.floor(b)
        if b > 1:
            maxregret[col] = maxregret[col] / (10**b)

    # plot the max_regret on a parcoords plot.
    limits = parcoords.get_limits(maxregret)
    paraxes = parcoords.ParallelAxes(maxregret)
    paraxes.plot(maxregret)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.show()

    # Filter out the solution with the number of deaths lower than 0.05 for both A1 and A2 and plot the parcoords plot
    # again with the limits of the previous plot in order to compare it with the other solutions.
    maxregret_filtered = maxregret.loc[(maxregret["A.1_Expected Number of Deaths"] < 0.05)
                                       & (maxregret["A.2_Expected Number of Deaths"] < 0.05)]
    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(maxregret_filtered)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.show()

    # Having filtered this policy out, its name was found and the performance of this policy across all the different
    # scenarios was plotted.
    scenario_and_option = maxregret_filtered.first_valid_index()
    policy = experiments.loc[experiments["policy"] == scenario_and_option]

    # Here, the appropriate indices for this particular policy are extracted from the original outcomes dataframe.
    indices = [int(s) for s in scenario_and_option.split() if s.isdigit()]
    indices = range((indices[0] * 5 * 1000 + indices[1] * 1000), indices[0] * 5 * 1000 + (indices[1]+1) * 1000)
    df_outcomes = pd.DataFrame(outcomes)
    all_scenarios = df_outcomes.loc[indices, :]

    limits = parcoords.get_limits(all_scenarios)
    paraxes = parcoords.ParallelAxes(limits)
    paraxes.plot(all_scenarios)
    plt.gcf().subplots_adjust(bottom=0.5)
    plt.show()

    # Finally, a sensitivity analysis was performed on this policy
    df_policies = policy.iloc[:, 19:50]
    df_policies = df_policies.to_frame().T.reset_index(drop=True)

    dict_policies = df_policies.to_dict('index')

    # A policy was created from the dataframe with all the policies (which is technically just 1 policy)
    policies = []
    for dict_index in range(len(dict_policies)):
        policy_dict = dict_policies[dict_index]
        policies.append(Policy(f"pol_{dict_index}", **policy_dict))

    model, array = get_model_for_problem_formulation(3)

    problem = get_SALib_problem(model.uncertainties)

    # Number of experiments set at: 2^10 = 1024
    n_exp = 1024

    with MultiprocessingEvaluator(model) as evaluator:
        experiments_sobol, results_sobol = evaluator.perform_experiments(scenarios=n_exp,
                                                                         uncertainty_sampling=SOBOL,
                                                                         policies=policies)

    # There were some difficulties plotting this in this file, the results were thus saved as a csv and used in the
    # plotting file.
    sobol_results = {}
    for policy in experiments_sobol.policy.unique() :
        logical = experiments_sobol.policy == policy
        for outcome in results_sobol:
            y = results_sobol[outcome][logical]
            indices = sobol.analyze(problem, y)
            sobol_results[policy] = indices
            np.savetxt(f"{outcome}.csv", y, delimiter=",")
