import pandas as pd
import numpy as np

from ema_workbench import Policy, MultiprocessingEvaluator, ema_logging, ScalarOutcome, Scenario, SequentialEvaluator
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.analysis import parcoords, prim
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from ema_workbench import save_results

from scipy.spatial.distance import pdist, squareform
from sklearn import preprocessing
import itertools
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import functools
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)
import random


if __name__ == '__main__':

    ema_logging.log_to_stderr(ema_logging.INFO)

    #load and print the 403 solutions found during first optimization iteration (nfe = 10.000)
    results = pd.read_csv('second_optimization_results.csv')

    #Make a list of all the column titles
    #
    # so we can filter the ones of interest
    cols = [results.columns]
    cols_of_interest = ['0_RfR 0', '0_RfR 1', '0_RfR 2', '1_RfR 0', '1_RfR 1', '1_RfR 2',
                        'A.1 Total Costs', 'A.1_Expected Number of Deaths',
                        'A.2 Total Costs', 'A.2_Expected Number of Deaths',
                         'A.4_Expected Number of Deaths']

    interests = results[cols_of_interest]

    #Filters the best performing 20% of scenarios in our actors interest
    candidates = interests.nsmallest(81, ['A.2_Expected Number of Deaths', 'A.1_Expected Number of Deaths'])

    candidates['better'] = np.where((candidates['A.1_Expected Number of Deaths'] <= candidates['A.4_Expected Number of Deaths']) &
                                   (candidates['A.2_Expected Number of Deaths'] <= candidates['A.4_Expected Number of Deaths']),
                                   candidates['A.1_Expected Number of Deaths'],
                                   np.nan)
    candidates = candidates.dropna()

    candidates['RfR ring 1'] = np.where((candidates['0_RfR 0'] + candidates['0_RfR 1'] + candidates['0_RfR 1']) > 0,
                                        True, False)

    candidates['RfR ring 2'] = np.where((candidates['1_RfR 0'] + candidates['1_RfR 1'] + candidates['1_RfR 1']) > 0,
                                        True, False)

    candidates['possible_sol_ring1'] = (candidates['RfR ring 1'] == True) & (candidates['A.1 Total Costs'] < 1e8)
    candidates['possible_sol_ring2'] = (candidates['RfR ring 2'] == True) & (candidates['A.2 Total Costs'] < 1.8e8)

    RfR_sol = candidates.drop(candidates[(candidates.possible_sol_ring1 == False) | (candidates.possible_sol_ring2 == False)].index)

    df_policies = results.copy()

    policies_index = RfR_sol.index.tolist()

    df_policies = df_policies.loc[policies_index, :]
    try:
        df_policies = df_policies.drop(columns=["Unnamed: 0"])
    except Exception as e:
        print(f"Exception occurred while trying to drop column --> {e}\n"
              f"Column does not exist.")

    df_policies = df_policies.iloc[:, :31].reset_index(drop=True)

    dict_policies = df_policies.to_dict('index')

    policies = []
    for dict_index in range(len(dict_policies)):
        policy_dict = dict_policies[dict_index]
        policies.append(Policy(f"pol_{dict_index}", **policy_dict))

    model, array = get_model_for_problem_formulation(3)
    n_scenarios = 12
    with MultiprocessingEvaluator(model) as evaluator:
        results_w_pol = evaluator.perform_experiments(scenarios=n_scenarios, policies=policies)


    def s_to_n(data, direction) :
        mean = np.mean(data)
        std = np.std(data)

        if direction == ScalarOutcome.MAXIMIZE :
            return mean / std
        else :
            return mean * std


    experiments, outcomes = results_w_pol

    overall_scores = {}
    for policy in np.unique(experiments['policy']) :
        scores = {}

        logical = experiments['policy'] == policy

        for outcome in model.outcomes :
            value = outcomes[outcome.name][logical]
            sn_ratio = s_to_n(value, outcome.kind)
            scores[outcome.name] = sn_ratio
        overall_scores[policy] = scores
    scores = pd.DataFrame.from_dict(overall_scores).T


    def calculate_regret(data, best) :
        return np.abs(best - data)


    overall_regret = {}
    max_regret = {}
    for outcome in model.outcomes :
        policy_column = experiments['policy']

        # create a DataFrame with all the relevent information
        # i.e., policy, scenario_id, and scores
        data = pd.DataFrame({outcome.name : outcomes[outcome.name],
                             "policy" : experiments['policy'],
                             "scenario" : experiments['scenario']})

        # reorient the data by indexing with policy and scenario id
        data = data.pivot(index='scenario', columns='policy')

        # flatten the resulting hierarchical index resulting from
        # pivoting, (might be a nicer solution possible)
        data.columns = data.columns.get_level_values(1)

        # we need to control the broadcasting.
        # max returns a 1d vector across scenario id. By passing
        # np.newaxis we ensure that the shape is the same as the data
        # next we take the absolute value
        #
        # basically we take the difference of the maximum across
        # the row and the actual values in the row
        #
        outcome_regret = (data.max(axis=1)[:, np.newaxis] - data).abs()

        overall_regret[outcome.name] = outcome_regret
        max_regret[outcome.name] = outcome_regret.max()

    max_regret = pd.DataFrame(max_regret)
    sns.heatmap(max_regret / max_regret.max(), cmap='viridis', annot=True)
    # plt.show()

    colors = sns.color_palette()

    data = max_regret

    # makes it easier to identify the policy associated with each line
    # in the parcoords plot
    # data['policy'] = data.index.astype("float64")

    limits = parcoords.get_limits(data)
    limits.loc[0, ["A.1 Total Costs", "A.1_Expected Number of Deaths",
                   "A.2 Total Costs", "A.2_Expected Number of Deaths",
                   "A.3 Total Costs", "A.3_Expected Number of Deaths",
                   "A.4 Total Costs", "A.4_Expected Number of Deaths",
                   "A.5 Total Costs", "A.5_Expected Number of Deaths",
                   "RfR Total Costs", "Expected Evacuation Costs"]] = 0

    paraxes = parcoords.ParallelAxes(limits)
    for i, (index, row) in enumerate(data.iterrows()) :
        paraxes.plot(row.to_frame().T, label=str(index), color=colors[i])
    paraxes.legend()

    # plt.show()

    policy_regret = defaultdict(dict)
    for key, value in overall_regret.items() :
        for policy in value :
            policy_regret[policy][key] = value[policy]

    # this generates a 2 by 2 axes grid, with a shared X and Y axis
    # accross all plots
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 10),
                             sharey=True, sharex=True)

    # to ensure easy iteration over the axes grid, we turn it
    # into a list. Because there are four plots, I hard coded
    # this.
    axes = [axes[0, 0], axes[0, 1],
            axes[1, 0], ]

    # zip allows us to zip together the list of axes and the list of
    # key value pairs return by items. If we iterate over this
    # it returns a tuple of length 2. The first item is the ax
    # the second items is the key value pair.
    for ax, (policy, regret) in zip(axes, policy_regret.items()) :
        data = pd.DataFrame(regret)

        # we need to scale the regret to ensure fair visual
        # comparison. We can do that by divding by the maximum regret
        data = data / max_regret.max(axis=0)
        sns.boxplot(data=data, ax=ax)

        # removes top and left hand black outline of axes
        sns.despine()

        # ensure we know which policy the figure is for
        ax.set_title(str(policy))
    # plt.show()
    experiments_copy = experiments.copy()
    for col in experiments.columns:
        if "RfR" in col or "Days" in col or "DikeIncrease" in col or "policy" in col or "scenario" in col:
            experiments_copy = experiments_copy.drop(columns=col)
        elif "model" not in col:
            experiments_copy[col] = pd.to_numeric(experiments_copy[col])

    x = experiments_copy
    y = outcomes["A.1 Total Costs"] < 7e7

    prim_alg = prim.Prim(x, y, threshold=0.5)
    box = prim_alg.find_box()

    # box.inspect(3)

    scens_in_box = experiments.iloc[box.yi]
    outcomes_in_box = {k : v[box.yi] for k, v in outcomes.items()}


    # save_results((scens_in_box, outcomes_in_box), 'mordm_42.tar.gz')

    data = pd.DataFrame({k : v[y] for k, v in outcomes.items()})

    max_total_1 = max(data["A.1 Total Costs"])
    max_total_2 = max(data["A.2 Total Costs"])
    max_death_1 = max(data["A.1_Expected Number of Deaths"])
    max_death_2 = max(data["A.2_Expected Number of Deaths"])

    indices = []
    indices.append(data.index[data["A.1 Total Costs"] == max_total_1].tolist()[0])
    indices.append(data.index[data["A.2 Total Costs"] == max_total_2].tolist()[-1])
    indices.append(data.index[data["A.1_Expected Number of Deaths"] == max_death_1].tolist()[0])
    indices.append(data.index[data["A.2_Expected Number of Deaths"] == max_death_2].tolist()[-1])

    selected = experiments.loc[indices, ['discount rate 0', 'discount rate 1', 'discount rate 2',
                                         'A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_pfail', 'A.1_Brate', 'A.2_Bmax',
                                         'A.2_pfail', 'A.2_Brate', 'A.3_Bmax', 'A.3_pfail', 'A.3_Brate', 'A.4_Bmax',
                                         'A.4_pfail', 'A.4_Brate', 'A.5_Bmax', 'A.5_pfail', 'A.5_Brate']]

    scenarios = [Scenario(f"{index}", **row) for index, row in selected.iterrows()]

    experiments_of_interest = experiments.loc[y]
    outcomes_df = pd.DataFrame({k : v[y] for k, v in outcomes.items()})

    # normalize outcomes on unit interval to ensure equal weighting of outcomes
    x = outcomes_df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    normalized_outcomes = pd.DataFrame(x_scaled, columns=outcomes_df.columns)

    n_scen = experiments.loc[y].shape[0]
    indices = range(n_scen)
    set_size = 12
    combinations = itertools.combinations(indices, set_size)
    combinations = list(combinations)
    print(len(combinations))



    def evaluate_diversity_single(indices, distances, weight=0.5, distance='euclidean') :
        '''
        takes the outcomes and selected scenario set (decision variables),
        returns a single 'diversity' value for the scenario set.
        outcomes : outcomes dictionary of the scenario ensemble
        decision vars : indices of the scenario set
        weight : weight given to the mean in the diversity metric. If 0, only minimum; if 1, only mean
        '''
        i, j = [e for e in zip(*itertools.combinations(indices, 2))]
        subset_distances = distances[i, j]
        minimum = np.min(subset_distances)
        mean = np.mean(subset_distances)
        diversity = (1 - weight) * minimum + weight * mean

        return [diversity]


    def find_maxdiverse_scenarios(distances, combinations) :
        scores = []
        for indices in combinations :
            diversity = evaluate_diversity_single(indices, distances)
            scores.append((diversity, indices))

        return scores



    distances = squareform(pdist(normalized_outcomes.values))

    cores = 4
    partial_function = functools.partial(find_maxdiverse_scenarios, distances)

    with ThreadPoolExecutor(max_workers=2) as executor :
        worker_data = np.array_split(combinations, cores)
        results = [e for e in executor.map(partial_function, worker_data)]
        results = list(itertools.chain.from_iterable(results))

    results.sort(key=lambda entry : entry[0], reverse=True)
    most_diverse = results[0]

    from ema_workbench import Scenario

    selected = experiments.loc[most_diverse[1], ['discount rate 0', 'discount rate 1', 'discount rate 2',
                                         'A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_pfail', 'A.1_Brate', 'A.2_Bmax',
                                         'A.2_pfail', 'A.2_Brate', 'A.3_Bmax', 'A.3_pfail', 'A.3_Brate', 'A.4_Bmax',
                                         'A.4_pfail', 'A.4_Brate', 'A.5_Bmax', 'A.5_pfail', 'A.5_Brate']]
    scenarios = [Scenario(f"{index}", **row) for index, row in selected.iterrows()]


    def optimize(scenario, nfe, model, converge_metrics, epsilons) :

        with MultiprocessingEvaluator(model) as evaluator :
            results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                      convergence=convergence_metrics,
                                                      epsilons=epsilons,
                                                      reference=scenario)
        return results, convergence


    results = []
    for scenario in scenarios :
        convergence_metrics = [HyperVolume(minimum=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], maximum=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                               EpsilonProgress()]
        # epsilons = [1e7, 0.01, 1e8, 1e4]
        epsilons = [1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e8, 1e4]

        results.append(optimize(scenario, 1e4, model, convergence_metrics, epsilons))

    for i, (results, convergence) in enumerate(results) :
        save_results((results, convergence), f'mordm_last_{i}.tar.gz')