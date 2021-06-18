# Imports:
# External:
import itertools
import functools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import pdist, squareform

# EMA Workbench
from ema_workbench import Policy, MultiprocessingEvaluator, ema_logging, ScalarOutcome, Scenario, save_results
from ema_workbench.analysis import parcoords, prim
from ema_workbench.em_framework.optimization import HyperVolume, EpsilonProgress

# Internal:
from quaquel_github_files_kopie.problem_formulation import get_model_for_problem_formulation

if __name__ == '__main__':
    # NOTE: This file is the same as the notebook, up to the line with '='

    ema_logging.log_to_stderr(ema_logging.INFO)

    # Load and print the 403 solutions found during second optimization iteration (nfe = 10.000)
    results = pd.read_csv('../final_data/second_optimization_results.csv')

    # Make a list of all the column titles, so we can filter the ones of interest
    cols = [results.columns]
    cols_of_interest = ['0_RfR 0', '0_RfR 1', '0_RfR 2', '1_RfR 0', '1_RfR 1', '1_RfR 2',
                        'A.1 Total Costs', 'A.1_Expected Number of Deaths',
                        'A.2 Total Costs', 'A.2_Expected Number of Deaths',
                         'A.4_Expected Number of Deaths']
    interests = results[cols_of_interest]

    # Filters the best performing 20% of scenarios in our actors interest
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

    # Copy the results to a new DataFrame
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

    # ==================================================================================================================
    # The notebook goes up to here.

    model, array = get_model_for_problem_formulation(3)
    n_scenarios = 12
    with MultiprocessingEvaluator(model) as evaluator:
        results_w_pol = evaluator.perform_experiments(scenarios=n_scenarios, policies=policies)

    # Function that gives the signal to noise ratio as output.
    def s_to_n(data, direction) :
        mean = np.mean(data)
        std = np.std(data)

        if direction == ScalarOutcome.MAXIMIZE :
            return mean / std
        else :
            return mean * std

    experiments, outcomes = results_w_pol

    # Function that calculates regret for the different outcomes.
    overall_regret = {}
    max_regret = {}
    for outcome in model.outcomes :
        policy_column = experiments['policy']

        # Create a DataFrame with all the relevant information
        data = pd.DataFrame({outcome.name : outcomes[outcome.name],
                             "policy" : experiments['policy'],
                             "scenario" : experiments['scenario']})

        # Reorient the data by indexing with policy and scenario id
        data = data.pivot(index='scenario', columns='policy')

        # Flatten the resulting hierarchical index resulting from pivoting
        data.columns = data.columns.get_level_values(1)

        # Calculating the regret of the outcome
        outcome_regret = (data.max(axis=1)[:, np.newaxis] - data).abs()
        overall_regret[outcome.name] = outcome_regret
        max_regret[outcome.name] = outcome_regret.max()

    # Turn max_regret to a Dataframe and calculate the ratio, the overal mean and the means of DR 1 & 2.
    max_regret_df = pd.DataFrame(max_regret)
    max_regret_ratio = max_regret_df / max_regret_df.max()
    max_regret_ratio['mean'] = max_regret_ratio.mean(axis=1)

    cols_of_interest_2 = ["A.1 Total Costs", "A.1_Expected Number of Deaths", "A.2 Total Costs", "A.2_Expected Number of Deaths"]
    max_regret_ratio["Mean DR1 & DR2"] = max_regret_ratio[cols_of_interest_2].mean(axis=1)

    # Plotting a heatmap with the viridis colourmap of the regret ratios.
    sns.heatmap(max_regret_ratio, cmap='viridis', annot=True)
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.show()

    # Create a parcoords plot of the same output as a above, this will give a different perspective and see
    # where all the solution are located across all the outcomes.
    colors = sns.color_palette()
    data = max_regret_df
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
    plt.tight_layout()
    plt.show()

    # Creating a boxplot of all the different solutions found above.
    policy_regret = defaultdict(dict)
    for key, value in overall_regret.items() :
        for policy in value :
            policy_regret[policy][key] = value[policy]

    # this generates a 2 by 2 axes grid, with a shared X and Y axis across all plots
    fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(10, 10))

    # In order to ensure iteration over the axes grid, we turn it into a list.
    axes = [axes[0, 0], axes[0, 1], axes[1, 0], ]

    # zip allows us to zip together the list of axes and the list of key value pairs return by items. If we iterate over
    # this it returns a tuple of length 2. The first item is the ax the second items is the key value pair.
    for ax, (policy, regret) in zip(axes, policy_regret.items()) :
        data = pd.DataFrame(regret)

        # Standardizing the regret function, in order to ensure a fair comparison between all outcomes.
        data = data / max_regret_df.max(axis=0)
        sns.boxplot(data=data, ax=ax)

        # removes top and left hand black outline of axes
        sns.despine()

        # ensure we know which policy the figure is for
        ax.set_title(str(policy))
        plt.xticks(rotation=90)
    plt.tight_layout()
    plt.tick_params(axis='x', rotation=90)
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.show()

    # Select some outcomes based on the PRIM function.
    experiments_copy = experiments.copy()
    for col in experiments.columns:
        if "RfR" in col or "Days" in col or "DikeIncrease" in col or "policy" in col or "scenario" in col:
            experiments_copy = experiments_copy.drop(columns=col)
        elif "model" not in col:
            experiments_copy[col] = pd.to_numeric(experiments_copy[col])

    # The main requirement is to keep the costs low, is done by filtering for the total costs of A.1
    x = experiments_copy
    y = outcomes["A.1 Total Costs"] < 1.2e8

    #prim_alg = prim.Prim(x, y, threshold=0.5)
    #box = prim_alg.find_box()

    #a = box.inspect_tradeoff(0)

    #box.inspect(0)
    #box.select(0)

    #scens_in_box = experiments.iloc[box.yi]
    #outcomes_in_box = {k : v[box.yi] for k, v in outcomes.items()}

    # Saving the results, in case future use is needed.
    #save_results((scens_in_box, outcomes_in_box), 'mordm_0.tar.gz')

    # Using the same filter as for PRIM, we try to filter out the results and create scenarios for these.
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

    # Create a list of combinations based on the indices and set_size
    n_scen = experiments.loc[y].shape[0]
    indices = range(n_scen)
    set_size = 12
    combinations = itertools.combinations(indices, set_size)
    combinations = list(combinations)

    # Function to evaluate diversity on a single axis
    def evaluate_diversity_single(indices, distances, weight=0.5, distance='euclidean') :
        # takes the outcomes and selected scenario set (decision variables), returns a single 'diversity' value for the
        # scenario set.
        i, j = [x for x in zip(*itertools.combinations(indices, 2))]
        subset_distances = distances[i, j]
        minimum = np.min(subset_distances)
        mean = np.mean(subset_distances)
        diversity = (1 - weight) * minimum + weight * mean

        return [diversity]

    # Function to evaluate diversity for whole set.
    def find_maxdiverse_scenarios(distances, combinations) :
        scores = []
        for indices in combinations :
            diversity = evaluate_diversity_single(indices, distances)
            scores.append((diversity, indices))

        return scores

    distances = squareform(pdist(normalized_outcomes.values))
    cores = 4
    partial_function = functools.partial(find_maxdiverse_scenarios, distances)

    # Since finding the combinations takes a lot of work, a ThreadPoolExecutor was used to speed it up.
    with ThreadPoolExecutor(max_workers=cores) as executor :
        worker_data = np.array_split(combinations, cores)
        results = [e for e in executor.map(partial_function, worker_data)]
        results = list(itertools.chain.from_iterable(results))

    # Optimizing the model using the most diverse scenarios.
    results.sort(key=lambda entry : entry[0], reverse=True)
    most_diverse = results[0]

    selected = experiments.loc[most_diverse[1], ['discount rate 0', 'discount rate 1', 'discount rate 2',
                                         'A.0_ID flood wave shape', 'A.1_Bmax', 'A.1_pfail', 'A.1_Brate', 'A.2_Bmax',
                                         'A.2_pfail', 'A.2_Brate', 'A.3_Bmax', 'A.3_pfail', 'A.3_Brate', 'A.4_Bmax',
                                         'A.4_pfail', 'A.4_Brate', 'A.5_Bmax', 'A.5_pfail', 'A.5_Brate']]
    scenarios = [Scenario(f"{index}", **row) for index, row in selected.iterrows()]

    # Function that will be run in order to optimize each scenario
    def optimize(scenario, nfe, model, converge_metrics, epsilons) :

        with MultiprocessingEvaluator(model) as evaluator :
            results, convergence = evaluator.optimize(nfe=nfe, searchover='levers',
                                                      convergence=convergence_metrics,
                                                      epsilons=epsilons,
                                                      reference=scenario)
        return results, convergence

    # For loop that will iterate over the scenarios and save the results in a list
    results = []
    for scenario in scenarios :
        convergence_metrics = [HyperVolume(minimum=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                           maximum=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                               EpsilonProgress()]
        # epsilons = [1e7, 0.01, 1e8, 1e4]
        epsilons = [1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e7, 0.01, 1e8, 1e4]

        results.append(optimize(scenario, 1e4, model, convergence_metrics, epsilons))

    # Results are lastly saved one by one.
    for i, (results, convergence) in enumerate(results) :
        save_results((results, convergence), f'mordm_last_{i}.tar.gz')