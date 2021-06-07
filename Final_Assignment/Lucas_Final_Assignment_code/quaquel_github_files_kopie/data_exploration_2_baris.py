import pandas as pd
import numpy as np

from ema_workbench import Policy, MultiprocessingEvaluator, ema_logging
from problem_formulation import get_model_for_problem_formulation
from ema_workbench.analysis import parcoords
import matplotlib.pyplot as plt


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

    model, array = get_model_for_problem_formulation(30)
    n_scenarios = 5

    with MultiprocessingEvaluator(model) as evaluator:
        results_w_pol, outcomes_w_policy = evaluator.perform_experiments(scenarios=n_scenarios, policies=policies)

    df_outcomes_w_policy = pd.DataFrame(outcomes_w_policy)

    df_full = pd.concat([df_outcomes_w_policy, results_w_pol], axis=1)

    signal_noise_r = []
    for policy_analyzed in range(len(policies)):
        df = (df_outcomes_w_policy.tail((len(policies) - policy_analyzed)*n_scenarios)).head(n_scenarios)
        df.loc["mean"] = df.mean()
        df.loc["stdev"] = df.std()
        df.loc["ratio"] = df.loc["mean"] / df.loc["stdev"]
        partial_dict = df.loc["ratio"].to_dict()
        signal_noise_r.append(partial_dict)

    df_plot_signals = pd.DataFrame(signal_noise_r)

    limits = parcoords.get_limits(df_plot_signals)

    df_plot_signals.replace([np.inf, -np.inf], 0, inplace=True)
    limits.replace([np.inf, -np.inf], 0, inplace=True)

    axes = parcoords.ParallelAxes(limits)
    axes.plot(df_plot_signals)
    plt.show()


    def calculate_regret() :
        df_regret = pd.DataFrame(columns=["policy_num", "Expected Annual Damage", "Total Investment Costs",
                                          "Expected Number of Deaths"])
        for unique_item in df_full["policy"].unique() :
            df_partial = df_full.loc[df_full["policy"] == unique_item]
            df_partial.reset_index(drop=True, inplace=True)
            max_damage = max(df_partial["Expected Annual Damage"])
            max_cost = max(df_partial["Total Investment Costs"])
            max_death = max(df_partial["Expected Number of Deaths"])
            for index in range(len(df_partial)) :
                new_row = {'policy_num' : unique_item,
                           'Expected Annual Damage' : max_damage - df_partial.loc[index, 'Expected Annual Damage'],
                           'Total Investment Costs' : max_cost - df_partial.loc[index, 'Total Investment Costs'],
                           'Expected Number of Deaths' : max_death - df_partial.loc[index, 'Expected Number of Deaths']}
                df_regret = df_regret.append(new_row, ignore_index=True)

        columns_to_sum = ['Expected Annual Damage', "Total Investment Costs", 'Expected Number of Deaths']
        df_regret['total_regret'] = df_regret[columns_to_sum].sum(axis=1)

        return df_regret


    df_regret = calculate_regret()

    df_regret_plot = pd.DataFrame(columns=['policy', 'max_regret', 'min_regret'])
    for unique_item in df_regret['policy_num'].unique() :
        df_partial = df_regret.loc[df_regret['policy_num'] == unique_item]
        new_row = {'policy' : unique_item,
                   'max_regret' : max(df_partial['total_regret']),
                   'min_regret' : min(df_partial['total_regret'])}
        df_regret_plot = df_regret_plot.append(new_row, ignore_index=True)

    df_regret_plot.sort_values(['max_regret'], inplace=True)
    best_policy = df_regret_plot.loc[0, 'policy']

    print(best_policy)