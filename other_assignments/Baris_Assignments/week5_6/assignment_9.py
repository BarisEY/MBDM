import pandas as pd
from ema_workbench import (load_results, MultiprocessingEvaluator, ema_logging, Model, RealParameter, ScalarOutcome,
                           Constant, Policy)
from ema_workbench.analysis import (parcoords, prim)
from ema_workbench.em_framework.optimization import (HyperVolume, EpsilonProgress)

from Week_5_6_quaq.dps_lake_model import lake_model as lake_problem

import matplotlib.pyplot as plt

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    lake_model = Model('lakemodel', function=lake_problem)
    lake_model.time_horizon = 100

    lake_model.uncertainties = [RealParameter("mean", 0.01, 0.05),
                                RealParameter("stdev", 0.001, 0.005),
                                RealParameter("b", 0.1, 0.45),
                                RealParameter("q", 2, 4.5),
                                RealParameter("delta", 0.93, 0.99)]

    lake_model.levers = [RealParameter("c1", -2, 2),
                         RealParameter("c2", -2, 2),
                         RealParameter("r1", 0, 2),
                         RealParameter("r2", 0, 2),
                         RealParameter("w1", 0, 1)]

    lake_model.outcomes = [ScalarOutcome("max_P", kind=ScalarOutcome.MINIMIZE, expected_range=(0, 5)),
                           ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 2)),
                           ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1)),
                           ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE, expected_range=(0, 1))]

    lake_model.constants = [Constant("alpha", 0.4),
                            Constant("nsamples", 100),
                            Constant("myears", 100)]

    n_scenarios = 1000

    results, outcomes = load_results('results/selected_results.tar.gz')

    df_outcomes = pd.DataFrame(outcomes)
    median = df_outcomes['utility'].median()

    df_median = df_outcomes.iloc[((df_outcomes['utility']-median).abs().argsort()[:2])]
    df_outcomes = df_outcomes.loc[(df_outcomes['utility'] == max(df_outcomes['utility'])) |
                                  (df_outcomes['utility'] == min(df_outcomes['utility']))]
    df_outcomes = pd.concat([df_outcomes, df_median])
    index_list = df_outcomes.index.tolist()
    results = results.iloc[index_list].reset_index(drop=True)
    df_outcomes.reset_index(drop=True, inplace=True)

    convergence_metrics = [HyperVolume.from_outcomes(lake_model.outcomes),
                           EpsilonProgress()]

    with MultiprocessingEvaluator(lake_model) as evaluator:
        results_, convergence = evaluator.optimize(nfe=2000, searchover='levers',
                                                   epsilons=[0.1, 0.1, 0.05, 0.05],
                                                   convergence=convergence_metrics)

    fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, figsize=(8,4))
    ax1.plot(convergence.nfe, convergence.epsilon_progress)
    ax1.set_ylabel('$\epsilon$-progress')
    ax2.plot(convergence.nfe, convergence.hypervolume)
    ax2.set_ylabel('hypervolume')

    ax1.set_xlabel('number of function evaluations')
    ax2.set_xlabel('number of function evaluations')
    # plt.show()

    outcomes = results_.loc[:, ['max_P', 'utility', "inertia", 'reliability']]
    limits = parcoords.get_limits(outcomes)
    axes = parcoords.ParallelAxes(limits)
    axes.plot(outcomes)

    # we invert this axis so direction of desirability is the same
    axes.invert_axis('max_P')
    # plt.show()

    policy_results = results_.drop(['max_P', 'utility', "inertia", 'reliability'], axis=1)
    new_results_dict = results_.to_dict('index')
    policies = []
    for dict_index in range(len(new_results_dict)) :
        policy_dict = new_results_dict[dict_index]
        policies.append(Policy(f"pol_{dict_index}", **policy_dict))

    n_scenarios = 10
    LHS = 'lhs'
    with MultiprocessingEvaluator(lake_model) as evaluator:
        results__, outcomes__ = evaluator.perform_experiments(scenarios=n_scenarios,
                                                              policies=policies,
                                                              uncertainty_sampling=LHS)


    def calculate_regret(outcomes_w_policy, results_w_policy) :
        df_outcomes_w_policy = pd.DataFrame(outcomes_w_policy)

        results_w_policy.rename(columns={"max_P": "max_P_initial",
                                         "inertia": "inertia_initial",
                                         "utility": "utility_initial",
                                         "reliability": "reliability_initial"},
                                inplace=True)
        df_full = pd.concat([df_outcomes_w_policy, results_w_policy], axis=1)
        df_regret = pd.DataFrame(columns=['policy_num', 'max_P', 'utility', "inertia", 'reliability'])
        for unique_item in df_full['policy'].unique() :
            df_partial = df_full.loc[df_full['policy'] == unique_item]
            df_partial.reset_index(drop=True, inplace=True)
            max_max_P = max(df_partial['max_P'])
            max_utility = max(df_partial['utility'])
            max_inertia = max(df_partial['inertia'])
            max_reliability = max(df_partial['reliability'])
            for index in range(len(df_partial)) :
                loc_max_p = df_partial.loc[index, 'max_P']
                loc_utility = df_partial.loc[index, 'utility']
                loc_inertia = df_partial.loc[index, 'inertia']
                loc_reliability = df_partial.loc[index, 'reliability']
                new_row = {'policy_num' : unique_item,
                           'max_P' : max_max_P - loc_max_p,
                           'utility' : max_utility - loc_utility,
                           'inertia' : max_inertia - loc_inertia,
                           'reliability' : max_reliability - loc_reliability}
                df_regret = df_regret.append(new_row, ignore_index=True)

        columns_to_sum = ['max_P', 'utility', "inertia", 'reliability']
        df_regret['total_regret'] = df_regret[columns_to_sum].sum(axis=1)

        return df_regret

    df_regret = calculate_regret(outcomes__, results__)

    df_regret_plot = pd.DataFrame(columns=['policy', 'max_regret', 'min_regret'])
    for unique_item in df_regret['policy_num'].unique() :
        df_partial = df_regret.loc[df_regret['policy_num'] == unique_item]
        new_row = {'policy' : unique_item,
                   'max_regret' : max(df_partial['total_regret']),
                   'min_regret' : min(df_partial['total_regret'])}
        df_regret_plot = df_regret_plot.append(new_row, ignore_index=True)

    df_outcomes__ = pd.DataFrame(outcomes__)
    limits = parcoords.get_limits(df_outcomes__)
    axes = parcoords.ParallelAxes(limits)
    axes.plot(df_outcomes__)
    # plt.show()

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111)
    df_regret_plot.plot(y='max_regret', style='o', use_index=True)
    plt.show()

    z = 1