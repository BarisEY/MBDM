import pandas as pd

from Week_5_6_quaq.dps_lake_model import lake_model as lake_problem

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant, SequentialEvaluator, MultiprocessingEvaluator,
                           ema_logging, Constraint, Policy, save_results)
from ema_workbench.analysis import (parcoords, prim)
import matplotlib.pyplot as plt

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

lake_model.outcomes = [ScalarOutcome("max_P", kind=ScalarOutcome.MINIMIZE),
                       ScalarOutcome("utility", kind=ScalarOutcome.MAXIMIZE),
                       ScalarOutcome("inertia", kind=ScalarOutcome.MAXIMIZE),
                       ScalarOutcome("reliability", kind=ScalarOutcome.MAXIMIZE)]

lake_model.constants = [Constant("alpha", 0.4),
                        Constant("nsamples", 100),
                        Constant("myears", 100)]

n_scenarios = 10

if __name__ == '__main__':
    nfes = [1000, 10000]
    epsilons = [0.01, 0.02, 0.05, 0.1]
    for nfe in nfes:
        for epsilon in epsilons:
            epsilon = [epsilon, epsilon, epsilon, epsilon]
            constraints = [Constraint("reliability", outcome_names="reliability",
                                      function=lambda x : max(0.9, 1.8-x))]
            with MultiprocessingEvaluator(lake_model) as evaluator:
                results = evaluator.optimize(epsilons=epsilon, nfe=nfe, constraints=constraints)
            new_results = results.loc[results["reliability"] > 0.9]

            policy_results = new_results.drop(['max_P', 'utility', "inertia", 'reliability'], axis=1)
            new_results_dict = new_results.to_dict('index')
            policies = []
            for dict_index in range(len(new_results_dict)):
                policy_dict = new_results_dict[dict_index]
                policies.append(Policy(f"pol_{dict_index}", **policy_dict))

            # for policy in policies:
            with MultiprocessingEvaluator(lake_model) as evaluator:
                results_w_policy, outcomes_w_policy = evaluator.perform_experiments(n_scenarios, policies=policies)
            results_w_policy_ext = results_w_policy.rename(columns={"max_P": "max_P_initial",
                                                                    "inertia": "inertia_initial",
                                                                    "utility": "utility_initial",
                                                                    "reliability": "reliability_initial"})

            df_outcomes_w_policy = pd.DataFrame(outcomes_w_policy)

            df_full = pd.concat([df_outcomes_w_policy, results_w_policy_ext], axis=1)

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
            axes = parcoords.ParallelAxes(limits)
            axes.plot(df_plot_signals)
            # plt.show()

            def calculate_regret() :
                df_regret = pd.DataFrame(columns=['policy_num', 'max_P', 'utility', "inertia", 'reliability'])
                for unique_item in df_full['policy'].unique() :
                    df_partial = df_full.loc[df_full['policy'] == unique_item]
                    df_partial.reset_index(drop=True, inplace=True)
                    max_max_P = max(df_partial['max_P'])
                    max_utility = max(df_partial['utility'])
                    max_inertia = max(df_partial['inertia'])
                    max_reliability = max(df_partial['reliability'])
                    for index in range(len(df_partial)) :
                        new_row = {'policy_num' : unique_item,
                                   'max_P' : max_max_P - df_partial.loc[index, 'max_P'],
                                   'utility' : max_utility - df_partial.loc[index, 'utility'],
                                   'inertia' : max_inertia - df_partial.loc[index, 'inertia'],
                                   'reliability' : max_reliability - df_partial.loc[index, 'reliability']}
                        df_regret = df_regret.append(new_row, ignore_index=True)

                columns_to_sum = ['max_P', 'utility', "inertia", 'reliability']
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
            best_c1 = results_w_policy_ext.loc[results_w_policy_ext['policy'] == best_policy, "c1"].iloc[0]
            best_c2 = results_w_policy_ext.loc[results_w_policy_ext['policy'] == best_policy, "c2"].iloc[0]
            best_r1 = results_w_policy_ext.loc[results_w_policy_ext['policy'] == best_policy, "r1"].iloc[0]
            best_r2 = results_w_policy_ext.loc[results_w_policy_ext['policy'] == best_policy, "r2"].iloc[0]
            best_w1 = results_w_policy_ext.loc[results_w_policy_ext['policy'] == best_policy, "w1"].iloc[0]
            print(f"Best policy is policy number: {best_policy}\n"
                  f"This policy has the following values: \n"
                  f"c1: {best_c1}\n"
                  f"c2: {best_c2}\n"
                  f"r1: {best_r1}\n"
                  f"r2: {best_r2}\n"
                  f"w1: {best_w1}")


            x = results_w_policy
            y = outcomes_w_policy['utility'] < 0.25
            prim_alg = prim.Prim(x, y, threshold=0.8)
            box1 = prim_alg.find_box()
            box1.show_tradeoff()
            plt.show()

            selected_experiments = results_w_policy.iloc[box1.yi]
            selected_outcomes = {k : v[box1.yi] for k, v in outcomes_w_policy.items()}

            save_results((selected_experiments, selected_outcomes), './results/selected_results.tar.gz')

            outcomes = results.loc[:, ['max_P', 'utility', "inertia", 'reliability']]

            limits = parcoords.get_limits(outcomes)
            axes = parcoords.ParallelAxes(limits)
            axes.plot(outcomes)

            # we invert this axis so direction of desirability is the same
            axes.invert_axis('max_P')
            plt.show()

z = 1