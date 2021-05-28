import pandas as pd

from Week_5_6_quaq.dps_lake_model import lake_model as lake_problem

from ema_workbench import Model, RealParameter, ScalarOutcome, Constant, SequentialEvaluator, \
    MultiprocessingEvaluator, ema_logging, analysis, Constraint, Policy
from ema_workbench.analysis import parcoords
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
                       ScalarOutcome("intertia", kind=ScalarOutcome.MAXIMIZE),
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

            policy_results = new_results.drop(['max_P', 'utility', "intertia", 'reliability'], axis=1)
            new_results_dict = new_results.to_dict('index')
            policies = []
            for dict_index in range(len(new_results_dict)):
                policy_dict = new_results_dict[dict_index]
                policies.append(Policy(f"pol_{dict_index}", **policy_dict))

            # for policy in policies:
            with MultiprocessingEvaluator(lake_model) as evaluator:
                results_w_policy, outcomes_w_policy = evaluator.perform_experiments(n_scenarios, policies=policies)

            df_outcomes_w_policy = pd.DataFrame(outcomes_w_policy)

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
            plt.show()

            outcomes = results.loc[:, ['max_P', 'utility', "intertia", 'reliability']]

            limits = parcoords.get_limits(outcomes)
            axes = parcoords.ParallelAxes(limits)
            axes.plot(outcomes)

            # we invert this axis so direction of desirability is the same
            axes.invert_axis('max_P')
            plt.show()

z = 1