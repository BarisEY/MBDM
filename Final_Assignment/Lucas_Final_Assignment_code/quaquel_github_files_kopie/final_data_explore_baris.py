import matplotlib.pyplot as plt
from ema_workbench import load_results, Policy, MultiprocessingEvaluator, save_results, ema_logging
from ema_workbench.analysis import parcoords
import seaborn as sns
from problem_formulation import get_model_for_problem_formulation
import numpy as np
import pandas as pd

if __name__ == '__main__':

    ema_logging.log_to_stderr(ema_logging.INFO)

    results = []
    for i in range(12):
        results.append(load_results(f"..//..//final_data//mordm_last_{i}.tar.gz"))

    #     fig, (ax1, ax2) = plt.subplots(ncols=2)
    #     _, convergence = results
    #     ax1.plot(convergence["nfe"], convergence["hypervolume"], label=f'scenario {i}')
    #     ax2.plot(convergence["nfe"], convergence["epsilon_progress"], label=f'scenario {i}')
    #
    # ax1.set_ylabel('hypervolume')
    # ax1.set_xlabel('nfe')
    # ax2.set_ylabel('$\epsilon$ progress')
    # ax2.set_xlabel('nfe')
    # fig.legend()
    # plt.show()

    colors = iter(sns.color_palette())



    policies = []
    for i, (result, _) in enumerate(results):
        try:
            color = next(colors)
        except:
            pass
        result = result.sample(n=5).reset_index(drop=True)
        data = result.iloc[:, 31 : :]
        limits = parcoords.get_limits(data)

        limits.loc[0, ["A.1 Total Costs", "A.1_Expected Number of Deaths",
                       "A.2 Total Costs", "A.2_Expected Number of Deaths",
                       "A.3 Total Costs", "A.3_Expected Number of Deaths",
                       "A.4 Total Costs", "A.4_Expected Number of Deaths",
                       "A.5 Total Costs", "A.5_Expected Number of Deaths",
                       "RfR Total Costs", "Expected Evacuation Costs"]] = 0
        # limits.loc[0, ['inertia', 'reliability']] = 1
        # limits.loc[0, 'max_P'] = 4 # max over results based on quick inspection not shown here
        # limits.loc[0, 'utility'] = 1 # max over results based on quick inspection not shown here
        # limits.loc[1, :] = 0
        paraxes = parcoords.ParallelAxes(limits)

        data = result.iloc[:, 31::]
        paraxes.plot(data, label=f'scenario {i}', color=color)

        paraxes.legend()

        plt.show()



        result = result.iloc[:, 0:31]
        for j, row in result.iterrows():
            policy = Policy(f'scenario {i} option {j}', **row.to_dict())
            policies.append(policy)

    model, array = get_model_for_problem_formulation(3)

    # This was to run the experiment again with the different policies:
    #
    # with MultiprocessingEvaluator(model) as evaluator:
    #     reevaluation_results = evaluator.perform_experiments(1000, policies=policies)
    #
    # experiments, outcomes = reevaluation_results
    # save_results((experiments, outcomes), f'deep_uncertainty.tar.gz')