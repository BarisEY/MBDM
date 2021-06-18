# Imports:
# External
import seaborn as sns
import matplotlib.pyplot as plt

# EMA Workbench
from ema_workbench import load_results, Policy, MultiprocessingEvaluator, save_results, ema_logging
from ema_workbench.analysis import parcoords

# Internal
from quaquel_github_files_kopie.problem_formulation import get_model_for_problem_formulation

if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Load all the results
    full_results = []
    for i in range(12):
        full_results.append(load_results(f"..//final_data//mordm_last_{i}.tar.gz"))

    # Set colors in order to iterate over them for the parcoord plots
    colors = iter(sns.color_palette())
    for i, (result, _) in enumerate(full_results) :
        # Randomly sample 5 scenarios.
        results = result.sample(n=5).reset_index(drop=True)
        data = results.iloc[:, 31 : :]
        if i == 0:
            limits = parcoords.get_limits(data)
        else:
            for col in data.columns:
                if limits.loc[1, col] < result[col].max():
                    a = result[col].max()
                    limits.loc[1, col] = a

    limits.loc[0, ["A.1 Total Costs", "A.1_Expected Number of Deaths",
                   "A.2 Total Costs", "A.2_Expected Number of Deaths",
                   "A.3 Total Costs", "A.3_Expected Number of Deaths",
                   "A.4 Total Costs", "A.4_Expected Number of Deaths",
                   "A.5 Total Costs", "A.5_Expected Number of Deaths",
                   "RfR Total Costs", "Expected Evacuation Costs"]] = 0

    # Plot all the parcoords plots one be one.
    paraxes = parcoords.ParallelAxes(limits)
    for i, (result, _) in enumerate(full_results):
        try:
            color = next(colors)
        except:
            pass
        result = result.sample(n=5).reset_index(drop=True)
        # Selecting the columns of interest
        data = result.iloc[:, 31::]
        paraxes.plot(data, label=f'scenario {i}', color=color)

    paraxes.legend()
    plt.gcf().subplots_adjust(bottom=0.4)
    plt.show()


    # Filter out the columns of interest for the results
    result = result.iloc[:, 0:31]

    # Create different policies from these results.
    policies = []
    for j, row in result.iterrows():
        policy = Policy(f'scenario {i} option {j}', **row.to_dict())
        policies.append(policy)

    model, array = get_model_for_problem_formulation(3)

    # This was to run the experiment again with the different policies and scenarios.
    with MultiprocessingEvaluator(model) as evaluator:
        reevaluation_results = evaluator.perform_experiments(1000, policies=policies)

    experiments, outcomes = reevaluation_results
    save_results((experiments, outcomes), f'deep_uncertainty.tar.gz')