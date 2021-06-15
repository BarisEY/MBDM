import matplotlib.pyplot as plt
from ema_workbench import load_results, Policy, MultiprocessingEvaluator, save_results, ema_logging
from ema_workbench.analysis import parcoords
import seaborn as sns
from problem_formulation import get_model_for_problem_formulation
import numpy as np
import pandas as pd


experiments, outcomes = load_results("deep_uncertainty.tar.gz")

thresholds = {'A.1_Expected Number of Deaths' : 0.0, 'A.2_Expected Number of Deaths' : 0.0}

overall_scores = {}
for policy in experiments.policy.unique() :
    logical = experiments.policy == policy
    scores = {}
    for k, v in outcomes.items() :
        try :
            a = v[logical] <= thresholds[k]
            n = np.sum(a)
        except KeyError :
            continue
        scores[k] = n / 1000
    overall_scores[policy] = scores

overall_scores = pd.DataFrame(overall_scores).T

limits = parcoords.get_limits(overall_scores)
paraxes = parcoords.ParallelAxes(limits)
paraxes.plot(overall_scores)
# plt.show()

overall_scores = {}
regret = []
for scenario in experiments.policy.unique() :
    logical = experiments.policy == scenario
    temp_results = {k : v[logical] for k, v in outcomes.items()}
    temp_results = pd.DataFrame(temp_results)
    temp_experiments = experiments[experiments.scenario == scenario]

    best = temp_results.min()
    worst = temp_results.max()
    scenario_regret = worst - best
    scenario_regret['policy'] = temp_experiments.policy.values
    regret.append(scenario_regret)

regret = pd.DataFrame(regret)
maxregret = regret.groupby('policy').max()

limits = parcoords.get_limits(maxregret)
paraxes = parcoords.ParallelAxes(maxregret)
paraxes.plot(maxregret)
plt.show()