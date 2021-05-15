import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ema_workbench
from ema_workbench import (Model, RealParameter, TimeSeriesOutcome, perform_experiments, ema_logging)

from ema_workbench.em_framework.evaluators import LHS, SOBOL, MORRIS

from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis.scenario_discovery_util import RuleInductionType
from ema_workbench.em_framework.salib_samplers import get_SALib_problem
from SALib.analyze import sobol
import statsmodels.api as sm
import seaborn as sns
import statsmodels.formula.api as smf
from ema_workbench import (RealParameter, ScalarOutcome, Constant, Model)

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 10000)

def pred_prey(prey_birth_rate=0.025, predation_rate=0.0015, predator_efficiency=0.002,
              predator_loss_rate=0.06, initial_prey=50, initial_predators=20, dt=0.25,
              final_time=365, reps=1):
    # Initial values
    predators = np.zeros((reps, int(final_time / dt) + 1))
    prey = np.zeros((reps, int(final_time / dt) + 1))
    sim_time = np.zeros((reps, int(final_time / dt) + 1))

    for r in range(reps):
        predators[r, 0] = initial_predators
        prey[r, 0] = initial_prey

    # Calculate the time series
    for t in range(0, sim_time.shape[1] - 1):
        dx = (prey_birth_rate * prey[r, t]) - (predation_rate * prey[r, t] * predators[r, t])
        dy = (predator_efficiency * predators[r, t] * prey[r, t]) - (predator_loss_rate * predators[r, t])

        prey[r, t + 1] = max(prey[r, t] + dx * dt, 0)
        predators[r, t + 1] = max(predators[r, t] + dy * dt, 0)
        sim_time[r, t + 1] = (t + 1) * dt

    # Return outcomes
    return {'TIME': sim_time,
            'predators': predators,
            'prey': prey}

def analyze_results(results, variable, analyze_array):
    _, outcomes = results
    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[variable]
    sobol_indices = sobol.analyze(problem, analyze_array)
    sobol_stats = {key: sobol_indices[key] for key in ['ST', 'ST_conf', 'S1',
                                                       'S1_conf']}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
    sobol_stats.sort_values(by='ST', ascending=False)
    s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'],
                      columns=problem['names'])
    s2_conf = pd.DataFrame(sobol_indices['S2_conf'], index=problem['names'],
                           columns=problem['names'])

    return sobol_stats, s2, s2_conf


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)
    #Creating a model instance
    model = Model('predator', function=pred_prey)

    #specify uncertainties
    model.uncertainties = [RealParameter('prey_birth_rate', 0.015, 0.035),
                           RealParameter('predation_rate', 0.0005, 0.003),
                           RealParameter('predator_efficiency', 0.001, 0.004),
                           RealParameter('predator_loss_rate', 0.04, 0.08)]

    model.outcomes = [TimeSeriesOutcome('TIME'),
                      TimeSeriesOutcome('predators'),
                      TimeSeriesOutcome('prey')]

    model.constants = [Constant('initial_prey', 50),
                       Constant('initial_predators', 20),
                       Constant('dt', 0.25),
                       Constant('final_time', 365),
                       Constant('reps', 1)]

    n_scenarios_1 = 50
    n_scenarios_2 = 250
    n_scenarios_3 = 1000

    with ema_workbench.MultiprocessingEvaluator(model) as evaluator:
        results = evaluator.perform_experiments(n_scenarios_1, uncertainty_sampling=SOBOL)

# with ema_workbench.MultiprocessingEvaluator(model) as evaluator:
#     results = evaluator.perform_experiments(n_scenarios_1, uncertainty_sampling=SOBOL)

    x, outcomes = results
    y = outcomes['prey']
    last_values = np.zeros(shape=len(y))
    mean_values = np.zeros(shape=len(y))
    standard_dev_values = np.zeros(shape=len(y))
    for i in range(len(y)):
        last_values[i] = y[i][-1][-1]
        mean_values[i] = y[i][-1].mean()
        standard_dev_values[i] = y[i][-1].std()
    # y[:] = y[:][-1]
    x = x.drop(['model', 'policy'], axis=1)
    y = np.max(outcomes['prey'], axis=1)# sobol_stats, s2, s2_conf = analyze_results(results, 'prey', last_values)
    # print(sobol_stats)
    # print(s2)
    all_scores = []# print(s2_conf)
    # Change this to 100, although it increases the time by a lot
    for i in range(10):#
        print(f"Running {i}")
        indices = np.random.choice(np.arange(0, x.shape[0]), size=x.shape[0])# sobol_stats, s2, s2_conf = analyze_results(results, 'prey', mean_values)
        selected_x = x.iloc[indices, :]# print(sobol_stats)
        selected_y = y[indices]# print(s2)
        # print(s2_conf)
        scores = feature_scoring.get_ex_feature_scores(selected_x,
                                                       selected_y,
                                                       max_features=0.6,
                                                       mode=RuleInductionType.REGRESSION)[0]# sobol_stats, s2, s2_conf = analyze_results(results, 'prey', standard_dev_values)
        all_scores.append(scores)# print(sobol_stats)
    all_scores = pd.concat(all_scores, axis=1, sort=False)# print(s2)
    # print(s2_conf)
    sns.boxplot(data=all_scores.T)
    plt.show()
