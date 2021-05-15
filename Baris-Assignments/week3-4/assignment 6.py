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


def analyze_results_final_value(results, variable):
    _, outcomes = results
    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[variable]
    final_values = np.zeros(shape=len(y))
    for i in range(len(y)):
        a = y[i]
        b = a[-1][-1]
        final_values[i] = b
    # y[:] = y[:][-1]
    sobol_indices = sobol.analyze(problem, final_values)
    sobol_stats = {key: sobol_indices[key] for key in ['ST', 'ST_conf', 'S1',
                                                       'S1_conf']}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
    sobol_stats.sort_values(by='ST', ascending=False)
    s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'],
                      columns=problem['names'])
    s2_conf = pd.DataFrame(sobol_indices['S2_conf'], index=problem['names'],
                           columns=problem['names'])

    return sobol_stats, s2, s2_conf

def analyze_results_mean_value(results, variable):
    _, outcomes = results
    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[variable]
    final_values = np.zeros(shape=len(y))
    for i in range(len(y)):
        a = y[i]
        b = a[-1]
        c = b.mean()
        final_values[i] = c
    # y[:] = y[:][-1]
    sobol_indices = sobol.analyze(problem, final_values)
    sobol_stats = {key: sobol_indices[key] for key in ['ST', 'ST_conf', 'S1',
                                                       'S1_conf']}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
    sobol_stats.sort_values(by='ST', ascending=False)
    s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'],
                      columns=problem['names'])
    s2_conf = pd.DataFrame(sobol_indices['S2_conf'], index=problem['names'],
                           columns=problem['names'])

    return sobol_stats, s2, s2_conf

def analyze_results_standard_dev_value(results, variable):
    _, outcomes = results
    problem = get_SALib_problem(model.uncertainties)
    y = outcomes[variable]
    final_values = np.zeros(shape=len(y))
    for i in range(len(y)):
        a = y[i]
        b = a[-1]
        c = b.std()
        final_values[i] = c
    # y[:] = y[:][-1]
    sobol_indices = sobol.analyze(problem, final_values)
    sobol_stats = {key: sobol_indices[key] for key in ['ST', 'ST_conf', 'S1',
                                                       'S1_conf']}
    sobol_stats = pd.DataFrame(sobol_stats, index=problem['names'])
    sobol_stats.sort_values(by='ST', ascending=False)
    s2 = pd.DataFrame(sobol_indices['S2'], index=problem['names'],
                      columns=problem['names'])
    s2_conf = pd.DataFrame(sobol_indices['S2_conf'], index=problem['names'],
                           columns=problem['names'])

    return sobol_stats, s2, s2_conf

# if __name__ == '__main__':
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

with ema_workbench.SequentialEvaluator(model) as evaluator:
    results = evaluator.perform_experiments(n_scenarios_1, uncertainty_sampling=SOBOL)

# with ema_workbench.MultiprocessingEvaluator(model) as evaluator:
#     results = evaluator.perform_experiments(n_scenarios_1, uncertainty_sampling=SOBOL)

sobol_stats, s2, s2_conf = analyze_results_final_value(results, 'prey')
print(sobol_stats)
print(s2)
print(s2_conf)

sobol_stats, s2, s2_conf = analyze_results_mean_value(results, 'prey')
print(sobol_stats)
print(s2)
print(s2_conf)

sobol_stats, s2, s2_conf = analyze_results_standard_dev_value(results, 'prey')
print(sobol_stats)
print(s2)
print(s2_conf)
